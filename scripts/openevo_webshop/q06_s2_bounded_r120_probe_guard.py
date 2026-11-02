#!/usr/bin/env python3
"""Add a UID write/rename gate after Qwen loads and before q17's first generate."""
from __future__ import annotations

import argparse
import codecs
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Callable

from scripts.openevo_webshop.q06_s2_bounded_r120_host_orchestrator import (
    GateError,
    output_tree_bytes,
    uid_write_rename_gate,
)


def install_post_load_gate(probe: ModuleType, output: Path,
                           gate: Callable[[Path], None] = uid_write_rename_gate,
                           after_gate: Callable[[], None] | None = None) -> None:
    """Wrap q17's runtime constructor; the gate completes before main can generate."""
    original_loader = probe.load_science_modules

    def guarded_loader(runtime_source_root: Path):
        modules = original_loader(runtime_source_root)
        runtime_type = modules["Runtime"]

        class RuntimeWithPostLoadGate(runtime_type):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                gate(output)
                if after_gate is not None:
                    after_gate()

        modules["Runtime"] = RuntimeWithPostLoadGate
        return modules

    probe.load_science_modules = guarded_loader


def install_output_byte_guard(output: Path, max_bytes: int):
    """Wrap Path.open so cumulative UTF-8 output bytes are checked before writes."""
    if max_bytes <= 0:
        raise GateError("output byte cap must be positive")
    output = output.resolve()
    original_open = Path.open
    total = [output_tree_bytes(output)]

    class CappedFile:
        def __init__(self, stream):
            self._stream = stream

        def write(self, data):
            if isinstance(data, str):
                encoding = self._stream.encoding or "utf-8"
                errors = self._stream.errors or "strict"
                byte_count = len(data.encode(encoding, errors=errors))
            else:
                byte_count = len(data)
            if total[0] + byte_count > max_bytes:
                raise GateError("S2 cumulative persistent-output byte ceiling exceeded before write")
            result = self._stream.write(data)
            total[0] += byte_count
            return result

        def writelines(self, lines):
            for line in lines:
                self.write(line)

        def __enter__(self):
            self._stream.__enter__()
            return self

        def __exit__(self, exc_type, exc, traceback):
            return self._stream.__exit__(exc_type, exc, traceback)

        def __getattr__(self, name):
            return getattr(self._stream, name)

    def capped_open(path, mode="r", *args, **kwargs):
        target = Path(path)
        try:
            resolved = target.resolve(strict=False)
            in_output = resolved == output or output in resolved.parents
        except (OSError, RuntimeError) as exc:
            raise GateError(f"cannot resolve output write target: {target}") from exc
        writing = any(flag in mode for flag in ("w", "a", "x", "+"))
        if not (in_output and writing):
            return original_open(target, mode, *args, **kwargs)
        if target.is_symlink():
            raise GateError(f"refusing symlinked output write: {target}")
        if "b" not in mode:
            encoding = kwargs.get("encoding")
            if encoding is None or encoding == "locale":
                kwargs["encoding"] = "utf-8"
            else:
                try:
                    canonical_encoding = codecs.lookup(encoding).name
                except LookupError as exc:
                    raise GateError(f"unsupported output text encoding: {encoding}") from exc
                if canonical_encoding != "utf-8":
                    raise GateError(f"S2 output text must use UTF-8: {target}")
        previous_size = target.stat().st_size if target.is_file() else 0
        stream = original_open(target, mode, *args, **kwargs)
        if "w" in mode and "+" not in mode:
            total[0] = max(0, total[0] - previous_size)
        return CappedFile(stream)

    Path.open = capped_open

    def reconcile() -> None:
        total[0] = output_tree_bytes(output)

    def restore() -> None:
        Path.open = original_open

    return reconcile, restore


def install_scorer_cost_guard(probe: ModuleType, *, max_forward_calls: int,
                              max_input_tokens: int):
    """Count and stop Q17 click-scorer forwards before either ceiling is crossed."""
    if max_forward_calls <= 0 or max_input_tokens <= 0:
        raise GateError("Q06 scorer ceilings must be positive")
    state = {
        "schema": "openevo.q06.s2-scorer-cost-receipt.v1",
        "max_forward_calls": max_forward_calls,
        "max_input_tokens": max_input_tokens,
        "forward_calls": 0,
        "input_tokens": 0,
        "budget_exceeded": False,
        "accounting": "Counts input_ids.numel() and model invocations in GenerateObserver.score_click_destinations; checks both ceilings before calling the model.",
    }
    original_score = probe.GenerateObserver.score_click_destinations
    original_run_episode = probe.run_episode
    original_atomic_json = probe.atomic_json

    class CountedModel:
        def __init__(self, model):
            self.model = model

        def __getattr__(self, name):
            return getattr(self.model, name)

        def __call__(self, *args, **kwargs):
            if state["budget_exceeded"]:
                raise GateError("Q06 scorer budget was already exhausted")
            input_ids = kwargs.get("input_ids")
            if input_ids is None:
                raise GateError("Q06 scorer forward lacks explicit input_ids for cost accounting")
            numel = getattr(input_ids, "numel", None)
            input_count = int(numel()) if callable(numel) else len(input_ids[0])
            next_calls = state["forward_calls"] + 1
            next_tokens = state["input_tokens"] + input_count
            if next_calls > max_forward_calls or next_tokens > max_input_tokens:
                state["budget_exceeded"] = True
                raise GateError("Q06 scorer cost ceiling reached before model forward")
            state["forward_calls"] = next_calls
            state["input_tokens"] = next_tokens
            return self.model(*args, **kwargs)

    def guarded_score(self, *args, **kwargs):
        original_model = self.model
        self.model = CountedModel(original_model)
        try:
            return original_score(self, *args, **kwargs)
        finally:
            self.model = original_model

    def guarded_run_episode(*args, **kwargs):
        if state["budget_exceeded"]:
            raise GateError("Q06 scorer budget exhausted; remaining episodes were not started")
        return original_run_episode(*args, **kwargs)

    def guarded_atomic_json(path, payload):
        if isinstance(payload, dict) and Path(path).name == "summary.json":
            receipt = {**state, "within_budget": not state["budget_exceeded"]}
            payload["scorer_cost_receipt"] = receipt
            probe_receipt = payload.get("probe_receipt")
            if isinstance(probe_receipt, dict):
                probe_receipt["scorer_cost"] = receipt
        elif isinstance(payload, dict) and Path(path).name == "probe-receipt.json":
            payload["scorer_cost"] = {**state, "within_budget": not state["budget_exceeded"]}
        return original_atomic_json(path, payload)

    probe.GenerateObserver.score_click_destinations = guarded_score
    probe.run_episode = guarded_run_episode
    probe.atomic_json = guarded_atomic_json

    def restore():
        probe.GenerateObserver.score_click_destinations = original_score
        probe.run_episode = original_run_episode
        probe.atomic_json = original_atomic_json

    return state, restore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority-repo", type=Path, required=True)
    parser.add_argument("--probe-script", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-output-bytes", type=int, required=True)
    parser.add_argument("--max-scorer-forward-calls", type=int, required=True)
    parser.add_argument("--max-scorer-input-tokens", type=int, required=True)
    parser.add_argument("probe_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    repo = args.authority_repo.resolve()
    guard_script = Path(__file__).resolve()
    if repo not in guard_script.parents or Path(__file__).is_symlink():
        raise RuntimeError("probe guard must run from its exact authority-repository file")
    if args.probe_script.is_symlink():
        raise RuntimeError("q17 probe script may not be a symlink")
    probe_path = args.probe_script.resolve()
    if not probe_path.is_file():
        raise RuntimeError("q17 fixed checkpoint probe is missing")
    spec = importlib.util.spec_from_file_location("q06_s2_fixed_checkpoint_probe", probe_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import the pinned q17 fixed checkpoint probe")
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    reconcile_output_guard, restore_output_guard = install_output_byte_guard(
        args.output_dir, args.max_output_bytes)
    _, restore_scorer_cost_guard = install_scorer_cost_guard(
        probe, max_forward_calls=args.max_scorer_forward_calls,
        max_input_tokens=args.max_scorer_input_tokens)
    install_post_load_gate(probe, args.output_dir, after_gate=reconcile_output_guard)
    probe_args = args.probe_args[1:] if args.probe_args[:1] == ["--"] else args.probe_args
    sys.argv = [str(probe_path), *probe_args]
    try:
        return int(probe.main())
    finally:
        restore_scorer_cost_guard()
        restore_output_guard()


if __name__ == "__main__":
    raise SystemExit(main())
