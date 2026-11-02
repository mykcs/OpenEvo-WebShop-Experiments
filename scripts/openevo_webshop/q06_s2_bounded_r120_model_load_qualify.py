#!/usr/bin/env python3
"""Zero-task Q06 S2 model/adapter load qualification using the Q17 runtime.

This helper intentionally loads the exact frozen Q06 runtime/model/adapter and
then exits before any WebShop environment, panel, generation, scorer, or
training path is entered. It is intended to run inside the already-qualified
Q06 image; it does not create or launch a container itself.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from scripts.openevo_webshop import q06_s2_bounded_r120_host_orchestrator as host
from scripts.openevo_webshop import q17_fixed_checkpoint_probe as q17

SCHEMA = "openevo.q06.s2-bounded-r120-model-load-qualification.v1"
GPU_RE = re.compile(r"^GPU-[0-9a-f-]{36}$")


class QualificationError(RuntimeError):
    """Fail-closed model-load qualification rejection."""


def require(ok: bool, message: str) -> None:
    if not ok:
        raise QualificationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source(source: Path, expected_sha: str, expected_repository: str,
                  *, run: Callable[..., Any] = subprocess.run) -> str:
    """Require the exact clean execution checkout mounted into the image."""
    require(re.fullmatch(r"[0-9a-f]{40}", expected_sha) is not None,
            "source SHA must be a full Git commit")
    source = source.resolve()

    def git(*args: str) -> str:
        return run(["git", "-C", str(source), *args], check=True,
                   capture_output=True, text=True).stdout.strip()

    root = Path(git("rev-parse", "--show-toplevel")).resolve()
    remote = git("remote", "get-url", "origin")
    head = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain", "--untracked-files=all")
    require(root == source and remote == expected_repository and head == expected_sha and not dirty,
            "mounted execution source is not the exact clean pinned checkout")
    return head


def load_q17_probe(source: Path):
    """Import the Q17 probe from the exact pinned execution source checkout."""
    probe_path = source.resolve() / host.PROBE_PATH
    require(probe_path.is_file() and not probe_path.is_symlink(),
            "pinned execution source lacks the canonical Q17 fixed-checkpoint probe")
    spec = importlib.util.spec_from_file_location("q06_pinned_q17_fixed_checkpoint_probe", probe_path)
    require(spec is not None and spec.loader is not None,
            "cannot import Q17 fixed-checkpoint probe from pinned execution source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_load_identity(*, manifest_path: Path, checkpoint_id: str,
                           model_path: Path, model_revision: str, adapter_path: Path,
                           source: Path, source_sha: str, source_repository: str,
                           image_id: str,
                           gpu_uuid: str, output_dir: Path,
                           environ: dict[str, str] | None = None) -> dict[str, Any]:
    """Validate all execution identities before loading any model weights."""
    env = os.environ if environ is None else environ
    require(image_id == host.IMAGE_ID and env.get("OPENEVO_RUNTIME_IMAGE_ID") == host.IMAGE_ID,
            "runtime image does not match the frozen Q06 image identity")
    require(GPU_RE.fullmatch(gpu_uuid) is not None
            and env.get("CUDA_VISIBLE_DEVICES", "").strip() == gpu_uuid,
            "CUDA_VISIBLE_DEVICES must bind exactly the selected GPU UUID")
    require(manifest_path.is_file() and not manifest_path.is_symlink(),
            "Q06 source manifest is missing or symlinked")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    require(isinstance(manifest, dict)
            and manifest.get("schema") == "openevo.q06.s2-bounded-r120-probe-manifest.v1"
            and manifest.get("cell_id") == host.CELL_ID
            and manifest.get("checkpoint_id") == host.CHECKPOINT_ID == checkpoint_id,
            "manifest is not the exact Q06 S2 Bounded-R120 checkpoint")
    checkpoint = manifest.get("checkpoints", {}).get(checkpoint_id)
    require(isinstance(checkpoint, dict)
            and checkpoint.get("adapter_sha256") == manifest.get("endpoint", {}).get(
                "source_files", {}).get("adapter", {}).get("sha256"),
            "manifest checkpoint does not bind the adapter payload")
    adapter_ref = manifest["endpoint"]["source_files"]["adapter"]
    expected_adapter_dir = Path(str(manifest["endpoint"]["adapter_path"]))
    adapter_file = adapter_path / "adapter_model.safetensors"
    require(adapter_path.resolve() == expected_adapter_dir.resolve()
            and adapter_file.is_file() and not adapter_file.is_symlink()
            and adapter_ref.get("path") == str(adapter_file)
            and adapter_ref.get("bytes") == adapter_file.stat().st_size
            and q17.sha256_file(adapter_file) == checkpoint["adapter_sha256"],
            "adapter path/size/hash differs from the frozen Q06 manifest")
    model = manifest.get("model", {})
    require(model_path.resolve() == Path(str(model.get("local_model_path", ""))).resolve()
            and model_revision == model.get("revision"),
            "Qwen base model path or revision differs from the frozen Q06 manifest")
    require(output_dir.is_dir() and not output_dir.is_symlink(),
            "qualification output directory must already exist and be a real directory")
    runtime_source = source.resolve()
    source_sha256 = verify_source(source, source_sha, source_repository)
    pinned_q17 = load_q17_probe(source)
    pinned_q17.validate_runtime_files(manifest, runtime_source)
    return {
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "checkpoint_id": checkpoint_id,
        "adapter_path": str(adapter_path.resolve()),
        "adapter_sha256": checkpoint["adapter_sha256"],
        "model_path": str(model_path.resolve()),
        "model_revision": model_revision,
        "image_id": image_id,
        "source_path": str(source.resolve()),
        "source_sha": source_sha256,
        "q17_probe_sha256": sha256_file(runtime_source / host.PROBE_PATH),
        "gpu_uuid": gpu_uuid,
        "output_dir": str(output_dir.resolve()),
    }


def make_receipt(identity: dict[str, Any], *, status: str,
                 model_forward_calls: int, generate_calls: int,
                 optimizer_updates: int, output_write_gate: str,
                 failure_reason: str | None = None) -> dict[str, Any]:
    """Build an append-only receipt that never claims scientific work."""
    receipt = {
        "schema": SCHEMA,
        "status": status,
        "identity": dict(identity),
        "qualification_scope": "model-and-adapter-load-only",
        "task_bearing_calls": 0,
        "model_forward_calls": int(model_forward_calls),
        "generate_calls": int(generate_calls),
        "training_calls": 0,
        "optimizer_updates": int(optimizer_updates),
        "formal_task_consumed": False,
        "scientific_state_mutated": False,
        "final_panel_access": 0,
        "output_write_gate": output_write_gate,
        "scientific_authority_granted_by_receipt": False,
    }
    if failure_reason:
        receipt["failure_reason"] = failure_reason
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return receipt


def _install_no_execution_guards(torch: Any) -> tuple[dict[str, int], Callable[[], None]]:
    """Count and reject module execution, generation, backward, and optimizer steps."""
    from transformers.generation import GenerationMixin

    counts = {"module_calls": 0, "generate_calls": 0, "optimizer_updates": 0}
    original_call_impl = torch.nn.Module._call_impl
    original_generate = GenerationMixin.generate
    original_backward = torch.Tensor.backward
    original_step = torch.optim.Optimizer.step

    def no_module_calls(module, *args, **kwargs):
        counts["module_calls"] += 1
        raise QualificationError("model forward/module execution is forbidden in load-only qualification")

    def no_generate(self, *args, **kwargs):
        counts["generate_calls"] += 1
        raise QualificationError("generation is forbidden in load-only qualification")

    def no_backward(self, *args, **kwargs):
        raise QualificationError("backward/training is forbidden in load-only qualification")

    def no_optimizer_step(self, *args, **kwargs):
        counts["optimizer_updates"] += 1
        raise QualificationError("optimizer updates are forbidden in load-only qualification")

    torch.nn.Module._call_impl = no_module_calls
    GenerationMixin.generate = no_generate
    torch.Tensor.backward = no_backward
    torch.optim.Optimizer.step = no_optimizer_step

    def restore() -> None:
        torch.nn.Module._call_impl = original_call_impl
        GenerationMixin.generate = original_generate
        torch.Tensor.backward = original_backward
        torch.optim.Optimizer.step = original_step

    return counts, restore


def load_and_qualify(identity: dict[str, Any], *, runtime_source_root: Path,
                     memory_path: Path, skill_path: Path, agent_path: Path,
                     output_dir: Path, runtime_factory: Callable[..., Any] | None = None,
                     torch_module: Any | None = None,
                     output_gate: Callable[[Path], None] = host.uid_write_rename_gate) -> dict[str, Any]:
    """Load the canonical Q17 Runtime, verify CUDA placement, gate output, close."""
    if torch_module is None:
        import torch as torch_module
    pinned_q17 = load_q17_probe(runtime_source_root)
    modules = pinned_q17.load_science_modules(runtime_source_root)
    factory = runtime_factory or modules["Runtime"]
    counts, restore = _install_no_execution_guards(torch_module)
    runtime = None
    gate_status = "NOT_RUN"
    status = "FAIL"
    failure_reason = None
    try:
        runtime = factory(
            identity["model_path"], identity["model_revision"],
            sd_adapter=Path(identity["adapter_path"]),
            memory=memory_path.read_text(), skill_bundle=skill_path.read_text(),
            agent_system=agent_path.read_text(),
        )
        model = runtime.base._model
        require(str(getattr(model, "device", "")).startswith("cuda"),
                "loaded Qwen model is not placed on CUDA")
        require(not bool(getattr(model, "training", False)),
                "loaded Qwen model is in training mode")
        output_gate(output_dir)
        gate_status = "PASS"
        status = "PASS"
    except BaseException as exc:
        failure_reason = f"{type(exc).__name__}: {exc}"
    finally:
        if runtime is not None:
            try:
                runtime.close()
            except BaseException as exc:
                status = "FAIL"
                failure_reason = failure_reason or f"runtime close failed: {type(exc).__name__}: {exc}"
        restore()
    return make_receipt(
        identity, status=status, model_forward_calls=counts["module_calls"],
        generate_calls=counts["generate_calls"], optimizer_updates=counts["optimizer_updates"],
        output_write_gate=gate_status, failure_reason=failure_reason,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--runtime-source-root", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--source-repository", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--sd-adapter", type=Path, required=True)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--skill", type=Path, required=True)
    parser.add_argument("--agent", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    try:
        identity = validate_load_identity(
            manifest_path=args.manifest, checkpoint_id=args.checkpoint_id,
            model_path=args.model, model_revision=args.revision,
            adapter_path=args.sd_adapter, source=args.runtime_source_root,
            source_sha=args.source_sha, source_repository=args.source_repository,
            image_id=args.image_id,
            gpu_uuid=args.gpu_uuid, output_dir=args.output_dir,
        )
        manifest = json.loads(args.manifest.read_text())
        require(os.geteuid() == 1001, "model-load qualification must run as the frozen image UID 1001")
        q17.validate_payload(args.memory, manifest["common_carriers"]["memory_sha256"], "memory")
        q17.validate_payload(args.skill, manifest["common_carriers"]["skill_sha256"], "skill")
        q17.validate_payload(args.agent, manifest["common_carriers"]["agent_system_sha256"], "agent_system")
        receipt = load_and_qualify(
            identity, runtime_source_root=args.runtime_source_root,
            memory_path=args.memory, skill_path=args.skill, agent_path=args.agent,
            output_dir=args.output_dir,
        )
    except BaseException as exc:
        receipt = make_receipt({}, status="FAIL", model_forward_calls=0,
                               generate_calls=0, optimizer_updates=0,
                               output_write_gate="NOT_RUN",
                               failure_reason=f"{type(exc).__name__}: {exc}")
    target = args.receipt.absolute()
    require(not target.exists() and not target.is_symlink(),
            "receipt path must be fresh; qualification receipts are append-only")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": receipt["status"], "receipt": str(target),
                      "task_bearing_calls": receipt["task_bearing_calls"],
                      "model_forward_calls": receipt["model_forward_calls"]}, sort_keys=True))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
