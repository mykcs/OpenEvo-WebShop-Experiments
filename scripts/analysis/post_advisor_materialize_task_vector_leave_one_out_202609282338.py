#!/usr/bin/env python3
"""Materialize exact CPU-only E4 leave-one-update-out LoRA states.

For each t in R152..R159 this writes W159 - (Wt - Wt-1) as rank-384 LoRA
factors. It reads only the sealed adapter/config/state/summary files, never
loads a model, imports a task evaluator, or accesses a benchmark panel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


SOURCE_RUN = "bounded-effective-state-gdr-recovery-1e7c4952-20260916/off"
SOURCE_CONFIG_SHA256 = "cbd30da3262c20cd0df9542d7577d14568db14e7d9cbb69291c0182da96d4ee8"
SOURCE_STATES = {
    151: ("afb7ff3a3e91536576d2661091c0af04411c834b33c9cb7dc4fbaa1eac1bd3e6", "812e64cff1efc55d3eb7e293876c0ab8658734b42486898306e3a338335d784a"),
    152: ("30061f4a4e7275695f487e8d68ebf34945793acba2a96e8999c87f62e18aca96", "e1f55b33807c03495b9c1b3df03d4827d3ff5f8cd1ec0b4f6084a55c0ce471b8"),
    153: ("ce341beec205c22f7db1b5c2f8fdfd5e6ced67db17cde4c3ba045c9758fffff4", "e81003489a4a41a7c9055135450690808d4ccb972a4649ec1a5995969a579a0c"),
    154: ("5dd4d8d60a85cb32d27ce8ea7e4b5e5679db3f9bacb98fcae367474875b5720a", "00e84ea051f3e3d10d70961f9a74f89ebe40cbbd971d296d53418948b79bf823"),
    155: ("ac9750603346765ff09468af1fb721d7fd31add1b4619f1ddd28588479fb6be2", "d238a8f94cd62d3a3bb86890ffb91dc2cf38a0220379d0a9fbe3d6d4c0ba19dc"),
    156: ("968083a2db3c1463f139188fcafe7d788d161bdb57aec9ba8e610b4ec4bbaaff", "284c4aecacff4130f528efb35f29dec7efa798858e7fe4b1959b0594f78ce3d5"),
    157: ("297dbf20b2275f4535f02c05f6f48a73122e79428ead20425d72e292fa09108a", "980756040b8a47068b54e6eb0e5a3b2ef1dfc7fd0e46538d495acb9db352f226"),
    158: ("d6e78aa9f6189a338642c46fbd1d76c27d09fbfb08a5ba5d027c9d535bf611ef", "a4f3d315aa6a642217b87a2436f98d731cc3eca75f9ec69d7febbff6253e24ef"),
    159: ("6333af5da5b4cb850cd065a83670f877fb102ea1be5bc8d27438e10b3195dc2d", "1bc2f7f62c1bde68cd4fd34e86c25af7f9663153a9ded631759dda52c87a21a5"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def materialize_module_factors(torch, a_final, b_final, a_t, b_t, a_prev, b_prev):
    """Represent BfAf - BtAt + BpAp exactly by rank-concatenated factors."""
    pairs = ((a_final, b_final), (a_t, b_t), (a_prev, b_prev))
    for a, b in pairs:
        if a.ndim != 2 or b.ndim != 2 or int(a.shape[0]) != int(b.shape[1]):
            raise ValueError("invalid LoRA factor dimensions")
        if a.device.type != "cpu" or b.device.type != "cpu":
            raise ValueError("E4 materializer accepts CPU tensors only")
        if a.dtype != a_final.dtype or b.dtype != b_final.dtype:
            raise ValueError("source factor dtype drift")
        if int(a.shape[1]) != int(a_final.shape[1]) or int(b.shape[0]) != int(b_final.shape[0]):
            raise ValueError("source factor module dimensions drift")
    a_out = torch.cat((a_final, a_t, a_prev), dim=0).contiguous()
    b_out = torch.cat((b_final, -b_t, b_prev), dim=1).contiguous()
    return a_out, b_out


def _load_and_validate_sources(torch, source_root: Path) -> dict[int, dict[str, Any]]:
    from safetensors.torch import load_file

    if source_root.name != "off" or source_root.as_posix().endswith(SOURCE_RUN) is False:
        raise ValueError(f"source root must be the sealed {SOURCE_RUN} directory")
    records: dict[int, dict[str, Any]] = {}
    reference_config: dict[str, Any] | None = None
    for round_index, (expected_state_id, expected_adapter_sha) in SOURCE_STATES.items():
        round_dir = source_root / f"round-{round_index}"
        adapter_dir = round_dir / "sd-lora" / "bounded-state"
        adapter_path = adapter_dir / "adapter_model.safetensors"
        config_path = adapter_dir / "adapter_config.json"
        state_path = round_dir / "state-after.json"
        summary_path = round_dir / "summary.json"
        for required in (adapter_path, config_path, state_path, summary_path):
            if not required.is_file():
                raise FileNotFoundError(f"missing sealed E4 source file: {required}")
        adapter_sha = sha256_file(adapter_path)
        if adapter_sha != expected_adapter_sha:
            raise ValueError(f"R{round_index} adapter SHA mismatch: {adapter_sha}")
        if sha256_file(config_path) != SOURCE_CONFIG_SHA256:
            raise ValueError(f"R{round_index} adapter config SHA mismatch")
        state = json.loads(state_path.read_text())
        if state.get("state_id") != expected_state_id or int(state.get("bounded_state_rank", -1)) != 128:
            raise ValueError(f"R{round_index} state identity/rank mismatch")
        if Path(state.get("current_sd_adapter", "")).as_posix() != adapter_dir.as_posix():
            raise ValueError(f"R{round_index} state points to a different adapter directory")
        summary = json.loads(summary_path.read_text())
        if str(summary.get("status", "")).upper() != "PASS" or int(summary.get("round_index", -1)) != round_index:
            raise ValueError(f"R{round_index} sealed round summary is not PASS")
        if round_index > 151:
            prior_state_id = SOURCE_STATES[round_index - 1][0]
            if summary.get("state_id") != prior_state_id:
                raise ValueError(f"R{round_index} summary does not consume sealed R{round_index - 1} state")

        config = json.loads(config_path.read_text())
        if reference_config is None:
            reference_config = config
        elif config != reference_config:
            raise ValueError(f"R{round_index} adapter config differs from source lineage")
        if (
            int(config.get("r", -1)) != 128
            or int(config.get("lora_alpha", -1)) != 128
            or config.get("rank_pattern") != {}
            or config.get("alpha_pattern") != {}
            or config.get("use_rslora") is not False
            or config.get("use_dora") is not False
            or config.get("lora_bias") is not False
            or config.get("fan_in_fan_out") is not False
            or config.get("modules_to_save") is not None
            or config.get("base_model_name_or_path") != "Qwen/Qwen3-1.7B"
        ):
            raise ValueError("source adapter is not the frozen uniform rank-128 vanilla LoRA contract")
        records[round_index] = {
            "adapter_path": adapter_path,
            "config_path": config_path,
            "adapter_sha256": adapter_sha,
            "state_id": expected_state_id,
            "config": config,
        }

    return records


def _dense_reconstruction_audit(torch, a_out, b_out, a_final, b_final, a_t, b_t, a_prev, b_prev):
    expected = (
        b_final.double() @ a_final.double()
        - b_t.double() @ a_t.double()
        + b_prev.double() @ a_prev.double()
    )
    actual = b_out.double() @ a_out.double()
    difference = actual - expected
    error = float(torch.linalg.vector_norm(difference).item())
    expected_norm = float(torch.linalg.vector_norm(expected).item())
    relative_error = error / expected_norm if expected_norm > 0 else error
    max_abs_error = float(torch.max(torch.abs(difference)).item()) if difference.numel() else 0.0
    if not math.isfinite(relative_error) or relative_error > 1e-10:
        raise RuntimeError(f"rank-concatenated factors failed dense audit: rel={relative_error}")
    return {"relative_frobenius_error": relative_error, "max_abs_error": max_abs_error}


def materialize(source_root: Path, output_root: Path) -> dict[str, Any]:
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before E4 CPU materialization")
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite E4 output: {output_root}")
    records = _load_and_validate_sources(torch, source_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        final_record = records[159]
        final_tensors = load_file(str(final_record["adapter_path"]), device="cpu")
        a_keys = sorted(key for key in final_tensors if key.endswith(".lora_A.weight"))
        if len(a_keys) != 112 or len(final_tensors) != 224:
            raise ValueError("R159 tensor inventory differs from the sealed 112-module LoRA adapter")
        b_keys = {key.replace(".lora_A.weight", ".lora_B.weight") for key in a_keys}
        if set(final_tensors) != set(a_keys) | b_keys:
            raise ValueError("R159 contains missing, unmatched, or unexpected adapter tensors")
        module_names = [key.removesuffix(".lora_A.weight") for key in a_keys]
        module_reports: dict[int, dict[str, Any]] = {}
        for round_index in range(152, 160):
            previous_tensors = load_file(str(records[round_index - 1]["adapter_path"]), device="cpu")
            current_tensors = load_file(str(records[round_index]["adapter_path"]), device="cpu")
            if set(previous_tensors) != set(final_tensors) or set(current_tensors) != set(final_tensors):
                raise ValueError(f"R{round_index} tensor inventory differs from R159")
            result_tensors: dict[str, Any] = {}
            max_relative_error = 0.0
            max_abs_error = 0.0
            for a_key in a_keys:
                b_key = a_key.replace(".lora_A.weight", ".lora_B.weight")
                a_out, b_out = materialize_module_factors(
                    torch,
                    final_tensors[a_key], final_tensors[b_key],
                    current_tensors[a_key], current_tensors[b_key],
                    previous_tensors[a_key], previous_tensors[b_key],
                )
                if int(a_out.shape[0]) != 384 or int(b_out.shape[1]) != 384:
                    raise RuntimeError(f"R{round_index} materialized rank is not 384 for {a_key}")
                audit = _dense_reconstruction_audit(
                    torch, a_out, b_out,
                    final_tensors[a_key], final_tensors[b_key],
                    current_tensors[a_key], current_tensors[b_key],
                    previous_tensors[a_key], previous_tensors[b_key],
                )
                max_relative_error = max(max_relative_error, audit["relative_frobenius_error"])
                max_abs_error = max(max_abs_error, audit["max_abs_error"])
                result_tensors[a_key] = a_out
                result_tensors[b_key] = b_out
            name = f"leave-one-update-out-r{round_index}"
            adapter_out = temp_root / name
            adapter_out.mkdir()
            with safe_open(str(final_record["adapter_path"]), framework="pt", device="cpu") as handle:
                metadata = handle.metadata()
            save_file(result_tensors, str(adapter_out / "adapter_model.safetensors"), metadata=metadata)
            config = dict(final_record["config"])
            config["r"] = 384
            config["lora_alpha"] = 384
            config["rank_pattern"] = {}
            config["alpha_pattern"] = {}
            config["inference_mode"] = True
            (adapter_out / "adapter_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
            module_reports[round_index] = {
                "intervention": f"W159 - (W{round_index} - W{round_index - 1})",
                "output_rank": 384,
                "module_count": len(module_names),
                "max_dense_relative_frobenius_error": max_relative_error,
                "max_dense_abs_error": max_abs_error,
                "adapter_sha256": sha256_file(adapter_out / "adapter_model.safetensors"),
                "config_sha256": sha256_file(adapter_out / "adapter_config.json"),
            }
            del previous_tensors, current_tensors, result_tensors

        receipt = {
            "schema": "openevo.post-advisor.task-vector-leave-one-out-materialization.202609282338",
            "status": "PASS",
            "source_run_root": str(source_root),
            "source_rounds": {
                str(r): {"state_id": records[r]["state_id"], "adapter_sha256": records[r]["adapter_sha256"]}
                for r in SOURCE_STATES
            },
            "formula": "W159 - (Wt - Wt-1) = B159 A159 - Bt At + Bt-1 At-1, per target module",
            "representation": "exact rank concatenation; A=cat(A159,At,At-1), B=cat(B159,-Bt,Bt-1); all source scales=1 and output r=alpha=384",
            "source_base_model": "Qwen/Qwen3-1.7B",
            "source_rank": 128,
            "output_rank": 384,
            "modules_per_state": 112,
            "states": module_reports,
            "cpu_only": True,
            "model_loaded": False,
            "inference_started": False,
            "task_panel_accessed": False,
            "formal_rollouts_consumed": 0,
            "final_panel_access": 0,
        }
        (temp_root / "materialization-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        os.replace(temp_root, output_root)
        return receipt
    except BaseException:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True, help="sealed run's off directory")
    parser.add_argument("--output-root", type=Path, required=True, help="new directory for eight counterfactual adapters")
    args = parser.parse_args()
    print(json.dumps(materialize(args.source_root, args.output_root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
