#!/usr/bin/env python3
"""Build E4 reverse-direction adapters on CPU only.

This treatment adds W159 - 2Δt (the equal-magnitude reverse update) and references
the already-sealed W159 - Δt adapters by receipt and file hashes. It writes
only eight reverse-direction adapters to a new directory.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from scripts.analysis import post_advisor_materialize_task_vector_leave_one_out_202609282338 as leave_one_out


INTERVENTIONS = {
    "reverse-direction": {
        "output_id": "reverse-direction-r{round_index}",
        "coefficients": (1, -2, 2),
        "formula": "W159 - 2*(Wt - Wt-1)",
    },
}

LOO_RECEIPT_SCHEMA = "openevo.post-advisor.task-vector-leave-one-out-materialization.202609282338"


def materialize_module_factors(torch, a_final, b_final, a_t, b_t, a_prev, b_prev, coefficients):
    """Represent c_f BfAf + c_t BtAt + c_p BpAp by exact rank concatenation."""
    c_final, c_t, c_prev = coefficients
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
    # Source and output LoRA scales are one. Multiplication by these integer
    # doses is exact for the source floating-point factors; the dense audit
    # below rejects any overflow or representation drift.
    b_out = torch.cat((c_final * b_final, c_t * b_t, c_prev * b_prev), dim=1).contiguous()
    return a_out, b_out


def _dense_audit(torch, factors, source_factors, coefficients) -> dict[str, float]:
    c_final, c_t, c_prev = coefficients
    a_final, b_final, a_t, b_t, a_prev, b_prev = source_factors
    a_out, b_out = factors
    expected = (
        c_final * (b_final.double() @ a_final.double())
        + c_t * (b_t.double() @ a_t.double())
        + c_prev * (b_prev.double() @ a_prev.double())
    )
    actual = b_out.double() @ a_out.double()
    difference = actual - expected
    error = float(torch.linalg.vector_norm(difference).item())
    expected_norm = float(torch.linalg.vector_norm(expected).item())
    relative_error = error / expected_norm if expected_norm > 0 else error
    max_abs_error = float(torch.max(torch.abs(difference)).item()) if difference.numel() else 0.0
    if not math.isfinite(relative_error) or relative_error > 1e-10:
        raise RuntimeError(f"rank-concatenated reverse-direction factors failed dense audit: rel={relative_error}")
    return {"relative_frobenius_error": relative_error, "max_abs_error": max_abs_error}


def validate_loo_root(loo_root: Path, expected_sources: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Bind the reverse-direction treatment to sealed leave-one-out artifacts."""
    loo_root = loo_root.resolve()
    receipt_path = loo_root / "materialization-receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"missing sealed leave-one-out receipt: {receipt_path}")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("schema") != LOO_RECEIPT_SCHEMA or receipt.get("status") != "PASS":
        raise ValueError("--loo-root does not contain a PASS leave-one-out materialization receipt")
    if (
        receipt.get("task_panel_accessed") is not False
        or receipt.get("formal_rollouts_consumed") != 0
        or receipt.get("final_panel_access") != 0
    ):
        raise ValueError("leave-one-out receipt crosses the zero-panel/zero-rollout boundary")
    if receipt.get("source_rounds") != {
        str(r): {"state_id": expected_sources[r]["state_id"], "adapter_sha256": expected_sources[r]["adapter_sha256"]}
        for r in leave_one_out.SOURCE_STATES
    }:
        raise ValueError("leave-one-out receipt source identities do not match the pinned E4 source lineage")
    if receipt.get("output_rank") != 384 or receipt.get("modules_per_state") != 112:
        raise ValueError("leave-one-out receipt rank/module contract drift")

    states = receipt.get("states")
    if not isinstance(states, dict):
        raise ValueError("leave-one-out receipt has no state map")
    if set(states) != {str(r) for r in range(152, 160)}:
        raise ValueError("leave-one-out receipt must contain exactly the eight frozen R152-R159 states")
    referenced: dict[str, Any] = {}
    for round_index in range(152, 160):
        output_id = f"leave-one-update-out-r{round_index}"
        state = states.get(str(round_index))
        if not isinstance(state, dict) or state.get("intervention") != f"W159 - (W{round_index} - W{round_index - 1})":
            raise ValueError(f"leave-one-out receipt is missing the frozen R{round_index} intervention")
        if state.get("output_rank") != 384 or state.get("module_count") != 112:
            raise ValueError(f"leave-one-out R{round_index} receipt rank/module contract drift")
        adapter_path = loo_root / output_id / "adapter_model.safetensors"
        config_path = loo_root / output_id / "adapter_config.json"
        if not adapter_path.is_file() or not config_path.is_file():
            raise FileNotFoundError(f"leave-one-out adapter files missing for R{round_index}: {output_id}")
        adapter_sha = leave_one_out.sha256_file(adapter_path)
        config_sha = leave_one_out.sha256_file(config_path)
        if adapter_sha != state.get("adapter_sha256") or config_sha != state.get("config_sha256"):
            raise ValueError(f"leave-one-out R{round_index} artifact hash mismatch")
        config = json.loads(config_path.read_text())
        if int(config.get("r", -1)) != 384 or int(config.get("lora_alpha", -1)) != 384:
            raise ValueError(f"leave-one-out R{round_index} adapter config rank/scale mismatch")
        referenced[str(round_index)] = {
            "output_id": output_id,
            "adapter_sha256": adapter_sha,
            "config_sha256": config_sha,
        }
    return {
        "root": str(loo_root),
        "receipt_path": str(receipt_path),
        "receipt_sha256": leave_one_out.sha256_file(receipt_path),
        "states": referenced,
    }


def _write_intervention(torch, save_file, safe_open, final_tensors, current_tensors,
                        previous_tensors, a_keys, source_record, output_dir: Path,
                        round_index: int, treatment: str) -> dict[str, Any]:
    treatment_spec = INTERVENTIONS[treatment]
    coefficients = treatment_spec["coefficients"]
    result_tensors: dict[str, Any] = {}
    max_relative_error = 0.0
    max_abs_error = 0.0
    for a_key in a_keys:
        b_key = a_key.replace(".lora_A.weight", ".lora_B.weight")
        source_factors = (
            final_tensors[a_key], final_tensors[b_key],
            current_tensors[a_key], current_tensors[b_key],
            previous_tensors[a_key], previous_tensors[b_key],
        )
        a_out, b_out = materialize_module_factors(
            torch, *source_factors, coefficients=coefficients
        )
        if int(a_out.shape[0]) != 384 or int(b_out.shape[1]) != 384:
            raise RuntimeError(f"R{round_index} materialized rank is not 384 for {a_key}")
        audit = _dense_audit(torch, (a_out, b_out), source_factors, coefficients)
        max_relative_error = max(max_relative_error, audit["relative_frobenius_error"])
        max_abs_error = max(max_abs_error, audit["max_abs_error"])
        result_tensors[a_key] = a_out
        result_tensors[b_key] = b_out

    output_id = treatment_spec["output_id"].format(round_index=round_index)
    adapter_out = output_dir / output_id
    adapter_out.mkdir()
    with safe_open(str(source_record["adapter_path"]), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    save_file(result_tensors, str(adapter_out / "adapter_model.safetensors"), metadata=metadata)
    config = dict(source_record["config"])
    config["r"] = 384
    config["lora_alpha"] = 384
    config["rank_pattern"] = {}
    config["alpha_pattern"] = {}
    config["inference_mode"] = True
    (adapter_out / "adapter_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return {
        "output_id": output_id,
        "treatment": treatment,
        "round_index": round_index,
        "formula": treatment_spec["formula"],
        "effective_coefficients_final_t_previous": list(coefficients),
        "output_rank": 384,
        "module_count": len(a_keys),
        "max_dense_relative_frobenius_error": max_relative_error,
        "max_dense_abs_error": max_abs_error,
        "adapter_sha256": leave_one_out.sha256_file(adapter_out / "adapter_model.safetensors"),
        "config_sha256": leave_one_out.sha256_file(adapter_out / "adapter_config.json"),
    }


def materialize_direction_interventions(source_root: Path, loo_root: Path, output_root: Path) -> dict[str, Any]:
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before E4 CPU materialization")
    source_root = source_root.resolve()
    loo_root = loo_root.resolve()
    output_root = output_root.resolve()
    if output_root == source_root or output_root.is_relative_to(source_root):
        raise ValueError("reverse-direction output root must be outside the sealed source run tree")
    if output_root == loo_root or output_root.is_relative_to(loo_root):
        raise ValueError("reverse-direction output root must be outside the sealed leave-one-out tree")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite E4 output (including leave-one-out artifacts): {output_root}")
    records = leave_one_out._load_and_validate_sources(torch, source_root)
    loo_reference = validate_loo_root(loo_root, records)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        final_record = records[159]
        final_tensors = load_file(str(final_record["adapter_path"]), device="cpu")
        a_keys = sorted(key for key in final_tensors if key.endswith(".lora_A.weight"))
        b_keys = {key.replace(".lora_A.weight", ".lora_B.weight") for key in a_keys}
        if len(a_keys) != 112 or len(final_tensors) != 224 or set(final_tensors) != set(a_keys) | b_keys:
            raise ValueError("R159 tensor inventory differs from the sealed 112-module LoRA adapter")

        states: dict[str, Any] = {}
        for round_index in range(152, 160):
            previous_tensors = load_file(str(records[round_index - 1]["adapter_path"]), device="cpu")
            current_tensors = load_file(str(records[round_index]["adapter_path"]), device="cpu")
            if set(previous_tensors) != set(final_tensors) or set(current_tensors) != set(final_tensors):
                raise ValueError(f"R{round_index} tensor inventory differs from R159")
            for treatment in INTERVENTIONS:
                report = _write_intervention(
                    torch, save_file, safe_open, final_tensors, current_tensors,
                    previous_tensors, a_keys, final_record, temp_root, round_index, treatment,
                )
                if report["output_id"] in states:
                    raise RuntimeError(f"duplicate reverse-direction output id: {report['output_id']}")
                report["paired_leave_one_out"] = loo_reference["states"][str(round_index)]
                states[report["output_id"]] = report
            del previous_tensors, current_tensors

        receipt = {
            "schema": "openevo.post-advisor.task-vector-interventions-materialization.202609282338",
            "revision": "202609282338",
            "status": "PASS",
            "source_run_root": str(source_root),
            "source_rounds": {
                str(r): {"state_id": records[r]["state_id"], "adapter_sha256": records[r]["adapter_sha256"]}
                for r in leave_one_out.SOURCE_STATES
            },
            "leave_one_out_reference": loo_reference,
            "interventions": {
                name: {"formula": spec["formula"], "coefficients_final_t_previous": list(spec["coefficients"])}
                for name, spec in INTERVENTIONS.items()
            },
            "representation": "exact rank concatenation; source scale=1; output rank=alpha=384; no SVD or truncation",
            "source_base_model": "Qwen/Qwen3-1.7B",
            "source_rank": 128,
            "output_rank": 384,
            "modules_per_state": len(a_keys),
            "states": states,
            "cpu_only": True,
            "model_loaded": False,
            "inference_started": False,
            "task_panel_accessed": False,
            "formal_rollouts_consumed": 0,
            "final_panel_access": 0,
            "leave_one_out_artifact_modified": False,
            "leave_one_out_adapters_copied_or_rewritten": False,
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
    parser.add_argument("--loo-root", type=Path, required=True, help="existing sealed leave-one-out materialization root")
    parser.add_argument("--output-root", type=Path, required=True, help="new, distinct directory for eight reverse-direction adapters")
    args = parser.parse_args()
    print(json.dumps(materialize_direction_interventions(args.source_root, args.loo_root, args.output_root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
