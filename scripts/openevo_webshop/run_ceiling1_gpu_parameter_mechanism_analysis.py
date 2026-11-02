#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any

SCHEMA = "openevo.ceiling1-gpu-parameter-mechanism-analysis.202609050634"
MODULE_RE = re.compile(r"layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import numpy as np
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError("GPU mechanism analysis requires numpy, torch, peft, transformers") from exc
    return np, torch, PeftModel, (AutoModelForCausalLM, AutoTokenizer)


def gpu_identity(torch: Any) -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected exactly one scheduler-assigned CUDA device, got {torch.cuda.device_count()}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=False,
    )
    rows = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    return {
        "cuda_visible_devices": visible,
        "torch_device_count": torch.cuda.device_count(),
        "torch_device_name": torch.cuda.get_device_name(0),
        "nvidia_smi_visible_rows": rows,
    }


def deterministic_replay(path: Path, samples: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = row.get("metadata") or {}
            identity_fields = {
                "event_id": metadata.get("event_id"),
                "session_id": metadata.get("session_id"),
                "task_id": metadata.get("task_id"),
                "trace_index": metadata.get("trace_index"),
            }
            if all(value is None for value in identity_fields.values()):
                identity_text = line.strip()
            else:
                identity_text = json.dumps(identity_fields, sort_keys=True, separators=(",", ":"))
            row["_selection_key"] = hashlib.sha256(identity_text.encode("utf-8")).hexdigest()
            rows.append(row)
    if not rows:
        raise RuntimeError("empty SD-LoRA replay")
    rows.sort(key=lambda row: row["_selection_key"])
    return rows[: min(samples, len(rows))]


def replay_identity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        metadata = row.get("metadata") or {}
        output.append({
            "selection_key": row["_selection_key"],
            "event_id": metadata.get("event_id"),
            "session_id": metadata.get("session_id"),
            "task_id": metadata.get("task_id"),
            "trace_index": metadata.get("trace_index"),
            "reward": metadata.get("reward"),
        })
    return output


def input_for(tok: Any, row: dict[str, Any], max_len: int, target_tokens: int, torch: Any) -> tuple[Any, int, list[int]]:
    messages = list(row["messages"])
    target_start = int(row.get("target_message_start", len(messages) - 1))
    prompt_messages = messages[:target_start]
    target_messages = messages[target_start:]
    if not target_messages:
        raise RuntimeError("replay row has no target message")
    target_text = str(target_messages[0].get("content") or "")
    prompt = tok.apply_chat_template(prompt_messages, tokenize=True, add_generation_prompt=True)
    full = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if len(full) >= len(prompt) and full[: len(prompt)] == prompt:
        target = full[len(prompt): len(prompt) + target_tokens]
    else:
        target = tok(target_text, add_special_tokens=False)["input_ids"][:target_tokens]
    if not target:
        raise RuntimeError("empty target after chat-template tokenization")
    ids = prompt + target
    if len(ids) > max_len:
        overflow = len(ids) - max_len
        prompt = prompt[overflow:] if overflow < len(prompt) else prompt[-1:]
        ids = (prompt + target)[-max_len:]
    plen = len(ids) - len(target)
    return torch.tensor(ids, device="cuda").unsqueeze(0), plen, target


def metric_norm(np: Any, gram: Any, vector: Any) -> float:
    return float(math.sqrt(max(float(vector @ gram @ vector), 0.0)))


def load_theme_contributions(np: Any, measurement_root: Path, component_count: int) -> tuple[Any, list[dict[str, Any]]]:
    grams = np.load(measurement_root / "layer_grams.npz")
    if "global__all" not in grams:
        raise RuntimeError("global component Gram missing from measurement pack")
    gram = np.array(grams["global__all"], dtype=float)
    theme_rows = list(csv.DictReader((measurement_root / "principal_parameter_themes.csv").open()))
    loading_rows = list(csv.DictReader((measurement_root / "parameter_theme_stage_loadings.csv").open()))
    themes: list[dict[str, Any]] = []
    indexes = sorted({
        int(row["theme_index"])
        for row in theme_rows
        if row["group_kind"] == "global" and row["group"] == "all" and row["family"] == "cumulative"
    })
    for index in indexes:
        coeff_rows = sorted(
            [row for row in theme_rows if row["group_kind"] == "global" and row["group"] == "all" and row["family"] == "cumulative" and int(row["theme_index"]) == index],
            key=lambda row: int(row["component_index"]),
        )
        vector = np.array([float(row["component_coefficient"]) for row in coeff_rows[:component_count]], dtype=float)
        singular_value = float(coeff_rows[0]["singular_value"])
        match = [
            row for row in loading_rows
            if row["group_kind"] == "global" and row["group"] == "all" and row["family"] == "cumulative"
            and int(row["theme_index"]) == index and int(row["component_count"]) == component_count
        ]
        if len(match) != 1:
            raise RuntimeError(f"expected one final stage loading for theme {index}, got {len(match)}")
        loading = float(match[0]["stage_loading"])
        contribution = singular_value * loading * vector
        themes.append({
            "theme_index": index,
            "vector": vector,
            "singular_value": singular_value,
            "stage_loading": loading,
            "contribution": contribution,
            "contribution_norm": metric_norm(np, gram, contribution),
            "direction_norm": metric_norm(np, gram, vector),
        })
    themes.sort(key=lambda row: row["contribution_norm"], reverse=True)
    return gram, themes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2-root", type=Path, required=True)
    parser.add_argument("--measurement-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--target-tokens", type=int, default=48)
    parser.add_argument("--top-themes", type=int, default=3)
    parser.add_argument("--layer-bands", type=int, default=4)
    args = parser.parse_args()

    np, torch, PeftModel, transformers = dependencies()
    AutoModelForCausalLM, AutoTokenizer = transformers
    device_identity = gpu_identity(torch)
    stage2_root = args.stage2_root.resolve()
    measurement_root = args.measurement_root.resolve()
    if args.out.exists():
        raise FileExistsError(f"output already exists: {args.out}")
    args.out.mkdir(parents=True)

    pack = read_json(measurement_root / "measurement-pack-summary.json")
    if pack.get("status") != "PASS":
        raise RuntimeError("CPU parameter measurement pack is not PASS")
    component_count = int(pack["parameter_update_count"])
    source_round = int(pack["source_round"])
    state = read_json(stage2_root / "execution-state.json")
    checkpoint = next(
        (row for row in state.get("sd_checkpoints", []) if int(row["component_count"]) == component_count),
        None,
    )
    if checkpoint is None or int(read_json(Path(checkpoint["geometry_path"]))["round_index"]) != source_round:
        raise RuntimeError("measurement pack no longer maps to accepted checkpoint")
    adapter = Path(checkpoint["path"])
    adapter_manifest = read_json(adapter / "openevo_sd_lora_state.json")
    if adapter_manifest["state_weights_sha256"] != pack["adapter_state_sha256"]:
        raise RuntimeError("adapter tensor identity drift")
    current = np.array([float(row["coefficient"]) for row in adapter_manifest["components"]], dtype=float)
    if len(current) != component_count:
        raise RuntimeError("coefficient count drift")
    base_root = Path(state["merged_base_receipt"]).parent
    replay_path = adapter / "openevo_sd_lora_replay.jsonl"
    selected = deterministic_replay(replay_path, args.samples)
    selected_identity = replay_identity(selected)

    tok = AutoTokenizer.from_pretrained(base_root, local_files_only=True)
    model0 = AutoModelForCausalLM.from_pretrained(
        base_root, torch_dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa"
    ).cuda().eval()
    model = PeftModel.from_pretrained(
        model0, str(adapter), is_trainable=False, local_files_only=True, autocast_adapter_dtype=False
    ).eval()
    base = model.get_base_model()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    modules: list[dict[str, Any]] = []
    adapter_config = read_json(adapter / "adapter_config.json")
    configured_total_rank = int(adapter_config["r"])
    rank: int | None = None
    for name, module in model.named_modules():
        if not hasattr(module, "lora_B") or "default" not in getattr(module, "lora_B", {}):
            continue
        match = MODULE_RE.search(name)
        if not match:
            continue
        master = module.lora_B["default"].weight.detach().clone()
        if master.shape[1] % component_count != 0:
            raise RuntimeError(f"LoRA-B width is not divisible by component count for {name}: {tuple(master.shape)}")
        module_rank = int(master.shape[1] // component_count)
        if rank is None:
            rank = module_rank
        elif rank != module_rank:
            raise RuntimeError(f"per-component rank drift for {name}: {module_rank} vs {rank}")
        if int(master.shape[1]) != configured_total_rank:
            raise RuntimeError(f"PEFT configured total rank drift for {name}: {master.shape[1]} vs {configured_total_rank}")
        modules.append({
            "name": name, "layer": int(match.group(1)), "projection": match.group(2),
            "module": module, "master": master,
        })
    if not modules or rank is None:
        raise RuntimeError("no LoRA modules discovered")
    layer_count = max(row["layer"] for row in modules) + 1

    def set_coefficients(coefficients: Any) -> None:
        ratio = torch.zeros(component_count * rank, device="cuda", dtype=torch.float32)
        for index in range(component_count):
            if abs(float(current[index])) < 1.0e-20:
                if abs(float(coefficients[index])) > 1.0e-20:
                    raise RuntimeError("cannot rescale a zero-current component")
                continue
            ratio[index * rank:(index + 1) * rank] = float(coefficients[index]) / float(current[index])
        with torch.no_grad():
            for row in modules:
                row["module"].lora_B["default"].weight.copy_(
                    (row["master"].float() * ratio[None, :]).to(row["master"].dtype)
                )

    def eval_panel(coefficients: Any, hidden: bool = False) -> tuple[list[dict[str, Any]], dict[int, list[Any]]]:
        set_coefficients(coefficients)
        metrics: list[dict[str, Any]] = []
        hidden_rows: dict[int, list[Any]] = {}
        with torch.inference_mode():
            for sample_index, row in enumerate(selected):
                ids, plen, target = input_for(tok, row, args.max_len, args.target_tokens, torch)
                output = base.model(
                    input_ids=ids, use_cache=False, return_dict=True, output_hidden_states=hidden
                )
                positions = torch.arange(plen - 1, plen - 1 + len(target), device="cuda")
                logits = base.lm_head(output.last_hidden_state[0, positions, :]).float()
                targets = torch.tensor(target, device="cuda")
                nll = float(torch.nn.functional.cross_entropy(logits, targets).item())
                vals, indexes = torch.topk(logits[0], 2)
                metrics.append({
                    "sample_index": sample_index, "nll": nll,
                    "first_target_margin": float((vals[0] - vals[1]).item()),
                    "first_target_top1": int(indexes[0]),
                })
                if hidden:
                    chosen_layers = sorted(set([0, layer_count // 4, layer_count // 2, 3 * layer_count // 4, layer_count]))
                    for hidden_index in chosen_layers:
                        tensor = output.hidden_states[hidden_index][0, plen:, :].float().mean(dim=0).cpu()
                        hidden_rows.setdefault(hidden_index, []).append(tensor)
        return metrics, hidden_rows

    gram, themes = load_theme_contributions(np, measurement_root, component_count)
    selected_themes = themes[: min(args.top_themes, len(themes))]
    current_norm = metric_norm(np, gram, current)
    epsilon = max(current_norm * 0.10, 1.0e-4)

    full_metrics, full_hidden = eval_panel(current, hidden=True)
    base_metrics, base_hidden = eval_panel(np.zeros_like(current), hidden=True)
    full_nll = np.array([row["nll"] for row in full_metrics], dtype=float)
    full_top = np.array([row["first_target_top1"] for row in full_metrics], dtype=int)

    hidden_drift: list[dict[str, Any]] = []
    for hidden_index in sorted(full_hidden):
        per_sample = []
        for full_tensor, base_tensor in zip(full_hidden[hidden_index], base_hidden[hidden_index], strict=True):
            diff = full_tensor - base_tensor
            denom = float(torch.linalg.vector_norm(base_tensor)) + 1.0e-30
            cosine = float(torch.nn.functional.cosine_similarity(full_tensor, base_tensor, dim=0).item())
            per_sample.append((float(torch.linalg.vector_norm(diff)), float(torch.linalg.vector_norm(diff)) / denom, cosine))
        hidden_drift.append({
            "hidden_state_index": hidden_index,
            "sample_count": len(per_sample),
            "l2_drift_mean": float(np.mean([x[0] for x in per_sample])),
            "relative_l2_drift_mean": float(np.mean([x[1] for x in per_sample])),
            "cosine_mean": float(np.mean([x[2] for x in per_sample])),
        })

    theme_ablation: list[dict[str, Any]] = []
    base_nll = np.array([row["nll"] for row in base_metrics], dtype=float)
    base_top = np.array([row["first_target_top1"] for row in base_metrics], dtype=int)
    theme_ablation.append({
        "configuration": "all_sd_off", "theme_index": "",
        "task_vector_norm": 0.0, "distance_from_full_norm": current_norm,
        "nll_mean": float(base_nll.mean()), "delta_nll_vs_full": float((base_nll - full_nll).mean()),
        "top1_flip_rate_vs_full": float(np.mean(base_top != full_top)),
    })
    for theme in selected_themes:
        ablated = current - theme["contribution"]
        metrics, _ = eval_panel(ablated)
        nll = np.array([row["nll"] for row in metrics], dtype=float)
        top = np.array([row["first_target_top1"] for row in metrics], dtype=int)
        theme_ablation.append({
            "configuration": f"remove_theme_{theme['theme_index']}",
            "theme_index": theme["theme_index"],
            "task_vector_norm": metric_norm(np, gram, ablated),
            "distance_from_full_norm": theme["contribution_norm"],
            "nll_mean": float(nll.mean()), "delta_nll_vs_full": float((nll - full_nll).mean()),
            "top1_flip_rate_vs_full": float(np.mean(top != full_top)),
        })

    sensitivity: list[dict[str, Any]] = []
    full_mean = float(full_nll.mean())
    for theme in selected_themes:
        direction = theme["vector"]
        direction_norm = max(float(theme["direction_norm"]), 1.0e-30)
        direction = direction / direction_norm
        plus, _ = eval_panel(current + epsilon * direction)
        minus, _ = eval_panel(current - epsilon * direction)
        plus_mean = float(np.mean([row["nll"] for row in plus]))
        minus_mean = float(np.mean([row["nll"] for row in minus]))
        sensitivity.append({
            "theme_index": theme["theme_index"], "epsilon_frobenius": epsilon,
            "nll_full": full_mean, "nll_plus": plus_mean, "nll_minus": minus_mean,
            "directional_sensitivity": (plus_mean - minus_mean) / (2.0 * epsilon),
            "directional_curvature": (plus_mean - 2.0 * full_mean + minus_mean) / (epsilon * epsilon),
        })

    band_edges = [round(index * layer_count / args.layer_bands) for index in range(args.layer_bands + 1)]
    topology: list[dict[str, Any]] = []
    set_coefficients(current)
    for band_index in range(args.layer_bands):
        low, high = band_edges[band_index], band_edges[band_index + 1] - 1
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            set_coefficients(current)
            with torch.no_grad():
                for row in modules:
                    if low <= row["layer"] <= high and row["projection"] == projection:
                        row["module"].lora_B["default"].weight.zero_()
            # Evaluate the already-mutated model directly; eval_panel() intentionally restores coefficients.
            with torch.no_grad():
                for row in modules:
                    if low <= row["layer"] <= high and row["projection"] == projection:
                        row["module"].lora_B["default"].weight.zero_()
                compact = []
                for sample_index, replay_row in enumerate(selected):
                    ids, plen, target = input_for(tok, replay_row, args.max_len, args.target_tokens, torch)
                    output = base.model(input_ids=ids, use_cache=False, return_dict=True)
                    positions = torch.arange(plen - 1, plen - 1 + len(target), device="cuda")
                    logits = base.lm_head(output.last_hidden_state[0, positions, :]).float()
                    targets = torch.tensor(target, device="cuda")
                    compact.append({
                        "nll": float(torch.nn.functional.cross_entropy(logits, targets).item()),
                        "top1": int(torch.argmax(logits[0]).item()),
                    })
            nll = np.array([row["nll"] for row in compact], dtype=float)
            top = np.array([row["top1"] for row in compact], dtype=int)
            topology.append({
                "layer_low": low, "layer_high": high, "projection": projection,
                "delta_nll_vs_full": float((nll - full_nll).mean()),
                "nll_mean": float(nll.mean()),
                "top1_flip_rate_vs_full": float(np.mean(top != full_top)),
            })
            print(json.dumps({"band": [low, high], "projection": projection, "delta_nll": topology[-1]["delta_nll_vs_full"]}), flush=True)

    write_csv(args.out / "gpu_hidden_state_drift.csv", hidden_drift)
    write_csv(args.out / "gpu_theme_ablation.csv", theme_ablation)
    write_csv(args.out / "gpu_theme_sensitivity_curvature.csv", sensitivity)
    write_csv(args.out / "gpu_layer_projection_ablation.csv", topology)
    analysis_summary = {
        "schema": SCHEMA, "status": "PASS", "arm": pack["arm"],
        "parameter_update_count": component_count, "source_round": source_round,
        "measurement_pack_sha256": sha256(measurement_root / "GENERATION_RECEIPT.json"),
        "adapter_state_sha256": pack["adapter_state_sha256"],
        "replay_path": str(replay_path), "replay_sha256": sha256(replay_path),
        "selected_replay": selected_identity, "sample_count": len(selected),
        "gpu_identity": device_identity, "current_task_vector_norm": current_norm,
        "sensitivity_epsilon_frobenius": epsilon,
        "selected_themes": [
            {key: value for key, value in theme.items() if key not in {"vector", "contribution"}}
            for theme in selected_themes
        ],
        "hidden_state_drift": hidden_drift,
        "theme_ablation": theme_ablation,
        "theme_sensitivity_curvature": sensitivity,
        "layer_projection_ablation": topology,
        "analysis_contract": {
            "posthoc_only": True, "formal_rollout_calls_delta": 0,
            "formal_training_calls_delta": 0, "final_panel_access": False,
            "panel_source": "accepted_sd_lora_training_replay",
        },
    }
    write_json(args.out / "GPU_ANALYSIS_SUMMARY.json", analysis_summary)
    generated = [
        args.out / "gpu_hidden_state_drift.csv", args.out / "gpu_theme_ablation.csv",
        args.out / "gpu_theme_sensitivity_curvature.csv", args.out / "gpu_layer_projection_ablation.csv",
        args.out / "GPU_ANALYSIS_SUMMARY.json",
    ]
    write_json(args.out / "GPU_ANALYSIS_RECEIPT.json", {
        "schema": "openevo.ceiling1-gpu-parameter-mechanism-receipt.202609050634",
        "status": "PASS", "arm": pack["arm"], "parameter_update_count": component_count,
        "source_round": source_round, "posthoc_only": True, "final_panel_access": False,
        "formal_rollout_calls_delta": 0, "formal_training_calls_delta": 0,
        "files": [{"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in generated],
        "measurement_pack_sha256": sha256(measurement_root / "GENERATION_RECEIPT.json"),
        "replay_sha256": sha256(replay_path), "gpu_identity": device_identity,
    })
    print(json.dumps({
        "status": "PASS", "arm": pack["arm"], "parameter_update_count": component_count,
        "source_round": source_round, "samples": len(selected), "out": str(args.out),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
