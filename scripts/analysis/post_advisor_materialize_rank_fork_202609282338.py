#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def compress_pair(torch, a, b, target_rank: int, device: str):
    a = a.float().contiguous().to(device)
    b = b.float().contiguous().to(device)
    qa, ra = torch.linalg.qr(a.T, mode="reduced")
    qb, rb = torch.linalg.qr(b, mode="reduced")
    u, s, vh = torch.linalg.svd(rb @ ra.T, full_matrices=False)
    rr = min(int(target_rank), int(s.numel()))
    sq = s[:rr].sqrt()
    b_new = (qb @ (u[:, :rr] * sq.unsqueeze(0))).contiguous()
    a_new = ((sq.unsqueeze(1) * vh[:rr, :]) @ qa.T).contiguous()
    total = float((s * s).sum().item())
    rem = float((s[rr:] * s[rr:]).sum().item())
    residual = math.sqrt(rem / total) if total else 0.0
    retained = 1.0 - residual * residual
    next_sv = float(s[rr].item()) if rr < int(s.numel()) else 0.0
    return a_new.cpu(), b_new.cpu(), residual, retained, next_sv


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--target-rank", type=int, choices=(8, 16, 32, 64), required=True)
    p.add_argument("--source-state-id", required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = p.parse_args()

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise SystemExit("output directory already exists")
    model_path = source / "adapter_model.safetensors"
    config_path = source / "adapter_config.json"
    replay_path = source / "openevo_sd_lora_replay.jsonl"
    for required in (model_path, config_path, replay_path):
        if not required.is_file():
            raise SystemExit(f"missing source asset: {required}")

    tensors = load_file(str(model_path), device="cpu")
    a_keys = sorted(k for k in tensors if k.endswith(".lora_A.weight"))
    if not a_keys:
        raise SystemExit("no LoRA factors found")
    out = {}
    rows = []
    source_rank = None
    for a_key in a_keys:
        b_key = a_key.replace(".lora_A.weight", ".lora_B.weight")
        if b_key not in tensors:
            raise SystemExit(f"missing paired factor: {b_key}")
        a = tensors[a_key]
        b = tensors[b_key]
        rank = int(a.shape[0])
        if int(b.shape[1]) != rank:
            raise SystemExit(f"factor rank mismatch: {a_key}")
        source_rank = rank if source_rank is None else source_rank
        if rank != source_rank:
            raise SystemExit("mixed source ranks")
        if args.target_rank >= rank:
            raise SystemExit("target rank must be lower than source rank")
        a_new, b_new, residual, retained, next_sv = compress_pair(torch, a, b, args.target_rank, args.device)
        out[a_key] = a_new
        out[b_key] = b_new
        rows.append({
            "a_key": a_key,
            "b_key": b_key,
            "source_rank": rank,
            "target_rank": args.target_rank,
        "compute_device": args.device,
            "relative_frobenius_residual": residual,
            "retained_energy_fraction": retained,
            "next_singular_value": next_sv,
        })

    with safe_open(str(model_path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    output.mkdir(parents=True)
    save_file(out, str(output / "adapter_model.safetensors"), metadata=metadata)

    config = json.loads(config_path.read_text())
    source_alpha = int(config.get("lora_alpha", source_rank))
    config["r"] = args.target_rank
    config["lora_alpha"] = args.target_rank
    config["rank_pattern"] = {}
    config["alpha_pattern"] = {}
    config["inference_mode"] = True
    write_json(output / "adapter_config.json", config)
    shutil.copyfile(replay_path, output / "openevo_sd_lora_replay.jsonl")

    residuals = [row["relative_frobenius_residual"] for row in rows]
    retained = [row["retained_energy_fraction"] for row in rows]
    receipt = {
        "schema": "openevo.post-advisor.rank-materialization.202609282338",
        "status": "PASS",
        "source_dir": str(source),
        "source_state_id": args.source_state_id,
        "source_adapter_sha256": sha256_file(model_path),
        "source_rank": source_rank,
        "source_lora_alpha": source_alpha,
        "source_alpha_over_rank": source_alpha / source_rank,
        "target_rank": args.target_rank,
        "target_lora_alpha": args.target_rank,
        "target_alpha_over_rank": 1.0,
        "module_count": len(rows),
        "relative_residual_max": max(residuals),
        "relative_residual_median": float(torch.tensor(residuals).median().item()),
        "retained_energy_fraction_min": min(retained),
        "retained_energy_fraction_mean": sum(retained) / len(retained),
        "retained_energy_fraction_median": float(torch.tensor(retained).median().item()),
        "output_adapter_sha256": sha256_file(output / "adapter_model.safetensors"),
        "output_config_sha256": sha256_file(output / "adapter_config.json"),
        "output_replay_sha256": sha256_file(output / "openevo_sd_lora_replay.jsonl"),
        "historical_source_mutated": False,
        "formal_rollouts_consumed": 0,
        "final_panel_access": 0,
        "modules": rows,
    }
    write_json(output / "compression-receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
