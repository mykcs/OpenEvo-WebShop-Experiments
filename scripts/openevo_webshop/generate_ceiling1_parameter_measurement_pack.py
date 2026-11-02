#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

SCHEMA = "openevo.ceiling1-parameter-measurement-pack.202609050634"
COMPONENT_RE = re.compile(r"^components\.(\d+)\.(.+)\.([AB])$")
LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty CSV: {path.name}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def dependencies() -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        import torch
        from safetensors import safe_open
    except Exception as exc:
        raise RuntimeError(
            "measurement generation requires numpy, torch, and safetensors; "
            "run inside the declared scientific runtime"
        ) from exc
    return np, torch, safe_open


def spectrum_metrics(values: Any) -> dict[str, Any]:
    vals = [float(v) for v in values if float(v) > 1.0e-12]
    if not vals:
        return {
            "stable_rank": 0.0, "participation_ratio_energy": 0.0,
            "entropy_rank_energy": 0.0, "entropy_rank_singular": 0.0,
            "nuclear_effective_rank": 0.0, "rank90_energy": 0,
            "rank95_energy": 0, "rank99_energy": 0,
            "top1_energy_share": 0.0, "top5_energy_share": 0.0,
            "top10_energy_share": 0.0, "spectral_gap_s1_s2": "",
        }
    energy = [v * v for v in vals]
    total = sum(energy)
    p = [x / total for x in energy]
    ssum = sum(vals)
    ps = [v / ssum for v in vals]

    def entropy_rank(prob: list[float]) -> float:
        return math.exp(-sum(x * math.log(x) for x in prob if x > 0.0))

    def energy_rank(frac: float) -> int:
        acc = 0.0
        for index, value in enumerate(p, 1):
            acc += value
            if acc + 1.0e-15 >= frac:
                return index
        return len(p)

    return {
        "stable_rank": total / energy[0],
        "participation_ratio_energy": 1.0 / sum(x * x for x in p),
        "entropy_rank_energy": entropy_rank(p),
        "entropy_rank_singular": entropy_rank(ps),
        "nuclear_effective_rank": (ssum * ssum) / total,
        "rank90_energy": energy_rank(0.90),
        "rank95_energy": energy_rank(0.95),
        "rank99_energy": energy_rank(0.99),
        "top1_energy_share": sum(p[:1]),
        "top5_energy_share": sum(p[:5]),
        "top10_energy_share": sum(p[:10]),
        "spectral_gap_s1_s2": vals[0] / vals[1] if len(vals) > 1 else "",
    }


def load_components(path: Path, safe_open: Any) -> dict[int, dict[str, tuple[Any, Any]]]:
    pieces: dict[int, dict[str, dict[str, Any]]] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            match = COMPONENT_RE.match(key)
            if not match:
                continue
            index, module, which = int(match.group(1)), match.group(2), match.group(3)
            pieces.setdefault(index, {}).setdefault(module, {})[which] = handle.get_tensor(key).float().cpu()
    if not pieces or sorted(pieces) != list(range(max(pieces) + 1)):
        raise RuntimeError("component tensor indexes are missing or non-contiguous")
    result: dict[int, dict[str, tuple[Any, Any]]] = {}
    for index in sorted(pieces):
        result[index] = {}
        for module, matrices in pieces[index].items():
            if set(matrices) != {"A", "B"}:
                raise RuntimeError(f"incomplete component pair: {index} {module}")
            result[index][module] = (matrices["A"], matrices["B"])
    inventory = set(result[0])
    if any(set(result[index]) != inventory for index in result):
        raise RuntimeError("component module inventories differ")
    return result


def component_gram(torch: Any, components: dict[int, dict[str, tuple[Any, Any]]], modules: list[str]) -> Any:
    count = len(components)
    gram = torch.zeros((count, count), dtype=torch.float64)
    for module in modules:
        for i in range(count):
            ai, bi = components[i][module]
            for j in range(i, count):
                aj, bj = components[j][module]
                ga = ai.double() @ aj.double().transpose(0, 1)
                gb = bi.double().transpose(0, 1) @ bj.double()
                value = float((ga * gb).sum())
                gram[i, j] += value
                if i != j:
                    gram[j, i] += value
    return gram


def module_groups(modules: list[str]) -> dict[tuple[str, str], list[str]]:
    groups: dict[tuple[str, str], list[str]] = {("global", "all"): list(modules)}
    for module in modules:
        layer = LAYER_RE.search(module)
        if layer:
            groups.setdefault(("layer", layer.group(1)), []).append(module)
        projection = module.rsplit(".", 1)[-1]
        if projection in PROJECTIONS:
            groups.setdefault(("projection", projection), []).append(module)
    return groups


def accepted_history(state: dict[str, Any], final_factors: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    checkpoints = sorted(state.get("sd_checkpoints", []), key=lambda x: int(x["component_count"]))
    for checkpoint in checkpoints:
        path = Path(checkpoint["geometry_path"])
        geometry = read_json(path)
        count = int(geometry["component_count"])
        if list(geometry.get("component_factor_sha256", [])) != final_factors[:count]:
            raise RuntimeError(f"append-only component identity mismatch at component_count={count}")
        rows.append({
            "round": int(geometry["round_index"]),
            "component_count": count,
            "generation_index": int(geometry["generation_index"]),
            "current_coefficients": [float(v) for v in geometry["current_coefficients"]],
            "delta_coefficients": [float(v) for v in geometry["delta_coefficients"]],
            "cumulative_delta_norm": float(geometry["cumulative_delta_norm"]),
            "delta_norm": float(geometry["delta_norm"]),
            "path_length": float(geometry["path_length"]),
            "adjacent_update_cosine": geometry.get("adjacent_update_cosine"),
            "cumulative_alignment": geometry.get("cumulative_alignment"),
            "coefficient_effective_component_count": geometry.get("coefficient_effective_component_count"),
            "geometry_sha256": sha256(path),
        })
    if not rows:
        raise RuntimeError("no accepted SD-LoRA checkpoint exists")
    return rows


def metric_svd(torch: Any, matrix: Any, gram: Any) -> tuple[Any, Any, Any]:
    gram = (gram.double() + gram.double().transpose(0, 1)) * 0.5
    eigvals, eigvecs = torch.linalg.eigh(gram)
    scale = max(float(eigvals.max()), 1.0)
    keep = eigvals > scale * 1.0e-10
    sqrt_diag = torch.where(keep, eigvals.clamp_min(0).sqrt(), torch.zeros_like(eigvals))
    inv_diag = torch.where(keep, eigvals.clamp_min(0).rsqrt(), torch.zeros_like(eigvals))
    sqrt_g = (eigvecs * sqrt_diag.unsqueeze(0)) @ eigvecs.transpose(0, 1)
    inv_sqrt_g = (eigvecs * inv_diag.unsqueeze(0)) @ eigvecs.transpose(0, 1)
    u, s, vh = torch.linalg.svd(matrix.double() @ sqrt_g, full_matrices=False)
    return u, s, vh @ inv_sqrt_g


def lowrank_svals(torch: Any, factors: list[tuple[Any, Any]], coefficients: list[float]) -> Any:
    u = torch.cat([b.double() * float(c) for (a, b), c in zip(factors, coefficients, strict=True)], dim=1)
    v = torch.cat([a.double() for (a, b) in factors], dim=0)
    if float(u.abs().max()) == 0.0:
        return torch.zeros(u.shape[1], dtype=torch.float64)
    _, ru = torch.linalg.qr(u, mode="reduced")
    _, rv = torch.linalg.qr(v.transpose(0, 1), mode="reduced")
    return torch.linalg.svdvals(ru @ rv.transpose(0, 1))


def power_spectral(torch: Any, weight: Any, factors: list[tuple[Any, Any]], coefficients: list[float], x: Any | None, iterations: int) -> tuple[float, float, Any]:
    weight = weight.float()
    if x is None or int(x.numel()) != int(weight.shape[1]):
        x = torch.arange(1, weight.shape[1] + 1, dtype=torch.float32)
        x = x / torch.linalg.vector_norm(x)

    def forward(vec: Any) -> Any:
        out = weight @ vec
        for (a, b), coefficient in zip(factors, coefficients, strict=True):
            if coefficient:
                out = out + float(coefficient) * (b @ (a @ vec))
        return out

    def backward(vec: Any) -> Any:
        out = weight.transpose(0, 1) @ vec
        for (a, b), coefficient in zip(factors, coefficients, strict=True):
            if coefficient:
                out = out + float(coefficient) * (a.transpose(0, 1) @ (b.transpose(0, 1) @ vec))
        return out

    for _ in range(max(1, iterations)):
        y = forward(x)
        yn = torch.linalg.vector_norm(y)
        if float(yn) == 0.0:
            return 0.0, 0.0, x
        y = y / yn
        x = backward(y)
        xn = torch.linalg.vector_norm(x)
        if float(xn) == 0.0:
            return 0.0, 0.0, x
        x = x / xn
    raw = forward(x)
    sigma = float(torch.linalg.vector_norm(raw))
    if sigma == 0.0:
        return 0.0, 0.0, x
    y = raw / sigma
    residual = float(torch.linalg.vector_norm(backward(y) - sigma * x) / sigma)
    return sigma, residual, x


class WeightSource:
    def __init__(self, root: Path, safe_open: Any) -> None:
        self.root = root
        self.safe_open = safe_open
        index = root / "model.safetensors.index.json"
        self.weight_map = read_json(index)["weight_map"] if index.is_file() else None
        if self.weight_map is None and not (root / "model.safetensors").is_file():
            raise FileNotFoundError(f"no model safetensors under {root}")
        self.stack = contextlib.ExitStack()
        self.handles: dict[str, Any] = {}

    def __enter__(self) -> "WeightSource":
        self.stack.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stack.__exit__(*args)

    def get(self, key: str) -> Any:
        shard = self.weight_map.get(key) if self.weight_map is not None else "model.safetensors"
        if shard is None:
            raise KeyError(f"base model missing {key}")
        if shard not in self.handles:
            self.handles[shard] = self.stack.enter_context(
                self.safe_open(str(self.root / shard), framework="pt", device="cpu")
            )
        return self.handles[shard].get_tensor(key).float().cpu()


def parameter_tables(torch: Any, components: dict[int, dict[str, tuple[Any, Any]]], history: list[dict[str, Any]], base_root: Path, safe_open: Any, base_iters: int, stage_iters: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    modules = sorted(components[0])
    spectral_rows: list[dict[str, Any]] = []
    w_rows: list[dict[str, Any]] = []
    rms_rows: list[dict[str, Any]] = []
    method = f"deterministic_power_iteration_base{base_iters}_stage{stage_iters}"
    with WeightSource(base_root, safe_open) as weights:
        for module_index, module in enumerate(modules):
            factors = [components[i][module] for i in range(len(components))]
            weight = weights.get(module + ".weight")
            base_fro_sq = float((weight.double() * weight.double()).sum())
            base_fro = math.sqrt(max(base_fro_sq, 0.0))
            base_sigma, base_residual, x = power_spectral(torch, weight, [], [], None, base_iters)
            base_row = {
                "module": module, "round": -1, "component_count": 0, "state": "base",
                "fro_norm": base_fro, "rms": base_fro / math.sqrt(weight.numel()),
                "spectral_norm_est": base_sigma, "spectral_residual": base_residual,
                "stable_rank_est": base_fro_sq / max(base_sigma * base_sigma, 1.0e-30),
                "spectral_method": method,
            }
            w_rows.append(dict(base_row))
            rms_rows.append(dict(base_row))
            for stage in history:
                count = int(stage["component_count"])
                coefficients = list(stage["current_coefficients"])
                active = factors[:count]
                svals = lowrank_svals(torch, active, coefficients)
                sm = spectrum_metrics(svals.tolist())
                delta_fro_sq = float((svals.double() * svals.double()).sum())
                delta_fro = math.sqrt(max(delta_fro_sq, 0.0))
                inner = 0.0
                for (a, b), coefficient in zip(active, coefficients, strict=True):
                    inner += float(coefficient) * float((b.double() * (weight.double() @ a.double().transpose(0, 1))).sum())
                full_fro_sq = max(base_fro_sq + 2.0 * inner + delta_fro_sq, 0.0)
                full_fro = math.sqrt(full_fro_sq)
                full_sigma, residual, x = power_spectral(torch, weight, active, coefficients, x, stage_iters)
                full_stable = full_fro_sq / max(full_sigma * full_sigma, 1.0e-30)
                spectral_rows.append({
                    "module": module, "round": stage["round"], "component_count": count,
                    "fro_norm": delta_fro, "rms": delta_fro / math.sqrt(weight.numel()),
                    "spectral_norm": float(svals[0]) if len(svals) else 0.0,
                    "stable_rank_est": sm["stable_rank"],
                    "nuclear_effective_rank": sm["nuclear_effective_rank"],
                    "rank95_energy": sm["rank95_energy"],
                    "spectral_update_over_base": float(svals[0]) / base_sigma if len(svals) and base_sigma > 0 else "",
                    "fro_update_over_base": delta_fro / base_fro if base_fro > 0 else "",
                    "full_W_spectral_est": full_sigma, "full_W_stable_rank_est": full_stable,
                })
                full_row = {
                    "module": module, "round": stage["round"], "component_count": count,
                    "state": "base_plus_sd_lora", "fro_norm": full_fro,
                    "rms": full_fro / math.sqrt(weight.numel()), "spectral_norm_est": full_sigma,
                    "spectral_residual": residual, "stable_rank_est": full_stable,
                    "delta_fro_norm": delta_fro,
                    "delta_spectral_norm": float(svals[0]) if len(svals) else 0.0,
                    "spectral_method": method,
                }
                w_rows.append(dict(full_row))
                rms_rows.append(dict(full_row))
            if (module_index + 1) % 16 == 0:
                print(json.dumps({"progress_modules": module_index + 1, "module_total": len(modules)}), flush=True)
    return spectral_rows, w_rows, rms_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-power-iters", type=int, default=6)
    parser.add_argument("--stage-power-iters", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    np, torch, safe_open = dependencies()
    torch.set_num_threads(max(1, args.threads))
    stage2_root = args.stage2_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)

    execution_state_path = stage2_root / "execution-state.json"
    state = read_json(execution_state_path)
    checkpoints = sorted(state.get("sd_checkpoints", []), key=lambda x: int(x["component_count"]))
    if not checkpoints:
        raise RuntimeError("no accepted parameter state exists yet")
    latest = checkpoints[-1]
    adapter = Path(latest["path"])
    manifest_path = adapter / "openevo_sd_lora_state.json"
    tensor_path = adapter / "openevo_sd_lora_state.safetensors"
    adapter_manifest = read_json(manifest_path)
    component_count = int(adapter_manifest["component_count"])
    if component_count != int(latest["component_count"]):
        raise RuntimeError("execution-state / adapter component-count mismatch")
    geometry = read_json(Path(latest["geometry_path"]))
    final_factors = list(geometry.get("component_factor_sha256", []))
    if len(final_factors) != component_count:
        raise RuntimeError("final geometry does not bind every component factor")
    base_receipt_path = Path(state["merged_base_receipt"])
    base_root = base_receipt_path.parent
    components = load_components(tensor_path, safe_open)
    if len(components) != component_count:
        raise RuntimeError("state tensor component count mismatch")
    modules = sorted(components[0])
    history = accepted_history(state, final_factors)

    groups = module_groups(modules)
    group_grams = {key: component_gram(torch, components, names) for key, names in groups.items()}
    global_gram = group_grams[("global", "all")]
    expected_gram = torch.tensor(geometry["gram_matrix"], dtype=torch.float64)
    gram_error = float((global_gram - expected_gram).abs().max())
    if gram_error > 2.0e-4:
        raise RuntimeError(f"component Gram reconstruction mismatch: max_abs={gram_error}")
    np.savez_compressed(
        output / "layer_grams.npz",
        **{re.sub(r"[^A-Za-z0-9_]+", "_", f"{kind}__{group}"): gram.numpy()
           for (kind, group), gram in group_grams.items()},
    )

    cumulative = torch.zeros((len(history), component_count), dtype=torch.float64)
    updates = torch.zeros((len(history), component_count), dtype=torch.float64)
    geometry_rows: list[dict[str, Any]] = []
    for row_index, stage in enumerate(history):
        count = int(stage["component_count"])
        cumulative[row_index, :count] = torch.tensor(stage["current_coefficients"], dtype=torch.float64)
        updates[row_index, :count] = torch.tensor(stage["delta_coefficients"], dtype=torch.float64)
        geometry_rows.append({
            "round": stage["round"], "component_count": count,
            "generation_index": stage["generation_index"],
            "cumulative_delta_norm": stage["cumulative_delta_norm"], "delta_norm": stage["delta_norm"],
            "path_length": stage["path_length"], "adjacent_update_cosine": stage["adjacent_update_cosine"],
            "cumulative_alignment": stage["cumulative_alignment"],
            "coefficient_effective_component_count": stage["coefficient_effective_component_count"],
            "current_coefficients_json": json.dumps(stage["current_coefficients"], separators=(",", ":")),
            "delta_coefficients_json": json.dumps(stage["delta_coefficients"], separators=(",", ":")),
            "geometry_sha256": stage["geometry_sha256"],
        })
    write_csv(output / "historical_decomposed_geometry.csv", geometry_rows)

    theme_rows: list[dict[str, Any]] = []
    loading_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    global_summaries: dict[str, Any] = {}
    for (kind, group), gram in group_grams.items():
        for family, matrix in {"cumulative": cumulative, "updates": updates}.items():
            u, s, coefficient_themes = metric_svd(torch, matrix, gram)
            metrics = spectrum_metrics(s.tolist())
            summary_rows.append({
                "group_kind": kind, "group": group, "family": family,
                "stage_count": len(history), "component_count": component_count, **metrics,
            })
            if kind == "global" and group == "all":
                global_summaries[family] = metrics
            energy_total = float((s * s).sum())
            for theme_index in range(len(s)):
                share = float((s[theme_index] * s[theme_index]) / energy_total) if energy_total else 0.0
                for component_index in range(component_count):
                    theme_rows.append({
                        "group_kind": kind, "group": group, "family": family,
                        "theme_index": theme_index, "component_index": component_index,
                        "component_coefficient": float(coefficient_themes[theme_index, component_index]),
                        "singular_value": float(s[theme_index]), "explained_energy_share": share,
                    })
                for stage_index, stage in enumerate(history):
                    loading_rows.append({
                        "group_kind": kind, "group": group, "family": family,
                        "theme_index": theme_index, "round": stage["round"],
                        "component_count": stage["component_count"],
                        "stage_loading": float(u[stage_index, theme_index]),
                        "singular_value": float(s[theme_index]),
                    })
    write_csv(output / "trajectory_spectrum_summary.csv", summary_rows)
    write_csv(output / "principal_parameter_themes.csv", theme_rows)
    write_csv(output / "parameter_theme_stage_loadings.csv", loading_rows)

    spectral_rows, w_rows, rms_rows = parameter_tables(
        torch, components, history, base_root, safe_open,
        args.base_power_iters, args.stage_power_iters,
    )
    write_csv(output / "parameter_spectral_stability.csv", spectral_rows)
    write_csv(output / "parameter_W_spectral_over_time.csv", w_rows)
    write_csv(output / "parameter_W_rms_stablerank_over_time.csv", rms_rows)

    source_round = int(history[-1]["round"])
    source_files = {
        "execution_state_observed": execution_state_path,
        "adapter_manifest": manifest_path,
        "adapter_state_tensors": tensor_path,
        "task_vector_geometry": Path(latest["geometry_path"]),
        "merged_base_receipt": base_receipt_path,
    }
    source_hashes = {name: sha256(path) for name, path in source_files.items()}
    method = f"deterministic_power_iteration_base{args.base_power_iters}_stage{args.stage_power_iters}"
    summary = {
        "schema": SCHEMA, "status": "PASS", "arm": state["arm"],
        "component_count": component_count, "parameter_update_count": component_count,
        "source_round": source_round, "adapter_path": str(adapter),
        "adapter_id": adapter_manifest["adapter_id"],
        "adapter_state_sha256": adapter_manifest["state_weights_sha256"],
        "base_model": adapter_manifest["base_model"], "model_revision": adapter_manifest["model_revision"],
        "component_factor_sha256": final_factors, "source_sha256": source_hashes,
        "global_component_gram_max_abs_error_vs_geometry": gram_error,
        "global_cumulative_spectrum": global_summaries["cumulative"],
        "global_update_spectrum": global_summaries["updates"],
        "accepted_state_history": [
            {"round": row["round"], "component_count": row["component_count"], "geometry_sha256": row["geometry_sha256"]}
            for row in history
        ],
        "analysis_contract": {
            "posthoc_only": True, "formal_rollout_calls_delta": 0,
            "formal_training_calls_delta": 0, "final_panel_access": False,
            "full_W_spectral_method": method,
            "delta_W_spectrum": "exact_thin_low_rank_svd",
            "component_geometry": "exact_frobenius_gram_from_append_only_normalized_SD-LoRA_factors",
        },
    }
    write_json(output / "measurement-pack-summary.json", summary)
    write_json(output / "trajectory-spectrum-manifest.json", {
        "schema": "openevo.ceiling1-trajectory-spectrum-manifest.202609050634",
        "status": "PASS", "arm": state["arm"], "update_count": component_count,
        "through_round": source_round, "component_factor_sha256": final_factors,
        "global_component_gram_max_abs_error_vs_geometry": gram_error,
        "families": ["cumulative", "updates"],
        "groups": [
            {"group_kind": kind, "group": group, "module_count": len(names)}
            for (kind, group), names in groups.items()
        ],
    })
    generated = [
        output / "measurement-pack-summary.json", output / "trajectory-spectrum-manifest.json",
        output / "parameter_spectral_stability.csv", output / "parameter_W_spectral_over_time.csv",
        output / "parameter_W_rms_stablerank_over_time.csv", output / "trajectory_spectrum_summary.csv",
        output / "principal_parameter_themes.csv", output / "parameter_theme_stage_loadings.csv",
        output / "historical_decomposed_geometry.csv", output / "layer_grams.npz",
    ]
    write_json(output / "GENERATION_RECEIPT.json", {
        "schema": "openevo.ceiling1-measurement-pack-generation-receipt.202609050634",
        "status": "PASS", "arm": state["arm"], "parameter_update_count": component_count,
        "source_round": source_round, "output": str(output),
        "files": [{"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in generated],
        "source_sha256": source_hashes, "posthoc_only": True, "final_panel_access": False,
    })
    print(json.dumps({
        "status": "PASS", "arm": state["arm"], "parameter_update_count": component_count,
        "source_round": source_round, "output": str(output), "gram_error": gram_error,
        "file_count": len(generated) + 1,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
