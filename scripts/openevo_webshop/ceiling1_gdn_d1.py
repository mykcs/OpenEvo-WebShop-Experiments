#!/usr/bin/env python3
"""Preregistered D1 prospective Task-Vector bounded-state diagnostic.

This tool is analysis-only. It never trains, calls a teacher, or steps WebShop.
Heavy numeric/model dependencies are imported lazily so CPU/static CI can test
selection and classification without installing the scientific runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Iterable


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in paths:
        text = path.read_text()
        if path.suffix == ".json":
            value = json.loads(text)
            rows = value if isinstance(value, list) else value.get("rows", [])
            out.extend(dict(x) for x in rows)
        else:
            out.extend(json.loads(line) for line in text.splitlines() if line.strip())
    return out


def row_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("task_id"),
        int(row.get("source_schedule_position", -1)),
        int(row.get("step_idx", row.get("source_step", -1))),
        row.get("trajectory_id", row.get("session_id")),
    )


def select_probe_rows(rows: list[dict[str, Any]], *, tasks: int = 16, per_task: int = 2) -> list[dict[str, Any]]:
    dedup: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        if not row.get("task_id") or not row.get("prompt_messages") or not row.get("response_messages"):
            continue
        if row.get("final_panel") or row.get("heldout_final"):
            raise ValueError("final-panel row is forbidden in D1 probe source")
        dedup.setdefault(row_identity(row), row)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in dedup.values():
        grouped.setdefault(str(row["task_id"]), []).append(row)
    eligible = []
    for task_id, group in grouped.items():
        group.sort(key=lambda r: (int(r.get("source_schedule_position", -1)), int(r.get("step_idx", r.get("source_step", -1))), str(r.get("trajectory_id", r.get("session_id", "")))))
        if len(group) >= per_task:
            eligible.append((int(group[0].get("source_schedule_position", -1)), task_id, group))
    eligible.sort(key=lambda x: (x[0], x[1]))
    if len(eligible) < tasks:
        return []
    chosen: list[dict[str, Any]] = []
    for _, _, group in eligible[:tasks]:
        chosen.extend(group[:per_task])
    return chosen


def cmd_build_probe(args: argparse.Namespace) -> int:
    prereg = load_json(args.prereg)
    rows = read_rows(args.records)
    selected = select_probe_rows(
        rows,
        tasks=int(prereg["probe"]["required_distinct_task_identities"]),
        per_task=int(prereg["probe"]["rows_per_task"]),
    )
    if not selected:
        status = "D1_INSUFFICIENT_PROBE_NOT_A_FAILURE"
        atomic_json(args.manifest, {
            "schema": "openevo.ceiling1-gdn-d1-probe-manifest.v1",  # legacy-revision-compat: historical serialized schema identity
            "status": status,
            "source_files": [{"path": str(p.resolve()), "sha256": sha256(p)} for p in args.records],
            "eligible_source_rows": len(rows),
            "final_panel_access": 0,
            "new_environment_calls": 0,
            "external_teacher_calls": 0,
        })
        return 3
    atomic_json(args.output, selected)
    manifest = {
        "schema": "openevo.ceiling1-gdn-d1-probe-manifest.v1",  # legacy-revision-compat: historical serialized schema identity
        "status": "PASS",
        "prereg_sha256": sha256(args.prereg),
        "source_files": [{"path": str(p.resolve()), "sha256": sha256(p)} for p in args.records],
        "row_count": len(selected),
        "distinct_tasks": len({r["task_id"] for r in selected}),
        "panel_path": str(args.output.resolve()),
        "panel_sha256": sha256(args.output),
        "selection_rule": prereg["probe"]["selection_rule"],
        "final_panel_access": 0,
        "new_environment_calls": 0,
        "external_teacher_calls": 0,
    }
    atomic_json(args.manifest, manifest)
    print(json.dumps({"status": "PASS", "rows": len(selected), "tasks": manifest["distinct_tasks"], "panel_sha256": manifest["panel_sha256"]}, sort_keys=True))
    return 0


def _pinv(a, rtol: float):
    import numpy as np
    return np.linalg.pinv((a + a.T) / 2.0, rcond=rtol, hermitian=True)


def _spectrum(gram) -> dict[str, Any]:
    import numpy as np
    vals = np.linalg.eigvalsh((gram + gram.T) / 2.0)[::-1]
    vals = np.clip(vals, 0.0, None)
    vals = vals[vals > 1e-12]
    if len(vals) == 0:
        return {"eigenvalues": [], "rank90": 0, "rank95": 0, "rank99": 0, "stable_rank": 0.0}
    total = float(vals.sum())
    cumulative = np.cumsum(vals) / total
    rank = lambda q: int(np.searchsorted(cumulative, q) + 1)
    return {
        "eigenvalues": [float(x) for x in vals],
        "trace": total,
        "stable_rank": total / float(vals[0]),
        "rank90": rank(0.90), "rank95": rank(0.95), "rank99": rank(0.99),
    }


def _sequential_residuals(gram, rtol: float) -> list[dict[str, float]]:
    import numpy as np
    gram = (gram + gram.T) / 2.0
    out = []
    for j in range(gram.shape[0]):
        norm2 = max(0.0, float(gram[j, j]))
        projected = 0.0
        if j and norm2 > 0.0:
            b = gram[:j, j]
            projected = float(b @ _pinv(gram[:j, :j], rtol) @ b)
            projected = min(norm2, max(0.0, projected))
        residual = max(0.0, norm2 - projected)
        out.append({
            "index": j,
            "norm2": norm2,
            "explained_energy_fraction": projected / norm2 if norm2 else 0.0,
            "residual_energy_fraction": residual / norm2 if norm2 else 0.0,
            "residual_norm_fraction": math.sqrt(residual / norm2) if norm2 else 0.0,
        })
    return out


def _median(values: list[float]) -> float:
    values = sorted(values)
    n = len(values)
    if not n:
        return float("nan")
    if n % 2:
        return float(values[n // 2])
    return float((values[n // 2 - 1] + values[n // 2]) / 2.0)


def _load_accepted_geometry(manifest_path: Path):
    manifest = load_json(manifest_path)
    if manifest.get("schema") != "openevo.ceiling1-gdn-d1-accepted-geometry-manifest.v1":  # legacy-revision-compat: historical serialized schema identity
        raise ValueError("wrong accepted-geometry manifest schema")
    if manifest.get("stage2_sealed") is not True:
        raise ValueError("D1 requires a sealed Stage2 arm")
    if int(manifest.get("final_panel_access_count", -1)) != 0:
        raise ValueError("D1 must begin before any final-panel access")
    if int(manifest.get("external_teacher_calls_delta", -1)) != 0 or int(manifest.get("new_environment_calls_delta", -1)) != 0:
        raise ValueError("D1 provenance forbids new teacher/environment calls")
    entries = manifest.get("entries") or []
    rows = []
    for entry in entries:
        p = Path(entry["geometry_path"])
        if sha256(p) != entry["geometry_sha256"]:
            raise ValueError(f"geometry SHA mismatch: {p}")
        row = load_json(p)
        if row.get("status") != "PASS":
            raise ValueError(f"non-PASS geometry: {p}")
        if int(row["component_count"]) != int(entry["component_count"]):
            raise ValueError("manifest/geometry component count mismatch")
        rows.append(row)
    rows.sort(key=lambda r: int(r["component_count"]))
    counts = [int(r["component_count"]) for r in rows]
    if counts and counts != list(range(1, counts[-1] + 1)):
        raise ValueError("accepted component counts must be exactly sequential 1..N")
    return manifest, rows


def cmd_geometry(args: argparse.Namespace) -> int:
    import numpy as np
    prereg = load_json(args.prereg)
    provenance, rows = _load_accepted_geometry(args.accepted_geometry_manifest)
    minimum = int(prereg["activation"]["minimum_accepted_parametric_updates_for_primary_test"])
    if len(rows) < minimum:
        out = {
            "schema": "openevo.ceiling1-gdn-d1-geometry.v1",  # legacy-revision-compat: historical serialized schema identity
            "status": "D1_INSUFFICIENT_PARAMETER_UPDATES_NOT_A_FAILURE",
            "accepted_parametric_updates": len(rows),
            "required": minimum,
            "prereg_sha256": sha256(args.prereg),
            "accepted_geometry_manifest_sha256": sha256(args.accepted_geometry_manifest),
        }
        atomic_json(args.output, out)
        return 3
    n = len(rows)
    final = rows[-1]
    gram = np.asarray(final["gram_matrix"], dtype=np.float64)
    if gram.shape != (n, n):
        raise ValueError(f"final component Gram must be {n}x{n}, got {gram.shape}")
    gram = (gram + gram.T) / 2.0
    deltas = np.zeros((n, n), dtype=np.float64)
    for i, row in enumerate(rows):
        delta = np.asarray(row["delta_coefficients"], dtype=np.float64)
        if len(delta) != i + 1:
            raise ValueError("delta coefficient length must equal component_count")
        deltas[: i + 1, i] = delta
    final_coeff = np.asarray(final["current_coefficients"], dtype=np.float64)
    if len(final_coeff) != n:
        raise ValueError("final coefficient count mismatch")
    qnorm2 = lambda x: float(x.T @ gram @ x)
    telescope = deltas.sum(axis=1)
    tel_rel = math.sqrt(max(0.0, qnorm2(telescope - final_coeff))) / max(1e-30, math.sqrt(max(0.0, qnorm2(final_coeff))))
    task_gram = deltas.T @ gram @ deltas
    task_gram = (task_gram + task_gram.T) / 2.0
    residuals = _sequential_residuals(task_gram, float(prereg["geometry"]["pinv_relative_tolerance"]))
    late_start = max(1, int(math.floor(0.75 * n)))
    late = residuals[late_start:]
    late_median = _median([x["residual_energy_fraction"] for x in late])
    spec = _spectrum(task_gram)
    vals, vecs = np.linalg.eigh(task_gram)
    order = np.argsort(vals)[::-1]
    vecs = vecs[:, order]
    state_norm2 = max(1e-30, qnorm2(final_coeff))
    candidates: dict[str, Any] = {
        "CONTROL": {"K": n, "coefficients": [float(x) for x in final_coeff], "parameter_frobenius_relative_error_vs_control": 0.0}
    }
    for k in [int(prereg["geometry"]["bounded_candidates"]["primary"]), *[int(x) for x in prereg["geometry"]["bounded_candidates"]["secondary"]]]:
        label = f"K{k}"
        if k >= n:
            candidates[label] = {"K": k, "status": "NOT_COMPRESSIVE_DIAGNOSTIC_ONLY"}
            continue
        basis = deltas @ vecs[:, :k]
        core = basis.T @ gram @ basis
        rhs = basis.T @ gram @ final_coeff
        alpha = _pinv(core, 1e-10) @ rhs
        projected = basis @ alpha
        error = math.sqrt(max(0.0, qnorm2(final_coeff - projected)) / state_norm2)
        candidates[label] = {
            "K": k,
            "status": "COMPRESSIVE",
            "coefficients": [float(x) for x in projected],
            "parameter_frobenius_relative_error_vs_control": error,
        }
    mg = prereg["mechanism_gate"]
    mechanism_checks = {
        "telescoping_relative_error": tel_rel <= float(prereg["geometry"]["telescoping_relative_error_max"]),
        "late_window_median_residual_energy": late_median <= float(mg["late_window_median_task_vector_residual_energy_lte"]),
        "task_vector_rank95_fraction": float(spec["rank95"]) / n <= float(mg["task_vector_rank95_fraction_of_update_count_lte"]),
        "primary_K64_parameter_relative_error": candidates["K64"].get("status") == "COMPRESSIVE" and float(candidates["K64"]["parameter_frobenius_relative_error_vs_control"]) <= float(mg["primary_K64_parameter_relative_error_lte"]),
    }
    out = {
        "schema": "openevo.ceiling1-gdn-d1-geometry.v1",  # legacy-revision-compat: historical serialized schema identity
        "status": "PASS" if all(mechanism_checks.values()) else "FAIL_MECHANISM_GATE",
        "prereg_sha256": sha256(args.prereg),
        "accepted_geometry_manifest_sha256": sha256(args.accepted_geometry_manifest),
        "arm": provenance.get("arm"),
        "stage2_seal_sha256": provenance.get("stage2_seal_sha256"),
        "accepted_parametric_updates": n,
        "telescoping_relative_error": tel_rel,
        "late_window": {"start_update_index_zero_based": late_start, "count": len(late), "median_residual_energy": late_median},
        "task_vector_spectrum": spec,
        "mechanism_checks": mechanism_checks,
        "final_panel_access": 0,
        "external_teacher_calls_delta": 0,
        "new_environment_calls_delta": 0,
    }
    atomic_json(args.output, out)
    atomic_json(args.candidates_output, {
        "schema": "openevo.ceiling1-gdn-d1-candidate-coefficients.v1",  # legacy-revision-compat: historical serialized schema identity
        "status": "PASS",
        "prereg_sha256": sha256(args.prereg),
        "geometry_output_sha256": sha256(args.output),
        "component_count": n,
        "component_rank": int(final["component_rank"]),
        "candidates": candidates,
        "scientific_semantics": "K counts Task-Vector basis directions, not physical LoRA rank",
    })
    print(json.dumps({"status": out["status"], "updates": n, "late_median_residual_energy": late_median, "rank95": spec["rank95"], "K64_error": candidates["K64"].get("parameter_frobenius_relative_error_vs_control")}, sort_keys=True))
    return 0 if out["status"] == "PASS" else 2


def _action(text: str) -> str | None:
    matches = list(re.finditer(r"<action>\s*(.*?)\s*</action>", text, re.I | re.S))
    return matches[0].group(1).strip() if len(matches) == 1 else None


def _softmax_np(x):
    import numpy as np
    x = x.astype(np.float64)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _kl_js(a, b) -> tuple[float, float]:
    import numpy as np
    p, q = _softmax_np(a), _softmax_np(b)
    m = (p + q) / 2.0
    eps = 1e-300
    kl = float(np.sum(p * (np.log(p + eps) - np.log(q + eps))))
    js = float(0.5 * np.sum(p * (np.log(p + eps) - np.log(m + eps))) + 0.5 * np.sum(q * (np.log(q + eps) - np.log(m + eps))))
    return kl, js


def _template_ids(tok, messages, thinking: str):
    kwargs: dict[str, Any] = {"tokenize": True, "add_generation_prompt": True}
    if thinking != "auto":
        kwargs["enable_thinking"] = thinking == "true"
    return list(tok.apply_chat_template(messages, **kwargs))


def _make_ids(tok, row: dict[str, Any], *, max_len: int, target_mode: bool, thinking: str):
    import torch
    prompt = _template_ids(tok, row["prompt_messages"], thinking)
    target = list(tok(row["response_messages"][0]["content"], add_special_tokens=False).input_ids)[:64]
    if not target_mode:
        return torch.tensor(prompt[-max_len:], dtype=torch.long), None
    ids = prompt + target
    if len(ids) > max_len:
        ids = ids[-max_len:]
    plen = len(ids) - len(target)
    if plen <= 0:
        raise ValueError("D1 function probe target leaves no prompt tokens")
    return torch.tensor(ids, dtype=torch.long), (plen, target)


def cmd_function_probe(args: argparse.Namespace) -> int:
    import gc
    import os
    import numpy as np
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prereg = load_json(args.prereg)
    if args.arm == "qwen3-1p7b" and args.chat_template_enable_thinking != "false":
        raise ValueError("Qwen3-1.7B D1 must explicitly use enable_thinking=false")
    panel = load_json(args.panel)
    panel_manifest = load_json(args.panel_manifest)
    candidates_doc = load_json(args.candidates)
    if panel_manifest.get("status") != "PASS" or sha256(args.panel) != panel_manifest.get("panel_sha256"):
        raise ValueError("D1 panel identity mismatch")
    if len(panel) != int(prereg["probe"]["rows"]):
        raise ValueError("D1 panel must have exactly 32 rows")
    if int(panel_manifest.get("final_panel_access", -1)) != 0:
        raise ValueError("final-panel access is forbidden")
    cand = candidates_doc["candidates"]
    control = np.asarray(cand["CONTROL"]["coefficients"], dtype=np.float64)
    component_rank = int(candidates_doc["component_rank"])
    if np.any(np.abs(control) <= 1e-12):
        raise ValueError("control coefficient too close to zero for safe column-ratio probe; materialize explicit candidate adapters instead")
    labels = ["CONTROL"] + [f"K{k}" for k in [int(prereg["geometry"]["bounded_candidates"]["primary"]), *[int(x) for x in prereg["geometry"]["bounded_candidates"]["secondary"]]] if cand.get(f"K{k}", {}).get("status") == "COMPRESSIVE"]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible and args.gpu_uuid not in visible:
        raise ValueError("declared GPU UUID does not match CUDA_VISIBLE_DEVICES")
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, local_files_only=True).to(device).eval()
    model = PeftModel.from_pretrained(base, args.control_adapter, is_trainable=False, local_files_only=True, autocast_adapter_dtype=False).eval()
    peft_base = model.get_base_model()
    transformer, lm_head = peft_base.model, peft_base.lm_head
    pairs = []
    for name, module in model.named_modules():
        if hasattr(module, "lora_B") and "default" in getattr(module, "lora_B", {}):
            weight = module.lora_B["default"].weight
            pairs.append((name, module, weight.detach().clone()))
    expected_rank = len(control) * component_rank
    if not pairs or any(pair[2].shape[1] != expected_rank for pair in pairs):
        raise ValueError("control adapter rank/component accounting mismatch")
    logits: dict[str, Any] = {}
    nlls: dict[str, list[float]] = {}
    generations: dict[str, list[str]] = {}
    top1: dict[str, list[int]] = {}
    top5: dict[str, list[list[int]]] = {}
    started = time.time()
    for label in labels:
        coeff = np.asarray(cand[label]["coefficients"], dtype=np.float64)
        ratio = torch.tensor(coeff / control, dtype=torch.float32, device=device).repeat_interleave(component_rank)
        with torch.no_grad():
            for _, module, master in pairs:
                module.lora_B["default"].weight.copy_((master.float() * ratio.unsqueeze(0)).to(master.dtype))
        label_logits, label_nll, label_gen, label_top1, label_top5 = [], [], [], [], []
        for row in panel:
            ids, meta = _make_ids(tok, row, max_len=args.max_len, target_mode=True, thinking=args.chat_template_enable_thinking)
            plen, target = meta
            ids = ids.unsqueeze(0).to(device)
            with torch.inference_mode():
                attention = torch.ones_like(ids)
                hidden = transformer(input_ids=ids, attention_mask=attention, use_cache=False, return_dict=True).last_hidden_state
                next_logit = lm_head(hidden[:, plen - 1, :]).float()[0]
                _, indices = torch.topk(next_logit, 5)
                label_top1.append(int(indices[0]))
                label_top5.append([int(x) for x in indices])
                label_logits.append(next_logit.cpu().numpy().astype(np.float32))
                positions = torch.arange(plen - 1, plen - 1 + len(target), device=device)
                shifted = lm_head(hidden[0, positions, :]).float()
                target_tensor = torch.tensor(target, device=device)
                label_nll.append(float(torch.nn.functional.cross_entropy(shifted, target_tensor, reduction="mean").item()))
            prompt_ids, _ = _make_ids(tok, row, max_len=args.max_len, target_mode=False, thinking=args.chat_template_enable_thinking)
            prompt_ids = prompt_ids.unsqueeze(0).to(device)
            with torch.inference_mode():
                prompt_attention = torch.ones_like(prompt_ids)
                generated = model.generate(prompt_ids, attention_mask=prompt_attention, do_sample=False, max_new_tokens=args.max_new_tokens, use_cache=True, pad_token_id=tok.eos_token_id)
            label_gen.append(tok.decode(generated[0, prompt_ids.shape[1] :], skip_special_tokens=True))
        logits[label] = np.stack(label_logits)
        nlls[label] = label_nll
        generations[label] = label_gen
        top1[label] = label_top1
        top5[label] = label_top5
        torch.cuda.empty_cache()
        gc.collect()

    control_logits = logits["CONTROL"]
    control_generation = generations["CONTROL"]
    control_actions = [_action(x) for x in control_generation]
    results: dict[str, Any] = {}
    for label in labels:
        kls, jss, rel, overlaps = [], [], [], []
        for i in range(len(panel)):
            kl, js = _kl_js(control_logits[i], logits[label][i])
            kls.append(kl); jss.append(js)
            c = control_logits[i] - control_logits[i].mean()
            z = logits[label][i] - logits[label][i].mean()
            rel.append(float(np.linalg.norm(z - c) / (np.linalg.norm(c) + 1e-30)))
            overlaps.append(len(set(top5["CONTROL"][i]) & set(top5[label][i])) / 5.0)
        actions = [_action(x) for x in generations[label]]
        new_invalid = [i for i, (c, a) in enumerate(zip(control_actions, actions)) if c is not None and a is None]
        results[label] = {
            "task_vector_basis_capacity": int(cand[label]["K"]),
            "parameter_frobenius_relative_error_vs_control": float(cand[label]["parameter_frobenius_relative_error_vs_control"]),
            "next_token_kl_control_to_candidate_mean": float(np.mean(kls)),
            "next_token_js_mean": float(np.mean(jss)),
            "centered_next_logit_relative_l2_mean": float(np.mean(rel)),
            "top1_token_agreement_vs_control": float(np.mean(np.asarray(top1[label]) == np.asarray(top1["CONTROL"]))),
            "top5_token_set_overlap_mean": float(np.mean(overlaps)),
            "teacher_forced_target_nll_mean": float(np.mean(nlls[label])),
            "teacher_forced_target_nll_delta_vs_control": float(np.mean(nlls[label]) - np.mean(nlls["CONTROL"])),
            "greedy_response_exact_agreement_vs_control": float(np.mean([a == b for a, b in zip(control_generation, generations[label])])),
            "greedy_action_parse_rate": float(np.mean([a is not None for a in actions])),
            "greedy_action_exact_agreement_vs_control": float(np.mean([a == b for a, b in zip(control_actions, actions)])),
            "greedy_action_changed_rows": [i for i, (a, b) in enumerate(zip(control_actions, actions)) if a != b],
            "new_parse_invalid_rows": new_invalid,
        }
    out = {
        "schema": "openevo.ceiling1-gdn-d1-function-preservation.v1",  # legacy-revision-compat: historical serialized schema identity
        "status": "PASS_MEASUREMENT_COMPLETE",
        "prereg_sha256": sha256(args.prereg),
        "panel_sha256": sha256(args.panel),
        "panel_manifest_sha256": sha256(args.panel_manifest),
        "candidate_coefficients_sha256": sha256(args.candidates),
        "control_adapter_sha256": sha256(args.control_adapter / "adapter_model.safetensors"),
        "results": results,
        "control_action_parse_rate": float(np.mean([a is not None for a in control_actions])),
        "execution": {
            "training": False,
            "environment_calls": 0,
            "external_teacher_calls": 0,
            "final_panel_access": 0,
            "chat_template_enable_thinking": args.chat_template_enable_thinking,
            "max_len": args.max_len,
            "greedy_max_new_tokens": args.max_new_tokens,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "declared_gpu_uuid": args.gpu_uuid,
            "wall_time_seconds": time.time() - started,
        },
    }
    atomic_json(args.output, out)
    print(json.dumps({"status": out["status"], "labels": labels, "panel_sha256": out["panel_sha256"]}, sort_keys=True))
    return 0


def _function_candidate_pass(prereg: dict[str, Any], function: dict[str, Any], label: str) -> tuple[bool, dict[str, bool]]:
    gate = prereg["function_preservation_gate"]
    result = function["results"].get(label)
    if result is None:
        return False, {"candidate_present": False}
    control_parse = float(function["control_action_parse_rate"])
    checks = {
        "candidate_present": True,
        "greedy_action_exact_agreement": float(result["greedy_action_exact_agreement_vs_control"]) == float(gate["greedy_action_exact_agreement_vs_control"]),
        "parse_rate_nonregression": float(result["greedy_action_parse_rate"]) >= control_parse,
        "new_parse_invalid_rows": len(result.get("new_parse_invalid_rows", [])) <= int(gate["new_parse_invalid_rows_max"]),
        "top1_token_agreement": float(result["top1_token_agreement_vs_control"]) == float(gate["top1_token_agreement_vs_control"]),
        "next_token_js": float(result["next_token_js_mean"]) <= float(gate["next_token_js_mean_lte"]),
        "centered_logit_relative_l2": float(result["centered_next_logit_relative_l2_mean"]) <= float(gate["centered_next_logit_relative_l2_mean_lte"]),
        "teacher_forced_nll_delta": float(result["teacher_forced_target_nll_delta_vs_control"]) <= float(gate["teacher_forced_target_nll_delta_vs_control_lte"]),
    }
    return all(checks.values()), checks


def classify_payload(prereg: dict[str, Any], geometry: dict[str, Any], function: dict[str, Any] | None) -> dict[str, Any]:
    if geometry.get("status") == "D1_INSUFFICIENT_PARAMETER_UPDATES_NOT_A_FAILURE":
        return {"classification": "D1_INSUFFICIENT_PARAMETER_UPDATES_NOT_A_FAILURE", "primary_pass": False}
    if geometry.get("final_panel_access") != 0 or geometry.get("external_teacher_calls_delta") != 0 or geometry.get("new_environment_calls_delta") != 0:
        return {"classification": "D1_INVALID_PROVENANCE", "primary_pass": False}
    mechanism_checks = geometry.get("mechanism_checks") or {}
    mechanism_pass = bool(mechanism_checks) and all(bool(x) for x in mechanism_checks.values())
    if not mechanism_pass:
        return {"classification": "D1_FAIL_MECHANISM_NOT_REPLICATED", "primary_pass": False, "mechanism_checks": mechanism_checks}
    if function is None:
        return {"classification": "D1_INVALID_PROVENANCE", "primary_pass": False, "reason": "function-preservation result missing"}
    execution = function.get("execution") or {}
    if function.get("status") != "PASS_MEASUREMENT_COMPLETE" or int(execution.get("environment_calls", -1)) != 0 or int(execution.get("external_teacher_calls", -1)) != 0 or int(execution.get("final_panel_access", -1)) != 0 or bool(execution.get("training")):
        return {"classification": "D1_INVALID_PROVENANCE", "primary_pass": False, "reason": "function-probe provenance gate failed"}
    primary_label = f"K{int(prereg['function_preservation_gate']['primary_candidate'])}"
    primary_pass, primary_checks = _function_candidate_pass(prereg, function, primary_label)
    secondary = []
    for k in prereg["geometry"]["bounded_candidates"]["secondary"]:
        label = f"K{int(k)}"
        passed, checks = _function_candidate_pass(prereg, function, label)
        if label in function.get("results", {}):
            secondary.append({"label": label, "pass": passed, "checks": checks})
    if primary_pass:
        classification = "D1_PRIMARY_PASS"
    elif any(x["pass"] for x in secondary):
        classification = "D1_PRIMARY_FAIL_SECONDARY_PRESERVATION_ONLY"
    else:
        classification = "D1_FAIL_FUNCTION_NOT_PRESERVED"
    return {
        "classification": classification,
        "primary_pass": primary_pass,
        "mechanism_checks": mechanism_checks,
        "primary_candidate": {"label": primary_label, "checks": primary_checks},
        "secondary_candidates": secondary,
    }


def cmd_classify(args: argparse.Namespace) -> int:
    prereg = load_json(args.prereg)
    geometry = load_json(args.geometry)
    function = load_json(args.function) if args.function and args.function.is_file() else None
    prereg_sha = sha256(args.prereg)
    if geometry.get("prereg_sha256") != prereg_sha:
        result = {"classification": "D1_INVALID_PROVENANCE", "primary_pass": False, "reason": "geometry/prereg SHA mismatch"}
    elif function is not None and function.get("prereg_sha256") != prereg_sha:
        result = {"classification": "D1_INVALID_PROVENANCE", "primary_pass": False, "reason": "function/prereg SHA mismatch"}
    else:
        result = classify_payload(prereg, geometry, function)
    closeout = {
        "schema": "openevo.ceiling1-gdn-d1-closeout.v1",  # legacy-revision-compat: historical serialized schema identity
        "status": "PASS_CLOSEOUT_WRITTEN",
        "prereg_sha256": prereg_sha,
        "geometry_sha256": sha256(args.geometry),
        "function_preservation_sha256": sha256(args.function) if args.function and args.function.is_file() else None,
        **result,
        "activation_authority": False,
        "current_lineage_mutation_authority": False,
        "final_panel_access_delta": 0,
    }
    atomic_json(args.output, closeout)
    print(json.dumps({"classification": closeout["classification"], "primary_pass": closeout["primary_pass"]}, sort_keys=True))
    return 0 if closeout["classification"] == "D1_PRIMARY_PASS" else 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build-probe")
    b.add_argument("--prereg", type=Path, required=True)
    b.add_argument("--records", type=Path, nargs="+", required=True)
    b.add_argument("--output", type=Path, required=True)
    b.add_argument("--manifest", type=Path, required=True)
    b.set_defaults(fn=cmd_build_probe)

    g = sub.add_parser("geometry")
    g.add_argument("--prereg", type=Path, required=True)
    g.add_argument("--accepted-geometry-manifest", type=Path, required=True)
    g.add_argument("--output", type=Path, required=True)
    g.add_argument("--candidates-output", type=Path, required=True)
    g.set_defaults(fn=cmd_geometry)

    f = sub.add_parser("function-probe")
    f.add_argument("--prereg", type=Path, required=True)
    f.add_argument("--arm", choices=("qwen25-3b", "qwen3-1p7b"), required=True)
    f.add_argument("--base", type=Path, required=True)
    f.add_argument("--control-adapter", type=Path, required=True)
    f.add_argument("--panel", type=Path, required=True)
    f.add_argument("--panel-manifest", type=Path, required=True)
    f.add_argument("--candidates", type=Path, required=True)
    f.add_argument("--output", type=Path, required=True)
    f.add_argument("--gpu-uuid", required=True)
    f.add_argument("--chat-template-enable-thinking", choices=("auto", "true", "false"), default="auto")
    f.add_argument("--max-len", type=int, default=1536)
    f.add_argument("--max-new-tokens", type=int, default=64)
    f.set_defaults(fn=cmd_function_probe)

    c = sub.add_parser("classify")
    c.add_argument("--prereg", type=Path, required=True)
    c.add_argument("--geometry", type=Path, required=True)
    c.add_argument("--function", type=Path)
    c.add_argument("--output", type=Path, required=True)
    c.set_defaults(fn=cmd_classify)
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
