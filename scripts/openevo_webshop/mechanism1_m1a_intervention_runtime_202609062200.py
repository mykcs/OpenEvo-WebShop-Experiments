#!/usr/bin/env python3
"""M1-A adjacent-state intervention runtime and zero-WebShop qualification.

This module only constructs the preregistered parameter interventions.  It does
not import or instantiate WebShop.  Formal evaluation must be released by a
separate activation artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def qform(x: list[float], gram: list[list[float]]) -> float:
    return sum(x[i] * gram[i][j] * x[j] for i in range(len(x)) for j in range(len(x)))


def component_factor_digest(tensors: dict[str, Any], component_index: int) -> str:
    prefix = f"components.{component_index}."
    names = sorted({k[len(prefix) : -2] for k in tensors if k.startswith(prefix) and k.endswith(".A")})
    if not names:
        raise RuntimeError(f"component {component_index} has no modules")
    h = hashlib.sha256()
    for name in names:
        for suffix in ("A", "B"):
            key = f"{prefix}{name}.{suffix}"
            value = tensors[key].contiguous().cpu()
            h.update(key.encode())
            h.update(str(tuple(value.shape)).encode())
            h.update(value.numpy().tobytes())
    return h.hexdigest()


def verify_manifests(manifest: dict[str, Any], provenance: dict[str, Any]) -> None:
    if manifest.get("status") != "FROZEN_PREPARATION_EXECUTION_LOCKED":
        raise RuntimeError("M1-A probe manifest is not preparation-locked")
    if manifest.get("execution_authority") is not False or manifest.get("formal_task_consumption_allowed") is not False:
        raise RuntimeError("preparation manifest unexpectedly grants execution authority")
    if provenance.get("status") != "PASS" or provenance.get("execution_authority") is not False:
        raise RuntimeError("source provenance binding is not locked PASS evidence")
    if int(manifest.get("cell_count", -1)) != 9 or int(manifest.get("episodes_per_cell", -1)) != 64:
        raise RuntimeError("frozen M1-A matrix shape drift")
    if len(manifest["probe_panel"]["tasks"]) != 64 or len(manifest["cells"]) != 9:
        raise RuntimeError("frozen M1-A task/cell inventory drift")
    if manifest["probe_panel"].get("r27_to_r49_overlap_count") != 0:
        raise RuntimeError("probe panel overlaps the source-vector update tasks")
    if manifest["probe_panel"].get("ceiling_final_panel_goal_overlap_count") != 0 or manifest["probe_panel"].get("ceiling_final_panel_instruction_hash_overlap_count") != 0:
        raise RuntimeError("probe panel overlaps the Ceiling final panel")
    tau_norm = float(manifest["states"]["tau_gram_norm"])
    gram = manifest["parameter_basis"]["gram_matrix"]
    tau = [float(x) for x in manifest["states"]["tau_adjacent_late_coefficients"]]
    if not math.isclose(math.sqrt(max(0.0, qform(tau, gram))), tau_norm, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("tau Gram norm drift")
    for row in manifest["random_directions"]:
        direction = [float(x) for x in row["direction_coefficients"]]
        if not math.isclose(math.sqrt(max(0.0, qform(direction, gram))), tau_norm, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"random direction {row['seed']} is not Gram-norm matched")


def verify_canonical_provenance(repo_root: Path, provenance: dict[str, Any]):
    binding = provenance["canonical_remote_provenance_seal"]
    seal_path = repo_root / binding["path"]
    if sha256(seal_path) != binding["sha256"]:
        raise RuntimeError("canonical remote provenance seal hash drift")
    seal = load_json(seal_path)
    if seal.get("status") != "SOURCE_PROVENANCE_SEALED_EXECUTION_LOCKED":
        raise RuntimeError("canonical remote provenance seal is not locked")
    if seal.get("execution_authority") is not False or seal.get("formal_task_consumption_allowed") is not False:
        raise RuntimeError("canonical remote provenance seal unexpectedly grants execution authority")
    roles = provenance["guard_identity_roles"]
    shared_head = roles["shared_component_store_head_sha256"]
    if seal["component_store"]["head"]["sha256"] != shared_head:
        raise RuntimeError("canonical remote component-store HEAD drift")
    if seal["component_store"]["guard"]["sha256"] != roles["remote_archive_guard"]["sha256"]:
        raise RuntimeError("canonical remote archive guard drift")
    if provenance["component_store_immutable_snapshot"]["head"]["sha256"] != shared_head:
        raise RuntimeError("capsule component-store HEAD no longer matches canonical remote HEAD")
    if provenance["component_store_immutable_snapshot"]["guard"]["sha256"] != roles["source_capsule_guard"]["sha256"]:
        raise RuntimeError("source-capsule guard role drift")
    if provenance["component_store_live_readback_202609062200"]["guard_sha256"] != roles["qualification_live_guard"]["sha256"]:
        raise RuntimeError("qualification live guard role drift")
    if seal["source_capsule"]["archive_sha256"] != provenance["source_capsule"]["archive_sha256"]:
        raise RuntimeError("canonical source-capsule archive identity drift")
    if seal["source_capsule"]["manifest_sha256"] != provenance["source_capsule"]["manifest_sha256"]:
        raise RuntimeError("canonical source-capsule manifest identity drift")
    if seal["terminal_seal"]["sha256"] != provenance["terminal_seal"]["sha256"]:
        raise RuntimeError("canonical terminal-seal identity drift")
    return seal_path, seal


def load_component_store(component_store: Path, manifest: dict[str, Any], provenance: dict[str, Any]):
    from safetensors.torch import load_file

    head_path = component_store / "HEAD.json"
    guard_path = component_store / "GUARD_STATUS.json"
    immutable = provenance["component_store_immutable_snapshot"]
    if sha256(head_path) != immutable["head"]["sha256"]:
        raise RuntimeError("component-store HEAD no longer matches immutable source identity")
    head = load_json(head_path)
    guard = load_json(guard_path)
    if head.get("status") != "PASS" or guard.get("status") != "PASS":
        raise RuntimeError("component-store HEAD/GUARD is not PASS")
    if guard.get("component_store_head_sha256") != sha256(head_path):
        raise RuntimeError("component-store GUARD does not bind current HEAD")
    index_path = component_store / head["index"]
    if sha256(index_path) != head["index_sha256"]:
        raise RuntimeError("component-store index hash drift")
    snapshot_path = component_store / head["latest_snapshot"]
    if sha256(snapshot_path) != head["latest_snapshot_sha256"]:
        raise RuntimeError("component-store snapshot hash drift")
    index = load_json(index_path)
    expected_factors = list(manifest["parameter_basis"]["component_factor_sha256"])
    if int(index.get("component_count", -1)) != 7 or [x["factor_sha256"] for x in index["components"]] != expected_factors:
        raise RuntimeError("component-store factor lineage drift")
    shards = []
    for item in index["components"]:
        path = component_store / item["file"]
        if sha256(path) != item["file_sha256"]:
            raise RuntimeError(f"component shard hash drift: {path.name}")
        tensors = load_file(str(path), device="cpu")
        idx = int(item["component_index"])
        if component_factor_digest(tensors, idx) != item["factor_sha256"]:
            raise RuntimeError(f"component factor digest drift: {idx}")
        shards.append((idx, tensors))
    return head, guard, index, snapshot_path, shards


def module_factor(shard: tuple[int, dict[str, Any]], module_name: str):
    idx, tensors = shard
    return tensors[f"components.{idx}.{module_name}.A"], tensors[f"components.{idx}.{module_name}.B"]


def install_component_basis(model: Any, module_names: list[str], shards: list[tuple[int, dict[str, Any]]]) -> None:
    import torch

    core = model.base_model.model
    with torch.no_grad():
        for name in module_names:
            module = core.get_submodule(name)
            if "default" not in module.lora_A or "default" not in module.lora_B:
                raise RuntimeError(f"PEFT LoRA module missing at {name}")
            a_cat = torch.cat([module_factor(shard, name)[0] for shard in shards], dim=0)
            target_a = module.lora_A["default"].weight
            if tuple(a_cat.shape) != tuple(target_a.shape):
                raise RuntimeError(f"LoRA A shape drift at {name}: {tuple(a_cat.shape)} != {tuple(target_a.shape)}")
            target_a.copy_(a_cat.to(device=target_a.device, dtype=target_a.dtype))


def set_coefficients(model: Any, module_names: list[str], shards: list[tuple[int, dict[str, Any]]], coefficients: list[float]) -> None:
    import torch

    if len(coefficients) != len(shards):
        raise RuntimeError("coefficient/component count mismatch")
    core = model.base_model.model
    with torch.no_grad():
        for name in module_names:
            blocks = []
            for coefficient, shard in zip(coefficients, shards):
                b = module_factor(shard, name)[1]
                blocks.append(float(coefficient) * b)
            b_cat = torch.cat(blocks, dim=1)
            target_b = core.get_submodule(name).lora_B["default"].weight
            if tuple(b_cat.shape) != tuple(target_b.shape):
                raise RuntimeError(f"LoRA B shape drift at {name}: {tuple(b_cat.shape)} != {tuple(target_b.shape)}")
            target_b.copy_(b_cat.to(device=target_b.device, dtype=target_b.dtype))


def qualify(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    manifest = load_json(args.manifest)
    provenance = load_json(args.provenance)
    verify_manifests(manifest, provenance)
    repo_root = args.manifest.resolve().parents[3]
    canonical_seal_path, canonical_seal = verify_canonical_provenance(repo_root, provenance)
    head, guard, index, snapshot_path, shards = load_component_store(args.component_store, manifest, provenance)
    if index.get("model_revision") != "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e":
        raise RuntimeError("Qwen3-1.7B revision drift")
    model_path = args.model.resolve()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True, torch_dtype=torch.bfloat16)
    peft_cfg = LoraConfig(
        r=56,
        lora_alpha=56,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
        inference_mode=True,
    )
    model = get_peft_model(base, peft_cfg).to(args.device).eval()
    module_names = [row["name"] for row in index["modules"]]
    installed = sorted(name for name, module in model.base_model.model.named_modules() if hasattr(module, "lora_A") and "default" in module.lora_A)
    if sorted(module_names) != installed:
        raise RuntimeError("PEFT target-module inventory does not match component-store index")
    install_component_basis(model, module_names, shards)
    encoded = tokenizer("M1-A zero-task runtime qualification.", return_tensors="pt")
    encoded = {k: v.to(args.device) for k, v in encoded.items()}
    cell_receipts = []
    for cell in manifest["cells"]:
        coeff = [float(x) for x in cell["coefficients"]]
        set_coefficients(model, module_names, shards, coeff)
        with torch.no_grad():
            logits = model(**encoded).logits[:, -1, :32].float().cpu()
        if not torch.isfinite(logits).all():
            raise RuntimeError(f"non-finite synthetic forward for {cell['cell_id']}")
        cell_receipts.append({
            "cell_id": cell["cell_id"],
            "coefficient_sha256": hashlib.sha256(json.dumps(coeff, separators=(",", ":")).encode()).hexdigest(),
            "synthetic_logit_sha256": hashlib.sha256(logits.numpy().tobytes()).hexdigest(),
            "synthetic_forward_finite": True,
        })
    return {
        "schema": "openevo.mechanism1-m1a-zero-webshop-runtime-qualification.202609062200",
        "status": "PASS",
        "qualification_only": True,
        "formal_webshop_tasks_consumed": 0,
        "formal_webshop_episodes_consumed": 0,
        "final_panel_access": 0,
        "external_teacher_calls": 0,
        "experiment_git_sha": args.experiment_git_sha,
        "manifest_path": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "provenance_path": str(args.provenance),
        "provenance_sha256": sha256(args.provenance),
        "canonical_remote_provenance_seal_path": str(canonical_seal_path),
        "canonical_remote_provenance_seal_sha256": sha256(canonical_seal_path),
        "canonical_remote_component_store_guard_sha256": canonical_seal["component_store"]["guard"]["sha256"],
        "source_capsule_guard_sha256": provenance["guard_identity_roles"]["source_capsule_guard"]["sha256"],
        "qualification_live_guard_sha256": provenance["guard_identity_roles"]["qualification_live_guard"]["sha256"],
        "component_store_head_sha256": sha256(args.component_store / "HEAD.json"),
        "component_store_guard_sha256_at_qualification": sha256(args.component_store / "GUARD_STATUS.json"),
        "component_store_guard_status": guard["status"],
        "component_store_snapshot_path": str(snapshot_path),
        "component_store_snapshot_sha256": sha256(snapshot_path),
        "component_count": int(index["component_count"]),
        "component_module_count": len(module_names),
        "model_path": str(model_path),
        "model_config_sha256": sha256(model_path / "config.json"),
        "device": args.device,
        "cuda_device_name": torch.cuda.get_device_name(torch.device(args.device)) if args.device.startswith("cuda") else None,
        "cell_count": len(cell_receipts),
        "cells": cell_receipts,
        "webshop_runtime_imported": False,
        "execution_authority": False,
        "formal_task_consumption_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--component-store", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--experiment-git-sha", required=True)
    args = parser.parse_args()
    receipt = qualify(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "cells": receipt["cell_count"], "formal_webshop_tasks_consumed": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
