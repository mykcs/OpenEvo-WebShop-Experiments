#!/usr/bin/env python3
"""Recover a packed PEFT Q17 SD-LoRA adapter from sealed native state, CPU only.

Scientific identity: exact original SD-LoRA state manifest + the original
coefficients and rank-8 factors. Reconstruct module LoRA_A by concatenating
the components vertically, and LoRA_B by concatenating component B scaled by
the component coefficient horizontally. Reference R127 proves byte-wise tensor
parity against a retained original exported adapter before any R126/R150
reconstruction is eligible for GPU qualification.

Never modify, delete, chmod or replace the original checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

Q17 = Path("/workspace/runs/"
    "q17-direct-apply-only-w2-202609082022/"
    "q17-direct-apply-only-w2-202609082022-direct")
ALLOWED_ROOT = Path("/workspace/maintenance/"
                    "Q17-SD-LORA-RECONSTRUCT-20261013")
EXPECTED_ADAPTER = {
  126:"74cdc9d37d4adc9bd01db22cdafa3a7b5f35d3cde5c2247d8a2467ed529ae973",
  127:"2c52bcc73586bd130e1b965149eb24033e8124346b7d1df990923231844fc06a",
  150:"decf98f62076f66aff1a54c71a91e7df6bbfcdf20700271cfa76808d2e405a54",
}
EXPECTED_COMPONENTS = {126:126,127:127,150:150}
EXPECTED_BASE = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
FILEMODEL = "adapter_model.safetensors"
FILESOURCE = "openevo_sd_lora_state.safetensors"


class IntegrityError(RuntimeError):
    pass


def digest(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<22),b""):
            h.update(b)
    return h.hexdigest()


def original(round_index:int)->tuple[Path,dict,dict,dict]:
    if round_index not in EXPECTED_COMPONENTS:
        raise IntegrityError("round not in frozen reference/target allowlist")
    update=json.loads((Q17/f"round-{round_index:03d}"/"sd-lora"/"update.json").read_bytes())
    root=Path(update["adapter_path"])
    if root.is_symlink() or not root.resolve().is_relative_to((Q17/f"round-{round_index:03d}").resolve()):
        raise IntegrityError("original adapter path outside exact round or symlink")
    if update.get("adapter_digest_after")!=EXPECTED_ADAPTER[round_index]:
        raise IntegrityError("frozen producer final adapter hash changed")
    cfg=json.loads((root/"adapter_config.json").read_bytes())
    manifest=json.loads((root/"openevo_sd_lora_state.json").read_bytes())
    count=EXPECTED_COMPONENTS[round_index]
    if (int(manifest.get("component_count",-1))!=count
            or int(manifest.get("effective_rank",-1))!=8*count
            or len(manifest.get("components",[]))!=count
            or len(manifest.get("modules",[]))!=112
            or manifest.get("model_revision")!=EXPECTED_BASE
            or manifest.get("schema_version")!="openevo.sd_lora_state.v4"
            or manifest.get("direction_parameterization")!="frozen_global_unit_frobenius_direction"):
        raise IntegrityError("native state scientific manifest mismatch")
    if (cfg.get("r")!=8*count or cfg.get("lora_alpha")!=8*count
            or cfg.get("lora_dropout")!=0
            or set(cfg.get("target_modules",[]))!={"q_proj","k_proj","v_proj","o_proj"}
            or cfg.get("peft_type")!="LORA"):
        raise IntegrityError("native PEFT config mismatch")
    for i,c in enumerate(manifest["components"]):
        if (int(c["task_index"])!=i or int(c["rank"])!=8
                or not math.isfinite(float(c["coefficient"]))):
            raise IntegrityError(f"native component index/rank/coefficient mismatch: {i}")
    p=root/FILESOURCE
    if not p.is_file() or p.is_symlink() or p.stat().st_size!=int(manifest["state_weights_size_bytes"]):
        raise IntegrityError("native state file inaccessible/size mismatch")
    if digest(p)!=manifest["state_weights_sha256"]:
        raise IntegrityError("native weights do not match original full SHA256")
    return root,update,manifest,cfg


def factor_pair(native:Any,name:str,coeff:list[float],count:int):
    import torch
    a=torch.cat([native.get_tensor(f"components.{i}.{name}.A")
                 for i in range(count)],dim=0).contiguous()
    b=torch.cat([native.get_tensor(f"components.{i}.{name}.B")*coeff[i]
                 for i in range(count)],dim=1).contiguous()
    return a,b


def rebuild(round_index:int, *, output_root:Path|None, verify_only:bool)->dict:
    from safetensors import safe_open
    from safetensors.torch import save_file
    started=time.monotonic()
    root,update,manifest,cfg=original(round_index)
    count=EXPECTED_COMPONENTS[round_index]
    components=manifest["components"]
    coeff=[float(c["coefficient"]) for c in components]
    source=root/FILESOURCE
    input_keys_expected={
        f"components.{i}.{name}.{factor}"
        for i in range(count) for name in [m["name"] for m in manifest["modules"]]
        for factor in ("A","B")
    }
    reference = root/FILEMODEL
    if verify_only:
        if round_index!=127 or not reference.is_file() or digest(reference)!=EXPECTED_ADAPTER[127]:
            raise IntegrityError("full original R127 reference adapter required")
        output=None
    else:
        if round_index not in (126,150):
            raise IntegrityError("only R126 and R150 may be materialized in separate storage")
        if output_root is None or output_root.resolve()!=ALLOWED_ROOT:
            raise IntegrityError("destination must be the exact approved independent new staging root")
        if output_root.is_symlink() or not output_root.is_dir():
            raise IntegrityError("staging root must already exist and not be a symlink")
        output=output_root/f"round-{round_index:03d}"
        # No reset/rerun: cannot overwrite a previous attempt or foreign file.
        output.mkdir(mode=0o700,parents=False,exist_ok=False)

    packed={}
    checked=0
    with safe_open(str(source),framework="pt",device="cpu") as native:
        keys=set(native.keys())
        if keys!=input_keys_expected:
            raise IntegrityError("tensor keys do not match exact N components x 112 modules x A/B")
        if verify_only:
            orig_ctx=safe_open(str(reference),framework="pt",device="cpu")
        else:
            orig_ctx=None
        try:
            if orig_ctx is not None:
                orig_ctx.__enter__()
            for module in manifest["modules"]:
                name=module["name"]
                A,B=factor_pair(native,name,coeff,count)
                inp=int(module["in_features"]);out=int(module["out_features"])
                if tuple(A.shape)!=(8*count,inp) or tuple(B.shape)!=(out,8*count):
                    raise IntegrityError("module packed factor shape mismatch")
                prefix="base_model.model."+name
                key_a=prefix+".lora_A.weight"
                key_b=prefix+".lora_B.weight"
                if orig_ctx is not None:
                    import torch
                    if not torch.equal(A,orig_ctx.get_tensor(key_a)):
                        raise IntegrityError("R127 reference A factor exact parity failed")
                    if not torch.equal(B,orig_ctx.get_tensor(key_b)):
                        raise IntegrityError("R127 reference B factor exact parity failed")
                else:
                    packed[key_a]=A
                    packed[key_b]=B
                checked+=1
        finally:
            if orig_ctx is not None:
                orig_ctx.__exit__(None,None,None)
    if checked!=112:
        raise IntegrityError("not exactly 112 QKV/O modules")
    original_cfg_sha=digest(root/"adapter_config.json")
    generated_hash=None
    if output is not None:
        produced=output/FILEMODEL
        if produced.exists():raise IntegrityError("refuse adapter output overwrite")
        save_file(packed,str(produced),metadata={"format":"pt"})
        shutil.copyfile(root/"adapter_config.json",output/"adapter_config.json")
        generated_hash=digest(produced)
        # The old adapter was removed but its producing round recorded its exact
        # expected SHA. This check is intentionally strict. If serialization
        # changes, fail closed; never relabel as original output.
        if generated_hash!=EXPECTED_ADAPTER[round_index]:
            raise IntegrityError("reconstructed adapter not byte-identical to frozen producer digest: observed "+generated_hash)
    return {
       "schema":"openevo.q17.sd_lora_native_to_peft_exact_cpu.v1",
       "status":"PASS_EXACT_REFERENCE_PARITY" if verify_only else "PASS_EXACT_ORIGINAL_ADAPTER_RECONSTRUCTION",
       "round":round_index,
       "components":count,"effective_nominal_rank":8*count,
       "modules_verified_or_packed":checked,
       "factor_pairs_verified_or_packed":checked*2,
       "reconstruction":"A=torch.cat(native component A vertically); B=torch.cat(native component B * coefficient horizontally)",
       "native_state_sha256":manifest["state_weights_sha256"],
       "original_producer_expected_adapter_sha256":EXPECTED_ADAPTER[round_index],
       "observed_reconstructed_adapter_sha256":generated_hash,
       "original_adapter_config_sha256":original_cfg_sha,
       "model_revision":manifest["model_revision"],
       "output_root":str(output) if output else None,
       "no_source_model_modification":True,"no_GPU_calls":True,"no_new_webshop":True,
       "wall_seconds":time.monotonic()-started
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--phase",choices=("verify-reference","reconstruct"),required=True)
    ap.add_argument("--round",type=int,required=True)
    ap.add_argument("--output-root",type=Path)
    arg=ap.parse_args()
    try:
        out=rebuild(arg.round,output_root=arg.output_root,verify_only=arg.phase=="verify-reference")
        print(json.dumps(out,ensure_ascii=False,sort_keys=True),flush=True)
        return 0
    except (IntegrityError,KeyError,TypeError,ValueError,FileNotFoundError) as e:
        print(json.dumps({"status":"BLOCKED_EXACT_ADAPTER_RECONSTRUCTION","round":arg.round,
            "reason":str(e),"GPU_calls":0,"original_files_mutated":0},sort_keys=True),file=sys.stderr)
        return 2

if __name__=="__main__":
    raise SystemExit(main())
