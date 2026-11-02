#!/usr/bin/env python3
"""Run the bounded, synthetic native SD-LoRA trainer qualification gate.

The gate is engineering-only.  It resolves a formal experiment onto the Golden
Runtime/model asset, builds two deterministic synthetic WebShop-shaped records,
runs exactly two native optimizer steps, reloads the resulting adapter twice,
and emits a fail-closed receipt.  It never consumes a formal schedule position
or reads the formal held-out panel.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.openevo_webshop import experiment_resolver, golden_runtime_registry, qualification_dataset
from scripts.openevo_webshop.gpu_resource_gate import idle, snapshot_gpu


SCHEMA = "openevo.native-trainer-qualification.v1"
POLICY_PATH = ROOT / "configs/execution/qualification-policy-v1.json"


class QualificationError(RuntimeError):
    pass


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualificationError(f"unreadable JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise QualificationError(f"JSON object required: {path}")
    return value


def _git_identity(root: Path) -> str:
    root = root.resolve()
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise QualificationError(f"Git checkout is dirty: {root}")
    return head


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)
    path.chmod(0o600)


def load_policy() -> dict[str, Any]:
    policy = _read(POLICY_PATH)
    if policy.get("schema") != "openevo.execution-qualification-policy.v1":
        raise QualificationError("unexpected execution qualification policy schema")
    trainer = policy.get("trainer_gate")
    if not isinstance(trainer, dict) or trainer.get("record_count") != 2 or trainer.get("optimizer_steps") != 2:
        raise QualificationError("trainer qualification must remain fixed at two records / two optimizer steps")
    if trainer.get("formal_schedule_positions_consumed") != 0:
        raise QualificationError("trainer qualification policy consumes formal schedule positions")
    return policy


def _file_uri_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        raise QualificationError(f"adapter URI is not a local file URI: {uri}")
    return Path(unquote(parsed.path)).resolve()


def _native_trainer_model_id(model: dict[str, Any]) -> str:
    value = model.get("hf_id")
    if (
        not isinstance(value, str)
        or value.count("/") != 1
        or value.startswith(("/", "."))
        or "://" in value
    ):
        raise QualificationError(f"native trainer requires a Hugging Face model ID, got: {value!r}")
    return value


def _prepare_local_hf_cache(
    root: Path,
    *,
    model_id: str,
    model_path: Path,
    effective_revision: str,
) -> Path:
    """Expose the verified canonical mirror through the HF-ID cache contract.

    Frozen OpenEvo requires ``base_model`` to remain an HF model ID. The
    qualification gate is offline, so it materializes only HF cache metadata
    and a snapshot symlink to the already-verified local asset; it never copies
    or downloads model weights.
    """
    owner, name = model_id.split("/", 1)
    repo_root = root / "hub" / f"models--{owner}--{name}"
    refs = repo_root / "refs"
    snapshots = repo_root / "snapshots"
    refs.mkdir(mode=0o700, parents=True, exist_ok=True)
    snapshots.mkdir(mode=0o700, parents=True, exist_ok=True)
    snapshot = snapshots / effective_revision
    if snapshot.is_symlink() or snapshot.exists():
        if snapshot.resolve() != model_path.resolve():
            raise QualificationError(f"qualification HF cache snapshot mismatch: {snapshot}")
    else:
        snapshot.symlink_to(model_path.resolve(), target_is_directory=True)
    ref = refs / effective_revision
    if ref.exists() and ref.read_text(encoding="utf-8") != effective_revision:
        raise QualificationError(f"qualification HF cache ref mismatch: {ref}")
    ref.write_text(effective_revision, encoding="utf-8")
    return root


def _verify_declared_file(source_root: Path, item: dict[str, Any], label: str) -> Path:
    relative = item.get("path")
    expected = item.get("sha256")
    if not isinstance(relative, str) or not relative or not isinstance(expected, str) or len(expected) != 64:
        raise QualificationError(f"{label} path/SHA-256 declaration is incomplete")
    path = (source_root / relative).resolve()
    source_root = source_root.resolve()
    if path != source_root and source_root not in path.parents:
        raise QualificationError(f"{label} path escapes source root")
    if not path.is_file():
        raise QualificationError(f"{label} does not exist: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise QualificationError(f"{label} SHA-256 mismatch: expected={expected} observed={actual}")
    return path


def _write_runtime_trainer_launcher(
    output: Path,
    *,
    trainer_wrapper: Path,
    runtime_python: Path,
    openevo_root: Path,
) -> Path:
    """Bridge the frozen trainer wrapper through Core's isolated env allowlist.

    The pinned Core service invokes its executable with ``-I`` and keeps only
    an explicit environment allowlist.  That drops the parent ``PYTHONPATH``
    and the wrapper's ``OPENEVO_TRAINER_CORE`` override.  This private launcher
    restores those two resolved paths before chaining to the declared wrapper;
    it does not install, copy, or modify the pinned Core checkout.
    """

    launcher = output / "runtime-trainer-launcher.sh"
    launcher.write_text(
        "#!/bin/sh\n"
        "set -eu\n"
        "if [ \"${1:-}\" = \"-I\" ]; then\n"
        "  shift\n"
        "fi\n"
        f"export OPENEVO_TRAINER_PYTHON={shlex.quote(str(runtime_python))}\n"
        f"export OPENEVO_TRAINER_CORE={shlex.quote(str(openevo_root))}\n"
        f"exec {shlex.quote(str(trainer_wrapper))} \"$@\"\n",
        encoding="utf-8",
    )
    launcher.chmod(0o700)
    return launcher


def validate_smoke_payload(
    payload: dict[str, Any],
    *,
    resolution: dict[str, Any],
    dataset: dict[str, Any],
    effective_revision: str,
    policy: dict[str, Any],
) -> dict[str, Any]:
    trainer = policy["trainer_gate"]
    errors: list[str] = []
    if payload.get("official_open_evo_stable") != resolution["source"]["openevo_revision"]:
        errors.append("trainer payload OpenEvo revision mismatch")
    if payload.get("dataset_records_sha256") != dataset["records_sha256"]:
        errors.append("trainer payload synthetic records SHA-256 mismatch")
    if payload.get("training_record_count") != trainer["record_count"]:
        errors.append("trainer payload training_record_count mismatch")
    if payload.get("optimizer_steps") != trainer["optimizer_steps"]:
        errors.append("trainer payload optimizer_steps mismatch")
    manifest = payload.get("artifact_manifest")
    if not isinstance(manifest, dict):
        errors.append("trainer payload has no artifact_manifest")
    else:
        if manifest.get("model_revision") != effective_revision:
            errors.append("trainer artifact model_revision mismatch")
        if manifest.get("steps_completed") != trainer["optimizer_steps"]:
            errors.append("trainer artifact steps_completed mismatch")
        if manifest.get("training_record_count") != trainer["record_count"]:
            errors.append("trainer artifact training_record_count mismatch")
        if manifest.get("method") != "parametric_memory_sd_lora":
            errors.append("trainer artifact is not native parametric_memory_sd_lora")
    if errors:
        raise QualificationError("; ".join(errors))
    return {
        "optimizer_steps": payload["optimizer_steps"],
        "training_record_count": payload["training_record_count"],
        "training_loss": payload.get("training_loss"),
        "artifact_id": manifest.get("adapter_id") if isinstance(manifest, dict) else None,
    }


def validate_reload_payload(payload: dict[str, Any], *, effective_revision: str) -> dict[str, Any]:
    errors: list[str] = []
    if payload.get("status") != "pass":
        errors.append("adapter reload status is not pass")
    if payload.get("model_revision") != effective_revision:
        errors.append("adapter reload model_revision mismatch")
    if payload.get("loaded_parameter_digest_match") is not True:
        errors.append("two adapter reloads do not have identical parameter digests")
    if not isinstance(payload.get("loaded_lora_tensor_count"), int) or payload["loaded_lora_tensor_count"] < 1:
        errors.append("adapter reload exposed no LoRA tensors")
    if payload.get("base_trainable_parameter_count") != 0:
        errors.append("adapter reload left base parameters trainable")
    if errors:
        raise QualificationError("; ".join(errors))
    return {
        "first_load_digest": payload["first_load_digest"],
        "second_load_digest": payload["second_load_digest"],
        "loaded_lora_tensor_count": payload["loaded_lora_tensor_count"],
        "loaded_lora_parameter_count": payload.get("loaded_lora_parameter_count"),
    }


def _sample_live_idle(uuid: str, policy: dict[str, Any]) -> dict[str, Any]:
    gpu_policy = policy["gpu_gate"]
    sample = snapshot_gpu(uuid)
    if not idle(
        sample,
        max_memory_mib=int(gpu_policy["max_idle_memory_mib"]),
        max_util_percent=int(gpu_policy["max_idle_util_percent"]),
    ):
        raise QualificationError(f"GPU is not live-idle: {uuid}: {sample}")
    return sample


def _runtime_context() -> tuple[dict[str, Any], dict[str, Any]]:
    runtime = golden_runtime_registry.load_runtime()
    evidence = golden_runtime_registry._read(ROOT / runtime["runtime_evidence"]["path"])
    return runtime, evidence


def run(args: argparse.Namespace) -> dict[str, Any]:
    policy = load_policy()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise QualificationError(f"qualification output must be empty: {output}")
    output.mkdir(mode=0o700, parents=True, exist_ok=True)

    experiment_path = args.experiment.resolve()
    experiment_source_root = args.experiment_source_root.resolve()
    if experiment_path != experiment_source_root and experiment_source_root not in experiment_path.parents:
        raise QualificationError("experiment config is outside --experiment-source-root")
    experiment_source_revision = _git_identity(experiment_source_root)
    platform_source_revision = _git_identity(args.source_root.resolve())
    resolution = experiment_resolver.resolve(experiment_path, verify_local_assets=True)
    model = golden_runtime_registry.load_model(resolution["model"]["asset_id"])
    model_path = Path(model["canonical_path"]).resolve()
    model_id = _native_trainer_model_id(model)
    effective_revision = str(model.get("revision") or model["content_pin"])
    source_root = args.source_root.resolve()
    experiment = _read(args.experiment.resolve())
    execution = experiment.get("execution_contract")
    if not isinstance(execution, dict):
        raise QualificationError("experiment has no execution_contract")
    trainer_script = _verify_declared_file(source_root, execution.get("native_trainer_entrypoint", {}), "native trainer entrypoint")
    trainer_wrapper = _verify_declared_file(source_root, execution.get("trainer_wrapper", {}), "trainer wrapper")
    reload_script = source_root / "scripts/openevo_webshop/verify_adapter_reload.py"
    if not reload_script.is_file():
        raise QualificationError(f"adapter reload verifier missing: {reload_script}")

    dataset = qualification_dataset.build(output / "synthetic-dataset")
    smoke_evidence = output / "trainer-smoke.json"
    reload_evidence = output / "adapter-reload.json"
    artifact_root = output / "artifacts"
    hf_home = _prepare_local_hf_cache(
        output / "runtime-hf-cache",
        model_id=model_id,
        model_path=model_path,
        effective_revision=effective_revision,
    )
    runtime, runtime_evidence = _runtime_context()
    runtime_python = Path(args.runtime_python or runtime_evidence["runtime"]["python_executable"]).resolve()
    openevo_root = Path(args.openevo_root or runtime_evidence["sources"]["openevo_path"]).resolve()
    if not runtime_python.is_file() or not openevo_root.is_dir():
        raise QualificationError("Golden Runtime python/OpenEvo source is unavailable")
    observed_openevo = _git_identity(openevo_root)
    if observed_openevo != resolution["source"]["openevo_revision"]:
        raise QualificationError(f"OpenEvo checkout mismatch: {observed_openevo}")
    runtime_trainer_launcher = _write_runtime_trainer_launcher(
        output,
        trainer_wrapper=trainer_wrapper,
        runtime_python=runtime_python,
        openevo_root=openevo_root,
    )

    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": args.gpu_uuid,
            "HF_HOME": str(hf_home),
            "HF_HUB_CACHE": str(hf_home / "hub"),
            "TRANSFORMERS_CACHE": str(hf_home / "hub"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "WANDB_MODE": "offline",
            "OPENEVO_TRAINER_PYTHON": str(runtime_python),
            "OPENEVO_TRAINER_CORE": str(openevo_root),
        }
    )
    pythonpath = [str(source_root), str(source_root / "vendor/openevo-webshop/src"), str(openevo_root / "src")]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)

    lock_root = Path(os.environ.get("OPENEVO_WEBSHOP_GPU_LOCK_ROOT", policy["gpu_gate"]["lock_root"]))
    lock_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = lock_root / f"{args.gpu_uuid}.lock"
    started = time.monotonic()
    prelaunch: list[dict[str, Any]] = []
    smoke_completed: subprocess.CompletedProcess[str] | None = None
    reload_completed: subprocess.CompletedProcess[str] | None = None
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise QualificationError(f"GPU lock busy: {lock_path}") from exc
        for index in range(int(policy["gpu_gate"]["prelaunch_samples"])):
            prelaunch.append(_sample_live_idle(args.gpu_uuid, policy))
            if index + 1 < int(policy["gpu_gate"]["prelaunch_samples"]):
                time.sleep(0.25)

        trainer_cmd = [
            str(runtime_python),
            str(trainer_script),
            "--dataset-manifest", dataset["manifest"],
            "--dataset-artifact-id", f"fast-start-{resolution['model']['asset_id']}-synthetic-v1",
            "--artifact-root", str(artifact_root),
            "--evidence", str(smoke_evidence),
            "--run-id", f"fast-start-{resolution['model']['asset_id']}-trainer-v1",
            "--benchmark", "webshop",
            "--python", str(runtime_trainer_launcher),
            "--base-model", model_id,
            "--model-revision", effective_revision,
            "--max-steps", str(policy["trainer_gate"]["optimizer_steps"]),
            "--max-records", str(policy["trainer_gate"]["max_records"]),
            "--minimum-reward", str(policy["trainer_gate"]["minimum_reward"]),
        ]
        smoke_completed = subprocess.run(trainer_cmd, env=env, text=True, capture_output=True, check=False)
        if smoke_completed.returncode != 0:
            raise QualificationError(
                f"native trainer qualification failed rc={smoke_completed.returncode}: "
                f"{smoke_completed.stderr[-4000:]}"
            )
        smoke = _read(smoke_evidence)
        smoke_summary = validate_smoke_payload(
            smoke,
            resolution=resolution,
            dataset=dataset,
            effective_revision=effective_revision,
            policy=policy,
        )
        adapter_uri = smoke.get("adapter_uri")
        if not isinstance(adapter_uri, str):
            raise QualificationError("native trainer emitted no adapter URI")
        adapter_path = _file_uri_path(adapter_uri)
        missing = [name for name in policy["trainer_gate"]["required_adapter_files"] if not (adapter_path / name).is_file()]
        if missing:
            raise QualificationError(f"trainer adapter is incomplete: {missing}")

        reload_cmd = [
            str(runtime_python),
            str(reload_script),
            "--model", str(model_path),
            "--adapter", str(adapter_path),
            "--model-revision", effective_revision,
            "--output", str(reload_evidence),
            "--require-cuda",
        ]
        reload_completed = subprocess.run(reload_cmd, env=env, text=True, capture_output=True, check=False)
        if reload_completed.returncode != 0:
            raise QualificationError(
                f"adapter reload qualification failed rc={reload_completed.returncode}: "
                f"{reload_completed.stderr[-4000:]}"
            )
        reload = _read(reload_evidence)
        reload_summary = validate_reload_payload(reload, effective_revision=effective_revision)

    after = _sample_live_idle(args.gpu_uuid, policy)
    wall = time.monotonic() - started
    receipt = {
        "schema": SCHEMA,
        "status": "PASS",
        "experiment": resolution["experiment"],
        "resolution_sha256": resolution["resolution_sha256"],
        "runtime_family": runtime["runtime_family"],
        "model": {
            "asset_id": model["asset_id"],
            "hf_id": model["hf_id"],
            "canonical_path": str(model_path),
            "revision": model.get("revision"),
            "content_pin": model["content_pin"],
            "trainer_base_model": model_id,
            "trainer_effective_revision": effective_revision,
            "qualification_hf_home": str(hf_home),
        },
        "source": {
            **resolution["source"],
            "experiment_source_revision": experiment_source_revision,
            "platform_source_revision": platform_source_revision,
            "openevo_checkout_revision": observed_openevo,
        },
        "synthetic_dataset": dataset,
        "trainer": smoke_summary,
        "reload": reload_summary,
        "gpu": {
            "uuid": args.gpu_uuid,
            "prelaunch_samples": prelaunch,
            "postrelease_sample": after,
            "wall_seconds": wall,
            "gpu_hours": wall / 3600.0,
        },
        "subprocess": {
            "trainer_return_code": smoke_completed.returncode if smoke_completed else None,
            "reload_return_code": reload_completed.returncode if reload_completed else None,
        },
        "authority_boundary": {
            "qualification_only": True,
            "execution_authorized": False,
            "formal_task_consumption_allowed": False,
            "formal_schedule_positions_consumed": 0,
            "formal_panel_accessed": False,
            "qualification_adapter_may_enter_formal_state": False,
        },
    }
    _atomic_json(output / "qualification-receipt.json", receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--experiment-source-root", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-python", type=Path)
    parser.add_argument("--openevo-root", type=Path)
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except (QualificationError, experiment_resolver.ExperimentResolutionError, golden_runtime_registry.RegistryError, OSError, subprocess.SubprocessError) as exc:
        print(json.dumps({"schema": SCHEMA, "status": "FAIL", "errors": [str(exc)]}, indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
