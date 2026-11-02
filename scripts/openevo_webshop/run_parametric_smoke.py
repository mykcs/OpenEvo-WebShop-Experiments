#!/usr/bin/env python3
"""Invoke the official OpenEvo SD-LoRA method for one WebShop P0 smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from openevo.evolution.framework.contracts import canonical_digest
from openevo.evolution.framework.execution import (
    MethodExecutionContext,
    MethodExecutionServices,
    ResolvedMethodInputBinding,
    build_execution_envelope,
    worker_input_artifact_digest,
)
from openevo.evolution.models import WorkerClaimInputArtifact, WorkerClaimedJob
from openevo.evolution.parametric.sd_lora import parametric_memory_sd_lora
from openevo.evolution.parametric import sd_lora as sd_lora_module
from openevo.evolution.parametric.trainer_service import SubprocessSdLoraTrainerService


class _NoInferenceHarness:
    def infer(self, request):  # pragma: no cover - the parametric method must not call it.
        raise AssertionError(f"unexpected inference request: {request!r}")


def _install_runtime_compatibility() -> None:
    """Bridge one Python build capability without changing official Core code.

    Some Linux CPython builds omit ``follow_symlinks`` from ``os.chmod`` even
    though the path is already private and owned. The official method uses the
    keyword while creating its worker directory. This narrow downstream shim
    preserves mode/ownership checks and is recorded as an upstream issue.
    """

    if os.chmod in os.supports_follow_symlinks:
        return

    def compatible_private_work_dir(artifact_root: Path, job_id: str, lease_id: str) -> Path:
        root = Path(os.path.abspath(artifact_root))
        if not root.is_dir():
            raise ValueError("SD-LoRA artifact root does not exist")
        workers = root / "workers"
        workers.mkdir(mode=0o700, exist_ok=True)
        os.chmod(workers, 0o700)
        identity = hashlib.sha256(f"{job_id}\0{lease_id}".encode("utf-8")).hexdigest()[:32]
        work_dir = workers / f"sd-lora-{identity}"
        work_dir.mkdir(mode=0o700)
        return work_dir

    sd_lora_module._private_work_dir = compatible_private_work_dir


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_PRIOR_ADAPTER_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "openevo_sd_lora_replay.jsonl",
    "openevo_sd_lora_state.json",
    "openevo_sd_lora_state.safetensors",
)


def _prior_artifact(args: argparse.Namespace) -> WorkerClaimInputArtifact | None:
    if (args.prior_adapter is None) != (args.prior_artifact_id is None):
        raise ValueError("--prior-adapter and --prior-artifact-id must be supplied together")
    if args.prior_adapter is None:
        return None
    prior_artifact_id = args.prior_artifact_id.strip()
    if not prior_artifact_id:
        raise ValueError("--prior-artifact-id must not be empty")
    prior_dir = args.prior_adapter.resolve()
    if not prior_dir.is_dir():
        raise ValueError(f"prior adapter directory not found: {prior_dir}")
    missing = [name for name in _PRIOR_ADAPTER_FILES if not (prior_dir / name).is_file()]
    if missing:
        raise ValueError("prior adapter is incomplete: " + ", ".join(missing))
    return WorkerClaimInputArtifact(
        artifact_id=prior_artifact_id,
        type="parametric_memory",
        uri=prior_dir.as_uri(),
        name=prior_dir.name,
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    _install_runtime_compatibility()
    manifest_path = args.dataset_manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records_path = manifest_path.with_name("records.jsonl")
    benchmark = args.benchmark
    dataset = WorkerClaimInputArtifact(
        artifact_id=args.dataset_artifact_id or f"{benchmark}-parametric-{args.run_id}",
        type="dataset",
        uri=manifest_path.as_uri(),
        name=str(manifest.get("name") or f"{benchmark}-parametric-smoke"),
        manifest_sha256=canonical_digest(manifest),
        records_byte_size=int(manifest["records_byte_size"]),
        records_sha256=str(manifest["records_sha256"]),
    )
    prior = _prior_artifact(args)
    prior_artifacts = (prior,) if prior is not None else ()
    bindings = (
        ResolvedMethodInputBinding(
            binding_id="current_dataset",
            artifact_ids=(dataset.artifact_id,),
            artifact_digests=(worker_input_artifact_digest(dataset),),
        ),
        ResolvedMethodInputBinding(
            binding_id="prior_target_artifacts",
            artifact_ids=tuple(artifact.artifact_id for artifact in prior_artifacts),
            artifact_digests=tuple(
                worker_input_artifact_digest(artifact) for artifact in prior_artifacts
            ),
        ),
    )
    user_config = {
        "base_model": args.base_model,
        "model_revision": args.model_revision,
        "rank": 4,
        "target_modules": ["q_proj", "v_proj"],
        "learning_rate": 2.0e-4,
        "coefficient_learning_rate": 1.0e-2,
        "epochs": 1,
        "max_steps": args.max_steps,
        "max_length": 512,
        "max_records": args.max_records,
        "replay_capacity": 64,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "max_grad_norm": 1.0,
        "dtype": "bfloat16",
        "load_in_4bit": False,
        "gradient_checkpointing": True,
        "coefficient_init": 0.8,
        "minimum_reward": args.minimum_reward,
        "seed": 1993,
        "timeout_seconds": 3600.0,
    }
    core_config = {
        "name": f"seed3090-{benchmark}-parametric-{args.run_id}",
        "experiment_id": f"seed3090-{benchmark}-parametric-p0",
        "lineage": {
            "seed3090_run_id": args.run_id,
            "dataset_manifest_sha256": canonical_digest(manifest),
            "dataset_records_sha256": _sha256(records_path),
            "official_open_evo_stable": "796e6248a61112a61c880885844d3b0420c0b6ae",
            "prior_parametric_memory_artifact_id": (
                prior.artifact_id if prior is not None else None
            ),
        },
        "tags": ["seed3090", benchmark, "parametric-p0", "sd-lora"],
        "scores": {},
        "compatibility": {"agent_harnesses": ["local_transformers"]},
        "promoted": False,
    }
    envelope = build_execution_envelope(
        plan_id=f"seed3090-parametric-plan-{args.run_id}",
        plan_digest="a" * 64,
        registry_snapshot_digest="b" * 64,
        target_id="parametric_memory",
        method_id="parametric_memory_sd_lora",
        method_identity_digest="c" * 64,
        user_config=user_config,
        core_config=core_config,
        input_bindings=bindings,
        output_artifact_types=("parametric_memory",),
    )
    job = WorkerClaimedJob(
        job_id=f"seed3090-parametric-job-{args.run_id}",
        lease_id=f"seed3090-parametric-lease-{args.run_id}",
        job_type="parametric_memory",
        method="parametric_memory_sd_lora",
        input_artifacts=[dataset, *prior_artifacts],
        config={**user_config, **core_config},
    )
    args.artifact_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    args.artifact_root.chmod(0o700)
    context = MethodExecutionContext(
        job=job,
        artifact_root=args.artifact_root,
        envelope=envelope,
        services=MethodExecutionServices(
            harness=_NoInferenceHarness(),
            parametric_trainer=None,
        ),
    )
    with SubprocessSdLoraTrainerService(args.artifact_root, python_executable=args.python) as trainer:
        context = MethodExecutionContext(
            job=context.job,
            artifact_root=context.artifact_root,
            envelope=context.envelope,
            services=MethodExecutionServices(
                harness=context.services.harness,
                parametric_trainer=trainer,
            ),
        )
        artifacts = parametric_memory_sd_lora(context)
    if len(artifacts) != 1:
        raise RuntimeError(f"expected one parametric artifact, got {len(artifacts)}")
    artifact = artifacts[0]
    payload = {
        "schema": "seed3090.parametric_smoke.v1",
        "benchmark": benchmark,
        "run_id": args.run_id,
        "official_open_evo_stable": "796e6248a61112a61c880885844d3b0420c0b6ae",
        "dataset_manifest_sha256": canonical_digest(manifest),
        "dataset_records_sha256": _sha256(records_path),
        "minimum_reward": args.minimum_reward,
        "prior_parametric_memory_artifact_id": (
            prior.artifact_id if prior is not None else None
        ),
        "prior_adapter_uri": prior.uri if prior is not None else None,
        "artifact": artifact.model_dump(mode="json"),
        "artifact_manifest": artifact.manifest,
        "training_record_count": artifact.manifest.get("training_record_count"),
        "optimizer_steps": artifact.manifest.get("steps_completed"),
        "training_loss": artifact.manifest.get("training_loss"),
        "adapter_uri": artifact.uri,
    }
    evidence_path = args.evidence.resolve()
    evidence_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    evidence_path.chmod(0o600)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--dataset-artifact-id", default=None)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--benchmark", choices=("webshop", "alfworld"), default="webshop")
    parser.add_argument("--python", default=None)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument(
        "--model-revision",
        default="aa8e72537993ba99e69dfaafa59ed015b17504d1",
    )
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--max-records", type=int, default=30)
    parser.add_argument("--minimum-reward", type=float, default=0.0)
    parser.add_argument(
        "--prior-adapter",
        type=Path,
        help="Directory of one official cumulative SD-LoRA artifact to continue.",
    )
    parser.add_argument(
        "--prior-artifact-id",
        help="Immutable artifact ID for --prior-adapter; required together.",
    )
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
