#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scripts.openevo_webshop.post_advisor_e3_guard_202609300248 import verify_formal_activation

SUCCESSOR_NAME = "post_advisor_bounded_rank_sweep_successor_202609282338.py"
SUCCESSOR_SHA256 = "1dbed9b7a286aedfd52242517f0f11ca0d8c65fbbd1c321b3059e111793f5a20"
OFF_MODE = "BOUNDED_OFF"
ON_MODE = "EFFECTIVE_STATE_GDR_LORA_V1"  # legacy-revision-compat: exact historical frozen identifier
MAG_MODE = "MAGNITUDE_SCALAR_GATE_V1"  # legacy-revision-compat: exact historical frozen identifier
TRACE_QUALIFICATION_REL = "docs/evidence/bounded-recurrence-gdr-20260915/gdr-step-trace-qualification-202609152040/result.json"
TRACE_QUALIFICATION_SHA256 = "eb9c4b54cf097bb4f829fe683d0b211ac60524dddf2737b624570dc452cdbc5c"
TRACE_QUALIFICATION_STATUS = "PASS_EFFECTIVE_STATE_GDR_STEP_TRACE_EXACT_EQUIVALENCE"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_trace_qualification() -> None:
    repo = Path(__file__).resolve().parents[2]
    path = repo / TRACE_QUALIFICATION_REL
    if not path.is_file() or sha(path) != TRACE_QUALIFICATION_SHA256:
        raise RuntimeError("GDR step-trace qualification receipt SHA drift")
    payload = json.loads(path.read_text())
    if payload.get("status") != TRACE_QUALIFICATION_STATUS:
        raise RuntimeError("GDR step-trace qualification is not PASS")
    boundary = payload.get("boundary") or {}
    if int(boundary.get("formal_rows_consumed", -1)) != 0 or int(boundary.get("final_panel_access", -1)) != 0:
        raise RuntimeError("GDR step-trace qualification protected-counter drift")


def load_successor():
    validate_trace_qualification()
    path = Path(__file__).resolve().parents[1] / "analysis" / SUCCESSOR_NAME
    if sha(path) != SUCCESSOR_SHA256:
        raise RuntimeError("formal transition successor SHA drift")
    spec = importlib.util.spec_from_file_location("bounded_effective_state_successor", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen bounded successor")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def replay_expectations(successor, args):
    engine = successor.load_engine()
    from openevo.evolution.parametric import sd_lora_trainer as trainer
    from openevo.evolution.parametric.sd_lora import _training_examples
    from openevo.evolution.framework.contracts import canonical_json

    dataset = engine.build_dataset_artifact(args.dataset_manifest, args.dataset_artifact_id)
    current = _training_examples((dataset,), maximum=256, minimum_reward=0.5)
    if not current:
        raise RuntimeError("formal bounded transition requires a non-empty current dataset")
    if args.round_index == 0:
        prior_replay = []
    else:
        if args.prior_replay is None:
            raise RuntimeError("Round1+ requires prior replay")
        prior_count = sum(1 for line in args.prior_replay.read_text().splitlines() if line.strip())
        prior_replay = trainer._read_examples(args.prior_replay, expected_count=prior_count)
    replay_seed = args.seed + args.round_index
    replay = trainer._select_replay_examples(prior_replay, count=len(current), seed=replay_seed)
    replay_buffer = trainer._bounded_replay_buffer(prior_replay, current, capacity=args.replay_capacity)
    blob = "".join(canonical_json(row) + "\n" for row in replay_buffer).encode()
    return {
        "current_count": len(current),
        "prior_replay_count": len(prior_replay),
        "replay_count": len(replay),
        "output_replay_count": len(replay_buffer),
        "steps": len(current) + len(replay),
        "output_replay_sha256": hashlib.sha256(blob).hexdigest(),
    }

def invoke_successor(successor, args, exp: dict) -> int:
    records = args.dataset_manifest.with_name("records.jsonl")
    argv = [
        SUCCESSOR_NAME,
        "--transition-kind", "empty-first-state" if args.round_index == 0 else "nonempty-steady-state",
        "--indexing-mode", "new-zero-based",
        "--dataset-manifest", str(args.dataset_manifest),
        "--dataset-artifact-id", args.dataset_artifact_id,
        "--output-dir", str(args.output_dir),
        "--model-path", str(args.model_path),
        "--round-index", str(args.round_index),
        "--logical-history-count", str(args.round_index),
        "--state-rank", str(args.state_rank),
        "--seed", str(args.seed),
        "--replay-capacity", str(args.replay_capacity),
        "--expected-current-count", str(exp["current_count"]),
        "--expected-prior-replay-count", str(exp["prior_replay_count"]),
        "--expected-replay-count", str(exp["replay_count"]),
        "--expected-output-replay-count", str(exp["output_replay_count"]),
        "--expected-steps", str(exp["steps"]),
        "--expected-records-sha", sha(records),
        "--expected-output-replay-sha", exp["output_replay_sha256"],
        "--transition-id", args.transition_id,
        "--treatment-mode", args.treatment_mode,
    ]
    if args.round_index == 0:
        argv += ["--empty-history-manifest", str(args.empty_history_manifest)]
        argv += ["--adapter-config-template", str(args.adapter_config_template)]
    else:
        argv += ["--prior-adapter", str(args.prior_adapter)]
        argv += ["--prior-replay", str(args.prior_replay)]
        argv += ["--expected-prior-sha", sha(args.prior_adapter / "adapter_model.safetensors")]
    if args.treatment_mode == ON_MODE:
        argv += ["--gdr-runtime-path", str(args.gdr_runtime_path)]
        argv += ["--beta-policy-path", str(args.beta_policy_path)]
    prior = sys.argv
    try:
        sys.argv = argv
        return int(successor.main())
    finally:
        sys.argv = prior

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-manifest", type=Path, required=True)
    p.add_argument("--dataset-artifact-id", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--round-index", type=int, required=True)
    p.add_argument("--transition-id", required=True)
    p.add_argument("--treatment-mode", choices=(OFF_MODE,), required=True)
    p.add_argument("--prior-adapter", type=Path)
    p.add_argument("--prior-replay", type=Path)
    p.add_argument("--empty-history-manifest", type=Path, required=True)
    p.add_argument("--adapter-config-template", type=Path, required=True)
    p.add_argument("--gdr-runtime-path", type=Path)
    p.add_argument("--beta-policy-path", type=Path)
    p.add_argument("--seed", type=int, default=1993)
    p.add_argument("--replay-capacity", type=int, default=64)
    p.add_argument("--state-rank", type=int, choices=(8, 16, 64), required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.round_index < 152 or args.round_index > 159:
        raise SystemExit("post-advisor short fork round must be 152..159")
    if args.round_index == 0 and (args.prior_adapter is not None or args.prior_replay is not None):
        raise SystemExit("Round0 EMPTY_BOUNDED_HISTORY forbids prior state")
    if args.round_index > 0 and (args.prior_adapter is None or args.prior_replay is None):
        raise SystemExit("short fork requires matching-rank prior state and replay")
    if args.treatment_mode == ON_MODE and (args.gdr_runtime_path is None or args.beta_policy_path is None):
        raise SystemExit("Effective-State ON requires runtime and policy")
    if args.treatment_mode != ON_MODE and (args.gdr_runtime_path is not None or args.beta_policy_path is not None):
        raise SystemExit("non-GDR treatment forbids runtime/policy inputs")
    try:
        verify_formal_activation(REPO_ROOT, Path(os.environ["OPENEVO_E3_AGGREGATE_AUTHORITY"]))
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    successor = load_successor()
    exp = replay_expectations(successor, args)
    print(json.dumps({"status": "EXPECTATIONS_PASS", **exp}, sort_keys=True), flush=True)
    return invoke_successor(successor, args, exp)


if __name__ == "__main__":
    raise SystemExit(main())
