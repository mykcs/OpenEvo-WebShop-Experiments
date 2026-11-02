#!/usr/bin/env python3
"""CPU-only reconciliation: frozen Q17 R159 eight independent-seed dev attempts.

Source: one SHA-bound instrumented original R159 dev64 + seven frozen R159
repeat-1...repeat-7 runs. No model inference. No task text or raw actions
leave the server. This DOES NOT manufacture missing historical logits, and
does not access the sealed final WebShop panel.

Primary scientific targets (declared in CHECKPOINT_METRIC_BLINDSPOTS_V1.json):
- repeated_pass_all_k: mean_i product_{j=0}^{k-1} exact_success[i,j]
  for k=1..8 with frozen seed ordering. It is NOT (mean success)^k.
- task_success_heterogeneity: q_i, within-task Bernoulli variance and the
  across-task q_i dispersion corrected for finite k; task-cluster CI.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

MODEL_SHA = "18032c5cabd135b2464d5cbb2a3949aeb6130046dcc9f33acb33aee38c1baf46"
PANEL_SHA = "4033523c01697576b7f9ce8cddc3a63cf6c1514ef5554e01a8c82b59ee62d478"
MANIFEST_SHA = "2d5d1620065a15840ae20fed324038312aec74b0895e13c9d283177928bf1a3b"
MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
REPEAT_STRIDE = 10_000_000
N_TASKS = 64
N_REPEATS = 8
SAMPLING = {"temperature": 1.0, "top_p": 1.0, "top_k": 0,
            "max_new_tokens": 512, "max_steps": 15}
BASE = Path("/workspace/maintenance/"
            "Q17-GPU67-METRIC-20261012/q17gpu67-20261012-q1/"
            "r159/dev64/summary.json")
REPEATS = Path("/workspace/maintenance/"
               "Q17-TEMP-GPU0-5-20261013-M1-V2/m1")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def fail(message: str) -> None:
    raise ValueError(message)


def task_key(row: dict) -> tuple:
    return (row["task_identity"], int(row["probe_ordinal"]),
            int(row["source_session_index"]), int(row["worker_seed"]))


def load(base: Path, repeats: Path) -> tuple[list[tuple[dict, ...]], list[dict]]:
    raw = base.read_bytes()
    original = json.loads(raw)
    if (original.get("status") != "PASS"
            or original.get("task_count") != N_TASKS
            or original.get("panel_sha256") != PANEL_SHA
            or original.get("task_panel_manifest_sha256") != MANIFEST_SHA
            or original.get("instrumentation") != "on"
            or original.get("final_panel_access") != 0
            or original.get("formal_task_consumption") != 0
            or original.get("base_model_revision") != MODEL_REVISION
            or original.get("checkpoint", {}).get("adapter_sha256") != MODEL_SHA
            or len(original.get("rows", [])) != N_TASKS):
        fail("original R159 development model/panel/provenance mismatch")
    original_rows = original["rows"]
    base_map = {}
    base_seed_by_key = {}
    for row in original_rows:
        key = task_key(row)
        if key in base_map or row.get("engineering_invalid"):
            fail("duplicate or engineering-invalid original task")
        ep = row["episode"]
        if not isinstance(ep.get("won"), bool) or not (0.0 <= float(ep["score"]) <= 1.0):
            fail("invalid frozen initial outcome")
        if int(row["paired_generation_seed"]) < 0:
            fail("invalid original seed")
        base_map[key] = row
        base_seed_by_key[key] = int(row["paired_generation_seed"])
    if len(base_map) != N_TASKS or set(int(x[1]) for x in base_map) != set(range(N_TASKS)):
        fail("original task identity/ordinal panel not complete")

    matrices = {k: [base_map[k]] for k in base_map}
    sources = [{
        "repeat_index": 0, "source": "W0_instrumented_R159_dev64",
        "summary_sha256": sha(raw),
        "task_count": N_TASKS,
        "exact": sum(int(r["episode"]["won"]) for r in base_map.values()),
        "mean_score": statistics.mean(float(r["episode"]["score"]) for r in base_map.values()),
        "seed_stride": 0, "original_instrumentation": "on",
    }]

    for rep in range(1, N_REPEATS):
        dirs = sorted(repeats.glob("gpu*/repeat-" + str(rep)))
        if len(dirs) != 1:
            fail("expected exactly one distinct repeat root for rep " + str(rep))
        root = dirs[0]
        meta_raw = (root / "summary.json").read_bytes()
        meta = json.loads(meta_raw)
        attempt_raw = (root / "attempts.jsonl").read_bytes()
        expected = {
            "status": "PASS", "task_count": N_TASKS, "repeat_index": rep,
            "repeat_seed_stride": REPEAT_STRIDE,
            "checkpoint_adapter_sha256": MODEL_SHA,
            "model_revision": MODEL_REVISION,
            "panel_sha256": PANEL_SHA,
            "source_development_manifest_sha256": MANIFEST_SHA,
            "formal_task_consumption": 0, "final_panel_access": 0,
            "training_updates": 0, "engineering_invalid": 0,
            "generation_sampling": SAMPLING
        }
        for name, value in expected.items():
            if meta.get(name) != value:
                fail("repeat " + str(rep) + " contract mismatch: " + name)
        if sha(attempt_raw) != meta.get("raw_attempts_sha256"):
            fail("repeat " + str(rep) + " raw attempts SHA mismatch")
        rows = [json.loads(line) for line in attempt_raw.splitlines()]
        if len(rows) != N_TASKS:
            fail("repeat " + str(rep) + " count not 64")
        mapping = {}
        for row in rows:
            key = task_key(row)
            if key in mapping or key not in base_map:
                fail("repeat " + str(rep) + " unexpected/duplicated task")
            if (row.get("adapter_sha256") != MODEL_SHA
                    or int(row["repeat_index"]) != rep
                    or int(row["paired_generation_seed_original"]) != base_seed_by_key[key]
                    or int(row["repeat_seed_base"]) != base_seed_by_key[key] + rep * REPEAT_STRIDE
                    or row.get("engineering_invalid") is not False):
                fail("repeat " + str(rep) + " model, seed or engineering identity mismatch")
            ep = row["episode"]
            if (not isinstance(ep.get("won"), bool)
                    or not (0 <= float(ep["score"]) <= 1)):
                fail("repeat " + str(rep) + " invalid outcome")
            mapping[key] = row
        if set(mapping) != set(base_map):
            fail("repeat " + str(rep) + " incomplete frozen task identities")
        n_exact = sum(int(r["episode"]["won"]) for r in mapping.values())
        score_mean = statistics.mean(float(r["episode"]["score"]) for r in mapping.values())
        if (n_exact != int(meta["exact_success"])
                or abs(score_mean - float(meta["mean_score"])) > 1e-12):
            fail("repeat " + str(rep) + " outcome does not match sealed summary")
        for key in base_map:
            matrices[key].append(mapping[key])
        sources.append({
            "repeat_index": rep, "source": "M1_GPU0_5_V2_repeat-" + str(rep),
            "summary_sha256": sha(meta_raw), "attempts_sha256": sha(attempt_raw),
            "task_count": N_TASKS, "exact": n_exact, "mean_score": score_mean,
            "seed_stride": REPEAT_STRIDE, "original_instrumentation": "off",
            "new_development_rollouts": N_TASKS,
            "gpu_uuid": meta["gpu_uuid"],
            "wall_time_seconds": float(meta["wall_time_seconds"]),
        })
    if len(sources) != N_REPEATS or any(len(x) != N_REPEATS for x in matrices.values()):
        fail("eight complete aligned runs required")
    ordered = [tuple(matrices[k]) for k in sorted(matrices, key=lambda key: key[1])]
    return ordered, sources


def rate_metrics(m: list[tuple[dict, ...]]) -> dict:
    successes = [[int(x["episode"]["won"]) for x in rows] for rows in m]
    scores = [[float(x["episode"]["score"]) for x in rows] for rows in m]
    n = len(successes)
    k = len(successes[0])
    if n != 64 or k != 8:
        fail("invalid matrix")
    q = [sum(z)/k for z in successes]
    histogram = Counter(sum(z) for z in successes)
    within_unbiased = [k/(k-1) * v*(1-v) for v in q]
    mean_within = statistics.mean(within_unbiased)
    observed_between_qhat_variance = statistics.variance(q)
    between_corrected = observed_between_qhat_variance - mean_within/k
    all_success = {str(p): sum(all(seq[:p]) for seq in successes)/n for p in range(1,k+1)}
    any_success = {str(p): sum(any(seq[:p]) for seq in successes)/n for p in range(1,k+1)}
    within_mixed = sum(0 < sum(seq) < k for seq in successes)
    return {
        "task_count": n, "repeat_count_including_frozen_base": k,
        "binary_success_observations": n*k,
        "exact_success_counts_by_repeat": [sum(seq[j] for seq in successes) for j in range(k)],
        "mean_exact_success_probability": statistics.mean(q),
        "overall_mean_task_score": statistics.mean(sum(s)/k for s in scores),
        "pass_all_k_by_exact_seed_prefix": all_success,
        "success_at_least_once_k_diagnostic": any_success,
        "success_count_per_task_histogram": {str(i): histogram.get(i,0) for i in range(k+1)},
        "unreliable_some_but_not_all_success_tasks": within_mixed,
        "always_successful_tasks": histogram.get(k,0),
        "never_successful_tasks": histogram.get(0,0),
        "mean_within_task_bernoulli_sample_variance": mean_within,
        "across_task_observed_qhat_sample_variance": observed_between_qhat_variance,
        "across_task_finite_k_corrected_variance_raw": between_corrected,
        "across_task_finite_k_corrected_variance_clipped": max(0.0,between_corrected),
        "all_task_qhat_standard_error_proxy": statistics.stdev(q)/math.sqrt(n),
        "naive_p_to_8_is_not_primary": statistics.mean(q)**8,
    }


def boot_ci(m: list[tuple[dict, ...]], loops: int=20_000) -> dict:
    random_seed = 15920261013
    rng = random.Random(random_seed)
    succ = [[int(x["episode"]["won"]) for x in row] for row in m]
    prefixes = [[int(all(z[:k])) for k in range(1,9)] for z in succ]
    qs = [sum(z)/8 for z in succ]
    n = len(succ)
    collects = {str(k):[] for k in (1,2,4,8)}
    collects["mean_q"] = []
    collects["within_unbiased_var"] = []
    for _ in range(loops):
        indices = [rng.randrange(n) for _ in range(n)]
        for k in (1,2,4,8):
            collects[str(k)].append(sum(prefixes[i][k-1] for i in indices)/n)
        collects["mean_q"].append(sum(qs[i] for i in indices)/n)
        collects["within_unbiased_var"].append(sum(8/7*qs[i]*(1-qs[i]) for i in indices)/n)
    ci={}
    for tag, sample in collects.items():
        a=sorted(sample)
        ci[tag]={"lower_95_task_cluster_percentile":a[int(.025*loops)],
                 "upper_95_task_cluster_percentile":a[math.ceil(.975*loops)-1]}
    return {"method":"nonparametric resample 64 matched task clusters with their 8 repeated outcomes intact",
            "bootstrap_repetitions":loops,"bootstrap_seed":random_seed,"estimates":ci}


def run(base:Path, repeats:Path, *, bootstrap:int=20_000) -> dict:
    matrix, sources = load(base,repeats)
    result = rate_metrics(matrix)
    ci = boot_ci(matrix,bootstrap)
    return {
        "schema":"openevo.q17.r159.repeated8_exact_success_cpu.v1",
        "status":"PASS_SHA_BOUND_FROZEN_64_X_8",
        "science":{"checkpoint_adapter_sha256":MODEL_SHA, "frozen_dev_panel_sha256":PANEL_SHA,
                   "source_panel_manifest_sha256":MANIFEST_SHA,"base_model_revision":MODEL_REVISION,
                   "task_horizon":15,"temperature":1.0,"top_p":1.0,"top_k":0,
                   "base_seed_policy":"paired_generation_seed","repeat_seed_stride":REPEAT_STRIDE,
                   "seed_order":"original W0 first, then preselected repeat indexes 1..7",
                   "original_generation_probabilities_not_recovered":True,
                   "model_checkpoint_changes":0,"formal_task_consumption":0,"final_panel_access":0},
        "measurement":result,"task_cluster_95ci":ci,"sources":sources,
        "guardrails":{"repeated_pass_all_k_is_not_mean_p_to_k":True,
                      "source_repeat_raw_sha256_verified":True,
                      "same_exact_64_task_identities_across_eight_repetitions":True,
                      "repeat_seed_vectors_pairwise_disjoint":True,
                      "no_task_text_or_actions_in_result":True,
                      "finite_repetitions_bias_on_q_i_corrected_as_sample_variance":True,
                      "new_repeat_task_consumption":64*7,
                      "old_w0_development_tasks_reused":64,
                      "cpu_only":True,"gpu_calls":0,"new_model_inference":0,
                      "model_or_checkpoint_deletions":0}
    }


def main(argv:list[str]|None=None)->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base",type=Path,default=BASE)
    ap.add_argument("--repeats",type=Path,default=REPEATS)
    ap.add_argument("--bootstrap",type=int,default=20_000)
    args=ap.parse_args(argv)
    if args.bootstrap not in (100,20_000):
        raise SystemExit("bootstrap must be exactly 100 (dev test) or 20000 (full receipt)")
    try:
        print(json.dumps(run(args.base,args.repeats,bootstrap=args.bootstrap),sort_keys=True,ensure_ascii=False),flush=True)
        return 0
    except (ValueError,KeyError,FileNotFoundError,TypeError) as exc:
        print(json.dumps({"status":"BLOCKED_SCIENCE_IDENTITY_MISMATCH","reason":str(exc),
                          "gpu_calls":0,"new_webshop_episodes":0},ensure_ascii=False),file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
