#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re, statistics
from collections import Counter, defaultdict
from pathlib import Path

def load(path: Path):
    return json.loads(path.read_text())

def percentile(values: list[float], p: float) -> float:
    xs = sorted(values)
    pos = (len(xs) - 1) * p
    lo = int(pos); hi = min(lo + 1, len(xs) - 1); w = pos - lo
    return xs[lo] * (1 - w) + xs[hi] * w

def seconds_stats(values: list[float]) -> dict:
    return {"count": len(values), "sum_seconds": sum(values),
        "mean_seconds": statistics.mean(values),
        "median_seconds": statistics.median(values),
        "p95_seconds": percentile(values, 0.95),
        "min_seconds": min(values), "max_seconds": max(values)}
def analyze_arm(root: Path, arm: str, successor_runtime: Path) -> dict:
    round_wall, shard_wall = [], []
    gpu_seconds, gpu_shards = defaultdict(float), Counter()
    gpu_rounds = defaultdict(list)
    trans, train, model_load, compression, peaks, losses = [], [], [], [], [], []
    engineering_invalid = gdr_writes = gdr_events = gdr_zero = 0
    transition_status = Counter()
    beta_min, beta_max, write_error = [], [], []
    carrier_status = {k: Counter() for k in ("text-memory", "skill-bundle", "agent-system")}
    carrier_calls, health_status, health_action = Counter(), Counter(), Counter()
    health_coverage = content_rejections = 0
    for index in range(160):
        rd = root / arm / f"round-{index:03d}"
        summary = load(rd / "summary.json")
        if summary["status"] != "PASS" or int(summary["attempt_count"]) != 128:
            raise RuntimeError(f"{arm} round {index} rollout denominator drift")
        round_wall.append(float(summary["wall_time_seconds"]))
        engineering_invalid += int(summary["engineering_invalid_count"])
        for shard_index in range(4):
            shard = load(rd / f"shard-{shard_index:02d}" / "summary.json")
            seconds = float(shard["wall_time_seconds"]); gpu = shard["gpu_uuid"]
            shard_wall.append(seconds); gpu_seconds[gpu] += seconds; gpu_shards[gpu] += 1
            gpu_rounds[gpu].append(index)
        terminal = load(rd / "sd-lora" / "terminal-receipt.json")
        transition_status[str(terminal.get("status"))] += 1
        receipt_path = rd / "sd-lora" / "bounded-state" / "bounded_recurrence_receipt.json"
        if receipt_path.exists():
            receipt = load(receipt_path)
            timing = receipt["timing"]
            trans.append(float(timing["total_seconds"]))
            train.append(float(timing["train_loop_seconds"]))
            model_load.append(float(timing["model_load_seconds"]))
            compression.append(float(timing["compression_seconds"]))
            peaks.append(int(receipt["gpu_peak_memory_bytes_total"]))
            losses.append(float(receipt["training_loss_mean"]))
            gdr_writes += int(receipt.get("gdr_applied_write_count") or 0)
            gdr_events += int(receipt.get("gdr_optimizer_event_count") or 0)
            gdr_zero += int(receipt.get("gdr_zero_direction_noop_count") or 0)
            if receipt.get("gdr_effective_beta_min") is not None:
                beta_min.append(float(receipt["gdr_effective_beta_min"]))
            if receipt.get("gdr_effective_beta_max") is not None:
                beta_max.append(float(receipt["gdr_effective_beta_max"]))
            if receipt.get("gdr_max_intended_effective_write_abs_error") is not None:
                write_error.append(float(receipt["gdr_max_intended_effective_write_abs_error"]))
        for kind in carrier_status:
            path = rd / kind / "terminal-receipt.json"
            if path.exists():
                item = load(path); carrier_status[kind][str(item.get("status"))] += 1
                carrier_calls[kind] += int(item.get("model_calls_consumed") or 0)
        health_path = rd / "carrier-health-effective-boundary.json"
        if health_path.exists():
            item = load(health_path); health_coverage += 1
            health_status[str(item.get("status"))] += 1
            health_action[str(item.get("action"))] += 1
            content_rejections += len(item.get("content_rejections") or [])
    seal = load(root / arm / "FORMAL_ARM_SEALED.json")
    values = list(gpu_seconds.values())
    log_path = successor_runtime / f"{arm}.log"
    text = log_path.read_text(errors="replace") if log_path.exists() else ""
    return {"stage2": {"seal_status": seal["status"], "final_state_id": seal["final_state_id"],
        "final_panel_access_count": seal["final_panel_access_count"]},
      "rollout": {"round_critical_path": seconds_stats(round_wall),
        "shard_wall": seconds_stats(shard_wall), "gpu_active_hours": sum(shard_wall) / 3600,
        "critical_path_hours": sum(round_wall) / 3600,
        "engineering_invalid_total": engineering_invalid,
        "gpu_uuid_shard_count": dict(sorted(gpu_shards.items())),
        "gpu_uuid_active_seconds": dict(sorted(gpu_seconds.items())),
        "gpu_uuid_rounds": {gpu: rounds for gpu, rounds in sorted(gpu_rounds.items())},
        "uuid_active_seconds_max_minus_min": max(values) - min(values),
        "uuid_active_seconds_ratio_max_min": max(values) / min(values)},
      "bounded_transition": {"terminal_status_counts": dict(transition_status),
        "timed_update_count": len(trans), "untimed_noop_count": transition_status.get("NOOP", 0),
        "total": seconds_stats(trans), "train_loop": seconds_stats(train),
        "model_load": seconds_stats(model_load), "compression": seconds_stats(compression),
        "gpu_active_equivalent_hours": sum(trans) / 3600,
        "peak_vram_gib_max": max(peaks) / (1024 ** 3),
        "peak_vram_gib_mean": statistics.mean(peaks) / (1024 ** 3),
        "training_loss_mean_across_rounds": statistics.mean(losses),
        "training_loss_round0": losses[0], "training_loss_round159": losses[-1]},
      "gdr": {"applied_write_count_total": gdr_writes,
        "optimizer_event_count_total": gdr_events, "zero_direction_noop_count_total": gdr_zero,
        "effective_beta_min_observed": min(beta_min) if beta_min else None,
        "effective_beta_max_observed": max(beta_max) if beta_max else None,
        "max_intended_effective_write_abs_error_observed": max(write_error) if write_error else None},
      "carriers": {"terminal_status_counts": {k: dict(v) for k, v in carrier_status.items()},
        "model_calls_consumed_total": dict(carrier_calls),
        "effective_health_receipt_round_coverage": health_coverage,
        "effective_health_missing_rounds": 160 - health_coverage,
        "effective_health_status_counts": dict(health_status),
        "effective_health_action_counts": dict(health_action),
        "content_rejection_entries_total": content_rejections},
      "active_successor_log_diagnostics": {
        "out_of_memory_literal_count": len(re.findall(r"out of memory|CUDA OOM|OutOfMemory", text, re.I)),
        "traceback_literal_count": len(re.findall(r"Traceback \(most recent call last\)", text))},
      "measured_gpu_active_hours_lower_bound": (sum(shard_wall) + sum(trans)) / 3600}

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--successor-runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"schema": "openevo.bounded-effective-state.stage2-gpu-resource-analysis.20260918",
      "status": "PASS_DESCRIPTIVE_ONLY", "source_root": str(args.root.resolve()),
      "rounds_per_arm": 160, "formal_rollouts_per_arm": 20480,
      "scientific_denominator_changed": False, "new_model_calls": 0,
      "final_panel_used_for_resource_analysis": False,
      "arms": {arm: analyze_arm(args.root.resolve(), arm, args.successor_runtime.resolve())
               for arm in ("off", "on")}}
    result["all_rollout_gpu_uuids"] = sorted({gpu for arm in result["arms"].values()
        for gpu in arm["rollout"]["gpu_uuid_shard_count"]})
    result["measurement_notes"] = [
      "Rollout GPU-active time is exact sum of per-shard wall_time_seconds.",
      "Rollout critical-path time excludes inter-round queue/resource wait.",
      "Transition GPU-active equivalent sums bounded recurrence total_seconds; transition UUID is not recorded.",
      "Carrier effective-health coverage varies because that receipt was introduced mid-campaign; missing rounds are counted, not imputed.",
      "Exact carrier call wall time and scheduler/lease wait seconds are unavailable in sealed receipts.",
      "OOM evidence is a literal scan of active-successor arm logs plus sealed-receipt validity, not a host-wide historical audit."]
    result["unavailable_exact_metrics"] = [
      "carrier model-call wall time by kind", "Ray queue/reservation/lease wait seconds",
      "transition GPU UUID attribution", "host-wide historical OOM/exit accounting before active successor"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({arm: {"gpu_hours_lower_bound": result["arms"][arm]["measured_gpu_active_hours_lower_bound"],
      "peak_vram_gib": result["arms"][arm]["bounded_transition"]["peak_vram_gib_max"],
      "gdr_writes": result["arms"][arm]["gdr"]["applied_write_count_total"]}
      for arm in ("off", "on")}, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
