from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_spectrum_metrics_rank_and_energy() -> None:
    module = load_module(
        "ceiling1_measurement_generator_test",
        "scripts/openevo_webshop/generate_ceiling1_parameter_measurement_pack.py",
    )
    metrics = module.spectrum_metrics([3.0, 1.0])
    assert math.isclose(metrics["stable_rank"], 10.0 / 9.0)
    assert math.isclose(metrics["top1_energy_share"], 0.9)
    assert metrics["rank90_energy"] == 1
    assert metrics["rank95_energy"] == 2
    assert math.isclose(metrics["spectral_gap_s1_s2"], 3.0)


def test_module_groups_are_global_layer_and_projection() -> None:
    module = load_module(
        "ceiling1_measurement_generator_groups_test",
        "scripts/openevo_webshop/generate_ceiling1_parameter_measurement_pack.py",
    )
    modules = [
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.1.self_attn.q_proj",
    ]
    groups = module.module_groups(modules)
    assert groups[("global", "all")] == modules
    assert groups[("layer", "0")] == modules[:2]
    assert groups[("layer", "1")] == modules[2:]
    assert groups[("projection", "q_proj")] == [modules[0], modules[2]]
    assert groups[("projection", "k_proj")] == [modules[1]]


def test_gpu_replay_selection_is_stable_and_record_bound(tmp_path: Path) -> None:
    module = load_module(
        "ceiling1_gpu_parameter_analysis_test",
        "scripts/openevo_webshop/run_ceiling1_gpu_parameter_mechanism_analysis.py",
    )
    path = tmp_path / "replay.jsonl"
    rows = [
        {
            "messages": [],
            "metadata": {"event_id": event_id, "session_id": "session", "task_id": "task", "trace_index": trace_index},
            "target_message_start": 0,
        }
        for event_id, trace_index in (("event-c", 0), ("event-a", 0), ("event-a", 1), ("event-b", 0))
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    selected = module.deterministic_replay(path, 3)
    expected = []
    for row in rows:
        metadata = row["metadata"]
        identity = json.dumps(
            {
                "event_id": metadata.get("event_id"),
                "session_id": metadata.get("session_id"),
                "task_id": metadata.get("task_id"),
                "trace_index": metadata.get("trace_index"),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        expected.append((hashlib.sha256(identity.encode("utf-8")).hexdigest(), metadata["event_id"], metadata["trace_index"]))
    expected.sort()
    expected = expected[:3]
    assert [(row["metadata"]["event_id"], row["metadata"]["trace_index"]) for row in selected] == [
        (event_id, trace_index) for _, event_id, trace_index in expected
    ]
    assert [row["_selection_key"] for row in selected] == [digest for digest, _, _ in expected]
