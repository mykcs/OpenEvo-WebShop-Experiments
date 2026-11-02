import json

from scripts.openevo_webshop.reconcile_h1_18 import reconcile


def test_reconcile_h1_18_requires_complete_three_arm_panel(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "task_selection": {"partitions": {"P1": [744, 1263], "P2": [1623, 3020], "P3": [3848, 4415], "P4": [5593, 6091]}},
                "rollout": {"seeds": [2026082700, 2026082701, 2026082702, 2026082703]},
            }
        ),
        encoding="utf-8",
    )
    cells = []
    for arm in ("base", "success-state", "student-recovery"):
        for partition, goals in (("P1", [744, 1263]), ("P2", [1623, 3020]), ("P3", [3848, 4415]), ("P4", [5593, 6091])):
            cell = tmp_path / f"{arm}-{partition}"
            (cell / "episodes").mkdir(parents=True)
            with (cell / "attempts.jsonl").open("w", encoding="utf-8") as stream:
                for goal in goals:
                    for seed in [2026082700, 2026082701, 2026082702, 2026082703]:
                        stream.write(
                            json.dumps(
                                {
                                    "arm": arm,
                                    "task_id": f"webshop_{goal:06d}",
                                    "goal_idx": goal,
                                    "rollout_seed": seed,
                                    "final_task_score": 0.0,
                                    "identity_ok": True,
                                    "trace_complete": True,
                                    "parser_or_fallback_failure": False,
                                    "error": None,
                                    "qualified_positive": False,
                                    # Each partition models a UUID-bound GPU cell;
                                    # cross-cell fingerprints may differ.
                                    "protocol_fingerprint": (
                                        ("a" if arm == "base" else ("b" if arm == "success-state" else "c"))
                                        * 64
                                    )[:64 - len(str(goals[0]))] + str(goals[0]),
                                }
                            )
                            + "\n"
                        )
            cells.append(f"{arm}:{partition}:{cell}")

    result = reconcile(
        manifest_path,
        cells,
        tmp_path / "reconciliation.json",
        train_task_ids={"webshop_003420", "webshop_006624"},
        expected_git_sha="3d69137",
        expected_upstream_sha="796e6248a61112a61c880885844d3b0420c0b6ae",
    )

    assert result["status"] == "complete"
    assert result["attempt_count"] == 96
    assert result["valid_count"] == 96
    assert result["paired_count"] == 32
    assert result["qualified_positive_count"] == 0
