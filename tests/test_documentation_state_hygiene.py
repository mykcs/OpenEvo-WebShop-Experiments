from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARENT_POLICY = (
    "https://github.com/mykcs/zju-server/blob/main/docs/experiment-execution-policy.md"
)
PARENT_GOVERNANCE = (
    "https://github.com/mykcs/zju-server/blob/main/docs/knowledge-state-governance.md"
)


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _normalized(text: str) -> str:
    return " ".join(text.lower().split())


def test_active_server_guidance_delegates_parent_policy() -> None:
    """Long-lived experiment guidance must not become a second server policy."""

    for path in ("README.md", "AGENTS.md"):
        text = _read(path)
        normalized = _normalized(text)

        assert PARENT_POLICY in text, f"{path} must link the canonical parent policy"
        assert PARENT_GOVERNANCE in text, f"{path} must link knowledge-state governance"
        assert not re.search(
            r"(?i)\beffective\s+20\d{2}-\d{2}-\d{2}\b", text
        ), f"{path} must not copy the parent policy effective date"

        forbidden = (
            "approved non-root path `experiment-host`",
            "use `experiment-host` or container uid/gid",
            "root reserved for control-plane actions",
            "root is the administrative and docker control-plane entry only",
            "root` is the administrative and docker control-plane entry only",
        )
        for fragment in forbidden:
            assert fragment not in normalized, (
                f"{path} reinstates a superseded host-account rule: {fragment!r}"
            )


def test_agent_front_door_exposes_canonical_host_storage_route() -> None:
    """A fresh Agent should not rediscover legacy host aliases before finding parent truth."""

    agents = _normalized(_read("AGENTS.md"))

    for fragment in (
        "/workspace",
        "/data/share",
        "/workspace/workspace",
        "managed-bind compatibility alias",
        "historical evidence keeps the path actually recorded at execution time",
        "container-internal destinations",
        "mykcs/zju-server/agents.md",
        "docs/server-facts.md",
    ):
        assert fragment in agents, f"AGENTS.md lost host-storage front-door route: {fragment}"

    assert "do not introduce `/workspace/workspace` as a new canonical host path" in agents
    assert "this is a startup hint, not a second host policy" in agents


def test_long_lived_indexes_route_instead_of_cache_current_state() -> None:
    """Index/router files must not freeze a concrete experiment/date as current."""

    config_index = _read("configs/experiment/README.md")
    evidence_index = _read("docs/evidence/README.md")
    run_index = _read("docs/science/webshop/runs/README.md")

    assert "current-campaign.json" in config_index
    assert "current-campaign.json" in evidence_index
    assert "CURRENT_FINDINGS.md" in evidence_index
    assert "current-campaign.json" in run_index
    assert "CURRENT_FINDINGS.md" in run_index

    stale_phrases = {
        "configs/experiment/README.md": (
            "for current work, the latest preregistration is",
            "latest preregistration is",
        ),
        "docs/evidence/README.md": (
            "current scientific status and decision-tree position",
            "— current scientific status",
        ),
        "docs/science/webshop/runs/README.md": (
            "current preregistered",
            "for the current research state, read:",
        ),
    }

    for path, phrases in stale_phrases.items():
        normalized = _normalized(_read(path))
        for phrase in phrases:
            assert phrase not in normalized, (
                f"{path} caches mutable current state via {phrase!r}"
            )


def test_current_index_lines_do_not_name_a_dated_or_h_numbered_current_truth() -> None:
    """Catch future variants of the same drift without banning historical prose."""

    paths = (
        "configs/experiment/README.md",
        "docs/evidence/README.md",
        "docs/science/webshop/runs/README.md",
    )
    dated = re.compile(r"20\d{2}-\d{2}-\d{2}")
    h_number = re.compile(r"\bH\d+(?:\.\d+)?\b", re.I)

    for path in paths:
        for lineno, line in enumerate(_read(path).splitlines(), start=1):
            lowered = line.lower()
            if "histor" in lowered or "do not" in lowered or "not " in lowered:
                continue
            claims_current = "current" in lowered or "latest" in lowered
            if not claims_current:
                continue
            assert not dated.search(line), (
                f"{path}:{lineno} labels a concrete date as durable current/latest state"
            )
            assert not h_number.search(line), (
                f"{path}:{lineno} labels a concrete H number as durable current/latest state"
            )


def test_agents_promotes_repeated_operator_and_external_auth_boundaries() -> None:
    """Repeated human corrections must remain visible at the Agent front door."""

    agents = _normalized(_read("AGENTS.md"))
    autonomy = _normalized(_read("docs/operations/governance/GOAL_MODE_AUTONOMY.md"))
    troubleshooting = _read("docs/troubleshooting/README.md")

    assert "select the shell explicitly for complex remote logic" in agents
    assert "unknown dirty/untracked worktrees as owned until proven otherwise" in agents
    assert "external compute and authentication boundary" in agents
    assert "missing legacy token file or environment variable" in agents

    assert "external notebook / saas authentication" in autonomy
    assert "missing `~/.kaggle/kaggle.json`" in autonomy
    assert "ssl timeout after auth pass" in autonomy

    assert "CASE-EXTERNAL-NOTEBOOK-AUTH-QWEN3-SMOKE-20260902.md" in troubleshooting

def test_stage2_continuation_anti_regression_is_front_door_visible() -> None:
    agents = _normalized(_read("AGENTS.md"))
    execution_router = _read("docs/operations/execution/README.md")
    holder = _normalized(_read("docs/operations/execution/GPU_LEASE_AND_HOLDER_HANDOFF.md"))
    old_case = _read("docs/troubleshooting/experiment-ops/CASE-CEILING1-END-TO-END-SCIENCE-ENGINEERING-RETROSPECTIVE-20260831.md")
    friction = _read("docs/operations/execution/FRICTION_REGISTER.md")

    for fragment in (
        "a pass is scoped",
        "crash is not a scheduling boundary",
        "reconcile before replay",
        "free vram is not permission to add workers",
    ):
        assert fragment in agents, f"AGENTS.md lost Stage2 anti-regression rule: {fragment}"

    assert "STAGE2_SAFE_CONTINUATION_AND_SHARED_GPU.md" in execution_router
    assert "conditional on the **active** resource policy" in holder
    assert "holder_policy.enabled=false" in holder
    assert "2026-09-04 anti-regression warning" in old_case
    assert "A1-era historical rule" in friction
    assert "当前 A1 固定" not in friction


def test_storage_reclaim_auth_and_commit_boundaries_stay_front_door_visible() -> None:
    """Repeated storage/auth corrections must not sink back into dated cases only."""

    agents = _normalized(_read("AGENTS.md"))
    playbook = _normalized(_read("docs/operations/publication/SERVER_ARTIFACT_PUBLICATION_PLAYBOOK.zh-CN.md"))
    friction = _read("docs/operations/execution/FRICTION_REGISTER.md")

    for fragment in (
        "credential repair is not mutation authority",
        "blob transfer is not archive closure",
        "one checkpoint transaction at a time",
    ):
        assert fragment in agents, f"AGENTS.md lost storage/auth anti-friction rule: {fragment}"

    assert "授权修复本身不是 publication/cleanup 的继续授权" in playbook
    assert "blob/cas/xet 上传完成不等于 immutable archive commit 完成" in playbook
    assert "do-not-delete" in friction
    assert "no commit = keep local" in friction


def test_ceiling1_conversation_lessons_stay_promoted_and_scope_safe() -> None:
    """2026-09-04 repeated corrections must remain startup-visible and mechanical."""

    agents = _normalized(_read("AGENTS.md"))
    standard = _normalized(_read("docs/operations/governance/EXPERIMENT_STANDARD.md"))
    friction = _read("docs/operations/execution/FRICTION_REGISTER.md")
    old_case = _read(
        "docs/troubleshooting/experiment-ops/"
        "CASE-CEILING1-END-TO-END-SCIENCE-ENGINEERING-RETROSPECTIVE-20260831.md"
    )
    current_case = _read(
        "docs/troubleshooting/experiment-ops/"
        "CASE-CEILING1-STAGE2-CONTINUATION-SHARED-GPU-RETROSPECTIVE-20260904.md"
    )

    for fragment in (
        "one shared engineering/method implementation plus an explicit arm spec",
        "str.splitlines()",
        "completion-evidence ladder",
        "parameter_update_count",
        "goal-first",
    ):
        assert fragment in agents, f"AGENTS.md lost promoted conversation rule: {fragment}"

    for fragment in (
        "parameter_state_update_count",
        "candidate_update_attempted=1",
        "active_update_accepted=0",
        "one `by_rollout/*` behavior point per sealed round",
        "closed parameter state",
    ):
        assert fragment in standard, f"EXPERIMENT_STANDARD lost Stage2 metric rule: {fragment}"

    old_case_norm = _normalized(old_case.replace("\n> ", "\n"))
    assert "historical-method warning" in old_case_norm
    assert "a1.1/4096" in old_case_norm
    assert "current text memory successor" in old_case_norm
    assert "max split depth 1" in old_case_norm
    for heading in (
        "## 11. Config intent is not runtime-effective behavior",
        "## 12. Interrupted collection and JSONL recovery",
        "## 13. MiniMax/provider resume",
        "## 14. OPSD resume",
        "## 15. Checkpoint/HF/W&B",
        "## 16. Completion claims need an evidence ladder",
        "## 17. Repeated-correction priority list",
    ):
        assert heading in current_case, f"current retrospective lost section: {heading}"

    for heading in (
        "### AM. Completion claims must match the strongest verified evidence rung",
        "### AN. JSONL records are physical LF records",
        "### AO. Shared-arm engineering means shared fixes",
        "### AP. Candidate training is not an accepted model-state update",
    ):
        assert heading in friction, f"Friction Register lost promoted rule: {heading}"


def test_ceiling1_master_plan_current_authority_does_not_regress() -> None:
    """Known stale A1/MiniMax control wording must not re-enter the current plan."""

    master = _read(
        "docs/experiment-tracking/OPEN_EVO_CEILING_1.0_END_TO_END_MASTER_PLAN.md"
    )
    current = _read("docs/experiment-tracking/OPEN_EVO_CEILING_1.0_CURRENT.md")

    forbidden = (
        "只有两个 arm 的 canonical seal 与 post-collection audit 都 PASS",
        "两个 arm 都 seal 后 pipeline **明确停止**",
        "pipeline governance commit 为 `8667c4",
        "MiniMax analyzer execution：`c60419",
    )
    for fragment in forbidden:
        assert fragment not in master, f"Master Plan revived stale authority: {fragment}"

    for fragment in (
        "每个 arm **独立**完成自己的 canonical seal",
        "A1/2048 与对称 A1.1/4096 都是历史/已否决方案",
        "deterministic `10+10`",
        "`max_split_depth=1`",
        "`PRE_STAGE2_READY` 仍不等于 Stage2 activation",
    ):
        assert fragment in master, f"Master Plan lost current authority rule: {fragment}"

    assert "A2 conditional fixed 10+10" in current
    assert "A1 and A1.1/4096 are historical/rejected" in current


def test_partial_round_storage_and_observability_rules_are_front_door_visible() -> None:
    """Repeated Stage2 recovery/storage mistakes need startup-visible guards."""

    agents = _normalized(_read("AGENTS.md"))
    runbook = _read("docs/operations/execution/STAGE2_SAFE_CONTINUATION_AND_SHARED_GPU.md")

    for fragment in (
        "one scientific run root has one writer/controller",
        "partial-round reuse preserves producer provenance",
        "container scratch has lifecycle semantics",
        "declared runtime interpreter",
        "a populated directory is not automatically a valid git execution checkout",
    ):
        assert fragment in agents, f"AGENTS.md lost recovery guard: {fragment}"

    for fragment in (
        "Partial round 恢复必须 single-writer + provenance-preserving",
        "host `/dev/shm`",
        "`accepted_rollouts_total`",
        "`parameter_update_count`",
        "measurement pack",
        "`SCIENTIFIC_BLOCKED`",
    ):
        assert fragment in runbook, f"Stage2 runbook lost durable rule: {fragment}"


def test_compatibility_document_paths_cannot_regain_independent_authority() -> None:
    """Compatibility paths must stay tiny pointers to one canonical editable rule set."""

    expectations = {
        "docs/experiment-tracking/FRICTION_REGISTER.md": (
            "../operations/execution/FRICTION_REGISTER.md",
            "docs/operations/execution/FRICTION_REGISTER.md",
        ),
        "docs/experiment-tracking/BRANCH_AND_PR_CONVENTIONS.md": (
            "../infrastructure/repository/BRANCH_AND_PR_CONVENTIONS.md",
            "docs/infrastructure/repository/BRANCH_AND_PR_CONVENTIONS.md",
        ),
    }

    for compatibility_path, (relative_target, canonical_path) in expectations.items():
        compatibility = _read(compatibility_path)
        canonical = _read(canonical_path)
        assert "compatibility pointer" in compatibility.lower()
        assert "not an independent authority" in compatibility.lower()
        assert relative_target in compatibility
        assert len(compatibility.splitlines()) < 20, (
            f"{compatibility_path} became an editable duplicate again"
        )
        assert len(canonical.splitlines()) > 50, f"{canonical_path} lost canonical content"


def test_ci_closeout_authority_is_router_driven_and_exact_head_scoped() -> None:
    """Provider migrations must not leave startup/reliability prose hard-coded to an old gate."""

    agents = _normalized(_read("AGENTS.md"))
    reliability = _normalized(_read("docs/infrastructure/ci/CI_RELIABILITY_STANDARD.md"))
    ci_index = _normalized(_read("docs/infrastructure/ci/README.md"))

    assert "ci authority is routed, not remembered" in agents
    assert "current_cpu_ci_strategy.md" in agents
    assert "live required circleci check" not in reliability
    assert "current_cpu_ci_strategy.md" in reliability
    assert "provider login is therefore a capability" in reliability
    assert "current required exact-head final-candidate gate" in ci_index
    assert "explicit api/pipeline recovery only; no automatic pr/main executor allocation" in ci_index
    assert "current mac required gate" not in ci_index
    assert "mac self-hosted gate is retired from ordinary ci" in ci_index

    gpu_smoke = _normalized(_read(".github/workflows/gpu-smoke.yml"))
    assert "cloudflare pages builds" not in gpu_smoke
    assert "github-hosted base-owned final gate" in gpu_smoke

    branch_conventions = _normalized(_read("docs/infrastructure/repository/BRANCH_AND_PR_CONVENTIONS.md"))
    revision_standard = _normalized(_read("docs/operations/governance/EXPERIMENT_REVISION_STANDARD_202609042007.md"))
    assert "current required automatic cpu checks are provided by circleci" not in branch_conventions
    assert "current_cpu_ci_strategy.md" in branch_conventions
    assert "required circleci cloud cpu gate" not in revision_standard
    assert "current_cpu_ci_strategy.md" in revision_standard


# These are documentation-discovery guards, not a new scientific launch gate.
def test_repeat_correction_is_routed_to_an_actual_use_site_check() -> None:
    agents = _read("AGENTS.md")
    autonomy = _read("docs/operations/governance/GOAL_MODE_AUTONOMY.md")
    assert "correction-to-action-witness" in agents
    assert "REPEAT-CORRECTION" in autonomy
    assert "trigger -> current owner -> checked artifact -> allowed next action -> invalidation cue" in autonomy
    assert "not a new manifest schema" in autonomy


def test_checklist_resume_rule_is_startup_routed() -> None:
    agents = _read("AGENTS.md")
    autonomy = _read("docs/operations/governance/GOAL_MODE_AUTONOMY.md")
    assert "GOAL_MODE_AUTONOMY.md" in agents
    assert "current integration branch's first unchecked item" in autonomy
    assert "Definition of Done fully complete, stop and report completion" in autonomy


def test_consumer_equivalence_and_parallelism_keep_their_scope() -> None:
    agents = _read("AGENTS.md")
    runtime = _read("docs/operations/execution/RUNTIME_PROVISIONING.md")
    assert "Consumer-equivalent qualification" in runtime
    for boundary in ("Shell and entrypoint", "Mounts and permissions", "Forwarded CLI", "Result lifetime"):
        assert boundary in runtime
    assert "not a permanent ban on parallel experiments" in agents
    assert "do not prove identical policy outputs" in agents


def test_analyzer_publication_requires_paired_identity_not_just_file_count() -> None:
    publication = _read("docs/operations/publication/RESULT_PUBLICATION_HANDOFF.md")
    assert "Corpus and analyzer completion are paired identities" in publication
    assert "parse_ok" in publication
    assert "not correctness of every judgment" in publication
    assert "not a seal for a new SEED-style corpus" in publication

def test_gated_delta_d1_closeout_lessons_are_promoted_and_scope_safe() -> None:
    agents = _read("AGENTS.md")
    autonomy = _read("docs/operations/governance/GOAL_MODE_AUTONOMY.md")
    standard = _read("docs/operations/governance/EXPERIMENT_STANDARD.md")
    ci_policy = _read("docs/infrastructure/ci/CI_POLICY.md")
    router = _read("docs/troubleshooting/README.md")
    assert "official sealed percentage" in agents
    assert "continuation receipts need identity-scoped append-only paths" in agents.lower()
    assert "minimal path to the next durable seal" in autonomy
    assert "inherited-contract parity ledger" in standard
    assert "append-only continuation receipt identity" in standard.lower()
    assert "historical execution bytes from current executable authority" in ci_policy
    assert "already-executed bytes" in ci_policy
    assert "CASE-GATED-DELTA-D1-PAIRED-EXECUTION-INTEGRATION-CLOSEOUT-202609141703.md" in router
    case = _read("docs/troubleshooting/experiment-ops/CASE-GATED-DELTA-D1-PAIRED-EXECUTION-INTEGRATION-CLOSEOUT-202609141703.md")
    assert "not execution authority" in case
    assert "dynamic recovery staging is a fresh uid/gid write boundary" in case.lower()
    assert "append-only receipt filenames are part of recovery identity" in case.lower()

def test_agent_handoff_compatibility_readme_remains_symlink() -> None:
    """Path-based API writes must not materialize the compatibility symlink as prose."""

    alias = ROOT / "docs/agent-handoffs/README.md"
    assert alias.is_symlink(), "docs/agent-handoffs/README.md must remain a compatibility symlink"
    assert alias.readlink() == Path("../agents/handoffs/README.md")
    assert (alias.parent / alias.readlink()).resolve() == (ROOT / "docs/agents/handoffs/README.md").resolve()


def test_bounded_gdr_wandb_closeout_lessons_are_promoted() -> None:
    rules = _read("docs/operations/execution/SHARED_ARM_EXECUTION_RULES.md")
    governance = _read("docs/DOCUMENTATION_GOVERNANCE.md")
    friction = _read("docs/operations/execution/FRICTION_REGISTER.md")
    router = _read("docs/troubleshooting/README.md")
    case = _read(
        "docs/troubleshooting/experiment-ops/"
        "CASE-BOUNDED-GDR-WANDB-OBSERVABILITY-CLOSEOUT-202609152235.md"
    )

    for fragment in (
        "Bind each logical offline run, sidecar state, and spool namespace",
        "separately admitted read-only checkout",
        "deterministic bounded samples for expensive quantiles",
        "fresh clean snapshot of sealed receipts",
    ):
        assert fragment in rules
    assert "120000" in governance and "symlink" in governance.lower()
    assert "use-site noncompliance" in friction
    assert "CASE-BOUNDED-GDR-WANDB-OBSERVABILITY-CLOSEOUT-202609152235.md" in router
    assert "Coverage limit" in case
    assert "Temporary state intentionally not promoted" in case


def test_recovery_disclosure_policy_stays_repository_wide_and_single_source() -> None:
    """Disaster recovery and disclosure must not collapse back into one visibility field."""

    policy = _normalized(_read(
        "docs/operations/publication/RESEARCH_ASSET_RECOVERY_AND_DISCLOSURE_POLICY.zh-CN.md"
    ))
    agents = _normalized(_read("AGENTS.md"))
    docs_index = _read("docs/DOCS_INDEX.v1.json")  # legacy-revision-compat
    experiment = _normalized(_read("docs/operations/governance/EXPERIMENT_STANDARD.md"))
    cross_platform = _normalized(_read(
        "docs/operations/governance/CROSS_PLATFORM_EXPERIMENT_ASSET_STANDARD.md"
    ))
    wandb = _normalized(_read("docs/operations/tracking/WANDB_WEAVE_STRATEGY.md"))
    storage = _normalized(_read(
        "docs/infrastructure/server/SERVER_STORAGE_PRESSURE_AND_RECLAIM_SOP.md"
    ))
    recovery_runbook = _normalized(_read(
        "docs/operations/publication/FULL_SERVER_DISASTER_RECOVERY_RUNBOOK.zh-CN.md"
    ))
    server_passport_generator = _normalized(_read(
        "scripts/governance/build_server_run_passports.py"
    ))

    for fragment in (
        "server_only",
        "remote_backed_up",
        "recovery_verified",
        "private",
        "embargoed",
        "public",
        "never_public",
        "provider_visibility",
    ):
        assert fragment in policy, f"canonical recovery/disclosure policy lost: {fragment}"

    assert "recovery/disclosure state" in agents
    assert "research_asset_recovery_disclosure" in docs_index
    assert "scientific identity freeze does **not** automatically publish artifacts" in experiment
    assert "experiment-asset-graph-node-202609201821.schema.json" in cross_platform
    assert "w&b never grants those states" in wandb
    assert "容量压力只改变存储放置优先级，不改变 disclosure" in storage
    assert "公开不是灾备 fallback" in recovery_runbook
    assert "gate_contract" in policy
    assert "gate_instance" in policy
    assert "durable private placement gate" in recovery_runbook
    assert "top-level provider listing is not proof of bundle-member absence" in recovery_runbook
    assert "'recovery_state':'server_only'" in server_passport_generator
    assert "'disclosure_state':'private'" in server_passport_generator

def test_hf_publication_transport_closeout_lessons_are_promoted() -> None:
    playbook = _normalized(_read(
        "docs/operations/publication/SERVER_ARTIFACT_PUBLICATION_PLAYBOOK.zh-CN.md"
    ))
    case = _normalized(_read(
        "docs/troubleshooting/publication/CASE-HUGGINGFACE-ASSET-GOVERNANCE-20260829.md"
    ))

    assert "控制面与数据面分离" in playbook
    assert "hf upload-large-folder --num-workers n" in playbook
    assert "同一 hf repo 的同一 target branch/path namespace" in playbook
    assert "不要为了省掉 mac 中转而把 workstation 的 token 复制到服务器" in playbook
    assert "remote head 在本 publisher 之外变化" in playbook

    assert "personal namespace 的 cli 404" in case
    assert "not_verified_api_timeout" in case
    assert "content_equivalent_by_member_identity / container_bytes_differ" in case
    assert "member path/size/sha" in case
    assert "临时状态故意不保存" in case

