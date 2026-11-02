from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_root_agents_is_unique_agent_bootstrap_authority() -> None:
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "unique repository-root Agent bootstrap authority" in agents
    assert "docs/agents/README.md" in agents


def test_docs_agents_readme_is_navigation_only() -> None:
    readme = (ROOT / "docs/agents/README.md").read_text(encoding="utf-8")
    assert "Authority: **navigation only**" in readme
    assert "../../AGENTS.md" in readme

    forbidden_sections = (
        "## Owner interaction conventions",
        "## Scientific artifact preservation preference",
        "### Repeated-correction and memory boundary",
        "### Portable memory candidates",
    )
    for section in forbidden_sections:
        assert section not in readme


def test_documentation_governance_names_agent_entry_hierarchy() -> None:
    governance = (ROOT / "docs/DOCUMENTATION_GOVERNANCE.md").read_text(encoding="utf-8")
    assert "repository-root Agent bootstrap authority" in governance
    assert "`docs/agents/README.md`" in governance
    assert "navigation-only" in governance


def test_root_agents_requires_full_consumer_parity() -> None:
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "Enumerate every downstream consumer" in agents
    assert "Fixing only the first rejecting consumer is incomplete" in agents

def test_scoped_workline_resolves_before_repository_default_campaign() -> None:
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "declared workline's actual control/execution checkout first" in agents
    assert "do **not** substitute repository-main's default campaign" in agents
    assert "default router is a fallback only when no narrower current execution owner exists" in agents


def test_passport_plan_separates_lifecycle_from_active_compute() -> None:
    plan = (
        ROOT
        / "docs/experiment-tracking/EXPERIMENT_PASSPORT_REGISTRY_IMPLEMENTATION_PLAN_202609111800.md"
    ).read_text(encoding="utf-8")
    assert "lifecycle_status" in plan
    assert "activity_status" in plan
    assert "IDLE_AT_BOUNDARY" in plan
    assert "禁止把 `lifecycle_status=RUNNING` 渲染成“GPU 正在训练”" in plan
    assert "next_round=N" in plan


def test_scientific_blocker_lifecycle_is_typed_and_routed() -> None:
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    rules = (
        ROOT / "docs/operations/execution/SHARED_ARM_EXECUTION_RULES.md"
    ).read_text(encoding="utf-8")

    assert "Scientific blocker state is typed, not a generic PASS" in agents
    assert "#scientific-blocker-lifecycle-and-successor-routing" in agents
    assert "`RESOLVED`" in rules
    assert "`SUPERSEDED/REMOVED`" in rules
    assert "`BLOCKED`" in rules
    assert "this is **not** a PASS of the old requirement" in rules
    assert "identity resolution" in rules
    assert "new-arm materialization" in rules
    assert "never relabel it as GitHub CI" in rules


def test_parallel_integration_searches_semantic_overlap_before_writing() -> None:
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    case = (ROOT / "docs/troubleshooting/experiment-ops/CASE-GDR-160-LAUNCH-CARRIER-HEALTH-ACCELERATION-CONVERGENCE-202609141740.md").read_text(encoding="utf-8")
    assert "Semantic-overlap scan precedes a new integration path" in agents
    assert "current integration code/configs/current routers first" in agents
    assert "then live open PRs, branches, and relevant worktrees by semantic target/outcome" in agents
    assert "do not create a second mutable integration or resource authority" in agents
    assert "Parallel-lineage reconciliation must happen before implementation" in case
    assert "one mutable integration path" in case

def test_concurrent_overlap_and_passport_coverage_checks_are_use_site_visible() -> None:
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    dag = (ROOT / "docs/operations/governance/EXPERIMENT_PASSPORT_ARTIFACT_DAG.md").read_text(encoding="utf-8")
    ci = (ROOT / "docs/infrastructure/ci/CI_RELIABILITY_STANDARD.md").read_text(encoding="utf-8")
    standard = (ROOT / "docs/operations/governance/EXPERIMENT_STANDARD.md").read_text(encoding="utf-8")
    assert "another Agent/window is concurrently working" in agents
    assert "before local implementation" in agents
    assert "again immediately before the first branch/PR/provider write" in agents
    assert "Mechanism readiness is not production-coverage readiness" in dag
    assert "current-experiment-registry.json" in dag
    assert "current-passport-artifact-index.json" in dag
    assert "mechanism ready / production provenance not yet admitted" in dag
    assert "git ls-files --stage -- <path>" in ci
    assert "Only a clean committed candidate can become exact-head evidence" in ci
    assert "Recovery coverage is not identity/index coverage" in standard
    assert "RECOVERY_VERIFIED" in standard
    assert "Server Run Passport control-plane index" in standard
    assert "canonical_run_id" in standard
    assert "Never rewrite a FROZEN" in standard
    assert "invent a `canonical_run_id`" in standard


def test_workflow_resume_rechecks_identity_and_topology_scope() -> None:
    plan = (
        ROOT / "docs/experiment-tracking/OPEN_EVO_CEILING_1.0_END_TO_END_MASTER_PLAN.md"
    ).read_text(encoding="utf-8")
    assert "不是 operation/scientific identity authority" in plan
    assert "resource-topology identity" in plan
    assert "不能自行换 topology/fallback" in plan
    assert "SHARED_ARM_EXECUTION_RULES.md#advisory-deduplication-is-evidence-not-reuse-authority" in plan
    assert "不得因 preferred attempt 失败而自行把 fallback 升格" in plan


def test_budgeted_acceleration_and_layered_replay_rules_are_durable() -> None:
    standard = (ROOT / "docs/operations/governance/EXPERIMENT_STANDARD.md").read_text(encoding="utf-8")
    rules = (ROOT / "docs/operations/execution/SHARED_ARM_EXECUTION_RULES.md").read_text(encoding="utf-8")
    case = (ROOT / "docs/troubleshooting/experiment-ops/CASE-GDR-160-LAUNCH-CARRIER-HEALTH-ACCELERATION-CONVERGENCE-202609141740.md").read_text(encoding="utf-8")
    standard_flat = " ".join(standard.split())
    rules_flat = " ".join(rules.split())
    case_flat = " ".join(case.split())
    assert "selection criterion before GPU spend" in standard_flat
    assert "Deferring an unselected lineage is not evidence" in standard_flat
    assert "Cross-line engineering ideas may be borrowed only when" in standard_flat
    assert "Acceleration promotion is a **two-axis gate**" in standard_flat
    assert "a microbenchmark or helper-level PASS does not promote a full successor" in standard_flat
    assert "Historical replay exactness is layered" in rules_flat
    assert "trainer-input exact; outer-policy provenance unresolved" in rules_flat
    assert "do **not** substitute the nearest/current file" in rules_flat
    assert "Budget-constrained acceleration selection follow-up" in case_flat
    assert "parser rejection as `NOT_EXECUTED`" in case_flat

def test_advisory_cache_reuse_stays_separate_from_scientific_authority_and_savings() -> None:
    rules = (ROOT / "docs/operations/execution/SHARED_ARM_EXECUTION_RULES.md").read_text(encoding="utf-8")
    assert "Advisory deduplication is evidence, not reuse authority" in rules
    assert "Count **work actually avoided**, not candidates discovered" in rules
    assert "Resource qualification reuse is topology-specific" in rules
    assert "scientific disposition are separate" in rules


def test_successor_absorption_revalidates_current_naming_before_hash_generation() -> None:
    policy = (ROOT / "docs/operations/governance/REVISION_NAMING_POLICY_202609041833.md").read_text(encoding="utf-8")
    assert "Successor absorption uses the current integration policy" in policy
    assert "before generating content-addressed indexes" in policy
    assert "regenerate operation digests, node hashes, indexes, and routers from canonical bytes" in policy


def test_aggregated_validation_cannot_hide_a_failed_subcheck() -> None:
    standard = (ROOT / "docs/infrastructure/ci/CI_RELIABILITY_STANDARD.md").read_text(encoding="utf-8")
    assert "Aggregated validation must fail closed" in standard
    assert "A later successful test must not erase an earlier failed preflight" in standard
    assert "aggregate PASS marker only after every required subcommand" in standard


def test_root_agents_keeps_explicit_outer_shell_rule_visible() -> None:
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "execution tool itself has a shell selector" in agents
    assert "set that outer shell explicitly" in agents


def test_repository_current_uses_live_remote_ref_not_tracking_cache() -> None:
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    closeout = (ROOT / "docs/operations/governance/CONVERSATION_LESSONS_CLOSEOUT.md").read_text(encoding="utf-8")
    assert "Remote-tracking refs are caches, not live authority" in agents
    assert "A local remote-tracking ref is a cache, not proof of live provider state" in closeout
    assert "git ls-remote" in closeout


def test_closeout_branch_setup_applies_shell_rule_before_first_write() -> None:
    closeout = (ROOT / "docs/operations/governance/CONVERSATION_LESSONS_CLOSEOUT.md").read_text(encoding="utf-8")
    assert "select the outer shell explicitly before the first branch/worktree command" in closeout
    assert "parser/quoting failure" in closeout
    assert "NOT_EXECUTED" in closeout


def test_run_manifest_identity_namespaces_are_type_specific() -> None:
    contract = (ROOT / "docs/operations/governance/RUN_MANIFEST_CONTRACT.md").read_text(encoding="utf-8")
    assert "Identity namespaces are type-specific" in contract
    assert "Git commit identity" in contract
    assert "40 lowercase hex" in contract
    assert "bare SHA-256" in contract
    assert "sha256:<64 lowercase hex>" in contract


def test_execution_runtime_closeout_case_is_routed() -> None:
    router = (ROOT / "docs/troubleshooting/README.md").read_text(encoding="utf-8")
    case = ROOT / "docs/troubleshooting/experiment-ops/CASE-EXECUTION-RUNTIME-MODERNIZATION-CLOSEOUT-20260917.md"
    assert case.is_file()
    assert case.name in router



def test_task_stream_denominators_and_exposure_are_separate_contracts() -> None:
    standard = (ROOT / "docs/operations/governance/EXPERIMENT_STANDARD.md").read_text(encoding="utf-8")
    readme = (ROOT / "docs/science/webshop/README.md").read_text(encoding="utf-8")
    case = (ROOT / "docs/troubleshooting/experiment-ops/CASE-GDR-160-LAUNCH-CARRIER-HEALTH-ACCELERATION-CONVERGENCE-202609141740.md").read_text(encoding="utf-8")

    assert "task-stream-denominator-and-exposure-contract" in standard
    assert "A formal task denominator is not automatically a unique-task denominator" in standard
    assert "Historical exposure to a training-eligible task is provenance" in standard
    assert "never silently coerce unknown to zero" in standard
    standard_flat = " ".join(standard.split())
    assert "do not call a historically used protected panel a new \"untouched\" evaluation" in standard_flat
    assert "#task-stream-denominator-and-exposure-contract" in readme
    assert "Supersession note — canonical-stream correction" in case
    assert "**superseded/removed**" in case


def test_eli5_hinge_and_representation_symmetry_rules_are_startup_visible() -> None:
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    standard = (ROOT / "docs/operations/governance/EXPERIMENT_STANDARD.md").read_text(encoding="utf-8")
    agents_flat = " ".join(agents.split())
    standard_flat = " ".join(standard.split())
    assert "spend the ELI5 explanation on that causal/scientific hinge first" in agents_flat
    assert "A coordinate-level bound is not automatically the original effective-state gate bound" in agents_flat
    assert "Treatment onset and representation contract" in standard_flat
    assert "empty prior/history is a state condition, not automatic permission to delay the intervention" in standard_flat
    assert "realized intervention state" in standard_flat


def test_declared_workline_scope_latches_and_server_maintenance_case_is_routed() -> None:
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    friction = (ROOT / "docs/operations/execution/FRICTION_REGISTER.md").read_text(encoding="utf-8")
    router = (ROOT / "docs/troubleshooting/README.md").read_text(encoding="utf-8")
    case = (
        ROOT
        / "docs/troubleshooting/publication/CASE-SERVER-MAINTENANCE-PASSPORT-AUTHORITY-CLOSEOUT-202609160018.md"
    ).read_text(encoding="utf-8")

    assert "Declared workline scope is latched" in agents
    assert "treat adjacent dependencies as handoffs" in agents
    assert "Declared workline scope is a latch, not a suggestion" in friction
    assert "CASE-SERVER-MAINTENANCE-PASSPORT-AUTHORITY-CLOSEOUT-202609160018.md" in router
    assert "bundle proof != member proof" in case
    assert "Capacity pressure cannot authorize public visibility" in case


def test_hf_quota_pressure_keeps_storage_visibility_and_completion_separate() -> None:
    policy = (ROOT / "docs/operations/publication/RESEARCH_ASSET_RECOVERY_AND_DISCLOSURE_POLICY.zh-CN.md").read_text(encoding="utf-8")
    standard = (ROOT / "docs/operations/publication/ARTIFACT_ARCHIVE_AND_PUBLICATION_STANDARD.md").read_text(encoding="utf-8")
    case = (ROOT / "docs/troubleshooting/publication/CASE-HUGGINGFACE-ASSET-GOVERNANCE-20260829.md").read_text(encoding="utf-8")
    policy_flat = " ".join(policy.split())
    standard_flat = " ".join(standard.split())
    case_flat = " ".join(case.split())
    assert "SERVER_ONLY" in policy_flat
    assert "REMOTE_BACKED_UP" in policy_flat
    assert "RECOVERY_VERIFIED" in policy_flat
    assert "SERVER_ONLY_UNVERIFIED" in policy_flat
    assert "它不是第四个 Recovery 状态" in policy_flat
    assert "PRIVATE" in policy_flat and "EMBARGOED" in policy_flat
    assert "PUBLIC" in policy_flat and "NEVER_PUBLIC" in policy_flat
    assert "以后可能公开" in policy_flat
    assert "private quota / 磁盘压力是**存储放置问题**" in policy_flat
    assert "Quota pressure is a **storage-placement signal, not visibility or deletion authority**" in standard_flat
    assert "public model-state archive can be complete while private raw-evidence disaster recovery is still partial" in standard_flat
    assert "Repository-tree bytes are composition evidence, not provider quota accounting" in standard_flat
    assert "A Draft publication PR is neither positive nor negative visibility authority" in standard_flat
    assert "PUBLIC / PRIVATE_ONLY / POINTER / RETIRED" in case_flat
    assert "storage/publication roles" in case_flat
    assert "Draft PR 不能授予 visibility；但 Draft PR 也不能证明 visibility authority 不存在" in case_flat


def test_matched_estimator_does_not_imply_runtime_lockstep() -> None:
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    rules = (ROOT / "docs/operations/execution/SHARED_ARM_EXECUTION_RULES.md").read_text(encoding="utf-8")
    assert "does **not** by itself authorize runtime lockstep" in agents
    assert "Matched analysis is not an implicit runtime barrier" in rules
    assert "either arm may be ahead in wall-clock rounds" in rules
    assert "preserve those barriers for that lineage" in rules


def test_carrier_content_rejection_is_not_automatically_integrity_failure() -> None:
    standard = (ROOT / "docs/operations/governance/EXPERIMENT_STANDARD.md").read_text(encoding="utf-8")
    flat = " ".join(standard.split())
    assert "A **content rejection**" in flat
    assert "engineering or integrity failure" in flat
    assert "keep the prior legal carrier" in flat
    assert "continue the main trajectory" in flat
    assert "do not retroactively reinterpret a" in flat

def test_concurrent_agents_isolate_control_sessions_and_parser_failures_latch() -> None:
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    agents_flat = " ".join(agents.split())
    assert "Concurrent Agents require isolated control sessions as well as isolated worktrees" in agents_flat
    assert "same interactive SSH/terminal/REPL session" in agents_flat
    assert "NOT_EXECUTED_OR_UNKNOWN" in agents_flat
    assert "After any nested-shell/quoting/parser failure" in agents_flat
    assert "keep the rest of that task in file/tool mode" in agents_flat


def test_bounded_ray_live_handoff_closeout_rules_are_startup_visible() -> None:
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    rules = (ROOT / "docs/operations/execution/SHARED_ARM_EXECUTION_RULES.md").read_text(encoding="utf-8")
    case = (ROOT / "docs/troubleshooting/experiment-ops/CASE-BOUNDED-RAY-CAPACITY-LIVE-HANDOFF-CLOSEOUT-20260916.md").read_text(encoding="utf-8")
    assert "Do not move the finish line with future hardening" in agents
    assert "current integration code/configs/current routers first" in agents
    assert "Bind the exact successor identity at the activation boundary" in agents
    assert "serves as provenance rather than a conversational permission token" in agents
    assert "Discovery is read-only" in agents
    assert "CURRENT_GOAL_BLOCKER" in rules
    assert "A successor branch name is not an execution generation" in rules
    assert "typed machine dispositions" in rules
    assert "one live-control owner per lineage" in case
    assert "Temporary state intentionally not promoted" in case
    assert "Attribute live resource authority by bound identity, not process basename" in agents
    assert "Keep verifier identity separate from execution identity" in agents
    assert "Process names are not live-authority identity" in rules
    assert "Verifier identity is separate from resource execution identity" in rules
    assert "Pre-adoption and post-adoption gates are different states" in rules
    assert "A process basename does not tell you whether it is the old or new scheduler" in case
    assert "The verifier SHA is not the formal execution SHA" in case
    assert "Post-adoption verification is not a second pre-adoption migration gate" in case


def test_partial_round_control_rebind_receipt_survives_post_rollout() -> None:
    rules = (ROOT / "docs/operations/execution/SHARED_ARM_EXECUTION_RULES.md").read_text(encoding="utf-8")
    case = (ROOT / "docs/troubleshooting/experiment-ops/CASE-BOUNDED-GDR-RECOVERY-CONTINUATION-CLOSEOUT-20260916.md").read_text(encoding="utf-8")
    rules_flat = " ".join(rules.split())
    case_flat = " ".join(case.split())
    assert "Control-only continuation receipts are phase-stable after creation." in rules_flat
    assert "Automatic recovery closure requires a semantic handoff witness." in rules_flat
    assert "later rollout/model side effects do not invalidate it" in rules_flat
    assert "post-rollout re-entry" in case_flat
    assert "same live control plane to create/dispatch the next admissible workload" in case_flat


def test_resume_rechecks_live_authority_and_parser_rule_is_early() -> None:
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    rules = (ROOT / "docs/operations/execution/SHARED_ARM_EXECUTION_RULES.md").read_text(encoding="utf-8")
    case = (ROOT / "docs/troubleshooting/experiment-ops/CASE-BOUNDED-GDR-RECOVERY-CONTINUATION-CLOSEOUT-20260916.md").read_text(encoding="utf-8")
    agents_flat = " ".join(agents.split())
    rules_flat = " ".join(rules.split())
    case_flat = " ".join(case.split())
    assert "Owner `resume`/`continue` authorization is goal-level authority" in agents_flat
    assert "Mutation-boundary authorization is re-resolved, not cached." in rules_flat
    assert "audit/adopt it and record the requested mutation as a no-op/not-executed" in rules_flat
    assert "owner intent and execution identity are separate facts" in case_flat
    assert agents.index("For commands that clearly cross multiple parser layers") < agents.index("## Agent entry hierarchy")

def test_closeout_protocol_uses_declared_validation_runtime() -> None:
    protocol = (ROOT / "docs/operations/governance/CONVERSATION_LESSONS_CLOSEOUT.md").read_text(encoding="utf-8")
    source = Path(__file__).read_text(encoding="utf-8")
    assert "Use the declared validation runtime" in protocol
    assert "Do not default to system Python plus `pytest`" in protocol
    assert 'if __name__ == "__main__":' in source


def test_wandb_native_workspace_and_same_conversation_capability_rules_are_durable() -> None:
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    strategy = (ROOT / "docs/operations/tracking/WANDB_WEAVE_STRATEGY.md").read_text(encoding="utf-8")
    router = (ROOT / "docs/troubleshooting/README.md").read_text(encoding="utf-8")
    assert "Same-conversation capability claims require a fresh probe" in agents
    assert "last-known-good authorized route" in agents
    assert "Workspace visualization defaults" in strategy
    assert "native W&B `LinePlot` panels" in strategy
    assert "do not interpolate or fabricate intermediate points" in strategy
    assert "prefer muted/low-saturation presentation colors" in strategy
    assert "CASE-WANDB-NATIVE-WORKSPACE-CAPABILITY-ROUTING-CLOSEOUT-20260919.md" in router


if __name__ == "__main__":
    tests = sorted((name, obj) for name, obj in globals().items() if name.startswith("test_") and callable(obj))
    for name, test in tests:
        test()
        print(f"PASS {name}")
    print(f"DOCUMENTATION_AUTHORITY_ASSERTIONS={len(tests)}/{len(tests)} PASS")
