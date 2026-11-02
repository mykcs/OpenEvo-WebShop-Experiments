#!/usr/bin/env python3
from __future__ import annotations
"""Scientific arm state machine for the formal Bounded OFF / Effective-State ON study.

Runtime safety lives in bounded_effective_state_runtime_safety_202609282338.py;
matched-pair orchestration lives in bounded_effective_state_pair_runtime_202609152300.py.
Keep this file focused on treatment/state transitions and round sealing.
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scripts.openevo_webshop import ceiling1_stage2_arm_controller as base
from scripts.openevo_webshop import bounded_effective_state_recovery_202609282338 as recovery
from scripts.openevo_webshop import bounded_effective_state_ray_formal_adapter_202609282338 as ray_capacity
from scripts.openevo_webshop.bounded_effective_state_launch_guard_202609282338 import validate_launch_pair, validate_post_advisor_activation, sha as guard_sha
from scripts.openevo_webshop.bounded_effective_state_runtime_safety_202609282338 import (
    GPU_UUID, SCIENCE_IMAGE, SCIENCE_IMAGE_ID, CARRIER_NOOP_SCHEMA, CARRIER_NOOP_REASON,
    RuntimeSafetyMixin, exclusive_arm_lock, validate_carrier_noop_amendment, exact_image_gate, fresh_gpu_gate,
)

OFF_MODE = "BOUNDED_OFF"
ON_MODE = "EFFECTIVE_STATE_GDR_LORA_V1"  # legacy-revision-compat: exact historical frozen identifier

def load_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value):
    base.write_once_json(path, value)



class BoundedEffectiveStateController(RuntimeSafetyMixin, base.Controller):
    def __init__(self, args: argparse.Namespace):
        self.rank32_short_fork = bool(getattr(args, "rank32_short_fork_authority", None))
        self.rank32_short_fork_authority_path = args.rank32_short_fork_authority.resolve() if self.rank32_short_fork else None
        self.experiment_arm_id = args.experiment_arm_id
        self.treatment_mode = args.treatment_mode
        self.launch_package = args._launch_package
        self.ray_capacity_policy = ray_capacity.load_policy(args.repo.resolve(), self.launch_package)
        self._prevalidated_resource_gates = {}
        self.launch_release = args._launch_release
        self.ready_package_path = args.ready_package.resolve()
        self.owner_launch_release_path = args.owner_launch_release.resolve()
        self.shared_rollout_gate_path = args.shared_rollout_gate.resolve() if args.shared_rollout_gate else None
        out = args.output.resolve()
        exec_state = out / "execution-state.json"
        if not exec_state.is_file():
            partial = []
            for rr in sorted(out.glob("round-*")) if out.is_dir() else []:
                if any((rr / name).exists() for name in ("state-before.json", "attempts.jsonl", "summary.json", "state-after.json", "ROUND_COMPLETE.json")):
                    partial.append(str(rr))
            if partial:
                raise RuntimeError("formal round artifacts exist without execution-state; refuse implicit Round0 reinitialization")
        self.carrier_noop_amendment_path = args.carrier_contract_fail_noop_amendment.resolve()
        self.carrier_noop_amendment_sha256 = validate_carrier_noop_amendment(self.carrier_noop_amendment_path, self.launch_package, args.repo.resolve())
        self.formal_transition_launcher = args.formal_transition_launcher.resolve()
        self.empty_history_manifest = args.empty_history_manifest.resolve()
        self.adapter_config_template = args.adapter_config_template.resolve()
        self.gdr_runtime_path = args.gdr_runtime_path.resolve() if args.gdr_runtime_path else None
        self.beta_policy_path = args.beta_policy_path.resolve() if args.beta_policy_path else None
        exact_image_gate()
        carrier_inputs = {
            "scope": args.carrier_scope_package.resolve(),
            "health": args.carrier_health_policy.resolve(),
            "identity": args.carrier_contract_identities.resolve(),
            "adoption": args.carrier_adoption_receipt.resolve(),
        }
        self._inherited_carrier_inputs = carrier_inputs
        # The inherited #480 carrier receipt has a dedicated Bounded adoption schema,
        # so never send it through the base generic Carrier-v2 adoption validator.  # legacy-revision-compat: exact historical frozen identifier
        args.carrier_scope_package = None
        args.carrier_health_policy = None
        args.carrier_contract_identities = None
        args.carrier_adoption_receipt = None
        super().__init__(args)
        self.install_inherited_carrier_v2(carrier_inputs)  # legacy-revision-compat: exact historical frozen identifier
        step_ref=self.launch_package.get("step_seed_formal_binding") or {}
        step_binding_path=self.wt/str(step_ref.get("path") or "")
        if not step_binding_path.is_file() or step_ref.get("sha256")!=guard_sha(step_binding_path): raise RuntimeError("step-seed formal binding missing or drifted")
        step_binding=load_json(step_binding_path); asset=step_binding.get("asset") or {}; expected_step=(self.wt/str(asset.get("path") or "")).resolve()
        if self.step_seed_amendment_path.resolve()!=expected_step or self.step_seed_amendment_sha256!=asset.get("sha256") or guard_sha(expected_step)!=asset.get("sha256"):
            raise RuntimeError("formal step-seed runtime does not match Qwen3 prospective binding")
        self.step_seed_formal_binding_sha256=guard_sha(step_binding_path)
        self.carrier_contract_fail_noop_authorized = True
        base.bind_preformal_identity(self.state, "carrier_contract_fail_noop_amendment_sha256", self.carrier_noop_amendment_sha256, "Carrier contract-fail NOOP amendment")
        base.bind_preformal_identity(self.state, "step_seed_formal_binding_sha256", self.step_seed_formal_binding_sha256, "Qwen3 injective step-seed formal binding")
        base.bind_preformal_identity(self.state, "formal_ready_package_sha256", guard_sha(self.ready_package_path), "formal READY package")
        base.bind_preformal_identity(self.state, "owner_launch_release_sha256", guard_sha(self.owner_launch_release_path), "owner launch release")
        if self.rank32_short_fork:
            base.bind_preformal_identity(self.state, "rank32_short_fork_authority_sha256", guard_sha(self.rank32_short_fork_authority_path), "post-advisor rank32 short-fork authority")
        self.save()

    def install_inherited_carrier_v2(self, carrier_inputs: dict) -> None:  # legacy-revision-compat: exact historical frozen identifier
        schedule = self.wt / "configs/experiment/manifests/h1.44-on-3b-training-schedule-v1.json"  # legacy-revision-compat: exact historical frozen identifier
        self.carrier_scope_package_path = carrier_inputs["scope"]
        self.carrier_scope_package = base.load_carrier_scope_package(self.carrier_scope_package_path, expected_schedule_sha256=base.sha(schedule), expected_round_count=160)
        self.carrier_scope_package_identity = self.carrier_scope_package["package_identity_sha256"]
        self.carrier_health_policy_path = carrier_inputs["health"]
        self.carrier_health_policy = base.load_carrier_health_policy(self.carrier_health_policy_path)
        self.carrier_health_policy_identity = base.carrier_health_policy_identity(self.carrier_health_policy)
        self.carrier_contract_identities_path = carrier_inputs["identity"]
        self.carrier_contract_identities = base.load_carrier_contract_identity(self.carrier_contract_identities_path, self.wt)
        self.carrier_contract_identities_sha256 = base.sha(self.carrier_contract_identities_path)
        self.carrier_contract_identity = self.carrier_contract_identities["carrier_contract_identity_sha256"]
        self.carrier_health_component_identities = self.carrier_contract_identities["health_component_identities"]
        receipt = load_json(carrier_inputs["adoption"])
        self.validate_inherited_adoption(receipt)
        self.carrier_adoption_receipt_path = carrier_inputs["adoption"]
        self.carrier_adoption_receipt = receipt
        self.carrier_adoption_identity = receipt["adoption_identity_sha256"]
        base.bind_carrier_scope_package_identity(self.state, self.carrier_scope_package_identity)
        base.bind_preformal_identity(self.state, "carrier_health_policy_identity_sha256", self.carrier_health_policy_identity, "Carrier Health policy")
        base.bind_preformal_identity(self.state, "carrier_contract_identities_sha256", self.carrier_contract_identities_sha256, "Carrier Contract identity manifest")
        base.bind_preformal_identity(self.state, "carrier_contract_identity_sha256", self.carrier_contract_identity, "Carrier Contract content identity")
        base.bind_preformal_identity(self.state, "carrier_adoption_identity_sha256", self.carrier_adoption_identity, "inherited Carrier v2 adoption")  # legacy-revision-compat: exact historical frozen identifier
        self.save()

    def validate_inherited_adoption(self, receipt: dict) -> None:
        if receipt.get("schema") != "openevo.bounded-gdr.inherited-carrier-v2-adoption.202609151800" or receipt.get("status") != "PASS_INHERITED_COMMON_START_PROSPECTIVE_V2":  # legacy-revision-compat: exact historical frozen identifier
            raise RuntimeError("inherited Carrier v2 adoption receipt invalid")  # legacy-revision-compat: exact historical frozen identifier
        if receipt.get("execution_git_sha") != ("a6f67b66e7d0fce0dc445890a68f9f0364da452a" if self.rank32_short_fork else self.execution_git_sha):
            raise RuntimeError("inherited Carrier adoption execution SHA drift")
        if receipt.get("carrier_contract_identity_sha256") != self.carrier_contract_identity or receipt.get("scope_package_identity_sha256") != self.carrier_scope_package_identity or receipt.get("health_policy_identity_sha256") != self.carrier_health_policy_identity:
            raise RuntimeError("inherited Carrier v2 mechanism identity drift")  # legacy-revision-compat: exact historical frozen identifier
        if receipt.get("initial_state_origin") != "FROZEN_PREEXISTING_COMMON_START" or receipt.get("generated_under_carrier_v2") is not False or receipt.get("prospective_updates_from_round0") != "CARRIER_CONTRACT_V2_REQUIRED":  # legacy-revision-compat: exact historical frozen identifier
            raise RuntimeError("inherited Carrier start semantics drift")
        expected = receipt.get("initial_carrier_sha256") or {}
        observed = {"text_memory": base.sha(Path(self.harness201_carrier_seed["memory"])), "skill_bundle": base.sha(Path(self.harness201_carrier_seed["skill"])), "agent_system": base.sha(Path(self.harness201_carrier_seed["agent"]))}
        if observed != expected:
            raise RuntimeError("inherited #480 carrier bytes drift")
        if int(receipt.get("final_panel_access", -1)) != 0:
            raise RuntimeError("inherited Carrier adoption final-panel boundary drift")

    def load_or_init(self):
        # The #480 common-start carriers predate Carrier-v2. On a fresh formal root,  # legacy-revision-compat: exact historical frozen identifier
        # accept only their exact frozen hashes; do not pretend they were authored
        # under the later Carrier-v2 contract. New/repair carriers from Round0 onward  # legacy-revision-compat: exact historical frozen identifier
        # still use the full prospective Carrier-v2 machinery installed after init.  # legacy-revision-compat: exact historical frozen identifier
        if self.exec_path.is_file():
            # Base resume binds Carrier identities before the subclass installs the
            # inherited Carrier contract. Prime only the identity fields here; the
            # dedicated inherited-adoption validator still runs after base init.
            ci=self._inherited_carrier_inputs
            schedule=self.wt / "configs/experiment/manifests/h1.44-on-3b-training-schedule-v1.json"  # legacy-revision-compat: exact historical frozen identifier
            self.carrier_scope_package_path=ci["scope"]
            self.carrier_scope_package=base.load_carrier_scope_package(self.carrier_scope_package_path, expected_schedule_sha256=base.sha(schedule), expected_round_count=160)
            self.carrier_scope_package_identity=self.carrier_scope_package["package_identity_sha256"]
            self.carrier_health_policy_path=ci["health"]
            self.carrier_health_policy=base.load_carrier_health_policy(self.carrier_health_policy_path)
            self.carrier_health_policy_identity=base.carrier_health_policy_identity(self.carrier_health_policy)
            self.carrier_contract_identities_path=ci["identity"]
            self.carrier_contract_identities=base.load_carrier_contract_identity(self.carrier_contract_identities_path,self.wt)
            self.carrier_contract_identities_sha256=base.sha(self.carrier_contract_identities_path)
            self.carrier_contract_identity=self.carrier_contract_identities["carrier_contract_identity_sha256"]
            self.carrier_health_component_identities=self.carrier_contract_identities["health_component_identities"]
            receipt=load_json(ci["adoption"])
            self.carrier_adoption_receipt_path=ci["adoption"]
            self.carrier_adoption_receipt=receipt
            self.carrier_adoption_identity=receipt["adoption_identity_sha256"]
            state = super().load_or_init()
        else:
            stage1 = base.load(Path(self.entry_authority["stage1_complete_path"]))
            merged = self.root / "stage2-base" / self.arm / "verification-receipt.json"
            if not self.harness201_successor:
                raise RuntimeError("Bounded formal inherited start requires Harness201 successor mode")
            mem = Path(self.harness201_carrier_seed["memory"])
            skill = Path(self.harness201_carrier_seed["skill"])
            agent = Path(self.harness201_carrier_seed["agent"])
            expected_seed = {
                "text_memory": self.harness201_carrier_seed["memory_sha256"],
                "skill_bundle": self.harness201_carrier_seed["skill_sha256"],
                "agent_system": self.harness201_carrier_seed["agent_sha256"],
            }
            adoption = load_json(self._inherited_carrier_inputs["adoption"])
            if adoption.get("schema") != "openevo.bounded-gdr.inherited-carrier-v2-adoption.202609151800" or adoption.get("status") != "PASS_INHERITED_COMMON_START_PROSPECTIVE_V2":  # legacy-revision-compat: exact historical frozen identifier
                raise RuntimeError("inherited Carrier v2 adoption receipt invalid at initial state")  # legacy-revision-compat: exact historical frozen identifier
            if adoption.get("execution_git_sha") != self.execution_git_sha:
                raise RuntimeError("inherited Carrier adoption execution SHA drift at initial state")
            if adoption.get("initial_state_origin") != "FROZEN_PREEXISTING_COMMON_START" or adoption.get("generated_under_carrier_v2") is not False or adoption.get("prospective_updates_from_round0") != "CARRIER_CONTRACT_V2_REQUIRED":  # legacy-revision-compat: exact historical frozen identifier
                raise RuntimeError("inherited Carrier initial-state semantics drift")
            if int(adoption.get("final_panel_access", -1)) != 0:
                raise RuntimeError("inherited Carrier initial-state final-panel boundary drift")
            expected_adoption = adoption.get("initial_carrier_sha256") or {}
            actual = {"text_memory": base.sha(mem), "skill_bundle": base.sha(skill), "agent_system": base.sha(agent)}
            if actual != expected_seed or actual != expected_adoption:
                raise RuntimeError("inherited #480 carrier byte identity drift")
            merged_receipt = base.load(merged)
            if merged_receipt.get("status") != "PASS" or merged_receipt.get("opsd_adapter_sha256") != stage1["opsd"]["adapter_sha256"] or merged_receipt.get("stage2_config_sha256") != self.config_sha:
                raise RuntimeError("post-OPSD Stage2 base identity drift")
            sid = base.state_id(merged, None, mem, skill, agent, -1)
            state = {
                "schema": "openevo.ceiling1-stage2-execution-state.v1", "status": "RUNNING", "arm": self.arm,  # legacy-revision-compat: exact historical frozen identifier
                "config_sha256": self.config_sha, "stage2_activation_sha256": self.entry_authority["activation_sha256"],
                "next_round": 0, "rollouts_consumed": 0, "state_id": sid, "merged_base_receipt": str(merged),
                "current_sd_adapter": None, "current_sd_artifact_id": None, "sd_component_count": 0, "sd_generation_count": 0,
                "memory": str(mem), "skill": str(skill), "agent_system": str(agent), "evidence_history": None,
                "rounds_complete": [], "external_teacher_calls": 0, "final_panel_access_count": 0,
                "initial_carrier_origin": "FROZEN_PREEXISTING_COMMON_START",
                "initial_carrier_generated_under_v2": False,  # legacy-revision-compat: exact historical frozen identifier
                "carrier_v2_prospective_from_round0": True,  # legacy-revision-compat: exact historical frozen identifier
            }
            if self.capacity_policy:
                state.update({
                    "capacity_policy_sha256": self.capacity_policy_sha256,
                    "capacity_effective_rank_limit": self.sd_effective_rank_limit,
                    "capacity_independent_component_stop": None,
                    "capacity_core_sha": self.capacity_policy["openevo_core"]["execution_sha"],
                })
            state.update({
                "harness_profile": self.harness_profile,
                "step_seed_policy": "injective-v1",  # legacy-revision-compat: exact historical frozen identifier
                "step_seed_amendment_sha256": self.step_seed_amendment_sha256,
                "harness201_carrier_seed_receipt_sha256": self.harness201_carrier_seed["receipt_sha256"],
            })
            base.atomic_json(self.exec_path, state)
        expected = {
            "bounded_experiment_arm_id": self.experiment_arm_id,
            "bounded_treatment_mode": self.treatment_mode,
            "bounded_state_rank": 32 if self.rank32_short_fork else 128,
        }
        for key, value in expected.items():
            if key in state and state[key] != value:
                raise RuntimeError(f"bounded formal state drift: {key}")
            state[key] = value
        state.setdefault("bounded_logical_history_count", int(state.get("next_round", 0)))
        if int(state["bounded_logical_history_count"]) != int(state.get("next_round", -1)):
            raise RuntimeError("bounded logical history must equal next round index")
        self.state = state
        self.save()
        return state

    def allowed_rollout_shas(self, r: int) -> set[str]:
        if self.rank32_short_fork and r >= 152:
            return {self.execution_git_sha}
        arm = "off" if self.experiment_arm_id == "BOUNDED_OFF" else "on"
        return recovery.validate_import(self.launch_package, arm, r, self.out/f"round-{r:03d}")

    def ensure_round_before(self, r: int, rr: Path) -> None:
        if self.rank32_short_fork and r >= 152:
            base.ensure_round_state_before(rr / "state-before.json", self.state)
            return
        arm = "off" if self.experiment_arm_id == "BOUNDED_OFF" else "on"
        recovery.ensure_before(self.launch_package, arm, r, rr, self.state, base)

    def save(self):
        base.atomic_json(self.exec_path, self.state)

    def run_bounded_transition(self, r: int, dataset_root: Path, prior: Path | None, output: Path):
        gpus, gpu_gate = self.select_post_gpu("transition", r)
        env={"BOUNDED_EXECUTION_REPO":str(self.wt),"BOUNDED_ANALYSIS_ROOT":str(self.root),"BOUNDED_GPU_UUID":GPU_UUID[gpus[0]],
             "BOUNDED_ROUND_INDEX":str(r),"BOUNDED_TREATMENT_MODE":self.treatment_mode,"BOUNDED_DATASET_ROOT":str(dataset_root),
             "BOUNDED_TRANSITION_OUTPUT":str(output),"BOUNDED_PRIOR_ADAPTER":str(prior) if prior else ''}
        self._run_managed_container('transition',r,self.formal_transition_launcher,env,gpus,container_env_key='BOUNDED_TRANSITION_CONTAINER',prevalidated_gate=gpu_gate)
        receipt=load_json(output/'bounded_recurrence_receipt.json')
        if receipt.get('status')!='PASS' or int(receipt.get('round_index',-1))!=r: raise RuntimeError('bounded transition receipt invalid')
        if receipt.get('treatment_mode')!=self.treatment_mode: raise RuntimeError('bounded treatment receipt drift')
        return receipt

    def validate_bounded_terminal(self, terminal: dict, r: int, prior: Path | None):
        if terminal.get("schema") != "openevo.bounded-gdr.formal-parametric-terminal.202609151730":
            raise RuntimeError("bounded formal terminal schema drift")
        if int(terminal.get("round_index", -1)) != r or terminal.get("treatment_mode") != self.treatment_mode:
            raise RuntimeError("bounded formal terminal identity drift")
        status = terminal.get("status")
        if status == "UPDATE":
            path = Path(str(terminal.get("adapter_path") or ""))
            receipt = path / "bounded_recurrence_receipt.json"
            if not receipt.is_file() or base.sha(path / "adapter_model.safetensors") != terminal.get("adapter_sha256"):
                raise RuntimeError("bounded UPDATE payload drift")
            return path, terminal.get("artifact_id"), (32 if self.rank32_short_fork else 128), True
        if status == "NOOP":
            if r == 0:
                raise RuntimeError("Round0 bounded NOOP is not a legal successor state")
            path = Path(str(terminal.get("adapter_path") or ""))
            if prior is None or path.resolve() != prior.resolve():
                raise RuntimeError("bounded NOOP must preserve predecessor")
            if base.sha(path / "adapter_model.safetensors") != terminal.get("adapter_sha256"):
                raise RuntimeError("bounded NOOP predecessor digest drift")
            return prior, terminal.get("artifact_id"), (32 if self.rank32_short_fork else 128), False
        raise RuntimeError("bounded formal terminal is neither UPDATE nor NOOP")

    def run_sd(self, r: int, rr: Path):
        directory = rr / "sd-lora"
        directory.mkdir(parents=True, exist_ok=True)
        self.ensure_writable(directory)
        terminal_path = directory / "terminal-receipt.json"
        prior = Path(self.state["current_sd_adapter"]) if self.state.get("current_sd_adapter") else None
        if terminal_path.is_file():
            return self.validate_bounded_terminal(load_json(terminal_path), r, prior)
        dataset = directory / "dataset"
        selection_path = dataset / "selection.json"
        if not selection_path.is_file():
            base.cmd(["python3", self.wt / "scripts/openevo_webshop/ceiling1_stage2_parametric_dataset_v2.py",  # legacy-revision-compat: exact historical frozen identifier
                      "--attempts", rr / "attempts.jsonl", "--output", dataset,
                      "--round-index", r, "--max-records", "256"])
        # Dataset authoring runs in the control namespace; hand off owned files
        # to the immutable UID1001 training consumer before opening its manifest.
        self.ensure_writable(dataset)
        for name in ("manifest.json", "records.jsonl", "selection.json"):
            path = dataset / name
            if path.is_file(): path.chmod(0o600)
        selection = load_json(selection_path)
        base.validate_parametric_selection(selection, dataset, rr / "attempts.jsonl", r)
        if selection.get("status") != "DATASET_READY":
            if r == 0:
                blocker = {
                    "schema": "openevo.bounded-gdr.round0-no-current-update-blocker.202609151730",
                    "status": "BLOCKED_OWNER_DECISION_REQUIRED",
                    "round_index": 0,
                    "selection_status": selection.get("status"),
                    "scientific_reason": "EMPTY_BOUNDED_HISTORY has no legal physical S1 without a current update; no zero adapter, fake replay, or delayed treatment is authorized.",
                    "formal_rollouts_already_consumed_this_round": 128,
                    "final_panel_access": 0,
                }
                base.write_once_json(directory / "ROUND0_NO_CURRENT_UPDATE_BLOCKER.json", blocker)
                raise RuntimeError("ROUND0_EMPTY_NO_CURRENT_UPDATE_OWNER_DECISION_REQUIRED")
            if prior is None:
                raise RuntimeError("Round1+ NOOP unexpectedly lacks bounded predecessor")
            terminal = {
                "schema": "openevo.bounded-gdr.formal-parametric-terminal.202609151730",
                "status": "NOOP",
                "round_index": r,
                "reason": selection.get("status"),
                "treatment_mode": self.treatment_mode,
                "adapter_path": str(prior),
                "adapter_sha256": base.sha(prior / "adapter_model.safetensors"),
                "artifact_id": self.state.get("current_sd_artifact_id"),
                "logical_history_count_before": r,
                "logical_history_count_after": r + 1,
                "replay_preserved": True,
                "gdr_events": 0,
                "scientific_semantics": "no current update; bounded parameter state and replay remain unchanged; round identity advances",
            }
            base.write_once_json(terminal_path, terminal)
            return self.validate_bounded_terminal(terminal, r, prior)
        transition_root = directory / "bounded-state"
        receipt = self.run_bounded_transition(r, dataset, prior, transition_root)
        terminal = {
            "schema": "openevo.bounded-gdr.formal-parametric-terminal.202609151730",
            "status": "UPDATE",
            "round_index": r,
            "treatment_mode": self.treatment_mode,
            "adapter_path": str(transition_root),
            "adapter_sha256": base.sha(transition_root / "adapter_model.safetensors"),
            "artifact_id": f"bounded-{self.experiment_arm_id.lower()}-round-{r:03d}",
            "logical_history_count_before": r,
            "logical_history_count_after": r + 1,
            "bounded_receipt_sha256": base.sha(transition_root / "bounded_recurrence_receipt.json"),
            "dataset_selection_sha256": base.sha(selection_path),
            "gdr_optimizer_event_count": int(receipt.get("gdr_optimizer_event_count", 0)),
            "formal_rows_consumed_by_transition": 0,
            "final_panel_access": int(receipt.get("protected_final_panel_access", 0)),
        }
        base.write_once_json(terminal_path, terminal)
        return self.validate_bounded_terminal(terminal, r, prior)

    def task_vector_receipt(self, r: int, rr: Path, sd_term: dict, prior_sd: Path | None, new_sd: Path | None):
        path = rr / "task-vector.json"
        if path.is_file():
            return load_json(path)
        receipt = {
            "schema": "openevo.bounded-gdr.task-vector-diagnostic.202609151730",
            "status": "PASS" if sd_term.get("status") == "UPDATE" else "STRUCTURAL_NULL",
            "round_index": r,
            "prior_adapter_sha256": base.sha(prior_sd / "adapter_model.safetensors") if prior_sd else None,
            "current_adapter_sha256": base.sha(new_sd / "adapter_model.safetensors") if new_sd else None,
            "online_use": "diagnostic-only",
            "modify_training": False,
            "modify_next_round": False,
        }
        base.write_once_json(path, receipt)
        return receipt

    def retention(self, r: int, rr: Path, current_geometry: dict):
        receipt = {"schema": "openevo.bounded-gdr.retention.202609151730", "round_index": r,
                   "status": "RETAIN_ALL_FORMAL_BOUNDED_STATES", "payload_deleted": False}
        base.write_once_json(rr / "checkpoint-retention.json", receipt)

    def apply_carrier_health_boundary(self, r: int, rr: Path):
        if self.carrier_health_policy is None:
            return None
        complete = rr / "ROUND_COMPLETE.json"
        after = rr / "state-after.json"
        if not complete.is_file() or not after.is_file():
            raise RuntimeError("Carrier Health boundary requires a sealed round")
        ledger = base.build_round_ledger(rr, self.experiment_arm_id, self.carrier_health_component_identities)
        ledger_path = rr / "carrier-health-ledger.json"
        base.write_carrier_health_json(ledger_path, ledger)
        decision = base.formal_round_decision(self.carrier_health_policy, ledger)
        decision_path = rr / "carrier-health-boundary.json"
        base.write_once_json(decision_path, decision)
        if decision["action"] == "PAUSE_FOR_DIAGNOSTIC":
            pause_path = self.out / "carrier-health-pauses" / f"round-{r:03d}.json"
            pause = {
                "schema": "openevo.carrier-health-pause.v1", "status": "PAUSE_FOR_DIAGNOSTIC",  # legacy-revision-compat: exact historical frozen identifier
                "arm": self.experiment_arm_id, "through_round": r, "next_round": r + 1,
                "policy_identity_sha256": self.carrier_health_policy_identity,
                "carrier_contract_identity_sha256": self.carrier_contract_identity,
                "carrier_adoption_identity_sha256": self.carrier_adoption_identity,
                "health_ledger_sha256": base.sha(ledger_path), "boundary_decision_sha256": base.sha(decision_path),
                "round_complete_sha256": base.sha(complete), "state_after_sha256": base.sha(after),
                "automatic_scientific_change": False, "formal_transaction_preserved": True,
            }
            base.write_once_json(pause_path, pause)
            self.state = load_json(after)
            self.state.update({"status": "PAUSE_FOR_DIAGNOSTIC", "blocker": f"Carrier Health FAIL through round {r}",
                               "carrier_health_pause_path": str(pause_path), "carrier_health_pause_sha256": base.sha(pause_path)})
            self.save()
            raise RuntimeError(self.state["blocker"])
        return decision

    def round_complete_payload(self, r: int, rr: Path, state_before: dict, state_after: dict) -> dict:
        return {
            "schema": "openevo.bounded-gdr.formal-round-complete.202609151730",
            "status": "PASS", "experiment_arm_id": self.experiment_arm_id,
            "model_arm": self.arm, "treatment_mode": self.treatment_mode, "persistent_state_rank": (32 if self.rank32_short_fork else 128), "round_index": r,
            "rollout_summary": load_json(rr / "summary.json"),
            "text_memory": load_json(rr / "text-memory" / "terminal-receipt.json"),
            "parametric_bounded_state": load_json(rr / "sd-lora" / "terminal-receipt.json"),
            "skill_bundle": load_json(rr / "skill-bundle" / "terminal-receipt.json"),
            "agent_system": load_json(rr / "agent-system" / "terminal-receipt.json"),
            "task_vector": load_json(rr / "task-vector.json"),
            "state_id_before": state_before["state_id"], "state_id_after": state_after["state_id"],
            "formal_rollouts_consumed_total": (r + 1) * 128,
            "execution_git_sha": self.execution_git_sha,
            "formal_ready_package_sha256": guard_sha(self.ready_package_path),
            "owner_launch_release_sha256": guard_sha(self.owner_launch_release_path),
            "carrier_contract_fail_noop_amendment_sha256": self.carrier_noop_amendment_sha256,
            "step_seed_formal_binding_sha256": self.step_seed_formal_binding_sha256,
            "external_teacher_calls": 0, "final_panel_access_count": 0,
        }

    def validate_completed_round(self, r: int, rr: Path) -> dict:
        before = load_json(rr / "state-before.json")
        after = load_json(rr / "state-after.json")
        complete = load_json(rr / "ROUND_COMPLETE.json")
        if complete.get("status") != "PASS" or int(complete.get("round_index", -1)) != r:
            raise RuntimeError("bounded completed round receipt drift")
        if complete.get("experiment_arm_id") != self.experiment_arm_id or complete.get("treatment_mode") != self.treatment_mode:
            raise RuntimeError("bounded completed round arm/treatment drift")
        base.validate_rollout_summary(complete["rollout_summary"], r, before["state_id"])
        base.validate_attempt_rows(rr, r, before["state_id"], self.allowed_rollout_shas(r))
        if int(after.get("next_round", -1)) != r + 1 or int(after.get("rollouts_consumed", -1)) != (r + 1) * 128:
            raise RuntimeError("bounded successor accounting drift")
        if int(after.get("bounded_logical_history_count", -1)) != r + 1:
            raise RuntimeError("bounded logical-history accounting drift")
        terminal = load_json(rr / "sd-lora" / "terminal-receipt.json")
        prior = Path(before["current_sd_adapter"]) if before.get("current_sd_adapter") else None
        adapter, artifact_id, _, _ = self.validate_bounded_terminal(terminal, r, prior)
        observed = Path(after["current_sd_adapter"]) if after.get("current_sd_adapter") else None
        if (adapter.resolve() if adapter else None) != (observed.resolve() if observed else None) or after.get("current_sd_artifact_id") != artifact_id:
            raise RuntimeError("bounded completed round adapter state drift")
        return after

    def run_round(self, r: int):
        if self.execution_git_sha:
            base.assert_execution_source(self.wt, self.execution_git_sha)
        self.ensure_previous_carrier_health_boundary(r)
        rr = self.out / f"round-{r:03d}"
        rr.mkdir(parents=True, exist_ok=True)
        self.write_carrier_scope_round_binding(r, rr)
        complete_path = rr / "ROUND_COMPLETE.json"
        if complete_path.is_file():
            self.state = self.validate_completed_round(r, rr)
            self.save()
            self.apply_carrier_health_boundary(r, rr)
            return
        if (rr / "state-after.json").is_file():
            before = load_json(rr / "state-before.json")
            after = self.validate_completed_round(r, rr) if complete_path.is_file() else load_json(rr / "state-after.json")
            payload = self.round_complete_payload(r, rr, before, after)
            base.write_once_json(complete_path, payload)
            self.state = self.validate_completed_round(r, rr)
            self.save(); self.apply_carrier_health_boundary(r, rr); return
        self.ensure_round_before(r, rr)
        self.run_rollout(r, rr)
        self.evidence(r, rr)
        prior_sd = Path(self.state["current_sd_adapter"]) if self.state.get("current_sd_adapter") else None
        prior_mem = Path(self.state["memory"]); prior_skill = Path(self.state["skill"]); prior_agent = Path(self.state["agent_system"])
        self._prime_reflector_resource_gate(r, rr, "text_memory")
        new_mem = self.reflect(r, rr, "text_memory", prior_mem, sd_adapter=prior_sd)
        new_sd, new_sd_id, _, updated = self.run_sd(r, rr)
        self._prime_reflector_resource_gate(r, rr, "skill_bundle")
        new_skill = self.reflect(r, rr, "skill_bundle", prior_skill, sd_adapter=prior_sd)
        self._prime_reflector_resource_gate(r, rr, "agent_system")
        new_agent = self.reflect(r, rr, "agent_system", prior_agent, memory=prior_mem, skill=prior_skill, sd_adapter=prior_sd)
        sd_terminal = load_json(rr / "sd-lora" / "terminal-receipt.json")
        tv = self.task_vector_receipt(r, rr, sd_terminal, prior_sd, new_sd)
        self.retention(r, rr, tv)
        if updated:
            self.state["sd_generation_count"] = int(self.state.get("sd_generation_count", 0)) + 1
        new_sid = base.state_id(Path(self.state["merged_base_receipt"]), new_sd, new_mem, new_skill, new_agent, r)
        self.state.update({
            "next_round": r + 1,
            "rollouts_consumed": (r + 1) * 128,
            "state_id": new_sid,
            "current_sd_adapter": str(new_sd) if new_sd else None,
            "current_sd_artifact_id": new_sd_id,
            "sd_component_count": (32 if self.rank32_short_fork else 128) if new_sd else 0,
            "memory": str(new_mem), "skill": str(new_skill), "agent_system": str(new_agent),
            "evidence_history": str(rr / "evidence" / "history-after.json"),
            "bounded_logical_history_count": r + 1,
            "external_teacher_calls": 0, "final_panel_access_count": 0,
            "rounds_complete": list(range(r + 1)),
        })
        state_after = dict(self.state)
        base.write_once_json(rr / "state-after.json", state_after)
        base.write_once_json(rr / "ROUND_COMPLETE.json", self.round_complete_payload(r, rr, load_json(rr / "state-before.json"), state_after))
        self.save()
        self.apply_carrier_health_boundary(r, rr)

    def _validate_previous_pair_barrier(self, r: int) -> None:
        barriers=Path(self.launch_package["pair_output_root"])/"barriers"
        if (barriers/f"round-{r:03d}.json").exists():
            raise RuntimeError("current matched barrier already exists; refuse duplicate arm execution")
        if r == 0:
            return
        prev=barriers/f"round-{r-1:03d}.json"
        if not prev.is_file(): raise RuntimeError("previous matched barrier missing; arm may not advance")
        b=load_json(prev)
        if b.get("schema")!="openevo.bounded-gdr.matched-round-barrier.202609151930" or b.get("status")!="PASS" or int(b.get("round_index",-1))!=r-1:
            raise RuntimeError("previous matched barrier identity drift")
        key="off_state_id" if self.experiment_arm_id=="BOUNDED_OFF" else "on_state_id"
        if b.get(key)!=self.state.get("state_id"):
            raise RuntimeError("previous matched barrier does not bind current arm state")

    def validate_pair_rollout_entry(self, r: int) -> None:
        self._validate_previous_pair_barrier(r)

    def _rollout_phase_receipt(self, r: int, rr: Path) -> dict:
        before=load_json(rr/"state-before.json"); summary=load_json(rr/"summary.json")
        base.validate_rollout_summary(summary,r,before["state_id"])
        producer_shas = base.validate_attempt_rows(rr,r,before["state_id"],self.allowed_rollout_shas(r))
        self.validate_rollout_harness_readback(summary,rr)
        payload={
            "schema":"openevo.bounded-gdr.formal-rollout-phase-complete.20260916",
            "status":"PASS","round_index":r,"experiment_arm_id":self.experiment_arm_id,
            "treatment_mode":self.treatment_mode,"state_id_before":before["state_id"],
            "summary_sha256":base.sha(rr/"summary.json"),"attempts_sha256":base.sha(rr/"attempts.jsonl"),
            "gpu_uuids":summary.get("gpu_uuids"),"attempt_count":int(summary.get("attempt_count",-1)),
            "execution_git_sha":next(iter(producer_shas)),"formal_rows_consumed_by_phase":128,
            "final_panel_access":0,
        }
        base.write_once_json(rr/"ROLLOUT_PHASE_COMPLETE.json",payload)
        return payload

    def run_rollout_phase(self, r: int) -> dict:
        if self.execution_git_sha: base.assert_execution_source(self.wt,self.execution_git_sha)
        self.ensure_previous_carrier_health_boundary(r)
        rr=self.out/f"round-{r:03d}"; rr.mkdir(parents=True,exist_ok=True)
        self.write_carrier_scope_round_binding(r,rr)
        if (rr/"ROUND_COMPLETE.json").is_file():
            self.validate_completed_round(r,rr)
            return self._rollout_phase_receipt(r,rr)
        self.ensure_round_before(r, rr)
        self.run_rollout(r,rr)
        return self._rollout_phase_receipt(r,rr)

    def run_post_rollout_phase(self, r: int) -> None:
        rr=self.out/f"round-{r:03d}"; phase=rr/"ROLLOUT_PHASE_COMPLETE.json"
        if not phase.is_file(): raise RuntimeError("post-rollout phase requires sealed rollout phase")
        expected=self._rollout_phase_receipt(r,rr)
        if load_json(phase)!=expected: raise RuntimeError("rollout phase receipt drift")
        self.run_round(r)

    def validate_pair_round_entry(self, r: int) -> None:
        pair_root=Path(self.launch_package["pair_output_root"]); barriers=pair_root/"barriers"
        if (barriers/f"round-{r:03d}.json").exists(): raise RuntimeError("current matched barrier already exists; refuse duplicate arm execution")
        if r>0:
            prev=barriers/f"round-{r-1:03d}.json"
            if not prev.is_file(): raise RuntimeError("previous matched barrier missing; arm may not advance")
            b=load_json(prev)
            if b.get("schema")!="openevo.bounded-gdr.matched-round-barrier.202609151930" or b.get("status")!="PASS" or int(b.get("round_index",-1))!=r-1: raise RuntimeError("previous matched barrier identity drift")
            key="off_state_id" if self.experiment_arm_id=="BOUNDED_OFF" else "on_state_id"
            if b.get(key)!=self.state.get("state_id"): raise RuntimeError("previous matched barrier does not bind current arm state")
        other_key="on" if self.experiment_arm_id=="BOUNDED_OFF" else "off"
        other_root=Path(self.launch_package["arms"][other_key]["output_root"])/f"round-{r:03d}"
        if self.experiment_arm_id=="BOUNDED_OFF" and (other_root/"ROUND_COMPLETE.json").exists(): raise RuntimeError("ON completed before OFF for current round; paired order drift")
        if self.experiment_arm_id=="BOUNDED_EFFECTIVE_STATE_ON":
            for name in ("ROUND_COMPLETE.json","state-after.json","carrier-health-boundary.json"):
                if not (other_root/name).is_file(): raise RuntimeError("ON requires sealed healthy OFF round first")
            h=load_json(other_root/"carrier-health-boundary.json")
            if h.get("action")!="CONTINUE": raise RuntimeError("ON cannot start while OFF Carrier Health is paused")

    def validate_pair_seal_entry(self) -> None:
        barriers=Path(self.launch_package["pair_output_root"])/"barriers"
        last=None
        for r in range(160):
            path=barriers/f"round-{r:03d}.json"
            if not path.is_file(): raise RuntimeError(f"arm seal requires matched barrier {r}")
            b=load_json(path)
            if b.get("schema")!="openevo.bounded-gdr.matched-round-barrier.202609151930" or b.get("status")!="PASS" or int(b.get("round_index",-1))!=r: raise RuntimeError(f"arm seal matched barrier drift at {r}")
            last=b
        key="off_state_id" if self.experiment_arm_id=="BOUNDED_OFF" else "on_state_id"
        if last is None or last.get(key)!=self.state.get("state_id"): raise RuntimeError("arm seal final matched barrier does not bind final state")

    def seal(self):
        if int(self.state.get("next_round", -1)) != 160 or int(self.state.get("rollouts_consumed", -1)) != 20480:
            raise RuntimeError("cannot seal bounded formal arm before 160x128")
        expected_state = load_json(self.out / "round-000" / "state-before.json")["state_id"]
        updates = noops = 0
        for r in range(160):
            rr = self.out / f"round-{r:03d}"
            after = self.validate_completed_round(r, rr)
            before = load_json(rr / "state-before.json")
            if before["state_id"] != expected_state:
                raise RuntimeError(f"round {r} state chain mismatch")
            expected_state = after["state_id"]
            terminal = load_json(rr / "sd-lora" / "terminal-receipt.json")
            if terminal["status"] == "UPDATE": updates += 1
            else: noops += 1
            self.validate_carrier_scope_round_binding(r, rr)
        if expected_state != self.state["state_id"]:
            raise RuntimeError("final bounded state chain mismatch")
        result = {
            "schema": "openevo.bounded-gdr.formal-arm-sealed.202609151730", "status": "PASS",
            "experiment_arm_id": self.experiment_arm_id, "treatment_mode": self.treatment_mode,
            "rounds": 160, "rollouts": 20480, "updates": updates, "noops": noops,
            "final_state_id": expected_state, "final_adapter": self.state.get("current_sd_adapter"),
            "execution_git_sha": self.execution_git_sha,
            "formal_ready_package_sha256": guard_sha(self.ready_package_path),
            "owner_launch_release_sha256": guard_sha(self.owner_launch_release_path),
            "carrier_contract_fail_noop_amendment_sha256": self.carrier_noop_amendment_sha256,
            "step_seed_formal_binding_sha256": self.step_seed_formal_binding_sha256,
            "external_teacher_calls": 0, "final_panel_access_count": 0,
        }
        base.write_once_json(self.out / "FORMAL_ARM_SEALED.json", result)
        self.state["status"] = "STAGE2_SEALED"; self.save(); return result

    def short_fork_seal(self):
        if not self.rank32_short_fork: raise RuntimeError("short_fork_seal is rank32-only")
        if int(self.state.get("next_round",-1)) != 160: raise RuntimeError("rank32 short fork incomplete")
        expected=load_json(self.out/"round-152"/"state-before.json")["state_id"]; updates=noops=0; rounds=[]
        for r in range(152,160):
            rr=self.out/f"round-{r:03d}"; before=load_json(rr/"state-before.json")
            if before.get("state_id")!=expected: raise RuntimeError(f"rank32 chain mismatch before R{r}")
            after=self.validate_completed_round(r,rr); expected=after["state_id"]; term=load_json(rr/"sd-lora"/"terminal-receipt.json")
            updates+=int(term.get("status")=="UPDATE"); noops+=int(term.get("status")!="UPDATE")
            rounds.append({"round_index":r,"state_id":expected,"adapter":after.get("current_sd_adapter")})
        result={"schema":"openevo.post-advisor.bounded-rank32-short-fork-sealed.202609282338","status":"PASS","fork_origin":"R151_AFTER",
                "persistent_state_rank":32,"current_update_rank":8,"rounds":rounds,"new_frontiers":8,"updates":updates,"noops":noops,
                "final_state_id":expected,"final_adapter":self.state.get("current_sd_adapter"),"execution_git_sha":self.execution_git_sha,
                "parent_execution_git_sha":"a6f67b66e7d0fce0dc445890a68f9f0364da452a",
                "rank32_short_fork_authority_sha256":guard_sha(self.rank32_short_fork_authority_path),
                "historical_rank128_control_rerun":False,"external_teacher_calls":0,"final_panel_access_count":0}
        base.write_once_json(self.out/"RANK32_SHORT_FORK_SEALED.json",result); self.state["status"]="RANK32_SHORT_FORK_SEALED"; self.save(); return result

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-arm-id", choices=("BOUNDED_OFF", "BOUNDED_EFFECTIVE_STATE_ON"), required=True)
    p.add_argument("--treatment-mode", choices=(OFF_MODE, ON_MODE), required=True)
    p.add_argument("--arm", choices=("qwen3-1p7b",), default="qwen3-1p7b")
    p.add_argument("--repo", type=Path, required=True); p.add_argument("--analysis-root", type=Path, required=True)
    p.add_argument("--seed-root", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stage2-activation", type=Path, required=True); p.add_argument("--capacity-policy", type=Path, required=True)
    p.add_argument("--harness-profile", choices=("201.1",), default="201.1")
    p.add_argument("--step-seed-policy", choices=("injective-v1",), default="injective-v1")  # legacy-revision-compat: exact historical frozen identifier
    p.add_argument("--step-seed-amendment", type=Path, required=True); p.add_argument("--harness201-carrier-root", type=Path, required=True)
    p.add_argument("--carrier-scope-package", type=Path, required=True); p.add_argument("--carrier-health-policy", type=Path, required=True)
    p.add_argument("--carrier-contract-identities", type=Path, required=True); p.add_argument("--carrier-adoption-receipt", type=Path, required=True)
    p.add_argument("--carrier-contract-fail-noop-amendment", type=Path, required=True)
    p.add_argument("--ready-package", type=Path, required=True); p.add_argument("--owner-launch-release", type=Path, required=True)
    p.add_argument("--execution-git-sha", required=True)
    p.add_argument("--formal-transition-launcher", type=Path, required=True)
    p.add_argument("--empty-history-manifest", type=Path, required=True); p.add_argument("--adapter-config-template", type=Path, required=True)
    p.add_argument("--gdr-runtime-path", type=Path); p.add_argument("--beta-policy-path", type=Path)
    p.add_argument("--round-index", type=int)
    p.add_argument("--rollout-only", action="store_true")
    p.add_argument("--post-rollout-only", action="store_true")
    p.add_argument("--shared-rollout-gate", type=Path)
    p.add_argument("--recover-completed-round", type=int)
    p.add_argument("--seal-only", action="store_true")
    p.add_argument("--recovery-amendment", type=Path); p.add_argument("--manual-hold-release", type=Path)
    p.add_argument("--carrier-health-resolution", type=Path); p.add_argument("--science-recovery-amendment", type=Path); p.add_argument("--rank32-short-fork-authority", type=Path)
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.experiment_arm_id == "BOUNDED_OFF" and args.treatment_mode != OFF_MODE:
        raise SystemExit("OFF arm treatment mismatch")
    if args.experiment_arm_id == "BOUNDED_EFFECTIVE_STATE_ON" and args.treatment_mode != ON_MODE:
        raise SystemExit("ON arm treatment mismatch")
    if args.treatment_mode == ON_MODE and (args.gdr_runtime_path is None or args.beta_policy_path is None):
        raise SystemExit("ON arm requires Effective-State runtime and beta policy")
    repo=args.repo.resolve(); ready=args.ready_package.resolve(); release=args.owner_launch_release.resolve()
    if args.rank32_short_fork_authority is not None:
        authority_path=args.rank32_short_fork_authority.resolve(); authority=load_json(authority_path)
        if authority.get("schema")!="openevo.post-advisor.bounded-rank32-short-fork-authority.202609282338" or authority.get("status")!="AUTHORIZED": raise SystemExit("rank32 authority invalid")
        if authority.get("fork_origin")!="R151_AFTER" or authority.get("rounds")!=list(range(152,160)): raise SystemExit("rank32 scope drift")
        if int(authority.get("final_panel_access_limit",-1))!=0 or authority.get("historical_rank128_control_rerun") is not False: raise SystemExit("rank32 protected boundary drift")
        if guard_sha(ready)!="5f9297eaa21c0dd0e385cc5731ec090476cf09464239e58f3c41e8680bad640c": raise SystemExit("parent READY drift")
        if guard_sha(release)!="2ea603e403641cec3494aa12bde869130efcca946657b54c6c5a97bf9cbd6caa": raise SystemExit("parent release drift")
        package=load_json(ready); release_doc=load_json(release); package=json.loads(json.dumps(package))
        package["arms"]["off"]["output_root"]=str(args.output.resolve()); package["execution_git_sha"]=args.execution_git_sha
        package["pair_output_root"]=str(args.output.resolve().parent); package.setdefault("execution_policy",{})["rollout_arm_concurrency"]="RANK32_SHORT_FORK_SINGLE_ARM"
    else:
        package, release_doc = validate_launch_pair(repo, ready, release)
    arm_key = "off" if args.experiment_arm_id == "BOUNDED_OFF" else "on"
    arm = package.get("arms", {}).get(arm_key) or {}
    if arm.get("experiment_arm_id") != args.experiment_arm_id or arm.get("treatment_mode") != args.treatment_mode:
        raise SystemExit("controller arm identity does not match READY package")
    if Path(str(arm.get("output_root") or "")).resolve() != args.output.resolve():
        raise SystemExit("controller output root does not match READY package")
    if package.get("execution_git_sha") != args.execution_git_sha:
        raise SystemExit("controller execution SHA does not match READY package")
    args._launch_package=package; args._launch_release=release_doc
    phase_modes=int(args.rollout_only)+int(args.post_rollout_only)
    if phase_modes and args.round_index is None:
        raise SystemExit("split rollout/post phase requires --round-index")
    ordinary_round = args.round_index is not None and phase_modes == 0
    modes=sum((int(ordinary_round), int(args.rollout_only), int(args.post_rollout_only), int(args.recover_completed_round is not None), int(args.seal_only)))
    if modes != 1:
        raise SystemExit("choose exactly one formal controller mode")
    if args.rank32_short_fork_authority is not None:
        try:
            validate_post_advisor_activation(repo, args.execution_git_sha)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
    if args.rollout_only and args.shared_rollout_gate is None:
        raise SystemExit("co-resident rollout phase requires --shared-rollout-gate")
    with exclusive_arm_lock(args.output.resolve()):
        controller=BoundedEffectiveStateController(args)
        if args.seal_only:
            if args.rank32_short_fork_authority is not None: result=controller.short_fork_seal()
            else: controller.validate_pair_seal_entry(); result=controller.seal()
            print(json.dumps(result,sort_keys=True)); return 0
        if args.recover_completed_round is not None:
            r=int(args.recover_completed_round); rr=controller.out/f"round-{r:03d}"
            if not (rr/"ROUND_COMPLETE.json").is_file():
                raise SystemExit("completed-round recovery requires ROUND_COMPLETE.json")
            controller.state=controller.validate_completed_round(r,rr); controller.save()
            boundary=controller.apply_carrier_health_boundary(r,rr)
            print(json.dumps({"status":"COMPLETED_ROUND_RECOVERY_PASS","round_index":r,"state_id":controller.state["state_id"],"carrier_health_action":boundary.get("action")},sort_keys=True)); return 0
        r=int(args.round_index)
        if r != int(controller.state.get("next_round",-1)):
            raise SystemExit("requested round is not the exact next round")
        if args.rollout_only:
            controller.validate_pair_rollout_entry(r)
            receipt=controller.run_rollout_phase(r)
            print(json.dumps({"status":"ROLLOUT_PHASE_PASS","round_index":r,"experiment_arm_id":controller.experiment_arm_id,"receipt":receipt},sort_keys=True)); return 0
        if args.post_rollout_only:
            controller.validate_pair_round_entry(r)
            controller.run_post_rollout_phase(r)
            print(json.dumps({"status":"ROUND_PASS","round_index":r,"experiment_arm_id":controller.experiment_arm_id,"state_id":controller.state["state_id"]},sort_keys=True)); return 0
        if (package.get("execution_policy") or {}).get("rollout_arm_concurrency") == "CORESIDENT_OFF_ON_SHARED_FOUR_GPU":
            raise SystemExit("formal co-resident package forbids unsplit --round-index execution")
        if args.rank32_short_fork_authority is None: controller.validate_pair_round_entry(r)
        controller.run_round(r)
        print(json.dumps({"status":"ROUND_PASS","round_index":r,"experiment_arm_id":controller.experiment_arm_id,"state_id":controller.state["state_id"]},sort_keys=True)); return 0

if __name__ == "__main__":
    raise SystemExit(main())
