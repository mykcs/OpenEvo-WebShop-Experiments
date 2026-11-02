from pathlib import Path
import unittest


ROOT = Path(__file__).parents[1]
POOLED = ROOT / "scripts/openevo_webshop/launch_h138a_pooled_diversity.sh"
CONTINUATION = ROOT / "scripts/openevo_webshop/launch_h138a_c_frozen_panel_completion.sh"
WORKSPACE_MOUNT = (
    '-v /workspace/openevo-workspace:'
    '/workspace/openevo-workspace:ro'
)


def probe_command(script: str) -> str:
    start = script.index('probe="$(timeout')
    return script[start : script.index('2>&1)"', start)]


class H138ALauncherContractTest(unittest.TestCase):
    def test_pooled_online_probe_mounts_absolute_python_workspace(self):
        command = probe_command(POOLED.read_text(encoding="utf-8"))
        self.assertIn(WORKSPACE_MOUNT, command)
        self.assertIn('--entrypoint "${PYTHON}"', command)

    def test_continuation_probes_online_before_any_gpu_cell(self):
        script = CONTINUATION.read_text(encoding="utf-8")
        command = probe_command(script)
        self.assertIn(WORKSPACE_MOUNT, command)
        self.assertIn('--entrypoint "${PYTHON}"', command)
        self.assertLess(script.index('probe="$(timeout'), script.index('available=( $(idle_gpus)'))
        self.assertIn('PROBE_PATH', script)
        self.assertIn('effective_mode:"online"', script)
        self.assertIn('effective_mode:"offline"', script)
        self.assertIn('--arg output "${probe}"', script)

    def test_continuation_resume_uses_frozen_orchestration_identity(self):
        script = CONTINUATION.read_text(encoding="utf-8")
        self.assertIn('--arg orchestration_sha "${ORCHESTRATION_SHA}"', script)
        self.assertIn('.orchestration_sha == $orchestration_sha', script)
        self.assertIn('merge-base --is-ancestor "${ORCHESTRATION_SHA}"', script)
        self.assertIn('merge-base --is-ancestor "${RECORDED_EXECUTION_SHA}" "${EXECUTION_SHA}"', script)
        self.assertNotIn('--arg sha "${EXECUTION_SHA}"', script)

    def test_continuation_writes_stable_tracking_receipt(self):
        script = CONTINUATION.read_text(encoding="utf-8")
        self.assertIn('wandb-fallback-receipt.json', script)
        self.assertIn('wandb-offline-spool.sha256', script)
        self.assertIn('effective_mode:"offline"', script)
        self.assertIn('effective_mode:"online"', script)
        self.assertIn('wandb-online-probe-${EXECUTION_SHA}.json', script)
        self.assertIn('probe_sha256', script)
        self.assertIn('spool_manifest_sha256', script)
        self.assertIn('offline W&B spool missing', script)
        self.assertIn('.mode_counts.online + .mode_counts.offline', script)
        self.assertIn(".continuation_tracking", script)
        self.assertIn(".offline_spools[]", script)
        self.assertIn('run_ids', script)
        self.assertIn('fallback_receipts', script)

    def test_continuation_never_synthesizes_cell_fallback_receipts(self):
        script = CONTINUATION.read_text(encoding="utf-8")
        self.assertNotIn('seal_offline_cell()', script)
        self.assertNotIn('require_offline_cell_receipt()', script)
        self.assertNotIn('spool_path:$spool', script)
        self.assertIn('complete_cell "${arm}" "${task}";', script)

    def test_continuation_seals_per_execution_provenance(self):
        script = CONTINUATION.read_text(encoding="utf-8")
        self.assertIn('provenance-${EXECUTION_SHA}.json', script)
        self.assertIn('openevo.webshop-h1.38a-c-execution-provenance.v1', script)
        self.assertIn('existing per-execution provenance mismatch', script)
        self.assertIn('mv -n "${provenance_tmp}" "${EXECUTION_PROVENANCE_PATH}"', script)
        self.assertIn('for execution_provenance in "${HOST_ROOT}"/evidence/provenance-*.json', script)
        self.assertIn('cat-file -e "${provenance_execution_sha}^{commit}"', script)
        self.assertIn('merge-base --is-ancestor "${provenance_execution_sha}" "${EXECUTION_SHA}"', script)

    def test_continuation_probe_is_immutable_and_offline_is_fail_closed(self):
        script = CONTINUATION.read_text(encoding="utf-8")
        self.assertIn('[[ -f "${PROBE_PATH}" ]]', script)
        self.assertIn('.execution_sha == $execution_sha', script)
        self.assertIn('explicit_offline_allowed', script)
        self.assertIn('explicit offline requires', script)
        self.assertIn('h138a-c-online-probe-', script)
        self.assertIn('${EXECUTION_SHA:0:12}', script)

    def test_continuation_validates_exact_sources_and_full_combined_panel(self):
        script = CONTINUATION.read_text(encoding="utf-8")
        self.assertIn('reconciliation_evidence_path', script)
        self.assertIn('closeout_summary_evidence_path', script)
        self.assertIn('wandb_online_probe_evidence_path', script)
        self.assertIn('sealed_fallback_receipt_manifest.path', script)
        self.assertIn('sealed fallback receipt manifest SHA mismatch', script)
        self.assertIn('continuation control files differ from execution commit', script)
        self.assertIn('sealed fallback receipt manifest absent from execution commit', script)
        self.assertIn('--validate-source original', script)
        self.assertIn('--validate-source continuation', script)
        self.assertIn('--validate-cell', script)
        self.assertIn('completed_diversity_8_task_ids', script)
        self.assertIn('--original-root "${ORIGINAL_ROOT}"', script)
        self.assertIn('--continuation-root "${ROOT}"', script)

    def test_continuation_does_not_expand_frozen_scientific_contract(self):
        script = CONTINUATION.read_text(encoding="utf-8")
        self.assertIn('SEEDS=(2026090500 2026090501 2026090502 2026090503)', script)
        self.assertIn('for arm_adapter in "diversity-8|${d8}" "diversity-16|${d16}"', script)
        self.assertNotIn('run_parametric_smoke_tracked', script)
        self.assertNotIn('train_arm', script)


if __name__ == "__main__":
    unittest.main()
