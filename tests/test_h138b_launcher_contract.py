from pathlib import Path
import unittest


ROOT = Path(__file__).parents[1]
LAUNCHER = ROOT / "scripts/openevo_webshop/launch_h138b_method_control.sh"


class H138BLauncherContractTests(unittest.TestCase):
    def test_launcher_is_uuid_bound_capped_and_qualification_gated(self):
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn('CORE_SHA="796e6248a61112a61c880885844d3b0420c0b6ae"', text)
        self.assertIn('CAP=6.0', text)
        self.assertIn('AMENDMENT="configs/experiment/h1.38b-continuation-amendment-2026-08-20.json"', text)
        self.assertIn(".window.no_new_cell_after_epoch", text)
        self.assertIn('OPENEVO_H138B_GPU_UUIDS:?', text)
        self.assertNotIn('GPU_UUIDS=(GPU-', text)
        self.assertIn('CORE_RUNTIME_PATHS=(src/openevo/evolution/parametric/sd_lora_trainer.py', text)
        self.assertIn('expected_core_dirty_paths=(desktop/.env.openevo-desktop', text)
        self.assertIn('--gpus "device=${uuid}"', text)
        self.assertEqual(text.count('-e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory -e GIT_CONFIG_VALUE_0=/workspace/repo'), 2)
        self.assertIn('selected GPU ${uuid} is not immediately live-idle', text)
        self.assertIn('selected GPU UUID sequence differs from the exact published authorization set', text)
        self.assertIn(".gpu_authorization_status == \"campaign-contract-authorized\"", text)
        self.assertIn(".gpu_authorization.session_reconfirmation_required == false", text)
        self.assertIn('validate_h138b_preregistration.py', text)
        self.assertLess(text.index('validate_h138b_preregistration.py'), text.index('mkdir -p'))
        self.assertIn('verify_h138b_qualification', text)
        self.assertIn('run_structural_gpu_gate', text)
        self.assertIn('h138b-pinned-7b-gpu-gate', text)
        self.assertIn('--artifact-root "${ROOT}/artifacts/sd-lora"', text)
        self.assertIn('--dataset-artifact-id "h138b-increment-$(printf', text)
        self.assertIn('openevo.webshop-h1.38b-source-transition.v1', text)
        self.assertIn('--tokenizer-root "${MODEL}"', text)
        self.assertIn('--model-root "${MODEL}"', text)
        self.assertIn('models--Qwen--Qwen2.5-7B-Instruct', text)
        self.assertIn('-e HF_HUB_CACHE="${runtime_out}/hf-hub"', text)
        self.assertIn('ln -s "${MODEL}" "${host_model_cache}/snapshots/${MODEL_REVISION}"', text)
        self.assertIn('printf \'%s\' "${MODEL_REVISION}"', text)
        qualification = text.index('verify_h138b_qualification')
        self.assertGreater(text.index('matrix_stage fresh'), qualification)
        self.assertIn('partial formal cell', text)
        self.assertIn('safe_time\n  within_cap\n  line="$(nvidia-smi -i "${uuid}"', text)
        self.assertIn('while true; do safe_time; within_cap; mapfile -t values', text)
        self.assertIn('openevo.webshop-h1.38b-gpu-job-wall-time.v1', text)
        self.assertIn('local runtime_out="${ROOT}/runtime/${name}"', text)
        self.assertIn('local host_runtime="${HOST_ROOT}/runtime/${name}"', text)
        self.assertNotIn('mkdir -p "${host_out}"/{home,tmp,xdg-cache', text)
        self.assertIn('if docker run --rm --name "${name}"', text)
        self.assertIn('status=failed', text)
        self.assertIn('mv "${receipt_tmp}" "${receipt}"', text)
        run_gpu = text[text.index('run_gpu() {'):text.index('gpu_job_accounting() {')]
        self.assertLess(run_gpu.index('started_epoch="$(date +%s)"'), run_gpu.index('if docker run'))
        self.assertLess(run_gpu.index('if docker run'), run_gpu.index('ended_epoch="$(date +%s)"'))
        self.assertLess(run_gpu.index('ended_epoch="$(date +%s)"'), run_gpu.index('mv "${receipt_tmp}" "${receipt}"'))
        self.assertNotIn('find "${HOST_ROOT}/evaluation" -name summary.json', text)
        self.assertIn("source:\"outer-docker-wall-time-receipts-v1\"", text)
        self.assertIn('attach_gpu_job_accounting "${partial_reconciliation}"', text)
        self.assertEqual(text.count('attach_gpu_job_accounting "${HOST_ROOT}/evidence/h1.38b-reconciliation.json"'), 1)
        self.assertNotIn('docker restart', text)
        self.assertNotIn('nvidia-smi --gpu-reset', text)


if __name__ == "__main__":
    unittest.main()
