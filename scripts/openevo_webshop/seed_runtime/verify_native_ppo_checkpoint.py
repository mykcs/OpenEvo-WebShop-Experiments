#!/usr/bin/env python3
"""Target-weight CUDA/FSDP component qualification; never a formal rollout."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
from datetime import timedelta
from pathlib import Path


def digest_tensor(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.reshape(-1).view(dtype=torch.uint8).numpy()).hexdigest()


def digest_file(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temporary-checkpoints", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve()
    output = args.output.resolve()
    temporary = args.temporary_checkpoints.resolve()
    if not str(temporary).startswith("/dev/shm/"):
        raise ValueError("This qualification writes ephemeral checkpoints only in task-owned /dev/shm")
    if int(os.environ["WORLD_SIZE"]) != 8:
        raise ValueError("The target component topology is DP8/SP1")
    sys.path.insert(0, str(source))
    # Reuse the existing padding-only shim for source imports. This fixture
    # explicitly uses SDPA and never claims FlashAttention qualification.
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    global torch
    import torch
    import numpy as np
    from omegaconf import OmegaConf
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from verl import DataProto
    from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
    from verl.utils.fsdp_utils import get_fsdp_wrap_policy
    from verl.workers.actor.dp_actor import DataParallelPPOActor

    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_num_threads(1)
    torch.distributed.init_process_group("nccl", timeout=timedelta(seconds=180))
    timestamps = {}

    def phase(name):
        torch.cuda.synchronize()
        timestamps[name] = time.monotonic()
        print(json.dumps({"phase": name, "rank": rank, "cuda_peak_bytes": torch.cuda.max_memory_allocated()}), flush=True)

    try:
        torch.manual_seed(73)
        random.seed(73)
        np.random.seed(73)
        phase("before_model_load")
        raw = AutoModelForCausalLM.from_pretrained(
            str(args.model.resolve()), local_files_only=True,
            torch_dtype=torch.float32, attn_implementation="sdpa",
        )
        raw.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        config = OmegaConf.load(source / "verl/trainer/config/ppo_trainer.yaml")
        actor_config = config.actor_rollout_ref.actor
        actor_config.ppo_mini_batch_size = 8  # global64 / DP8
        actor_config.ppo_micro_batch_size_per_gpu = 8
        actor_config.use_remove_padding = False
        actor_config.use_fused_kernels = False
        actor_config.use_torch_compile = False
        actor_config.use_kl_loss = True
        actor_config.kl_loss_coef = 0.01
        actor_config.opd_loss_coef = 0.01
        actor_config.opd_gate_beta = 5.0
        mesh = init_device_mesh("cuda", (8,), mesh_dim_names=("fsdp",))
        model = FSDP(
            raw, device_id=torch.cuda.current_device(), device_mesh=mesh,
            use_orig_params=False, sync_module_states=True, forward_prefetch=False,
            auto_wrap_policy=get_fsdp_wrap_policy(raw, actor_config.fsdp_config.wrap_policy),
            mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, betas=(0.9, 0.999), weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
        actor = DataParallelPPOActor(actor_config, model, optimizer)
        tokenizer = AutoTokenizer.from_pretrained(str(args.model.resolve()), local_files_only=True)
        before = [digest_tensor(p) for p in model.parameters()]
        phase("model_loaded")
        # Fixed no-task numerical fixture, shared with the accepted CPU actor
        # fixture. It neither samples a new policy output nor reuses a score.
        ids = torch.tensor([[1, 4, 5, 6, 7, 8]], device="cuda").repeat(8, 1)
        batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                 "position_ids": torch.arange(6, device="cuda").repeat(8, 1), "responses": ids[:, -2:]}
        with torch.no_grad():
            _, old = actor._forward_micro_batch(batch, temperature=1.0)
        batch.update(old_log_probs=old.detach(), advantages=torch.ones_like(old),
                     ref_log_prob=old.detach(), teacher_log_prob=old.detach() + 0.2,
                     teacher_signal_mask=torch.ones_like(old))
        count = {"steps": 0}

        def count_step(optim, positional, keyword):
            count["steps"] += 1

        hook = optimizer.register_step_post_hook(count_step)
        phase("before_update")
        metrics = actor.update_policy(DataProto.from_dict(tensors=batch, meta_info={"temperature": 1.0, "global_step": 1}))
        phase("after_update")
        hook.remove()
        assert count["steps"] == 1, count
        assert all(np.isfinite(np.asarray(v)).all() for v in metrics.values()), metrics
        after = [digest_tensor(p) for p in model.parameters()]
        assert before != after, "The target model parameters did not change"
        manager = FSDPCheckpointManager(model, optimizer, scheduler, processing_class=tokenizer)
        checkpoint = temporary / "native-checkpoint"
        manager.save_checkpoint(str(checkpoint), global_step=1, max_ckpt_to_keep=None)
        phase("checkpoint_saved")
        expected_rng = (torch.rand(3), torch.rand(3, device="cuda"), np.random.rand(3), [random.random() for _ in range(3)])
        optimizer_before = {
            str(key): {name: digest_tensor(value) if torch.is_tensor(value) else value for name, value in values.items()}
            for key, values in optimizer.state_dict()["state"].items()
        }
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1.0)
            for values in optimizer.state.values():
                for name in ("exp_avg", "exp_avg_sq"):
                    if name in values:
                        values[name].add_(1.0)
        manager.load_checkpoint(str(checkpoint), del_local_after_load=False)
        assert [digest_tensor(p) for p in model.parameters()] == after
        optimizer_after = {
            str(key): {name: digest_tensor(value) if torch.is_tensor(value) else value for name, value in values.items()}
            for key, values in optimizer.state_dict()["state"].items()
        }
        assert optimizer_before == optimizer_after
        assert torch.equal(expected_rng[0], torch.rand(3))
        assert torch.equal(expected_rng[1], torch.rand(3, device="cuda"))
        assert np.array_equal(expected_rng[2], np.random.rand(3))
        assert expected_rng[3] == [random.random() for _ in range(3)]
        phase("checkpoint_restored")
        shard_manifest = {p.name: {"bytes": p.stat().st_size, "sha256": digest_file(p)}
                          for p in checkpoint.glob(f"*rank_{rank}.pt")}
        rank_receipt = {"rank": rank, "pid": os.getpid(), "gpu_uuid": os.environ["CUDA_VISIBLE_DEVICES"].split(",")[rank],
                        "optimizer_steps": count["steps"], "parameter_change": before != after,
                        "exact_parameter_restore": True, "exact_optimizer_restore": True, "learner_cpu_cuda_rng_restore": True,
                        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(), "phase_seconds": timestamps,
                        "checkpoint_shards": shard_manifest}
        with (output / f"rank-{rank}.json").open("x") as f:
            json.dump(rank_receipt, f, sort_keys=True, indent=2)
        torch.distributed.barrier()
        # Every CUDA/FSDP assertion and rank receipt is complete. Release the
        # collective group before rank zero performs standalone CPU export/load;
        # other ranks must not wait in NCCL while that serial work runs.
        torch.distributed.destroy_process_group()
        if rank == 0:
            spec = importlib.util.spec_from_file_location("q03_target_merger", source / "scripts/model_merger.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            merger = module.FSDPModelMerger(module.ModelMergerConfig(
                operation="merge", backend="fsdp", local_dir=str(checkpoint),
                hf_model_config_path=str(checkpoint), target_dir=str(temporary / "merged-bf16")))
            merged = merger._load_and_merge_state_dicts(8, 8, (8,), ("fsdp",))
            merger.save_hf_model_and_tokenizer(merged)
            restored = AutoModelForCausalLM.from_pretrained(
                temporary / "merged-bf16", local_files_only=True, attn_implementation="sdpa", torch_dtype=torch.bfloat16)
            assert all(torch.equal(merged[k], v) for k, v in restored.state_dict().items())
            receipt = {"status": "PASS_TARGET_CUDA_NATIVE_PPO_CHECKPOINT_COMPONENTS", "world_size": 8,
                       "scope": "FP32 master/BF16 mixed-precision target weights, one native PPO optimizer step on fixed synthetic short rows, exact stock save/restore and BF16 merger/HF load; SDPA/no-remove-padding component route only",
                       "formal_activation_allowed": False, "persistent_storage_qualified": False,
                       "formal_task_consumption": 0, "policy_generations": 0, "webshop_resets": 0,
                       "webshop_actions": 0, "full_native_trainer_qualified": False,
                       "rollout_rng_and_teacher_interleaving_qualified": False, "canonical_q17_qualified": False,
                       "exact_bf16_export_reload": True, "temporary_checkpoint_root": str(temporary),
                       "rank_receipts": [json.loads((output / f"rank-{i}.json").read_text()) for i in range(8)]}
            with (output / "receipt.json").open("x") as f:
                json.dump(receipt, f, sort_keys=True, indent=2)
            print(json.dumps(receipt, sort_keys=True), flush=True)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
