#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, os, random, time
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scripts.openevo_webshop.bounded_effective_state_launch_guard_202609282338 import validate_post_advisor_activation


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


BOUNDED_SEALED_HEAD = "3a2128e5fb424403987e925c80adc15cfadbd9f7"
BOUNDED_EXECUTED_TRAINER_SHA256 = "c70c74bd549ded9a83987f8c973676740d0c8d0de0311abe8704ab2b79ff673e"
GDR_CORE_SHA = "a6207c13fc9b6215613424bb2d721bee3ae018e0"
GDR_RUNTIME_SHA256 = "16837b7730b76ba866d61f37468848a9564012300be3d0809c427ca426420267"
GDR_BETA_SHA256 = "3a1d283ac0f6e379a47fcaae0a8226be9006ac0b936361b30f07d9a8a024391e"
GDR_MAPPING_PREREG_SHA256 = "d7fc320d8fbb8b01721e472920b41d15d69ba93df1332c4194c370dc5c4cc79d"
OFF_MODE = "BOUNDED_OFF"
ON_MODE = "EFFECTIVE_STATE_GDR_LORA_V1"  # legacy-revision-compat: exact historical frozen identifier


def configure_deterministic_cuda_environment() -> None:
    expected = ":4096:8"
    current = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if current not in (None, expected):
        raise ValueError(f"CUBLAS_WORKSPACE_CONFIG drift: {current!r}")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = expected


def configure_deterministic_torch(torch) -> None:
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    controls = ("enable_flash_sdp", "enable_mem_efficient_sdp", "enable_math_sdp")
    available = tuple(hasattr(torch.backends.cuda, name) for name in controls)
    if any(available) and not all(available):
        raise RuntimeError("CUDA SDP determinism controls are only partially available")
    if all(available):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def load_effective_state_gdr(runtime_path: Path):
    if sha256_file(runtime_path) != GDR_RUNTIME_SHA256:
        raise RuntimeError("Effective-State GDR runtime SHA mismatch")
    spec = importlib.util.spec_from_file_location("bounded_gdr_effective_state_runtime", runtime_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Effective-State GDR runtime")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GatedDeltaFactorController


def existing_transition_disposition(output_dir: Path, transition_id: str) -> str:
    if not output_dir.exists():
        return "START"
    receipt_path = output_dir / "bounded_recurrence_receipt.json"
    if not receipt_path.is_file():
        return "BLOCKED_PARTIAL_OUTPUT"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") == "PASS" and receipt.get("transition_id") == transition_id:
        return "REUSE_COMPLETE"
    return "BLOCKED_IDENTITY_MISMATCH"


def compress_factors(torch, prior_a, prior_b, current_a, current_b, c_prior, c_current, target_rank):
    a_cat = torch.cat((prior_a.float(), current_a.float()), dim=0).cuda()
    b_cat = torch.cat((prior_b.float() * c_prior, current_b.float() * c_current), dim=1).cuda()
    qa, ra = torch.linalg.qr(a_cat.T, mode='reduced')
    qb, rb = torch.linalg.qr(b_cat, mode='reduced')
    u, s, vh = torch.linalg.svd(rb @ ra.T, full_matrices=False)
    rr = min(target_rank, s.numel())
    sq = s[:rr].sqrt()
    b_new = (qb @ (u[:, :rr] * sq.unsqueeze(0))).cpu().contiguous()
    a_new = ((sq.unsqueeze(1) * vh[:rr, :]) @ qa.T).cpu().contiguous()
    total = float((s * s).sum().item())
    rem = float((s[rr:] * s[rr:]).sum().item())
    err = math.sqrt(rem / total) if total else 0.0
    return a_new, b_new, err


def synthetic_test() -> dict[str, Any]:
    import torch
    torch.manual_seed(7)
    pa = torch.randn(4, 16)
    pb = torch.randn(12, 4)
    ca = torch.randn(2, 16)
    cb = torch.randn(12, 2)
    cp, cc = 0.9, -0.3
    an, bn, err = compress_factors(torch, pa, pb, ca, cb, cp, cc, 6)
    dense = cp * (pb @ pa) + cc * (cb @ ca)
    rel = float(torch.linalg.vector_norm(bn @ an - dense) / torch.linalg.vector_norm(dense))
    if rel > 1e-5 or err > 1e-5:
        raise RuntimeError(f'synthetic compression mismatch rel={rel} err={err}')
    return {'status': 'PASS', 'relative_error': rel, 'spectral_residual': err}


def build_dataset_artifact(manifest_path: Path, artifact_id: str):
    from openevo.evolution.framework.contracts import canonical_digest
    from openevo.evolution.models import WorkerClaimInputArtifact
    man = json.loads(manifest_path.read_text())
    return WorkerClaimInputArtifact(
        artifact_id=artifact_id,
        type='dataset',
        uri=manifest_path.as_uri(),
        name=str(man.get('name') or artifact_id),
        manifest_sha256=canonical_digest(man),
        records_byte_size=int(man['records_byte_size']),
        records_sha256=str(man['records_sha256']),
    )


def load_prior_factors(load_file, adapter_dir: Path, target_names: tuple[str, ...], expected_rank: int):
    tensors = load_file(str(adapter_dir / 'adapter_model.safetensors'), device='cpu')
    pairs = {}
    for name in target_names:
        stem = f'base_model.model.{name}'
        ka = f'{stem}.lora_A.weight'
        kb = f'{stem}.lora_B.weight'
        if ka not in tensors or kb not in tensors:
            raise RuntimeError(f'missing prior factors for {name}')
        a = tensors[ka].float().contiguous()
        b = tensors[kb].float().contiguous()
        if a.shape[0] != expected_rank or b.shape[1] != expected_rank:
            raise RuntimeError(f'prior rank mismatch for {name}: {a.shape}, {b.shape}')
        pairs[name] = (a, b, ka, kb)
    return pairs


def install_wrappers(torch, trainer, model, prior_pairs, coefficients, current_rank, dtype):
    class BoundedSdLoraLinear(torch.nn.Module):
        def __init__(self, base_layer, prior_a, prior_b):
            super().__init__()
            self.base_layer = base_layer
            self.prior_a = torch.nn.Parameter(prior_a, requires_grad=False)
            self.prior_b = torch.nn.Parameter(prior_b, requires_grad=False)
            self.current_a = torch.nn.Parameter(torch.empty(current_rank, base_layer.in_features, device=coefficients.device, dtype=dtype))
            self.current_b = torch.nn.Parameter(torch.empty(base_layer.out_features, current_rank, device=coefficients.device, dtype=dtype))
            torch.nn.init.kaiming_uniform_(self.current_a, a=math.sqrt(5))
            torch.nn.init.zeros_(self.current_b)

        def forward(self, inputs):
            output = self.base_layer(inputs)
            p = torch.nn.functional.linear(torch.nn.functional.linear(inputs, self.prior_a), self.prior_b)
            c = torch.nn.functional.linear(torch.nn.functional.linear(inputs, self.current_a), self.current_b)
            delta = p * coefficients[0].to(p.dtype) + c * coefficients[1].to(c.dtype)
            return output + delta.to(output.dtype)
    wrappers = {}
    for name, (pa, pb, _, _) in prior_pairs.items():
        base = model.get_submodule(name)
        wrapper = BoundedSdLoraLinear(
            base,
            pa.to(device=coefficients.device, dtype=dtype),
            pb.to(device=coefficients.device, dtype=dtype),
        )
        trainer._set_submodule(model, name, wrapper)
        wrappers[name] = wrapper
    return wrappers


def write_replay(path: Path, records: list[dict[str, Any]]) -> str:
    from openevo.evolution.framework.contracts import canonical_json
    payload = ''.join(canonical_json(x) + '\n' for x in records).encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def train(args) -> dict[str, Any]:
    validate_post_advisor_activation(REPO_ROOT)
    configure_deterministic_cuda_environment()
    import torch
    from openevo.evolution.parametric import sd_lora_trainer as trainer
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from openevo.evolution.parametric.sd_lora import _training_examples

    started = time.monotonic()
    if sha256_file(args.prior_adapter / 'adapter_model.safetensors') != args.expected_prior_sha:
        raise RuntimeError('prior adapter SHA mismatch')
    manifest = json.loads(args.dataset_manifest.read_text())
    records_path = args.dataset_manifest.with_name('records.jsonl')
    if sha256_file(records_path) != args.expected_records_sha or manifest['records_sha256'] != args.expected_records_sha:
        raise RuntimeError('dataset records SHA mismatch')
    dataset = build_dataset_artifact(args.dataset_manifest, args.dataset_artifact_id)
    current = _training_examples((dataset,), maximum=256, minimum_reward=0.5)
    prior_replay = trainer._read_examples(args.prior_replay, expected_count=args.expected_prior_replay_count)
    replay_seed = args.seed + args.logical_history_count
    replay = trainer._select_replay_examples(prior_replay, count=len(current), seed=replay_seed)
    replay_buffer = trainer._bounded_replay_buffer(prior_replay, current, capacity=args.replay_capacity)
    if len(current) != args.expected_current_count or len(replay) != args.expected_replay_count:
        raise RuntimeError('workload count mismatch')
    if len(current) + len(replay) != args.expected_steps:
        raise RuntimeError('optimizer example count mismatch')
    from openevo.evolution.framework.contracts import canonical_json
    replay_blob = ''.join(canonical_json(x) + '\n' for x in replay_buffer).encode()
    replay_sha = hashlib.sha256(replay_blob).hexdigest()
    if replay_sha != args.expected_output_replay_sha:
        raise RuntimeError(f'replay output SHA mismatch: {replay_sha}')

    configure_deterministic_torch(torch)
    trainer._initialize_cuda_runtime(torch)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    load_started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path), trust_remote_code=False, torch_dtype=torch.bfloat16, local_files_only=True
    )
    model.to(torch.device('cuda', 0))
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=False, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_load_seconds = time.monotonic() - load_started

    target_suffixes = ('q_proj', 'k_proj', 'v_proj', 'o_proj')
    target_names = trainer._target_module_names(model, target_suffixes)
    prior_pairs = load_prior_factors(load_file, args.prior_adapter, target_names, args.state_rank)
    original_truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side = 'left'
    try:
        encoded = [trainer._encode_example(tokenizer, x, args.max_length) for x in [*current, *replay]]
    finally:
        tokenizer.truncation_side = original_truncation_side

    for parameter in model.parameters():
        parameter.requires_grad = False
    coefficients = torch.nn.Parameter(torch.tensor([1.0, args.current_coefficient_init], dtype=torch.float32, device='cuda'))
    wrappers = install_wrappers(torch, trainer, model, prior_pairs, coefficients, args.current_rank, torch.bfloat16)
    model.gradient_checkpointing_enable()
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = False
    direction_parameters = []
    for wrapper in wrappers.values():
        direction_parameters.extend((wrapper.current_a, wrapper.current_b))
    trainable = [coefficients, *direction_parameters]
    optimizer = torch.optim.AdamW(
        (
            {'params': [coefficients], 'lr': args.coefficient_learning_rate, 'weight_decay': 0.0},
            {'params': direction_parameters, 'lr': args.learning_rate, 'weight_decay': 0.0},
        )
    )
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    optimizer.zero_grad(set_to_none=True)
    model.train()
    gdr_controller = None
    gdr_diagnostics: list[dict[str, float]] = []
    if args.treatment_mode == ON_MODE:
        controller_class = load_effective_state_gdr(args.gdr_runtime_path)
        if sha256_file(args.beta_policy_path) != GDR_BETA_SHA256:
            raise RuntimeError("Effective-State GDR beta checkpoint SHA mismatch")
        gdr_controller = controller_class(
            torch=torch, wrappers=wrappers, coefficients=coefficients, optimizer=optimizer,
            checkpoint_path=args.beta_policy_path, expected_sha256=GDR_BETA_SHA256,
            planned_steps=args.expected_steps,
        )
    losses = []
    coefficient_trajectory = []
    steps = 0
    train_started = time.monotonic()
    for batch in trainer._batches(encoded, batch_size=1, seed=args.seed, epochs=1):
        tensors = {k: v.to('cuda', non_blocking=True) for k, v in trainer._collate(torch, batch, int(tokenizer.pad_token_id)).items()}
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
            loss = model(**tensors).loss
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('non-finite loss')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        if gdr_controller is not None:
            gdr_controller.capture_before_step()
        scaler.step(optimizer)
        if gdr_controller is not None:
            gdr_diagnostics.append(gdr_controller.apply_after_shadow_step())
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().float().cpu()))
        coefficient_trajectory.append([steps + 1, *[float(x) for x in coefficients.detach().float().cpu().tolist()]])
        steps += 1
    train_loop_seconds = time.monotonic() - train_started
    if steps != args.expected_steps:
        raise RuntimeError(f'optimizer step mismatch: {steps}')

    learned_coefficients = [float(x) for x in coefficients.detach().float().cpu().tolist()]
    current_pairs = {
        name: (
            wrapper.current_a.detach().float().cpu().contiguous(),
            wrapper.current_b.detach().float().cpu().contiguous(),
        )
        for name, wrapper in wrappers.items()
    }
    for name, wrapper in wrappers.items():
        trainer._set_submodule(model, name, wrapper.base_layer)
    del optimizer
    torch.cuda.empty_cache()
    training_peak_memory_bytes = int(torch.cuda.max_memory_allocated())
    del model, tokenizer
    torch.cuda.empty_cache()

    compression_started = time.monotonic()
    out_tensors = {}
    module_errors = {}
    cp, cc = learned_coefficients
    for name in target_names:
        pa, pb, ka, kb = prior_pairs[name]
        ca, cb = current_pairs[name]
        an, bn, err = compress_factors(torch, pa, pb, ca, cb, cp, cc, args.state_rank)
        out_tensors[ka] = an
        out_tensors[kb] = bn
        module_errors[name] = err
    compression_seconds = time.monotonic() - compression_started

    if args.output_dir.exists():
        raise RuntimeError('output directory already exists')
    args.output_dir.mkdir(parents=True)
    with safe_open(str(args.prior_adapter / 'adapter_model.safetensors'), framework='pt', device='cpu') as f:
        metadata = f.metadata()
    save_file(out_tensors, str(args.output_dir / 'adapter_model.safetensors'), metadata=metadata)
    cfg = json.loads((args.prior_adapter / 'adapter_config.json').read_text())
    cfg['r'] = args.state_rank
    cfg['lora_alpha'] = args.state_rank
    cfg['rank_pattern'] = {}
    cfg['alpha_pattern'] = {}
    cfg['inference_mode'] = True
    write_json(args.output_dir / 'adapter_config.json', cfg)
    replay_written_sha = write_replay(args.output_dir / 'openevo_sd_lora_replay.jsonl', replay_buffer)
    if replay_written_sha != args.expected_output_replay_sha:
        raise RuntimeError('written replay SHA mismatch')

    total_seconds = time.monotonic() - started
    candidate_sha = sha256_file(args.output_dir / 'adapter_model.safetensors')
    receipt = {
        'schema': 'openevo.post-advisor.bounded-rank32-update.202609282338',  # legacy-revision-compat: sealed historical receipt schema
        'status': 'PASS',
        'round_index': args.round_index,
        'logical_history_count_before': args.logical_history_count,
        'logical_history_count_after': args.logical_history_count + 1,
        'physical_state_rank': args.state_rank,
        'current_update_rank': args.current_rank,
        'prior_adapter_sha256': args.expected_prior_sha,
        'output_adapter_sha256': candidate_sha,
        'dataset_records_sha256': args.expected_records_sha,
        'current_training_example_count': len(current),
        'replay_training_example_count': len(replay),
        'optimizer_steps': steps,
        'replay_seed': replay_seed,
        'replay_buffer_record_count': len(replay_buffer),
        'replay_output_sha256': replay_written_sha,
        'replay_output_byte_identical_to_formal_r150': replay_written_sha == args.expected_output_replay_sha,
        'training_loss_mean': sum(losses) / len(losses),
        'loss_trajectory': losses,
        'learned_coefficients': {'prior': cp, 'current': cc},
        'coefficient_trajectory': coefficient_trajectory,
        'deterministic_runtime': {
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'torch_deterministic_algorithms': bool(torch.are_deterministic_algorithms_enabled()),
            'cudnn_benchmark': bool(torch.backends.cudnn.benchmark),
            'cudnn_deterministic': bool(torch.backends.cudnn.deterministic),
            'cudnn_allow_tf32': bool(torch.backends.cudnn.allow_tf32),
            'cuda_matmul_allow_tf32': bool(torch.backends.cuda.matmul.allow_tf32),
        },
        'module_compression_relative_errors': module_errors,
        'mean_module_compression_relative_error': sum(module_errors.values()) / len(module_errors),
        'max_module_compression_relative_error': max(module_errors.values()),
        'timing': {
            'model_load_seconds': model_load_seconds,
            'train_loop_seconds': train_loop_seconds,
            'compression_seconds': compression_seconds,
            'total_seconds': total_seconds,
        },
        'gpu_peak_memory_bytes_training': training_peak_memory_bytes,
        'gpu_peak_memory_bytes_total': int(torch.cuda.max_memory_allocated()),
        'protected_final_panel_access': 0,
        'formal_rows_consumed': 0,
        'transition_id': args.transition_id,
        'treatment_mode': args.treatment_mode,
        'bounded_sealed_head': BOUNDED_SEALED_HEAD,
        'bounded_executed_trainer_sha256': BOUNDED_EXECUTED_TRAINER_SHA256,
        'rank32_parent_execution_sha': 'a6f67b66e7d0fce0dc445890a68f9f0364da452a',
        'gdr_core_sha': GDR_CORE_SHA if args.treatment_mode == ON_MODE else None,
        'gdr_runtime_sha256': GDR_RUNTIME_SHA256 if args.treatment_mode == ON_MODE else None,
        'gdr_beta_sha256': GDR_BETA_SHA256 if args.treatment_mode == ON_MODE else None,
        'gdr_mapping_version': 'EFFECTIVE_STATE_GDR_LORA_V1' if args.treatment_mode == ON_MODE else None,  # legacy-revision-compat: exact historical frozen identifier
        'gdr_mapping_prereg_sha256': GDR_MAPPING_PREREG_SHA256 if args.treatment_mode == ON_MODE else None,
        'gdr_optimizer_event_count': len(gdr_diagnostics),
        'gdr_applied_write_count': int(sum(row['applied_write_count'] for row in gdr_diagnostics)),
        'gdr_zero_direction_noop_count': int(sum(row['zero_direction_noop_count'] for row in gdr_diagnostics)),
        'gdr_g_values': sorted({float(row['g']) for row in gdr_diagnostics}),
        'gdr_retention_values': sorted({float(row['retention']) for row in gdr_diagnostics}),
        'gdr_effective_beta_min': min((float(row['effective_beta_min']) for row in gdr_diagnostics), default=None),
        'gdr_effective_beta_max': max((float(row['effective_beta_max']) for row in gdr_diagnostics), default=None),
        'gdr_factor_displacement_min': min((float(row['factor_displacement_min']) for row in gdr_diagnostics), default=None),
        'gdr_factor_displacement_max': max((float(row['factor_displacement_max']) for row in gdr_diagnostics), default=None),
        'gdr_max_intended_effective_write_abs_error': max((float(row['max_intended_effective_write_abs_error']) for row in gdr_diagnostics), default=None),
        'gdr_optimizer_step_trace': [
            {'optimizer_step': index + 1, **{key: float(value) for key, value in row.items()}}
            for index, row in enumerate(gdr_diagnostics)
        ],
        'ordering': 'current_optimizer_step:gdr_capture->adam_shadow_step->gdr_apply; round_end:merge->recompress',
    }
    write_json(args.output_dir / 'bounded_recurrence_receipt.json', receipt)
    print(json.dumps(receipt, sort_keys=True))
    return receipt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('synthetic', 'train'), required=True)
    p.add_argument('--dataset-manifest', type=Path)
    p.add_argument('--dataset-artifact-id')
    p.add_argument('--prior-adapter', type=Path)
    p.add_argument('--prior-replay', type=Path)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--model-path', type=Path)
    p.add_argument('--round-index', type=int, default=150)
    p.add_argument('--logical-history-count', type=int, default=149)
    p.add_argument('--state-rank', type=int, default=128)
    p.add_argument('--current-rank', type=int, default=8)
    p.add_argument('--seed', type=int, default=1993)
    p.add_argument('--learning-rate', type=float, default=2e-4)
    p.add_argument('--coefficient-learning-rate', type=float, default=1e-2)
    p.add_argument('--current-coefficient-init', type=float, default=0.8)
    p.add_argument('--max-length', type=int, default=2048)
    p.add_argument('--max-grad-norm', type=float, default=1.0)
    p.add_argument('--replay-capacity', type=int, default=64)
    p.add_argument('--expected-current-count', type=int, default=65)
    p.add_argument('--expected-prior-replay-count', type=int, default=64)
    p.add_argument('--expected-replay-count', type=int, default=65)
    p.add_argument('--expected-steps', type=int, default=130)
    p.add_argument('--expected-prior-sha')
    p.add_argument('--expected-records-sha')
    p.add_argument('--expected-output-replay-sha')
    p.add_argument('--transition-id', required=True)
    p.add_argument('--treatment-mode', choices=(OFF_MODE, ON_MODE), default=OFF_MODE)
    p.add_argument('--gdr-runtime-path', type=Path)
    p.add_argument('--beta-policy-path', type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == 'synthetic':
        print(json.dumps(synthetic_test(), sort_keys=True))
        return 0
    required = (
        'dataset_manifest', 'dataset_artifact_id', 'prior_adapter', 'prior_replay',
        'output_dir', 'model_path', 'expected_prior_sha', 'expected_records_sha',
        'expected_output_replay_sha',
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise SystemExit('missing train args: ' + ', '.join(missing))
    if args.state_rank != 32 or args.current_rank != 8:
        raise SystemExit('post-advisor rank32 fork requires state_rank=32 and current_rank=8')
    disposition = existing_transition_disposition(args.output_dir, args.transition_id)
    if disposition == 'REUSE_COMPLETE':
        print(json.dumps({'status': 'REUSE_COMPLETE', 'transition_id': args.transition_id}, sort_keys=True))
        return 0
    if disposition != 'START':
        raise SystemExit(f'bounded transition cannot restart: {disposition}')
    if args.treatment_mode == ON_MODE and (args.gdr_runtime_path is None or args.beta_policy_path is None):
        raise SystemExit('GDR ON requires --gdr-runtime-path and --beta-policy-path')
    if args.treatment_mode == OFF_MODE and (args.gdr_runtime_path is not None or args.beta_policy_path is not None):
        raise SystemExit('GDR OFF forbids GDR runtime/beta inputs')
    if args.round_index < 150 or args.round_index > 159 or args.logical_history_count != args.round_index - 1:
        raise SystemExit('bounded recurrence requires round 150..159 and logical_history_count=round_index-1')
    train(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
