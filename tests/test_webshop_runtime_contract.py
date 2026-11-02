import inspect
import sys
import types
from pathlib import Path


VENDOR_SRC = Path(__file__).parents[1] / "vendor/openevo-webshop/src"
sys.path.insert(0, str(VENDOR_SRC))

from openevo_webshop.env_compat import WebshopEnvConfig
from openevo_webshop.inference import TransformersBackend
from openevo_webshop.prompts import COMPACT_PROMPT_CONTRACT
from openevo_webshop.runner import run_episode
from scripts.openevo_webshop.run_parametric_eval import (
    atomic_json,
    summarize,
    validate_existing_episode,
)
from scripts.openevo_webshop.h1_closed_loop_tracked import _require_uuid_cuda_binding
from scripts.openevo_webshop.webshop_comparison_contract import (
    PRODUCT_GROUNDED_PROMPT_CONTRACT,
    format_product_candidates,
    render_product_grounded_webshop_prompt,
)


class _FakeEnv:
    def reset(self, idx, seed=None):
        return "Instruction: [SEP] buy a blue shirt [SEP] Search", {}

    def get_available_actions(self):
        return {"has_search_bar": True, "clickables": []}

    def step(self, action):
        return "done", 0.0, True, {"task_score": 0.0, "won": False}


class _RecordingBackend:
    model_name = "fake-model"

    def __init__(self):
        self.calls = []

    def generate(self, prompt, max_new=128, temperature=0.4, generation_seed=None):
        self.calls.append(
            {
                "prompt": prompt,
                "max_new": max_new,
                "temperature": temperature,
                "generation_seed": generation_seed,
            }
        )
        return "<think>search</think><action>search[blue shirt]</action>"



def test_product_grounding_maps_visible_product_identity_to_exact_action():
    observation = (
        "WebShop [SEP] B09MLG4MXX [SEP] Gloaiidjonh Women Cotton Crew Socks "
        "[SEP] $4.93 [SEP] B00ITCBD3Y [SEP] Other Product [SEP] $12.00"
    )
    actions = {"has_search_bar": False, "clickables": ["b09mlg4mxx", "next >"]}
    candidates = format_product_candidates(observation, actions)
    assert "click[b09mlg4mxx] -> Gloaiidjonh Women Cotton Crew Socks | $4.93" in candidates
    assert "click[next >]" not in candidates


def test_product_grounded_prompt_is_a_distinct_interface_contract():
    prompt = render_product_grounded_webshop_prompt(
        task_description="Find a sock under 30 dollars",
        current_observation="WebShop [SEP] B09MLG4MXX [SEP] Sock [SEP] $4.93",
        available_actions={"has_search_bar": False, "clickables": ["b09mlg4mxx"]},
    )
    assert PRODUCT_GROUNDED_PROMPT_CONTRACT == "seed3090.webshop-product-grounded-action-prompt.v1"
    assert "Deterministic product candidates visible on this page" in prompt
    assert "click[b09mlg4mxx] -> Sock | $4.93" in prompt

def test_transformers_backend_exposes_seeded_generation_contract():
    parameters = inspect.signature(TransformersBackend.generate).parameters
    assert "temperature" in parameters
    assert "generation_seed" in parameters


def test_runner_forwards_generation_contract_and_records_seed_trace():
    backend = _RecordingBackend()
    result = run_episode(
        _FakeEnv(),
        backend,
        6680,
        max_steps=1,
        max_new=256,
        temperature=0.4,
        generation_seed=202652,
        environment_seed=2026,
    )

    assert backend.calls[0]["temperature"] == 0.4
    assert backend.calls[0]["generation_seed"] == 202652
    assert result.trace[0]["generation_seed"] == 202652
    assert result.trace[0]["environment_seed"] == 2026
    assert "search[<your query>]" in result.trace[0]["prompt"]
    assert result.trace[0]["used_fallback"] is False


def test_runner_forwards_compact_prompt_contract():
    backend = _RecordingBackend()
    result = run_episode(
        _FakeEnv(),
        backend,
        6680,
        max_steps=1,
        generation_seed=202652,
        prompt_contract=COMPACT_PROMPT_CONTRACT,
    )

    assert result.prompt_contract == COMPACT_PROMPT_CONTRACT
    assert "exactly two tagged lines" in backend.calls[0]["prompt"]


def test_environment_config_freezes_environment_seed():
    config = WebshopEnvConfig(seed_root="/seed", num_products=1000, seed=2026)
    assert config.seed == 2026


def test_transformers_backend_reuses_same_generation_seed(monkeypatch):
    class _Tensor:
        shape = (1, 1)

        def to(self, device):
            return self

    class _Tokenizer:
        eos_token_id = 0

        def apply_chat_template(self, *args, **kwargs):
            return {"input_ids": _Tensor()}

        def decode(self, tokens, skip_special_tokens=True):
            return str(tokens)

    class _Model:
        device = "cuda"

        def generate(self, **kwargs):
            assert "generator" not in kwargs
            return [[0, fake_torch.seed, kwargs["temperature"]]]

    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class _SeedScope:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    fake_torch = types.SimpleNamespace(no_grad=_NoGrad, seed=None)
    fake_torch.random = types.SimpleNamespace(fork_rng=lambda devices: _SeedScope())
    fake_torch.manual_seed = lambda seed: setattr(fake_torch, "seed", seed)
    backend = object.__new__(TransformersBackend)
    backend._loaded = True
    backend._tok = _Tokenizer()
    backend._model = _Model()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    first = backend.generate("prompt", max_new=16, temperature=0.4, generation_seed=202652)
    second = backend.generate("prompt", max_new=16, temperature=0.4, generation_seed=202652)
    third = backend.generate("prompt", max_new=16, temperature=0.4, generation_seed=202653)

    assert first == second
    assert first != third


def test_transformers_backend_omits_generator_for_greedy_generation(monkeypatch):
    class _Tensor:
        shape = (1, 1)

        def to(self, device):
            return self

    class _Tokenizer:
        eos_token_id = 0

        def apply_chat_template(self, *args, **kwargs):
            return {"input_ids": _Tensor()}

        def decode(self, tokens, skip_special_tokens=True):
            return str(tokens)

    class _Model:
        device = "cuda"

        def generate(self, **kwargs):
            assert kwargs["do_sample"] is False
            assert "generator" not in kwargs
            return [[0]]

    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class _SeedScope:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    fake_torch = types.SimpleNamespace(no_grad=_NoGrad)
    fake_torch.random = types.SimpleNamespace(fork_rng=lambda devices: _SeedScope())
    fake_torch.manual_seed = lambda seed: None
    backend = object.__new__(TransformersBackend)
    backend._loaded = True
    backend._tok = _Tokenizer()
    backend._model = _Model()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert backend.generate("prompt", max_new=16, temperature=0.0, generation_seed=202652) == "[]"


def test_h1_runner_requires_single_uuid_cuda_binding(monkeypatch):
    gpu_uuid = "GPU-example"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", gpu_uuid)
    assert _require_uuid_cuda_binding(gpu_uuid) == gpu_uuid

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    try:
        _require_uuid_cuda_binding(gpu_uuid)
    except SystemExit as exc:
        assert "CUDA_VISIBLE_DEVICES" in str(exc)
    else:
        raise AssertionError("runner accepted an ordinal-only unbound CUDA namespace")


def test_resume_requires_matching_protocol_identity(tmp_path):
    record = {
        "task_id": "webshop_006680",
        "goal_idx": 6680,
        "split": "promotion-dev",
        "arm": "base",
        "generation_seed": 202652,
        "environment_seed": 2026,
        "model_revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "adapter_identity_digest": None,
        "protocol_hash": "abc123",
        "status": "complete",
        "episode": {"task_score": 0.0, "won": False, "trace": []},
    }
    path = tmp_path / "episode.json"
    atomic_json(path, record)

    expected = {key: record[key] for key in (
        "task_id", "goal_idx", "split", "arm", "generation_seed",
        "environment_seed", "model_revision", "adapter_identity_digest",
    )}
    assert validate_existing_episode(path, expected, "abc123")["status"] == "complete"

    try:
        validate_existing_episode(path, expected, "different-protocol")
    except ValueError as exc:
        assert "protocol" in str(exc)
    else:
        raise AssertionError("resume accepted a different protocol hash")


def test_phase_g_summary_computes_paired_deltas():
    records = []
    for task in (6680, 2590):
        for seed in (202652, 202653):
            for arm, score in (("base", 0.0), ("adapter_1x", 0.25), ("adapter_cumulative", 0.5)):
                records.append({
                    "goal_idx": task,
                    "generation_seed": seed,
                    "arm": arm,
                    "episode": {"task_score": score, "won": score == 1.0, "trace": [{"used_fallback": False}]},
                })
    summary = summarize(records)
    assert summary["by_arm"]["adapter_1x"]["average_reward"] == 0.25
    assert summary["paired_deltas"]["base_to_adapter_cumulative"][0]["delta"] == 0.5


def test_seed_runtime_freezes_flashinfer_after_all_pip_installs():
    dockerfile = (Path(__file__).parents[1] / "scripts/openevo_webshop/seed_runtime/Dockerfile").read_text()
    flashinfer_override = dockerfile.index('"flashinfer-python==0.6.16.post4"')
    pip_check = dockerfile.index("RUN python3 -m pip check")
    assert flashinfer_override < pip_check
    assert '"flashinfer-cubin==0.6.16.post4"' in dockerfile
    assert '"flashinfer-jit-cache==0.6.16.post4+cu130"' in dockerfile
    assert "inspect.getsourcefile(flashinfer.comm)" in dockerfile
    assert '"array.array[int]" not in open(source' in dockerfile


def test_seed_runtime_installs_worker_visible_legacy_exports():
    runtime_dir = Path(__file__).parents[1] / "scripts/openevo_webshop/seed_runtime"
    dockerfile = (runtime_dir / "Dockerfile").read_text()
    assert "vllm_lora_models.py" in dockerfile
    assert "normalize_transformers_export.py" in dockerfile
    assert (runtime_dir / "vllm_lora_models.py").is_file()
    assert (runtime_dir / "normalize_transformers_export.py").is_file()


def test_no_docker_bootstrap_records_identity_and_requires_jdk():
    script = (Path(__file__).parents[1] / "scripts/openevo_webshop/seed_runtime/bootstrap-venv.sh").read_text()
    assert 'for tool in java javac' in script
    assert 'openevo.seed-venv-receipt.v1' in script
    assert 'formal_task_consumption_allowed' in script
    assert 'pip-freeze.txt' in script
