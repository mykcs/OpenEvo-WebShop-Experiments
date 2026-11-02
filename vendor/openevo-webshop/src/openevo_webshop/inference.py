"""Inference backends (decision D2).

One unified ``generate(prompt, max_new) -> str`` protocol with two implementations:

- ``TransformersBackend`` — local HF transformers single-process (used on Colab
  this round; no vLLM/flash-attn/gateway available there). Reuses SEED's
  ``gen()`` logic verbatim. Model is loaded once and cached (module-level) so
  repeated calls are idempotent.
- ``GatewayBackend`` — OpenAI-compatible chat completions against
  ``$OPENAI_BASE_URL`` (Linux container path; fail-closed until a runtime
  endpoint is injected). Selection is deferred to runtime because
  ``$OPENAI_BASE_URL`` is not set during setup().

The runner talks only to the ``generate`` protocol, so it is backend-agnostic.
"""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import os
from pathlib import Path
from typing import Any, Protocol

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class InferenceBackend(Protocol):
    def generate(
        self,
        prompt: str,
        max_new: int = 128,
        temperature: float = 0.4,
        generation_seed: int | None = None,
    ) -> str: ...

    @property
    def model_name(self) -> str: ...


# ---------------------------------------------------------------------------
# Transformers backend (Colab path)
# ---------------------------------------------------------------------------


class TransformersBackend:
    """Local HF transformers backend. Reuses SEED webshop_rollout.gen() verbatim.

    The heavy model+tokenizer load happens once at construction and is cached;
    transformers/torch are lazily imported (decision D1) so the module imports
    on a bare python without the deps installed.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        revision: str | None = None,
        dtype: str = "bfloat16",
        adapter_path: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._revision = revision
        self._dtype = dtype
        self._adapter_path = adapter_path
        self._local_files_only = local_files_only
        self._loaded = False
        self._tok = None
        self._model = None
        self._load_receipt: dict[str, Any] | None = None

    @property
    def model_name(self) -> str:
        return self._model_id

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _tensor_digest(model: Any) -> str:
        """Digest the active in-memory LoRA tensors, including names and dtype."""

        digest = hashlib.sha256()
        torch = __import__("torch")
        count = 0
        for name, parameter in sorted(model.named_parameters()):
            if "lora_" not in name:
                continue
            tensor = parameter.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(b"\0")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(b"\0")
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
            count += 1
        if count < 1:
            raise RuntimeError("requested PEFT adapter exposed no active LoRA tensors")
        return digest.hexdigest()

    @property
    def load_receipt(self) -> dict[str, Any]:
        """Force the lazy load and prove whether PEFT tensors are active."""

        self._load()
        if self._load_receipt is None:  # pragma: no cover - defensive invariant
            raise RuntimeError("transformers backend did not produce a load receipt")
        return dict(self._load_receipt)

    def _load(self) -> None:
        if self._loaded:
            return
        import torch  # noqa: WPS433 (lazy by design)
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: WPS433

        from_pretrained_kwargs = {}
        if self._revision is not None:
            from_pretrained_kwargs["revision"] = self._revision
        if self._local_files_only:
            from_pretrained_kwargs["local_files_only"] = True
        torch_dtype = getattr(torch, self._dtype)
        self._tok = AutoTokenizer.from_pretrained(self._model_id, **from_pretrained_kwargs)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            torch_dtype=torch_dtype,
            **from_pretrained_kwargs,
        ).to(self._device)
        if self._adapter_path is not None:
            from peft import PeftModel, get_peft_model_state_dict  # noqa: WPS433

            adapter_root = Path(self._adapter_path).resolve()
            config_path = adapter_root / "adapter_config.json"
            weights_path = adapter_root / "adapter_model.safetensors"
            if not config_path.is_file() or not weights_path.is_file():
                raise ValueError("requested PEFT adapter is incomplete")
            self._model = PeftModel.from_pretrained(
                self._model,
                str(adapter_root),
                is_trainable=False,
                local_files_only=self._local_files_only,
                autocast_adapter_dtype=False,
            )
            state = get_peft_model_state_dict(self._model)
            if not state:
                raise RuntimeError("requested PEFT adapter loaded no tensors")
            active = getattr(self._model, "active_adapters", None)
            if callable(active):
                active = active()
            if active is None:
                active = [getattr(self._model, "active_adapter", "default")]
            self._load_receipt = {
                "schema": "openevo.webshop-adapter-load-receipt.v1",
                "verified": True,
                "adapter_loaded": True,
                "requested_adapter_path": str(adapter_root),
                "adapter_config_sha256": self._file_sha256(config_path),
                "adapter_file_sha256": self._file_sha256(weights_path),
                "loaded_tensor_digest": self._tensor_digest(self._model),
                "loaded_tensor_count": len(state),
                "active_adapters": list(active) if isinstance(active, (list, tuple)) else [str(active)],
                "backend_model_type": type(self._model).__name__,
            }
        else:
            self._load_receipt = {
                "schema": "openevo.webshop-adapter-load-receipt.v1",
                "verified": True,
                "adapter_loaded": False,
                "requested_adapter_path": None,
                "adapter_config_sha256": None,
                "adapter_file_sha256": None,
                "loaded_tensor_digest": None,
                "loaded_tensor_count": 0,
                "active_adapters": [],
                "backend_model_type": type(self._model).__name__,
            }
        self._model.eval()
        self._loaded = True

    def generate(
        self,
        prompt: str,
        max_new: int = 128,
        temperature: float = 0.4,
        generation_seed: int | None = None,
    ) -> str:
        self._load()
        import torch  # noqa: WPS433

        tok, model = self._tok, self._model
        # Verbatim from webshop_rollout.py:83-91.
        enc = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        n_in = enc["input_ids"].shape[1]
        generation_kwargs = {
            "max_new_tokens": max_new,
            "do_sample": temperature > 0,
            "pad_token_id": tok.eos_token_id,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = float(temperature)
        if generation_seed is None:
            rng_scope = nullcontext()
        else:
            device_index = getattr(model.device, "index", None)
            devices = (
                [device_index]
                if getattr(model.device, "type", None) == "cuda" and device_index is not None
                else []
            )
            rng_scope = torch.random.fork_rng(devices=devices)
        with rng_scope:
            if generation_seed is not None:
                torch.manual_seed(int(generation_seed))
            with torch.no_grad():
                out = model.generate(**enc, **generation_kwargs)
        return tok.decode(out[0][n_in:], skip_special_tokens=True)

    def close(self) -> None:
        if self._model is None:
            return
        import torch  # noqa: WPS433

        self._model = None
        self._tok = None
        self._loaded = False
        self._load_receipt = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Gateway backend (Linux path; runtime endpoint required)
# ---------------------------------------------------------------------------


class GatewayBackend:
    """OpenAI-compatible chat-completions backend against $OPENAI_BASE_URL.

    The base URL is read lazily on first generate(), not at construction,
    because $OPENAI_BASE_URL is injected by the runtime only once the rollout
    session starts (it is absent during setup()).
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        device: str | None = None,
    ) -> None:
        del device  # Shared rollout CLI compatibility; the gateway owns placement.
        self._model_name = model_name
        self._explicit_base_url = base_url
        self._usage_totals = {
            "requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def usage_totals(self) -> dict[str, int]:
        """Return provider-reported token usage accumulated by this backend."""
        return dict(self._usage_totals)

    def _base_url(self) -> str:
        url = self._explicit_base_url or os.environ.get("OPENAI_BASE_URL")
        if not url:
            raise RuntimeError(
                "OPENAI_BASE_URL not set; gateway backend requires the runtime "
                "to inject it at run_steps time"
            )
        return url.rstrip("/")

    def generate(
        self,
        prompt: str,
        max_new: int = 128,
        temperature: float = 0.4,
        generation_seed: int | None = None,
    ) -> str:
        base_url = self._base_url()
        import httpx  # noqa: WPS433

        payload = {
            "model": self._model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new,
            "temperature": temperature,
        }
        if generation_seed is not None:
            payload["seed"] = int(generation_seed)
        api_key = os.environ.get("OPENAI_API_KEY")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        resp = httpx.post(
            f"{base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=120.0,
        )
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage") if isinstance(data, dict) else None
        self._usage_totals["requests"] += 1
        if isinstance(usage, dict):
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = usage.get(key)
                if isinstance(value, int) and not isinstance(value, bool):
                    self._usage_totals[key] += value
        return data["choices"][0]["message"]["content"]


def make_backend(kind: str, model_name: str, **kwargs: Any) -> InferenceBackend:
    """Factory; selection deferred to runtime (run_steps), not setup()."""
    if kind == "transformers":
        return TransformersBackend(model_name, **kwargs)
    if kind == "gateway":
        return GatewayBackend(model_name, **kwargs)
    raise ValueError(f"unknown inference backend kind: {kind!r}")
