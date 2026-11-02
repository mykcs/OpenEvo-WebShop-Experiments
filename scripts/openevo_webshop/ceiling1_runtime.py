#!/usr/bin/env python3
"""Ceiling-1.0 WebShop runtime composition helpers."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from openevo.evolution.runtime_injection import instruction_with_evolution_context
from openevo_webshop.inference import TransformersBackend


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_adapter_activation(active: list[str], expected: list[str]) -> None:
    if active != expected:
        raise RuntimeError(f"adapter activation mismatch: active={active!r} expected={expected!r}")


def build_context_messages(prompt: str, *, memory: str = "", skill_bundle: str = "", agent_system: str = "") -> tuple[list[dict[str, str]], str]:
    memory = memory.strip()
    skill_bundle = skill_bundle.strip()
    agent_system = agent_system.strip()
    effective = (
        instruction_with_evolution_context(prompt, {"memory": {"rendered_text": memory}})
        if memory else prompt
    )
    if skill_bundle:
        effective = (
            "Use the following reusable WebShop skills when applicable. They are procedural guidance, not task answers.\n"
            "=== Reusable WebShop Skills Begin ===\n"
            f"{skill_bundle}\n"
            "=== Reusable WebShop Skills End ===\n\n"
            f"{effective}"
        )
    messages: list[dict[str, str]] = []
    if agent_system:
        messages.append({"role": "system", "content": agent_system})
    messages.append({"role": "user", "content": effective})
    return messages, effective


class CeilingRuntime:
    def __init__(self, model: str, revision: str, *, bootstrap_adapter: Path | None = None,
                 sd_adapter: Path | None = None, memory: str = "", skill_bundle: str = "", agent_system: str = "") -> None:
        self.memory = memory.strip()
        self.skill_bundle = skill_bundle.strip()
        self.agent_system = agent_system.strip()
        self.base = TransformersBackend(model, device="cuda", revision=revision,
                                        adapter_path=None, local_files_only=True)
        self.base._load()
        self._adapter_receipts: list[dict[str, Any]] = []
        self._install_adapters(bootstrap_adapter, sd_adapter)
        expected = (["bootstrap"] if bootstrap_adapter is not None else []) + (["sd"] if sd_adapter is not None else [])
        validate_adapter_activation(self.active_adapters, expected)

    def _install_adapters(self, bootstrap: Path | None, sd: Path | None) -> None:
        if bootstrap is None and sd is None:
            return
        from peft import PeftModel
        model = self.base._model
        active: list[str] = []
        if bootstrap is not None:
            bootstrap = bootstrap.resolve()
            model = PeftModel.from_pretrained(model, str(bootstrap), adapter_name="bootstrap",
                                               is_trainable=False, local_files_only=True,
                                               autocast_adapter_dtype=False)
            active.append("bootstrap")
            self._adapter_receipts.append(self._adapter_receipt("bootstrap", bootstrap))
        if sd is not None:
            sd = sd.resolve()
            if isinstance(model, PeftModel):
                model.load_adapter(str(sd), adapter_name="sd", is_trainable=False)
            else:
                model = PeftModel.from_pretrained(model, str(sd), adapter_name="sd",
                                                   is_trainable=False, local_files_only=True,
                                                   autocast_adapter_dtype=False)
            active.append("sd")
            self._adapter_receipts.append(self._adapter_receipt("sd", sd))
        model.base_model.set_adapter(active)
        model.eval()
        self.base._model = model

    @staticmethod
    def _adapter_receipt(name: str, root: Path) -> dict[str, Any]:
        config = root / "adapter_config.json"
        weights = root / "adapter_model.safetensors"
        if not config.is_file() or not weights.is_file():
            raise ValueError(f"{name} adapter is incomplete")
        return {
            "name": name,
            "path": str(root),
            "config_sha256": sha256(config),
            "weights_sha256": sha256(weights),
        }

    @property
    def active_adapters(self) -> list[str]:
        model = self.base._model
        try:
            value = getattr(model, "active_adapters")
            if callable(value):
                value = value()
        except AttributeError:
            return []
        except ValueError as exc:
            # A merged Stage-2 base intentionally has no live PEFT adapter.
            # Transformers exposes active_adapters but raises in that state.
            if "No adapter loaded" in str(exc):
                return []
            raise
        return [value] if isinstance(value, str) else list(value)

    def prepare(self, prompt: str) -> tuple[dict[str, Any], dict[str, Any]]:
        messages, effective = build_context_messages(
            prompt, memory=self.memory, skill_bundle=self.skill_bundle, agent_system=self.agent_system
        )
        tok = self.base._tok
        enc = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        decoded = tok.decode(enc["input_ids"][0], skip_special_tokens=False)
        requested = {"text_memory": self.memory, "skill_bundle": self.skill_bundle, "agent_system": self.agent_system}
        readback = {}
        for name, text in requested.items():
            present = (not text) or (text in decoded)
            readback[name] = {
                "requested": bool(text),
                "present_in_decoded_input": present,
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest() if text else None,
            }
            if text and not present:
                raise RuntimeError(f"{name} runtime injection readback failed")
        receipt = {
            "messages": messages,
            "effective_user_prompt": effective,
            "decoded_input": decoded,
            "input_tokens": int(enc["input_ids"].shape[1]),
            "active_adapters": self.active_adapters,
            "adapters": list(self._adapter_receipts),
            "carrier_readback": readback,
            "skill_runtime_bridge": "webshop_qwen_skill_bridge_v1",
        }
        return enc, receipt

    def generate(self, prompt: str, *, generation_seed: int, max_new_tokens: int = 128,
                 do_sample: bool = True, temperature: float = 1.0, top_p: float = 1.0,
                 top_k: int = 0, repetition_penalty: float = 1.0) -> tuple[str, dict[str, Any]]:
        import torch
        enc, receipt = self.prepare(prompt)
        model, tok = self.base._model, self.base._tok
        enc = {key: value.to(model.device) for key, value in enc.items()}
        n_in = int(enc["input_ids"].shape[1])
        device_index = getattr(model.device, "index", None)
        devices = [device_index] if getattr(model.device, "type", None) == "cuda" and device_index is not None else []
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": bool(do_sample),
            "repetition_penalty": float(repetition_penalty),
            "pad_token_id": tok.eos_token_id,
        }
        if do_sample:
            kwargs.update(temperature=float(temperature), top_p=float(top_p), top_k=int(top_k))
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(generation_seed))
            with torch.no_grad():
                out = model.generate(**enc, **kwargs)
        receipt["sampling"] = {
            "do_sample": bool(do_sample), "temperature": float(temperature) if do_sample else None,
            "top_p": float(top_p) if do_sample else None, "top_k": int(top_k) if do_sample else None,
            "repetition_penalty": float(repetition_penalty), "max_new_tokens": int(max_new_tokens),
        }
        return tok.decode(out[0][n_in:], skip_special_tokens=True), receipt

    def close(self) -> None:
        self.base.close()
