#!/usr/bin/env python3
"""Fail-open W&B helpers for real OpenEvo/WebShop experiment runs.

Local JSON/Run Manifest evidence remains authoritative. W&B is an observability
layer: ``online`` gives live dashboards, ``offline`` preserves local runs for
later sync, and ``disabled`` must not change scientific execution.

Run identity follows the repository experiment-governance contract:
- campaign_id == W&B group;
- run name is short/human-readable;
- W&B run ID is auto-generated unless an interrupted run is explicitly resumed;
- config/tags carry reproducibility/search dimensions rather than the display name.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


_DISABLED = {"0", "false", "no", "off", "disable", "disabled", "none"}
_ALLOWED = {"offline", "online", "disabled"}
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CURRENT_CAMPAIGN = _REPO_ROOT / "configs" / "experiment" / "current-campaign.json"
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_MAX_WANDB_TAG_LENGTH = 64


def normalize_wandb_mode(value: str | None) -> str:
    raw = (value or "online").strip().lower()
    if raw in _DISABLED:
        return "disabled"
    if raw not in _ALLOWED:
        raise ValueError(f"unsupported W&B mode: {value!r}")
    return raw


def _warn(reason: str) -> None:
    print(json.dumps({"wandb_tracking": "warning", "reason": reason}, sort_keys=True), file=sys.stderr)


def load_current_campaign(path: Path | None = None) -> dict[str, Any] | None:
    campaign_path = (path or _CURRENT_CAMPAIGN).resolve()
    if not campaign_path.is_file():
        return None
    try:
        payload = json.loads(campaign_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _warn(f"current campaign unreadable: {type(exc).__name__}: {exc}")
        return None
    if not isinstance(payload, dict):
        _warn("current campaign is not a JSON object")
        return None
    tracking = payload.get("tracking")
    if not isinstance(tracking, dict):
        _warn("current campaign missing tracking object")
        return None
    campaign_id = str(payload.get("campaign_id") or "").strip()
    group = str(tracking.get("group") or "").strip()
    if not campaign_id or group != campaign_id:
        _warn("current campaign tracking.group must equal campaign_id")
        return None
    return payload


def campaign_tracking_config(campaign: Mapping[str, Any] | None) -> dict[str, Any]:
    if not campaign:
        return {}
    tracking = campaign.get("tracking") if isinstance(campaign.get("tracking"), dict) else {}
    return {
        "standard_core": campaign.get("standard_core"),
        "campaign_id": campaign.get("campaign_id"),
        "campaign_status": campaign.get("status"),
        "campaign_title": campaign.get("title"),
        "scientific_question": campaign.get("scientific_question"),
        "benchmark": campaign.get("benchmark"),
        "phase": campaign.get("phase"),
        "split": campaign.get("split"),
        "tracking_group": tracking.get("group"),
    }


def campaign_notes(campaign: Mapping[str, Any] | None) -> str | None:
    if not campaign:
        return None
    objective = str(campaign.get("objective") or "").strip()
    question = str(campaign.get("scientific_question") or "").strip()
    if objective and question:
        return f"{objective}\n\nScientific question: {question}"
    return objective or question or None


def _slug(value: str) -> str:
    return _SLUG_RE.sub("-", value.strip().lower()).strip("-") or "run"


def _bounded_tag(value: str) -> str:
    """Return a deterministic W&B tag accepted by the SDK (max 64 chars)."""
    normalized = value.strip().lower() or "run"
    if len(normalized) <= _MAX_WANDB_TAG_LENGTH:
        return normalized
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:8]
    return f"{normalized[:55]}-{digest}"


def short_run_name(*, arm: str, unit: str, seed: int | str | None = None) -> str:
    parts = [_slug(arm), _slug(unit)]
    if seed is not None:
        parts.append(f"s{_slug(str(seed))}")
    return "-".join(parts)


def standard_tags(campaign: Mapping[str, Any] | None, *, method: str | None = None, arm: str | None = None, extra: Sequence[str] = ()) -> tuple[str, ...]:
    tags: list[str] = []
    if campaign:
        for key in ("benchmark", "phase", "split"):
            value = campaign.get(key)
            if value:
                tags.append(_bounded_tag(f"{key}:{_slug(str(value))}"))
    if method:
        tags.append(_bounded_tag(f"method:{_slug(method)}"))
    if arm:
        tags.append(_bounded_tag(f"arm:{_slug(arm)}"))
    tags.extend(_bounded_tag(_slug(tag) if ":" not in tag else tag) for tag in extra if tag)
    return tuple(dict.fromkeys(tags))


def add_wandb_arguments(parser: argparse.ArgumentParser, *, default_project: str = "openevo-webshop-experiments") -> None:
    campaign = load_current_campaign()
    tracking = campaign.get("tracking") if campaign and isinstance(campaign.get("tracking"), dict) else {}
    default_mode = os.environ.get("OPENEVO_WANDB_MODE", os.environ.get("WANDB_MODE", str(tracking.get("default_mode") or "online")))
    default_entity = os.environ.get("WANDB_ENTITY", tracking.get("entity"))
    project = os.environ.get("WANDB_PROJECT", str(tracking.get("project") or default_project))
    campaign_id = os.environ.get("OPENEVO_CAMPAIGN_ID", campaign.get("campaign_id") if campaign else None)
    group = os.environ.get("WANDB_RUN_GROUP", str(tracking.get("group") or campaign_id or "")) or None
    parser.add_argument("--wandb-mode", default=default_mode, help="W&B mode: online for live dashboards, offline, disabled, or no.")
    parser.add_argument("--wandb-project", default=project)
    parser.add_argument("--wandb-entity", default=default_entity)
    parser.add_argument("--campaign-id", default=campaign_id)
    parser.add_argument("--wandb-group", default=group)
    parser.add_argument("--wandb-dir", type=Path, default=None, help="Optional local W&B root. Offline runs are kept here for later sync.")


def _start_offline_fallback_receipt(directory: Path | None, reason: str, project: str) -> Path | None:
    if directory is None:
        return None
    receipt = directory / "wandb-fallback-receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "openevo.wandb-fallback-receipt.v1",
                "requested_mode": "online",
                "effective_mode": "offline",
                "project": project,
                "online_error": reason,
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "spool_sha256": None,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return receipt


def _complete_fallback_receipt(receipt: Path) -> None:
    if not receipt.is_file():
        return
    digest = hashlib.sha256()
    for path in sorted(receipt.parent.rglob("*")):
        if not path.is_file() or path == receipt:
            continue
        digest.update(str(path.relative_to(receipt.parent)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["spool_sha256"] = "sha256:" + digest.hexdigest()
    payload["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    receipt.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@dataclass
class WandbTracker:
    mode: str
    project: str
    run: Any | None = None
    disabled_reason: str | None = None
    fallback_reason: str | None = None
    fallback_receipt: Path | None = None

    def log(self, metrics: Mapping[str, Any], *, step: int | None = None) -> None:
        if self.run is None:
            return
        clean = {key: value for key, value in metrics.items() if value is not None}
        if not clean:
            return
        try:
            self.run.log(clean) if step is None else self.run.log(clean, step=step)
        except Exception as exc:
            _warn(f"wandb.log failed: {type(exc).__name__}: {exc}")

    def define_metrics(self, metric_axes: Mapping[str, str]) -> None:
        if self.run is None:
            return
        try:
            for step_metric in dict.fromkeys(metric_axes.values()):
                self.run.define_metric(step_metric)
            for metric, step_metric in metric_axes.items():
                self.run.define_metric(metric, step_metric=step_metric)
        except Exception as exc:
            _warn(f"wandb.define_metric failed: {type(exc).__name__}: {exc}")

    def set_summary(self, values: Mapping[str, Any]) -> None:
        if self.run is None:
            return
        try:
            for key, value in values.items():
                if value is not None:
                    self.run.summary[key] = value
        except Exception as exc:
            _warn(f"wandb summary update failed: {type(exc).__name__}: {exc}")

    def log_table(self, key: str, *, columns: Sequence[str], data: Sequence[Sequence[Any]]) -> None:
        if self.run is None or not data:
            return
        try:
            import wandb  # type: ignore
            self.run.log({key: wandb.Table(columns=list(columns), data=[list(row) for row in data])})
        except Exception as exc:
            _warn(f"wandb table upload failed: {type(exc).__name__}: {exc}")

    def log_artifact_file(self, path: Path, *, name: str, artifact_type: str = "experiment-evidence", metadata: Mapping[str, Any] | None = None) -> None:
        if self.run is None or not path.is_file():
            return
        try:
            import wandb  # type: ignore
            artifact = wandb.Artifact(name=name, type=artifact_type, metadata=dict(metadata or {}))
            artifact.add_file(str(path), name=path.name)
            self.run.log_artifact(artifact)
        except Exception as exc:
            _warn(f"wandb artifact upload failed: {type(exc).__name__}: {exc}")

    def metadata(self) -> dict[str, Any]:
        if self.run is None:
            return {"status": "disabled", "mode": "disabled", "project": self.project, "reason": self.disabled_reason}
        settings = getattr(self.run, "settings", None)
        sync_dir = getattr(settings, "sync_dir", None) if settings is not None else None
        return {
            "status": "active", "mode": self.mode, "entity": getattr(self.run, "entity", None), "project": self.project,
            "run_id": getattr(self.run, "id", None), "run_path": getattr(self.run, "path", None), "run_url": getattr(self.run, "url", None),
            "run_name": getattr(self.run, "name", None), "run_group": getattr(self.run, "group", None), "job_type": getattr(self.run, "job_type", None),
            "run_dir": str(getattr(self.run, "dir", "") or "") or None, "sync_dir": str(sync_dir) if sync_dir else None,
            "offline": bool(getattr(self.run, "offline", self.mode == "offline")),
            "fallback_reason": self.fallback_reason,
            "fallback_receipt": str(self.fallback_receipt) if self.fallback_receipt else None,
        }

    def finish(self) -> None:
        if self.run is None:
            return
        try:
            self.run.finish()
        except Exception as exc:
            _warn(f"wandb.finish failed: {type(exc).__name__}: {exc}")
        finally:
            if self.fallback_receipt is not None:
                _complete_fallback_receipt(self.fallback_receipt)
            self.run = None


def start_wandb_run(*, mode: str, project: str, entity: str | None, name: str, job_type: str, config: Mapping[str, Any], tags: Sequence[str] = (), directory: Path | None = None, group: str | None = None, notes: str | None = None, metric_axes: Mapping[str, str] | None = None, run_id: str | None = None, resume: str | None = None) -> WandbTracker:
    resolved_mode = normalize_wandb_mode(mode)
    if resolved_mode == "disabled":
        return WandbTracker(mode="disabled", project=project, disabled_reason="explicitly disabled by --wandb-mode/OPENEVO_WANDB_MODE")
    if directory is not None:
        directory = directory.resolve(); directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        import wandb  # type: ignore
    except Exception as exc:
        reason = f"wandb import failed: {type(exc).__name__}: {exc}"; _warn(reason); return WandbTracker(mode="disabled", project=project, disabled_reason=reason)
    init_kwargs: dict[str, Any] = {"project": project, "entity": entity, "name": name, "job_type": job_type, "config": dict(config), "tags": list(tags), "group": group, "notes": notes, "mode": resolved_mode, "dir": str(directory) if directory is not None else None, "reinit": "create_new"}
    if run_id:
        init_kwargs["id"] = run_id
    if resume:
        init_kwargs["resume"] = resume
    try:
        run = wandb.init(**init_kwargs)
    except Exception as exc:
        reason = f"wandb.init failed: {type(exc).__name__}: {exc}"
        if resolved_mode != "online":
            _warn(reason)
            return WandbTracker(mode="disabled", project=project, disabled_reason=reason)
        fallback_receipt = _start_offline_fallback_receipt(directory, reason, project)
        fallback_kwargs = dict(init_kwargs)
        fallback_kwargs["mode"] = "offline"
        try:
            run = wandb.init(**fallback_kwargs)
        except Exception as fallback_exc:
            final_reason = f"{reason}; offline fallback failed: {type(fallback_exc).__name__}: {fallback_exc}"
            _warn(final_reason)
            return WandbTracker(mode="disabled", project=project, disabled_reason=final_reason)
        _warn(f"{reason}; explicit offline fallback started")
        tracker = WandbTracker(mode="offline", project=project, run=run, fallback_reason=reason, fallback_receipt=fallback_receipt)
        if metric_axes:
            tracker.define_metrics(metric_axes)
        return tracker
    tracker = WandbTracker(mode=resolved_mode, project=project, run=run)
    if metric_axes:
        tracker.define_metrics(metric_axes)
    return tracker


@contextmanager
def trainer_wandb_environment(tracker: WandbTracker, *, mode: str, project: str, entity: str | None, group: str, run_name: str, campaign_id: str | None = None, notes: str | None = None, trainer_run_id: str | None = None) -> Iterator[None]:
    resolved_mode = normalize_wandb_mode(mode)
    effective_mode = tracker.mode if tracker.mode == "offline" else resolved_mode
    keys = ("OPENEVO_WANDB_ENABLE", "OPENEVO_WANDB_RUN_ID", "OPENEVO_WANDB_PARENT_RUN_ID", "OPENEVO_WANDB_JOB_TYPE", "OPENEVO_CAMPAIGN_ID", "WANDB_RUN_ID", "WANDB_RUN_GROUP", "WANDB_NAME", "WANDB_NOTES", "WANDB_MODE", "WANDB_PROJECT", "WANDB_ENTITY")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        if resolved_mode == "disabled":
            os.environ["OPENEVO_WANDB_ENABLE"] = "0"; os.environ["WANDB_MODE"] = "disabled"
        else:
            os.environ["OPENEVO_WANDB_ENABLE"] = "1"; os.environ["WANDB_MODE"] = effective_mode; os.environ["WANDB_PROJECT"] = project
            os.environ["WANDB_RUN_GROUP"] = group; os.environ["WANDB_NAME"] = run_name; os.environ["OPENEVO_WANDB_JOB_TYPE"] = "trainer"
            if notes: os.environ["WANDB_NOTES"] = notes
            else: os.environ.pop("WANDB_NOTES", None)
            if campaign_id: os.environ["OPENEVO_CAMPAIGN_ID"] = campaign_id
            else: os.environ.pop("OPENEVO_CAMPAIGN_ID", None)
            parent_id = getattr(tracker.run, "id", None) if tracker.run is not None else None
            if parent_id: os.environ["OPENEVO_WANDB_PARENT_RUN_ID"] = str(parent_id)
            else: os.environ.pop("OPENEVO_WANDB_PARENT_RUN_ID", None)
            if trainer_run_id: os.environ["OPENEVO_WANDB_RUN_ID"] = trainer_run_id
            else: os.environ.pop("OPENEVO_WANDB_RUN_ID", None); os.environ.pop("WANDB_RUN_ID", None)
            if entity: os.environ["WANDB_ENTITY"] = entity
            else: os.environ.pop("WANDB_ENTITY", None)
        yield
    finally:
        for key, value in previous.items():
            if value is None: os.environ.pop(key, None)
            else: os.environ[key] = value
