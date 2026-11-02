"""Shared structural checks for routers that supersede historical campaigns.

Historical validators use this module only to prove that the repository now
points at a complete, independently routed campaign.  It deliberately does
not decide whether that campaign may execute; execution-readiness and launcher
checks retain exact current-authority matching.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any


def is_external_successor_router(
    current: Mapping[str, Any],
    predecessor_campaign_ids: Collection[str],
    *,
    predecessor_routes: Collection[str] = (),
) -> bool:
    """Return whether ``current`` is a complete independent post-lineage route."""
    campaign_id = current.get("campaign_id")
    phase = current.get("phase")
    status = current.get("status")
    route = current.get("preregistration") or current.get("plan")
    return (
        isinstance(campaign_id, str)
        and bool(campaign_id)
        and campaign_id not in predecessor_campaign_ids
        and isinstance(phase, str)
        and bool(phase)
        and isinstance(status, str)
        and bool(status)
        and isinstance(route, str)
        and bool(route)
        and route not in predecessor_routes
        and isinstance(current.get("formal_task_consumption_allowed"), bool)
    )
