"""WebShop adapter for OpenEvo Core.

A downstream package (imports openevo Core, Core never imports this) that runs
the WebShop shopping environment through OpenEvo's rollout/evaluator contract.
"""

from __future__ import annotations

__version__ = "0.1.0"

from openevo_webshop.prompts import (
    WEBSHOP_TEMPLATE,
    build_prompt,
    extract_goal,
    fallback_action,
    fmt_actions,
    parse_action,
)

__all__ = [
    "WEBSHOP_TEMPLATE",
    "build_prompt",
    "extract_goal",
    "fallback_action",
    "fmt_actions",
    "parse_action",
    "__version__",
]
