#!/usr/bin/env python3
"""Qwen3 first-class binding for the shared Stage2 rollout worker."""
from __future__ import annotations

import scripts.openevo_webshop.ceiling1_stage2_rollout_worker as shared
from scripts.openevo_webshop.ceiling1_stage2_format_hard_runtime import (
    Qwen3Stage2FormatHardRuntime,
)

# Keep all rollout/schedule/action logic identical. Qwen3 differs only in its
# frozen thinking-off chat-template binding inside the shared format envelope.
shared.Harness2011Runtime = Qwen3Stage2FormatHardRuntime

if __name__ == "__main__":
    raise SystemExit(shared.main())
