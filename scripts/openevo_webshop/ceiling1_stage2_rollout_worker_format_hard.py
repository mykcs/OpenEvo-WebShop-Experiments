#!/usr/bin/env python3
"""Shared 3B Stage2 rollout binding for the interface-owned action envelope."""
from __future__ import annotations

import scripts.openevo_webshop.ceiling1_stage2_rollout_worker as shared
from scripts.openevo_webshop.ceiling1_stage2_format_hard_runtime import (
    Stage2FormatHardRuntime,
)

shared.Harness2011Runtime = Stage2FormatHardRuntime

if __name__ == "__main__":
    raise SystemExit(shared.main())
