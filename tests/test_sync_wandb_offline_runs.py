from __future__ import annotations

from scripts import sync_wandb_offline_runs as syncer


def test_sync_success_requires_zero_exit_and_no_provider_error_marker() -> None:
    assert syncer.sync_command_succeeded(0, "Syncing: https://wandb.ai/x ... done.") is True
    assert syncer.sync_command_succeeded(1, "network failure") is False
    assert syncer.sync_command_succeeded(0, "wandb: ERROR Error uploading artifact") is False
    assert syncer.sync_command_succeeded(0, "Artifact won't be committed") is False
    assert syncer.sync_command_succeeded(0, "Error uploading staging file") is False
