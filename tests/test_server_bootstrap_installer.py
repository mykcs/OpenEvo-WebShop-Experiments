import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "tools" / "install-server-bootstrap.sh"
REGISTRY = ROOT / "configs" / "platform" / "server-registry.json"


def test_server_registry_declares_supported_bootstrap_user() -> None:
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert data["schema"] == "openevo.server-workspace-registry.v1"
    assert data["bootstrap_user"] == "experiment-host"


def test_installer_keeps_pointer_private_to_supported_user_not_root() -> None:
    source = INSTALLER.read_text(encoding="utf-8")
    assert 'BOOTSTRAP_USER="$(python3 - "$REPO_REGISTRY"' in source
    assert 'BOOTSTRAP_GROUP="$(id -gn "$BOOTSTRAP_USER")"' in source
    assert 'install -d -o "$BOOTSTRAP_USER" -g "$BOOTSTRAP_GROUP" -m 700 "$POINTER_DIR"' in source
    assert 'install -o "$BOOTSTRAP_USER" -g "$BOOTSTRAP_GROUP" -m 600 "$REPO_REGISTRY" "$POINTER_PATH.tmp"' in source
    assert 'cp "$REPO_REGISTRY" "$POINTER_PATH.tmp"' not in source
    assert 'chmod 644 "$POINTER_PATH"' not in source


def test_server_bootstrap_installer_is_valid_bash() -> None:
    subprocess.run(["bash", "-n", str(INSTALLER)], check=True)
