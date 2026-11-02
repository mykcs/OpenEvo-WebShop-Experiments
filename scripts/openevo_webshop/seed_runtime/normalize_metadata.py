"""Normalize audited legacy dependency metadata for the text-only SEED path."""

from importlib.metadata import distribution
from pathlib import Path


def metadata_path(package: str) -> Path:
    return Path(distribution(package)._path) / "METADATA"


def normalize_vllm() -> None:
    path = metadata_path("vllm")
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    flashinfer = [i for i, line in enumerate(lines) if "Requires-Dist: flashinfer-python==0.6.16.post3" in line]
    opencv = [i for i, line in enumerate(lines) if line.startswith("Requires-Dist: opencv-python-headless")]
    if len(flashinfer) > 1 or len(opencv) > 1:
        raise SystemExit(f"unexpected vLLM metadata shape: {path}")
    if flashinfer:
        lines[flashinfer[0]] = lines[flashinfer[0]].replace(
            "flashinfer-python==0.6.16.post3", "flashinfer-python==0.6.16.post4"
        )
    if opencv:
        lines[opencv[0]] = ""
    path.write_text("".join(lines), encoding="utf-8")
    observed = distribution("vllm").metadata.get_all("Requires-Dist") or []
    if any("flashinfer-python==0.6.16.post3" in line for line in observed):
        raise SystemExit("vLLM metadata still advertises FlashInfer post3")
    if any(line.startswith("Requires-Dist: opencv-python-headless") for line in observed):
        raise SystemExit("vLLM metadata still advertises OpenCV")


def normalize_pyserini() -> None:
    path = metadata_path("pyserini")
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    for package in ("nmslib", "lightgbm"):
        matches = [i for i, line in enumerate(lines) if line.startswith(f"Requires-Dist: {package}")]
        if len(matches) > 1:
            raise SystemExit(f"unexpected pyserini metadata shape for {package}: {path}")
        if matches:
            lines[matches[0]] = ""
    path.write_text("".join(lines), encoding="utf-8")
    observed = distribution("pyserini").metadata.get_all("Requires-Dist") or []
    for package in ("nmslib", "lightgbm"):
        if any(line.startswith(f"Requires-Dist: {package}") for line in observed):
            raise SystemExit(f"pyserini metadata still advertises {package}")


if __name__ == "__main__":
    normalize_vllm()
    normalize_pyserini()
    print("normalized audited vLLM and pyserini metadata")
