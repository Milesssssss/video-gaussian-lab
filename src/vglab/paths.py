"""仓库内固定路径。"""

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def protocol_path() -> Path:
    return repo_root() / "configs" / "phase1_protocol.json"


def example_manifest_path() -> Path:
    return repo_root() / "examples" / "scene_manifest.json"
