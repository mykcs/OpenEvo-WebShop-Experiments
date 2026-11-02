import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY_SCHEMA = ROOT / "schemas/cross-platform-experiment-asset-record.schema.json"  # legacy-revision-compat
SCHEMA = ROOT / "schemas/cross-platform-experiment-asset-record-202609201821.schema.json"
EXAMPLE = ROOT / "templates/cross-platform-experiment-asset-record.example.json"
STANDARD = ROOT / "docs/operations/governance/CROSS_PLATFORM_EXPERIMENT_ASSET_STANDARD.md"
LANGUAGE_STANDARD = ROOT / "docs/operations/governance/RESEARCH_ASSET_LANGUAGE_AND_PRESENTATION_STANDARD.md"


def test_schema_and_example_are_valid_json():
    legacy = json.loads(LEGACY_SCHEMA.read_text())
    schema = json.loads(SCHEMA.read_text())
    example = json.loads(EXAMPLE.read_text())
    assert legacy["properties"]["schema"]["const"] == "openevo.cross-platform-experiment-asset-record.v1"  # legacy-revision-compat
    assert schema["$schema"].endswith("2020-12/schema")
    assert schema["properties"]["schema"]["const"] == "openevo.cross-platform-experiment-asset-record.202609201821"
    assert example["schema"] == "openevo.cross-platform-experiment-asset-record.202609201821"


def test_required_identity_and_provider_fields_are_frozen():
    schema = json.loads(SCHEMA.read_text())
    required = set(schema["required"])
    for field in {"canonical_id", "display_title_en", "source_authority", "provider_bindings", "language_policy"}:
        assert field in required
    providers = schema["properties"]["provider_bindings"]["items"]["properties"]["provider"]["enum"]
    assert providers == ["server", "github", "wandb", "huggingface", "kaggle", "basemodel"]


def test_language_policy_is_english_titles_chinese_first_body():
    schema = json.loads(SCHEMA.read_text())
    props = schema["properties"]["language_policy"]["properties"]
    assert props["title_language"]["const"] == "en"
    assert props["primary_explanatory_language"]["const"] == "zh-CN"
    text = LANGUAGE_STANDARD.read_text()
    assert "Status: **active operating standard**" in text
    assert "README.md" in text
    assert "README.en.md" in text
    assert "中文优先" in text
    assert "W&B Report" in text
    assert "Artifact description" in text
    assert "metric key" in text


def test_registry_is_explicitly_derived_not_scientific_authority():
    text = STANDARD.read_text()
    assert "Registry 是**生成索引**" in text
    assert "不成为第二套 experiment-state database" in text


def test_cross_platform_standard_defers_to_active_language_standard():
    text = STANDARD.read_text()
    assert "Status: **active operating standard**" in text
    assert "RESEARCH_ASSET_LANGUAGE_AND_PRESENTATION_STANDARD.md" in text
    assert "metadata-only" in text
