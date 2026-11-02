"""Schema-conformance guard for the accepted KNOWN_GOOD runtime identity.

`configs/runtime/known-good-runtime-v1.json` is only created after a real live
provisioning session passes (see configs/runtime/README.md). When it exists it
must stay conformant to `known-good-runtime-v1.schema.json`; this test guards
against silent drift of the checked-in claim. It is dependency-free (no
jsonschema) and skips when no runtime has been provisioned yet.
"""

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[1]
DOC = REPO_ROOT / "configs/runtime/known-good-runtime-v1.json"
SCHEMA = REPO_ROOT / "configs/runtime/known-good-runtime-v1.schema.json"

_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "null": type(None),
    "boolean": bool,
    "integer": int,
    "number": (int, float),
}


def _validate(value, schema, path, errors):
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: const {schema['const']!r} != {value!r}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in enum {schema['enum']}")
    declared = schema.get("type")
    if declared:
        types = declared if isinstance(declared, list) else [declared]
        ok = any(isinstance(value, _TYPES[t]) for t in types if t in _TYPES)
        if isinstance(value, bool) and "integer" in types and "boolean" not in types:
            ok = False
        if not ok:
            errors.append(f"{path}: type {type(value).__name__} not in {types}")
    if isinstance(value, dict):
        for required in schema.get("required", []):
            if required not in value:
                errors.append(f"{path}: missing required key {required!r}")
        additional = schema.get("additionalProperties")
        if additional is False:
            for key in value:
                if key not in schema.get("properties", {}):
                    errors.append(f"{path}: unexpected key {key!r}")
        elif isinstance(additional, dict):
            for key in value:
                if key not in schema.get("properties", {}):
                    _validate(value[key], additional, f"{path}.{key}", errors)
        for key, subschema in schema.get("properties", {}).items():
            if key in value:
                _validate(value[key], subschema, f"{path}.{key}", errors)
    if isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            _validate(item, schema["items"], f"{path}[{index}]", errors)


def test_known_good_runtime_document_matches_schema():
    if not DOC.exists():
        pytest.skip("no KNOWN_GOOD runtime provisioned yet")
    document = json.loads(DOC.read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    errors: list[str] = []
    _validate(document, schema, "$", errors)
    assert not errors, "known-good-runtime-v1.json drifted from its schema:\n" + "\n".join(errors)


def test_known_good_runtime_document_is_not_fabricated():
    if not DOC.exists():
        pytest.skip("no KNOWN_GOOD runtime provisioned yet")
    document = json.loads(DOC.read_text(encoding="utf-8"))
    # A KNOWN_GOOD claim must carry real qualification + smoke evidence, not a bare label.
    if document["status"] == "KNOWN_GOOD":
        assert document["qualification"]["status"] == "PASS"
        assert document["qualification"]["final_observation"] in {"PASS", "CACHE_HIT"}
        assert document["formal_heldout_tasks_consumed"] == 0
        for field in ("runtime_imports", "model_load", "webshop_boot", "openevo_zero_task_startup"):
            assert document["smoke"][field] == "PASS"


def test_known_good_requires_compliant_single_gpu_smoke():
    """Top-level KNOWN_GOOD must not coexist with a BLOCKED single-GPU smoke.

    A GPU runtime may only declare ``status == "KNOWN_GOOD"`` once its bounded
    non-heldout single-GPU smoke has actually passed (or is genuinely not
    required). While that smoke is ``BLOCKED`` (e.g. awaiting a policy-compliant
    live-idle GPU), the honest top-level status is ``PROVISIONING``. This stops a
    future reader from trusting a bare ``KNOWN_GOOD`` label over a pending GPU
    validation.
    """
    if not DOC.exists():
        pytest.skip("no KNOWN_GOOD runtime provisioned yet")
    document = json.loads(DOC.read_text(encoding="utf-8"))
    single_gpu = document["smoke"].get("single_gpu_non_heldout_smoke")
    if document["status"] == "KNOWN_GOOD":
        assert single_gpu in {"PASS", "NOT_REQUIRED"}, (
            "KNOWN_GOOD requires a compliant single-GPU smoke; got "
            f"single_gpu_non_heldout_smoke={single_gpu!r}"
        )
    if single_gpu == "BLOCKED":
        assert document["status"] != "KNOWN_GOOD", (
            "a BLOCKED single-GPU smoke is incompatible with top-level KNOWN_GOOD"
        )
