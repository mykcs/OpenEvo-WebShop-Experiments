"""Add the audited Transformers 5.15.0 legacy vision-model export."""

from importlib import metadata
from importlib.util import find_spec
from pathlib import Path


EXPECTED_VERSION = "5.15.0"
IMPORT_STRUCTURE_NEEDLE = (
    "    import_structure = define_import_structure(Path(__file__).parent / \"models\", prefix=\"models\")\n"
)
IMPORT_STRUCTURE_UPDATE_NEEDLE = "    import_structure[frozenset({})].update(_import_structure)\n"
IMPORT_STRUCTURE_PATCH = IMPORT_STRUCTURE_UPDATE_NEEDLE + (
    "    for _backend_modules in import_structure.values():\n"
    "        for _module_names in _backend_modules.values():\n"
    "            if \"AutoModelForImageTextToText\" in _module_names:\n"
    "                _module_names.add(\"AutoModelForVision2Seq\")\n"
)
MODEL_ALIAS = (
    "\n# OpenEvo/SEED compatibility: v5 renamed this export.\n"
    "AutoModelForVision2Seq = AutoModelForImageTextToText\n"
)


def main() -> None:
    version = metadata.version("transformers")
    if version != EXPECTED_VERSION:
        raise RuntimeError(f"unexpected transformers version: {version}")

    spec = find_spec("transformers")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("cannot locate the transformers package")
    package_root = Path(next(iter(spec.submodule_search_locations)))
    init_path = package_root / "__init__.py"
    model_path = package_root / "models" / "auto" / "modeling_auto.py"
    init_text = init_path.read_text(encoding="utf-8")
    model_text = model_path.read_text(encoding="utf-8")

    if "AutoModelForImageTextToText" not in model_text:
        raise RuntimeError("the expected Transformers source symbol is absent")
    if "AutoModelForVision2Seq = AutoModelForImageTextToText" not in model_text:
        model_path.write_text(model_text.rstrip() + MODEL_ALIAS + "\n", encoding="utf-8")

    if "_module_names.add(\"AutoModelForVision2Seq\")" not in init_text:
        if init_text.count(IMPORT_STRUCTURE_NEEDLE) != 1 or init_text.count(IMPORT_STRUCTURE_UPDATE_NEEDLE) != 1:
            raise RuntimeError("Transformers import-structure anchor changed")
        init_path.write_text(
            init_text.replace(IMPORT_STRUCTURE_UPDATE_NEEDLE, IMPORT_STRUCTURE_PATCH),
            encoding="utf-8",
        )

    print({"transformers": version, "model": str(model_path), "init": str(init_path)})


if __name__ == "__main__":
    main()
