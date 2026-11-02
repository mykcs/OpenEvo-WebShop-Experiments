"""Pinned SEED data.custom_cls adapter for the Step119 public val-only panel."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

from verl.utils.dataset.rl_dataset import RLHFDataset


_HELPER_PATH = Path(__file__).with_name("q03_source_faithful_manifest.py")
_HELPER_SPEC = importlib.util.spec_from_file_location("_q03_step119_manifest_contract", _HELPER_PATH)
if _HELPER_SPEC is None or _HELPER_SPEC.loader is None:
    raise ImportError("cannot load the sibling Q03 Step119 manifest contract")
_HELPER = importlib.util.module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(_HELPER)
validate_and_select = _HELPER.validate_and_select
validate_file_sha256 = _HELPER.validate_file_sha256
validate_environment_session_draw = _HELPER.validate_environment_session_draw


NO_TRAIN_DATASET = "q03://step119/empty-training-dataset"
EXPECTED_VALIDATION_ROWS = 128
TRAIN_SENTINEL_DATASET = "q03://step119/non-readable-training-sentinel"


def _get(config: Any, key: str, default: Any = None) -> Any:
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _paths(data_files: Any) -> list[str]:
    if isinstance(data_files, (str, Path)):
        return [str(data_files)]
    return [str(path) for path in data_files]


class Q03Step119ValidationDataset(RLHFDataset):
    """Keep the train dataset empty and bind 128 carrier rows to task IDs.

    SEED constructs a validation WebShop environment whose RNG selects the
    ``sampled_session_index`` values. The parquet row indices 0..127 identify
    those task slots; they are not session indices.
    """

    def __init__(self, data_files, tokenizer, processor, config):
        self._q03_train_sentinel = False
        self._q03_train_sentinel_length = 0
        self._q03_tasks: list[dict[str, Any]] = []
        paths = _paths(data_files)
        if paths == [NO_TRAIN_DATASET] or paths == [TRAIN_SENTINEL_DATASET]:
            self._q03_train_sentinel = True
            batch_size = _get(config, "gen_batch_size", None)
            if batch_size is None:
                batch_size = _get(config, "train_batch_size", 1)
            if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
                raise ValueError("Q03 Step119 train sentinel requires a positive integer batch size")
            # The pinned RayPPOTrainer rejects a zero-length train DataLoader
            # during construction, even though val_only returns before use.
            # Expose one synthetic batch of indices so the loader is nonempty;
            # __getitem__ always raises, making accidental training fail closed.
            self._q03_train_sentinel_length = batch_size
            return

        expected_path = str(_get(config, "q03_step119_validation_parquet"))
        if paths != [expected_path]:
            raise ValueError("Q03 Step119 dataset may read only its exact frozen validation parquet")

        parquet_path = Path(expected_path)
        observed_parquet_sha = hashlib.sha256(parquet_path.read_bytes()).hexdigest()
        if observed_parquet_sha != _get(config, "q03_step119_validation_parquet_sha256"):
            raise ValueError("Q03 Step119 validation parquet SHA-256 differs from the frozen design")

        manifest_path = Path(str(_get(config, "q03_step119_task_manifest")))
        validate_file_sha256(
            manifest_path,
            str(_get(config, "q03_step119_task_manifest_sha256")),
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self._q03_tasks = validate_and_select(manifest)
        validate_environment_session_draw(
            self._q03_tasks,
            int(_get(config, "q03_step119_validation_base_seed", 1000)),
        )

        super().__init__(
            data_files=data_files,
            tokenizer=tokenizer,
            processor=processor,
            config=config,
        )
        if len(self.dataframe) != EXPECTED_VALIDATION_ROWS:
            raise ValueError("Q03 Step119 validation parquet must contain exactly 128 carrier rows")

        # Validate the full ordered carrier before any model request is built.
        for slot in range(EXPECTED_VALIDATION_ROWS):
            row = self.dataframe[slot]
            extra = row.get("extra_info")
            if not isinstance(extra, dict) or extra.get("split") != "test" or extra.get("index") != slot:
                raise ValueError(f"Q03 Step119 validation carrier identity mismatch at slot {slot}")

    def __len__(self):
        if self._q03_train_sentinel:
            return self._q03_train_sentinel_length
        return super().__len__()

    def __getitem__(self, item):
        if self._q03_train_sentinel:
            raise IndexError("Q03 Step119 training sentinel never exposes a readable row")
        if isinstance(item, bool) or not isinstance(item, int) or not 0 <= item < EXPECTED_VALIDATION_ROWS:
            raise IndexError("Q03 Step119 validation slot is outside 0..127")
        row = super().__getitem__(item)
        slot = row.get("index")
        if isinstance(slot, bool) or slot != item:
            raise ValueError("Q03 Step119 loader changed the frozen sequential carrier order")
        task = self._q03_tasks[item]
        row.update({
            "q03_panel_slot": item,
            "q03_task_id": task["task_id"],
            "q03_worker_seed": task["worker_seed"],
            "q03_sampled_session_index": task["sampled_session_index"],
            "q03_generation_seed": task["generation_seed"],
            "q03_instruction_sha256": task["instruction_sha256"],
        })
        return row
