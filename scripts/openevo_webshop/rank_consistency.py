"""Static rank-consistency validator for rank-sensitive experiment successors.

The validator is deliberately explicit: a run profile declares the relevant
config/controller/state/checkpoint bindings and one expected state rank.  The
validator never guesses the experiment's scientific rank from filenames or chat.
It runs before GPU selection and grants no launch/scientific authority.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

SCHEMA = "openevo.rank-consistency.202609281939"
ALLOWED_ROLES = {"config", "controller", "state", "checkpoint"}


class RankConsistencyError(RuntimeError):
    """A declared rank-sensitive closure is incomplete, unreadable, or inconsistent."""


def canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _integer(value: Any, label: str) -> int:
    if type(value) is not int or value < 1:
        raise RankConsistencyError(f"{label} must be a positive integer")
    return value


def _path(resolve_path: Callable[[str], Path], raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise RankConsistencyError(f"{label}.path must be a non-empty string")
    path = resolve_path(raw).resolve()
    if not path.is_file():
        raise RankConsistencyError(f"{label} file is missing: {path}")
    return path


def _role(raw: Any, label: str) -> str:
    role = str(raw or "")
    if role not in ALLOWED_ROLES:
        raise RankConsistencyError(
            f"{label}.role must be one of {sorted(ALLOWED_ROLES)}"
        )
    return role
def _pointer(document: Any, pointer: str) -> Any:
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise RankConsistencyError(f"JSON pointer must start with '/': {pointer!r}")
    current = document
    for token in pointer.split("/")[1:]:
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            try:
                index = int(token)
            except ValueError as exc:
                raise RankConsistencyError(
                    f"JSON pointer list index is not an integer: {pointer}"
                ) from exc
            if index < 0 or index >= len(current):
                raise RankConsistencyError(f"JSON pointer out of range: {pointer}")
            current = current[index]
        elif isinstance(current, dict):
            if token not in current:
                raise RankConsistencyError(f"JSON pointer is missing: {pointer}")
            current = current[token]
        else:
            raise RankConsistencyError(
                f"JSON pointer traverses a scalar before the end: {pointer}"
            )
    return current


def _terminal_name(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _int_constant(node: ast.AST | None) -> int | None:
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return int(node.value)
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and type(node.operand.value) is int
    ):
        return -int(node.operand.value)
    return None


def _observation(
    path: Path, node: ast.AST, *, symbol: str, value: int, context: str
) -> dict[str, Any]:
    return {
        "path": str(path),
        "line": int(getattr(node, "lineno", 0)),
        "column": int(getattr(node, "col_offset", 0)),
        "symbol": symbol,
        "value": value,
        "context": context,
    }
def _python_observations(path: Path, symbols: set[str]) -> list[dict[str, Any]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise RankConsistencyError(f"cannot parse Python rank binding {path}: {exc}") from exc

    out: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets: Sequence[ast.AST]
            value_node: ast.AST | None
            if isinstance(node, ast.Assign):
                targets = node.targets
                value_node = node.value
            else:
                targets = (node.target,)
                value_node = node.value
            value = _int_constant(value_node)
            if value is not None:
                for target in targets:
                    name = _terminal_name(target)
                    if name in symbols:
                        out.append(
                            _observation(
                                path, node, symbol=name, value=value, context="assignment"
                            )
                        )

        if isinstance(node, ast.Compare):
            sequence = [node.left, *node.comparators]
            for left, right in zip(sequence, sequence[1:]):
                left_name, right_name = _terminal_name(left), _terminal_name(right)
                left_value, right_value = _int_constant(left), _int_constant(right)
                if left_name in symbols and right_value is not None:
                    out.append(
                        _observation(
                            path,
                            node,
                            symbol=left_name,
                            value=right_value,
                            context="comparison",
                        )
                    )
                if right_name in symbols and left_value is not None:
                    out.append(
                        _observation(
                            path,
                            node,
                            symbol=right_name,
                            value=left_value,
                            context="comparison",
                        )
                    )
        if isinstance(node, ast.Dict):
            for key, value_node in zip(node.keys, node.values):
                if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                    continue
                if key.value not in symbols:
                    continue
                value = _int_constant(value_node)
                if value is not None:
                    out.append(
                        _observation(
                            path,
                            node,
                            symbol=key.value,
                            value=value,
                            context="dict-literal",
                        )
                    )

        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg in symbols:
                    value = _int_constant(keyword.value)
                    if value is not None:
                        out.append(
                            _observation(
                                path,
                                node,
                                symbol=str(keyword.arg),
                                value=value,
                                context="keyword",
                            )
                        )
            option_names = {
                value.value.lstrip("-").replace("-", "_")
                for value in node.args
                if isinstance(value, ast.Constant) and isinstance(value.value, str)
                and value.value.startswith("--")
            }
            matched_options = option_names & symbols
            if matched_options:
                default = next(
                    (
                        _int_constant(keyword.value)
                        for keyword in node.keywords
                        if keyword.arg == "default"
                    ),
                    None,
                )
                if default is not None:
                    for symbol in sorted(matched_options):
                        out.append(
                            _observation(
                                path,
                                node,
                                symbol=symbol,
                                value=default,
                                context="argparse-default",
                            )
                        )
    return out
def evaluate_contract(
    contract: Mapping[str, Any],
    *,
    resolve_path: Callable[[str], Path],
) -> dict[str, Any]:
    expected = _integer(contract.get("expected_rank"), "expected_rank")
    required_roles_raw = contract.get("required_roles", sorted(ALLOWED_ROLES))
    if not isinstance(required_roles_raw, list) or not required_roles_raw:
        raise RankConsistencyError("required_roles must be a non-empty list")
    required_roles = {_role(value, "required_roles") for value in required_roles_raw}

    observations: list[dict[str, Any]] = []
    failures: list[str] = []
    observed_roles: set[str] = set()

    json_bindings = contract.get("json_bindings") or []
    python_bindings = contract.get("python_bindings") or []
    if not isinstance(json_bindings, list) or not isinstance(python_bindings, list):
        raise RankConsistencyError("rank bindings must be lists")

    for index, binding in enumerate(json_bindings):
        if not isinstance(binding, Mapping):
            raise RankConsistencyError(f"json_bindings[{index}] must be an object")
        label = f"json_bindings[{index}]"
        role = _role(binding.get("role"), label)
        path = _path(resolve_path, binding.get("path"), label)
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RankConsistencyError(f"cannot read JSON rank binding {path}: {exc}") from exc
        pointers = binding.get("pointers")
        if not isinstance(pointers, list) or not pointers:
            raise RankConsistencyError(f"{label}.pointers must be a non-empty list")
        observed_roles.add(role)
        for pointer in pointers:
            value = _pointer(document, str(pointer))
            row = {
                "role": role,
                "kind": "json",
                "path": str(path),
                "pointer": str(pointer),
                "value": value,
            }
            observations.append(row)
            if type(value) is not int:
                failures.append(f"{path}:{pointer} rank is not an integer: {value!r}")
            elif value != expected:
                failures.append(
                    f"{path}:{pointer} rank={value} != expected_rank={expected}"
                )
    for index, binding in enumerate(python_bindings):
        if not isinstance(binding, Mapping):
            raise RankConsistencyError(f"python_bindings[{index}] must be an object")
        label = f"python_bindings[{index}]"
        role = _role(binding.get("role"), label)
        path = _path(resolve_path, binding.get("path"), label)
        symbols_raw = binding.get("symbols")
        if not isinstance(symbols_raw, list) or not symbols_raw:
            raise RankConsistencyError(f"{label}.symbols must be a non-empty list")
        symbols = {str(value) for value in symbols_raw if str(value)}
        if not symbols:
            raise RankConsistencyError(f"{label}.symbols resolved empty")
        rows = _python_observations(path, symbols)
        if not rows:
            failures.append(
                f"{path} contains no numeric rank observation for {sorted(symbols)}"
            )
        observed_roles.add(role)
        for row in rows:
            row["role"] = role
            row["kind"] = "python"
            observations.append(row)
            if int(row["value"]) != expected:
                failures.append(
                    f"{path}:{row['line']} {row['symbol']}={row['value']} "
                    f"!= expected_rank={expected}"
                )

    missing_roles = sorted(required_roles - observed_roles)
    if missing_roles:
        failures.append(f"rank contract missing required roles: {missing_roles}")

    projection: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "PASS" if not failures else "FAIL",
        "expected_rank": expected,
        "required_roles": sorted(required_roles),
        "observed_roles": sorted(observed_roles),
        "observations": observations,
        "failures": failures,
        "formal_task_consumed": False,
        "gpu_requests": 0,
        "scientific_state_mutated": False,
        "scientific_authority_granted": False,
    }
    return {**projection, "receipt_sha256": canonical_sha256(projection)}
def evaluate_profile(
    profile: Mapping[str, Any],
    *,
    resolve_path: Callable[[str], Path],
) -> dict[str, Any] | None:
    prelaunch = profile.get("prelaunch")
    if not isinstance(prelaunch, Mapping):
        return None
    contract = prelaunch.get("rank_contract")
    if contract is None:
        return None
    if not isinstance(contract, Mapping):
        raise RankConsistencyError("prelaunch.rank_contract must be an object")
    return evaluate_contract(contract, resolve_path=resolve_path)


def require_pass(receipt: Mapping[str, Any] | None) -> None:
    if receipt is None:
        return
    if receipt.get("schema") != SCHEMA:
        raise RankConsistencyError("rank consistency receipt schema drift")
    projection = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt.get("receipt_sha256") != canonical_sha256(projection):
        raise RankConsistencyError("rank consistency receipt identity mismatch")
    if receipt.get("status") != "PASS":
        failures = receipt.get("failures") or []
        detail = "; ".join(map(str, failures[:5]))
        raise RankConsistencyError(f"rank consistency failed: {detail}")
    if receipt.get("formal_task_consumed") is not False:
        raise RankConsistencyError("rank consistency check is not zero-formal")
    if receipt.get("gpu_requests") != 0:
        raise RankConsistencyError("rank consistency check requested GPU")
    if receipt.get("scientific_state_mutated") is not False:
        raise RankConsistencyError("rank consistency check mutated scientific state")
