"""Source-exact lazy import compatibility for the pinned Step119 dense path."""

from __future__ import annotations

import ast


ACTOR_PATH = "verl/workers/actor/dp_actor.py"
ACTOR_SHA256 = "6da306319f326d0b8afe8f3d6f64a3e139719465868614f8afb96566e55c9bf3"
PADDING_NAMES = {"index_first_axis", "pad_input", "rearrange", "unpad_input"}


def patch_padding_imports(source: str) -> str:
    """Move only optional CUDA/NPU padding imports into their original branch."""
    tree = ast.parse(source)
    platform_nodes = [
        node for node in tree.body
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name)
        and node.test.id == "is_cuda_available"
    ]
    actor_classes = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DataParallelPPOActor"
    ]
    if len(platform_nodes) != 1 or len(actor_classes) != 1:
        raise ValueError("expected one original platform import and actor class")
    platform = platform_nodes[0]
    methods = [
        node for node in actor_classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == "_forward_micro_batch"
    ]
    if len(methods) != 1:
        raise ValueError("expected original actor forward method")
    branches = [
        node for node in ast.walk(methods[0])
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Attribute) and node.test.attr == "use_remove_padding"
        and isinstance(node.test.value, ast.Name) and node.test.value.id == "self"
        and any(isinstance(item, ast.Call) and isinstance(item.func, ast.Name)
                and item.func.id == "unpad_input" for item in ast.walk(node))
    ]
    if len(branches) != 1:
        raise ValueError("expected one padding-only forward branch")
    clauses = [(platform, "is_cuda_available", "flash_attn.bert_padding")]
    if len(platform.orelse) != 1 or not isinstance(platform.orelse[0], ast.If):
        raise ValueError("original NPU import branch differs")
    clauses.append((platform.orelse[0], "is_npu_available", "transformers.integrations.npu_flash_attention"))
    for node, flag, module in clauses:
        if (not isinstance(node.test, ast.Name) or node.test.id != flag or len(node.body) != 1
                or not isinstance(node.body[0], ast.ImportFrom) or node.body[0].module != module
                or {item.name for item in node.body[0].names} != PADDING_NAMES):
            raise ValueError("original padding import surface differs")
    if platform.orelse[0].orelse:
        raise ValueError("unexpected original platform fallback")
    method_nodes = {id(node) for node in ast.walk(methods[0])}
    if any(isinstance(node, ast.Name) and node.id in PADDING_NAMES
           and id(node) not in method_nodes for node in ast.walk(tree)):
        raise ValueError("padding operators are used outside the original forward method")
    lines = source.splitlines(keepends=True)
    moved = [" " * (branches[0].col_offset + 4) + line
             for line in lines[platform.lineno - 1:platform.end_lineno]]
    lines[branches[0].lineno:branches[0].lineno] = moved
    del lines[platform.lineno - 1:platform.end_lineno]
    patched = "".join(lines)
    after = ast.parse(patched)
    actor = next(node for node in after.body
                 if isinstance(node, ast.ClassDef) and node.name == "DataParallelPPOActor")
    forward = next(node for node in actor.body
                    if isinstance(node, ast.FunctionDef) and node.name == "_forward_micro_batch")
    destination = next(node for node in ast.walk(forward)
                       if isinstance(node, ast.If)
                       and isinstance(node.test, ast.Attribute)
                       and node.test.attr == "use_remove_padding"
                       and node.body
                       and isinstance(node.body[0], ast.If)
                       and ast.dump(node.body[0], include_attributes=False)
                       == ast.dump(platform, include_attributes=False))
    destination.body.pop(0)
    tree.body.remove(platform)
    if ast.dump(tree, include_attributes=False) != ast.dump(after, include_attributes=False):
        raise ValueError("compatibility overlay changed more than import timing")
    return patched
