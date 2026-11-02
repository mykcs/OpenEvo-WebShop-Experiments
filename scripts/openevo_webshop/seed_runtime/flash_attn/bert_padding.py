"""Pure-PyTorch compatibility for veRL's imported padding helpers.

The pinned route does not enable remove-padding kernels.  These helpers keep
the import contract available and provide correct tensor reshaping if a
bounded diagnostic reaches them; they are not a FlashAttention implementation.
"""

from __future__ import annotations

import torch
from einops import rearrange


def index_first_axis(input_tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Select rows from the flattened first axis."""
    return input_tensor[indices]


def unpad_input(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, None]:
    """Remove masked rows and return the metadata expected by veRL."""
    batch_size, seqlen = attention_mask.shape[:2]
    indices = torch.nonzero(attention_mask.reshape(-1), as_tuple=False).flatten()
    flat_hidden_states = hidden_states.reshape(batch_size * seqlen, *hidden_states.shape[2:])
    unpadded = flat_hidden_states.index_select(0, indices)
    seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=attention_mask.device)
    cu_seqlens[1:] = torch.cumsum(seqlens, dim=0)
    max_seqlen = int(seqlens.max().item()) if batch_size else 0
    return unpadded, indices, cu_seqlens, max_seqlen, None


def pad_input(
    hidden_states: torch.Tensor,
    indices: torch.Tensor,
    batch: int,
    seqlen: int,
) -> torch.Tensor:
    """Restore flattened rows to ``(batch, seqlen, ...)`` order."""
    output = hidden_states.new_zeros((batch * seqlen, *hidden_states.shape[1:]))
    output.index_copy_(0, indices, hidden_states)
    return output.reshape(batch, seqlen, *hidden_states.shape[1:])
