"""Minimal padding-API fallback for the no-remove-padding SEED route.

This package does not provide FlashAttention kernels.  The route keeps
``use_remove_padding=false`` and only needs the small padding helper API that
veRL imports unconditionally.
"""
