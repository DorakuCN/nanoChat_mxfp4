"""
Utilities for resolving experimental MXFP4 / NVFP4 precision settings.

The goal is to keep all precision related tweaks isolated from the
production code so we can iterate quickly as PyTorch adds support for
Blackwell 4-bit floating-point formats.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Optional

import torch

from nanochat.common import print0


class PrecisionNotAvailableError(RuntimeError):
    """Raised when a requested experimental dtype is not available."""


def _iter_modules() -> Iterable[object]:
    """Yield torch modules that may expose dtype objects."""
    modules = [torch]
    for attr in ("cuda", "dtypes"):
        module = getattr(torch, attr, None)
        if module is not None:
            modules.append(module)
    # walk a couple of obvious CUDA submodules
    cuda_module = getattr(torch, "cuda", None)
    if cuda_module is not None:
        for attr in ("amp", "nn", "float8", "formats"):
            sub = getattr(cuda_module, attr, None)
            if sub is not None:
                modules.append(sub)
    return modules


def _lookup_dtype(name: str) -> Optional[torch.dtype]:
    """Attempt to locate a dtype object that matches the given name."""
    canonical = name.lower()
    candidate_suffixes = [
        canonical,
        canonical.replace("_", ""),
        canonical.replace("fp", "float"),
        canonical.replace("nv", "nv_"),
        canonical + "_dtype",
    ]
    candidate_suffixes += [s.upper() for s in candidate_suffixes]

    for module in _iter_modules():
        for attr in dir(module):
            if not attr:
                continue
            value = getattr(module, attr)
            if not isinstance(value, torch.dtype):
                continue
            lowered = attr.lower()
            if (
                lowered == canonical
                or lowered.replace("_", "") == canonical.replace("_", "")
                or lowered in candidate_suffixes
            ):
                return value
    return None


_FALLBACK_MAP = {
    # Until PyTorch exposes official MXFP4/NVFP4 dtypes we default to BF16 so the
    # experimental pipeline can still be exercised end-to-end.
    "mxfp4": torch.bfloat16,
    "nvfp4": torch.bfloat16,
}


def resolve_precision(name: str, *, allow_fallback: bool = True) -> torch.dtype:
    """Resolve a textual precision name into a torch.dtype."""
    dtype = _lookup_dtype(name)
    if dtype is not None:
        return dtype

    canonical = name.lower()
    if allow_fallback:
        env_flag = os.environ.get("NANOCHAT_MXFP4_DISABLE_FALLBACK", "").lower()
        if env_flag not in ("1", "true", "yes"):
            fallback = _FALLBACK_MAP.get(canonical)
            if fallback is not None:
                print0(
                    f"[mxfp4] Requested precision '{name}' not available in this PyTorch build. "
                    f"Falling back to {fallback}."
                )
                return fallback

    raise PrecisionNotAvailableError(
        f"Requested precision '{name}' is not available in this PyTorch build. "
        "Upgrade to a version that ships MXFP4/NVFP4 kernels or build from source."
    )


@dataclass
class PrecisionConfig:
    """User provided precision preferences."""

    compute: str = "bfloat16"
    embedding: Optional[str] = None
    rotary: Optional[str] = None
    logits: str = "float32"
    autocast: Optional[str] = None
    allow_fallback: bool = True


@dataclass
class ResolvedPrecision:
    """Concrete torch.dtype objects derived from a PrecisionConfig."""

    compute: torch.dtype
    embedding: torch.dtype
    rotary: torch.dtype
    logits: torch.dtype
    autocast: torch.dtype


def resolve_precision_config(config: PrecisionConfig) -> ResolvedPrecision:
    """Resolve all precision strings in the provided config."""

    compute = resolve_precision(config.compute, allow_fallback=config.allow_fallback)
    embedding_name = config.embedding or config.compute
    rotary_name = config.rotary or config.compute
    logits_name = config.logits
    autocast_name = config.autocast or config.compute

    embedding = resolve_precision(embedding_name, allow_fallback=config.allow_fallback)
    rotary = resolve_precision(rotary_name, allow_fallback=config.allow_fallback)
    logits = resolve_precision(logits_name, allow_fallback=config.allow_fallback)
    autocast = resolve_precision(autocast_name, allow_fallback=config.allow_fallback)

    return ResolvedPrecision(
        compute=compute,
        embedding=embedding,
        rotary=rotary,
        logits=logits,
        autocast=autocast,
    )
