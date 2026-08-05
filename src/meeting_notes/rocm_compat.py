"""Compatibility helpers for AMD's native-Windows ROCm PyTorch build."""

# Dynamic modules intentionally gain the attributes imported by Transformers.
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import sys
import types
from typing import Any


def install_windows_rocm_transformers_compatibility(torch: Any) -> None:
    """Stub optional distributed APIs that native-Windows ROCm omits.

    Transformers imports these modules even for single-GPU inference.  The
    stubs make that import path available without pretending distributed
    execution itself is supported.
    """
    distributed = getattr(torch, "distributed", None)
    if distributed is not None and distributed.is_available():
        return

    fsdp = types.ModuleType("transformers.distributed.fsdp")

    def is_fsdp_managed_module(_module: object) -> bool:
        return False

    def is_fsdp_enabled() -> bool:
        return False

    def get_fsdp_ckpt_kwargs() -> dict[str, object]:
        return {}

    def update_fsdp_plugin_peft(*_args: object, **_kwargs: object) -> None:
        return None

    fsdp.is_fsdp_managed_module = is_fsdp_managed_module
    fsdp.is_fsdp_enabled = is_fsdp_enabled
    fsdp.get_fsdp_ckpt_kwargs = get_fsdp_ckpt_kwargs
    fsdp.update_fsdp_plugin_peft = update_fsdp_plugin_peft
    sys.modules[fsdp.__name__] = fsdp

    sharding = types.ModuleType("transformers.distributed.sharding_utils")

    class DtensorShardOperation:
        pass

    def distributed_tensors_unavailable(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("Distributed tensors are unavailable in this PyTorch build.")

    sharding.DtensorShardOperation = DtensorShardOperation
    sharding._dtensor_from_local_like = distributed_tensors_unavailable
    sys.modules[sharding.__name__] = sharding
