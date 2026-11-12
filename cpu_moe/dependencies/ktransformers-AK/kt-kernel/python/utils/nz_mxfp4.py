"""Borrow the offload manager's final NZ bytes; never load or repack weights."""
from dataclasses import dataclass

import torch

from ..experts_base import BaseMoEWrapper
from .. import kt_kernel_ext


@dataclass(frozen=True)
class NZWeights:
    w13: tuple
    w2: tuple
    scale13: tuple
    scale2: tuple

    @property
    def fields(self):
        return (self.w13, self.w2, self.scale13, self.scale2)

    def validate(self, experts, hidden, intermediate):
        if hidden % 32 or intermediate % 32:
            raise ValueError("NZ MXFP4 dimensions must be divisible by 32")
        sizes = (hidden * intermediate, hidden * intermediate // 2,
                 2 * intermediate * hidden // 32, hidden * intermediate // 32)
        for field, size in zip(self.fields, sizes):
            if len(field) != experts:
                raise ValueError("NZ expert source count mismatch")
            for tensor in field:
                if tensor.device.type != "cpu" or tensor.dtype != torch.uint8 or tensor.numel() != size:
                    raise ValueError("NZ backend requires equal-sized borrowed CPU uint8 sources")
                if tensor.storage_offset() + size > tensor.untyped_storage().nbytes():
                    raise ValueError("NZ source is outside its backing storage")
        # The logical scale view is [G/2,N,2], backed by row-major [N,G].
        # Reject a separately packed contiguous scale slab: its byte indexing differs.
        for field, n, groups in ((self.scale13, 2 * intermediate, hidden // 32),
                                  (self.scale2, hidden, intermediate // 32)):
            for tensor in field:
                if tuple(tensor.shape) != (groups // 2, n, 2) or tuple(tensor.stride()) != (2, groups, 1):
                    raise ValueError("NZ CPU requires the single final-stride scale buffer")
                if bool((tensor == 255).any()):
                    raise ValueError("Reserved E8M0 scale 255 in NZ source")
        return self

    def pointers(self):
        return [[tensor.data_ptr() for tensor in field] for field in self.fields]


class NZMoEWrapper(BaseMoEWrapper):
    def __init__(self, *, native_weights, **kwargs):
        if kwargs["threadpool_count"] != 1:
            raise ValueError("NZ CPU requires one NUMA pool")
        if kwargs.get("max_deferred_experts_per_token", 0):
            raise ValueError("NZ CPU does not support deferred experts")
        if not hasattr(kt_kernel_ext.moe, "NZMOE"):
            raise RuntimeError("Rebuild the CPU backend with shared NZ support")
        self.native_weights = native_weights.validate(kwargs["num_experts"], kwargs["hidden_size"],
                                                       kwargs["moe_intermediate_size"])
        super().__init__(method="LLAMAFILE_NZ", **kwargs)
        self.weights_to_keep = native_weights

    def load_weights(self, physical_to_logical_map_cpu=None):
        if self.moe is not None:
            raise RuntimeError("NZ weights are immutable after registration")
        if physical_to_logical_map_cpu is not None:
            expected = torch.arange(self.num_experts, dtype=torch.int32)
            if not torch.equal(physical_to_logical_map_cpu, expected):
                raise ValueError("NZ CPU requires logical expert order")
        self.moe = kt_kernel_ext.moe.NZMOE(
            self.cpu_infer.backend_, self.layer_idx, self.hidden_size, self.moe_intermediate_size,
            self.num_experts_per_tok, self.chunked_prefill_size, self.native_weights.pointers())
        if self.moe.weight_pointers() != self.native_weights.pointers():
            raise RuntimeError("NZ CPU is not borrowing the DMA source bytes")
        print(f"[CPU-NZ] layer={self.layer_idx} experts={self.num_experts} "
              "layout=MXFP4_NZ shared_dma_sources=True weight_copies=0 "
              "dispatch=CPP_FLAG compute=CPP result=CPP_EVENT_MAPPED_HOST", flush=True)

    def load_weights_from_tensors(self, *args, **kwargs):
        raise RuntimeError("NZ CPU must borrow the finalized offload sources")
