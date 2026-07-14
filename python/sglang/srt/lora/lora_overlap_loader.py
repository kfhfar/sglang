import logging
from enum import Enum, auto
from typing import Dict, Optional, Set

import torch
from torch.cuda import Event as CudaEvent
from torch.cuda import Stream as CudaStream
from torch.cuda import StreamContext as CudaStreamContext

from sglang.srt.lora.lora_manager import LoRAManager

logger = logging.getLogger(__name__)


class LoRAOverlapLoadStatus(Enum):
    LOADED = auto()
    LOADING = auto()
    NOT_LOADED = auto()


class LoRAOverlapLoader:
    def __init__(self, lora_manager):
        self.lora_manager: LoRAManager = lora_manager
        self.device_module = torch.get_device_module(self.lora_manager.device)
        self.load_stream: CudaStream = self.device_module.Stream()
        self.load_stream_context: CudaStreamContext = self.device_module.stream(
            self.load_stream
        )
        self.lora_to_overlap_load_event: Dict[Optional[str], CudaEvent] = {}

        # Track adapters whose pipelined loading has started (forward can proceed)
        self.pipelined_loading_loras: Set[Optional[str]] = set()

    def try_admit_overlap_lora(
        self, lora_id: Optional[str], running_loras: set[Optional[str]]
    ) -> bool:
        """
        Decides whether `lora_id` can join the batch currently being built.

        Returns True if the adapter is already resident / in flight, or there is
        capacity to admit it into this batch; False otherwise.
        """
        if lora_id in self.pipelined_loading_loras:
            # Verify adapter is still in GPU memory (not evicted)
            if lora_id in self.lora_manager.memory_pool.uid_to_buffer_id:
                return True
            self.pipelined_loading_loras.discard(lora_id)

        # Drain completed async loads before status/capacity checks so finished
        # adapters no longer count as in-flight.
        self._drain_completed_overlap_loads()

        status = self._check_overlap_load_status(lora_id)
        if status == LoRAOverlapLoadStatus.LOADING:
            return False
        if status == LoRAOverlapLoadStatus.LOADED:
            self.pipelined_loading_loras.discard(lora_id)
            return True

        # NOT_LOADED: gate on capacity against everything that will be resident
        # for this batch (already-admitted adapters live in `running_loras`,
        # which the scheduler grows as it admits, plus in-flight loads).
        prospective = (
            running_loras | self.lora_to_overlap_load_event.keys() | {lora_id}
        )
        return self.lora_manager.validate_lora_batch(prospective)

    def new_overlap_loads_lora(
        self, batch_loras: set[Optional[str]]
    ) -> None:
        """
        Issue a SINGLE layer-major transfer for every adapter admitted to the
        batch that is not already resident or in flight.

        Capacity was already validated per-adapter in `try_admit_overlap_lora`,
        so the combined set is guaranteed to fit; we only reissue the load here.
        """
        to_load = {
            lid
            for lid in batch_loras
            if lid not in self.lora_manager.memory_pool.uid_to_buffer_id
            and lid not in self.lora_to_overlap_load_event
        }
        if not to_load:
            return

        loras_to_be_loaded = (
            set(batch_loras) - to_load
        ) | self.lora_to_overlap_load_event.keys()

        with self.load_stream_context:
            self.lora_manager.fetch_new_loras(
                to_load,
                loras_to_be_loaded,
                loading_stream=self.load_stream,
            )
            event = self.device_module.Event()
            event.record(self.load_stream)

        # One completion event guards the whole layer-major transfer; each new
        # adapter references it so drain/eviction bookkeeping stays per-adapter.
        for lid in to_load:
            self.lora_to_overlap_load_event[lid] = event
            self.pipelined_loading_loras.add(lid)
        logger.debug(f"Layer-major loading LoRA adapters {to_load} asynchronously")

    def _check_overlap_load_status(
        self, lora_id: Optional[str]
    ) -> LoRAOverlapLoadStatus:
        if lora_id in self.lora_to_overlap_load_event:
            return LoRAOverlapLoadStatus.LOADING

        # After completed events have been drained, a memory-pool entry with no
        # pending event is safe to use on the current stream.
        if lora_id in self.lora_manager.memory_pool.uid_to_buffer_id:
            return LoRAOverlapLoadStatus.LOADED

        return LoRAOverlapLoadStatus.NOT_LOADED

    def _drain_completed_overlap_loads(self) -> None:
        completed_loads = [
            (lora_id, event)
            for lora_id, event in self.lora_to_overlap_load_event.items()
            if event.query()
        ]
        for lora_id, event in completed_loads:
            torch.cuda.current_stream().wait_event(event)
            del self.lora_to_overlap_load_event[lora_id]
