from __future__ import annotations

import time
from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch
from minisgl.core import Batch, Req
from minisgl.engine import ForwardOutput
from minisgl.env import ENV
from minisgl.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    DetokenizeMsg,
    ExitMsg,
    UserMsg,
)
from minisgl.utils import init_logger, load_tokenizer

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .prefill import ChunkedReq, PrefillManager
from .table import TableManager

if TYPE_CHECKING:
    from minisgl.engine import BatchSamplingArgs


logger = init_logger(__name__)

Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]


# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D  # (token_mapping, positions)
    write_tuple: Indice2D  # (req_mapping, seq_lens or -1)


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        from minisgl.engine import Engine

        self.engine = Engine(config)

        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)

        # initialize other managers
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        self.cache_manager = CacheManager(
            self.engine.num_pages, config.page_size, self.engine.page_table, config.cache_type
        )
        self.decode_manager = DecodeManager(config.page_size)
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )
        # GDN (linear-attention) models keep a single global recurrent state
        # that must be zeroed at the start of every new request. The prefill
        # adder calls ctx.request_gdn_reset() when it adds a NEW request
        # (first prefill chunk); Context.forward_batch consumes the flag.
        self.prefill_manager.ctx = self.engine.ctx

        # some alias for easy access
        self.finished_reqs: Set[Req] = set()
        self.tokenizer = load_tokenizer(config.model_path)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.token_pool = self.table_manager.token_pool
        self.prefill_budget = config.max_extend_tokens
        # self.config = config

        # Initialize the I/O mixin
        super().__init__(config, self.engine.tp_cpu_group)

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        blocking = not (
            last_data is not None  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(last_data)
        return ongoing_data

    def normal_loop(self) -> None:
        blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data)

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        # Spec-decode emits a variable number of tokens (L+1) per step, which
        # breaks the overlap loop's 1-iteration lag assumption (iteration i's
        # forward would re-read the position iteration i-1 already consumed).
        # Use the normal loop when the spec step is active.
        use_overlap = not ENV.DISABLE_OVERLAP_SCHEDULING and self.engine.config.mtp_n_max == 0
        if use_overlap:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)
        else:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()

    def shutdown(self) -> None:
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            return

        forward_input, forward_output = last_data
        batch = forward_input.batch
        next_tokens_cpu = forward_output.next_tokens_cpu
        copy_done = forward_output.copy_done_event
        spec_emit_counts = forward_output.spec_emit_counts
        copy_done.synchronize()
        reply: List[DetokenizeMsg] = []
        new_finished_reqs: Set[Req] = set()
        with self.cache_manager.lazy_free_region():
            if spec_emit_counts:
                # Spec-decode step: each req emits L+1 tokens (x_t + d_1..d_L).
                # The engine did NOT call complete_one; the scheduler advances
                # the Req by L+1 here.
                offset = 0
                for i, req in enumerate(batch.reqs):
                    if isinstance(req, ChunkedReq):
                        continue
                    emit_count = spec_emit_counts[i]
                    emitted = next_tokens_cpu[offset : offset + emit_count]
                    offset += emit_count
                    import sys as _sys
                    print(f"[spec-debug] _process_last_data req{i} emit_count={emit_count} "
                          f"emitted_len={len(emitted)} next_tokens_cpu_len={len(next_tokens_cpu)}", file=_sys.stderr)
                    # Append all L+1 tokens to the req's host input_ids.
                    req.append_host(emitted.to(torch.int32))
                    # Advance the Req by L+1 (not 1).
                    req.cached_len += emit_count
                    req.device_len += emit_count
                    finished = not req.can_decode
                    if not req.sampling_params.ignore_eos:
                        for t in emitted:
                            finished |= int(t.item()) == self.eos_token_id
                    for j in range(emit_count):
                        reply.append(DetokenizeMsg(
                            uid=req.uid,
                            next_token=int(emitted[j].item()),
                            finished=(finished and j == emit_count - 1),
                        ))
                    if finished and req not in self.finished_reqs:
                        self.decode_manager.remove_req(req)
                        self._free_req_resources(req)
                        new_finished_reqs.add(req)
            else:
                # Normal step: each req emits 1 token.
                for i, req in enumerate(batch.reqs):
                    if isinstance(req, ChunkedReq):
                        continue
                    next_token = next_tokens_cpu[i]
                    req.append_host(next_token.unsqueeze(0))
                    next_token = int(next_token.item())
                    finished = not req.can_decode
                    if not req.sampling_params.ignore_eos:
                        finished |= next_token == self.eos_token_id
                    reply.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))
                    # NOTE: overlap scheduling may make the request freed twice, skip second free
                    if finished and req not in self.finished_reqs:
                        self.decode_manager.remove_req(req)
                        self._free_req_resources(req)
                        new_finished_reqs.add(req)
                    elif batch.is_prefill:  # for prefill, non-chunk req, cache the prefix
                        self.cache_manager.cache_req(req, finished=False)

        self.finished_reqs = new_finished_reqs
        self.send_result(reply)

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            max_output_len = max_seq_len - input_len
            if max_output_len <= 0:
                return logger.warning_rank0(
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            self.prefill_manager.add_one_req(msg)
        elif isinstance(msg, AbortBackendMsg):
            logger.debug_rank0("Aborting request %d", msg.uid)
            req_to_free = self.prefill_manager.abort_req(msg.uid)
            req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
            if req_to_free is not None:
                self._free_req_resources(req_to_free)
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _free_req_resources(self, req: Req) -> None:
        self.table_manager.free(req.table_idx)
        self.cache_manager.cache_req(req, finished=True)

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        self.engine.graph_runner.pad_batch(batch)
        self.cache_manager.allocate_paged(batch.reqs)
        batch.positions = _make_positions(batch, self.device)
        input_mapping = _make_input_tuple(batch, self.device)
        write_mapping = _make_write_tuple(batch, self.device)
        batch.out_loc = self.engine.page_table[input_mapping]
        self.engine.attn_backend.prepare_metadata(batch)
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def _schedule_next_batch(self) -> ForwardInput | None:
        # TODO: support other policies: e.g. DECODE first
        batch = (
            self.prefill_manager.schedule_next_batch(self.prefill_budget)
            or self.decode_manager.schedule_next_batch()
        )
        import sys as _sys
        if batch is not None:
            reqs_info = [(r.cached_len, r.device_len, r.extend_len) for r in batch.reqs]
            print(f"[sched-debug] phase={batch.phase} reqs={batch.size} "
                  f"(cached,dev,ext)={reqs_info} "
                  f"pending={len(self.prefill_manager.pending_list)} "
                  f"decode_run={len(self.decode_manager.running_reqs)}", file=_sys.stderr)
        return self._prepare_batch(batch) if batch else None

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        batch, sample_args, input_mapping, output_mapping = forward_input
        batch.input_ids = self.token_pool[input_mapping]
        started = time.perf_counter()
        if self.engine.config.mtp_n_max > 0 and batch.is_decode:
            forward_output = self._forward_spec(batch, sample_args)
        else:
            forward_output = self.engine.forward_batch(batch, sample_args)
            self.token_pool[output_mapping] = forward_output.next_tokens_gpu
        step_ms = (time.perf_counter() - started) * 1000.0
        self.decode_manager.filter_reqs(forward_input.batch.reqs)
        self._record_step(batch, step_ms)
        return forward_output

    def _forward_spec(self, batch: Batch, sample_args: BatchSamplingArgs) -> ForwardOutput:
        """Drive the two-forward spec-decode step: forward A (draft) + forward B (verify).

        Forward A: the normal decode forward (input at position N) samples x_t at
        position N+1, then the MTP draft chain produces d_1..d_K for positions
        N+2..N+K+1. The K+1 tokens [x_t, d_1..d_K] are scattered into the token
        pool at positions N+1..N+K+1.

        Forward B: the verification forward (phase="prefill", K+1 tokens) reads
        the K+1 tokens from the token pool, runs the full model, and accepts the
        matching prefix of draft tokens. It emits L+1 tokens (x_t + d_1..d_L)
        per request. The bonus (v_K, the logit prediction for d_K) is NEVER
        emitted -- it is the output of the NEXT forward A, which consumes d_L at
        position N+L+1 and generates the bonus at N+L+2.

        The engine does NOT call complete_one for the spec step; the scheduler
        advances the Req by L+1 in _process_last_data via spec_emit_counts.
        """
        n_max = self.engine.config.mtp_n_max
        k_plus_1 = n_max + 1
        assert len(batch.reqs) == 1, "spec-decode currently requires concurrency 1"
        req = batch.reqs[0]
        # The draft chain writes the MTP KV slots N+1..N+K-1 through the page
        # table, but none of them is allocated yet: forward A's allocate_paged
        # covers only slot N, and the verify batch allocates N+1..N+K+1 AFTER
        # the chain has already run. Pre-allocate the chain's slots transiently
        # and release them once the draft completes (the verify batch allocates
        # them again immediately after; the freed pages are handed back to the
        # pool before that allocation, and the chain's GPU writes to those pages
        # are already enqueued on the stream, so no race).
        if n_max > 1:
            old_cached = req.cached_len
            old_device = req.device_len
            req.cached_len = old_cached + 1  # N+1
            req.device_len = old_cached + n_max  # N+K
            self.cache_manager.allocate_paged([req])
            req.cached_len, req.device_len = old_cached, old_device  # restore exactly
        # Forward A: the normal decode forward + the MTP draft chain.
        draft_output = self.engine.forward_mtp_draft(batch, sample_args)
        draft_output.copy_done_event.synchronize()
        if n_max > 1:
            self.cache_manager._free(
                self.engine.page_table[
                    req.table_idx, old_cached + 1 : old_cached + n_max
                ]
            )
        tokens_gpu = draft_output.next_tokens_gpu  # [K+1]
        # Scatter the K+1 tokens into the token pool at positions N+1..N+K+1.
        base = req.cached_len + 1  # position of x_t = N+1
        positions = torch.arange(base, base + k_plus_1, device=self.device)
        self.token_pool[req.table_idx, positions] = tokens_gpu
        # Temporarily extend the req so the verify batch allocates K+1 tokens
        # of KV (positions N+1..N+K+1).
        old_cached_len = req.cached_len
        old_device_len = req.device_len
        req.cached_len = old_cached_len + 1  # N+1
        req.device_len = old_cached_len + k_plus_1 + 1  # N+K+2
        try:
            verify_batch = Batch(reqs=[req], phase="prefill")
            verify_input = self._prepare_batch(verify_batch)
            verify_batch.input_ids = self.token_pool[verify_input.input_tuple]
            verify_output = self.engine.forward_mtp_verify(verify_batch, sample_args)
        finally:
            req.cached_len = old_cached_len
            req.device_len = old_device_len
        emit_count = len(verify_output.next_tokens_cpu)
        import sys as _sys
        print(f"[spec-debug] verify_batch.input_ids.shape={tuple(verify_batch.input_ids.shape)} "
              f"emit_count={emit_count} n_max={n_max} k_plus_1={k_plus_1}", file=_sys.stderr)
        # Free the rejected-draft KV. The verify batch allocated K+1 pages
        # (positions N+1..N+K+1) but only emit_count (L+1) tokens are emitted;
        # the req only advances by emit_count, so cache_req (which frees up to
        # req.cached_len) would never free the rejected tail (N+emit_count+1 ..
        # N+K+1). Left alone, each spec step leaks ~K pages and the CacheManager
        # integrity check (free + cached == num_pages) trips after a few steps.
        # page_size is 1, so _free token indices are page indices.
        #
        # There is a SECOND, smaller leak: _process_last_data advances
        # cached_len by emit_count (L+1) -> N+L+1, which covers positions
        # 0..N+L. The emitted token d_L lives at position N+L+1, so its page
        # (allocated by the verify batch) is beyond cached_len and is never
        # freed by cache_req either. That is 1 leaked page per spec step.
        # Free it here, immediately after the verify, before the req is
        # advanced. The next spec step's decode forward reprocesses d_L from
        # the GDN snapshot and recomputes its KV (the main model is causal),
        # so dropping the page is safe.
        if emit_count < k_plus_1:
            rejected = self.engine.page_table[
                req.table_idx,
                old_cached_len + emit_count + 1 : old_cached_len + k_plus_1 + 1,
            ]
            self.cache_manager._free(rejected)
        # The last accepted token's page (position N+emit_count) is allocated
        # but not cached (cached_len = N+emit_count covers 0..N+emit_count-1).
        last_tok_page = self.engine.page_table[
            req.table_idx, old_cached_len + emit_count
        ]
        self.cache_manager._free(last_tok_page.unsqueeze(0))
        return ForwardOutput(
            verify_output.next_tokens_gpu,
            verify_output.next_tokens_cpu,
            verify_output.copy_done_event,
            spec_emit_counts=(emit_count,),
        )

    def _record_step(self, batch: Batch, step_ms: float) -> None:
        from minisgl.distributed import get_tp_info
        from minisgl.step_ring import append_step

        if not get_tp_info().is_primary():
            return
        prefill_tokens = 0
        radix_hit_tokens = 0
        if batch.is_prefill:
            prefill_tokens = sum(req.extend_len for req in batch.reqs)
            radix_hit_tokens = sum(req.cached_len for req in batch.reqs)
        graph_hit = bool(
            batch.is_decode and self.engine.graph_runner.can_use_cuda_graph(batch)
        )
        append_step({
            "prefill_tokens": prefill_tokens,
            "decode_batch": batch.size if batch.is_decode else 0,
            "graph_hit": graph_hit,
            "radix_hit_tokens": radix_hit_tokens,
            "step_ms": round(step_ms, 3),
            "queue_depth": len(self.prefill_manager.pending_list),
        })


def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    needed_size = sum(r.extend_len for r in batch.padded_reqs)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        torch.arange(
            req.cached_len,
            req.device_len,
            dtype=torch.int32,
            out=indices_host[offset : offset + length],
        )
        offset += length
    return indices_host.to(device, non_blocking=True)


def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_host = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        mapping_host[offset : offset + length].fill_(req.table_idx)
        offset += length
    return mapping_host.to(device, non_blocking=True), batch.positions.to(torch.int64)


def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_list = [req.table_idx for req in batch.reqs]
    mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)
    write_list = [(req.device_len if req.can_decode else -1) for req in batch.reqs]
    write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)
    return mapping_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)
