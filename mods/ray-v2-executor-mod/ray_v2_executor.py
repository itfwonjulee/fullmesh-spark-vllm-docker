# SPDX-License-Identifier: Apache-2.0
"""
RayV2 Executor for vLLM V1 Engine.

Replaces Ray Compiled DAG with direct ray.remote() calls + native NCCL
for pipeline-parallel inter-stage communication.

Solves the second-prompt deadlock caused by Compiled DAG timing constraints
on switchless RoCE mesh clusters with custom NCCL plugins.

Architecture:
  - Ray manages process placement and lifecycle (placement groups, actors)
  - Worker.execute_model() handles PP send/recv via NCCL internally
  - Worker.sample_tokens() handles token broadcast via NCCL internally
  - Executor dispatches via collective_rpc (ray.remote + ray.get)
  - NO Compiled DAG, NO Ray shared-memory channels on the hot path

Usage:
  --distributed-executor-backend vllm.v1.executor.ray_v2_executor.RayV2Executor

Reference: https://github.com/vllm-project/vllm/issues/35848
"""

import os
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.v1.executor.ray_executor import RayDistributedExecutor
from vllm.v1.executor.ray_utils import FutureWrapper, ray
from vllm.v1.outputs import ModelRunnerOutput

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput

logger = init_logger(__name__)


# ── Module-level callables for cloudpickle serialization ──────────────
# These are passed through collective_rpc → cloudpickle → execute_method
# on each Ray worker. The first arg `wrapper` is the RayWorkerWrapper
# (a WorkerWrapperBase subclass) injected by run_method().

def _safe_execute_model(wrapper: Any, scheduler_output: "SchedulerOutput") -> Any:
    """Execute forward pass with Ray-safe output conversion.

    WorkerWrapperBase.execute_model applies mm_cache then delegates to
    Worker.execute_model which handles PP send/recv via NCCL.
    """
    output = wrapper.execute_model(scheduler_output)
    # AsyncModelRunnerOutput holds CUDA events that cannot be pickled
    # across Ray process boundaries. Convert to plain ModelRunnerOutput.
    if hasattr(output, 'get_output'):
        output = output.get_output()
    return output


def _safe_sample_tokens(wrapper: Any, grammar_output: "GrammarOutput | None") -> Any:
    """Sample tokens with Ray-safe output conversion.

    Worker.sample_tokens handles PP broadcast of sampled tokens via NCCL.
    Non-last PP ranks receive tokens and return None.
    Last PP rank samples, broadcasts, and returns ModelRunnerOutput.
    """
    # sample_tokens is not on WorkerWrapperBase; delegate via worker directly
    output = wrapper.worker.sample_tokens(grammar_output)
    if hasattr(output, 'get_output'):
        output = output.get_output()
    return output


# ── RayV2 Executor ────────────────────────────────────────────────────

class RayV2Executor(RayDistributedExecutor):
    """Ray V2 Executor — direct ray.remote() dispatch, no Compiled DAG.

    Uses Ray for process placement and lifecycle across nodes, but replaces
    the Compiled DAG hot path with:
      • Direct ray.remote() calls for execute_model / sample_tokens
      • Native NCCL for PP intermediate tensor communication
      • Native NCCL for PP sampled token broadcast

    This eliminates the Ray Compiled DAG timing assumptions that cause
    deadlocks on high-latency or jittery interconnects (e.g., switchless
    RoCE mesh networks with custom NCCL net plugins).
    """

    uses_ray: bool = True
    supports_pp: bool = True

    def _init_executor(self) -> None:
        """Initialize — reuses parent's Ray cluster + worker setup,
        but never builds a Compiled DAG."""
        logger.info(
            "RayV2Executor: Initializing (direct ray.remote + NCCL, "
            "no Compiled DAG)"
        )
        # Parent sets up Ray cluster, placement group, creates workers,
        # initializes distributed env, loads model. Also sets:
        #   self.forward_dag = None
        #   self.has_connector, self.uses_sampler, self.scheduler_output
        super()._init_executor()

        # Cache the output rank (first worker in the last PP stage)
        pp = self.parallel_config.pipeline_parallel_size
        tp = self.parallel_config.tensor_parallel_size
        self._output_rank = (pp - 1) * tp

        logger.info(
            "RayV2Executor ready: PP=%d, TP=%d, output_rank=%d, "
            "num_workers=%d, has_connector=%s",
            pp, tp, self._output_rank, len(self.workers),
            self.has_connector,
        )

    # ── Hot path ──────────────────────────────────────────────────────

    def _execute_dag(
        self,
        scheduler_output: "SchedulerOutput",
        grammar_output: "GrammarOutput | None",
        non_block: bool = False,
    ) -> "ModelRunnerOutput | None | Future[ModelRunnerOutput | None]":
        """Override: execute via direct ray calls instead of Compiled DAG.

        Called by the parent's execute_model() and sample_tokens() methods
        which handle the two-phase (forward → sample) split logic.

        Flow:
          1. collective_rpc("execute_model") → all workers run forward pass
             PP communication (intermediate tensors) via NCCL internally
          2. If output_rank worker returned a result → return it (empty batch)
          3. Otherwise, collective_rpc("sample_tokens") → all workers sample
             PP broadcast of sampled tokens via NCCL internally
          4. Return ModelRunnerOutput from output_rank worker
        """
        result = self._execute_direct(scheduler_output, grammar_output)

        if non_block:
            # Wrap in a resolved Future for API compatibility.
            # Since we execute synchronously, the result is already available.
            future: Future[ModelRunnerOutput | None] = Future()
            future.set_result(result)
            return future
        return result

    def _execute_direct(
        self,
        scheduler_output: "SchedulerOutput",
        grammar_output: "GrammarOutput | None",
    ) -> "ModelRunnerOutput | None":
        """Synchronous execution: forward pass + optional sampling."""

        # Phase 1: Forward pass on all workers
        # Worker.execute_model() handles:
        #   - Non-first PP rank: irecv intermediate tensors via NCCL
        #   - Run model layers
        #   - Non-last PP rank: isend intermediate tensors via NCCL, return None
        #   - Last PP rank: save state for sample_tokens, return None
        #   - Empty batch: return EMPTY_MODEL_RUNNER_OUTPUT directly
        execute_outputs = self.collective_rpc(
            _safe_execute_model, args=(scheduler_output,)
        )

        result = execute_outputs[self._output_rank]
        if result is not None:
            # Direct result from output_rank (e.g., empty batch, pooling model)
            if self.has_connector:
                return self.kv_output_aggregator.aggregate(execute_outputs)
            return result

        # Phase 2: Sampling on all workers
        # Worker.sample_tokens() handles:
        #   - Last PP rank: sample tokens, broadcast via NCCL, return output
        #   - Non-last PP rank: receive broadcast, update state, return None
        sample_outputs = self.collective_rpc(
            _safe_sample_tokens, args=(grammar_output,)
        )

        if self.has_connector:
            return self.kv_output_aggregator.aggregate(sample_outputs)
        return sample_outputs[self._output_rank]

    # ── Scheduling ────────────────────────────────────────────────────

    @property
    def max_concurrent_batches(self) -> int:
        """RayV2 executes synchronously — one batch at a time.

        Without Compiled DAG, there is no PP micro-batch pipelining.
        Each _execute_direct call blocks until the full forward + sample
        completes across all PP stages. This is correct but does not
        overlap compute across PP stages for multiple batches.
        """
        return 1

    # ── Lifecycle ─────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Clean shutdown: kill Ray worker actors (no DAG to tear down)."""
        if not logger:
            return
        logger.info(
            "RayV2Executor: Shutting down. SIGTERM warnings from Ray "
            "logging.cc are expected — ignore them."
        )
        if hasattr(self, 'workers'):
            for worker in self.workers:
                ray.kill(worker)

    def __del__(self):
        self.shutdown()

    def check_health(self) -> None:
        """Verify all workers are responsive via a lightweight ping."""
        try:
            ray.get(
                [w.get_node_ip.remote() for w in self.workers],  # type: ignore
                timeout=60,
            )
        except Exception as e:
            raise RuntimeError(
                f"RayV2Executor health check failed: {e}"
            ) from e
