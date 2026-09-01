"""Run pi0.5 inference with data parallelism (DP=N, TP=1).

The ``pi05_wn`` plugin loads a complete copy of the model on every GPU. Rank 0
splits the input batch into chunks of ``ceil(batch_size / dp)`` rows and sends
one chunk to each GPU. It then gathers the action chunks into a
``(batch, chunk, action_dim)`` tensor, which is kept on rank 0.

Use one torchrun process per GPU. ``--nproc_per_node`` must match ``--dp``::

    torchrun --nproc_per_node=8 examples/pi05/run_pi05_wn.py --dp 8 \\
        --checkpoint /path/to/pi05_base/ --batch-size 32

The script creates random tensors in the shapes expected by the model, so the
returned actions have no practical meaning. Use it to check the data-parallel
path and compare warmup and inference times. CUDA and NCCL are required.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import time

import torch

from phyai.utils import get_logger


logger = get_logger(__name__)


def _resolve_topology(dp: int) -> tuple[int, int, bool]:
    """Reconcile ``--dp`` with the torchrun launch env.

    Returns ``(local_rank, dp, is_main)``. World size == dp (one process/rank).
    ``dp == 1`` runs in-process (no torchrun needed). Rank 0 is the Router and
    holds the gathered result.
    """
    world = dp
    env_world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1 and env_world != world:
        raise SystemExit(
            f"--dp {dp} requires torchrun --nproc_per_node={world} "
            f"(saw WORLD_SIZE={env_world}). Example:\n"
            f"  torchrun --nproc_per_node={world} examples/pi05/run_pi05_wn.py "
            f"--dp {dp} --checkpoint <ckpt> --batch-size 32"
        )
    if world == 1 and env_world != 1:
        raise SystemExit(
            f"launched under torchrun (WORLD_SIZE={env_world}) but --dp is 1; "
            f"set --dp to use all ranks."
        )
    return local_rank, dp, local_rank == 0


@contextlib.contextmanager
def _timed(label: str, store: dict):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        store[label] = time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True, help="pi05_base checkpoint dir")
    parser.add_argument(
        "--dp",
        type=int,
        default=1,
        help="Data-parallel degree; world_size = dp must equal torchrun --nproc_per_node.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Total robots across all ranks (split ceil(batch/dp) per card).",
    )
    parser.add_argument("--num-images", type=int, default=3)
    parser.add_argument(
        "--vision-dtype", choices=("bfloat16", "float32"), default="bfloat16"
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")

    from phyai.engine import Engine, EngineArgs
    from phyai.engine_config import (
        DeviceConfig,
        EngineConfig,
        ParallelConfig,
        RuntimeConfig,
    )
    from phyai.models.pi05.configuration_pi05 import PI05Config
    from phyai.models.pi05.main_pi05_wn import PI05WNArgs
    from phyai.models.pi05.scheduler_ws1_pi05 import PI05Request
    from phyai.utils import load_config

    local_rank, dp_size, is_main = _resolve_topology(args.dp)
    device = f"cuda:{local_rank}"
    dtype = torch.bfloat16
    vision_dtype = torch.float32 if args.vision_dtype == "float32" else None

    plugin_cfg = load_config(args.checkpoint, PI05Config)
    inputs_image_shape = [
        [plugin_cfg.vision.image_size, plugin_cfg.vision.image_size, 3]
        for _ in range(args.num_images)
    ]

    engine = Engine(
        EngineArgs(
            plugin="pi05_wn",
            plugin_args=PI05WNArgs(
                checkpoint_dir=args.checkpoint,
                max_batch_size=args.batch_size,
                vision_params_dtype=vision_dtype,
                inputs_image_shape=inputs_image_shape,
            ),
            config=EngineConfig(
                device=DeviceConfig(target=device, params_dtype=dtype),
                parallel=ParallelConfig(world_size=dp_size, dp_size=dp_size, tp_size=1),
                runtime=RuntimeConfig(use_cuda_graph=True),
            ),
        )
    )
    logger.info_rank0("[engine] created pi0.5 data-parallel engine (dp=%d).", dp_size)

    timings: dict[str, float] = {}
    try:
        # Every rank builds a full canonical request; only rank 0's rows are
        # scattered (others' copies are ignored — they recv their shard).
        B = args.batch_size
        img = plugin_cfg.vision.image_size
        request = PI05Request(
            pixel_values=torch.rand(
                B, args.num_images, 3, img, img, dtype=dtype, device=device
            ),
            input_ids=torch.zeros(
                B, plugin_cfg.tokenizer_max_length, dtype=torch.int64, device=device
            ),
            lang_lens=torch.ones(B, dtype=torch.int64, device=device),
        )
        request.input_ids[:, 0] = 2

        with _timed("warmup", timings):
            engine.step(request)
        with _timed("inference", timings):
            actions = engine.step(request)

        logger.info_rank0("[run] dp=%d total_batch=%d", dp_size, B)
        logger.info_rank0("action chunk shape : %s", tuple(actions.shape))
        logger.info_rank0("action chunk device: %s", actions.device)
        logger.info_rank0("first action row   : %s", actions[0, 0].float().tolist())
        logger.info_rank0(
            "timing (s): warmup=%.2f inference=%.2f",
            timings["warmup"],
            timings["inference"],
        )
    finally:
        engine.close()


if __name__ == "__main__":
    main()
