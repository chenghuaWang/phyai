"""Discrete ``torch.distributed`` bootstrap entry point.

:func:`init_dist` is the single function the engine calls to bring up
the process group. Splitting it out of :class:`~phyai.engine.Engine`
keeps the orchestration shallow (Engine just decides *whether* to call
this; it doesn't own the bootstrap logic) and lets advanced users /
tests reach for the same routine standalone.

Single-rank short-circuit
-------------------------
When ``world_size == 1`` and no caller-owned process group exists, no
``init_process_group`` is issued — every collective in
:mod:`phyai.parallel.ops` short-circuits at world_size=1, so a real
group would be ceremony. The function returns ``False`` (we don't own
a process group) and the caller skips the matching teardown.

Device before group
-------------------
``torch.cuda.set_device`` runs *before* ``init_process_group``, not
after. NCCL binds to the current device as it builds its communicator,
and anything allocated between the two calls lands on whatever device
was current — so a late ``set_device`` means the group and the tensors
can disagree about which GPU this rank owns.
"""

from __future__ import annotations

import os
from datetime import timedelta

import torch
import torch.distributed as dist

from phyai.utils.cuda import resolve_device


def init_dist(
    *,
    world_size: int,
    device_type: str,
    timeout: timedelta | None = None,
) -> bool:
    """Bring up the process group for the requested ``world_size``.

    ``world_size`` is the total rank count for the global mesh — the
    product of every parallel axis (``dp * ep * sp * cp * tp``), not
    one specific axis. The caller (typically :class:`~phyai.engine.Engine`)
    has already done the multiplication.

    Args:
        world_size: total rank count for the global mesh.
        device_type: ``"cuda"`` / ``"cpu"``; picks the backend and decides
            whether a device gets pinned.
        timeout: collective timeout handed to ``init_process_group``.
            ``None`` keeps torch's own default. Engine passes
            :attr:`~phyai.engine_config.RuntimeConfig.dist_timeout_s`.

    Returns
    -------
    bool
        ``True`` if this call **owns** the process group (created it
        and is responsible for ``dist.destroy_process_group()`` on
        shutdown). ``False`` if a group was already up (e.g. under
        ``torchrun``) or single-rank short-circuited.

    Behaviour
    ---------
    * ``world_size == 1`` with no existing group -> no-op, returns ``False``.
    * ``world_size > 1`` with no existing group -> spin one up
      (``nccl`` for cuda, ``gloo`` for cpu) and return ``True``.
    * Any ``world_size`` with an existing group -> reuse it; raise
      :class:`ValueError` if its world size disagrees with the request.
    """
    if not dist.is_initialized():
        if world_size == 1:
            if device_type == "cuda":
                torch.cuda.set_device(resolve_device("cuda"))
            return False

        backend = "nccl" if device_type == "cuda" else "gloo"
        # Honour a launcher (torchrun / torchelastic) when it has populated the
        # rendezvous env: each process then has its OWN rank / local_rank. Falling
        # back to rank 0 / world_size only covers the single-process spin-up case
        # (one process creating the whole group). Reading RANK here is essential —
        # hardcoding rank=0 makes every torchrun process claim rank 0 and the NCCL
        # rendezvous deadlocks on the first collective.
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", str(world_size))
        os.environ.setdefault("LOCAL_RANK", "0")
        rank = int(os.environ["RANK"])
        # Pin the device first: NCCL reads the current device while building
        # its communicator, so binding after init_process_group leaves a
        # window where the group and this rank's allocations disagree.
        if backend == "nccl":
            torch.cuda.set_device(resolve_device("cuda"))
        kwargs = {} if timeout is None else {"timeout": timeout}
        dist.init_process_group(backend, rank=rank, world_size=world_size, **kwargs)
        return True

    actual = dist.get_world_size()
    if actual != world_size:
        raise ValueError(
            f"init_dist: world_size={world_size} does not match the existing "
            f"process-group world_size={actual}. Either set the parallel sizes "
            f"to match the launcher, or destroy the current process group first."
        )
    return False


__all__ = ["init_dist"]
