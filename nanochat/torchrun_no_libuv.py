"""Torchrun wrapper that forces libuv off on local TCPStore."""

from __future__ import annotations

import os
import sys
from importlib import import_module

# Ensure the environment flag is persisted for any downstream helpers that do check it.
os.environ["USE_LIBUV"] = "0"

# Patch torch.distributed TCPStore constructors to default to use_libuv=False.
import torch.distributed as dist  # noqa: E402
from torch.distributed.elastic.multiprocessing import (
    start_processes as _orig_start_processes,
)
from torch.distributed.elastic.agent.server import local_elastic_agent

_original_tcp_store = dist.TCPStore


def _tcp_store_no_libuv(*args, **kwargs):  # pragma: no cover - thin wrapper
    kwargs["use_libuv"] = False
    if "multi_tenant" in kwargs:
        kwargs["multi_tenant"] = False
    return _original_tcp_store(*args, **kwargs)


dist.TCPStore = _tcp_store_no_libuv  # type: ignore[assignment]

# Also patch the rendezvous backend which cached the symbol during import.
c10d_backend = import_module("torch.distributed.elastic.rendezvous.c10d_rendezvous_backend")
setattr(c10d_backend, "TCPStore", dist.TCPStore)
rendezvous_mod = import_module("torch.distributed.rendezvous")
setattr(rendezvous_mod, "TCPStore", dist.TCPStore)


def _start_processes_no_libuv(*args, **kwargs):  # pragma: no cover - env tweak
    args = list(args)
    envs = kwargs.get("envs")
    positional = False
    if envs is None and len(args) >= 4:
        envs = args[3]
        positional = True
    if envs is not None:
        patched = {}
        for rank, env in envs.items():
            patched_env = dict(env)
            patched_env['USE_LIBUV'] = '0'
            patched[rank] = patched_env
        if positional:
            args[3] = patched
        else:
            kwargs["envs"] = patched
    return _orig_start_processes(*args, **kwargs)


elastic_mp = import_module("torch.distributed.elastic.multiprocessing")
elastic_mp.start_processes = _start_processes_no_libuv  # type: ignore[attr-defined]
local_elastic_agent.start_processes = _start_processes_no_libuv  # type: ignore[attr-defined]



def main() -> int:  # pragma: no cover - delegated to torch.distributed.run
    from torch.distributed.run import main as torchrun_main

    return torchrun_main()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
