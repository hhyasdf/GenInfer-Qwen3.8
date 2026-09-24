from __future__ import annotations

import logging
import multiprocessing as mp
import os
import sys
import threading
import time
from dataclasses import replace
from typing import TYPE_CHECKING

from minisgl.distributed import DistributedInfo
from minisgl.utils import init_logger

if TYPE_CHECKING:
    from .args import ServerArgs


def _watch_workers(procs: list[mp.Process], logger) -> None:
    """Exit the main process when any backend worker dies.

    A crashed scheduler leaves the API server up but unable to serve:
    requests hang, and a supervisor boot gate polling ``kill -0`` waits for
    the whole deadline. The engine dies with its workers, so a crash fails
    fast. The watchdog starts before the readiness-ack wait, so a crash
    during startup (weight load, kernel JIT) also fails fast instead of
    blocking the ack loop forever.
    """

    def _loop() -> None:
        while True:
            for p in procs:
                if not p.is_alive():
                    logger.error(
                        "Backend worker %s died (exitcode=%s); terminating the engine",
                        p.name,
                        p.exitcode,
                    )
                    for q in procs:
                        if q.is_alive():
                            q.terminate()
                    time.sleep(1)
                    os._exit(1)
            time.sleep(1)

    threading.Thread(target=_loop, name="minisgl-worker-watchdog", daemon=True).start()


def _run_scheduler(args: ServerArgs, ack_queue: mp.Queue[str]) -> None:
    import torch
    from minisgl.scheduler import Scheduler

    with torch.inference_mode():
        scheduler = Scheduler(args)
        scheduler.sync_all_ranks()

        if args.tp_info.is_primary():
            ack_queue.put("Scheduler is ready")

        if args.silent_output:
            logging.disable(logging.INFO)

        try:
            scheduler.run_forever()
        except KeyboardInterrupt:
            logger = init_logger(__name__)
            if args.tp_info.is_primary():
                print()  # for a clean newline after ^C
                logger.info("Scheduler exiting gracefully...")
            scheduler.shutdown()


def start_subprocess(server_args: ServerArgs, logger) -> None:
    """Spawn the backend worker processes (TP schedulers + tokenizer/detokenizer)
    and wait for their readiness acknowledgments.

    Shared by ``launch_server`` (the normal server path) and ``minisgl.bench``
    (the in-process benchmark path) so both exercise the identical process
    model.
    """
    import multiprocessing as mp

    from minisgl.tokenizer import tokenize_worker

    mp.set_start_method("spawn", force=True)

    world_size = server_args.tp_info.size
    # a multiprocessing queue to receive ack from subprocesses
    # so that we can guarantee all subprocesses are ready
    ack_queue: mp.Queue[str] = mp.Queue()

    # Keep the handles: the watchdog below must see every worker.
    procs: list[mp.Process] = []
    for i in range(world_size):
        new_args = replace(
            server_args,
            tp_info=DistributedInfo(i, world_size),
        )
        p = mp.Process(
            target=_run_scheduler,
            args=(new_args, ack_queue),
            daemon=False,
            name=f"minisgl-TP{i}-scheduler",
        )
        p.start()
        procs.append(p)

    num_tokenizers = server_args.num_tokenizer
    # DeTokenizer, only 1
    p = mp.Process(
        target=tokenize_worker,
        kwargs={
            "tokenizer_path": server_args.model_path,
            "addr": server_args.zmq_detokenizer_addr,
            "backend_addr": server_args.zmq_backend_addr,
            "frontend_addr": server_args.zmq_frontend_addr,
            "local_bs": 1,
            "create": server_args.tokenizer_create_addr,
            "tokenizer_id": num_tokenizers,
            "ack_queue": ack_queue,
        },
        daemon=False,
        name="minisgl-detokenizer-0",
    )
    p.start()
    procs.append(p)
    for i in range(num_tokenizers):
        p = mp.Process(
            target=tokenize_worker,
            kwargs={
                "tokenizer_path": server_args.model_path,
                "addr": server_args.zmq_tokenizer_addr,
                "backend_addr": server_args.zmq_backend_addr,
                "frontend_addr": server_args.zmq_frontend_addr,
                "local_bs": 1,
                "create": server_args.tokenizer_create_addr,
                "tokenizer_id": i,
                "ack_queue": ack_queue,
            },
            daemon=False,
            name=f"minisgl-tokenizer-{i}",
        )
        p.start()
        procs.append(p)

    # The main process dies with its workers (a dead scheduler must not leave
    # a wedged API server behind — see _watch_workers).
    _watch_workers(procs, logger)

    # Wait for acknowledgments from all worker processes:
    # - world_size schedulers (but only primary rank sends ack)
    # - num_tokenizers tokenizers
    # - 1 detokenizer
    # Total acks expected: 1 + num_tokenizers + 1 = num_tokenizers + 2
    for _ in range(num_tokenizers + 2):
        logger.info(ack_queue.get())


def log_stack_versions(logger) -> None:
    """CUDA, driver, and the wheels this process actually imported."""
    import importlib.metadata as metadata
    import shutil
    import subprocess

    parts = []
    for dist in ("torch", "transformers", "flashinfer-python", "sgl_kernel"):
        try:
            parts.append(f"{dist}={metadata.version(dist)}")
        except metadata.PackageNotFoundError:
            parts.append(f"{dist}=missing")
    cuda = "unknown"
    try:
        import torch

        cuda = str(torch.version.cuda)
    except Exception:
        pass
    driver = "unknown"
    if shutil.which("nvidia-smi"):
        try:
            done = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            done = None
        if done is not None and done.returncode == 0 and done.stdout.strip():
            driver = done.stdout.strip().splitlines()[0].strip()
    logger.info("stack: cuda=%s driver=%s %s", cuda, driver, " ".join(parts))


def launch_server(run_shell: bool = False) -> None:
    from .api_server import run_api_server
    from .args import parse_args

    server_args, run_shell = parse_args(sys.argv[1:], run_shell)
    logger = init_logger(__name__, "initializer")
    log_stack_versions(logger)

    run_api_server(server_args, lambda: start_subprocess(server_args, logger), run_shell=run_shell)


if __name__ == "__main__":
    launch_server()
