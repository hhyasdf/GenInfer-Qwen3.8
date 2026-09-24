"""Admin control surface + request observability (gen-inference contract layer).

mini-sglang upstream ships no health/admin surface; the gen-inference
contract requires:

- ``GET  /health``      liveness probe + active backend marker
- ``GET  /admin/stats`` request/token counters, active requests, uptime,
  and the scheduler's recent step ring (``logs/step-ring.json``)
- ``POST /admin/drain`` stop accepting new generation requests (in-flight
  generations finish; new ones get ``503 + Retry-After: 1``)

plus request logging (every request logged on receipt and on completion)
and 422 logging (unparseable bodies logged with a 256-char body preview —
a silent 422 is undiagnosable from the supervisor log).

Notes:

- ``tokens_out`` is **approximate**: the detokenizer replies carry text,
  not token counts, so the server counts streamed chunks (one chunk ≈ one
  decoded token in the incremental detokenizer).
- ``tokens_in`` is **not tracked** at the API layer: the tokenizer lives in
  a separate worker process, so the API server never sees prompt token ids.
"""

from __future__ import annotations

import signal
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict

import uvicorn
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from minisgl.step_ring import read_ring
from minisgl.utils import init_logger

logger = init_logger(__name__, "Admin")


@dataclass
class AdminState:
    """Per-process admin/observability state (one API server process)."""

    start_time: float = field(default_factory=time.time)
    draining: bool = False
    active: int = 0
    requests_total: int = 0
    requests_completed: int = 0
    requests_failed: int = 0
    tokens_out: int = 0
    backend: str = "engine"


_ADMIN_STATE: AdminState | None = None


def init_admin_state(backend: str = "engine") -> AdminState:
    global _ADMIN_STATE
    _ADMIN_STATE = AdminState(backend=backend)
    return _ADMIN_STATE


def get_admin_state() -> AdminState:
    assert _ADMIN_STATE is not None, "Admin state is not initialized"
    return _ADMIN_STATE


def check_draining() -> JSONResponse | None:
    """Drain gate for generation endpoints: ``503 + Retry-After: 1`` while
    the server is draining (the supervisor drains before a restart/swap)."""
    state = get_admin_state()
    if not state.draining:
        return None
    return JSONResponse(
        status_code=503,
        content={
            "error": {
                "message": "server is draining; new requests are rejected until the drain completes",
                "type": "server_error",
                "code": "draining",
            }
        },
        headers={"Retry-After": "1"},
    )


async def track_stream(gen: AsyncGenerator[bytes, None], scope: Scope) -> AsyncGenerator[bytes, None]:
    """Wrap a streaming response generator: count streamed chunks (≈ tokens)
    for the request log and the admin counters, and hold the active-request
    counter for the lifetime of the stream (the ``finally`` runs on normal
    end, server-side error, and client disconnect)."""
    state = get_admin_state()
    state.active += 1
    n = 0
    try:
        async for chunk in gen:
            n += 1
            state.tokens_out += 1
            scope["minisgl_tokens_out"] = n
            yield chunk
    finally:
        state.active -= 1


class AdminMiddleware:
    """Raw ASGI middleware: capture the request body (for 422 logging),
    time every request, log on receipt and on completion.

    ``/health`` and ``/admin/*`` are excluded from the request log (the
    supervisor probes them every 15 s; logging them would drown real
    traffic).
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path == "/health" or path.startswith("/admin"):
            await self.app(scope, receive, send)
            return

        state = get_admin_state()
        state.requests_total += 1
        method = scope.get("method", "?")
        started = time.perf_counter()
        status_holder: Dict[str, int] = {}

        async def capture_receive() -> Dict:
            message = await receive()
            if message["type"] == "http.request":
                body = message.get("body", b"")
                if body:
                    scope["minisgl_body"] = scope.get("minisgl_body", b"") + body
            return message

        async def capture_send(message: Dict) -> None:
            if message["type"] == "http.response.start":
                status_holder["status"] = message["status"]
            await send(message)

        logger.info("[req] %s %s", method, path)
        try:
            await self.app(scope, capture_receive, capture_send)
            status = status_holder.get("status", 200)
        except Exception:
            state.requests_failed += 1
            raise
        else:
            if status >= 400:
                state.requests_failed += 1
            else:
                state.requests_completed += 1
        finally:
            elapsed = time.perf_counter() - started
            status = status_holder.get("status", "?")
            tokens = scope.get("minisgl_tokens_out")
            if tokens:
                tok_s = tokens / elapsed if elapsed > 0 else 0.0
                logger.info(
                    "[req] %s %s status=%s elapsed=%.3fs tokens~%d tok/s~%.1f",
                    method,
                    path,
                    status,
                    elapsed,
                    tokens,
                    tok_s,
                )
            else:
                logger.info("[req] %s %s status=%s elapsed=%.3fs", method, path, status, elapsed)


class EngineUvicornServer(uvicorn.Server):
    """Log signal-driven shutdowns so the supervisor log can distinguish a
    supervisor-initiated stop (drain + SIGTERM) from an external kill."""

    def handle_exit(self, sig: int, frame: Any) -> None:
        try:
            name = signal.Signals(sig).name
        except ValueError:
            name = str(sig)
        logger.info("Signal %s received — shutting down", name)
        super().handle_exit(sig, frame)


def install_admin(app: FastAPI, backend: str = "engine") -> None:
    """Mount the admin surface, the request-logging middleware, and the 422
    handler onto ``app``. Must be called before the first request."""
    init_admin_state(backend=backend)

    @app.get("/health")
    async def health() -> Dict:
        state = get_admin_state()
        return {"status": "ok", "backend": state.backend, "draining": state.draining}

    @app.get("/admin/stats")
    async def stats() -> Dict:
        state = get_admin_state()
        return {
            "active_requests": state.active,
            "requests_total": state.requests_total,
            "requests_completed": state.requests_completed,
            "requests_failed": state.requests_failed,
            "tokens_out": state.tokens_out,
            "uptime_s": round(time.time() - state.start_time, 1),
            "draining": state.draining,
            "backend": state.backend,
            "steps": read_ring(),
        }

    @app.post("/admin/drain")
    async def drain() -> Dict:
        state = get_admin_state()
        state.draining = True
        logger.info("Drain requested: new generation requests will be rejected")
        return {"status": "draining", "active_requests": state.active}

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(request: Request, exc: RequestValidationError):
        body: bytes = request.scope.get("minisgl_body", b"")
        preview = body.decode("utf-8", "replace")[:256]
        logger.warning(
            "422 unparseable request %s %s: %s | body_preview=%r",
            request.method,
            request.url.path,
            jsonable_encoder(exc.errors()),
            preview,
        )
        return JSONResponse(status_code=422, content={"detail": jsonable_encoder(exc.errors())})

    app.add_middleware(AdminMiddleware)
