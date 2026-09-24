"""OpenAI-compatible ``POST /v1/completions`` (gen-inference contract layer).

mini-sglang upstream ships only ``/v1/chat/completions``; the contract
requires ``/v1/completions`` as well. This module registers the endpoint on
the shared FastAPI app; ``api_server.py`` imports :func:`register_completions`
at the end of the module (after ``OpenAICompletionRequest`` is defined), so
there is no circular import.
"""

from __future__ import annotations

import json
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from minisgl.core import SamplingParams
from minisgl.message import TokenizeMsg
from minisgl.utils import init_logger

from .admin import check_draining, get_admin_state, track_stream
from .api_server import OpenAICompletionRequest, _first_prompt, get_global_state

logger = init_logger(__name__, "Completions")


def register_completions(app: FastAPI) -> None:
    @app.post("/v1/completions")
    async def v1_completions(req: OpenAICompletionRequest, request: Request):
        if (drain := check_draining()) is not None:
            return drain

        if req.prompt is None:
            return JSONResponse(
                status_code=422,
                content={"detail": "'prompt' is required for /v1/completions"},
            )

        prompt = _first_prompt(req.prompt)
        state = get_global_state()
        uid = state.new_user()
        await state.send_one(
            TokenizeMsg(
                uid=uid,
                text=prompt,
                sampling_params=SamplingParams(
                    ignore_eos=req.ignore_eos,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_k=req.top_k,
                    top_p=req.top_p,
                ),
            )
        )

        if req.stream:
            return StreamingResponse(
                track_stream(
                    state.stream_with_cancellation(
                        _stream_completions(state, uid), request, uid
                    ),
                    request.scope,
                ),
                media_type="text/event-stream",
            )

        # Non-streaming: collect all chunks and return a single JSON response
        n_tokens = 0
        full_content = ""
        state_admin = get_admin_state()
        state_admin.active += 1
        try:
            async for ack in state.wait_for_ack(uid):
                n_tokens += 1
                full_content += ack.incremental_output
                if ack.finished:
                    break
        finally:
            state_admin.active -= 1
        state_admin.tokens_out += n_tokens
        request.scope["minisgl_tokens_out"] = n_tokens

        return {
            "id": f"cmpl-{uid}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "text": full_content,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": n_tokens,
                "total_tokens": n_tokens,
            },
        }

    async def _stream_completions(state, uid: int):
        async for ack in state.wait_for_ack(uid):
            chunk = {
                "id": f"cmpl-{uid}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": state.config.model_path,
                "choices": [
                    {
                        "text": ack.incremental_output,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n".encode()
            if ack.finished:
                break

        end_chunk = {
            "id": f"cmpl-{uid}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": state.config.model_path,
            "choices": [{"text": "", "index": 0, "logprobs": None, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(end_chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"
