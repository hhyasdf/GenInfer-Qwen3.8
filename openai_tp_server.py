# OpenAI 兼容 API 服务 (Phase 9/10)。薄 HTTP 层, 委托 LLMEngine.generate。
# 端点: /v1/chat/completions, /v1/completions, /health。
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from llm_engine import LLMEngine

MODEL_DIR = os.environ.get("MODEL_DIR", ".")
_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = LLMEngine(MODEL_DIR, inference_backend="qwen_tp", max_num_seqs=4)
    return _engine


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok"})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            self._json(400, {"error": "bad json"})
            return
        if self.path == "/v1/chat/completions":
            prompt = ""
            for m in req.get("messages", []):
                prompt += m.get("content", "") + " "
            max_tokens = req.get("max_tokens", 64)
            temperature = req.get("temperature", 0.0)
            text = get_engine().generate(prompt, max_new_tokens=max_tokens,
                                         temperature=temperature)
            self._json(200, {"choices": [{"message": {"content": text},
                                          "finish_reason": "stop"}]})
        elif self.path == "/v1/completions":
            prompt = req.get("prompt", "")
            max_tokens = req.get("max_tokens", 64)
            temperature = req.get("temperature", 0.0)
            text = get_engine().generate(prompt, max_new_tokens=max_tokens,
                                         temperature=temperature)
            self._json(200, {"choices": [{"text": text, "finish_reason": "stop"}]})
        else:
            self._json(404, {"error": "not found"})


def main():
    port = int(os.environ.get("PORT", "8000"))
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
