"""Local fake OpenAI API for offline QA of the ``openai`` provider.

Serves fixed JSON for ``GET /v1/models`` and ``POST /v1/chat/completions``.
Point the provider at it with ``JEVPILOT_OPENAI_BASE_URL=http://127.0.0.1:<port>/v1``.
With ``--status 401`` every request is rejected, to exercise ``auth_invalid``.
Request logs never include headers, so API keys are not printed.

    python -m benchmarks.browser_use_eval.fake_openai_server --port 8799
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Final, override

MODEL_ID: Final = "fake-model"
DEFAULT_CONTENT: Final = json.dumps(
    {
        "goal": "find the north region price",
        "start_url": "http://127.0.0.1:8765/filter.html",
        "done_when": "the north row price is visible",
    }
)


def _models() -> dict[str, object]:
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "jevpilot"}],
    }


def _completion(content: str) -> dict[str, object]:
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "created": 0,
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": 30,
            "total_tokens": 150,
            "prompt_tokens_details": {"cached_tokens": 20},
        },
    }


def make_server(
    port: int = 0, *, status: int = 200, content: str = DEFAULT_CONTENT
) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict[str, object]) -> None:
            payload = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            _ = self.wfile.write(payload)

        def _reject_or(self, body: dict[str, object]) -> None:
            if status != 200:
                self._send(
                    status,
                    {"error": {"message": "fake rejection", "type": "invalid_request_error", "code": "invalid_api_key"}},
                )
            else:
                self._send(200, body)

        def do_GET(self) -> None:
            if self.path.split("?")[0] == "/v1/models":
                self._reject_or(_models())
            else:
                self._send(404, {"error": {"message": "not found"}})

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            _ = self.rfile.read(length)
            if self.path == "/v1/chat/completions":
                self._reject_or(_completion(content))
            else:
                self._send(404, {"error": {"message": "not found"}})

        @override
        def log_message(self, format: str, *args: object) -> None:
            _ = sys.stderr.write(f"fake-openai {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}\n")

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def start_in_thread(port: int = 0, **kwargs: object) -> tuple[ThreadingHTTPServer, str]:
    server = make_server(port, **kwargs)  # pyright: ignore[reportArgumentType]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--port", type=int, default=8799)
    _ = parser.add_argument("--status", type=int, default=200)
    _ = parser.add_argument("--content", default=DEFAULT_CONTENT)
    args = parser.parse_args()
    server = make_server(args.port, status=args.status, content=args.content)
    print(f"fake OpenAI server on http://127.0.0.1:{server.server_address[1]}/v1", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
