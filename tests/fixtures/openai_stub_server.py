"""Minimal OpenAI-compatible chat-completions stub for offline end-to-end tests.

Answers ``POST /v1/responses`` (Responses API, used by the gpt-5 specs) with a fixed
instruction in a fenced block, and ``POST /v1/chat/completions``:
- a request offering tools gets a call to ``end_execution`` when offered, else to the first
  tool, with arguments filled from its JSON schema (``answer`` comes from ``STUB_ANSWER``,
  default 7);
- a request with a ``json_schema`` response_format gets JSON filled from that schema;
- otherwise a text reply proposing a fixed instruction (fenced), which GEPA reflection
  calls adopt as their new candidate.
Each request is appended to ``STUB_LOG`` as one JSON line: path, model, kind, tool_choice,
tools offered, and ``text`` (the request's message contents, for assertions about what
reached the model).
Prints the bound port on stdout, then serves until killed.
"""

import json
import os
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

ANSWER = int(os.environ.get("STUB_ANSWER", "7"))
LOG = os.environ.get("STUB_LOG", "")


def fill(schema: dict[str, Any], defs: dict[str, Any]) -> Any:
    if "$ref" in schema:
        return fill(defs[schema["$ref"].split("/")[-1]], defs)
    for key in ("anyOf", "oneOf"):
        if key in schema:
            return fill(next(s for s in schema[key] if s.get("type") != "null"), defs)
    kind = schema.get("type")
    if kind == "object":
        props = schema.get("properties", {})
        return {
            name: (
                (str(ANSWER) if sub.get("type") == "string" else ANSWER)
                if name == "answer"
                else fill(sub, defs)
            )
            for name, sub in props.items()
        }
    if kind == "array":
        return []
    if kind in ("integer", "number"):
        return schema.get("maximum", 1)
    if kind == "boolean":
        return True
    if "enum" in schema:
        return schema["enum"][0]
    return "ok"


def _content_text(content: Any) -> str:
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return "" if content is None else str(content)


PROPOSAL = "```\nSolve the arithmetic expression and return the integer result.\n```"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass

    def _send(self, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _log(self, body: dict[str, Any], kind: str, tools: dict[str, Any]) -> None:
        messages = body.get("messages") or body.get("input") or []
        if isinstance(messages, str):
            messages = [{"content": messages}]
        text = "\n".join(_content_text(m.get("content")) for m in messages if isinstance(m, dict))
        if LOG:
            with open(LOG, "a") as fh:
                record = {
                    "path": self.path,
                    "model": body.get("model"),
                    "kind": kind,
                    "tool_choice": body.get("tool_choice"),
                }
                fh.write(json.dumps({**record, "tools": sorted(tools), "text": text}) + "\n")

    def _respond_responses_api(self, body: dict[str, Any]) -> None:
        self._log(body, "responses_text", {})
        text = PROPOSAL
        self._send(
            {
                "id": f"resp_{uuid.uuid4().hex[:8]}",
                "object": "response",
                "created_at": int(time.time()),
                "model": body.get("model", "stub"),
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "id": f"msg_{uuid.uuid4().hex[:8]}",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": text, "annotations": []}],
                    }
                ],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            }
        )

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
        if self.path.rstrip("/").endswith("/responses"):
            self._respond_responses_api(body)
            return
        tools = {t["function"]["name"]: t["function"] for t in body.get("tools", [])}
        message: dict[str, Any] = {"role": "assistant", "content": None}
        finish = "stop"
        response_format = body.get("response_format") or {}
        if tools:
            name = "end_execution" if "end_execution" in tools else next(iter(tools))
            params = tools[name].get("parameters", {})
            args = fill(params, params.get("$defs", {}))
            message["tool_calls"] = [
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
            ]
            finish, kind = "tool_calls", f"tool_call:{name}"
        elif response_format.get("type") == "json_schema":
            schema = response_format["json_schema"]["schema"]
            message["content"] = json.dumps(fill(schema, schema.get("$defs", {})))
            kind = "structured_output"
        else:  # e.g. a GEPA reflection call: propose a new instruction
            message["content"] = PROPOSAL
            kind = "text"
        self._log(body, kind, tools)
        payload = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", "stub"),
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        if body.get("stream"):
            self._stream(payload)
        else:
            self._send(payload)

    def _stream(self, payload: dict[str, Any]) -> None:
        """The same completion as one server-sent-events chunk."""
        choice = payload["choices"][0]
        delta = {**choice["message"]}
        if "tool_calls" in delta:
            delta["tool_calls"] = [{"index": i, **c} for i, c in enumerate(delta["tool_calls"])]
        chunk = {**payload, "object": "chat.completion.chunk"}
        chunk["choices"] = [{"index": 0, "delta": delta, "finish_reason": choice["finish_reason"]}]
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        self.wfile.write(f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode())


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    print(server.server_address[1], flush=True)
    sys.stdout.close()
    server.serve_forever()
