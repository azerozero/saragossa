"""Offline SDK contracts: no model, server, GPU or network connection."""

import importlib.util
import inspect
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from endpoints import Endpoints, response_contract, stream_contract
from matrix import Matrix
from transport import Clients


def response():
    return dict(id="resp_fixture", object="response", created_at=1, model="fixture", status="completed",
                error=None, incomplete_details=None, output=[dict(id="msg_fixture", type="message", role="assistant",
                status="completed", content=[dict(type="output_text", text="Bonjour", annotations=[], logprobs=[])])],
                parallel_tool_calls=True, tool_choice="auto", tools=[],
                usage=dict(input_tokens=12, output_tokens=1, total_tokens=13,
                           input_tokens_details=dict(cached_tokens=0, cache_write_tokens=0), output_tokens_details=dict(reasoning_tokens=0)))


def events():
    final = response()
    created = dict(final, output=[], status="in_progress", usage=None)
    item = final["output"][0]
    values = [
        dict(type="response.created", response=created),
        dict(type="response.output_item.added", output_index=0, item=dict(item, content=[], status="in_progress")),
        dict(type="response.content_part.added", output_index=0, content_index=0, item_id=item["id"],
             part=dict(type="output_text", text="", annotations=[], logprobs=[])),
        dict(type="response.output_text.delta", output_index=0, content_index=0, item_id=item["id"], delta="Bonjour", logprobs=[]),
        dict(type="response.output_text.done", output_index=0, content_index=0, item_id=item["id"], text="Bonjour", logprobs=[]),
        dict(type="response.content_part.done", output_index=0, content_index=0, item_id=item["id"], part=item["content"][0]),
        dict(type="response.output_item.done", output_index=0, item=item),
        dict(type="response.completed", response=final),
    ]
    return [dict(event, sequence_number=i) for i, event in enumerate(values)]


class EndpointCapabilities(unittest.TestCase):
    def test_tools_follow_configured_family(self):
        for family in ("flint", "qwen3_5", "qwen3_6", "qwen3_6_moe", "qwen3_8", "qwen3_8_moe", "gemma4", "gemma4_text", "gemma4_unified", "gemma4_unified_text", "gemma3", "qwen3", "unknown"):
            matrix = Matrix(None, "fixture", "chat", model_type=family)
            expected = family not in {"gemma3", "qwen3", "unknown"}
            for case in matrix.cases:
                if (case.endpoint == "/v1/responses" and case.option.startswith("function_")) or case.option == "tools_exact_prompt":
                    self.assertEqual(not bool(case.unsupported), expected, (family, case.option))


@unittest.skipUnless(importlib.util.find_spec("openai") and importlib.util.find_spec("anthropic"), "SDK non installés")
class EndpointSDK(unittest.TestCase):
    def setUp(self):
        import httpx2
        self.requests = []
        def respond(request):
            self.requests.append(request)
            body = json.loads(request.content)
            if request.url.path.endswith("/count_tokens"):
                return httpx2.Response(200, json={"input_tokens":12})
            if request.url.path.endswith("/messages"):
                return httpx2.Response(200, json=dict(id="msg_fixture",type="message",role="assistant",model="fixture",
                    content=[{"type":"text","text":"Bonjour"}],stop_reason="max_tokens",stop_sequence=None,
                    usage={"input_tokens":12,"output_tokens":1}))
            if body.get("stream"):
                payload = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events())
                return httpx2.Response(200,text=payload,headers={"content-type":"text/event-stream"})
            return httpx2.Response(200,json=response())
        transport = httpx2.MockTransport(respond)
        with patch("openai.DefaultHttpxClient", side_effect=lambda **kw: httpx2.Client(transport=transport, **kw)), \
             patch("anthropic.DefaultHttpxClient", side_effect=lambda **kw: httpx2.Client(transport=transport, **kw)):
            self.clients = Clients("http://127.0.0.1:1", "local-test", 1)
        self.addCleanup(self.clients.close)
        self.endpoint = Endpoints(Matrix(self.clients, "fixture", "chat"))

    def test_sdk_signatures_and_wire(self):
        inspect.signature(self.clients.openai.responses.create).bind(**self.endpoint.body(stream=True))
        inspect.signature(self.clients.anthropic.messages.count_tokens).bind(model="fixture",messages=[],system="règle",tools=[])
        self.endpoint.text()
        self.endpoint.count(tools=True)
        self.assertEqual([r.url.path for r in self.requests], ["/v1/responses","/v1/messages/count_tokens","/v1/messages"])
        count, generated = (json.loads(r.content) for r in self.requests[1:])
        self.assertNotIn("max_tokens", count)
        for key in ("model", "messages", "system", "tools"):
            self.assertEqual(count[key], generated[key])

    def test_sdk_stream_deserialization(self):
        with patch.object(self.clients, "stream_wire", return_value=events()):
            self.endpoint.text(stream=True)
        self.assertTrue(json.loads(self.requests[0].content)["stream"])

    def test_strict_schema_and_stream_order(self):
        response_contract(response(), "fixture")
        stream_contract(events(), "fixture")
        for corrupted in [events()[1:], events() + events()[-1:], list(reversed(events()))]:
            with self.assertRaises((AssertionError, ValueError)):
                stream_contract(corrupted, "fixture")

    def test_matrix_registers_all_cases(self):
        matrix = Matrix(self.clients, "fixture", "chat", flint=True)
        cases = {(case.endpoint, case.option) for case in matrix.cases}
        self.assertIn(("/v1/responses","function_stream_roundtrip"), cases)
        self.assertIn(("/v1/messages/count_tokens","tools_exact_prompt"), cases)


if __name__ == "__main__":
    unittest.main(verbosity=2)
