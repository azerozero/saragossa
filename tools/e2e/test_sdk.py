"""Verify installed SDK signatures and serialization entirely in memory."""

import copy
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from matrix import ANTHROPIC_TOOL, OPENAI_TOOL, Matrix, sdk_body
from test_contracts import events, message
from transport import Clients


@unittest.skipUnless(importlib.util.find_spec("openai") and importlib.util.find_spec("anthropic"),
                     "SDK officiels non installés; aucune installation automatique")
class SDK(unittest.TestCase):
    def setUp(self):
        import httpx2

        self.requests = []
        self.response = message()
        self.status = 200

        def respond(request):
            self.requests.append(request)
            if json.loads(request.content).get("stream"):
                payload = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
                                  for event in events())
                return httpx2.Response(200, text=payload, headers={"content-type": "text/event-stream"})
            return httpx2.Response(self.status, json=self.response)

        transport = httpx2.MockTransport(respond)
        def client(**kwargs):
            return httpx2.Client(transport=transport, **kwargs)
        with patch("openai.DefaultHttpxClient", side_effect=client), \
             patch("anthropic.DefaultHttpxClient", side_effect=client):
            self.clients = Clients("http://127.0.0.1:1", "local-test-key", 1)
        self.addCleanup(self.clients.close)
        self.matrix = Matrix(self.clients, "test", "chat")

    def test_every_used_resource_signature(self):
        """bind follows raw/streaming wrappers and rejects unexpected kwargs."""
        a, b = self.clients.openai, self.clients.anthropic
        chat = self.matrix.chat_body(seed=42, stop=["Bonjour"], n=2,
            response_format={"type": "json_object"}, stream_options={"include_usage": True},
            tools=[OPENAI_TOOL], tool_choice="auto", stream=True)
        anthropic = sdk_body(self.matrix.chat_body(system="Bonjour", stop_sequences=["Bonjour"],
            tools=[ANTHROPIC_TOOL], tool_choice={"type": "auto"}, stream=True), True)
        embedding = dict(model="test", input=["Bonjour"], encoding_format="float")
        speech = dict(model="test", input="Bonjour", voice="alloy", response_format="wav")
        transcription = dict(model="whisper", file=("probe.wav", b"fixture", "audio/wav"),
                             language="fr", response_format="json")
        calls = [
            (a.models.with_raw_response.list, {}),
            (a.chat.completions.create, chat),
            (a.chat.completions.with_raw_response.create, chat),
            (b.messages.create, anthropic),
            (b.messages.with_raw_response.create, anthropic),
            (a.embeddings.create, embedding),
            (a.embeddings.with_raw_response.create, embedding),
            (a.audio.speech.create, speech),
            (a.audio.speech.with_streaming_response.create, speech),
            (a.audio.transcriptions.create, transcription),
            (a.audio.transcriptions.with_raw_response.create, transcription),
        ]
        for method, kwargs in calls:
            with self.subTest(method=method.__qualname__):
                signature = inspect.signature(method)
                signature.bind(**kwargs)
                self.assertNotIn(inspect.Parameter.VAR_KEYWORD,
                                 [parameter.kind for parameter in signature.parameters.values()])
        self.assertNotIn("temperature", inspect.signature(b.messages.create).parameters)
        with self.assertRaises(TypeError):
            inspect.signature(b.messages.create).bind(**self.matrix.chat_body())

    def assert_temperature_on_wire(self):
        self.assertTrue(self.requests)
        for request in self.requests:
            payload = json.loads(request.content)
            self.assertEqual(payload["temperature"], 0)
            self.assertNotIn("extra_body", payload)

    def test_anthropic_raw_temperature_and_body_preservation(self):
        body = self.matrix.chat_body(system="Bonjour", stop_sequences=["fin"])
        original = copy.deepcopy(body)
        self.matrix.complete(body, anthropic=True)
        self.assertEqual(body, original)
        self.assert_temperature_on_wire()
        self.assertEqual(json.loads(self.requests[0].content), body)
        extended = dict(body, extra_body={"extension": True})
        self.assertEqual(sdk_body(extended, True)["extra_body"], {"extension": True, "temperature": 0})
        self.assertEqual(extended["extra_body"], {"extension": True})

    def test_anthropic_live_iterator_keeps_temperature(self):
        # The wire check remains separate; only SDK I/O is replaced in memory.
        with patch.object(self.clients, "stream_wire", return_value=events()) as wire:
            self.matrix.stream(anthropic=True)
        self.assertEqual(wire.call_args.args[1]["temperature"], 0)
        self.assert_temperature_on_wire()

    def test_anthropic_unknown_model_reaches_http_error(self):
        self.status = 404
        self.response = {"type": "error", "error": {
            "type": "not_found_error", "message": "modèle inconnu"}}
        self.matrix.unknown(True)
        self.assert_temperature_on_wire()

    def test_anthropic_tool_request_keeps_temperature(self):
        self.response = dict(message(), stop_reason="tool_use", content=[{
            "type": "tool_use", "id": "call-1", "name": "get_weather", "input": {"city": "Paris"}}])
        self.matrix.tool(anthropic=True, choice={"type": "auto"})
        self.assert_temperature_on_wire()
        self.assertEqual(json.loads(self.requests[0].content)["tools"], [ANTHROPIC_TOOL])


if __name__ == "__main__":
    unittest.main(verbosity=2)
