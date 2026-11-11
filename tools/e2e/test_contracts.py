"""CPU-only harness tests: no server, weights, network, or GPU."""

import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import contracts as c
from launcher import configuration, server, stop, verify_mode
from matrix import CHAT, MESSAGES, Case, Matrix
from run import FAIL, OK, UNSUPPORTED, execute_case, write_report
from transport import Clients


def chat():
    return {"id": "chatcmpl-test", "object": "chat.completion", "created": 1, "model": "test",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Bonjour"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}}


def chunks(with_usage=False):
    def chunk(delta, reason=None):
        return {"id": "chatcmpl-test", "object": "chat.completion.chunk", "created": 1, "model": "test",
                "choices": [{"index": 0, "delta": delta, "finish_reason": reason}]}
    result = [chunk({"role": "assistant"}), chunk({"content": "Bonjour"}), chunk({}, "stop")]
    if with_usage:
        for item in result:
            item["usage"] = None
        result.append(dict(result[0], choices=[], usage=chat()["usage"]))
    return result


def message():
    return {"id": "msg-test", "type": "message", "role": "assistant", "model": "test",
            "content": [{"type": "text", "text": "Bonjour"}], "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 3, "output_tokens": 1}}


def events():
    initial = dict(message(), content=[], stop_reason=None, usage={"input_tokens": 3, "output_tokens": 0})
    return [{"type": "message_start", "message": initial},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Bonjour"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 1}},
            {"type": "message_stop"}]


class Contracts(unittest.TestCase):
    def test_valid_chat_and_message(self):
        c.chat(chat(), "test", 8)
        c.anthropic(message(), "test", 8)

    def test_missing_fields_are_not_filled_by_sdk_defaults(self):
        for key in ("id", "created", "usage", "model", "choices"):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                value = chat()
                del value[key]
                c.chat(value, "test", 8)

    def test_reject_coercion_boolean_and_inconsistent_usage(self):
        for value in (True, "1", -1, 0.5, 9):
            with self.subTest(value=value), self.assertRaises(AssertionError):
                body = chat()
                body["usage"]["completion_tokens"] = value
                c.chat(body, "test", 8)

    def test_content_and_finish_reason(self):
        for content, reason in (("", "stop"), ([], "stop"), ("Bonjour", "unknown"), ("Bonjour", "tool_calls")):
            with self.subTest(content=content, reason=reason), self.assertRaises(AssertionError):
                value = chat()
                value["choices"][0]["message"]["content"] = content
                value["choices"][0]["finish_reason"] = reason
                c.chat(value, "test", 8)

    def test_embeddings_shape_dimension_normalization(self):
        body = {"object": "list", "model": "test", "data": [
            {"object": "embedding", "index": 0, "embedding": [1.0, 0.0]}]}
        c.embeddings(body, "test", 1, 2)
        for vector in ([1.0], [0.0, 0.0], [float("nan"), 0], [True, 0], [2.0, 0]):
            with self.subTest(vector=vector), self.assertRaises(AssertionError):
                body["data"][0]["embedding"] = vector
                c.embeddings(body, "test", 1, 2)

    def test_models_require_exact_registry(self):
        body = {"object": "list", "data": [{"id": "test", "object": "model", "created": 0, "owned_by": "local"}]}
        c.models(body, ["test"])
        with self.assertRaises(AssertionError):
            c.models(body, ["other"])

    def test_error_envelopes_are_not_interchangeable(self):
        value = {"error": {"type": "invalid_request_error", "message": "body trop grand"}}
        c.error(value, contains="body")
        with self.assertRaises(AssertionError):
            c.error(value, anthropic=True)
        c.error(dict(value, type="error"), anthropic=True)

    def test_sse_crlf_comments_and_multiline(self):
        payload = b': ping\r\nevent: message\r\ndata: {"a":\r\ndata: 1}\r\n\r\n'
        self.assertEqual(list(c.sse_frames(payload.splitlines(keepends=True))), [("message", '{"a":\n1}')])

    def test_sse_truncation_and_malformed_utf8(self):
        for payload in (b'data: {}\n', b'data: \xff\n\n', b'garbage\n\n'):
            with self.subTest(payload=payload), self.assertRaises((AssertionError, UnicodeDecodeError)):
                list(c.sse_frames(payload.splitlines(keepends=True)))

    def test_openai_stream_with_and_without_usage(self):
        for usage in (False, True):
            self.assertEqual(c.openai_stream(chunks(usage), "test", 8, usage), "Bonjour")

    def test_openai_stream_truncation_identity_and_order(self):
        invalid = []
        invalid.append(chunks()[:-1])
        invalid.append(chunks()[1:])
        invalid.append(chunks() + [chunks()[1]])
        changed = chunks()
        changed[1]["id"] = "other"
        invalid.append(changed)
        changed = chunks()
        changed[1]["created"] = 2
        invalid.append(changed)
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(AssertionError):
                c.openai_stream(value, "test", 8)

    def test_openai_missing_usage_and_tool_fragments(self):
        with self.assertRaises(AssertionError):
            c.openai_stream(chunks(), "test", 8, True)
        missing_null = chunks(True)
        del missing_null[0]["usage"]
        with self.assertRaises(AssertionError):
            c.openai_stream(missing_null, "test", 8, True)
        value = chunks()
        value[1]["choices"][0]["delta"] = {"tool_calls": [{"index": 0, "id": "call-1", "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city":'}}]}
        continuation = copy.deepcopy(value[1])
        continuation["choices"][0]["delta"] = {"tool_calls": [{"index": 0, "function": {"arguments": '"Paris"}'}}]}
        value.insert(2, continuation)
        value[-1]["choices"][0]["finish_reason"] = "tool_calls"
        c.openai_stream(value, "test", 8, tools=True)
        value[2]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] = "INVALID"
        with self.assertRaises(json.JSONDecodeError):
            c.openai_stream(value, "test", 8, tools=True)

    def test_anthropic_stream_reconstructs_message(self):
        self.assertEqual(c.anthropic_stream(events(), "test", 8), message())

    def test_anthropic_stream_rejects_order_and_truncation(self):
        invalid = [events()[:-1], events()[1:], events() + [events()[-1]]]
        wrong = events()
        wrong[2]["index"] = 1
        invalid.append(wrong)
        wrong = events()
        wrong[2], wrong[3] = wrong[3], wrong[2]
        invalid.append(wrong)
        wrong = events()
        del wrong[3]
        invalid.append(wrong)
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(AssertionError):
                c.anthropic_stream(value, "test", 8)

    def test_anthropic_stream_tool_input_json(self):
        value = events()
        value[1]["content_block"] = {"type": "tool_use", "id": "tool-1", "name": "get_weather", "input": {}}
        value[2]["delta"] = {"type": "input_json_delta", "partial_json": '{"city":"Paris"}'}
        value[4]["delta"]["stop_reason"] = "tool_use"
        c.anthropic_stream(value, "test", 8, tools=True)
        value[2]["delta"]["partial_json"] = '{"city":'
        with self.assertRaises(json.JSONDecodeError):
            c.anthropic_stream(value, "test", 8, tools=True)


class Runner(unittest.TestCase):
    def test_unsupported_is_never_a_pass(self):
        action = Mock()
        row = execute_case(Case(CHAT, "tools", action, "known unsupported"), "test", "ar")
        self.assertEqual(row["status"], UNSUPPORTED)
        self.assertFalse(row["executed"])
        action.assert_not_called()

    def test_expected_rejection_failure_is_not_hidden(self):
        case = Case(CHAT, "json_schema", Mock(side_effect=AssertionError("accepted")), "expected 501",
                    verify_rejection=True)
        row = execute_case(case, "test", "ar")
        self.assertEqual(row["status"], FAIL)
        self.assertTrue(row["executed"])

    def test_successful_rejection_remains_unsupported(self):
        case = Case(CHAT, "json_schema", lambda: None, "expected 501", verify_rejection=True)
        self.assertEqual(execute_case(case, "test", "ar")["status"], UNSUPPORTED)

    def test_report_counts_and_redaction(self):
        case = Case(CHAT, "simple", Mock(side_effect=ValueError("private-location")))
        row = execute_case(case, "test", "ar", ["private-location"])
        report = {"revision": "test", "suite": "full", "results": [row]}
        with tempfile.TemporaryDirectory() as temporary:
            write_report(Path(temporary), report)
            data = json.loads((Path(temporary) / "report.json").read_text())
            self.assertEqual(data["counts"], {FAIL: 1})
            self.assertNotIn("private-location", (Path(temporary) / "report.md").read_text())

    def test_matrix_is_unique_and_smoke_reuses_same_cases(self):
        for family in ("chat", "embed", "stt", "tts", "audio"):
            matrix = Matrix(None, "test", family)
            keys = [(case.endpoint, case.option) for case in matrix.cases]
            self.assertEqual(len(keys), len(set(keys)))
            self.assertTrue(all(case in matrix.cases for case in matrix.selected("smoke")))
            self.assertTrue(all(case.action or case.unsupported for case in matrix.cases))

    def test_n_rejection_is_a_gate_not_a_static_skip(self):
        matrix = Matrix(None, "test", "chat")
        case = next(case for case in matrix.cases if case.option == "n_rejected")
        self.assertFalse(case.unsupported)

    def test_families_do_not_mislabel_tools(self):
        for flint in (False, True):
            matrix = Matrix(None, "test", "chat", flint)
            tool = next(case for case in matrix.cases if case.endpoint == CHAT and case.option == "tools_auto")
            self.assertEqual(bool(tool.unsupported), not flint)
            none = next(case for case in matrix.cases if case.endpoint == MESSAGES and case.option == "tools_none")
            self.assertTrue(none.unsupported)

    def test_ar_clears_inherited_flags_and_uses_explicit_alias(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "config.json").write_text('{"model_type":"qwen3_5"}')
            with patch.dict(os.environ, {"E2E_QWEN38_MODEL": str(path), "SARAGOSSA_RUST_MTP_DECODE": "1",
                                         "SARAGOSSA_RUST_DFLASH_DRAFT_DIR": "inherited", "RETI_RUST_MTP_DECODE": "1"}):
                _, _, args, env, _ = configuration("qwen38", "ar")
            self.assertEqual(args, ["--model", "qwen38=" + str(path.resolve())])
            self.assertEqual(env["SARAGOSSA_RUST_MTP_DECODE"], "0")
            self.assertNotIn("SARAGOSSA_RUST_DFLASH_DRAFT_DIR", env)
            self.assertNotIn("RETI_RUST_MTP_DECODE", env)
            self.assertEqual(env["HF_HUB_OFFLINE"], "1")

    def test_stop_escalates_and_waits_without_starting_a_process(self):
        process = Mock(pid=123, poll=Mock(return_value=None))
        process.wait.side_effect = [subprocess.TimeoutExpired("fake", 10), 0]
        with patch("launcher.os.killpg") as kill:
            stop(process)
        self.assertEqual(kill.call_count, 2)
        self.assertEqual(process.wait.call_count, 2)

    def test_server_cleanup_after_case_failure_without_network_or_process(self):
        process = Mock(pid=123, poll=Mock(return_value=None))
        health = io.BytesIO(b'{"status":"ok"}')
        health.status = 200
        registry = io.BytesIO(b'{"object":"list","data":[]}')
        registry.status = 200
        opener = Mock()
        opener.open.side_effect = [health, registry]
        reservation = Mock()
        reservation.getsockname.return_value = ("127.0.0.1", 12345)
        with patch("launcher.subprocess.Popen", return_value=process), \
             patch("launcher.socket.socket") as sock, \
             patch("launcher.urllib.request.build_opener", return_value=opener), \
             patch("launcher.os.killpg") as kill:
            sock.return_value.__enter__.return_value = reservation
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                with server(Path("unused"), [], {}, 1) as (_, _, log, _):
                    self.assertTrue(log.exists())
                    raise RuntimeError("test failure")
        kill.assert_called_once()
        process.wait.assert_called_once()
        self.assertFalse(log.exists())

    def test_mode_cannot_silently_fall_back(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "log"
            path.write_text("saragossa serve completion model=test decode_path=ar\n")
            verify_mode("ar", path, "test")
            with self.assertRaises(AssertionError):
                verify_mode("mtp", path, "test")
            path.write_text("saragossa serve MTP actif model=test sidecar=x\n"
                            "saragossa serve drafter DFlash actif model=test block_size=8\n"
                            "saragossa serve completion model=test decode_path=mtp\n")
            verify_mode("dflash", path, "test")

    @unittest.skipUnless(importlib.util.find_spec("openai") and importlib.util.find_spec("anthropic"),
                         "SDK officiels non installés; tests de contrats stdlib restent disponibles")
    def test_pinned_sdk_clients_and_official_response_models_offline(self):
        clients = Clients("http://127.0.0.1:1", "local-test-key", 1)
        self.assertEqual(clients.anthropic.auth_headers, {"Authorization": "Bearer local-test-key"})
        self.assertEqual(clients.openai.auth_headers, {"Authorization": "Bearer local-test-key"})
        clients.close()
        from openai.types.chat import ChatCompletion, ChatCompletionChunk
        from anthropic.types import Message
        c.chat(ChatCompletion.model_validate(chat()).model_dump(exclude_unset=True), "test", 8)
        c.anthropic(Message.model_validate(message()).model_dump(exclude_unset=True), "test", 8)
        c.openai_stream([ChatCompletionChunk.model_validate(x).model_dump(exclude_unset=True)
                         for x in chunks(True)], "test", 8, True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
