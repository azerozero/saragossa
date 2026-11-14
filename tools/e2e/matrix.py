"""Executable capability matrix. Unsupported entries always carry source evidence."""

from dataclasses import dataclass
from difflib import SequenceMatcher
import io
import json
import os
import re
import wave

import contracts as c
from launcher import local_path
from transport import decoded, rejected

CHAT = "/v1/chat/completions"
MESSAGES = "/v1/messages"
EMBED = "/v1/embeddings"
SPEECH = "/v1/audio/speech"
STT = "/v1/audio/transcriptions"
BUDGET = 256
SCHEMA = {"type": "object", "properties": {"city": {"type": "string"}},
          "required": ["city"], "additionalProperties": False}
FUNCTION = {"name": "get_weather", "description": "Read the current weather for a city.",
            "parameters": SCHEMA}
OPENAI_TOOL = {"type": "function", "function": FUNCTION}
ANTHROPIC_TOOL = {"name": "get_weather", "description": FUNCTION["description"], "input_schema": SCHEMA}
TOOL_PROMPT = "Call get_weather for Paris. Use the tool before answering; do not invent the weather."
AUDIO_TEXT = "Bonjour, le petit chat regarde par la fenêtre."


def sdk_body(body, anthropic=False):
    """Keep the wire payload intact despite Anthropic 2.8's narrower signature."""
    kwargs = dict(body)
    if anthropic and "temperature" in kwargs:
        kwargs["extra_body"] = dict(kwargs.get("extra_body", {}), temperature=kwargs.pop("temperature"))
    return kwargs


@dataclass
class Case:
    endpoint: str
    option: str
    action: object = None
    unsupported: str = ""
    smoke: bool = False
    verify_rejection: bool = False


class Matrix:
    def __init__(self, clients, model, family, flint=False, similarity=0.6, model_type=None):
        self.clients, self.model, self.family = clients, model, family
        self.flint, self.similarity = flint, similarity
        if model_type is None:
            model_type = "flint" if flint else {"qwen38":"qwen3_8", "qwen36_moe":"qwen3_6_moe",
                                               "gemma3":"gemma3", "gemma4":"gemma4"}.get(model, "")
        self.supports_tools = family == "chat" and (model_type in {"flint", "gemma4", "gemma4_text", "gemma4_unified", "gemma4_unified_text"}
            or any(model_type == prefix or model_type.startswith(prefix + "_")
                   for prefix in ("qwen3_5", "qwen3_6", "qwen3_8")))
        self.cases = []
        self.build()

    def add(self, endpoint, option, action=None, unsupported="", smoke=False, verify_rejection=False):
        self.cases.append(Case(endpoint, option, action, unsupported, smoke, verify_rejection))

    def chat_body(self, **options):
        return dict(model=self.model, max_tokens=BUDGET, temperature=0,
                    messages=[{"role": "user", "content": "Reply with the single word Bonjour."}], **options)

    def complete(self, body, anthropic=False, tools=False, allow_empty=False):
        resource = self.clients.anthropic.messages if anthropic else self.clients.openai.chat.completions
        value = decoded(resource.with_raw_response.create(**sdk_body(body, anthropic)))
        check = c.anthropic if anthropic else c.chat
        check(value, self.model, body["max_tokens"], tools, allow_empty)
        return value

    def normal(self, option, anthropic=False):
        body = self.chat_body()
        if option == "system":
            system = "Answer each question with the single word SAPHIR."
            body["messages"] = [{"role": "user", "content": "What is your answer?"}]
            if anthropic:
                body["system"] = system
            else:
                body["messages"].insert(0, {"role": "system", "content": system})
        elif option == "multi_turn":
            body["messages"] = [
                {"role": "user", "content": "My secret word is ORCHIDEE."},
                {"role": "assistant", "content": "I will remember your secret word."},
                {"role": "user", "content": "Repeat my secret word, nothing else."},
            ]
        elif option == "max_tokens":
            body["max_tokens"] = 1
            body["messages"] = [{"role": "user", "content": "Count from one to twenty, one number per line."}]
        value = self.complete(body, anthropic, allow_empty=option == "max_tokens")
        if option == "max_tokens":
            reason = value["stop_reason"] if anthropic else value["choices"][0]["finish_reason"]
            c.require(reason == ("max_tokens" if anthropic else "length"), "limite tokens non signalée")
        else:
            output = self.output(value, anthropic).casefold()
            expected = {"simple": "bonjour", "system": "saphir", "multi_turn": "orchidee"}[option]
            c.require(expected in output, f"contenu minimal absent: {expected}")

    @staticmethod
    def output(value, anthropic=False):
        if anthropic:
            return "".join(block["text"] for block in value["content"] if block["type"] == "text")
        return value["choices"][0]["message"]["content"] or ""

    def deterministic(self):
        body = self.chat_body(seed=42)
        first, second = self.complete(body), self.complete(body)
        c.require(first["choices"] == second["choices"], "seed + température 0 non déterministe")
        c.require(first["usage"] == second["usage"], "usage non déterministe")

    def stops(self, anthropic=False):
        # Derive a stop from a deterministic baseline: absence alone is not proof
        # that the model actually emitted (and the server removed) a stop string.
        body = self.chat_body()
        baseline = self.complete(body, anthropic)
        words = re.findall(r"\w+", self.output(baseline, anthropic))
        c.require(bool(words), "aucun mot pour le contrôle stop")
        stop = words[0]
        body["stop_sequences" if anthropic else "stop"] = [stop]
        value = self.complete(body, anthropic, allow_empty=True)
        c.require(stop not in self.output(value, anthropic), "stop divulgué dans la sortie")
        if anthropic:
            c.require(value["stop_reason"] == "stop_sequence" and value["stop_sequence"] == stop,
                      "stop_sequences non appliqué")
        else:
            c.require(value["choices"][0]["finish_reason"] == "stop", "finish_reason stop")
            c.require(value["usage"]["completion_tokens"] <= baseline["usage"]["completion_tokens"],
                      "stop ne réduit pas la génération")

    def stream(self, anthropic=False, include_usage=False, tools=False):
        body = self.tool_body(anthropic) if tools else self.chat_body()
        if include_usage:
            body["stream_options"] = {"include_usage": True}
        endpoint = MESSAGES if anthropic else CHAT
        # Wire probe checks [DONE]/event names. A second call exercises the SDK's
        # own live iterator, which deliberately hides [DONE] from the application.
        wire = self.clients.stream_wire(endpoint, body)
        check = c.anthropic_stream if anthropic else c.openai_stream
        kwargs = {"tools": tools}
        if not anthropic:
            kwargs["include_usage"] = include_usage
        check(wire, self.model, BUDGET, **kwargs)
        resource = self.clients.anthropic.messages if anthropic else self.clients.openai.chat.completions
        stream = resource.create(**sdk_body(body, anthropic), stream=True)
        try:
            events = [event.model_dump(exclude_unset=True) for event in stream]
        finally:
            stream.close()
        result = check(events, self.model, BUDGET, **kwargs)
        if not tools:
            output = self.output(result, True) if anthropic else result
            c.require("bonjour" in output.casefold(), "contenu minimal du flux")

    def tool_body(self, anthropic=False, choice=None):
        body = self.chat_body()
        body["messages"] = [{"role": "user", "content": TOOL_PROMPT}]
        body["tools"] = [ANTHROPIC_TOOL if anthropic else OPENAI_TOOL]
        if choice is not None:
            body["tool_choice"] = choice
        return body

    def tool(self, anthropic=False, choice=None, roundtrip=False):
        body = self.tool_body(anthropic, choice)
        none = choice == "none"
        if none:
            body["messages"] = [{"role": "user", "content": "Do not use tools. Reply Bonjour."}]
        value = self.complete(body, anthropic, tools=not none)
        if not roundtrip:
            return
        result = '{"temperature":17,"unit":"C","condition":"sunny"}'
        if anthropic:
            body["messages"].append({"role": "assistant", "content": value["content"]})
            tool_results = [{"type": "tool_result", "tool_use_id": block["id"], "content": result}
                            for block in value["content"] if block["type"] == "tool_use"]
            body["messages"].append({"role": "user", "content": tool_results})
        else:
            message = value["choices"][0]["message"]
            body["messages"].append(message)
            body["messages"].extend({"role": "tool", "tool_call_id": call["id"], "content": result}
                                     for call in message["tool_calls"])
            body["tool_choice"] = "none"
        final = self.complete(body, anthropic)
        c.require("17" in self.output(final, anthropic), "résultat outil absent de la réponse finale")

    def json_object(self):
        body = self.chat_body(response_format={"type": "json_object"})
        body["messages"] = [{"role": "user", "content": 'Return a JSON object with a greeting: "Bonjour".'}]
        value = self.complete(body)
        # json_object garantit un objet JSON, pas son contenu : `{}` est conforme.
        c.require(type(json.loads(self.output(value))) is dict, "objet JSON attendu")

    def unsupported_request(self, option):
        if option == "json_schema":
            body = self.chat_body(response_format={"type": "json_schema", "json_schema": {
                "name": "weather", "schema": SCHEMA, "strict": True}})
            status, contains = 501, "json_schema"
        elif option == "n":
            body, status, contains = self.chat_body(n=2), 400, "n"
        else:
            choice = "required" if option == "required" else {"type": "function", "function": {"name": "get_weather"}}
            body, status, contains = self.tool_body(choice=choice), 400, "tool_choice"
        rejected(lambda: self.clients.openai.chat.completions.create(**body), status, contains=contains)

    def embed(self, many=False):
        inputs = ["Bonjour Paris", "Bonsoir Brest"] if many else "Bonjour Paris"
        value = decoded(self.clients.openai.embeddings.with_raw_response.create(
            model=self.model, input=inputs, encoding_format="float"))
        c.embeddings(value, self.model, 2 if many else 1)

    def speech(self):
        with self.clients.openai.audio.speech.with_streaming_response.create(
            model=self.model, input=AUDIO_TEXT, voice="alloy", response_format="wav"
        ) as response:
            c.require(response.headers.get("content-type", "").startswith("audio/wav"), "Content-Type WAV")
            audio = response.read()
        with wave.open(io.BytesIO(audio), "rb") as wav:
            c.require(wav.getnchannels() == 1 and wav.getsampwidth() == 2, "WAV mono PCM16 requis")
            c.require(8000 <= wav.getframerate() <= 192000 and wav.getnframes() > 0, "WAV vide/fréquence")
            frames = wav.readframes(wav.getnframes())
            c.require(len(frames) == wav.getnframes() * 2 and any(frames), "WAV tronqué/silencieux")
        return audio

    def transcribe(self, audio, expected, fmt):
        raw = self.clients.openai.audio.transcriptions.with_raw_response.create(
            model="whisper", file=("probe.wav", audio, "audio/wav"), language="fr", response_format=fmt)
        parsed = raw.parse()
        content_type = raw.headers.get("content-type", "")
        if fmt == "json":
            c.require(content_type.startswith("application/json"), "Content-Type transcription JSON")
            value = raw.http_response.json()
            output = c.text(value, "text")
            c.require(parsed.text == output, "décodage transcription SDK")
        else:
            c.require(content_type.startswith("text/plain"), "Content-Type transcription texte")
            c.require(type(parsed) is str and bool(parsed.strip()), "transcription texte vide")
            output = parsed
        words = lambda text: re.findall(r"\w+", text.casefold())
        score = SequenceMatcher(None, words(expected), words(output), autojunk=False).ratio()
        c.require(score >= self.similarity, f"similarité de mots {score:.3f} < {self.similarity:.3f}")
        return {"word_similarity": round(score, 4)}

    def audio(self, fmt):
        if self.family == "audio":
            audio, expected = self.speech(), AUDIO_TEXT
        else:
            audio = local_path("E2E_AUDIO_WAV", directory=False).read_bytes()
            expected = os.environ.get("E2E_AUDIO_TEXT", "")
            c.require(bool(expected.strip()), "E2E_AUDIO_TEXT requis pour vérifier la transcription")
        return self.transcribe(audio, expected, fmt)

    def error_wire(self, endpoint, option):
        anthropic = endpoint == MESSAGES
        if option == "auth_x_api_key_only":
            status, _, payload = self.clients.wire(endpoint, self.chat_body(), key=None,
                headers={"x-api-key": self.clients.key, "anthropic-version": "2023-06-01"})
            c.require(status == 200, f"x-api-key refusé: HTTP {status}")
            c.anthropic(json.loads(payload), self.model, BUDGET)
            return
        elif option.startswith("auth_"):
            key = None if option == "auth_missing" else "e2e-invalid"
            status, _, payload = self.clients.wire(endpoint, self.chat_body(), key=key)
            expected, contains = 401, None
        elif option == "body_invalid":
            status, _, payload = self.clients.wire(endpoint, b'{"messages":')
            expected, contains = 400, None
        elif option == "body_type":
            body = self.chat_body()
            body["messages"] = 17
            status, _, payload = self.clients.wire(endpoint, body)
            expected, contains = 400, None
        elif option == "body_excessive":
            status, value = self.clients.excessive(endpoint)
            payload, expected, contains = json.dumps(value), 400, "body"
        else:
            raise AssertionError("cas d'erreur inconnu")
        c.require(status == expected, f"HTTP attendu {expected}, reçu {status}")
        value = json.loads(payload)
        c.error(value, anthropic, contains)
        expected_type = ("authentication_error" if status == 401 else "invalid_request_error") if anthropic else (
            "invalid_request_error" if status == 400 else "saragossa_error")
        c.require(value["error"]["type"] == expected_type, "type d'erreur incorrect")

    def unknown(self, anthropic):
        body = self.chat_body()
        body["model"] = "e2e-unknown-model"
        resource = self.clients.anthropic.messages if anthropic else self.clients.openai.chat.completions
        rejected(lambda: resource.create(**sdk_body(body, anthropic)), 404, anthropic, "modèle inconnu")

    def unloaded(self, endpoint):
        if endpoint == EMBED:
            call = lambda: self.clients.openai.embeddings.create(model="absent", input="Bonjour", encoding_format="float")
        elif endpoint == SPEECH:
            call = lambda: self.clients.openai.audio.speech.create(model="absent", input="Bonjour", voice="alloy")
        else:
            audio = io.BytesIO()
            with wave.open(audio, "wb") as wav:
                wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                wav.writeframes(b"\x00\x00" * 160)
            call = lambda: self.clients.openai.audio.transcriptions.create(
                model="absent", file=("silence.wav", audio.getvalue(), "audio/wav"))
        rejected(call, 400, contains="configur")

    def build(self):
        from endpoints import add_cases
        add_cases(self)
        chat_disabled = "Profil sans LLM; lancer un profil chat pour ce cas."
        self.add("/v1/models", "list", lambda: c.models(decoded(self.clients.openai.models.with_raw_response.list()),
                 [self.model] if self.family == "chat" else []), smoke=True)
        for endpoint, anthropic in ((CHAT, False), (MESSAGES, True)):
            for option in ("simple", "system", "multi_turn", "max_tokens"):
                self.add(endpoint, option, lambda a=anthropic, o=option: self.normal(o, a),
                         "" if self.family == "chat" else chat_disabled, smoke=option in {"simple", "max_tokens"})
            self.add(endpoint, "stop_sequences" if anthropic else "stop", lambda a=anthropic: self.stops(a),
                     "" if self.family == "chat" else chat_disabled)
            self.add(endpoint, "stream", lambda a=anthropic: self.stream(a),
                     "" if self.family == "chat" else chat_disabled, smoke=True)
            for option in ("auth_missing", "auth_invalid", "body_invalid", "body_type", "body_excessive"):
                self.add(endpoint, option, lambda e=endpoint, o=option: self.error_wire(e, o), smoke=True)
            self.add(endpoint, "model_unknown", lambda a=anthropic: self.unknown(a), smoke=True)
        for option, action in (("seed_temperature_zero", self.deterministic),
                               ("stream_usage", lambda: self.stream(include_usage=True)),
                               ("json_object", self.json_object),
                               ("n_rejected", lambda: self.unsupported_request("n"))):
            self.add(CHAT, option, action, "" if self.family == "chat" else chat_disabled,
                     smoke=option != "seed_temperature_zero")
        self.add(CHAT, "json_schema", lambda: self.unsupported_request("json_schema"),
                 "src/serve/protocol.rs: json_schema refusé avec HTTP 501; contrôle du rejet.", smoke=True,
                 verify_rejection=True)
        self.add(MESSAGES, "auth_x_api_key_only", lambda: self.error_wire(MESSAGES, "auth_x_api_key_only"),
                 "" if self.family == "chat" else chat_disabled, smoke=True)
        no_tools = "" if self.supports_tools else (
            "src/serve/tool_output.rs: outils réservés à flint, Qwen 3.5/3.6/3.8 et Gemma 4.")
        for option, action in (
            ("tools_auto", lambda: self.tool(choice="auto")),
            ("tools_none", lambda: self.tool(choice="none")),
            ("tools_roundtrip", lambda: self.tool(choice="auto", roundtrip=True)),
            ("tools_stream", lambda: self.stream(tools=True)),
        ):
            self.add(CHAT, option, action, no_tools)
        for choice in ("required", "named"):
            reason = no_tools or "src/serve/protocol.rs active_tools: choix contraint refusé HTTP 400; rejet vérifié."
            self.add(CHAT, "tools_" + choice,
                     (lambda o=choice: self.unsupported_request(o)) if not no_tools else None, reason,
                     verify_rejection=not no_tools)
        for option, action in (
            ("tools_auto", lambda: self.tool(True, {"type": "auto"})),
            ("tools_roundtrip", lambda: self.tool(True, roundtrip=True)),
            ("tools_stream_order", lambda: self.stream(True, tools=True)),
        ):
            self.add(MESSAGES, option, action, "" if self.family == "chat" else chat_disabled)
        for choice in ("any", "tool", "none"):
            self.add(MESSAGES, "tools_" + choice, unsupported=(
                "src/serve/anthropic.rs AnthropicMessagesRequest: tool_choice absent/ignoré; "
                "seule la sélection automatique par le prompt est disponible."))
        for many in (False, True):
            self.add(EMBED, "list_dimensions_l2_float" if many else "single_dimensions_l2_float",
                     lambda m=many: self.embed(m), "" if self.family == "embed" else "Profil sans embedder E5.")
        for option in ("dimensions_override", "encoding_base64", "usage"):
            self.add(EMBED, option, unsupported="src/serve/embeddings.rs: dimension fixe 384, float seulement, usage absent.")
        self.add(SPEECH, "wav", self.speech, "" if self.family in {"tts", "audio"} else "Profil sans Qwen3-TTS.")
        for fmt in ("mp3", "opus", "aac", "flac", "pcm"):
            self.add(SPEECH, "format_" + fmt, unsupported=(
                "src/serve/audio.rs SpeechRequest: response_format ignoré, WAV; PCM via extension stream seulement."))
        for fmt in ("json", "text"):
            self.add(STT, "roundtrip_" + fmt if self.family == "audio" else "fixture_" + fmt,
                     lambda f=fmt: self.audio(f), "" if self.family in {"stt", "audio"} else (
                         "Profil sans Whisper; profil audio requis pour texte → audio → texte."))
        for fmt in ("verbose_json", "srt", "vtt"):
            self.add(STT, "format_" + fmt, unsupported="src/serve/audio.rs handle_transcription: JSON et text seulement.")
        enabled = {EMBED: self.family == "embed", SPEECH: self.family in {"tts", "audio"},
                   STT: self.family in {"stt", "audio"}}
        for endpoint, active in enabled.items():
            if not active:
                self.add(endpoint, "model_not_loaded", lambda e=endpoint: self.unloaded(e), smoke=True)

    def selected(self, suite):
        return [case for case in self.cases if suite == "full" or case.smoke]
