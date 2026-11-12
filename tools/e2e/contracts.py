"""Strict wire contracts, independent of SDK coercion and inference."""

import json
import math


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def field(value, key, kind):
    require(isinstance(value, dict), "objet JSON requis")
    require(key in value, f"champ absent: {key}")
    item = value[key]
    require(type(item) is kind, f"type incorrect: {key}, attendu {kind.__name__}")
    return item


def text(value, key):
    item = field(value, key, str)
    require(bool(item.strip()), f"champ vide: {key}")
    return item


def number(value, key):
    item = field(value, key, int)
    require(item >= 0, f"compteur négatif: {key}")
    return item


def usage(value, anthropic=False):
    names = ("input_tokens", "output_tokens") if anthropic else (
        "prompt_tokens", "completion_tokens", "total_tokens"
    )
    counts = [number(value, name) for name in names]
    if not anthropic:
        require(counts[0] + counts[1] == counts[2], "usage.total_tokens incohérent")
    return counts


def models(value, expected):
    require(text(value, "object") == "list", "models.object")
    ids = []
    for model in field(value, "data", list):
        ids.append(text(model, "id"))
        require(text(model, "object") == "model", "model.object")
        number(model, "created")
        text(model, "owned_by")
    require(len(ids) == len(set(ids)), "identifiants modèles dupliqués")
    require(set(ids) == set(expected), "registre des modèles inattendu")


def tool_calls(calls):
    require(type(calls) is list and bool(calls), "tool_calls absent ou vide")
    ids = []
    for call in calls:
        ids.append(text(call, "id"))
        require(text(call, "type") == "function", "tool_call.type")
        function = field(call, "function", dict)
        require(text(function, "name") == "get_weather", "nom outil inattendu")
        args = json.loads(text(function, "arguments"))
        require(type(args) is dict and args.get("city") == "Paris", "arguments outil")
    require(len(ids) == len(set(ids)), "ids outils dupliqués")


def chat(value, model, budget, tools=False, allow_empty=False):
    text(value, "id")
    require(text(value, "object") == "chat.completion", "chat.object")
    number(value, "created")
    require(text(value, "model") == model, "chat.model")
    counts = usage(field(value, "usage", dict))
    require(counts[0] > 0 and 0 < counts[1] <= budget, "usage/budget chat")
    choices = field(value, "choices", list)
    require(len(choices) == 1, "une choice requise")
    choice = choices[0]
    require(number(choice, "index") == 0, "choice.index")
    reason = text(choice, "finish_reason")
    require(reason in {"stop", "length", "tool_calls"}, "finish_reason inconnu")
    message = field(choice, "message", dict)
    require(text(message, "role") == "assistant", "message.role")
    require("content" in message, "message.content absent")
    require(message["content"] is None or type(message["content"]) is str, "content type")
    if tools:
        require(reason == "tool_calls", "finish_reason attendu: tool_calls")
        tool_calls(message.get("tool_calls"))
    else:
        require(not message.get("tool_calls"), "outil non demandé")
        require(reason != "tool_calls", "finish_reason sans outil")
        if not allow_empty:
            text(message, "content")
    return message


def anthropic(value, model, budget, tools=False, allow_empty=False):
    text(value, "id")
    require(text(value, "type") == "message", "message.type")
    require(text(value, "role") == "assistant", "message.role")
    require(text(value, "model") == model, "message.model")
    counts = usage(field(value, "usage", dict), True)
    require(counts[0] > 0 and 0 < counts[1] <= budget, "usage/budget Anthropic")
    reason = text(value, "stop_reason")
    require(reason in {"end_turn", "max_tokens", "stop_sequence", "tool_use"}, "stop_reason")
    require("stop_sequence" in value, "stop_sequence absent")
    if reason == "stop_sequence":
        text(value, "stop_sequence")
    else:
        require(value["stop_sequence"] is None, "stop_sequence inattendue")
    content = field(value, "content", list)
    require(bool(content), "content vide")
    calls, ids, texts = [], [], []
    for block in content:
        kind = text(block, "type")
        if kind == "text":
            texts.append(field(block, "text", str))
        elif kind == "tool_use":
            ids.append(text(block, "id"))
            require(text(block, "name") == "get_weather", "tool_use.name")
            require(field(block, "input", dict).get("city") == "Paris", "tool_use.input")
            calls.append(block)
        else:
            raise AssertionError("type de bloc Anthropic inattendu")
    require(len(ids) == len(set(ids)), "tool_use.id dupliqué")
    require(bool(calls) == tools, "présence tool_use inattendue")
    require((reason == "tool_use") == tools, "stop_reason/tool_use incohérent")
    if not tools and not allow_empty:
        require(bool("".join(texts).strip()), "texte Anthropic vide")
    return content


def error(value, anthropic=False, contains=None):
    if anthropic:
        require(text(value, "type") == "error", "enveloppe erreur Anthropic")
    err = field(value, "error", dict)
    text(err, "type")
    message = text(err, "message")
    if contains:
        require(contains.casefold() in message.casefold(), f"erreur non explicite: {contains}")


def embeddings(value, model, count, dimension=384):
    require(text(value, "object") == "list", "embeddings.object")
    require(text(value, "model") == model, "embeddings.model")
    data = field(value, "data", list)
    require(len(data) == count, "nombre d'embeddings")
    for index, item in enumerate(data):
        require(text(item, "object") == "embedding", "embedding.object")
        require(number(item, "index") == index, "embedding.index/ordre")
        vector = field(item, "embedding", list)
        require(len(vector) == dimension, "dimension embedding")
        require(all(type(x) in (int, float) and math.isfinite(x) for x in vector), "vecteur non fini")
        require(abs(math.sqrt(sum(x * x for x in vector)) - 1) < 1e-3, "normalisation L2")


def sse_frames(lines):
    """Decode complete SSE frames, including the terminator hidden by OpenAI."""
    event, data = None, []
    for line in lines:
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="strict")
        line = line.rstrip("\r\n")
        if not line:
            if data:
                yield event, "\n".join(data)
            event, data = None, []
        elif line.startswith("event:"):
            event = line[6:].lstrip(" ")
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
        elif not line.startswith((":", "id:", "retry:")):
            raise AssertionError("ligne SSE invalide")
    require(not data, "trame SSE tronquée (séparateur final absent)")


def openai_stream(chunks, model, budget, include_usage=False, tools=False):
    require(bool(chunks), "flux OpenAI vide")
    identifier, created, finished, role_seen = None, None, False, False
    content, calls, usages = [], {}, []
    for position, chunk in enumerate(chunks):
        cid = text(chunk, "id")
        stamp = number(chunk, "created")
        if identifier is None:
            identifier, created = cid, stamp
        require(cid == identifier, "id instable dans le flux")
        require(stamp == created, "created instable dans le flux")
        require(text(chunk, "object") == "chat.completion.chunk", "chunk.object")
        require(text(chunk, "model") == model, "chunk.model")
        choices = field(chunk, "choices", list)
        if not choices:
            require(include_usage and finished and position == len(chunks) - 1, "chunk usage hors ordre")
            counts = usage(field(chunk, "usage", dict))
            require(counts[0] > 0 and 0 < counts[1] <= budget, "usage stream/budget")
            usages.append(counts)
            continue
        require(not finished and len(choices) == 1, "choice après fin ou cardinalité invalide")
        if include_usage:
            require("usage" in chunk, "usage:null absent d'un chunk intermédiaire")
        require(chunk.get("usage") is None, "usage avant chunk final")
        choice = choices[0]
        require(number(choice, "index") == 0, "stream choice.index")
        delta = field(choice, "delta", dict)
        if "role" in delta:
            require(delta["role"] == "assistant" and not role_seen and position == 0, "delta.role/ordre")
            role_seen = True
        if delta.get("content") is not None:
            content.append(field(delta, "content", str))
        for call in delta.get("tool_calls") or []:
            index = number(call, "index")
            if index not in calls:
                require(index == len(calls), "index outil non contigu")
                calls[index] = {"id": text(call, "id"), "type": text(call, "type"),
                                "function": {"name": "", "arguments": ""}}
            accumulated = calls[index]
            for key in ("id", "type"):
                if key in call:
                    require(call[key] == accumulated[key], "identité outil instable")
            function = field(call, "function", dict)
            for key in ("name", "arguments"):
                if function.get(key) is not None:
                    accumulated["function"][key] += field(function, key, str)
        require("finish_reason" in choice, "finish_reason absent du chunk")
        reason = choice["finish_reason"]
        if reason is not None:
            require(reason in ({"tool_calls"} if tools else {"stop", "length"}), "stream finish_reason")
            finished = True
    require(role_seen and finished, "flux sans role ou finish_reason final")
    require(len(usages) == int(include_usage), "chunk usage manquant/dupliqué")
    if tools:
        tool_calls(list(calls.values()))
    else:
        require(not calls and bool("".join(content).strip()), "contenu stream vide/inattendu")
    return "".join(content)


def anthropic_stream(events, model, budget, tools=False):
    require(len(events) >= 5, "flux Anthropic incomplet")
    require(events[0].get("type") == "message_start", "message_start doit être premier")
    initial = field(events[0], "message", dict)
    text(initial, "id")
    require(initial.get("type") == "message" and initial.get("role") == "assistant", "message_start schéma")
    require(text(initial, "model") == model, "stream model")
    require(field(initial, "content", list) == [], "content initial non vide")
    require("stop_reason" in initial and "stop_sequence" in initial, "champs stop initiaux absents")
    require(initial.get("stop_reason") is None and initial.get("stop_sequence") is None, "stop initial")
    usage(field(initial, "usage", dict), True)
    blocks, open_index, final, stopped = [], None, None, False
    partial = ""
    for event in events[1:]:
        kind = text(event, "type")
        require(not stopped, "événement après message_stop")
        if kind == "ping":
            continue
        if kind == "content_block_start":
            require(open_index is None and final is None, "début bloc hors ordre")
            require(number(event, "index") == len(blocks), "index bloc non contigu")
            block = field(event, "content_block", dict).copy()
            require(block.get("type") in {"text", "tool_use"}, "type bloc initial")
            if block["type"] == "text":
                field(block, "text", str)
            else:
                text(block, "id")
                text(block, "name")
                field(block, "input", dict)
            blocks.append(block)
            open_index, partial = len(blocks) - 1, ""
        elif kind in {"content_block_delta", "content_block_stop"}:
            require(open_index is not None and number(event, "index") == open_index, "delta/stop sans bloc ouvert")
            block = blocks[open_index]
            if kind == "content_block_stop":
                if block["type"] == "tool_use" and partial:
                    block["input"] = json.loads(partial)
                open_index = None
            else:
                delta = field(event, "delta", dict)
                expected = "text_delta" if block["type"] == "text" else "input_json_delta"
                require(text(delta, "type") == expected, "delta incompatible avec bloc")
                if expected == "text_delta":
                    block["text"] += field(delta, "text", str)
                else:
                    partial += field(delta, "partial_json", str)
        elif kind == "message_delta":
            require(open_index is None and final is None and bool(blocks), "message_delta hors ordre")
            final = field(event, "delta", dict)
            require("stop_reason" in final and "stop_sequence" in final, "champs stop finaux absents")
            final_usage = field(event, "usage", dict)
            number(final_usage, "output_tokens")
        elif kind == "message_stop":
            require(final is not None and open_index is None, "message_stop prématuré")
            stopped = True
        else:
            raise AssertionError("événement Anthropic inattendu")
    require(stopped, "message_stop absent")
    result = dict(initial, content=blocks, **final)
    result["usage"] = dict(initial["usage"], **final_usage)
    anthropic(result, model, budget, tools)
    return result
