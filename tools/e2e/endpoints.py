"""Responses and Anthropic token counting through the installed official SDKs."""

import inspect
import json

import contracts as c

RESPONSES = "/v1/responses"
COUNT = "/v1/messages/count_tokens"


def response_contract(value, model):
    from openai.types.responses import Response

    Response.model_validate(value)
    c.require(value["object"] == "response" and value["model"] == model, "identité Responses")
    c.require(value["status"] in {"completed", "incomplete"}, "statut Responses")
    c.require(value["error"] is None, "erreur Responses inattendue")
    usage = value["usage"]
    for field in ("input_tokens", "output_tokens", "total_tokens"):
        c.require(type(usage[field]) is int and usage[field] >= 0, "usage Responses")
    c.require(usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"], "total Responses")
    return value


def stream_contract(events, model):
    from openai.types.responses import ResponseStreamEvent
    from pydantic import TypeAdapter

    adapter = TypeAdapter(ResponseStreamEvent)
    c.require(bool(events) and events[0]["type"] == "response.created", "response.created absent")
    added, done, text, arguments = {}, {}, {}, {}
    terminals = []
    for sequence, event in enumerate(events):
        adapter.validate_python(event)
        c.require(event["sequence_number"] == sequence, "séquence SSE")
        kind = event["type"]
        if kind == "response.output_item.added":
            index = event["output_index"]
            c.require(index == len(added), "ordre output_index")
            added[index] = event["item"]
        elif kind in {"response.output_text.delta", "response.function_call_arguments.delta"}:
            index = event["output_index"]
            c.require(index in added and index not in done, "delta hors item ouvert")
            c.require(event["item_id"] == added[index]["id"], "item_id delta")
            target = text if kind == "response.output_text.delta" else arguments
            target[index] = target.get(index, "") + event["delta"]
        elif kind in {"response.output_text.done", "response.function_call_arguments.done"}:
            index = event["output_index"]
            c.require(index in added and index not in done, "done hors item ouvert")
            c.require(event["item_id"] == added[index]["id"], "item_id done")
            target, field = (text, "text") if kind == "response.output_text.done" else (arguments, "arguments")
            c.require(target.get(index, "") == event[field], "deltas et done différents")
        elif kind == "response.output_item.done":
            index = event["output_index"]
            c.require(index in added and index not in done, "item done absent/dupliqué")
            c.require(event["item"]["id"] == added[index]["id"], "item_id final")
            done[index] = event["item"]
        elif kind in {"response.completed", "response.incomplete", "response.failed"}:
            terminals.append(sequence)
    c.require(terminals == [len(events) - 1], "événement terminal unique et final")
    final = response_contract(events[-1]["response"], model)
    c.require(final["id"] == events[0]["response"]["id"], "response.id change")
    c.require(final["output"] == [done[i] for i in sorted(done)], "output final différent des items SSE")
    c.require(set(added) == set(done), "items non terminés")
    return final


class Endpoints:
    def __init__(self, matrix):
        self.matrix = matrix
        self.clients, self.model = matrix.clients, matrix.model

    def body(self, **options):
        return dict(model=self.model, input="Reply with the single word Bonjour.",
                    max_output_tokens=256, temperature=0, store=False, **options)

    def create(self, body):
        method = self.clients.openai.responses.create
        inspect.signature(method).bind(**body)
        value = method(**body).model_dump(mode="json")
        return response_contract(value, self.model)

    def text(self, stream=False, roles=False):
        body = self.body()
        if roles:
            body.update(instructions="Reply with SAPHIR.", input=[
                {"role":"system", "content":"Use one word."},
                {"role":"developer", "content":"Remember the requested word."},
                {"role":"user", "content":[{"type":"input_text", "text":"Bonjour é 🌍"}]},
                {"role":"assistant", "content":"Bonjour"},
                {"role":"user", "content":"Repeat the requested word."}])
        if stream:
            # Exercise the SDK iterator and the named SSE wire framing independently.
            with self.clients.openai.responses.create(**dict(body, stream=True)) as iterator:
                events = [event.model_dump(mode="json") for event in iterator]
            value = stream_contract(events, self.model)
            stream_contract(self.clients.stream_wire(RESPONSES, body), self.model)
        else:
            value = self.create(body)
        output = "".join(part["text"] for item in value["output"] if item["type"] == "message"
                         for part in item["content"] if part["type"] == "output_text")
        c.require(("saphir" if roles else "bonjour") in output.casefold(), "texte Responses absent")

    def count(self, tools=False):
        from matrix import ANTHROPIC_TOOL
        body = dict(model=self.model, system=[{"type":"text", "text":"Answer briefly."}],
                    messages=[{"role":"user", "content":"Bonjour é 🌍"}])
        if tools:
            body["tools"] = [ANTHROPIC_TOOL]
        method = self.clients.anthropic.messages.count_tokens
        inspect.signature(method).bind(**body)
        counted = method(**body)
        c.require(type(counted.input_tokens) is int and counted.input_tokens > 0, "input_tokens invalide")
        generated = self.clients.anthropic.messages.create(**dict(body, max_tokens=1),
                                                         extra_body={"temperature":0})
        c.require(counted.input_tokens == generated.usage.input_tokens, "comptage différent du prompt généré")

    def reject(self, option):
        import openai
        body = self.body()
        values = {"previous_response_id":"resp_absent", "store":True, "background":True,
                  "include":[], "tools":[{"type":"web_search"}]}
        body[option] = values[option]
        try:
            self.clients.openai.responses.create(**body)
        except openai.APIStatusError as error:
            c.require(error.status_code in {400, 501}, "option non supportée: statut")
            c.error(error.response.json())
            return
        raise AssertionError("option non supportée acceptée: " + option)

    def unknown(self, count=False):
        import openai
        import anthropic
        try:
            if count:
                self.clients.anthropic.messages.count_tokens(model="absent", messages=[{"role":"user","content":"x"}])
            else:
                self.clients.openai.responses.create(model="absent", input="x", store=False)
        except (openai.APIStatusError, anthropic.APIStatusError) as error:
            c.require(error.status_code == 404, "modèle inconnu: statut")
            c.error(error.response.json(), anthropic=count)
            return
        raise AssertionError("modèle inconnu accepté")

    def tool(self, stream=False):
        from matrix import FUNCTION, TOOL_PROMPT
        body = self.body()
        body.update(input=TOOL_PROMPT, tools=[dict(type="function", strict=False, **FUNCTION)])
        if stream:
            with self.clients.openai.responses.create(**dict(body, stream=True)) as iterator:
                first = stream_contract([event.model_dump(mode="json") for event in iterator], self.model)
        else:
            first = self.create(body)
        calls = [item for item in first["output"] if item["type"] == "function_call"]
        c.require(bool(calls), "function_call absent")
        history = [{"role":"user", "content":TOOL_PROMPT}] + first["output"]
        for call in calls:
            c.require(call["name"] == "get_weather" and "paris" in json.loads(call["arguments"])["city"].casefold(), "appel météo incorrect")
            history.append({"type":"function_call_output", "call_id":call["call_id"], "output":'{"temperature":17}'})
        body.update(input=history, tools=[], tool_choice="none")
        final = self.create(body)
        c.require("17" in json.dumps(final["output"], ensure_ascii=False), "retour outil absent")


def add_cases(matrix):
    endpoint = Endpoints(matrix)
    unavailable = "" if matrix.family == "chat" else "Profil sans LLM."
    matrix.add(COUNT, "exact_prompt", endpoint.count, unavailable, smoke=True)
    no_tools = "" if matrix.supports_tools else "src/serve/tool_output.rs: outils non pris en charge pour cette famille."
    matrix.add(COUNT, "tools_exact_prompt", lambda: endpoint.count(True), no_tools)
    matrix.add(RESPONSES, "text", endpoint.text, unavailable, smoke=True)
    matrix.add(RESPONSES, "roles_instructions", lambda: endpoint.text(roles=True), unavailable)
    matrix.add(RESPONSES, "stream_order_usage", lambda: endpoint.text(stream=True), unavailable, smoke=True)
    matrix.add(COUNT, "model_unknown", lambda: endpoint.unknown(True), smoke=True)
    matrix.add(RESPONSES, "model_unknown", endpoint.unknown, smoke=True)
    for option in ("previous_response_id", "store", "background", "tools", "include"):
        matrix.add(RESPONSES, "reject_" + option, lambda o=option: endpoint.reject(o), smoke=True)
    matrix.add(RESPONSES, "function_roundtrip", endpoint.tool, no_tools)
    matrix.add(RESPONSES, "function_stream_roundtrip", lambda: endpoint.tool(True), no_tools)
