"""Official SDK clients plus stdlib wire probes for malformed requests and SSE."""

import http.client
import json
import urllib.error
import urllib.request
from urllib.parse import urlsplit

import contracts as c

VERSIONS = {"openai": "3.49.0", "anthropic": "2.8.0"}
MAX_RESPONSE = 32 * 1024 * 1024


class Clients:
    def __init__(self, base, key, timeout):
        import openai
        import anthropic

        for module in (openai, anthropic):
            c.require(module.__version__ == VERSIONS[module.__name__],
                      f"version SDK requise: {module.__name__}=={VERSIONS[module.__name__]}")
        self.base, self.key, self.timeout = base, key, timeout
        self.openai = openai.OpenAI(
            base_url=base + "/v1", api_key=key, timeout=timeout, max_retries=0,
            http_client=openai.DefaultHttpxClient(trust_env=False),
        )
        self.anthropic = anthropic.Anthropic(
            base_url=base, auth_token=key, timeout=timeout, max_retries=0,
            http_client=anthropic.DefaultHttpxClient(trust_env=False),
        )
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def close(self):
        self.openai.close()
        self.anthropic.close()

    def wire(self, endpoint, body=None, key=True, method="POST", headers=None):
        request_headers = {"Content-Type": "application/json"}
        if key is not None:
            secret = self.key if key is True else key
            request_headers["Authorization"] = "Bearer " + secret
            if endpoint == "/v1/messages":
                request_headers["anthropic-version"] = "2023-06-01"
        request_headers.update(headers or {})
        data = json.dumps(body).encode() if isinstance(body, dict) else body
        request = urllib.request.Request(self.base + endpoint, data=data, method=method,
                                         headers=request_headers)
        try:
            response = self.opener.open(request, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            payload = response.read(MAX_RESPONSE + 1)
            c.require(len(payload) <= MAX_RESPONSE, "réponse trop volumineuse")
            return response.status, dict(response.headers), payload

    def excessive(self, endpoint):
        """Announce 16 MiB + 1 without allocating or uploading a large body."""
        target = urlsplit(self.base)
        connection = http.client.HTTPConnection(target.hostname, target.port, timeout=self.timeout)
        try:
            connection.putrequest("POST", endpoint)
            connection.putheader("Authorization", "Bearer " + self.key)
            connection.putheader("Content-Type", "application/json")
            connection.putheader("Content-Length", str(16 * 1024 * 1024 + 1))
            connection.endheaders()
            response = connection.getresponse()
            return response.status, json.loads(response.read(MAX_RESPONSE))
        finally:
            connection.close()

    def stream_wire(self, endpoint, body):
        status, headers, payload = self.wire(endpoint, dict(body, stream=True))
        c.require(status == 200, f"SSE HTTP {status}")
        content_type = next((v for k, v in headers.items() if k.lower() == "content-type"), "")
        c.require(content_type.startswith("text/event-stream"), "Content-Type SSE")
        frames = list(c.sse_frames(payload.splitlines(keepends=True)))
        if endpoint == "/v1/chat/completions":
            c.require(bool(frames) and frames[-1][1] == "[DONE]", "[DONE] absent ou non final")
            c.require(sum(data == "[DONE]" for _, data in frames) == 1, "[DONE] dupliqué")
            return [json.loads(data) for _, data in frames[:-1]]
        events = []
        for name, data in frames:
            value = json.loads(data)
            c.require(name == value.get("type"), "event SSE/type JSON incohérent")
            events.append(value)
        return events


def decoded(raw):
    """Exercise official SDK decoding AND validate the uncoerced JSON separately."""
    parsed = raw.parse()
    c.require(parsed is not None, "SDK: réponse absente")
    return raw.http_response.json()


def rejected(call, status, anthropic=False, contains=None):
    import openai
    import anthropic as sdk

    try:
        call()
    except (openai.APIStatusError, sdk.APIStatusError) as exc:
        c.require(exc.status_code == status, f"HTTP attendu {status}, reçu {exc.status_code}")
        c.error(exc.response.json(), anthropic, contains)
        expected_type = ({400: "invalid_request_error", 401: "authentication_error", 404: "not_found_error"}
                         .get(status, "api_error")) if anthropic else (
                             "invalid_request_error" if status == 400 else "saragossa_error")
        c.require(exc.response.json()["error"]["type"] == expected_type, "type d'erreur SDK incorrect")
        if contains == "n":
            import re
            c.require(re.search(r"\bn\b", exc.response.json()["error"]["message"]),
                      "erreur n=2 sans mention du paramètre n")
        return
    raise AssertionError(f"requête acceptée au lieu de HTTP {status}")
