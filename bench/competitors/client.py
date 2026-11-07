"""Client SSE du bench concurrents : une requête chat, chronométrée, stdlib seule.

Horloge : `time.perf_counter_ns`. La connexion TCP est ouverte avant t0 ;
t0 est pris juste avant l'envoi de la requête. Le HTTP/1.1 est lu sur le
socket brut (pas de tampon de bibliothèque) : chaque ligne SSE est datée à
l'arrivée de son dernier octet. Les mesures dérivées (TTFT, débit, latences)
sont calculées par `analyze.py` à partir des événements bruts conservés ici.
"""

import http.client
import json
import socket
import time

# Statuts d'une requête ; « ok » seul est une observation complète.
OK = "ok"
HTTP_ERROR = "http_error"
TIMEOUT = "timeout"
CONNECTION_LOST = "connection_lost"
STREAM_INCOMPLETE = "stream_incomplete"
# Événement `error` ou JSON invalide dans le flux : terminal, même si la fin
# du flux arrive ensuite (le contenu est alors incomplet ou inconnu).
STREAM_ERROR = "stream_error"


class _StreamError(Exception):
    pass


class _Wire:
    """Lecture HTTP/1.1 sur socket brut : chaque octet porte l'heure du `recv` qui l'a livré.

    Le tampon est entièrement consommé avant chaque nouveau `recv`, donc une
    ligne SSE est datée à l'arrivée de son dernier octet.
    """

    def __init__(self, sock, t0, deadline_ns, recv=None):
        self.sock = sock
        self.t0 = t0
        self.deadline_ns = deadline_ns
        self.buf = bytearray()
        self.t = 0
        self._recv = recv or self._socket_recv

    def _socket_recv(self, remaining):
        self.sock.settimeout(remaining)
        return self.sock.recv(65536)

    def fill(self):
        remaining = (self.deadline_ns - (time.perf_counter_ns() - self.t0)) / 1e9
        if remaining <= 0:
            raise TimeoutError
        data = self._recv(remaining)
        self.t = time.perf_counter_ns() - self.t0
        if not data:
            raise EOFError
        self.buf += data

    def line(self):
        while b"\r\n" not in self.buf:
            self.fill()
        index = self.buf.index(b"\r\n")
        out = bytes(self.buf[:index])
        del self.buf[:index + 2]
        return out

    def take(self, limit):
        if not self.buf:
            self.fill()
        out = bytes(self.buf[:limit])
        del self.buf[:limit]
        return out

    def body(self, chunked, length):
        """Morceaux décodés du corps, avec l'heure d'arrivée de leur dernier octet."""
        if chunked:
            while True:
                size = int(self.line().split(b";")[0], 16)
                if size == 0:
                    return
                while size:
                    piece = self.take(size)
                    size -= len(piece)
                    yield self.t, piece
                self.line()
        remaining = length
        while remaining is None or remaining > 0:
            try:
                piece = self.take(65536 if remaining is None else remaining)
            except EOFError:
                return
            if remaining is not None:
                remaining -= len(piece)
            yield self.t, piece


def chat_stream(host, port, body, headers=None, deadline_s=3600.0):
    """Envoie `body` en SSE et rend l'enregistrement brut de la réponse.

    `deadline_s` est absolu : de l'envoi à la fin du flux.
    """
    payload = json.dumps(body).encode("utf-8")
    lines = [
        "POST /v1/chat/completions HTTP/1.1", f"Host: {host}:{port}",
        "Content-Type: application/json", "Accept: text/event-stream",
        f"Content-Length: {len(payload)}", "Connection: close",
    ] + [f"{k}: {v}" for k, v in (headers or {}).items()]
    request = ("\r\n".join(lines) + "\r\n\r\n").encode("ascii") + payload
    record = {"events": [], "content": "", "reasoning": "", "usage": None,
              "finish_reason": None, "timings": None, "done": False}
    sock = None
    record["t0_wall"] = time.time()
    t0 = time.perf_counter_ns()
    # Origine commune aux échantillons mémoire (même horloge, valeur absolue).
    record["t0_ns"] = t0
    try:
        sock = socket.create_connection((host, port), timeout=60)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # t0 au plus près de l'envoi : l'ouverture de connexion ne compte pas.
        t0 = time.perf_counter_ns()
        record["t0_ns"] = t0
        wire = _Wire(sock, t0, int(deadline_s * 1e9))
        sock.sendall(request)
        _read_response(wire, record)
    except (socket.timeout, TimeoutError):
        record["status"] = TIMEOUT
    except _StreamError as error:
        record["status"] = STREAM_ERROR
        record["error"] = str(error)
    except (EOFError, OSError, ValueError, IndexError) as error:
        record["status"] = CONNECTION_LOST
        record["error"] = repr(error)
    finally:
        record["t_end_ns"] = time.perf_counter_ns() - t0
        if sock is not None:
            sock.close()
    if "status" not in record:
        complete = record["done"] and record["finish_reason"] is not None
        record["status"] = OK if complete else STREAM_INCOMPLETE
    return record


def _read_response(wire, record):
    status = wire.line().decode("latin-1")
    response_headers = {}
    while True:
        header = wire.line()
        if not header:
            break
        key, _, value = header.decode("latin-1").partition(":")
        response_headers[key.strip().lower()] = value.strip()
    record["t_headers_ns"] = wire.t
    record["http_status"] = int(status.split()[1])
    record["response_headers"] = {k: v for k, v in response_headers.items()
                                  if k.startswith(("x-", "content-type", "transfer-encoding"))}
    chunked = "chunked" in response_headers.get("transfer-encoding", "").lower()
    length = response_headers.get("content-length")
    pieces = wire.body(chunked, int(length) if length else None)
    if record["http_status"] != 200:
        body = b"".join(piece for _, piece in pieces)
        record["error_body"] = body[:65536].decode("utf-8", "replace")
        record["status"] = HTTP_ERROR
        return
    pending = b""
    for t, piece in pieces:
        pending += piece
        *complete, pending = pending.split(b"\n")
        for line in complete:
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                record["done"] = True
                return
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                record["events"].append({"t_ns": t, "bytes": len(line), "invalid": True,
                                         "raw": data[:2000].decode("utf-8", "replace")})
                raise _StreamError("JSON invalide dans le flux") from None
            _absorb(chunk, t, len(line), record)
            if "error" in chunk:
                raise _StreamError(f"événement error: {chunk['error']}")


def _absorb(chunk, t, nbytes, record):
    """Ajoute un chunk : seuls `content` et `reasoning_content` non vides comptent comme sortie."""
    event = {"t_ns": t, "bytes": nbytes, "content_chars": 0, "reasoning_chars": 0, "text": ""}
    if "error" in chunk:
        event["error"] = chunk["error"]
        record["stream_error"] = chunk["error"]
    for choice in chunk.get("choices") or []:
        delta = choice.get("delta") or {}
        content = delta.get("content") or ""
        reasoning = delta.get("reasoning_content") or ""
        record["content"] += content
        record["reasoning"] += reasoning
        event["content_chars"] += len(content)
        event["reasoning_chars"] += len(reasoning)
        # Fragment brut conservé : le comptage par événement se recalcule.
        event["text"] += reasoning + content
        if choice.get("finish_reason"):
            record["finish_reason"] = choice["finish_reason"]
            event["finish_reason"] = choice["finish_reason"]
    if chunk.get("usage"):
        record["usage"] = chunk["usage"]
        event["usage"] = chunk["usage"]
    if chunk.get("timings"):
        record["timings"] = chunk["timings"]
    record["events"].append(event)


def http_json(host, port, method, path, body=None, headers=None, timeout=600.0):
    """Petite requête JSON hors mesure (santé, /tokenize, /apply-template)."""
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        payload = None if body is None else json.dumps(body).encode("utf-8")
        conn.request(method, path, body=payload,
                     headers={"Content-Type": "application/json", **(headers or {})})
        resp = conn.getresponse()
        raw = resp.read()
        try:
            return resp.status, json.loads(raw) if raw else None
        except json.JSONDecodeError:
            return resp.status, raw.decode("utf-8", "replace")
    finally:
        conn.close()


def _self_check():
    # Réponse chunked découpée en deux recv : le second événement et [DONE]
    # arrivent ensemble et partagent leur heure.
    frames = [
        b'HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n'
        b'30\r\ndata: {"choices":[{"delta":{"content":"ab"}}]}\n\n\r\n',
        b'59\r\ndata: {"choices":[{"delta":{},"finish_reason":"stop"}],'
        b'"usage":{"completion_tokens":2}}\n\n\r\n'
        b'e\r\ndata: [DONE]\n\n\r\n0\r\n\r\n',
    ]
    wire = _Wire(None, time.perf_counter_ns(), int(1e9), recv=lambda _: frames.pop(0))
    record = {"events": [], "content": "", "reasoning": "", "usage": None,
              "finish_reason": None, "timings": None, "done": False}
    _read_response(wire, record)
    assert record["http_status"] == 200 and record["done"]
    assert record["content"] == "ab" and record["finish_reason"] == "stop"
    assert record["usage"] == {"completion_tokens": 2}
    assert [e["content_chars"] for e in record["events"]] == [2, 0]
    assert record["events"][0]["t_ns"] <= record["events"][1]["t_ns"]
    assert record["events"][0]["text"] == "ab"
    # Un événement error est terminal même si finish_reason et [DONE] suivent.
    frames = [b'HTTP/1.1 200 OK\r\nContent-Length: 200\r\n\r\n'
              b'data: {"error":{"message":"x"}}\n\n'
              b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n']
    wire = _Wire(None, time.perf_counter_ns(), int(1e9), recv=lambda _: frames.pop(0))
    record = {"events": [], "content": "", "reasoning": "", "usage": None,
              "finish_reason": None, "timings": None, "done": False}
    try:
        _read_response(wire, record)
        raise AssertionError("événement error non terminal")
    except _StreamError:
        pass
    assert record["finish_reason"] is None and not record["done"]


if __name__ == "__main__":
    _self_check()
    print("client: auto-test ok")
