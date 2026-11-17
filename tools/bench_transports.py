"""Latences des transports `saragossa serve` : temps au premier octet et latence par trame.

Sous-commandes (stdlib seule, serveur déjà lancé, rien n'est démarré ici) :
  sse  POST /v1/chat/completions en flux : TTFB, premier événement, écart entre événements.
  ws   /v1/audio/enhance/ws : TTFB (handshake 101), première trame PCM, aller-retour par trame.
HTTP/2 et QUIC/WebTransport n'ont pas de client stdlib : ils se mesurent par les tests
ignorés `h2_abba_latency` et `webtransport_abba_latency` (voir docs/voice-isolation.md).
Résultat : une ligne JSON par métrique, en millisecondes (p50, p95, n).
"""
import argparse
import base64
import http.client
import json
import os
import socket
import struct
import sys
import time

HOP = 480  # échantillons f32 par trame, comme le banc ABBA Rust


def load1():
    return os.getloadavg()[0]


def pct(values, p):
    ordered = sorted(values)
    return ordered[(len(ordered) - 1) * p // 100]


def report(transport, metric, values):
    print(json.dumps({
        "transport": transport, "metric": metric, "n": len(values),
        "p50_ms": round(pct(values, 50) * 1e3, 3), "p95_ms": round(pct(values, 95) * 1e3, 3),
    }))


def gate(max_load):
    # Une mesure sous charge n'est pas une mesure : échec franc, comme les bancs Rust.
    if load1() >= max_load:
        sys.exit(f"mesure annulée: load1={load1():.2f} >= {max_load}")


def sse_trial(args):
    conn = http.client.HTTPConnection(args.host, args.port, timeout=120)
    body = {"messages": [{"role": "user", "content": args.prompt}], "stream": True,
            "max_tokens": args.max_tokens, "temperature": 0}
    if args.model:
        body["model"] = args.model
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    start = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", json.dumps(body), headers)
    resp = conn.getresponse()
    ttfb = time.perf_counter() - start
    if resp.status != 200:
        sys.exit(f"HTTP {resp.status}: {resp.read()[:200]!r}")
    first, gaps, last = None, [], None
    while True:
        line = resp.fp.readline()
        if not line or line.strip() == b"data: [DONE]":
            break
        if not line.startswith(b"data:"):
            continue
        now = time.perf_counter()
        if first is None:
            first = now - start
        else:
            gaps.append(now - last)
        last = now
    conn.close()
    return ttfb, first, gaps


def recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise EOFError("connexion fermée")
        buf += chunk
    return buf


def ws_send(sock, opcode, payload):
    mask = os.urandom(4)
    n = len(payload)
    head = bytes([0x80 | opcode])
    head += bytes([0x80 | n]) if n < 126 else b"\xfe" + struct.pack(">H", n)
    sock.sendall(head + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload)))


def ws_recv(sock):
    b0, b1 = recv_exact(sock, 2)
    n = b1 & 0x7F
    if n == 126:
        n = struct.unpack(">H", recv_exact(sock, 2))[0]
    elif n == 127:
        n = struct.unpack(">Q", recv_exact(sock, 8))[0]
    return b0 & 0x0F, recv_exact(sock, n)


def ws_trial(args):
    sock = socket.create_connection((args.host, args.port), timeout=30)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    key = base64.b64encode(os.urandom(16)).decode()
    auth = f"Authorization: Bearer {args.api_key}\r\n" if args.api_key else ""
    req = (f"GET /v1/audio/enhance/ws?rate=16000&format=f32 HTTP/1.1\r\nHost: {args.host}:{args.port}\r\n"
           f"Upgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
           f"Sec-WebSocket-Version: 13\r\n{auth}\r\n")
    start = time.perf_counter()
    sock.sendall(req.encode())
    head = b""
    while not head.endswith(b"\r\n\r\n"):
        head += sock.recv(1)
    ttfb = time.perf_counter() - start
    if b" 101 " not in head.split(b"\r\n")[0]:
        sys.exit(f"handshake refusé: {head[:120]!r}")
    frame = b"".join(struct.pack("<f", 0.2 * __import__("math").sin(i * 0.03)) for i in range(HOP))
    first_pcm, hops = None, []
    t0 = time.perf_counter()
    for hop in range(args.frames):
        sent = time.perf_counter()
        ws_send(sock, 2, frame)
        if hop < 3:  # amorçage : le débruiteur rend sa sortie avec quelques trames de retard
            continue
        while ws_recv(sock)[0] != 2:
            pass
        if hop == 3:
            first_pcm = time.perf_counter() - t0
        else:
            hops.append(time.perf_counter() - sent)
    ws_send(sock, 2, b"")
    sock.close()
    return ttfb, first_pcm, hops


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("transport", choices=["sse", "ws"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--api-key")
    ap.add_argument("-n", "--reps", type=int, default=20, help="répétitions (connexions)")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--max-load", type=float, default=2.0)
    ap.add_argument("--frames", type=int, default=64, help="ws : trames par répétition")
    ap.add_argument("--model", help="sse : nom du modèle")
    ap.add_argument("--max-tokens", type=int, default=64, help="sse")
    ap.add_argument("--prompt", default="Compte de un à vingt.", help="sse")
    args = ap.parse_args()
    trial = sse_trial if args.transport == "sse" else ws_trial
    for _ in range(args.warmup):
        trial(args)
    ttfb, first, per_frame = [], [], []
    for _ in range(args.reps):
        gate(args.max_load)
        a, b, c = trial(args)
        gate(args.max_load)
        ttfb.append(a)
        first.append(b)
        per_frame += c
    first_name = "first_event" if args.transport == "sse" else "first_pcm"
    report(args.transport, "ttfb", ttfb)
    report(args.transport, first_name, first)
    report(args.transport, "per_frame", per_frame)


if __name__ == "__main__":
    main()
