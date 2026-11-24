"""Minimal local web chat: rule NPC on the left, sentence-atom model on the right.

    python tools/chat_web.py
    open http://127.0.0.1:7860

No web framework and no build step: one stdlib server, one HTML file, a JSON
endpoint. The rule engine owns state and permissions; the atom model only picks a
sentence from the vocabulary it learned.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bot.npc import engine
from expertnet.atoms import AtomSession, load_atoms
from expertnet.sentence import HFSentenceEncoder

PAGE = os.path.join(ROOT, "web", "chat.html")
PROFILE = "xiaohong"
DEFAULT_TRUST = 35


class ChatSession:
    def __init__(self, atom: AtomSession | None, atoms_available: bool):
        self.memory: dict = {}
        engine.activate(self.memory, PROFILE, trust=DEFAULT_TRUST)
        self.atom = atom
        self.atoms_available = atoms_available

    def say(self, text: str, mode: str) -> dict:
        result = engine.turn(text, self.memory)
        state = self.memory["npc"]
        out = {"rule": result["reply"], "atom": None,
               "state": {"trust": state["trust"], "turn": state["turn"],
                         "profile": state["profile"]},
               "atoms_available": self.atoms_available}
        if self.atom is not None and mode != "rule":
            line, conf = self.atom.reply(text, state)
            out["atom"] = {"text": line, "confidence": round(conf, 3)}
        return out

    def reset(self) -> None:
        engine.activate(self.memory, PROFILE, trust=DEFAULT_TRUST)
        if self.atom is not None:
            self.atom.reset()


class Server:
    def __init__(self, checkpoint: str):
        self.lock = threading.Lock()
        self.sessions: dict[str, ChatSession] = {}
        self.atoms_available = os.path.exists(checkpoint)
        self.hf = None
        self.template = None
        if self.atoms_available:
            try:
                print("loading sentence encoder...")
                self.hf = HFSentenceEncoder()
                self.template = load_atoms(checkpoint, self.hf)
                print("sentence-atom model ready")
            except Exception:
                traceback.print_exc()
                self.atoms_available = False
                self.template = None

    def page(self) -> bytes:
        with open(PAGE, encoding="utf-8") as fh:
            return fh.read().encode("utf-8")

    def session(self, sid: str) -> ChatSession:
        with self.lock:
            entry = self.sessions.get(sid)
            if entry is None:
                atom = None
                if self.atoms_available:
                    atom = AtomSession(self.template.model, self.hf, self.template.atoms)
                entry = ChatSession(atom, self.atoms_available)
                self.sessions[sid] = entry
            return entry


SERVER: Server | None = None


class Handler(BaseHTTPRequestHandler):
    server_version = "PyBotChat/0.1"

    def log_message(self, fmt, *args):
        pass

    def _write(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict) -> None:
        self._write(code, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    "application/json; charset=utf-8")

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._write(200, SERVER.page(), "text/html; charset=utf-8")
        elif path == "/api/health":
            self._json(200, {"ok": True, "atoms": SERVER.atoms_available,
                             "profile": PROFILE})
        elif path == "/favicon.ico":
            self._write(204, b"", "image/x-icon")
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._json(400, {"error": "bad json"})
            return
        sid = str(payload.get("sid") or uuid.uuid4())
        chat = SERVER.session(sid)

        if path == "/api/chat":
            text = (payload.get("text") or "").strip()
            if not text:
                self._json(400, {"error": "empty text"})
                return
            result = chat.say(text, str(payload.get("mode") or "both"))
            result["sid"] = sid
            self._json(200, result)
        elif path == "/api/reset":
            chat.reset()
            self._json(200, {"ok": True, "sid": sid})
        else:
            self._json(404, {"error": "not found"})


def main() -> int:
    global SERVER
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--checkpoint", default=os.path.join(ROOT, "expertnet", "atoms.pt"))
    args = parser.parse_args()

    SERVER = Server(args.checkpoint)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"serving on http://{args.host}:{args.port}")
    print("press Ctrl+C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
