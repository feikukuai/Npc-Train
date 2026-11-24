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
ZH_MODEL = ("BAAI/bge-small-zh-v1.5", "cls")


class ChatSession:
    def __init__(self, atom: AtomSession | None, atoms_available: bool, lang: str):
        self.lang = lang
        self.atom = atom
        self.atoms_available = atoms_available
        self.memory: dict = {}
        self.zh_state = None
        if lang == "zh":
            from expertnet.zh.persona import new_state
            self.zh_state = new_state()
        else:
            engine.activate(self.memory, PROFILE, trust=DEFAULT_TRUST)

    def say(self, text: str, mode: str) -> dict:
        if self.lang == "zh":
            from expertnet.zh.persona import teacher_turn
            rule, _ = teacher_turn(text, self.zh_state)
            state = self.zh_state
            profile = "ahe"
        else:
            result = engine.turn(text, self.memory)
            rule = result["reply"]
            state = self.memory["npc"]
            profile = state["profile"]
        out = {"rule": rule, "atom": None,
               "state": {"trust": state["trust"], "turn": state["turn"],
                         "profile": profile},
               "atoms_available": self.atoms_available}
        if self.atom is not None and mode != "rule":
            line, conf = self.atom.reply(text, state)
            out["atom"] = {"text": line, "confidence": round(conf, 3)}
        return out

    def reset(self) -> None:
        if self.lang == "zh":
            from expertnet.zh.persona import new_state
            self.zh_state = new_state()
        else:
            engine.activate(self.memory, PROFILE, trust=DEFAULT_TRUST)
        if self.atom is not None:
            self.atom.reset()


class Server:
    def __init__(self, checkpoint: str, lang: str = "en"):
        self.lock = threading.Lock()
        self.sessions: dict[str, ChatSession] = {}
        self.lang = lang
        self.name = "阿禾" if lang == "zh" else "小红"
        self.atoms_available = os.path.exists(checkpoint)
        self.hf = None
        self.template = None
        if self.atoms_available:
            try:
                print("loading sentence encoder...")
                if lang == "zh":
                    self.hf = HFSentenceEncoder(ZH_MODEL[0], pooling=ZH_MODEL[1])
                else:
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
                entry = ChatSession(atom, self.atoms_available, self.lang)
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
                             "profile": PROFILE, "lang": SERVER.lang,
                             "name": SERVER.name})
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
    parser.add_argument("--lang", choices=["en", "zh"], default="en")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = os.path.join(ROOT, "expertnet", "zh" if args.lang == "zh" else "",
                                  "atoms_zh.pt" if args.lang == "zh" else "atoms.pt")
    SERVER = Server(checkpoint, args.lang)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"serving {SERVER.name} on http://{args.host}:{args.port}")
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
