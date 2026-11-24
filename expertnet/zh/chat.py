"""Deployment core for 阿禾, independent of Gradio.

A session is plain Python data (``{"state": ..., "h": ...}``), so it can live in a
Gradio ``State`` or a web server. The sentence-atom model is optional: if its
checkpoint or encoder cannot load, the rule teacher still answers.
"""
from __future__ import annotations

import json
import os
import threading

import torch
import torch.nn.functional as F

from ..atoms import make_model
from ..sentence import HFSentenceEncoder
from .bank import get_bank
from .grammar import SPEC as GRAMMAR_SPEC
from .persona import FACTS, format_line, new_state, teacher_turn, zh_state_vector
from .thought import THOUGHTS

HERE = os.path.dirname(os.path.abspath(__file__))
SAFE = os.path.join(HERE, "atoms_zh.safetensors")
META = os.path.join(HERE, "atoms_zh.json")
PT = os.path.join(HERE, "atoms_zh.pt")
ENC_MODEL = "BAAI/bge-small-zh-v1.5"
SECRET_ATOMS = {fact["text"] for fact in FACTS.values() if fact.get("tier", 0) >= 1}
# Facts, permissions and names stay on the rule teacher; the learned selector is
# kept for social wording, taunt/apology_meta and episodic memory.
RULE_ONLY = {"boundary", "threat", "meta", "trade", "gift", "promise", "personal",
             "lore", "secret", "secret_denied", "name_saved", "name_recall",
             "name_unknown", "out_of_world", "fallback"}
# Every beat the grammar bank can render. Facts, secrets and safety lines are not
# in it: those keep the teacher's exact wording.
BANK_BEATS = frozenset(GRAMMAR_SPEC)

_LOCK = threading.Lock()
_MODEL = {"tried": False, "ready": False, "error": "",
          "model": None, "hf": None, "atoms": None, "beats": None}


def _load_checkpoint():
    if os.path.exists(SAFE) and os.path.exists(META):
        try:
            with open(META, encoding="utf-8") as fh:
                meta = json.load(fh)
            from safetensors.torch import load_file
            model = make_model(meta["config"])
            model.load_state_dict(load_file(SAFE))
            return model.eval(), meta["atoms"], meta["beats"]
        except Exception:
            pass
    ckpt = torch.load(PT, map_location="cpu")
    model = make_model(ckpt["config"])
    model.load_state_dict(ckpt["state"])
    return model.eval(), ckpt["atoms"], ckpt["beats"]


def _load_encoder():
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    try:
        return HFSentenceEncoder(ENC_MODEL, pooling="cls")
    except Exception as hf_error:
        try:
            from modelscope import snapshot_download
        except Exception:
            raise hf_error
        last = hf_error
        for model_id in ("AI-ModelScope/bge-small-zh-v1.5", ENC_MODEL):
            try:
                return HFSentenceEncoder(snapshot_download(model_id), pooling="cls")
            except Exception as exc:
                last = exc
        raise last


def ensure_model() -> bool:
    with _LOCK:
        if _MODEL["tried"]:
            return _MODEL["ready"]
        _MODEL["tried"] = True
        try:
            model, atoms, beats = _load_checkpoint()
            hf = _load_encoder()
            _MODEL.update(model=model, hf=hf, atoms=atoms, beats=beats, ready=True)
        except Exception as exc:
            _MODEL["error"] = f"{type(exc).__name__}: {exc}"[:300]
        return _MODEL["ready"]


def components():
    """(model, encoder, atoms, beats) or None; for training and evaluation."""
    if not ensure_model():
        return None
    return _MODEL["model"], _MODEL["hf"], _MODEL["atoms"], _MODEL["beats"]


def model_status() -> str:
    if _MODEL["ready"]:
        return "句库模型就绪"
    if not _MODEL["tried"]:
        return "句库模型加载中（首次会下载约 95MB 句向量模型）"
    return "规则模式（句库模型未就绪）"


def model_error() -> str:
    return _MODEL["error"]


def warmup() -> None:
    threading.Thread(target=ensure_model, daemon=True).start()
    threading.Thread(target=get_bank, daemon=True).start()


def new_session() -> dict:
    return {"state": new_state(), "h": None, "said": []}


def reply(session: dict | None, text: str) -> tuple[dict, dict]:
    session = session or new_session()
    state = dict(session.get("state") or new_state())
    state["recent"] = [dict(row) for row in state.get("recent") or []]
    said = list(session.get("said") or [])
    h = session.get("h")
    previous = state.get("last_beat")

    rule_template, beat = teacher_turn(text, state)
    rule = format_line(rule_template, state)
    result = {"rule": rule, "atom": None, "reply": rule, "used": "规则",
              "beat": beat, "confidence": None,
              "state": {"trust": state["trust"], "turn": state["turn"],
                        "player_name": state.get("player_name", "")}}

    # 句库是主要出口：同一个 beat 里有几千种说法，按情绪、玩家用词和最近说过的话挑。
    bank_line = None
    if beat in BANK_BEATS:
        try:
            bank = get_bank()
            bank_line = bank.pick(beat, state, text, said)
            result["bank_size"] = bank.size
            if bank_line:
                result.update(reply=bank_line, used="句库")
        except Exception as exc:
            result["bank_error"] = f"{type(exc).__name__}: {exc}"[:200]

    if ensure_model():
        try:
            atom_state = {**state, "last_beat": previous}
            vec = _MODEL["hf"].embed([text]).unsqueeze(0)
            st = zh_state_vector(atom_state).reshape(1, 1, -1)
            hidden = (torch.tensor(h, dtype=torch.float32).reshape(1, -1)
                      if h else torch.zeros(1, _MODEL["model"].hidden))
            with torch.no_grad():
                logits, _, new_h = _MODEL["model"](vec, st, hidden)
            logits = logits[0, 0]
            secret_locked = state["trust"] < 40 and "secret" not in state.get("revealed", [])
            no_previous = not state.get("last_player")
            blocked = torch.tensor([
                ("{player_name}" in atom and not state.get("player_name"))
                or ("{last_player}" in atom and no_previous)
                or (secret_locked and atom in SECRET_ATOMS)
                for atom in _MODEL["atoms"]
            ])
            if bool(blocked.any()):
                logits = logits.masked_fill(blocked, -1e9)
            probs = F.softmax(logits, dim=-1)
            line = _MODEL["atoms"][int(probs.argmax())]
            if "{player_name}" in line:
                line = line.replace("{player_name}", state.get("player_name") or "你")
            if "{last_player}" in line:
                line = line.replace("{last_player}",
                                    state.get("last_player") or "你刚才的话")
            thought = "-"
            if getattr(_MODEL["model"], "last_thoughts", None) is not None:
                thought = THOUGHTS[int(_MODEL["model"].last_thoughts[0, 0].argmax())]
            result.update(atom=line, confidence=round(float(probs.max()), 3),
                          thought=thought)
            if not bank_line:
                if beat in RULE_ONLY:
                    result.update(reply=rule, used="规则")
                else:
                    result.update(reply=line, used="句库")
            h = new_h[0].tolist()
        except Exception as exc:
            result["error"] = str(exc)[:200]

    if state.get("recent"):
        state["recent"][-1]["reply"] = result["reply"]
    said.append(result["reply"])
    del said[:-8]
    return {"state": state, "h": h, "said": said}, result
