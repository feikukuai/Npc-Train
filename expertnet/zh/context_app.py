"""测试页：纯上下文 → 426 个语义簇 → 一句台词。

模型是 context_train.py 用 `--dedup-threshold 0.8` 训出来的
`context_net_dedup08.pt`。输入只有对话历史，没有信任/情绪/名字参数；
名字和上一句话从历史里解析，用于渲染表里的占位符。
"""
from __future__ import annotations

import os
import threading

import gradio as gr
import torch
import torch.nn.functional as F

from ..sentence import HFSentenceEncoder
from .context_train import ContextGRUNet, format_context, line_list
from .exhaustive import PLACE, SUBJ
from .factorized_train import (FIELDS, MAX_LINES, PATTERN_NAMES,
                               PATTERN_TEMPLATE, TEMPLATE_KEY, VOCAB,
                               FactorizedNet)
from .persona import NAME_GET, extract_name, format_line

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LOCAL_ENC = os.path.join(ROOT, "models", "bge-small-zh-v1.5")
ENC_MODEL = LOCAL_ENC if os.path.isdir(LOCAL_ENC) else "BAAI/bge-small-zh-v1.5"
CKPT = os.environ.get("CONTEXT_CKPT",
                      os.path.join(HERE, "context_net_dedup09_30k.pt"))
FACTORIZED_CKPT = os.environ.get("FACTORIZED_CKPT",
                                 os.path.join(HERE, "factorized_deploy.pt"))
BACKEND = os.environ.get("CHAT_BACKEND", "factorized").lower()
MAX_TURNS = 4

_LOCK = threading.Lock()
_STATE = {"ready": False, "error": "", "backend": None, "model": None,
          "enc": None, "texts": None, "beats": None, "fixed_texts": None,
          "fixed_beats": None}


def ensure():
    with _LOCK:
        if _STATE["ready"] or _STATE["error"]:
            return _STATE["ready"]
        try:
            enc = HFSentenceEncoder(ENC_MODEL, pooling="cls", max_length=160)
            if BACKEND == "factorized" and os.path.exists(FACTORIZED_CKPT):
                ckpt = torch.load(FACTORIZED_CKPT, map_location="cpu",
                                  weights_only=False)
                model = FactorizedNet(**ckpt["config"]).eval()
                model.load_state_dict(ckpt["state"])
                _STATE.update(ready=True, backend="factorized", model=model,
                              enc=enc, fixed_texts=ckpt["fixed_texts"],
                              fixed_beats=ckpt.get("fixed_beats"))
                return True
            ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
            cfg = dict(ckpt["config"])
            cfg.pop("arch", None)
            model = ContextGRUNet(**cfg).eval()
            model.load_state_dict(ckpt["state"])
            _STATE.update(ready=True, backend="cluster", model=model, enc=enc,
                          texts=ckpt["texts"], beats=ckpt["beats"])
        except Exception as exc:
            _STATE["error"] = f"{type(exc).__name__}: {exc}"[:300]
        return _STATE["ready"]


def warmup():
    def run():
        if not ensure():
            return
        # 后台把 7000 条固定句编码好，第一句话就不必等 2–3 秒。
        texts = _STATE.get("fixed_texts") or []
        for i in range(0, len(texts), 256):
            _STATE["enc"].embed(texts[i:i + 256], batch=128)
    threading.Thread(target=run, daemon=True).start()


CONFLICTS = [("热", "冷"), ("大", "小"), ("多", "少"), ("快", "慢"),
             ("高", "低"), ("远", "近"), ("开心", "难过"), ("喜欢", "讨厌"),
             ("甜", "苦"), ("干", "湿"), ("亮", "暗")]


def _clash(a: str, b: str) -> bool:
    return any((x in a and y in b) or (y in a and x in b)
               for x, y in CONFLICTS)


def _answer_cluster(history: list[tuple[str, str]], message: str):
    if not ensure():
        return "模型未就绪：" + _STATE["error"], "", []
    context = format_context(history, message, MAX_TURNS)
    lines = line_list(context)
    vecs = _STATE["enc"].embed(lines).unsqueeze(0)
    mask = torch.ones(1, vecs.size(1), 1)
    with torch.no_grad():
        logits, _ = _STATE["model"](vecs, mask)
    probs = F.softmax(logits[0], dim=-1)
    top = torch.topk(probs, k=min(5, probs.numel()))
    candidates = [int(i) for i in top.indices]
    name = extract_name(message)
    for player, _ in reversed(history):
        if name:
            break
        name = extract_name(player)
    last_player = history[-1][0] if history else ""

    # 最近说过的不重复；没有名字/上一句时不选对应模板。
    avoid = {reply for _, reply in history[-8:]}
    valid = []
    for i in candidates:
        text = _STATE["texts"][i]
        if text in avoid:
            continue
        if "{player_name}" in text and not name:
            continue
        if "{last_player}" in text and not last_player:
            continue
        valid.append(i)
    if valid:
        candidates = valid

    # 问句里明确提到的主体/处所，优先选包含它的候选。
    for subject in SUBJ[2:]:
        if subject in message:
            hit = [i for i in candidates if subject in _STATE["texts"][i]]
            if hit:
                candidates = hit
            break
    if "你" in message:
        hit = [i for i in candidates if "我" in _STATE["texts"][i]]
        if hit:
            candidates = hit
    for place in PLACE:
        if place in message:
            hit = [i for i in candidates if place in _STATE["texts"][i]]
            if hit:
                candidates = hit
            break

    idx = candidates[0]
    # 问名字时，优先真的带名字的候选。
    if name and any(key in message for key in NAME_GET):
        named = [i for i in candidates
                 if "{player_name}" in _STATE["texts"][i]
                 or name in _STATE["texts"][i]]
        if named:
            idx = named[0]
    # 粗粒度合并可能把“热/冷”并进同一簇；只在同 beat 内回退。
    if _clash(_STATE["texts"][idx], message):
        beat = _STATE["beats"][idx]
        for i in candidates:
            if _STATE["beats"][i] == beat \
                    and not _clash(_STATE["texts"][i], message):
                idx = i
                break
    reply = format_line(_STATE["texts"][idx],
                        {"player_name": name, "last_player": last_player})
    beat = _STATE["beats"][idx]
    score = float(probs[idx])
    others = "；".join(
        f"{_STATE['texts'][int(i)]}({float(p):.2f})"
        for p, i in zip(top.values[1:], top.indices[1:]))
    status = (f"语义簇 {idx + 1}/{len(_STATE['texts'])} · beat {beat} · "
              f"置信 {score:.2f}")
    return reply, status, [beat, others]


def _name_and_last(history, message):
    name = extract_name(message)
    for player, _ in reversed(history):
        if name:
            break
        name = extract_name(player)
    last = history[-1][0] if history else ""
    return name, last


def _fixed_pool(beat: str) -> list[int]:
    pools = _STATE.get("fixed_pool")
    if pools is None:
        pools = {}
        for i, item in enumerate(_STATE["fixed_beats"] or []):
            pools.setdefault(item, []).append(i)
        _STATE["fixed_pool"] = pools
    return pools.get(beat, [])


def _pick_fixed(beat: str, message: str, name: str, last: str,
                recent: set[str], fallback: int) -> int:
    """在同一 beat 的固定句里挑一句：语义最贴近玩家这句话，且避开最近说过。"""
    ids = _fixed_pool(beat)
    valid = []
    for i in ids:
        text = _STATE["fixed_texts"][i]
        if "{player_name}" in text and not name:
            continue
        if "{last_player}" in text and not last:
            continue
        valid.append(i)
    if not valid:
        return fallback
    vec = _STATE["enc"].embed([message])[0]
    table = _STATE["enc"].embed([_STATE["fixed_texts"][i] for i in valid])
    sims = (table @ vec).tolist()
    order = sorted(range(len(valid)), key=lambda j: (-sims[j], valid[j]))
    best = valid[order[0]]
    if _STATE["fixed_texts"][best] in recent:
        for j in order[1:]:
            if _STATE["fixed_texts"][valid[j]] not in recent:
                # 只在这个候选几乎同样贴合时才换，避免把“我娘叫阿棠”换掉。
                if sims[order[0]] - sims[j] < 0.05:
                    return valid[j]
                break
    return best


def _answer_factorized(history: list[tuple[str, str]], message: str):
    # 逐行编码：最后一句玩家话不会被长历史淹没。
    lines = []
    for prev_player, prev_reply in history[-MAX_TURNS:]:
        lines.append(f"玩家：{prev_player}")
        lines.append(f"阿禾：{prev_reply}")
    lines.append(f"玩家：{message}")
    lines = lines[-MAX_LINES:]
    vec = _STATE["enc"].embed(lines).unsqueeze(0)
    lengths = torch.tensor([len(lines)])
    with torch.no_grad():
        raw = _STATE["model"](vec, lengths)
    out = {key: raw[key] + raw[key + "_local"]
           for key in ["mode", "fixed", "pattern", *VOCAB]}
    mode = int(out["mode"][0].argmax())
    confidence = float(F.softmax(out["mode"], dim=-1)[0, mode])
    name, last = _name_and_last(history, message)
    if mode == 1:
        best = int(out["fixed"][0].argmax())
        beat = (_STATE["fixed_beats"][best]
                if _STATE.get("fixed_beats") else "固定句")
        recent = {reply for _, reply in history[-8:]}
        idx = _pick_fixed(beat, message, name, last, recent, best)
        text = _STATE["fixed_texts"][idx]
        if ("{player_name}" in text and not name) \
                or ("{last_player}" in text and not last):
            text = next((item for item in _STATE["fixed_texts"]
                         if "{" not in item), "我没听明白。")
        reply = format_line(text, {"player_name": name, "last_player": last})
        return reply, f"因子化 · {beat} · 置信 {confidence:.2f}", [beat, ""]

    pattern = PATTERN_NAMES[int(out["pattern"][0].argmax())]
    values = {TEMPLATE_KEY[field]: VOCAB[field][int(out[field][0].argmax())]
              for field in FIELDS[pattern]}
    reply = PATTERN_TEMPLATE[pattern].format(**values)
    return reply, f"因子化 · {pattern} · 置信 {confidence:.2f}", [pattern, ""]


def answer(history: list[tuple[str, str]], message: str):
    if not ensure():
        return "模型未就绪：" + _STATE["error"], "", []
    if _STATE["backend"] == "factorized":
        return _answer_factorized(history, message)
    return _answer_cluster(history, message)


def chat(message, history, session):
    text = (message or "").strip()
    history = list(history or [])
    session = list(session or [])
    if not text:
        return history, session, "说点什么吧。", ""
    reply, status, info = answer(session, text)
    history.append({"role": "user", "content": text})
    history.append({"role": "assistant", "content": reply})
    session.append((text, reply))
    del session[:-8]
    if info and len(info) > 1 and info[1]:
        status += f" · 次选：{info[1]}"
    return history, session, status, ""


def reset():
    return [], [], "等待输入", ""


INTRO = """# 阿禾 · 五岁半

村口药铺家的孩子。只认识药铺、屋后药园、老井、橘猫阿团和娘。
和她说说话吧。
"""


def build():
    with gr.Blocks(title="阿禾 · 纯上下文测试") as demo:
        gr.Markdown(INTRO)
        chatbot = gr.Chatbot(height=460, show_label=False)
        with gr.Row():
            msg = gr.Textbox(placeholder="例如：你好。 / 我叫小明。 / 傻瓜，我走了。",
                             show_label=False, container=False, scale=7,
                             autofocus=True)
            send = gr.Button("发送", variant="primary", scale=1)
        status = gr.Markdown("模型加载中…")
        with gr.Row():
            reset_btn = gr.Button("重开")
        gr.Examples(["你好。", "我叫小明。", "你还记得我叫什么吗？",
                     "刚才发生了什么？", "你真笨。", "对不起。", "傻瓜，我走了。"],
                    inputs=msg)
        session = gr.State([])
        outputs = [chatbot, session, status, msg]
        send.click(chat, [msg, chatbot, session], outputs)
        msg.submit(chat, [msg, chatbot, session], outputs)
        reset_btn.click(reset, None, outputs)
    return demo


def main():
    warmup()
    build().launch(server_name="0.0.0.0",
                   server_port=int(os.environ.get("PORT", 7867)), share=False)


if __name__ == "__main__":
    main()
