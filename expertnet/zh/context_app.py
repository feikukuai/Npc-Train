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
from .persona import BEATS, NAME_GET, extract_name, format_line

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "context_net_dedup09_30k.pt")
ENC_MODEL = "BAAI/bge-small-zh-v1.5"
MAX_TURNS = 4

_LOCK = threading.Lock()
_STATE = {"ready": False, "error": "", "model": None, "enc": None,
          "texts": None, "beats": None, "class_beats": None}


def ensure():
    with _LOCK:
        if _STATE["ready"] or _STATE["error"]:
            return _STATE["ready"]
        try:
            try:
                ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
            except TypeError:
                ckpt = torch.load(CKPT, map_location="cpu")
            cfg = dict(ckpt["config"])
            cfg.pop("arch", None)
            model = ContextGRUNet(**cfg).eval()
            model.load_state_dict(ckpt["state"])
            enc = HFSentenceEncoder(ENC_MODEL, pooling="cls", max_length=160)
            class_beats = torch.tensor([BEATS.index(b) for b in ckpt["beats"]])
            _STATE.update(ready=True, model=model, enc=enc,
                          texts=ckpt["texts"], beats=ckpt["beats"],
                          class_beats=class_beats)
        except Exception as exc:
            _STATE["error"] = f"{type(exc).__name__}: {exc}"[:300]
        return _STATE["ready"]


def warmup():
    threading.Thread(target=ensure, daemon=True).start()


CONFLICTS = [("热", "冷"), ("大", "小"), ("多", "少"), ("快", "慢"),
             ("高", "低"), ("远", "近"), ("开心", "难过"), ("喜欢", "讨厌"),
             ("甜", "苦"), ("干", "湿"), ("亮", "暗")]


def _clash(a: str, b: str) -> bool:
    return any((x in a and y in b) or (y in a and x in b)
               for x, y in CONFLICTS)


def answer(history: list[tuple[str, str]], message: str):
    if not ensure():
        return "模型未就绪：" + _STATE["error"], "", []
    context = format_context(history, message, MAX_TURNS)
    lines = line_list(context)
    vecs = _STATE["enc"].embed(lines).unsqueeze(0)
    mask = torch.ones(1, vecs.size(1), 1)
    with torch.no_grad():
        logits, beat_logits = _STATE["model"](vecs, mask)
    probs = F.softmax(logits[0], dim=-1)
    top = torch.topk(probs, k=min(5, probs.numel()))
    candidates = [int(i) for i in top.indices]
    name = extract_name(message)
    for player, _ in reversed(history):
        if name:
            break
        name = extract_name(player)
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
    last_player = history[-1][0] if history else ""
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
    if info:
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
