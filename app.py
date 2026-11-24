"""阿禾：一个五岁半的中文小 NPC，整句预测演示。

界面只保留这一位角色。回复优先来自训练好的句库模型；模型未就绪时自动退回规则老师，
所以即使创空间第一次下载句向量失败，页面也不会空着。
"""
from __future__ import annotations

import os

import gradio as gr

from expertnet.zh.chat import model_status, new_session, reply, warmup
from expertnet.zh.persona import NAME

CSS = """
.gradio-container { max-width: 900px !important; }
#chat { border-radius: 14px; }
footer { display: none !important; }
"""

INTRO = f"""# {NAME} · 五岁半

村口药铺家的孩子。只认识药铺、屋后药园、老井、橘猫阿团和娘。
句库 **68 句**，记忆只有一个 **64 维向量**；事实和“刚才发生了什么”由规则层负责，
社交措辞由句库模型选择。被骂会记住一会儿，问名字会答。

> 她不会写新句子，只会从自己会的句子里挑一句。规则红线仍然优先。
"""


def chat(message, history, session):
    text = (message or "").strip()
    history = list(history or [])
    if not text:
        return history, session, "说点什么吧。", "", ""

    session, result = reply(session, text)
    answer = result.get("reply") or result["atom"] or result["rule"]
    history.append({"role": "user", "content": text})
    history.append({"role": "assistant", "content": answer})

    status = f"**{result['used']}**"
    if result["confidence"] is not None:
        status += f" · 置信度 {result['confidence']:.2f}"
    state = result["state"]
    status += f" · 信任 {state['trust']} · 回合 {state['turn']}"
    if state["player_name"]:
        status += f" · 叫你 {state['player_name']}"

    probe = ""
    if result["used"] == "句库" and result["atom"] != result["rule"]:
        probe = f"规则老师会说：{result['rule']}"
    elif result["used"] == "规则" and result["atom"]:
        probe = f"句库模型会选：{result['atom']}"
    return history, session, status, probe, ""


def reset():
    return [], new_session(), "68 句 · 64 维记忆", "", ""


with gr.Blocks(title=f"{NAME} · 句库 NPC") as demo:
    gr.Markdown(INTRO)
    chatbot = gr.Chatbot(height=460, elem_id="chat", label="和 阿禾 说话",
                         show_label=False)
    with gr.Row():
        msg = gr.Textbox(placeholder="和阿禾说点什么，例如：你好。 / 我叫小明。 / 傻瓜，我走了。",
                         show_label=False, container=False, scale=7, autofocus=True)
        send = gr.Button("发送", variant="primary", scale=1)
    with gr.Row():
        status = gr.Markdown("68 句 · 64 维记忆", scale=4)
        reset_btn = gr.Button("重开", scale=1)
    with gr.Accordion("规则老师对照", open=False):
        probe = gr.Markdown()
    gr.Examples(["你好。", "我叫小明。", "傻瓜，我走了。", "我问她名字。",
                 "刚才发生了什么？"], inputs=msg, label="试试")

    session = gr.State(new_session())
    outputs = [chatbot, session, status, probe, msg]
    send.click(chat, [msg, chatbot, session], outputs)
    msg.submit(chat, [msg, chatbot, session], outputs)
    reset_btn.click(reset, None, outputs)

warmup()
demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)),
            share=False, css=CSS)
