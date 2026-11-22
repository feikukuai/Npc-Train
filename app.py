"""PyBot 中文版：英文规则引擎 + LFM2.5-350M 中英互译。

流程：
    用户中文 → 350M 译成英文 → PyBot 规则引擎回答 → 350M 译回中文

PyBot 本体不含权重、不训练，全部是 if/else、正则和查表；350M 只做翻译，
不参与“怎么回答”的决策。翻译在 2 vCPU 上较慢，所以每轮可能需要几十秒。
"""
import os
import re
import threading

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
import gradio as gr
from modelscope import snapshot_download

REPO_ID = "LiquidAI/LFM2.5-350M"
MAX_INPUT_TOKENS = 4096

# --- 模型 -------------------------------------------------------------------
_state = {"model": None, "tokenizer": None, "error": None}
_ready = threading.Event()


def _warm_model():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = snapshot_download(REPO_ID)
        tokenizer = AutoTokenizer.from_pretrained(path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
        model.eval()
        _state["model"] = model
        _state["tokenizer"] = tokenizer
    except Exception as exc:  # noqa: BLE001 - 起不来也要能回话
        _state["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        _ready.set()


threading.Thread(target=_warm_model, daemon=True).start()


def _get_model():
    if not _ready.wait(timeout=600):
        raise RuntimeError("模型权重仍在下载，请稍后重试")
    if _state["error"]:
        raise RuntimeError(f"权重加载失败：{_state['error']}")
    return _state["model"], _state["tokenizer"]


# --- PyBot ------------------------------------------------------------------
_pybot = {"runtime": None, "error": None}
_pybot_ready = threading.Event()


def _warm_pybot():
    try:
        from bot import runtime

        _pybot["runtime"] = runtime
    except Exception as exc:  # noqa: BLE001
        _pybot["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        _pybot_ready.set()


threading.Thread(target=_warm_pybot, daemon=True).start()


def _get_pybot():
    if not _pybot_ready.wait(timeout=180):
        raise RuntimeError("PyBot 仍在加载，请稍后重试")
    if _pybot["error"]:
        raise RuntimeError(f"PyBot 加载失败：{_pybot['error']}")
    return _pybot["runtime"]


# --- 生成 -------------------------------------------------------------------
def _generate(messages, max_new_tokens=288):
    model, tokenizer = _get_model()
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
    input_ids = inputs.input_ids if hasattr(inputs, "input_ids") else inputs
    if input_ids.shape[1] > MAX_INPUT_TOKENS:
        raise RuntimeError(f"输入超过 {MAX_INPUT_TOKENS} tokens")
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=int(max_new_tokens),
            temperature=0.2,
            top_p=0.9,
            repetition_penalty=1.05,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)


# --- 翻译 -------------------------------------------------------------------
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
LATIN = re.compile(r"[A-Za-z]")

ZH2EN_SYSTEM = "You are a translator. Translate Chinese into natural English. Output only the translation, no notes."
ZH2EN_FEWSHOT = (
    "中文：你好，我叫小明。\nEnglish: Hello, my name is Xiaoming.\n\n"
    "中文：今天天气怎么样？\nEnglish: What is the weather like today?\n\n"
    "中文：水壶里的水垢怎么除？\nEnglish: How do I remove limescale from my kettle?\n\n"
    "中文：{text}\nEnglish:"
)
EN2ZH_SYSTEM = "You are a translator. Translate English into natural Chinese. Output only the translation, no notes."
EN2ZH_FEWSHOT = (
    "English: Hello, my name is Xiaoming.\n中文：你好，我叫小明。\n\n"
    "English: What is the weather like today?\n中文：今天天气怎么样？\n\n"
    "English: How do I remove limescale from my kettle?\n中文：水壶里的水垢怎么除？\n\n"
    "English: {text}\n中文："
)
SIMPLE = {
    "zh2en": "Translate the following Chinese into English. Output only the English translation.\n\n{text}",
    "en2zh": "Translate the following English into Chinese. Output only the Chinese translation.\n\n{text}",
}

_CACHE = {}
_CACHE_LOCK = threading.Lock()
_CACHE_MAX = 300


def _clean(text):
    text = (text or "").strip()
    text = re.sub(r"^(English|Chinese|Translation|翻译)\s*[:：]\s*", "", text, flags=re.I)
    for label in ("English:", "中文：", "中文:"):
        if label in text:
            text = text.rsplit(label, 1)[-1].strip()
    return text.strip().strip('"“”\'')


def _looks_right(text, direction):
    if not text or len(text) < 2:
        return False
    if direction == "zh2en":
        latin = len(LATIN.findall(text))
        cjk = len(CJK.findall(text))
        return latin >= 2 and cjk <= max(2, latin // 4)
    return len(CJK.findall(text)) >= 2


def _chunks(text, limit=420):
    """按句子切成不超过 limit 字符的块，避免 350M 一次翻太长。"""
    text = (text or "").strip()
    if len(text) <= limit:
        return [text] if text else []
    pieces = [p for p in re.split(r"(?<=[.!?。！？])\s*", text) if p]
    out, cur = [], ""
    for piece in pieces:
        while len(piece) > limit:
            out.append(piece[:limit])
            piece = piece[limit:]
        if cur and len(cur) + len(piece) + 1 > limit:
            out.append(cur.strip())
            cur = piece
        else:
            cur = (cur + " " + piece).strip() if cur else piece
    if cur.strip():
        out.append(cur.strip())
    return out


def _translate_chunk(text, direction):
    system = ZH2EN_SYSTEM if direction == "zh2en" else EN2ZH_SYSTEM
    few = ZH2EN_FEWSHOT if direction == "zh2en" else EN2ZH_FEWSHOT
    prompts = [few.format(text=text), SIMPLE[direction].format(text=text)]
    for prompt in prompts:
        try:
            raw = _generate(
                [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                max_new_tokens=288,
            )
        except Exception:  # noqa: BLE001 - 一次失败就换简单提示重试
            continue
        out = _clean(raw)
        if _looks_right(out, direction):
            return out
    return None


def translate(text, direction):
    """返回译文；失败返回 None。结果带缓存，重复句子不再生成。"""
    text = (text or "").strip()
    if not text:
        return ""
    key = (direction, text)
    with _CACHE_LOCK:
        if key in _CACHE:
            return _CACHE[key]
    parts = []
    for chunk in _chunks(text):
        out = _translate_chunk(chunk, direction)
        if out is None:
            # 失败不缓存：模型可能只是还没加载好，下次重试应当有机会成功
            return None
        parts.append(out)
    result = (" ".join(parts) if direction == "zh2en" else "".join(parts)).strip()
    with _CACHE_LOCK:
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = result
    return result


# --- 对话 -------------------------------------------------------------------
def chat(message, history, memory, show_en):
    history = list(history or [])
    memory = memory if isinstance(memory, dict) else {}
    text = (message or "").strip()
    if not text:
        return "", history, memory, _status("等待输入")

    user_is_zh = bool(CJK.search(text))
    note = ""
    try:
        if user_is_zh:
            zh_user = text
            en_user = translate(text, "zh2en")
            if not en_user:
                reply = "这句中文没能翻译成英文，PyBot 读不到。请换一种说法，或把句子缩短一点。"
                history += [
                    {"role": "user", "content": zh_user},
                    {"role": "assistant", "content": reply},
                ]
                return "", history, memory, _status("中→英失败")
        else:
            en_user = text
            zh_user = translate(text, "en2zh")
            if not zh_user:
                zh_user = text
                note = "（原句中文翻译失败，显示英文）"

        result = _get_pybot().turn(en_user, memory)
        en_reply = str(result.get("reply") or "")
        zh_reply = translate(en_reply, "en2zh")
        if not zh_reply:
            zh_reply = "（翻译失败，以下为 PyBot 英文原文）\n\n" + en_reply
            note = (note + " 回复翻译失败。").strip()

        shown = zh_reply
        if show_en and en_reply:
            shown += "\n\n> **PyBot:** " + en_reply
        if note:
            shown += "\n\n*" + note + "*"
        history += [
            {"role": "user", "content": zh_user},
            {"role": "assistant", "content": shown},
        ]
        step = "中→英" if user_is_zh else "英→中"
        status = (
            f"**模块**：`{result.get('module')}`　**流程**：{step} → PyBot → 英→中"
            f"　**PyBot 英文输入**：{en_user[:120]}"
        )
        return "", history, memory, status
    except Exception as exc:  # noqa: BLE001 - 一轮失败不能让页面 500
        history += [
            {"role": "user", "content": text},
            {"role": "assistant", "content": f"出错了：{type(exc).__name__}: {exc}"},
        ]
        return "", history, memory, _status("错误")


def _status(note="就绪"):
    model = "加载失败" if _state["error"] else ("就绪" if _ready.is_set() else "下载/加载中")
    bot = "加载失败" if _pybot["error"] else ("就绪" if _pybot_ready.is_set() else "加载中")
    return f"**{note}**　模型：{model}　PyBot：{bot}"


def reset():
    return [], {}, _status("新对话")


with gr.Blocks(title="PyBot 中文版") as demo:
    gr.Markdown(
        "# PyBot 中文版\n"
        "PyBot 是一个**没有权重、没有训练**的英文规则聊天机器人：全部由规则、正则和查表组成。"
        "这里用 `LiquidAI/LFM2.5-350M` 做中英互译，所以你说中文、它答中文，"
        "但真正决定“怎么回答”的仍然是 PyBot 的规则。\n\n"
        "**注意**：翻译在 2 vCPU 上较慢，每轮可能需要几十秒；首轮还要等模型下载。"
    )
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="对话（中文）", height=460)
            msg = gr.Textbox(
                label="你说（中文或英文）",
                placeholder="例如：水壶里的水垢怎么除？ / How long should I boil an egg?",
                lines=2,
            )
            with gr.Row():
                send = gr.Button("发送", variant="primary")
                clear = gr.Button("清空")
            show_en = gr.Checkbox(label="在回复下方显示 PyBot 英文原文", value=False)
        with gr.Column(scale=2):
            status = gr.Markdown(_status())
            refresh = gr.Button("刷新状态")
            gr.Markdown(
                "### 这一轮发生了什么\n"
                "1. 你的中文 → 350M 译成英文\n"
                "2. PyBot 规则引擎用英文回答\n"
                "3. 英文回答 → 350M 译回中文\n\n"
                "翻译失败时会自动回退：中文没译出来就不送 PyBot；"
                "回答没译出来就显示英文原文。"
            )
            gr.Examples(
                examples=[
                    "水壶里的水垢怎么除？",
                    "煮鸡蛋要几分钟？",
                    "为什么洋葱会让人流泪？",
                    "我叫小明，你记住了吗？",
                    "How long should I boil an egg?",
                ],
                inputs=msg,
            )

    state = gr.State({})
    inputs = [msg, chatbot, state, show_en]
    outputs = [msg, chatbot, state, status]
    msg.submit(chat, inputs, outputs, api_visibility="private")
    send.click(chat, inputs, outputs, api_name="ask")
    clear.click(reset, None, [chatbot, state, status], api_visibility="private")
    refresh.click(lambda: _status(), None, status, api_visibility="private")


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), share=False)
