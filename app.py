"""NPC 模拟：Grom Ironfoot / Lina Malt / 小红，规则驱动的角色对话。

流程（中文模式）：
    用户中文 → 译成英文 → NPC 规则引擎（权限/秘密/信任/任务）→ 英文回答
    → 译回中文 → LFM2.5 结合上下文润色

流程（纯英文快速模式，默认开启）：
    用户英文 → NPC 规则引擎 → 英文回答（跳过翻译与润色，几乎没有额外延迟）

NPC 的“说什么”由 bot/npc/ 的确定性规则决定；翻译和润色只改措辞，不参与决策。
翻译引擎：
- Opus-MT（默认）：Helsinki-NLP 的专用翻译模型，长句稳定。
- LFM2.5-350M（实验）：短句可用，也用于中文润色。
"""
import os
import re
import threading

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
import gradio as gr
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from bot.npc import activate as npc_activate
from bot.npc.profiles import PROFILES

LFM_ID = "LiquidAI/LFM2.5-350M"
OPUS_ID = {
    "en2zh": "Helsinki-NLP/opus-mt-en-zh",
    "zh2en": "Helsinki-NLP/opus-mt-zh-en",
}
MAX_INPUT_TOKENS = 4096

# --- Opus-MT ----------------------------------------------------------------
_opus = {"en2zh": None, "zh2en": None, "tokenizer": {}, "error": None}
_opus_ready = threading.Event()


def _warm_opus():
    try:
        for direction, repo in OPUS_ID.items():
            path = snapshot_download(repo)
            _opus[direction] = AutoModelForSeq2SeqLM.from_pretrained(path)
            _opus[direction].eval()
            _opus["tokenizer"][direction] = AutoTokenizer.from_pretrained(path)
    except Exception as exc:  # noqa: BLE001 - 起不来也要能回话
        _opus["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        _opus_ready.set()


threading.Thread(target=_warm_opus, daemon=True).start()


def _get_opus(direction):
    if not _opus_ready.wait(timeout=600):
        raise RuntimeError("翻译模型仍在下载，请稍后重试")
    if _opus["error"]:
        raise RuntimeError(f"Opus-MT 加载失败：{_opus['error']}")
    return _opus[direction], _opus["tokenizer"][direction]


# --- LFM2.5（实验） ----------------------------------------------------------
_lfm = {"model": None, "tokenizer": None, "error": None}
_lfm_ready = threading.Event()


def _warm_lfm():
    try:
        path = snapshot_download(LFM_ID)
        tokenizer = AutoTokenizer.from_pretrained(path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
        model.eval()
        _lfm["model"] = model
        _lfm["tokenizer"] = tokenizer
    except Exception as exc:  # noqa: BLE001
        _lfm["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        _lfm_ready.set()


threading.Thread(target=_warm_lfm, daemon=True).start()


def _get_lfm():
    if not _lfm_ready.wait(timeout=600):
        raise RuntimeError("LFM2.5 仍在下载，请稍后重试")
    if _lfm["error"]:
        raise RuntimeError(f"LFM2.5 加载失败：{_lfm['error']}")
    return _lfm["model"], _lfm["tokenizer"]


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
    text = re.sub(r">>[A-Za-z_]+<<", "", text)
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


def _chunks(text, limit=400):
    """按句号/逗号切成不超过 limit 字符的块，专用翻译模型对长句更稳。"""
    text = (text or "").strip()
    if len(text) <= limit:
        return [text] if text else []
    pieces = [p for p in re.split(r"(?<=[.!?。！？;；,])\s*", text) if p]
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


def _opus_chunk(text, direction):
    model, tokenizer = _get_opus(direction)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            num_beams=4,
            early_stopping=True,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def _lfm_generate(messages, max_new_tokens=288):
    model, tokenizer = _get_lfm()
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


def _lfm_chunk(text, direction):
    system = ZH2EN_SYSTEM if direction == "zh2en" else EN2ZH_SYSTEM
    few = ZH2EN_FEWSHOT if direction == "zh2en" else EN2ZH_FEWSHOT
    prompts = [few.format(text=text), SIMPLE[direction].format(text=text)]
    for prompt in prompts:
        try:
            raw = _lfm_generate(
                [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                max_new_tokens=288,
            )
        except Exception:  # noqa: BLE001 - 一次失败就换简单提示重试
            continue
        out = _clean(raw)
        if _looks_right(out, direction):
            return out
    return None


def _engine_order(direction, engine):
    if engine == "lfm":
        return ["lfm", "opus"]
    if engine == "opus":
        return ["opus", "lfm"]
    # 自动：中文短问句用 LFM（对短句更准），英文长回答用 Opus（长句更稳）
    return ["lfm", "opus"] if direction == "zh2en" else ["opus", "lfm"]


def _translate_chunk(text, direction, engine):
    for name in _engine_order(direction, engine):
        try:
            out = _opus_chunk(text, direction) if name == "opus" else _lfm_chunk(text, direction)
        except Exception:  # noqa: BLE001 - 引擎不可用就换另一个
            continue
        out = _clean(out)
        if _looks_right(out, direction):
            return out
    return None


def translate(text, direction, engine="opus"):
    """返回译文；失败返回 None。结果带缓存，重复句子不再生成。"""
    text = (text or "").strip()
    if not text:
        return ""
    key = (engine, direction, text)
    with _CACHE_LOCK:
        if key in _CACHE:
            return _CACHE[key]
    parts = []
    for chunk in _chunks(text):
        out = _translate_chunk(chunk, direction, engine)
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


# --- LFM2.5 润色 -------------------------------------------------------------
POLISH_SYSTEM = (
    "你是中文编辑。根据对话上下文，把待润色的中文回答改得自然、通顺、贴题。"
    "只输出改写后的中文回答，不要解释。"
)
POLISH_TEMPLATE = (
    "根据下面的对话，只改写“待改写句子”，让它更自然、通顺、贴题。\n"
    "要求：只输出改写后的这一句中文；保留原意，不新增事实、数字、人名；不要解释。\n\n"
    "[最近对话]\n{context}\n\n"
    "[本轮用户]\n{user}\n\n"
    "[PyBot 英文原答]\n{en_reply}\n\n"
    "[完整待润色回答]\n{draft}\n\n"
    "[待改写句子]\n{target}\n\n"
    "改写："
)
_POLISH_CACHE = {}
_POLISH_LOCK = threading.Lock()
_POLISH_MAX = 300


def _text_of(content):
    if isinstance(content, list):
        return "".join(x.get("text", "") for x in content if isinstance(x, dict))
    return str(content or "")


def _context_text(history):
    lines = []
    for message in (history or [])[-4:]:
        role = "用户" if message.get("role") == "user" else "助手"
        text = _text_of(message.get("content")).strip()
        if text:
            lines.append(f"{role}：{text[:240]}")
    return "\n".join(lines) or "（无）"


def _numbers(text):
    return set(re.findall(r"\d+(?:\.\d+)?", text or ""))


def _bigrams(text):
    text = re.sub(r"\s+", "", text or "")
    return {text[i:i + 2] for i in range(max(0, len(text) - 1))}


def _zh_sentences(text):
    parts = [p.strip() for p in re.split(r"(?<=[。！？；!?;])", text or "") if p.strip()]
    out = []
    for part in parts:
        while len(part) > 120:
            out.append(part[:120])
            part = part[120:]
        if part:
            out.append(part)
    return out


def _clean_polish(text):
    text = (text or "").strip()
    text = re.sub(r"^(改写|润色|修改|优化)(后)?(的)?(中文|句子|版本)?\s*[:：]\s*", "", text)
    for label in ("改写：", "润色：", "中文："):
        if label in text:
            text = text.rsplit(label, 1)[-1].strip()
    return text.strip().strip('"“”\'')


def _polish_ok(out, original):
    if not out or len(out) < 2:
        return False
    if len(CJK.findall(out)) < 2:
        return False
    ratio = len(out) / max(len(original), 1)
    if ratio < 0.5 or ratio > 2.0:
        return False
    if not _numbers(out) <= _numbers(original):
        return False
    if ("原文" in out or "改写" in out or "PyBot" in out) and not any(
        word in original for word in ("原文", "改写", "PyBot")
    ):
        return False
    # 改写后应与原句共享足够多的字词；否则视为跑题/套话
    a, b = _bigrams(out), _bigrams(original)
    if a and b and len(a & b) / max(len(b), 1) < 0.25:
        return False
    return True


def polish_zh(draft, history=None, user_text="", en_reply=""):
    """结合最近对话，用 LFM2.5 逐句润色中文回答；失败或未加载时原样返回。"""
    draft = (draft or "").strip()
    if len(draft) < 6 or not CJK.search(draft):
        return draft
    if not _lfm_ready.is_set() or _lfm["error"]:
        return draft
    context = _context_text(history)
    key = (draft, context, user_text[:200], en_reply[:300])
    with _POLISH_LOCK:
        if key in _POLISH_CACHE:
            return _POLISH_CACHE[key]
    parts = []
    changed = False
    for sentence in _zh_sentences(draft):
        prompt = POLISH_TEMPLATE.format(
            context=context,
            user=user_text[:200] or "（无）",
            en_reply=en_reply[:400] or "（无）",
            draft=draft[:600],
            target=sentence,
        )
        try:
            raw = _lfm_generate(
                [
                    {"role": "system", "content": POLISH_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                max_new_tokens=min(192, max(64, len(sentence) * 2)),
            )
        except Exception:  # noqa: BLE001 - 单句失败不影响整段
            parts.append(sentence)
            continue
        out = _clean_polish(raw)
        if out and out != sentence and _polish_ok(out, sentence):
            parts.append(out)
            changed = True
        else:
            parts.append(sentence)
    result = "".join(parts) if changed else draft
    with _POLISH_LOCK:
        if len(_POLISH_CACHE) >= _POLISH_MAX:
            _POLISH_CACHE.pop(next(iter(_POLISH_CACHE)))
        _POLISH_CACHE[key] = result
    return result


# --- NPC 对话 ---------------------------------------------------------------
NPC_CHOICES = {
    "Grom · 铁匠": "grom",
    "Lina · 老板娘": "lina",
    "小红 · 青梅竹马": "xiaohong",
}


def _npc_key(choice):
    return NPC_CHOICES.get(str(choice), "grom")


def _ensure_npc(memory, choice, guild=False):
    memory = memory if isinstance(memory, dict) else {}
    key = _npc_key(choice)
    state = memory.get("npc")
    if not (isinstance(state, dict) and state.get("enabled") and state.get("profile") == key):
        npc_activate(memory, key, guild=bool(guild))
    elif bool(guild) != bool(state.get("guild")):
        npc_activate(memory, key, guild=bool(guild))
    return memory, PROFILES[key]


def _npc_status(profile, memory, note="就绪"):
    st = memory.get("npc") or {}
    return (
        f"**{note}**　**NPC**：{profile['name']}（{profile['role']}）"
        f"　**信任**：{st.get('trust', 0)}/100"
        f"　**公会**：{'是' if st.get('guild') else '否'}"
        f"　**已知**：{len(st.get('revealed', []))}/{len(profile['facts'])}"
        f"　**轮次**：{st.get('turn', 0)}"
    )


def _chat_fast(message, history, memory, profile):
    """纯英文快速模式：不碰翻译与润色，英文进、英文出。"""
    result = _get_pybot().turn(message, memory)
    en_reply = str(result.get("reply") or "")
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": en_reply},
    ]
    status = (
        _npc_status(profile, memory, "就绪")
        + f"　**模块**：`{result.get('module')}`"
        "　**流程**：英文直通（无翻译）　**润色**：关"
    )
    return "", history, memory, status


def chat(message, history, memory, npc_choice, guild, show_en, engine, polish, fast_en=True):
    history = list(history or [])
    memory, profile = _ensure_npc(memory, npc_choice, guild)
    text = (message or "").strip()
    if not text:
        return "", history, memory, _npc_status(profile, memory, "等待输入")
    engine_key = "lfm" if str(engine).startswith("LFM") else (
        "opus" if str(engine).startswith("Opus") else "auto"
    )

    user_is_zh = bool(CJK.search(text))
    note = ""
    try:
        if fast_en:
            if CJK.search(text):
                reply = (
                    "纯英文快速模式已开启：请直接用英文和 NPC 说话，"
                    "或取消勾选“纯英文快速模式”改用中文。"
                )
                history += [
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": reply},
                ]
                return "", history, memory, _npc_status(profile, memory, "需要英文输入")
            return _chat_fast(text, history, memory, profile)
        if user_is_zh:
            zh_user = text
            en_user = translate(text, "zh2en", engine_key)
            if not en_user:
                reply = "这句中文没能翻译成英文，NPC 读不到。请换一种说法，或把句子缩短一点。"
                history += [
                    {"role": "user", "content": zh_user},
                    {"role": "assistant", "content": reply},
                ]
                return "", history, memory, _npc_status(profile, memory, "中→英失败")
        else:
            en_user = text
            zh_user = translate(text, "en2zh", engine_key)
            if not zh_user:
                zh_user = text
                note = "（原句中文翻译失败，显示英文）"

        result = _get_pybot().turn(en_user, memory)
        en_reply = str(result.get("reply") or "")
        zh_reply = translate(en_reply, "en2zh", engine_key)
        polish_note = "关"
        if not zh_reply:
            zh_reply = "（翻译失败，以下为 NPC 英文原文）\n\n" + en_reply
            note = (note + " 回复翻译失败。").strip()
            polish_note = "跳过"
        elif polish:
            polished = polish_zh(zh_reply, history, zh_user, en_reply)
            if polished != zh_reply:
                zh_reply = polished
                polish_note = "完成"
            else:
                polish_note = "跳过"

        shown = zh_reply
        if show_en and en_reply:
            shown += "\n\n> **" + profile["name"] + ":** " + en_reply
        if note:
            shown += "\n\n*" + note + "*"
        history += [
            {"role": "user", "content": zh_user},
            {"role": "assistant", "content": shown},
        ]
        step = "中→英" if user_is_zh else "英→中"
        status = (
            _npc_status(profile, memory, "就绪")
            + f"　**模块**：`{result.get('module')}`"
            f"　**流程**：{step} → NPC → 英→中"
            f"　**引擎**：{engine}　**润色**：{polish_note}"
        )
        return "", history, memory, status
    except Exception as exc:  # noqa: BLE001 - 一轮失败不能让页面 500
        history += [
            {"role": "user", "content": text},
            {"role": "assistant", "content": f"出错了：{type(exc).__name__}: {exc}"},
        ]
        return "", history, memory, _npc_status(profile, memory, "错误")


def _status(note="就绪"):
    opus = "失败" if _opus["error"] else ("就绪" if _opus_ready.is_set() else "加载中")
    lfm = "失败" if _lfm["error"] else ("就绪" if _lfm_ready.is_set() else "加载中")
    bot = "失败" if _pybot["error"] else ("就绪" if _pybot_ready.is_set() else "加载中")
    return f"**{note}**　Opus-MT：{opus}　LFM2.5：{lfm}　PyBot NPC：{bot}"


def reset(npc_choice="Grom · 铁匠", guild=False):
    memory = {}
    memory, profile = _ensure_npc(memory, npc_choice, guild)
    return [], memory, _npc_status(profile, memory, "新对话")


with gr.Blocks(title="NPC 模拟") as demo:
    gr.Markdown(
        "# NPC 模拟\n"
        "三位规则驱动的 NPC：**Grom Ironfoot**（矮人铁匠）、**Lina Malt**（醉灯笼旅店老板娘）"
        "和**小红**（青梅竹马）。\n\n"
        "**默认开启纯英文快速模式**：直接用英文对话，跳过翻译和润色，几乎没有额外等待。"
        "取消勾选后可用中文，NPC 的回答会译回中文并用 LFM2.5 润色。\n\n"
        "权限、秘密、信任和任务由 `bot/npc/` 的确定性规则决定，翻译和润色只改措辞，不改事实。"
    )
    with gr.Row():
        with gr.Column(scale=3):
            npc_choice = gr.Radio(
                choices=list(NPC_CHOICES), value="Grom · 铁匠", label="选择 NPC"
            )
            chatbot = gr.Chatbot(label="对话（快速模式为英文）", height=430)
            msg = gr.Textbox(
                label="你说（快速模式用英文；关闭快速模式可用中文）",
                placeholder="例如：Who are you? / Where is Tu? / Do you know this is a game world?",
                lines=2,
            )
            with gr.Row():
                send = gr.Button("发送", variant="primary")
                clear = gr.Button("重来")
            with gr.Row():
                guild = gr.Checkbox(label="以公会成员身份进入（会重置当前 NPC）", value=False)
                fast_en = gr.Checkbox(label="纯英文快速模式（跳过翻译与润色，默认开）", value=True)
            with gr.Row():
                show_en = gr.Checkbox(label="显示 NPC 英文原文（中文模式下）", value=True)
                polish = gr.Checkbox(label="用 LFM2.5 结合上下文润色（默认开启）", value=True)
            engine = gr.Radio(
                choices=["自动（推荐）", "LFM2.5（实验）", "Opus-MT"],
                value="自动（推荐）",
                label="翻译引擎",
            )
        with gr.Column(scale=2):
            status = gr.Markdown(_status())
            refresh = gr.Button("刷新状态")
            gr.Markdown(
                "### 这一轮发生了什么\n"
                "**快速模式（默认）**：你的英文 → NPC 规则引擎 → 英文回答，不经过翻译。\n\n"
                "**中文模式**：你的中文 → 英文 → NPC 规则引擎 → 英文回答 → 中文 → LFM2.5 润色。\n\n"
                "两种模式下，决定说什么的都是规则引擎（权限、秘密、信任、任务）。\n\n"
                "试试：`Who are you?`、`Where is Tu?`、`Do you know this is a game world?`、"
                "`What do you sell?`、`Tell me a secret.`、`What's in the cellar?`\n\n"
                "秘密分层：公开 / 公会 / 机密。机密需要信任 ≥ 50 或持有符文碎片。"
            )
            gr.Examples(
                examples=[
                    "Who are you?",
                    "Where is Tu?",
                    "Do you know this is a game world?",
                    "What do you sell?",
                    "Tell me a secret.",
                    "What's in the cellar?",
                ],
                inputs=msg,
            )

    state = gr.State({})
    inputs = [msg, chatbot, state, npc_choice, guild, show_en, engine, polish, fast_en]
    outputs = [msg, chatbot, state, status]
    msg.submit(chat, inputs, outputs, api_visibility="private")
    send.click(chat, inputs, outputs, api_name="ask")
    clear.click(reset, [npc_choice, guild], [chatbot, state, status], api_visibility="private")
    npc_choice.change(reset, [npc_choice, guild], [chatbot, state, status], api_visibility="private")
    guild.change(reset, [npc_choice, guild], [chatbot, state, status], api_visibility="private")
    refresh.click(lambda: _status(), None, status, api_visibility="private")


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), share=False)
