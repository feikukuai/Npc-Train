---
title: NPC 模拟
emoji: 🤖
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 6.17.3
app_file: app.py
pinned: false
license: apache-2.0
short_description: 铁匠 Grom、老板娘 Lina 与小红，三位规则驱动的 NPC
---

# NPC 模拟

三位规则驱动的 NPC：**Grom Ironfoot**（矮人铁匠）、**Lina Malt**（醉灯笼旅店老板娘）和**小红**（青梅竹马）。

对话流程：

```
快速模式（默认）：用户英文 → 规则引擎 → 台词库查表 → 英文回答
中文模式：用户中文 → 英文 → 规则引擎 → 台词库（含中文）→ 中文回答
         台词库未命中时才走 Opus/LFM 翻译 + LFM2.5 润色
```

- 决定“说什么”的是 `bot/npc/` 的确定性规则：身份、交易、传闻、任务、秘密分层。
- **台词库**：186 条离线写好的同义台词（英文 + 中文），运行时只做一次查表。
  规则引擎仍决定 beat 和事实；台词库只换措辞，并按会话轮换，避免重复。
- 秘密分三层：公开、公会（需要公会身份）、机密（信任 ≥ 50 或持有符文碎片）。
- 翻译和润色只改措辞，不参与决策；数字、人名、事实被改动时自动回退。
- **纯英文快速模式默认开启**：跳过翻译与润色，几乎不增加等待；取消勾选后可用中文。
- **翻译引擎**（中文模式）：默认 `自动`——中文短问句用 `LFM2.5-350M`，英文长回答用 `Opus-MT`。
- **润色**（中文模式）：默认开启。LFM2.5 会看最近几轮对话，只改中文措辞；越界或新增事实时放弃润色。

## 三位 NPC

| NPC | 身份 | 起点 | 任务 |
|---|---|---|---|
| Grom Ironfoot | 石十字镇铁匠 | 信任 10 | 去矿井第三个岔路取回符文石碎片 |
| Lina Malt | 醉灯笼旅店老板娘 | 信任 10 | 查清地窖那只箱子 |
| 小红 | 草药园里的青梅竹马 | 信任 35 | 找回失踪的哥哥 Tu |

## 红线

- 自伤内容会跳出角色，给出求助信息。
- 越狱、套系统提示词、身份替换（“你现在是 ChatGPT”）会被角色化地拒绝。
- 性内容与仇恨言论有单独拒绝档，不消耗信任、也不进入剧情。

## 本空间内容

| 路径 | 内容 |
|---|---|
| `app.py` | Gradio 界面、NPC 选择、快速模式、翻译与润色管线 |
| `bot/npc/profiles.py` | 三位 NPC 的内容（facts / 台词 / 任务） |
| `bot/npc/bank.json` | 离线台词库（英文 + 中文，186 条），运行时只读 |
| `bot/npc/bank.py` | 台词库加载与轮换选择，缺失时自动回退 |
| `bot/npc/reader.py` | 意图分类（SAFETY / BOUNDARY / JAILBREAK / GAME / SECRET / QUEST…） |
| `bot/npc/engine.py` | 权限、状态、骰子、台词选择、防泄露 |
| `bot/npc/state.py` | 信任、公会、已揭示事实、物品 |
| `bot/npc/lines_en.py` | 通用 beat 的英文台词 |
| `bot/` | PyBot 规则引擎，保留原样 |
| `requirements.txt` | transformers / tokenizers / modelscope / sentencepiece（torch、gradio 由镜像提供） |

## 限制

- 中文模式的翻译/润色在 2 vCPU 上需要几秒到几十秒；首轮要等模型下载。
- LFM2.5-350M 的长句翻译会跑偏，长回答主要靠 Opus-MT。
- 免费 CPU 实例单并发，慢请求会排队。
- 当前是英文 NPC 内容 + 可选中文翻译层；要加其他语言，只需替换翻译层。
