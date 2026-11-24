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
short_description: 铁匠 Grom 与老板娘 Lina 的规则驱动 NPC 模拟
---

# NPC 模拟

两位规则驱动的 NPC：**Grom Ironfoot**（矮人铁匠）和 **Lina Malt**（醉灯笼旅店老板娘）。

对话流程：

```
用户中文 → 译成英文 → NPC 规则引擎（权限/秘密/信任/任务）→ 英文回答 → 译回中文 → LFM2.5 润色
```

- 决定“说什么”的是 `bot/npc/` 的确定性规则：身份、交易、传闻、任务、秘密分层。
- 秘密分三层：公开、公会（需要公会身份）、机密（信任 ≥ 50 或持有符文碎片）。
- 翻译和润色只改措辞，不参与决策；数字、人名、事实被改动时自动回退。
- 支持中文或英文输入；NPC 的英文原文可以勾选显示。
- **翻译引擎**：默认 `自动`——中文短问句用 `LFM2.5-350M`，英文长回答用 `Opus-MT`。
- **润色**：默认开启。LFM2.5 会看最近几轮对话，只改中文措辞；越界或新增事实时放弃润色。

## 本空间内容

| 路径 | 内容 |
|---|---|
| `app.py` | Gradio 界面、NPC 选择、翻译与润色管线 |
| `bot/npc/` | NPC 内容、读数、状态、话术、引擎 |
| `bot/` | PyBot 规则引擎，保留原样 |
| `requirements.txt` | transformers / tokenizers / modelscope / sentencepiece（torch、gradio 由镜像提供） |

## 限制

- 翻译/润色在 2 vCPU 上需要几秒到几十秒；首轮要等模型下载。
- LFM2.5-350M 的长句翻译会跑偏，长回答主要靠 Opus-MT。
- 免费 CPU 实例单并发，慢请求会排队。
- 当前是英文 NPC 内容 + 中文翻译层；要加其他语言，只需替换翻译层。
