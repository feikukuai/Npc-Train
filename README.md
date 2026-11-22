---
title: PyBot 中文版
emoji: 🤖
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 6.17.3
app_file: app.py
pinned: false
license: apache-2.0
short_description: PyBot 纯规则聊天机器人 + Opus-MT/LFM2.5 中英互译
---

# PyBot 中文版

PyBot 是一个**没有权重、没有训练**的英文聊天机器人：528 个 Python 文件、约 64 万行代码，全部由 if/else、正则和查表组成。上游项目：[huggingface/pybot](https://github.com/huggingface/pybot)，Apache-2.0。

这个创空间在 PyBot 外面加了一层翻译：

```
用户中文 → 翻译模型译成英文 → PyBot 规则引擎回答 → 翻译模型译回中文
```

- 真正决定“怎么回答”的是 PyBot 的规则，翻译模型不参与知识或策略。
- 你说英文也可以：页面会先把你的话译成中文显示，再把英文原句交给 PyBot。
- **翻译引擎**：默认 `Opus-MT`（Helsinki-NLP 专用翻译模型，长句稳定）；`LFM2.5-350M` 仅作实验对照。实测 350M 能处理短句，但长回答会跑偏，所以不作为默认。
- 翻译失败会自动回退：中文没译成英文就不送 PyBot；英文回答没译成中文就显示英文原文。

## 本空间内容

| 路径 | 内容 |
|---|---|
| `app.py` | Gradio 界面、两个翻译引擎、中英互译管线 |
| `bot/` | PyBot 本体，原样保留，未改回答逻辑 |
| `requirements.txt` | transformers / tokenizers / modelscope / sentencepiece（torch、gradio 由镜像提供） |

## 限制

- Opus-MT 对长句、俚语、专有名词也会出错，但比 350M 稳定得多。
- LFM2.5-350M 的翻译质量有限：短句可用，长回答会跑偏，仅作对照。
- 免费 CPU 实例单并发，慢请求会排队。
- PyBot 是通用英文聊天机器人，不是游戏 NPC；NPC 版本在另一个创空间 `feihukuai/LFM2.5-350M` 的第二个 Tab。
