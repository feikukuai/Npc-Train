---
title: 阿禾 · 五岁半 NPC
emoji: 🌿
colorFrom: yellow
colorTo: green
sdk: gradio
sdk_version: 6.17.3
app_file: app.py
pinned: false
license: apache-2.0
short_description: 五岁半的中文小 NPC，从句库里选整句回答
---

# 阿禾 · 五岁半

一个不是“预测下一个字”的中文小 NPC。她把玩家的一句话编码成一个向量，用 64 维记忆携带上下文，然后从自己会的 **68 句**里选一句说出来。

她只认识村口药铺、屋后药园、老井、橘猫阿团和娘。知识面小、话语少、记忆短，但每次输出的都是一句完整的话。

## 怎么工作

```
玩家这句话 ──> 冻结的中文句向量 (bge-small-zh, 512 维)
                    │
状态：信任 / 回合 / 情绪 / 上一轮 beat / 已知名字
                    │
             64 维 GRU 记忆
                    │
        ┌───────────┴────────────┐
   句库模型：68 句里选一句      规则老师：意图 / 权限 / 事实
```

- **句库模型**：`expertnet/zh/train.py` 训练出的 64 维记忆选择器，参数约 18 万。
- **规则老师**：决定意图、秘密权限、名字记忆和情绪；句库模型只学“怎么接”。
- **上下文**：`mood` 被骂会下降，下一轮告别会变成“哼，你走吧”；上一轮 beat 和是否知道玩家名字都进入状态。
- **保底**：如果创空间第一次下载句向量失败，自动退回规则老师，页面仍然可用。

### 推理与记忆

部署时采用混合策略，避免“模型猜事实”：

- 事实、身份、母亲名字、学习、蚂蚁数量、名字、秘密权限、刚才发生了什么 → **规则老师**；
- 问候、告别、感谢、道歉、闲聊、被骂后的反应 → **句库模型**；
- `recent` 保存最近 6 轮（玩家原话 + 回复 + beat），所以“刚才发生了什么”能答出来；
- 规则老师的 beat 不在句库训练集时，用 `BEAT_ALIASES` 映射到旧槽位，不破坏 34 维状态。

例如：

```
玩家：你妈妈谁          -> 娘在药铺抓药，她认得所有草。
玩家：我问她名字        -> 我娘叫阿棠。
玩家：刚才发生了什么    -> 你刚才问我记不记得你的名字。
```

## 指标

1000 会话 × 6 轮（6000 个训练回合），HELDOUT 换说法测试：

| 指标 | 数值 |
|---|---|
| 整句完全命中 | 0.827 |
| 句子类型（beat）命中 | 0.833 |
| 整句 top-3 | 0.925 |
| 携带上下文 vs 每轮重置 | 0.827 vs 0.350（+0.477） |
| 固定上下文脚本 | 15/15 |

长对话对抗训练（Attacker / Student / Auditor，12 段固定长对话）：

| 指标 | 训练前 | 训练后 |
|---|---|---|
| 长对话整句命中 | 0.793 | **0.880** |
| 长对话行为命中 | 0.870 | **0.880** |
| 情绪类回合 | 0.750 | **1.000** |
| HELDOUT 整句 | 0.817 | 0.814 |

对抗只改善“接得对不对”，不增加新知识；句库和老师仍是天花板。详细交接说明见
[AGENTS.md](AGENTS.md)。

## 本地运行

```bash
python -m expertnet.zh.train --sessions 1000 --epochs 30 --demo
python -m expertnet.zh.adversarial --rounds 3 --sessions 80 --steps 14
python tools/chat_web.py --lang zh --port 7861
```

创空间直接运行 `app.py`。首次启动会下载约 95MB 的 `BAAI/bge-small-zh-v1.5`，之后走缓存。

- 代码与权重：<https://github.com/feikukuai/Npc-Train>
- 创空间：<https://www.modelscope.cn/studios/feihukuai/PyBot-ZH>

## 文件

| 路径 | 作用 |
|---|---|
| `app.py` | 创空间入口：单人 Gradio 界面 |
| `expertnet/zh/chat.py` | 无界面的会话核心，模型不可用时退回规则 |
| `expertnet/zh/persona.py` | 阿禾的事实、台词、情绪、名字记忆、状态向量 |
| `expertnet/zh/train.py` | 整句选择器的训练、评估与上下文脚本 |
| `expertnet/zh/atoms_zh.safetensors` | 训练好的 68 句选择器权重（约 0.7MB） |
| `expertnet/zh/atoms_zh.json` | 句库、beat 与模型配置 |
| `web/chat.html` | 本地对照网页（规则 vs 句库） |

旧的三位英文 NPC（Grom / Lina / 小红）仍保留在 `bot/npc/` 与 git 历史里，但不再作为本创空间的入口。
