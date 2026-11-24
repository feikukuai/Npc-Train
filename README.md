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

一个不是“预测下一个字”的中文小 NPC。她把玩家的一句话编码成一个向量，用 64 维记忆携带上下文，然后从自己会的 **84 句**里选一句说出来。

她只认识村口药铺、屋后药园、老井、橘猫阿团和娘。知识面小、话语少、记忆短，但每次输出的都是一句完整的话。

## 怎么工作

```
玩家这句话 ──> 冻结的中文句向量 (bge-small-zh, 512 维)
                    │
状态：信任 / 回合 / 情绪 / 上一轮 beat / 已知名字 (37 维)
                    │
             64 维 GRU 记忆
                    │
        思考标签 (10 类) ──> 软嵌入
                    │
        ┌───────────┴────────────┐
   句库模型：84 句里选一句      规则老师：意图 / 权限 / 事实
```

- **思考模式**：先预测一个可解释的上下文标签（`fresh`、`after_insult`、`memory_probe`、`fact_probe`…），
  再把它的软嵌入送进句子头。网页状态栏会显示这一轮的“思考”。
- **句库模型**：`expertnet/zh/train.py` 训练出的 64 维记忆选择器，参数约 18 万。
- **规则老师**：决定意图、秘密权限、名字记忆和情绪；句库模型只学“怎么接”。
- **上下文**：`mood` 被骂会下降，下一轮告别会变成“哼，你走吧”；上一轮 beat、是否知道玩家名字和最近一轮原话都进入状态。
- **保底**：如果创空间第一次下载句向量失败，自动退回规则老师，页面仍然可用。

### 推理与记忆

部署时采用混合策略，避免“模型猜事实”：

- 事实、身份、母亲名字、学习、蚂蚁数量、名字、秘密权限 → **规则老师**；
- 问候、告别、感谢、道歉、闲聊、被骂后的反应、taunt、apology_meta、memory_recall → **句库模型**；
- `recent` 保存最近 6 轮（玩家原话 + 回复 + beat），记忆模板 `{last_player}` 在显示时填入；
- 新增的 `memory_recall` / `taunt` / `apology_meta` 已经是正式 beat，31 个 slot 全部进入 37 维状态。

例如：

```
玩家：你妈妈谁          -> 娘在药铺抓药，她认得所有草。
玩家：我问她名字        -> 我娘叫阿棠。
玩家：刚才发生了什么    -> 你刚才问我记不记得你的名字。   （思考：memory_probe）
```

## 指标

1200 会话 × 6 轮（7200 个训练回合），HELDOUT 换说法测试：

| 指标 | 数值 |
|---|---|
| 整句完全命中 | 0.767 |
| 句子类型（beat）命中 | 0.809 |
| 思考标签命中 | 0.919 |
| 整句 top-3 | 0.888 |
| 携带上下文 vs 每轮重置 | 0.767 vs 0.351（+0.416） |
| 固定上下文脚本 | 19/20 |

长对话对抗训练（Attacker / Student / Auditor，14 段固定长对话）：

| 指标 | 训练前 | 训练后 |
|---|---|---|
| 长对话整句命中 | 0.815 | **0.889** |
| 长对话行为命中 | 0.889 | **0.898** |
| 长对话思考命中 | 0.938 | 0.901 |
| 情绪类回合 | 1.000 | 1.000 |
| 名字类回合（原始学生） | 0.500 | 0.500 |
| HELDOUT 整句 | 0.769 | 0.771 |

对抗只改善“接得对不对”，不增加新知识；句库和老师仍是天花板。名字类原始学生仍弱，
部署时走规则。详细交接说明见 [AGENTS.md](AGENTS.md)。

## 缩放实验：500 → 50,000 条长链

用 Player / Teacher / Attacker 生成 500、5,000、50,000 条长链路对话，从同一基线
微调，在 14 段固定长对话 + 200 条 HELDOUT 长链上评分：

| 模型 | atom | behaviour | thought | deploy reasoning |
|---|---|---|---|---|
| baseline | 0.791 | 0.804 | 0.859 | 0.651 |
| +500 | **0.798** | **0.808** | 0.871 | 0.651 |
| +5,000 | 0.780 | 0.796 | 0.904 | 0.643 |
| +50,000 | 0.713 | 0.729 | **0.913** | 0.607 |

结论：**没有智力涌现**。500 条只有噪声级提升；5,000/50,000 反而让句子选择变差，
只有思考标签继续上升。原因是数据来自同一个规则老师和同一套 84 句话，模型只有
64 维记忆；堆同类合成数据不会产生新推理。完整方法、测试集和评分见
[SCALING.md](SCALING.md)。

## 本地运行

```bash
python -m expertnet.zh.train --sessions 1200 --epochs 30 --demo
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
