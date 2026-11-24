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
- **台词库**：258 条离线写好的同义台词（英文 + 中文），运行时只做一次查表。
  规则引擎仍决定 beat 和事实；台词库只换措辞，并按会话轮换，避免重复。
- **信任分档**：小红的问候按信任分成 cold / warm / close 三档，同一个人在不同
  关系下开场白不同；高分事实还会给第二层追问（`more`）和事实专属的 `why`。
- **个人问题单独成档**：年龄、生日、家人、喜好、梦想、对 Grom/Lina 的看法等
  走 `personal` beat，不会再被当成“学者问题”答非所问。
- **后台自扩**：玩家读字/打字时，LFM2.5 会参考最近几轮对话，为刚说过的 beat
  生成新的同义台词，逐条校验后入库；模型调用用非阻塞锁，永远不会拖慢前台。
  词库按 beat 上限 16 条、总计 900 条封顶，并尝试写入 `bank_runtime.json`。
  设 `NPC_BG_GROW=0` 可关闭；`NPC_BANK_MAX_PER_BEAT` / `NPC_BANK_MAX_TOTAL`
  可调高上限。
- **语义判断（可选）**：规则 Reader 落到 `fallback / out_of_world` 时，交给
  开源的 GLiNER2.5-Decide（340M，Apache-2.0，英文单次前向）在固定意图集里选一个。
  它只补漏，不覆盖 SAFETY / BOUNDARY / JAILBREAK / 权限判断；装不上就直接停用。
  启用：在 `requirements.txt` 打开 `gliner2[local]>=2.1`；模型默认从
  `fastino/GLiNER2.5-Decide` 拉取，可用 `NPC_SEMANTIC_MODEL` 指向本地路径，
  `NPC_SEMANTIC_THRESHOLD` 调阈值，`NPC_SEMANTIC=0` 关闭。
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
| `bot/npc/bank.json` | 离线台词库（英文 + 中文，258 条），运行时只读 |
| `bot/npc/bank.py` | 台词库加载、轮换选择、运行时增量，缺失时自动回退 |
| `bot/npc/grower.py` | 后台自扩：按最近对话挑 beat、生成、校验、入库 |
| `bot/npc/semantic.py` | 可选语义判断：规则没接住时用 GLiNER2.5-Decide 补一个意图 |
| `bot/npc/reader.py` | 意图分类（SAFETY / BOUNDARY / JAILBREAK / GAME / SECRET / QUEST…） |
| `bot/npc/engine.py` | 权限、状态、骰子、台词选择、防泄露 |
| `bot/npc/state.py` | 信任、公会、已揭示事实、物品 |
| `bot/npc/lines_en.py` | 通用 beat 的英文台词 |
| `bot/` | PyBot 规则引擎，保留原样 |
| `requirements.txt` | transformers / tokenizers / modelscope / sentencepiece（torch、gradio 由镜像提供） |

## 限制

- 中文模式的翻译/润色在 2 vCPU 上需要几秒到几十秒；首轮要等模型下载。
- 自扩台词是**通用句**，不针对单个玩家：它只借鉴最近对话的语气和话题，
  不会把某个玩家说过的话写进公共库，避免串台和事实污染。
- LFM2.5-350M 的生成质量不稳定，多数候选会被校验器拒掉；空间跑几分钟后
  词库才会明显变多，这是预期行为。
- LFM2.5-350M 的长句翻译会跑偏，长回答主要靠 Opus-MT。
- 免费 CPU 实例单并发，慢请求会排队。
- 当前是英文 NPC 内容 + 可选中文翻译层；要加其他语言，只需替换翻译层。
