# 交接手册：阿禾（Npc-Train）

这份文档写给下一个接手训练的 Agent。目标不是“再训一个更大的模型”，而是维护一个可复现、
可评测、可部署的中文整句选择器。先读完再改代码。

## 0. 当前状态（2026-11）

> **2026-11-25 主线已切到因子化模型**：`expertnet/zh/factorized_train.py` 训练、
> `context_app.py` 默认 `CHAT_BACKEND=factorized`，3337 语义簇只作紧急回退。
> - 输出表：`exhaustive.py --fixed-bank 265` → 336,701 句（6,971 固定句 + 组合句）；
>   26 个 beat 每个约 260 条，按长度排序后等步长抽样，覆盖不同句法框架。
> - 架构：BGE 逐行编码 → GRU 读上下文；另一路只读最后一句，推理时两路 logits 相加。
> - 训练：12 万教师对话（真实说法直接监督固定句）+ 10.8 万合成槽位问句；
>   留存换说法的行为准确率 **0.68**，模式 0.99，合成组合整句 0.999。
> - 推理：**beat 概率 + 小决策头全局集成**（`decision_head.pt`，α=0.5）后选句；
>   事实问句由关键词规则**先于模型**兜底（`FACT_BEATS` 集合内），问“我叫什么”
>   由 `_name_pick` 锁定名字句，问“你叫什么”走事实句。
> - **评测口径（2026-11 起）**：`tools/quality_eval.py`，主指标是
>   beat **0.81** / 合法率 0.9825 / 近 8 句重复 5.5% / 事实命中 **0.985** / 多样性 0.446；
>   exact 只作参考（0.45），**不要把它当智力指标**。
>   exact 的上限由教师决定的：`persona._pick` 是 `lines[turn % len]` 机械轮换，
>   而 context 只保留最近 4 轮，turn 丢失；30/31 个 beat 只有 2–3 种目标句。
>   把 turn one-hot 直接喂给决策头，exact 从 0.364 升到 0.533（12 epoch 仍在涨），
>   证明瓶颈是“轮次信息不可见”，不是模型容量。`tools/turn_probe.py` 可复现。
> - 试过但**没有收益**，不要重复：
>   ① GRPO 只更新决策头 proj（组采样 8、奖励含 beat/事实/重复/长度），exact 0.429→0.419；
>   ② 无标签规则 Bo-4 重排 0.377；
>   ③ cross-encoder 验证器（随机同 beat 负例 + BCE）Bo-8 0.280、listwise 对齐策略分布
>   后 Bo-8 0.344 / Bo-16 0.355（覆盖 0.766）——候选里有 gold，但验证器选不出来，
>   因为 gold 的一半决定因素（turn）不在输入里；
>   ④ 输入闭集化：最后一句先映射到 SEEDS 最近句（阈值 0.86 命中 50.8%），
>   beat 0.7997→0.7966，无收益；阈值 0.80 更差。
> - 计划头（11 类）准确率 0.83，仅作辅助监督。
>   `warmup()` 会在后台预编码全部固定句并构建决策表，首句延迟从 ~2.8s 降到 ~0.05s。
> - 决策头（`tools/decision_head.py --epochs 12 --k 24`，同 beat 难负例 9 个）：
>   全表直选 beat 0.63 / exact 0.35；已定 beat 内选句 exact 0.53。
> - 说法扰动（`--augment 0.5`）+ 教师同义词/纠正规则 + 新增 seed，把留存行为准确率
>   从 0.68 提到 **0.703**；集成后 0.81。
> - 复现：`python -m expertnet.zh.exhaustive --out expertnet/zh/exhaustive.jsonl --fixed-bank 265`
>   然后 `python -m expertnet.zh.factorized_train --device cuda --samples 120000 --row-limit 120000 --augment 0.5`，
>   再跑 `tools/decision_head.py --epochs 12 --k 24` 训练决策头，
>   `tools/quality_eval.py` 出主指标，`tools/rerank_eval.py` 复核融合权重。
> - 局限：第一人称情绪（“我有点累”）还没有老师规则，会落到 fallback；秘密类在
>   信任不足时被规则拒答（评测里 2 条“失败”属于正确行为）。

- 角色：**阿禾**，五岁半，村口药铺家的孩子。只认识药铺、药园、老井、橘猫阿团和娘。
- 输出：规则老师先定 **beat**，再从 **191,600 句**整句库里按上下文挑一句；
  不再是 86 句固定台词，也不是逐字生成。
- 记忆：一个 **96 维** GRU 状态，跨轮携带。
- 思考：一个 **10 类**上下文标签（`fresh` / `after_insult` / `memory_probe`…），软嵌入送进句子头。
- 句向量：`BAAI/bge-small-zh-v1.5`，512 维，CLS pooling，冻结。
- 规则老师：`expertnet/zh/persona.py`，决定意图、事实权限、情绪、名字记忆。
- 基础训练：固定测试 atom 约 **0.808**，beat 0.816，思考标签 0.932。
- 长对话对抗训练（64 维版本历史）：固定长对话整句命中 **0.815 → 0.889**，行为 0.889 → 0.898。
- 名字专项：96 维候选 raw name atom **0.667 → 0.875**，部署 0.707 → 0.708。
- 整句库：26 个 beat，19.2 万句；406 轮评测不重复回复 **83.5%**，近 8 轮重复 1.2%，
  名字/刚才的话均命中，事实与边界 77/77 用规则原句，挑选延迟 avg 15.7 ms。
- 部署：ModelScope 创空间 `feihukuai/PyBot-ZH`，入口 `app.py`。
- GitHub：`https://github.com/feikukuai/Npc-Train`。

## 1. 环境

- Windows PowerShell + Python 3.12：
  `C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe`
- 依赖：`requirements.txt`；本地另需 `gradio>=6`、`torch`（创空间基础镜像自带）。
- 权重通过 **Git LFS** 存储：clone 前先 `git lfs install`，否则拿到的只是指针文件。
- 首次运行需要下载：
  - 中文句向量 `BAAI/bge-small-zh-v1.5`（约 95MB）
  - 英文句向量 `sentence-transformers/all-MiniLM-L6-v2`（约 90MB，仅英文实验用）
- 网络不好时设 `HF_ENDPOINT=https://hf-mirror.com`；缓存好之后设 `HF_HUB_OFFLINE=1`
  可以避免每次启动检查网络。`expertnet/zh/chat.py` 会先删掉 `HF_HUB_OFFLINE`，
  再尝试本地缓存 / HF 镜像 / ModelScope 快照，全失败则退回规则模式。

```powershell
cd PyBot-ZH
$env:PYTHONPATH=(Get-Location).Path
```

## 2. 目录地图

| 路径 | 作用 |
|---|---|
| `app.py` | 创空间入口，单人 Gradio 界面 |
| `expertnet/zh/persona.py` | 阿禾的事实、台词、老师逻辑、37 维状态向量 |
| `expertnet/zh/grammar.py` | 句法框架 × 人设词表，确定性生成整句库 |
| `expertnet/zh/bank.py` | 整句索引、缓存、按情绪/用词/最近回复挑选 |
| `expertnet/zh/bank_eval.py` | 整句库评测：规模、重复率、名字/记忆、安全、延迟 |
| `expertnet/zh/corpus.py` | 训练说法 `SEEDS` 与独立测试说法 `HELDOUT` |
| `expertnet/zh/train.py` | 基础训练、评估、上下文脚本、权重导出 |
| `expertnet/zh/adversary.py` | Attacker / Auditor，长对话生成与失败挖掘 |
| `expertnet/zh/adversarial.py` | 对抗训练主循环、接受/回滚、日志 |
| `expertnet/zh/name_train.py` | 名字记忆专项课程（设名→干扰→追问） |
| `expertnet/zh/scaling.py` | 500/5k/50k 长链缩放实验 |
| `expertnet/zh/scaling_eval.py` | 缩放变体的固定测试评分 |
| `expertnet/zh/long_suite.py` | 固定长对话测试集（禁止训练） |
| `expertnet/zh/chat.py` | 无界面部署核心：模型加载、句库选择、安全掩码 |
| `expertnet/atoms.py` | 通用 AtomNet 与训练/评估函数，中英文共用 |
| `expertnet/sentence.py` | 句向量编码器（MiniLM / BGE / 手写字符胶囊） |
| `expertnet/zh/atoms_zh.safetensors` | 当前权重（部署优先读取） |
| `expertnet/zh/atoms_zh.json` | 句库、beat、模型配置、编码器信息 |
| `expertnet/zh/atoms_zh.pt` | 同权重旧格式，safetensors 失败时兜底 |
| `tools/chat_web.py` | 本地对照网页（规则 vs 句库），`--lang zh` |
| `web/chat.html` | 上述网页的前端 |
| `bot/npc/` | 旧的三位英文 NPC，不参与阿禾部署，但英文 expertnet 训练仍用到其 Reader |

## 3. 架构

```
玩家这句话 ──> SentenceEncoder(text) ──> 512 维句向量
                                         │
状态（37 维）: trust, guild, turn, revealed,
              上一轮 beat one-hot(31),
              是否已知玩家名, mood
                                         │
                         concat ──> 手写 GRU（96 维记忆）
                                         │
                    思考头：10 类上下文标签 ──> 软嵌入
                                         │
                      ┌──────────────────┴─────────────────┐
                句子头：86 类 softmax               beat 头：31 类（辅助）
                                         │
                          规则老师给出 beat（意图/权限）
                                         │
              语法框架 × 词表 ──> 19 万句整句库 ──> 按上下文挑一句
```

- 模型约 **27 万参数**，CPU 训练/推理。
- 状态向量的 31 个槽位直接对应中文 beat，不再使用别名映射。
- `mood` 在 `[-1, 1]`，每轮先乘 0.85 衰减；被骂/被威胁下降，送礼/道歉上升。
- 名字和“刚才的话”用 `{player_name}` / `{last_player}` 模板放进句库，渲染时填入；
  没有名字时会屏蔽名字模板，避免输出“你叫你呀”。

### 3.1 部署混合策略（重要）

`expertnet/zh/chat.py` 不是“模型说什么就用什么”，而是：

- 事实、身份、母亲名字、学习、蚂蚁数量、秘密内容、边界与威胁一律用规则老师的原句；
- 其余 beat（问候、告别、感谢、道歉、闲聊、被骂后的反应、`taunt`、`apology_meta`、
  `memory_recall`、名字、交易、送礼、承诺、meta、fallback）从**整句库**里挑一句；
- 86 类选择器仍然前向一次，用来更新 96 维记忆、思考标签和网页对照，不再直接决定输出；
- 句库挑选会避开最近 8 句，按 `mood` 的符号、玩家这句话里的字、名字/上一句模板加权。

`recent` 是情景记忆：最近 6 轮的 `{player, reply, beat}`。`memory_recall` 的句子是模板
`你刚才说：{last_player}。`，显示时填入上一轮玩家原话；因此记忆能力已经进入学生模型，
不是规则硬编码。

## 4. 数据

### 4.1 语料

- `corpus.SEEDS`：训练用说法，按意图分组。
- `corpus.HELDOUT`：测试用换说法，**绝不能进入训练**。
- `long_suite.SUITE`：12 段长对话（6–14 轮），**绝不能进入训练**。
- `persona.LINES` / `FACTS`：老师会说的句子和事实；事实有 `tier`，tier≥1 需要信任 ≥40。

### 4.2 老师

`teacher_turn(text, state)` 是唯一的标签来源：

1. 先处理玩家名字（`我叫X` / `我叫什么`）；
2. `read()` 做关键词意图分类（boundary / threat / gift / promise / facts / farewell / insult…）；
3. 根据 `mood`、上一轮 beat、信任和是否已揭示选择台词；
4. 返回句子模板和 beat，并更新信任、情绪、`last_beat`、`revealed`。

改老师逻辑后，必须重跑基础训练；否则学生学的是旧分布。

### 4.3 长对话

`adversary.Attacker` 用弧线拼 12–16 轮：

```
insult → farewell          gift → thanks → insult → farewell
insult → greeting          name → name → name
threat → apology → farewell  secret ×3
out_of_world ×2 → personal   正常闲聊 + 短句“嗯/哦/那个……”
```

会加前缀（“阿禾，”“那个……”）、后缀和 30% 的复合告别（“傻瓜，我走了”）。

### 4.4 整句库（生成，不提交）

`grammar.py` 是新的输出空间来源：

- 26 个 beat，每个 beat 有十几条句法框架、若干槽位和人设词表；
- 生成是确定性的：`generate(per_beat=8000, seed=13)` → **191,600** 句；
- 三层变化：框架/槽位（语义）→ 句首语气词 → 句末语气词；问句不接第二小句；
- `{player_name}` / `{last_player}` 保留到运行时填充，不参与生成；
- 结果缓存在 `expertnet/zh/bank_runtime.json`（gitignore），首次约 5 秒，之后秒开。

改 `grammar.py` 后必须提升 `bank.VERSION` 或删除缓存，否则部署仍读旧句库。
整句库只增加“说法覆盖”，不增加事实；事实仍由 `persona.FACTS` 决定。

## 5. 训练

### 5.1 基础训练

```powershell
python -m expertnet.zh.train --sessions 1500 --epochs 40 --steps 10 --hidden 96
python -m expertnet.zh.name_train --chains 1200 --epochs 2 --lr 1e-4 --general 300
```

- 生成 6/10 轮会话，80% 走脚本弧线，20% 随机；
- 写出 `.pt` / `.safetensors` / `.json` 三个文件；
- `--demo` 会跑固定上下文脚本并打印分数。

### 5.2 对抗长对话训练

```powershell
python -m expertnet.zh.adversarial --rounds 3 --sessions 80 --steps 14 --epochs 6 --lr 5e-4
```

每轮：

1. Attacker 生成 80 段 14 轮对话（含难例插入）；
2. Student 带着 96 维记忆逐轮回答；
3. Auditor 用老师重放，记录每一轮的
   `text / trust / category / target / pred / hit / behaviour / confidence`；
4. 失败的会话取 **完整前缀**（保留名字和情绪状态）加入训练，另加一条改写；
5. 从当前最佳模型微调，评估 HELDOUT + 固定长对话；
6. 只有长对话分数提升且 HELDOUT 不下降超过 0.02 才保留，否则回滚。

产物（均被 gitignore）：

- `expertnet/zh/adversarial.jsonl`：逐轮审计记录；
- `expertnet/zh/adversarial_report.json`：基线、每轮、最佳结果。

### 5.3 类别

审计会把回合分成：

- `base`：普通回合；
- `mood`：上一轮被骂/威胁，这一轮问候或告别；
- `name`：设置/询问名字；
- `secret`：秘密与拒绝；
- `long`：第 9 轮及以后。

这些类别的准确率比总准确率更能说明上下文是否真的接上了。

## 6. 评测

| 指标 | 含义 |
|---|---|
| atom | 与老师选中的模板完全一致 |
| behaviour | 选中句子的 beat 与老师一致（同义句算对） |
| thought | 思考标签与 `thought_of` 的目标一致 |
| coverage | 目标句是否在句库内 |
| carried vs reset | 带着记忆 vs 每轮清零 |
| 类别准确率 | mood / name / secret / long |

当前基线：

- 基础训练：atom 0.808，beat 0.816，thought 0.932，name 0.875，deploy 0.708；
- 长对话（64 维版本）：atom 0.815 → 0.889，behaviour 0.889 → 0.898，thought 0.938 → 0.901。
- 整句库（`python -m expertnet.zh.bank_eval --turns 400`）：191,600 句；
  406 轮不重复 339 句（83.5%），近 8 轮重复 5 次；名字/刚才的话命中；
  事实与边界 77/77 原句；危险句 0；延迟 avg 15.7 ms / p95 26.7 ms。

注意：

- atom 会被同义句拉低，判断“会不会接话”更应该看 behaviour；
- 固定长对话只有 88 轮，1 轮的差异就是 1.1 个百分点，别过度解读；
- 评测 seed 不同结果会差 1–2 个百分点，比较时固定 seed；
- 整句库评测里的“不重复率”不是智力指标，只是说法覆盖；不要把它当成 reasoning 提升。

## 7. 网页

本地 Gradio（与创空间同入口）：

```powershell
$env:PORT=7862
python app.py
```

本地对照网页：

```powershell
python tools/chat_web.py --lang zh --port 7861
```

`app.py` 的回复优先用整句库；整句库不可用时退回规则老师。创空间首次启动会下载 BGE，
第一条消息可能慢十几秒；整句库首次构建约 5 秒，之后走 `bank_runtime.json` 缓存。

## 8. 安全边界

- **老师决定“能不能说”，句库只决定“怎么说”**。
- `chat.py` 只在 `BANK_BEATS`（`grammar.SPEC` 的 beat）里用整句库；事实、秘密内容、
  boundary、threat 不在库里，永远用老师原句。
- 句库挑选会屏蔽没有玩家名字时的 `{player_name}` 模板、没有上一句时的 `{last_player}` 模板；
  信任 < 40 且未揭示时的秘密句仍由老师的 `secret_denied` 负责，句库里没有秘密内容。
- 不要让学生模型或整句库绕过 `read()` 的 boundary / threat / secret 判断。
- 新的 secrets 必须在 `persona.FACTS` 里标 `tier`；若也要进句库，必须同步安全扫描。

## 9. 已知问题与下一步

### 9.1 缩放实验（已做）

`expertnet/zh/scaling.py` + `scaling_eval.py` 生成 500 / 5,000 / 50,000 条长链，
从同一基线微调，固定测试集评分。结论：**没有涌现**；atom/behaviour 从 +500 的轻微
正收益转为 50,000 的明显负收益，只有思考标签继续升。详见 `SCALING.md`。
变体权重 `atoms_zh_scale*.pt` 与报告被 gitignore，**不要替换主模型**。

1. ~~`name` 类别约 0.50~~：96 维候选上从 **0.667 → 0.875**；总体 raw atom 0.812 → 0.808，
   部署 0.707 → 0.708。后续若要继续优化，优先加不同间隔/不同干扰类型，而不是继续加量。
2. “回见”是 HELDOUT 说法，学生没见过；这是词表泛化问题，不是上下文问题。
   可以扩 `SEEDS`，但要同步换新的 HELDOUT 说法。
3. ~~句库只有 86 句，老师是天花板~~：已加 19.2 万句生成句库，输出覆盖大幅提升；
   但**语义核心仍来自手写框架和词表**，知识没有增加。要再提升，优先加框架/槽位，
   而不是只把 `per_beat` 调大；语气词和句首词只是表面变化。
4. 长对话测试集仍偏小；建议扩到 30 段、500 轮，并固定随机种子。
5. 思考头目前是单步标签；下一步可以试“先想两步”：先预测话题/情绪，再预测句子，
   或者在推理时把 top-2 思考标签都送进句子头。
6. 目前没有 19 万类 softmax：这个 27 万参数的小模型做不了，也没必要。整句库是
   “计划 + 检索/渲染”，不要误写成“模型学会了 19 万句”。

## 10. 交接规则

- 不要把 `HELDOUT` / `long_suite` 放进训练。
- 换句库或事实后，原子词表会变，必须从头训练并同时导出
  `.pt`、`.safetensors`、`.json` 三个文件；三者必须一致。
- 不要只更新 `.pt`：部署优先读 `.safetensors` + `.json`。
- 改 `grammar.py` 后提升 `bank.VERSION`（或删 `bank_runtime.json`），否则旧缓存还在。
- 提交前跑：`bank_eval --turns 200`（安全/占位符必须为 0）、基础训练 demo（15/15）、
  对抗 `--dry-run`、`python app.py` 手测一次。
- 记录每次改动的 seed、数据量、指标和失败样本；`adversarial_report.json` 是基线，
  `bank_report.json` 是整句库基线。
- 公开仓库：<https://github.com/feikukuai/Npc-Train>；创空间：
  <https://www.modelscope.cn/studios/feihukuai/PyBot-ZH>。
