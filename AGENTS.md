# 交接手册：阿禾（Npc-Train）

这份文档写给下一个接手训练的 Agent。目标不是“再训一个更大的模型”，而是维护一个可复现、
可评测、可部署的中文整句选择器。先读完再改代码。

## 0. 当前状态（2026-11）

- 角色：**阿禾**，五岁半，村口药铺家的孩子。只认识药铺、药园、老井、橘猫阿团和娘。
- 输出：从 **84 个句子原子**里选一句；不是逐字生成。
- 记忆：一个 **64 维** GRU 状态，跨轮携带。
- 思考：一个 **10 类**上下文标签（`fresh` / `after_insult` / `memory_probe`…），软嵌入送进句子头。
- 句向量：`BAAI/bge-small-zh-v1.5`，512 维，CLS pooling，冻结。
- 规则老师：`expertnet/zh/persona.py`，决定意图、事实权限、情绪、名字记忆。
- 基础训练：7200 回合，HELDOUT 整句命中约 **0.767**，beat 约 0.809，思考标签 0.919。
- 长对话对抗训练：固定长对话整句命中 **0.815 → 0.889**，行为 0.889 → 0.898；
  held-out 基本持平（0.769 → 0.771）。
- 名字专项：raw name atom **0.688 → 0.885**，部署 0.702 → 0.701（见 `name_train.py`）。
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
                         concat ──> 手写 GRU（64 维记忆）
                                         │
                    思考头：10 类上下文标签 ──> 软嵌入
                                         │
                      ┌──────────────────┴─────────────────┐
                句子头：84 类 softmax               beat 头：31 类（辅助）
```

- 模型约 **20 万参数**，CPU 训练/推理。
- 状态向量的 31 个槽位直接对应中文 beat，不再使用别名映射。
- `mood` 在 `[-1, 1]`，每轮先乘 0.85 衰减；被骂/被威胁下降，送礼/道歉上升。
- 名字和“刚才的话”用 `{player_name}` / `{last_player}` 模板放进句库，渲染时填入；
  没有名字时会屏蔽名字模板，避免输出“你叫你呀”。

### 3.1 部署混合策略（重要）

`expertnet/zh/chat.py` 不是“模型说什么就用什么”，而是：

- `RULE_ONLY` 里的 beat 一律用规则老师的原句：事实、身份、母亲名字、学习、蚂蚁数量、
  名字、秘密权限、out_of_world、fallback；
- 其余 beat（问候、告别、感谢、道歉、闲聊、被骂后的反应、`taunt`、`apology_meta`、
  `memory_recall`）用句库模型；
- 模型仍然会前向一次，用来更新 64 维记忆和思考标签，但输出可以被规则覆盖。

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

## 5. 训练

### 5.1 基础训练

```powershell
python -m expertnet.zh.train --sessions 1000 --epochs 30 --demo
```

- 生成 6 轮会话，80% 走脚本弧线，20% 随机；
- 空 batch 之后 `torch.save` 三个文件（.pt / .safetensors / .json）；
- `--demo` 会跑固定上下文脚本，当前应为 **15/15**。

### 5.2 对抗长对话训练

```powershell
python -m expertnet.zh.adversarial --rounds 3 --sessions 80 --steps 14 --epochs 6 --lr 5e-4
```

每轮：

1. Attacker 生成 80 段 14 轮对话（含难例插入）；
2. Student 带着 64 维记忆逐轮回答；
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

- 基础训练：atom 0.767，beat 0.809，thought 0.919，top-3 0.888，context +0.416；
- 长对话：atom 0.815 → 0.889，behaviour 0.889 → 0.898，thought 0.938 → 0.901；
- 名字类原始学生仍约 0.50（部署走规则）。

注意：

- atom 会被同义句拉低，判断“会不会接话”更应该看 behaviour；
- 固定长对话只有 88 轮，1 轮的差异就是 1.1 个百分点，别过度解读；
- 评测 seed 不同结果会差 1–2 个百分点，比较时固定 seed。

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

`app.py` 的回复优先用句库模型；模型未就绪时自动用规则老师。创空间首次启动会下载 BGE，
第一条消息可能慢十几秒。

## 8. 安全边界

- **老师决定“能不能说”，学生只决定“怎么说”**。
- `chat.py` 在推理时会屏蔽两类原子：
  - 没有玩家名字时的 `{player_name}` 模板；
  - 信任 < 40 且未揭示时的 tier≥1 秘密句。
- 不要让学生模型绕过 `read()` 的 boundary / threat / secret 判断。
- 新的 secrets 必须在 `persona.FACTS` 里标 `tier`，并加到 `SECRET_ATOMS` 的掩码逻辑里。

## 9. 已知问题与下一步

### 9.1 缩放实验（已做）

`expertnet/zh/scaling.py` + `scaling_eval.py` 生成 500 / 5,000 / 50,000 条长链，
从同一基线微调，固定测试集评分。结论：**没有涌现**；atom/behaviour 从 +500 的轻微
正收益转为 50,000 的明显负收益，只有思考标签继续升。详见 `SCALING.md`。
变体权重 `atoms_zh_scale*.pt` 与报告被 gitignore，**不要替换主模型**。

1. ~~`name` 类别约 0.50~~：已用 `expertnet/zh/name_train.py` 的“设名→干扰→追问”课程
   提升到 **0.885**；总体 raw atom 下降约 0.012，部署层 0.702 → 0.701。
   后续若要继续优化，优先加不同间隔/不同干扰类型，而不是继续加量。
2. “回见”是 HELDOUT 说法，学生没见过；这是词表泛化问题，不是上下文问题。
   可以扩 `SEEDS`，但要同步换新的 HELDOUT 说法。
3. 句库只有 84 句，老师是天花板；继续对抗只能改善路由，不会增加知识。
4. 长对话测试集仍偏小；建议扩到 30 段、500 轮，并固定随机种子。
5. 思考头目前是单步标签；下一步可以试“先想两步”：先预测话题/情绪，再预测句子，
   或者在推理时把 top-2 思考标签都送进句子头。

## 10. 交接规则

- 不要把 `HELDOUT` / `long_suite` 放进训练。
- 换句库或事实后，原子词表会变，必须从头训练并同时导出
  `.pt`、`.safetensors`、`.json` 三个文件；三者必须一致。
- 不要只更新 `.pt`：部署优先读 `.safetensors` + `.json`。
- 提交前跑：基础训练 demo（15/15）、对抗 `--dry-run`、`python app.py` 手测一次。
- 记录每次改动的 seed、数据量、指标和失败样本；`adversarial_report.json` 是基线。
- 公开仓库：<https://github.com/feikukuai/Npc-Train>；创空间：
  <https://www.modelscope.cn/studios/feihukuai/PyBot-ZH>。
