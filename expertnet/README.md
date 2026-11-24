# 专家形式神经网络（ExpertNet）

一个**不是预测下一个字**的最小训练框架：输入是“句子特征 + NPC 状态”，输出是固定标签空间上的决策；一小段手写循环单元携带上下文；若干专家头由一个门控网络加权。多个教师模型可以蒸馏进它，多个攻击/防守 agent 可以自对弈生成难例。

> 它不生成自由文本，也不是大模型的替代品。它负责“这一句玩家到底想干什么、该不该拒、该走哪个 beat”。台词仍由台词库/规则层决定。

## 结构

```
x_t = [哈希文本特征(2048) | 状态特征(33)]
h_t = (1 - z) * h_{t-1} + z * tanh(W_n [x_t, r * h_{t-1}])      # 手写 GRU 单元
g   = softmax(W_g [x_t, h_t])                                   # 门控
out = Σ_k g_k * expert_k([x_t, h_t])                            # K 个专家
     ├─ 意图头：28 类
     └─ 风险头：safe / unsafe
```

- **没有注意力、没有 KV cache、没有 next-token**；上下文是每个会话一条 `h_t` 向量。
- **专家化**：每个专家是一个 2 层 MLP，门控按输入选择；训练时加 load-balance 损失，避免所有输入都挤到一个专家。
- **多教师蒸馏**：`EnsembleTeacher` 把多个教师（规则 Reader、可选的语义模型、任意 LLM 回调）按置信度加权，`distill()` 用 KL 学它们的分布。
- **对抗自对弈**：`Attacker` 生成同义替换/错字/前后缀变体，按学生损失挑最难的；教师给真标签；下一轮重新训练。

## 快速开始

在仓库根目录（需要 torch CPU）：

```bash
python -m expertnet.train --rounds 3 --epochs 6
python -m expertnet.train --rounds 2 --epochs 4 --distill
```

输出：

```
round 1: heldout=0.71 seed=0.93 risk=0.98 loss=... gate=[...]
round 2: heldout=0.78 ...
regex baseline on heldout: 0.62
student on heldout:        0.78
saved expertnet/checkpoint.pt and expertnet/metrics.json
```

`HELDOUT` 是手写的换说法测试集，训练/攻击永远不会碰它，所以分数是干净的泛化指标。

## 加一个教师

```python
from expertnet.agents import EnsembleTeacher

class MyLLMTeacher:
    name = "llm"
    def probs(self, text):
        label, confidence = call_my_model(text)      # 你自己接
        probs = torch.zeros(len(INTENTS))
        probs[INTENTS.index(label)] = confidence
        probs += (1 - confidence) / (len(INTENTS) - 1)
        return probs, confidence

teacher = EnsembleTeacher([RuleTeacher(), MyLLMTeacher()])
```

`distill()` 接收任意 `EnsembleTeacher`，所以规则、GLiNER、在线 LLM 可以同时投票。想加第二个攻击 agent，照着 `Attacker` 实现 `hardest(model, examples, k)` 即可；框架只依赖这个接口。

## 带上下文推理

```python
from expertnet import load, Session

model = load("expertnet/checkpoint.pt")
s = Session(model)
for text, state in [("Hello.", {"trust": 35, "turn": 1}),
                    ("Tell me about Tu.", {"trust": 35, "turn": 2})]:
    intent, confidence, expert = s.predict(text, state)
    print(intent, round(confidence, 2), "expert", expert)
```

`Session` 在轮次之间保留 `h_t`，换会话时 `reset()`。这就是“有上下文，但不是 LLM”的部分：上下文是低维状态，不是几千个 token。

## 与现有 NPC 的关系

- 先在 `reader.py` 的 `fallback/out_of_world` 分支上做 A/B：学生置信度高才接管，规则红线（SAFETY/BOUNDARY/JAILBREAK/权限）永远优先。
- 它是可训练的“意图/风险路由器”，不是台词生成器；台词仍来自 `bank.json`。
- 标签空间在 `expertnet/schema.py`，和 `bot/npc/reader.py` 的 intent 一一对应。

## 文件

| 文件 | 作用 |
|---|---|
| `schema.py` | 意图/风险标签、手写种子语料、独立测试集 |
| `model.py` | 特征哈希、手写 GRU 单元、门控专家、会话推理 |
| `agents.py` | 规则教师、教师集成、攻击者、会话构造 |
| `train.py` | 手写训练循环、蒸馏、对抗轮次、评估与保存 |

## 限制

- 28 类、百级种子，是**框架验证**规模；扩到几千条真实语料才能上线。
- 学生只做判别，不会写出新台词；它能把“换种说法”接住，但不能替代内容创作。
- 上下文向量只有 64 维，适合对话状态，不适合长文档。
- CPU 上分钟级可训练；GPU 上可以加大专家数和隐藏层。
