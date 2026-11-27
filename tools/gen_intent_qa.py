"""给 35 个 beat 批量生成玩家说法并标注回答（用该 beat 的 canonical 句）。

输出 intent_qa.jsonl：{"input", "answer", "beat"}，用于补全社交意图覆盖。
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

from expertnet.zh.exhaustive import DEFAULT_OUT as EX
from expertnet.zh.factorized_train import canonical_text, load_rows

URL = "https://developer.amd.com.cn/radeon/api/v1/chat/completions"
MODEL = "DeepSeek-V4-Flash"
KEY = os.environ["AMD_KEY"]
OUT = "expertnet/zh/intent_qa.jsonl"
PER = 40

PROMPT = """角色：阿禾，五岁半，村口药铺家的孩子。
下面列出若干对话意图。请为每个意图生成 {per} 条玩家会说的话：
每条不超过 10 个字、口语、像真人对小孩说、句式多样、不要重复。
严格输出 JSON，键是意图名，值是字符串数组，不要解释：
{{"greeting": ["你好", "我来啦"], ...}}

意图：
{rows}
"""


def call(prompt, max_tokens=4096, temperature=0.9):
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(URL, data=payload, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {KEY}",
    })
    for _ in range(4):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                data = json.loads(r.read().decode())
            return data["choices"][0]["message"]["content"] or ""
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                time.sleep(60)
                continue
            print("HTTP", exc.code, flush=True)
        except Exception as exc:
            print("ERR", type(exc).__name__, flush=True)
        time.sleep(5)
    return ""


def parse(text):
    text = re.sub(r"```(?:json)?", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return {}


def main():
    _, fixed = load_rows(EX, 1, 31)
    texts = [r["text"] for r in fixed]
    beats = [r.get("beat", "fixed") for r in fixed]
    by_beat = {}
    for text, beat in zip(texts, beats):
        if "{" not in text:
            by_beat.setdefault(beat, []).append(text)
    canon = {beat: canonical_text(pool) for beat, pool in by_beat.items()
             if len(pool) >= 3}
    names = sorted(canon)
    print("意图", len(names), flush=True)

    banks = {}
    if os.path.exists(OUT):
        for line in open(OUT, encoding="utf-8"):
            row = json.loads(line)
            banks.setdefault(row["beat"], []).append(row["input"])

    todo = [b for b in names if len(banks.get(b, [])) < 10]
    with open(OUT, "a", encoding="utf-8") as fh:
        for start in range(0, len(todo), 8):
            chunk = todo[start:start + 8]
            text = call(PROMPT.format(per=PER, rows="\n".join(chunk)))
            got = parse(text)
            for beat in chunk:
                for utter in got.get(beat, []):
                    if not isinstance(utter, str):
                        continue
                    utter = utter.strip("。？！ ")
                    if not (1 < len(utter) <= 10):
                        continue
                    row = {"input": utter + "？", "answer": canon[beat],
                           "beat": beat}
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                print(beat, len(got.get(beat, [])), flush=True)
            time.sleep(3.2)
    total = sum(1 for _ in open(OUT, encoding="utf-8"))
    print("intent_qa 累计", total)


if __name__ == "__main__":
    main()
