"""用外部大模型给每个模式批量生成玩家口语说法，扩充输入闭集。

读 patterns.jsonl（模式表），按 8 个模式一批请求，要求输出 JSON：
{"模式id": ["说法1", ...]}。说法限 12 字内、口语、多样。
输出 patterns_utter.jsonl，供检索评测/路由使用。
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

URL = "https://developer.amd.com.cn/radeon/api/v1/chat/completions"
MODEL = "DeepSeek-V4-Flash"
KEY = os.environ["AMD_KEY"]
IN = "expertnet/zh/patterns.jsonl"
OUT = "expertnet/zh/patterns_utter.jsonl"
BATCH = 8
PER = 15

PROMPT = """角色：阿禾，五岁半，村口药铺家的孩子，只认识药铺、药园、老井、
橘猫阿团和娘。下面是若干「玩家问题模式」，每条包含实体、属性、标准回答。
请为每个模式生成 {per} 条不同的玩家口语说法：每条不超过 12 个字、自然、
像真人对小孩说话、不要重复句式。
严格输出 JSON，键是模式 id，值是说法数组，不要解释：
{{"f001": ["说法1", "说法2"], ...}}

模式：
{rows}
"""


def call(prompt, max_tokens=4096, temperature=0.8):
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
                print("429 sleep 60", flush=True)
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
    rows = [json.loads(line) for line in open(IN, encoding="utf-8")]
    done = {}
    if os.path.exists(OUT):
        for line in open(OUT, encoding="utf-8"):
            row = json.loads(line)
            done[row["id"]] = row["utterances"]
    todo = [r for r in rows if r["id"] not in done]
    print(f"模式 {len(rows)}，已完成 {len(done)}，本次 {len(todo)}", flush=True)

    with open(OUT, "a", encoding="utf-8") as fh:
        for start in range(0, len(todo), BATCH):
            chunk = todo[start:start + BATCH]
            lines = []
            for row in chunk:
                lines.append(f"{row['id']}｜实体：{row['entity']}"
                             f"｜属性：{row['attribute']}"
                             f"｜标准回答：{row['answer']}")
            text = call(PROMPT.format(per=PER, rows="\n".join(lines)))
            got = parse(text)
            for row in chunk:
                utters = [u.strip() for u in got.get(row["id"], [])
                          if isinstance(u, str) and 1 < len(u.strip()) <= 16]
                out = dict(row, utterances=utters)
                fh.write(json.dumps(out, ensure_ascii=False) + "\n")
                fh.flush()
                print(row["id"], len(utters), flush=True)
            time.sleep(3.2)


if __name__ == "__main__":
    main()
