"""统计现有 33 万句里 ≤10 字的主谓宾/主系表句子规模。"""
from __future__ import annotations

import collections
import json

PATH = "expertnet/zh/exhaustive.jsonl"
MAX = 10


def main():
    combo = collections.Counter()
    combo_short = collections.Counter()
    fixed = collections.Counter()
    fixed_short = collections.Counter()
    examples = {}
    fixed_examples = collections.defaultdict(list)
    with open(PATH, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            text = row["text"].rstrip("。？！")
            n = len(text)
            if row["pattern"] == "fixed":
                beat = row.get("beat", "")
                fixed[beat] += 1
                if n <= MAX:
                    fixed_short[beat] += 1
                    if len(fixed_examples[beat]) < 2:
                        fixed_examples[beat].append(text)
            else:
                combo[row["pattern"]] += 1
                if n <= MAX:
                    combo_short[row["pattern"]] += 1
                    examples.setdefault(row["pattern"], text)
    print(f"{'组合句式':14s} {'总数':>7s} {'≤10字':>7s}  示例")
    for pattern, total in combo.most_common():
        print(f"{pattern:14s} {total:7d} {combo_short[pattern]:7d}  "
              f"{examples.get(pattern, '')}")
    print(f"\n组合句 ≤10 字合计 {sum(combo_short.values())} / {sum(combo.values())}")
    print(f"\n{'beat':18s} {'固定句':>6s} {'≤10字':>6s}  示例")
    for beat, total in fixed.most_common():
        sample = " / ".join(fixed_examples.get(beat, []))
        print(f"{beat:18s} {total:6d} {fixed_short[beat]:6d}  {sample}")
    print(f"\n固定句 ≤10 字合计 {sum(fixed_short.values())} / {sum(fixed.values())}")


if __name__ == "__main__":
    main()
