#!/usr/bin/env python3
"""
P125 第 5 步：数据稀疏的直接测量——测试集的 n 元上下文有多少在训练集里从未出现。

这是 04_eval.py 里"平均实际阶数"偏低的原因：那些长上下文在训练集里压根没有，
模型只能回退。

做法：把训练集所有 k-gram 哈希进集合，再统计测试集 k-gram 的未命中率。
哈希用 64 位整数，碰撞概率可忽略。

用法：
    python3 scripts/05_sparsity.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPLIT = ROOT / "data" / "split"
OUT = ROOT / "out"
MAX_ORDER = 5


def kgrams(tokens, k):
    for i in range(len(tokens) - k + 1):
        yield hash(tuple(tokens[i:i + k]))


def load_lines(path):
    return [l.split() for l in path.open(encoding="utf-8") if l.strip()]


def main():
    train = load_lines(SPLIT / "train.txt")
    valid = load_lines(SPLIT / "valid.txt")
    test = load_lines(SPLIT / "test.txt")
    n_train_tok = sum(len(s) for s in train)
    print(f"训练集 {len(train):,} 篇 / {n_train_tok:,} 词", file=sys.stderr)

    rows = []
    for k in range(1, MAX_ORDER + 1):
        seen = set()
        total_train = 0
        for sent in train:
            for h in kgrams(sent, k):
                seen.add(h)
                total_train += 1

        row = {"k": k, "types": len(seen), "tokens": total_train}
        for name, data in (("valid", valid), ("test", test)):
            miss = tot = 0
            for sent in data:
                for h in kgrams(sent, k):
                    tot += 1
                    if h not in seen:
                        miss += 1
            row[name] = miss / tot if tot else 0.0
        rows.append(row)
        print(f"  {k}-gram: 训练集 {total_train:,} 个位置 / {len(seen):,} 种"
              f"（相异率 {len(seen)/total_train*100:.1f}%）"
              f"  测试集未见上下文 {row['test']*100:.2f}%", file=sys.stderr)

    lines = [
        "# P125 数据稀疏测量",
        "",
        "「相异率」= 训练集中互不相同的 k-gram 数 / k-gram 总出现位置数。",
        "相异率接近 100% 意味着几乎每个 k-gram 都只出现过一次，模型无法从中学到统计规律。",
        "",
        "「测试集未见上下文」= 测试集里有多少 k-gram 在训练集中从未出现，",
        "这些位置模型只能回退到低阶——正是 04_eval.py 中「平均实际阶数」上不去的原因。",
        "",
        "| 阶数 k | 训练集 k-gram 位置数 | 相异 k-gram 数 | 相异率 | 测试集未见上下文 | 验证集未见上下文 |",
        "|---|---|---|---|---|---|",
    ]
    lines += [f"| {r['k']} | {r['tokens']:,} | {r['types']:,} | "
              f"{r['types']/r['tokens']*100:.2f}% | {r['test']*100:.2f}% | {r['valid']*100:.2f}% |"
              for r in rows]
    (OUT / "sparsity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n写出 {OUT/'sparsity.md'}")


if __name__ == "__main__":
    main()
