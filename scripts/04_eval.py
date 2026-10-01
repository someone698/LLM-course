#!/usr/bin/env python3
"""
P125 第 4 步：在验证集/测试集上算困惑度、OOV 率，以及"平均实际阶数"。

    PPL = 10^(-Σ log10 P(w_i | h) / N)

kenlm 的 full_scores 每项是 (log10概率, 实际用到的阶数, 是否OOV)。第二个字段
是本脚本的关键：它告诉你预测这个词时，模型真正用上了多长的上下文。若 5-gram
在测试集上平均只用到 2.x 阶，就说明大部分 5 元上下文在训练集里根本没出现过、
只能回退——这是数据稀疏最直接的证据，也解释了为什么 n 不能无限增大。

用法：
    python3 scripts/04_eval.py --orders 2 3 4 5
"""
import argparse
import sys
from pathlib import Path

try:
    import kenlm
except ImportError:
    sys.exit("需要 kenlm python 模块：uv pip install kenlm")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"


def evaluate(model_path: Path, data: Path):
    model = kenlm.Model(str(model_path))
    n_tok = n_oov = order_sum = 0
    total = 0.0
    for line in data.open(encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            continue
        # bos=True 时 <s> 只作为上下文，不计入被预测的 token；eos=True 时 </s> 计入
        for log10p, order_used, oov in model.full_scores(line, bos=True, eos=True):
            total += log10p
            n_tok += 1
            n_oov += bool(oov)
            order_sum += order_used
    if not n_tok:
        return float("inf"), 0, 0.0, 0.0
    return (10 ** (-total / n_tok), n_tok, n_oov / n_tok, order_sum / n_tok)


def pick(models_dir: Path, order: int):
    b = models_dir / f"renmin.{order}gram.bin"
    if b.exists():
        return b
    a = models_dir / f"renmin.{order}gram.arpa"
    return a if a.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", type=int, nargs="+", default=[2, 3, 4, 5])
    ap.add_argument("--models", default=str(OUT / "models"))
    ap.add_argument("--data", default=str(ROOT / "data" / "split"))
    args = ap.parse_args()

    models_dir, data_dir = Path(args.models), Path(args.data)
    rows = []
    for o in args.orders:
        mp = pick(models_dir, o)
        if mp is None:
            print(f"[跳过] {o}-gram 模型不存在", file=sys.stderr)
            continue
        for name in ("valid", "test"):
            d = data_dir / f"{name}.txt"
            if not d.exists():
                continue
            ppl, n, oov, avg_order = evaluate(mp, d)
            rows.append((o, name, ppl, oov, avg_order, n))
            print(f"{o}-gram  {name:<5}  PPL = {ppl:9.2f}   OOV = {oov*100:6.2f}%   "
                  f"平均实际阶数 = {avg_order:5.3f}   ({n:,} 词)")

    if rows:
        lines = [
            "# P125 困惑度（人民日报 1998-01，文章级划分）",
            "",
            "「平均实际阶数」= 预测每个词时模型真正用上的上下文长度。",
            "例如 5-gram 若只有 2.3，说明约大多数情况下完整的 5 元上下文没出现过，",
            "只能回退到低阶——这就是数据稀疏在指标上的直接体现。",
            "",
            "| 阶数 | 数据集 | PPL | OOV 率 | 平均实际阶数 | 词数 |",
            "|---|---|---|---|---|---|",
        ]
        lines += [f"| {o} | {nm} | {p:.2f} | {v*100:.2f}% | {a:.3f} | {c:,} |"
                  for o, nm, p, v, a, c in rows]
        (OUT / "perplexity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n写出 {OUT/'perplexity.md'}")


if __name__ == "__main__":
    main()
