#!/usr/bin/env python3
"""
P125 第 3 步：从固定前缀出发，用训练好的 n-gram 续写。

前缀：在阳光明媚的五月，我们学校胜利召开了……

对比维度（"不同参数"）：
    阶数 n        2 / 3 / 4 / 5
    解码策略      greedy（每步取 argmax） / sample（按分布采样）
    温度 T        0.5（保守） / 1.0（原分布） / 1.5（发散）

实现要点：用 kenlm 的 State/BaseScore 增量打分接口，每步只在上下文状态上
算下一个词，无需重新拼接整句话调 score()，5.6 万词表也只需毫秒级。
候选词表直接取模型的 unigram 段，保证候选都是"模型认识的词"。

用法：
    python3 scripts/03_generate.py --orders 2 3 4 5 --gen 20
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

try:
    import kenlm
except ImportError:
    sys.exit("需要 kenlm python 模块：uv pip install kenlm")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"

PREFIX = "在阳光明媚的五月，我们学校胜利召开了"
# 语料是分词后的，前缀也得按同一套切分喂进去
PREFIX_TOKENS = ["在", "阳光", "明媚", "的", "五月", "，", "我们", "学校", "胜利", "召开", "了"]
SPECIAL = {"<s>", "</s>", "<unk>"}
LN10 = math.log(10)


def read_vocab_from_arpa(arpa: Path):
    """从 ARPA 的 \\1-grams: 段读出词表（保证候选词模型都认识）。"""
    vocab, in_uni = [], False
    with arpa.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "\\1-grams:":
                in_uni = True
                continue
            if in_uni:
                if not line.strip():
                    break
                parts = line.split("\t")
                if len(parts) >= 2 and parts[1] not in SPECIAL:
                    vocab.append(parts[1])
    return vocab


def pick_model(order: int, models_dir: Path):
    binp = models_dir / f"renmin.{order}gram.bin"
    arpa = models_dir / f"renmin.{order}gram.arpa"
    if binp.exists():
        return binp, arpa
    if arpa.exists():
        return arpa, arpa
    return None, arpa


def next_word(model, state, vocab, mode, temperature, rng):
    """返回 (词, log10概率, 打分过的候选数)。"""
    out = kenlm.State()
    best_w, best_s = None, -1e30
    logits = [] if mode == "sample" else None
    for w in vocab:
        s = model.BaseScore(state, w, out)          # log10 P(w | 状态)
        if s > best_s:
            best_s, best_w = s, w
        if logits is not None:
            logits.append((s * LN10 / temperature, w))
    if mode == "greedy":
        return best_w, best_s, len(vocab)

    m = max(l for l, _ in logits)
    weights = [math.exp(l - m) for l, _ in logits]
    total = sum(weights)
    r, acc = rng.random() * total, 0.0
    for wgt, (_, w) in zip(weights, logits):
        acc += wgt
        if acc >= r:
            return w, None, len(vocab)
    return logits[-1][1], None, len(vocab)


def generate(model, vocab, n_gen, mode, temperature, seed=0, prefix_tokens=None):
    rng = random.Random(seed)
    state = kenlm.State()
    model.BeginSentenceWrite(state)
    out = kenlm.State()
    for w in (prefix_tokens or PREFIX_TOKENS):   # 把前缀走成上下文状态
        model.BaseScore(state, w, out)
        state, out = out, kenlm.State()

    text, steps = [], []
    for _ in range(n_gen):
        w, s, ncand = next_word(model, state, vocab, mode, temperature, rng)
        text.append(w)
        steps.append({"word": w, "log10p": None if s is None else round(s, 4)})
        model.BaseScore(state, w, out)      # 选定后再推进状态
        state, out = out, kenlm.State()
    return text, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", type=int, nargs="+", default=[2, 3, 4, 5])
    ap.add_argument("--gen", type=int, default=20, help="续写多少个词")
    ap.add_argument("--temps", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--models", default=str(OUT / "models"))
    # 不清理版（06 对照实验）要传带词性的前缀——那个模型的词表里根本没有裸词
    ap.add_argument("--prefix", default=" ".join(PREFIX_TOKENS),
                    help="空格分隔的前缀词序列")
    ap.add_argument("--out-name", default="generations",
                    help="产物文件名前缀，写出 out/<name>.md 与 out/<name>.json")
    args = ap.parse_args()

    models_dir = Path(args.models)
    prefix_tokens = args.prefix.split()
    prefix_text = "".join(prefix_tokens)
    results = {}
    lines = [f"# P125 续写结果", "", f"前缀：**{prefix_text}**", "",
             f"（分词后喂入：{' '.join(prefix_tokens)}）", ""]

    for o in args.orders:
        mpath, arpa = pick_model(o, models_dir)
        if mpath is None:
            print(f"[跳过] {o}-gram 模型不存在：{arpa}", file=sys.stderr)
            continue
        print(f"载入 {mpath.name} ...", file=sys.stderr)
        model = kenlm.Model(str(mpath))
        vocab = read_vocab_from_arpa(arpa)
        print(f"  {o}-gram 词表 {len(vocab):,}", file=sys.stderr)
        per_order = {}

        # greedy
        text, steps = generate(model, vocab, args.gen, "greedy", 1.0, args.seed, prefix_tokens)
        per_order["greedy"] = {"text": "".join(text), "steps": steps}
        lines += [f"## {o}-gram · greedy（argmax）", "",
                  f"```\n{prefix_text}{''.join(text)}\n```",
                  "逐步 log10 P：", "",
                  "| 步 | 词 | log10 P |", "|---|---|---|"]
        lines += [f"| {i+1} | {s['word']} | {s['log10p']} |" for i, s in enumerate(steps)]
        lines.append("")

        # sampling at several temperatures
        for T in args.temps:
            text, _ = generate(model, vocab, args.gen, "sample", T, args.seed, prefix_tokens)
            per_order[f"sample_T{T}"] = "".join(text)
            lines += [f"### {o}-gram · sample T={T}", "",
                      f"```\n{prefix_text}{''.join(text)}\n```", ""]

        results[f"{o}gram"] = per_order

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.out_name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / f"{args.out_name}.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写出 {OUT/(args.out_name + '.md')}")


if __name__ == "__main__":
    main()
