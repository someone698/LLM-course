#!/usr/bin/env python3
"""
P125 对照实验：不清理会怎样。

动机：README 清洗规则表里"不清会怎样"一列，此前全是机制推断——
data/renmin.tagged.txt 生成之后从来没有真的训练过，五个模型全在清洗版上训。
本脚本把这条对照补上，让那一列变成实测。

对照设计（唯一变量是 token 粒度）：
    clean  语料 data/split/        模型 out/models/
    tagged 语料 data/split_tagged/ 模型 out/models_tagged/
两份语料由 01_clean.py 同一次循环产出、逐行对齐，**文章划分相同、token 位置数相同**，
只有 token 的写法不同（`迈向` vs `迈向/v`）。所以这几个指标是严格可比的：

    相异率 / 测试集未见上下文率 / OOV 率 / 平均实际阶数

⚠️ 困惑度（PPL）不在其列，本脚本仍然算出来，只为说明它为什么不能用：
tagged 模型预测的随机变量是 (词, 词性) 联合，clean 模型预测的是词边缘分布，
而 H(词,词性) ≥ H(词) 恒成立——即 tagged 模型**无论多好都不可能赢**，
这个比较是退化的。要看效果差异，只能看上面那几个。

用法：
    .venv/bin/python scripts/06_uncleaned_control.py
    .venv/bin/python scripts/06_uncleaned_control.py --skip-train --skip-generate
"""
import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

try:
    import kenlm
except ImportError:
    sys.exit("需要 kenlm python 模块：.venv/bin/pip install kenlm")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
CLEAN_SPLIT = ROOT / "data" / "split"
TAGGED_SPLIT = ROOT / "data" / "split_tagged"
TAGGED_MODELS = OUT / "models_tagged"
MAX_ORDER = 5
GEN_ORDER = 5

CORPORA = [
    ("clean", "清洗后", CLEAN_SPLIT, OUT / "models"),
    ("tagged", "不清理", TAGGED_SPLIT, TAGGED_MODELS),
]

PREFIX_WORDS = ["在", "阳光", "明媚", "的", "五月", "，", "我们", "学校", "胜利", "召开", "了"]
DOC_ID = re.compile(r"^(\d{8}-\d{2}-\d{3}-\d{3})/m$")


def load_lines(path: Path):
    return [l.split() for l in path.open(encoding="utf-8") if l.strip()]


def check_aligned(a_lines, b_lines, name):
    """整个对照实验都建立在"两版逐行对齐"上，所以这里真的验一遍：
    把 tagged 的每个 token 去掉 '/词性' 后，应与 clean 逐 token 相同。"""
    assert len(a_lines) == len(b_lines), f"{name}: 行数不一致 {len(a_lines)} vs {len(b_lines)}"
    for i, (a, b) in enumerate(zip(a_lines, b_lines)):
        # 与 01_clean.py 的 split_token 互逆：它在**第一个** '/' 处切，
        # 所以这里也必须切第一个（词本身按构造不含 '/'，标签可能含）。
        stripped = [t.split("/", 1)[0] for t in b]
        assert a == stripped, f"{name} 第 {i} 行未对齐：\n  clean  {a[:8]}\n  tagged {stripped[:8]}"
    return len(a_lines)


def vocab_stats(lines):
    """词表规模 + 碎片化程度。'只出现一次的类型'占比是碎片化的直接刻度。"""
    c = Counter()
    for s in lines:
        c.update(s)
    types = len(c)
    toks = sum(c.values())
    hapax = sum(1 for v in c.values() if v == 1)
    return {
        "types": types,
        "tokens": toks,
        "hapax": hapax,
        "hapax_ratio": hapax / types if types else 0.0,
        "mean_count": toks / types if types else 0.0,
    }


def kgram_stats(train, test, k):
    """相异 k-gram 数与测试集未见上下文率。逐行独立数，不含 <s>/</s>。"""
    seen = set()
    pos = 0
    for sent in train:
        for i in range(len(sent) - k + 1):
            seen.add(tuple(sent[i:i + k]))
            pos += 1
    miss = tot = 0
    for sent in test:
        for i in range(len(sent) - k + 1):
            tot += 1
            if tuple(sent[i:i + k]) not in seen:
                miss += 1
    return {"pos": pos, "types": len(seen), "unseen": miss / tot if tot else 0.0}


def eval_model(model_path: Path, sentences):
    model = kenlm.Model(str(model_path))
    n_tok = n_oov = order_sum = 0
    total = 0.0
    for toks in sentences:
        # bos=True 时 <s> 只作上下文不计入被预测 token；eos=True 时 </s> 计入
        for log10p, order_used, oov in model.full_scores(" ".join(toks), bos=True, eos=True):
            total += log10p
            n_tok += 1
            n_oov += bool(oov)
            order_sum += order_used
    if not n_tok:
        return None
    return {
        "ppl": 10 ** (-total / n_tok),
        "oov": n_oov / n_tok,
        "order": order_sum / n_tok,
        "tokens": n_tok,
    }


def pick(models_dir: Path, order: int):
    for ext in ("bin", "arpa"):
        p = models_dir / f"renmin.{order}gram.{ext}"
        if p.exists():
            return p
    return None


def tagged_prefix(train_lines):
    """前缀每个词取训练集里最高频的词性。
    不清理版的词表里没有裸词，喂 clean 前缀会整句 OOV、状态全坏。"""
    want = set(PREFIX_WORDS)
    tags = {w: Counter() for w in PREFIX_WORDS}
    for sent in train_lines:
        for tok in sent:
            w, _, t = tok.partition("/")
            if w in want:
                tags[w][t] += 1
    out, detail = [], []
    for w in PREFIX_WORDS:
        if not tags[w]:
            out.append(w)
            detail.append((w, None, 0, 0))
            continue
        top, n = tags[w].most_common(1)[0]
        out.append(f"{w}/{top}")
        detail.append((w, top, n, len(tags[w])))
    return out, detail


def train_tagged():
    """调 02_train.sh（和清洗版同一个脚本、同一套 lmplz 参数），只换输入与输出目录。"""
    env = dict(
        os.environ,
        TRAIN=str(TAGGED_SPLIT / "train.txt"),
        OUT_DIR=str(TAGGED_MODELS),
        LOG_DIR=str(OUT / "logs_tagged"),
        RESULT=str(OUT / "train_bench_tagged.tsv"),
    )
    print("=== 训练 tagged 模型（02_train.sh，同一脚本，换 TRAIN/OUT_DIR）===")
    subprocess.run(["bash", str(ROOT / "scripts" / "02_train.sh")], env=env, check=True)


def generate_tagged(prefix):
    print("=== tagged 版续写（03_generate.py，同一脚本，换 --models/--prefix）===")
    subprocess.run([
        sys.executable, str(ROOT / "scripts" / "03_generate.py"),
        "--orders", str(GEN_ORDER), "--gen", "20",
        "--models", str(TAGGED_MODELS),
        "--prefix", " ".join(prefix),
        "--out-name", "generations_tagged",
    ], check=True)


def article_id_stats():
    """文章编号那一行的代价：ID 是行首 token，每个基本只出现一次。"""
    ids = Counter()
    for line in (ROOT / "data" / "renmin.raw.txt").open(encoding="utf-8", errors="replace"):
        parts = line.split()
        if parts:
            m = DOC_ID.match(parts[0])
            if m:
                ids[m.group(1)] += 1
    return ids


def greedy_text(name):
    p = OUT / f"{name}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return d.get(f"{GEN_ORDER}gram", {}).get("greedy", {}).get("text")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true", help="模型已存在时不重训")
    ap.add_argument("--skip-generate", action="store_true", help="跳过续写")
    ap.add_argument("--orders", type=int, nargs="+", default=[2, 3, 4, 5])
    args = ap.parse_args()

    if not TAGGED_SPLIT.exists():
        sys.exit(f"找不到 {TAGGED_SPLIT}，先跑 python3 scripts/01_clean.py")

    if not args.skip_train:
        train_tagged()

    # --- 载入两版语料 + 对齐校验 ---
    data = {}
    for key, label, split_dir, models_dir in CORPORA:
        train = load_lines(split_dir / "train.txt")
        valid = load_lines(split_dir / "valid.txt")
        test = load_lines(split_dir / "test.txt")
        if key == "tagged":
            for name, s in (("train", train), ("valid", valid), ("test", test)):
                n = check_aligned(data["clean"][name], s, name)
                print(f"[对齐校验] {name}: {n:,} 行，逐 token 一致", file=sys.stderr)
        data[key] = {
            "label": label, "train": train, "valid": valid, "test": test,
            "models_dir": models_dir, "vocab": vocab_stats(train),
        }
        print(f"[{key}] train {len(train):,} 篇 / {sum(len(s) for s in train):,} 词  "
              f"词表 {data[key]['vocab']['types']:,}", file=sys.stderr)

    prefix, prefix_detail = tagged_prefix(data["tagged"]["train"])
    if not args.skip_generate:
        generate_tagged(prefix)

    # --- A. 词表碎片化 ---
    cv, tv = data["clean"]["vocab"], data["tagged"]["vocab"]

    # --- B/C. 相异率与未见上下文 ---
    sparse = {k: {o: kgram_stats(data[k]["train"], data[k]["test"], o)
                  for o in range(1, MAX_ORDER + 1)} for k in ("clean", "tagged")}

    # --- D. 模型行为 ---
    evals = {}
    for key in ("clean", "tagged"):
        evals[key] = {}
        for o in args.orders:
            mp = pick(data[key]["models_dir"], o)
            if mp is None:
                print(f"[跳过] {key} {o}-gram 模型不存在", file=sys.stderr)
                continue
            evals[key][o] = {"valid": eval_model(mp, data[key]["valid"]),
                             "test": eval_model(mp, data[key]["test"])}
            t = evals[key][o]["test"]
            print(f"{key:<6} {o}-gram  PPL = {t['ppl']:9.2f}  OOV = {t['oov']*100:6.2f}%  "
                  f"平均实际阶数 = {t['order']:5.3f}", file=sys.stderr)

    ids = article_id_stats()
    n_id_tok = sum(ids.values())
    n_id_typ = len(ids)
    dupes = {k: v for k, v in ids.items() if v > 1}
    with_tag = cv["types"] + n_id_typ
    all_bad = cv["types"] + n_id_typ + (tv["types"] - cv["types"])

    # --- 报告 ---
    L = [
        "# P125 对照实验：清理 vs 不清理",
        "",
        "唯一变量是 token 粒度。两份语料由 `01_clean.py` 同一次循环产出、逐行对齐，",
        "**文章划分相同、token 位置数相同**，只有 token 写法不同（`迈向` vs `迈向/v`）。",
        "",
        "（两版逐 token 对齐已由本脚本校验：tagged 去掉 `/词性` 后与 clean 完全相同。）",
        "",
        "## A. 词表碎片化（词性标签）",
        "",
        "（本节词表口径是**训练集**；`out/clean_report.txt` 里的 55,310 / 62,031 是全量语料口径。）",
        "",
        "| 指标 | 清洗后 | 不清理 | 变化 |",
        "|---|---|---|---|",
        "| 训练集词表规模（类型数） | %s | %s | +%.2f× |" % (
            f"{cv['types']:,}", f"{tv['types']:,}", tv['types'] / cv['types']),
        "| 平均每个类型的出现次数 | %.2f | %.2f | %.2f |" % (
            cv['mean_count'], tv['mean_count'], tv['mean_count'] - cv['mean_count']),
        "| 只出现一次的类型数 | %s | %s | %+.1f%% |" % (
            f"{cv['hapax']:,}", f"{tv['hapax']:,}", (tv['hapax'] / cv['hapax'] - 1) * 100),
        "| 其占词表比例 | %.2f%% | %.2f%% | %+.2f pp |" % (
            cv['hapax_ratio'] * 100, tv['hapax_ratio'] * 100,
            (tv['hapax_ratio'] - cv['hapax_ratio']) * 100),
        "",
        "## B. k-gram 相异率（训练集）",
        "",
        "| 阶数 k | 清洗后 相异数 | 清洗后 相异率 | 不清理 相异数 | 不清理 相异率 |",
        "|---|---|---|---|---|",
    ]
    for k in range(1, MAX_ORDER + 1):
        a, b = sparse["clean"][k], sparse["tagged"][k]
        L.append("| %d | %s | %.2f%% | %s | %.2f%% |" % (
            k, f"{a['types']:,}", a['types'] / a['pos'] * 100,
            f"{b['types']:,}", b['types'] / b['pos'] * 100))

    L += [
        "",
        "## C. 测试集未见上下文率（模型只能回退的比例）",
        "",
        "| 阶数 k | 清洗后 | 不清理 | 差值 |",
        "|---|---|---|---|",
    ]
    for k in range(1, MAX_ORDER + 1):
        a, b = sparse["clean"][k], sparse["tagged"][k]
        L.append("| %d | %.2f%% | %.2f%% | %+.2f pp |" % (
            k, a['unseen'] * 100, b['unseen'] * 100, (b['unseen'] - a['unseen']) * 100))

    if evals["clean"] and evals["tagged"]:
        L += [
            "",
            "## D. 模型行为（测试集）",
            "",
            "| 阶数 | OOV 率 清洗后 | OOV 率 不清理 | 平均实际阶数 清洗后 | 平均实际阶数 不清理 |",
            "|---|---|---|---|---|",
        ]
        for o in args.orders:
            if o not in evals["clean"] or o not in evals["tagged"]:
                continue
            a, b = evals["clean"][o]["test"], evals["tagged"][o]["test"]
            L.append("| %d | %.2f%% | %.2f%% | %.3f | %.3f |" % (
                o, a['oov'] * 100, b['oov'] * 100, a['order'], b['order']))
        L += [
            "",
            "### ⚠️ 困惑度：算出来了，但不能用来对比",
            "",
            "| 阶数 | PPL 清洗后 | PPL 不清理 |",
            "|---|---|---|",
        ]
        for o in args.orders:
            if o not in evals["clean"] or o not in evals["tagged"]:
                continue
            L.append("| %d | %.2f | %.2f |" % (
                o, evals["clean"][o]["test"]["ppl"], evals["tagged"][o]["test"]["ppl"]))
        L += [
            "",
            "tagged 模型预测的是 (词, 词性) 联合分布，clean 模型预测的是词的边缘分布，",
            "两者不是同一个随机变量。而 H(词, 词性) ≥ H(词) 恒成立——",
            "**tagged 模型无论训练得多好，这个数都必然更大**，所以它反映的不是模型质量。",
            "",
            "A–D 是可比的量化代价（幅度都不大），E 是定性代价（输出格式直接错了），",
            "F 是另一条清洗规则。",
        ]

    # --- E. 续写输出对照 ---
    gc, gt = greedy_text("generations"), greedy_text("generations_tagged")
    if gc or gt:
        L += [
            "",
            f"## E. {GEN_ORDER}-gram greedy 续写对照（定性差异在这里，不在上面那些指标里）",
            "",
            "| 版本 | 前缀 |",
            "|---|---|",
            "| 清洗后 | `%s` |" % " ".join(PREFIX_WORDS),
            "| 不清理 | `%s` |" % " ".join(prefix),
            "",
            "```",
            "[清洗后] …" + (gc or "（缺 out/generations.json）"),
            "",
            "[不清理] …" + (gt or "（缺 out/generations_tagged.json）"),
            "```",
            "",
            "不清理版的每个 token 都带 `/词性`——模型的输出格式本身是错的。",
            "",
            "前缀词性取自训练集最高频（不清理版词表里没有裸词，喂 clean 前缀会整句 OOV）：",
            "",
            "| 词 | 取的词性 | 该词性频次 | 该词共有几种词性 |",
            "|---|---|---|---|",
        ]
        for w, top, n, ntags in prefix_detail:
            L.append("| %s | %s | %s | %d |" % (
                w, f"`/{top}`" if top else "—", f"{n:,}", ntags))

    # --- F. 文章编号 ---
    L += [
        "",
        "## F. 文章编号那一行的代价（另一条清洗规则）",
        "",
        f"原始语料行首是 `19980101-01-001-001/m`，既是标注对象也是文章级划分的依据。",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| 编号出现次数 | {n_id_tok:,} |",
        f"| 不同编号数 | {n_id_typ:,} |",
        f"| 出现一次的编号 | {sum(1 for v in ids.values() if v == 1):,}"
        f"（{sum(1 for v in ids.values() if v == 1) / n_id_typ * 100:.2f}%） |",
        f"| 保留编号后词表 | {cv['types']:,} → **{with_tag:,}**（×{with_tag / cv['types']:.3f}） |",
        f"| 对比：保留词性标签后 | {cv['types']:,} → {tv['types']:,}（×{tv['types'] / cv['types']:.3f}） |",
        f"| 两样都保留 | ≈ {all_bad:,}（×{all_bad / cv['types']:.3f}） |",
        "",
        f"**编号造成的词表膨胀（×{with_tag / cv['types']:.3f}）比词性标签（×{tv['types'] / cv['types']:.3f}）更大。**"
        "99.99% 的编号只出现一次，与它相邻的 n-gram 组合也各只出现一次——"
        "这部分计数不可能被统计到任何规律。",
        "",
    ]
    if dupes:
        L += [
            "> ⚠️ 顺带一个语料事实：**`01_clean.py` 里「行首编号唯一」的说法不成立。**",
            f"> {n_id_tok:,} 行只有 {n_id_typ:,} 个不同编号，{len(dupes)} 个编号重复出现：",
        ]
        for k, v in sorted(dupes.items()):
            L.append(f"> `{k}` 出现 {v} 次。")
        L += [
            ">",
            "> 所以「19,484 篇」严格说是 19,484 **行**、19,483 篇。划分按编号哈希，",
            "> 重复编号的两行落在同一桶，不构成跨集泄漏。",
        ]

    report = "\n".join(L) + "\n"
    (OUT / "uncleaned_control.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"写出 {OUT/'uncleaned_control.md'}")


if __name__ == "__main__":
    main()
