#!/usr/bin/env python3
"""
P125 第 1 步：清洗人民日报切分标注语料。

原始格式（每行一篇完整文章，不是一句）：
    19980101-01-001-001/m  迈向/v  充满/v  希望/n  的/u ...
    └─ 文章编号/词性        └─ 词/词性 ...
    19,484 行 = 19,484 篇文章，行首编号唯一。

要清掉的东西：
    1. 行首文章编号  19980101-01-001-001/m
    2. 每个词的词性标签  /v /n /u ...（含语素族 Ng/Vg/Tg…，如 摄/Vg）
    3. 嵌套专名标注的括号：开括号粘在词左、闭括号粘在标签右，成对 9,238 组
           [中央/n  人民/n  广播/vn  电台/n]nt
           ^ 拆掉                          ^ 拆掉

产出：
    data/renmin.clean.txt    词序列，一行一句（KenLM 训练输入）
    data/renmin.tagged.txt   保留标签的对照版（用来量化"不清理会怎样"）
    out/clean_report.txt     统计报告

用法：python3 scripts/01_clean.py
"""
import re
import sys
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "renmin.raw.txt"
CLEAN = ROOT / "data" / "renmin.clean.txt"
TAGGED = ROOT / "data" / "renmin.tagged.txt"
REPORT = ROOT / "out" / "clean_report.txt"

# 文档编号形如 19980101-01-001-001/m
DOC_ID = re.compile(r"^(\d{8}-\d{2}-\d{3}-\d{3})/m$")


def split_token(tok: str):
    """把 '迈向/v' 拆成 ('迈向', 'v')；处理 [中央/n、电台/n]nt、摄/Vg 这类标注。

    返回 (词, 标签或 None)。
    """
    # 只在第一个 '/' 处切
    if "/" not in tok:
        return tok, None
    word, _, tag = tok.partition("/")
    # 嵌套专名标注，开闭两头都要拆：
    #   [中央/n  人民/n  广播/vn  电台/n]nt
    #   ^ 开括号粘在词左               ^ 闭括号粘在标签右
    word = word.lstrip("[")
    if "]" in tag:
        tag = tag.split("]")[0]
    if not word:
        return "", None
    return word, tag


def main():
    if not RAW.exists():
        sys.exit(f"找不到 {RAW}，先下载语料")

    clean_lines, tagged_lines = [], []
    tag_counter = Counter()
    residue = Counter()
    n_tok_raw = n_tok_clean = 0
    vocab_clean, vocab_tagged = set(), set()
    article_ids = set()
    word_tags = defaultdict(set)   # 词 -> 出现过的词性集合（兼类词统计）

    doc_of_line = []               # 每篇的行号 -> 文章编号，用于划分数据集
    for lineno, line in enumerate(RAW.open(encoding="utf-8", errors="replace"), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        words, tagged = [], []
        cur_id = f"line{lineno}"

        for i, tok in enumerate(parts):
            m = DOC_ID.match(tok)
            if i == 0 and m:
                # 文章 ID：既是清洗对象，也用来做"文章级"数据集划分
                cur_id = m.group(1)        # 19980101-01-001-001
                article_ids.add(cur_id)
                continue
            if i == 0 and tok.startswith("1998") and "/" in tok:
                # 编号格式略有出入的行首也丢掉，但记一笔
                residue["非常规文档编号"] += 1
                continue

            word, tag = split_token(tok)
            if tag is None:
                residue["无标签token"] += 1
            else:
                tag_counter[tag] += 1
                if any(c.isupper() for c in tag):
                    residue["语素族标签（Ng/Vg/Tg…）"] += 1
            if "[" in tok:
                residue["嵌套标注·开括号"] += 1
            if "]" in tok:
                residue["嵌套标注·闭括号"] += 1
            if not word:
                continue
            words.append(word)
            tagged.append(f"{word}/{tag}" if tag else word)
            n_tok_raw += 1
            vocab_clean.add(word)
            if tag:
                word_tags[word].add(tag)

        if not words:
            continue
        clean_lines.append(" ".join(words))
        tagged_lines.append(" ".join(tagged))
        doc_of_line.append(cur_id)
        n_tok_clean += len(words)
        vocab_tagged.update(tagged)

    CLEAN.write_text("\n".join(clean_lines) + "\n", encoding="utf-8")
    TAGGED.write_text("\n".join(tagged_lines) + "\n", encoding="utf-8")

    # --- 文章级划分 train/valid/test ---
    # 按文章编号哈希分桶（可复现），保证同一篇文章不会跨集泄漏。
    # 本语料一行即一篇，但按 ID 分桶对"一篇文章多行"的语料同样成立。
    SPLIT = ROOT / "data" / "split"
    SPLIT.mkdir(parents=True, exist_ok=True)
    buckets = {"train": [], "valid": [], "test": []}
    for art, text in zip(doc_of_line, clean_lines):
        h = int(hashlib.md5(art.encode()).hexdigest()[:8], 16) % 1000
        buckets["valid" if h < 10 else "test" if h < 20 else "train"].append(text)
    for name, lines in buckets.items():
        (SPLIT / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    n_doc = len(clean_lines)
    lens = [len(l.split()) for l in clean_lines]
    ambiguous = {w: ts for w, ts in word_tags.items() if len(ts) > 1}
    extra_types = sum(len(ts) - 1 for ts in ambiguous.values())
    report = [
        "=== P125 语料清洗报告 ===",
        f"源文件            : {RAW.name}",
        f"文章数（=行数）    : {n_doc:,}",
        f"token 总数        : {n_tok_clean:,}",
        f"篇均词数          : {sum(lens)/n_doc:.2f}（最长 {max(lens)}，最短 {min(lens)}）",
        "",
        "--- 词表规模：清理 vs 不清理 ---",
        f"清洗后词表（纯词）    : {len(vocab_clean):,}",
        f"不清理词表（词/词性） : {len(vocab_tagged):,}",
        f"膨胀倍数             : {len(vocab_tagged)/len(vocab_clean):.3f}x",
        "",
        "--- 膨胀从哪来：兼类词（同一个词带多种词性）---",
        f"兼类词数量           : {len(ambiguous):,} / {len(word_tags):,} ({len(ambiguous)/len(word_tags)*100:.1f}%)",
        f"它们贡献的额外类型   : {extra_types:,}（= 全部膨胀量；文章编号未计入任一侧词表）",
        "  最重的兼类词：",
    ]
    for w, ts in sorted(ambiguous.items(), key=lambda kv: -len(kv[1]))[:8]:
        report.append(f"    {w:<6} {len(ts)} 种  {'/'.join(sorted(ts))}")
    report += [
        "",
        "--- 词性标签分布（共 %d 种） ---" % len(tag_counter),
    ]
    for tag, c in tag_counter.most_common(40):
        report.append(f"  {tag:<6} {c:>8,}")
    report += ["", "--- 标注残渣 ---"]
    for k, v in residue.most_common():
        report.append(f"  {k:<16} {v:>8,}")
    report += [
        "",
        "--- 输出 ---",
        f"  {CLEAN.relative_to(ROOT)}   全量（词序列，一行一篇）",
        f"  {TAGGED.relative_to(ROOT)}  对照用（保留标签）",
        "  data/split/train.txt       训练集  %6d 篇  %8d 词" % (len(buckets["train"]), sum(len(l.split()) for l in buckets["train"])),
        "  data/split/valid.txt       验证集  %6d 篇  %8d 词" % (len(buckets["valid"]), sum(len(l.split()) for l in buckets["valid"])),
        "  data/split/test.txt        测试集  %6d 篇  %8d 词" % (len(buckets["test"]), sum(len(l.split()) for l in buckets["test"])),
        f"  sha256(clean) = {hashlib.sha256(CLEAN.read_bytes()).hexdigest()[:16]}",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
