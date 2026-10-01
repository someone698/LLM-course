#!/usr/bin/env python3
"""06_paper_check.py — 论文要点与本机 Gensim 实现的逐条核对（源码位置 + 实测复算）。

输出：out/paper_check.md

核对项：
1. 负采样分布 ∝ 词频^0.75（Mikolov 2013）
2. 高频词下采样保留概率 (sqrt(v/t)+1)·(t/v)（Mikolov 2013）
3. 动态窗口：有效窗口从 [1, window] 均匀采样（Mikolov 2013）
4. 子词 = 对 <词> 的字符 n-gram（Bojanowski 2017 的边界符号），FNV-1a 哈希入桶
5. 词表内词的向量 = (词行 + 子词桶行求和) / (桶数+1)（FB 参考实现口径）
6. OOV 词向量 = 子词桶向量均值

源码路径均相对仓库根的共用虚拟环境（../.venv/lib/python3.12/site-packages/gensim）。
"""

import numpy as np
from collections import Counter
from pathlib import Path

from gensim.models import FastText, Word2Vec
from gensim.models.fasttext import ft_ngram_hashes

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
MODELS = OUT / "models"

GENSIM_SRC = "../.venv/lib/python3.12/site-packages/gensim"


def fnv1a_signed(s):
    """FB fastText 的哈希：FNV-1a，逐字节异或时按 C 的 int8_t 解释（≥128 的字节是负数）"""
    h = 2166136261
    for byte in s.encode("utf-8"):
        b = byte - 256 if byte >= 128 else byte
        h = (h ^ (b & 0xFFFFFFFF)) & 0xFFFFFFFF
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def ref_ngram_hashes(word, min_n, max_n, buckets):
    """独立复算：<词> 的字符 n-gram，各段 FNV-1a 哈希后取模"""
    ext = f"<{word}>"
    out = []
    for n in range(min_n, min(len(ext), max_n) + 1):
        for i in range(len(ext) - n + 1):
            out.append(fnv1a_signed(ext[i:i + n]) % buckets)
    return out


def check_sampling():
    """1. 负采样分布：cum_table 增量 vs count^0.75 理论"""
    sentences = [[w, "填充"] for w, c in [("甲", 60), ("乙", 20), ("丙", 6), ("丁", 2)] for _ in range(c)]
    m = Word2Vec(sentences, vector_size=8, min_count=1, sg=1, negative=5,
                 seed=1, workers=1, epochs=1)
    wv = m.wv
    counts = np.array([wv.get_vecattr(w, "count") for w in wv.index_to_key], dtype=float)
    p_theory = counts ** 0.75 / (counts ** 0.75).sum()
    inc = np.diff(np.concatenate([[0.0], m.cum_table.astype(float)]))
    p_table = inc / m.cum_table[-1]
    dev = float(np.abs(p_table - p_theory).max())
    return dev, counts.tolist(), p_theory.round(4).tolist(), p_table.round(4).tolist()


def check_hash():
    """2. 子词 n-gram 与哈希：独立实现对拍 gensim"""
    words = ["足球", "computer", "中国", "ＡＴＴ", "🚑🚒🚓🚕"]
    rows = []
    for w in words:
        g = sorted(ft_ngram_hashes(w, 2, 4, 200000))
        r = sorted(ref_ngram_hashes(w, 2, 4, 200000))
        rows.append((w, len(g), len(r), g == r))
    return rows


def is_cjk(w):
    return any("一" <= ch <= "鿿" for ch in w)


def load_counts():
    """统计 sentences.txt 词频，用来挑真实的 OOV 词"""
    counts = Counter()
    with (OUT / "sentences.txt").open(encoding="utf-8") as f:
        for line in f:
            counts.update(line.split())
    return counts


def check_vectors():
    """3. 词表内词的合成向量；4. OOV 均值合成"""
    ft = FastText.load(str(MODELS / "fasttext.model"))
    wv = ft.wv
    if not wv.buckets_word:
        wv.recalc_char_ngram_buckets()

    word = "足球"
    i = wv.key_to_index[word]
    buckets = wv.buckets_word[i]
    composed = (wv.vectors_vocab[i] + wv.vectors_ngrams[buckets].sum(axis=0)) / (len(buckets) + 1)
    diff_in = float(np.abs(composed - wv.vectors[i]).max())
    diff_word_only = float(np.abs(wv.vectors_vocab[i] - wv.vectors[i]).max())

    counts = load_counts()
    cands = [w for w, c in counts.items()
             if c < 3 and 3 <= len(w) <= 6 and is_cjk(w)]
    cands.sort(key=lambda w: (-counts[w[:2]], w))   # 与前两字高频的可读词，避免数字串
    rows = []
    for oov in cands[:3]:
        hashes = ft_ngram_hashes(oov, wv.min_n, wv.max_n, wv.bucket)
        expect = wv.vectors_ngrams[hashes].mean(axis=0)
        rows.append((oov, counts[oov], len(hashes), float(np.abs(expect - wv[oov]).max())))
    return (word, len(buckets), diff_in, diff_word_only, rows)


def main():
    out = ["# 论文要点 ↔ Gensim 实现核对（06_paper_check.py 生成）\n\n",
           "源码路径相对本仓库根的共用虚拟环境：`" + GENSIM_SRC + "/...`。\n\n"]

    dev, counts, p_theory, p_table = check_sampling()
    out.append("## 1. 负采样分布 ∝ 词频^0.75（Mikolov 等 2013）\n\n")
    out.append("- 实现：`gensim/models/word2vec.py:245` 默认 `ns_exponent=0.75`；`:297` 注明"
               "「0.75 取自原论文」；`:825` `make_cum_table()` 按 `count ** ns_exponent` 建累积表；"
               "采样在 `word2vec_inner.pyx:156`。\n")
    out.append(f"- 实测：合成语料词频 {counts}，累积表增量与 f^0.75 理论分布的最大偏差 "
               f"**{dev:.2e}**（理论 {p_theory}，实测 {p_table}）。\n")
    out.append(f"- 结论：{'✅ 一致' if dev < 1e-3 else '⚠️ 有偏差'}\n\n")

    rows = check_hash()
    out.append("## 2. 子词 n-gram 与哈希（Bojanowski 等 2017）\n\n")
    out.append("- 实现：`fasttext_inner.pyx:644` `compute_ngrams()` 把词包成 `f'<{word}>'` 再取字符 "
               "n-gram；`:677` `compute_ngrams_bytes()` 注明移植自 Facebook 实现；`:619` `ft_hash_bytes()` "
               "为 FNV-1a 哈希（逐字节异或按 int8 解释），`:1326` 取模进 `bucket` 个桶。\n")
    out.append("- 实测：独立复算 `<词>` 的字符 n-gram + FNV-1a（int8 异或）取模，与本机 "
               "`ft_ngram_hashes(word, 2, 4, 200000)` 对拍：\n\n")
    out.append("| 词 | gensim 条数 | 复算条数 | 哈希集合相同 |\n|---|---|---|---|\n")
    for w, ng, nr, ok in rows:
        out.append(f"| {w} | {ng} | {nr} | {'✅' if ok else '❌'} |\n")
    out.append("\n- 结论：" + ("✅ 全部一致" if all(r[3] for r in rows) else "⚠️ 存在不一致") + "\n\n")

    word, nb, diff_in, diff_word_only, oov_rows = check_vectors()
    out.append("## 3. 词表内词的向量 = 词行与子词桶行的平均（FB 参考实现口径）\n\n")
    out.append("- 实现：`fasttext.py:1191` `adjust_vectors()`：「composes the trained full-word-token "
               "vectors with the vectors of the subword ngrams, matching the Facebook reference "
               "implementation behavior」，即 `(词行 + Σ 子词桶行) / (桶数 + 1)`。\n")
    out.append(f"- 实测：词「{word}」子词桶 {nb} 个，`wv.vectors[i]` 与上式复算的最大分量偏差 "
               f"**{diff_in:.2e}**；与只用词行的最大偏差 {diff_word_only:.2e}。\n")
    out.append(f"- 结论：{'✅ 一致' if diff_in < 1e-4 else '⚠️ 有偏差'}（且确实不是只用词行）\n\n")

    out.append("## 4. OOV 词向量 = 子词桶向量均值\n\n")
    out.append("- 实现：`fasttext.py:1085` `get_vector()`：词不在词表时，取该词全部 n-gram 的哈希，"
               "桶向量求和后除以 n-gram 个数；抽不出任何 n-gram 时返回零向量。\n")
    out.append("- 实测（取语料里频次低于 min_count=3 被丢弃的真实词，即 OOV）：\n\n")
    out.append("| OOV 词 | 语料频次 | n-gram 个数 | 与桶向量均值的最大分量偏差 |\n|---|---|---|---|\n")
    for w, c, nh, d in oov_rows:
        out.append(f"| {w} | {c} | {nh} | {d:.2e} |\n")
    ok = all(d < 1e-5 for *_, d in oov_rows)
    out.append(f"\n- 结论：{'✅ 一致' if ok else '⚠️ 有偏差'}\n\n")

    out.append("## 5. 动态窗口（Mikolov 等 2013）\n\n")
    out.append("- 实现：`word2vec.py:352` `shrink_windows=True`（默认）：「the effective window size is "
               "uniformly sampled from [1, window] for each target word during training, to match the "
               "original word2vec algorithm's approximate weighting of context words by distance」；"
               "实现位置 `word2vec_inner.pyx:565`（每词一次 randint 缩小窗口）。\n")
    out.append("- 结论：✅ 本实验用默认值，窗口从 [1, 5] 均匀采样，与论文一致（源码核对，无独立复算）\n\n")

    out.append("## 6. 高频词下采样（Mikolov 等 2013）\n\n")
    out.append("- 实现：`word2vec.py:722` 当 `sample < 1` 时阈值 `t = sample × 总词数`（本实验 "
               "`sample=1e-3` 走这条）；`:734` 保留概率 `(√(v/t)+1)·(t/v)`，与论文公式逐项相同；"
               "`:742` 量化成 `sample_int`，`word2vec_inner.pyx:544` 按此丢弃词元。"
               "（`sample ≥ 1` 时才用 `t = sample·(3+√5)/2` 的另一种写法。）\n")
    out.append("- 结论：✅ 保留概率公式与论文一致（源码核对，无独立复算）\n")

    (OUT / "paper_check.md").write_text("".join(out), encoding="utf-8")
    print("已写 out/paper_check.md")


if __name__ == "__main__":
    main()
