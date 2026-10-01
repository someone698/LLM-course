#!/usr/bin/env python3
"""03_report_gensim.py — Gensim 词向量报告：近邻对比、OOV 词、类别组内/组间余弦相似度。

输入：out/sentences.txt、out/articles.tsv、out/labels.tsv、out/models/{word2vec,fasttext}.model
输出：out/gensim_report.md

设计说明：
- 查询词固定，两模型同一批词对比 top-8 近邻（含余弦值）。
- OOV 演示两类：语料里因低于 min_count=3 被丢弃的真实词（取 3–6 个汉字、前两字为高频词者
  按前两字频次排序选 6 个，确定性选取），以及人工构造的复合词——检验 FastText 能否用子词
  合成向量，Word2Vec 则查不到。
- 组间相似度：每类取「类内高频标题词 top-10」（先剔除全语料 top-100 高频词与标题样板词），
  算组内两两余弦均值与 8×8 组间矩阵。
"""

import numpy as np
from collections import Counter
from pathlib import Path

from gensim.models import FastText, Word2Vec

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
MODELS = OUT / "models"

QUERIES = ["经济", "企业", "足球", "计算机", "教育", "电影", "银行", "总统"]

# 标题样板词（"附图片１张""本报讯"之类）与年份数字，选组内高频词时剔除
BOILER = {"图片", "记者", "本报", "报道", "通讯", "附", "张", "日讯", "新华社", "电",
          "举行", "进行", "指出", "要求", "强调", "日电", "专电"}
CONSTRUCTED_OOV = ["足球运动员", "计算机技术", "经济全球化", "国有企业改革"]


def cosine(v1, v2):
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))


def is_cjk(w):
    return any("一" <= ch <= "鿿" for ch in w)


def top_neighbors(wv, word, k=8):
    return [(w, round(s, 3)) for w, s in wv.most_similar(word, topn=k)]


def load_labels():
    """labels.tsv -> {aid: (类, 划分)}；articles.tsv -> {aid: (标题, 正文)}"""
    labels, articles = {}, {}
    for line in (OUT / "labels.tsv").read_text(encoding="utf-8").splitlines():
        aid, cls, split = line.split("\t")
        labels[aid] = (cls, split)
    for line in (OUT / "articles.tsv").read_text(encoding="utf-8").splitlines():
        aid, title, body = line.split("\t")
        articles[aid] = (title.split(), body.split())
    return labels, articles


def word_counts():
    """统计 sentences.txt 的词频（用于 OOV 选取与高频词剔除）"""
    counts = Counter()
    with (OUT / "sentences.txt").open(encoding="utf-8") as f:
        for line in f:
            counts.update(line.split())
    return counts


def group_words(labels, articles, counts, top_global=100):
    """每类取类内高频标题词 top-10；剔除全语料高频词、单字、数字、样板词"""
    global_top = {w for w, _ in counts.most_common(top_global)}
    by_class = {}
    for aid, (cls, _) in labels.items():
        by_class.setdefault(cls, Counter()).update(articles[aid][0])
    groups = {}
    for cls, cnt in by_class.items():
        words = [w for w, _ in cnt.most_common()
                 if len(w) >= 2 and w not in global_top and w not in BOILER
                 and not w.isdigit() and not w.isascii()]
        groups[cls] = words[:10]
    return groups


def main():
    w2v = Word2Vec.load(str(MODELS / "word2vec.model"))
    ft = FastText.load(str(MODELS / "fasttext.model"))
    labels, articles = load_labels()
    counts = word_counts()

    lines = ["# Gensim 词向量报告（03_report_gensim.py 生成）\n\n",
             f"- Word2Vec 词表 {len(w2v.wv)}，FastText 词表 {len(ft.wv)}（同语料同 min_count=3）\n\n"]

    # 1. 近邻对比
    lines.append("## 一、近邻对比（top-8，括号内为余弦）\n\n")
    lines.append("| 查询词 | 语料频次 | Word2Vec | FastText |\n|---|---|---|---|\n")
    for q in QUERIES:
        if q not in w2v.wv:
            lines.append(f"| {q} | - | 不在词表 | 不在词表 |\n")
            continue
        a = "、".join(f"{w} {s}" for w, s in top_neighbors(w2v.wv, q))
        b = "、".join(f"{w} {s}" for w, s in top_neighbors(ft.wv, q))
        lines.append(f"| {q} | {counts[q]} | {a} | {b} |\n")

    # 2. OOV
    lines.append("\n## 二、词表外的词（OOV）\n\n")
    oov = [w for w, c in counts.items()
           if c < 3 and 3 <= len(w) <= 6 and is_cjk(w)]
    oov.sort(key=lambda w: (-counts[w[:2]], w))     # 前两字高频者优先，近邻才可读
    lines.append("语料里被 min_count=3 丢弃的真实词（3–6 字、前两字高频的 6 个）：\n\n")
    lines.append("| 词 | 频次 | Word2Vec | FastText 近邻 |\n|---|---|---|---|\n")
    for w in oov[:6]:
        ft_nb = "、".join(f"{x} {s}" for x, s in top_neighbors(ft.wv, w, 5))
        lines.append(f"| {w} | {counts[w]} | 不在词表 | {ft_nb} |\n")
    lines.append("\n人工构造的复合词（检验子词合成）：\n\n")
    lines.append("| 词 | Word2Vec | FastText 近邻 |\n|---|---|---|\n")
    for w in CONSTRUCTED_OOV:
        w2v_cell = f"在词表内（频次 {counts[w]}）" if w in w2v.wv else "不在词表"
        ft_nb = "、".join(f"{x} {s}" for x, s in top_neighbors(ft.wv, w, 5))
        lines.append(f"| {w} | {w2v_cell} | {ft_nb} |\n")

    # 3. 组内/组间余弦
    groups = group_words(labels, articles, counts)
    classes = [c for c, _ in sorted(groups.items(), key=lambda kv: kv[0])]
    lines.append("\n## 三、类别组内 / 组间余弦相似度\n\n")
    lines.append("每类取类内高频标题词 top-10（已剔除全语料 top-100 高频词与标题样板词）：\n\n")
    for cls in classes:
        lines.append(f"- **{cls}**：{'、'.join(groups[cls])}\n")

    def matrix(wv):
        m = np.zeros((len(classes), len(classes)))
        for i, ci in enumerate(classes):
            for j, cj in enumerate(classes):
                vals = [cosine(wv[a], wv[b]) for a in groups[ci] for b in groups[cj]
                        if a in wv and b in wv]
                m[i, j] = np.mean(vals) if vals else float("nan")
        return m

    m = matrix(ft.wv)
    lines.append("\nFastText 组间余弦矩阵（对角线为组内均值）：\n\n")
    lines.append("| | " + " | ".join(classes) + " |\n")
    lines.append("|---|" + "---|" * len(classes) + "\n")
    for i, ci in enumerate(classes):
        lines.append(f"| **{ci}** | " + " | ".join(f"{m[i, j]:.3f}" for j in range(len(classes))) + " |\n")

    off = [m[i, j] for i in range(len(classes)) for j in range(len(classes)) if i != j]
    diag = [m[i, i] for i in range(len(classes))]
    lines.append(f"\n组内均值 {np.mean(diag):.3f}，组间均值 {np.mean(off):.3f}。\n")

    m2 = matrix(w2v.wv)
    lines.append("\n两模型组内均值对比：\n\n| 类别 | Word2Vec | FastText |\n|---|---|---|\n")
    for i, ci in enumerate(classes):
        lines.append(f"| {ci} | {m2[i, i]:.3f} | {m[i, i]:.3f} |\n")

    (OUT / "gensim_report.md").write_text("".join(lines), encoding="utf-8")
    print("已写 out/gensim_report.md")


if __name__ == "__main__":
    main()
