#!/usr/bin/env python3
"""05_train_supervised.py — 官方 fastText 监督分类：2/4/8 类 × 有/无预训练词向量。

数据：out/articles.tsv（正文）+ out/labels.tsv（弱标签、划分）
类别集嵌套：2 类 = 体育+经济；4 类 = 再加科技+教育；8 类 = 全部八类。
划分取自 labels.tsv 第三列（类内按 ID 排序每 5 篇取 1 篇作测试），
因此类别集嵌套时同一篇文章的划分不变，2/4/8 类的对照组可直接比较。

两组对照同参：dim=100、epoch=10、lr=0.1、wordNgrams=2、minn=0、maxn=0、
loss=softmax、thread=1、seed=42；实验组加 pretrainedVectors=out/models/fasttext.vec。
minn=maxn=0 表示分类器不加字符子词特征——这样两组只差「词向量初始化」一个变量。

输出：out/supervised_report.md（含每类 P/R/F1、混淆对、误分类样例）
"""

import time
from collections import Counter, defaultdict
from pathlib import Path

import fasttext

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
TMP = OUT / "tmp"
VEC = OUT / "models" / "fasttext.vec"

ALL_CLASSES = ["体育", "经济", "科技", "教育", "文化", "卫生", "军事", "政治外交"]
CLASS_SETS = {2: ["体育", "经济"], 4: ["体育", "经济", "科技", "教育"], 8: ALL_CLASSES}
PARAMS = dict(dim=100, epoch=10, lr=0.1, wordNgrams=2, minn=0, maxn=0,
              loss="softmax", thread=1, seed=42, verbose=0)


def load():
    articles = {}
    for line in (OUT / "articles.tsv").read_text(encoding="utf-8").splitlines():
        aid, _title, body = line.split("\t")
        articles[aid] = body
    labels = {}
    for line in (OUT / "labels.tsv").read_text(encoding="utf-8").splitlines():
        aid, cls, split = line.split("\t")
        labels[aid] = (cls, split)
    return articles, labels


def write_data(articles, labels, classes, split, path):
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for aid in sorted(labels):
            cls, sp = labels[aid]
            if sp == split and cls in classes:
                f.write(f"__label__{cls} {articles[aid]}\n")
                n += 1
    return n


def predict_all(model, path):
    """逐行预测。官方封装 _FastText.predict 在 NumPy 2 下会在
    `np.array(probs, copy=False)` 处报 ValueError（fasttext/FastText.py:239），
    这里直接调用底层 pybind 对象 model.f.predict，返回 [(概率, 标签)]。"""
    y_true, y_pred = [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            label = line.split()[0][len("__label__"):]
            preds = model.f.predict(" ".join(line.split()[1:]) + "\n", 1, 0.0, "strict")
            y_pred.append(preds[0][1][len("__label__"):])
            y_true.append(label)
    return y_true, y_pred


def metrics(y_true, y_pred, classes):
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    per = {}
    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[c] = (prec, rec, f1, sum(1 for t in y_true if t == c))
    macro_f1 = sum(v[2] for v in per.values()) / len(classes)
    return acc, macro_f1, per


def majority_baseline(labels, classes, train_and_test):
    """多数类基线：取训练集里最大的类，全部预测成它，在测试集上的准确率"""
    train, test = train_and_test
    maj = Counter(labels[a][0] for a in train if labels[a][0] in classes).most_common(1)[0][0]
    hits = sum(1 for a in test if labels[a][0] == maj)
    return maj, hits / len(test)


def main():
    TMP.mkdir(parents=True, exist_ok=True)
    articles, labels = load()
    lines = ["# 官方 fastText 监督分类报告（05_train_supervised.py 生成）\n\n"]
    lines.append("同参两组：`dim=100, epoch=10, lr=0.1, wordNgrams=2, minn=0, maxn=0, "
                 "loss=softmax, thread=1, seed=42`；实验组把 Gensim FastText 导出的 "
                 "`out/models/fasttext.vec` 传给 `pretrainedVectors` 作词向量初始化。\n\n")

    summary = []
    details = {}
    for n_cls in (2, 4, 8):
        classes = CLASS_SETS[n_cls]
        train_p = TMP / f"supervised_{n_cls}_train.txt"
        test_p = TMP / f"supervised_{n_cls}_test.txt"
        n_tr = write_data(articles, labels, classes, "train", train_p)
        n_te = write_data(articles, labels, classes, "test", test_p)

        for tag, use_vec in (("从零训练", False), ("预训练向量", True)):
            kwargs = dict(PARAMS)
            if use_vec:
                kwargs["pretrainedVectors"] = str(VEC)
            t0 = time.time()
            model = fasttext.train_supervised(input=str(train_p), **kwargs)
            y_true, y_pred = predict_all(model, test_p)
            acc, macro_f1, per = metrics(y_true, y_pred, classes)
            secs = time.time() - t0
            summary.append((n_cls, tag, n_tr, n_te, acc, macro_f1, secs))
            details[(n_cls, tag)] = (per, y_true, y_pred)

        maj, maj_acc = majority_baseline(labels, classes, (
            [a for a in labels if labels[a][1] == "train"],
            [a for a in labels if labels[a][1] == "test"]))
        details[(n_cls, "多数类基线")] = maj_acc
        print(f"{n_cls} 类：训练 {n_tr} / 测试 {n_te}，多数类 {maj} 基线准确率 {maj_acc:.3f}")

    lines.append("## 一、总表\n\n")
    lines.append("| 类别数 | 词向量 | 训练篇数 | 测试篇数 | 准确率 | 宏平均 F1 | 训练用时 s |\n")
    lines.append("|---|---|---|---|---|---|---|\n")
    for n_cls, tag, n_tr, n_te, acc, macro_f1, secs in summary:
        lines.append(f"| {n_cls} | {tag} | {n_tr} | {n_te} | {acc:.3f} | {macro_f1:.3f} | {secs:.1f} |\n")
    for n_cls in (2, 4, 8):
        lines.append(f"| {n_cls} | 多数类基线 | - | - | {details[(n_cls, '多数类基线')]:.3f} | - | - |\n")

    lines.append("\n## 二、8 类的逐类指标\n\n")
    for tag in ("从零训练", "预训练向量"):
        per, y_true, y_pred = details[(8, tag)]
        lines.append(f"**{tag}**\n\n| 类别 | 精确率 | 召回率 | F1 | 测试篇数 |\n|---|---|---|---|---|\n")
        for c in ALL_CLASSES:
            p, r, f1, n = per[c]
            lines.append(f"| {c} | {p:.3f} | {r:.3f} | {f1:.3f} | {n} |\n")
        conf = Counter((t, p) for t, p in zip(y_true, y_pred) if t != p)
        lines.append("\n最高频的 8 个混淆对（真实 → 预测）：\n\n")
        for (t, p), c in conf.most_common(8):
            lines.append(f"- {t} → {p}：{c}\n")
        lines.append("\n")

    lines.append("\n## 三、8 类（预训练向量）误分类样例\n\n")
    per, y_true, y_pred = details[(8, "预训练向量")]
    test_lines = (TMP / "supervised_8_test.txt").read_text(encoding="utf-8").splitlines()
    shown = 0
    for line, t, p in zip(test_lines, y_true, y_pred):
        if t != p and shown < 10:
            text = " ".join(line.split()[1:])
            lines.append(f"- 真实 {t} / 预测 {p}：{text[:60]}…\n")
            shown += 1

    (OUT / "supervised_report.md").write_text("".join(lines), encoding="utf-8")
    print("已写 out/supervised_report.md")


if __name__ == "__main__":
    main()
