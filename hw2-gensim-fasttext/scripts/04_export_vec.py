#!/usr/bin/env python3
"""04_export_vec.py — 把 Gensim FastText 的词向量导出为官方 fastText 的 .vec 文本格式。

输入：out/models/fasttext.model
输出：out/models/fasttext.vec（首行「词表大小 维度」，其后每行「词 各维分量」）

说明：.vec 只装「词」向量，装不了子词桶。官方 fastText 的 pretrainedVectors 按词查表
初始化，子词合成能力不会随 .vec 转移——这一点写进了 supervised_report.md 与 README。
"""

from pathlib import Path

from gensim.models import FastText

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "out" / "models"


def main():
    ft = FastText.load(str(MODELS / "fasttext.model"))
    dst = MODELS / "fasttext.vec"
    ft.wv.save_word2vec_format(str(dst), binary=False)
    n, dim = ft.wv.vectors.shape
    print(f"已写 {dst}：{n} 词 × {dim} 维，{dst.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
