#!/usr/bin/env python3
"""02_train_gensim.py — 用 Gensim 训练 Word2Vec 与 FastText（skip-gram）。

输入：out/sentences.txt（01_prepare_corpus.py 生成，每行一句）
输出：out/models/word2vec.model、out/models/fasttext.model（体积大，不进 git）

参数两模型完全一致（sg=1 跳元、100 维、窗口 5、min_count=3、5 轮、seed=42、
workers=1、负采样 5、高频词下采样 1e-3），FastText 另加字符子词 min_n=2、max_n=4、
bucket=200000。workers=1 + PYTHONHASHSEED=0 是为了结果可复现。
"""

import time
from pathlib import Path

from gensim.models import FastText, Word2Vec
from gensim.models.word2vec import LineSentence

ROOT = Path(__file__).resolve().parent.parent
SENTENCES = ROOT / "out" / "sentences.txt"
MODELS = ROOT / "out" / "models"

COMMON = dict(sg=1, vector_size=100, window=5, min_count=3, epochs=5,
              seed=42, workers=1, negative=5, sample=1e-3)


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    sentences = LineSentence(str(SENTENCES))

    t0 = time.time()
    w2v = Word2Vec(sentences, **COMMON)
    w2v.save(str(MODELS / "word2vec.model"))
    print(f"word2vec 词表 {len(w2v.wv)}，用时 {time.time() - t0:.0f}s")

    t0 = time.time()
    ft = FastText(sentences, min_n=2, max_n=4, bucket=200000, **COMMON)
    ft.save(str(MODELS / "fasttext.model"))
    print(f"fasttext 词表 {len(ft.wv)}，用时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
