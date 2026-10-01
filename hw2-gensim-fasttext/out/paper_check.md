# 论文要点 ↔ Gensim 实现核对（06_paper_check.py 生成）

源码路径相对本仓库根的共用虚拟环境：`../.venv/lib/python3.12/site-packages/gensim/...`。

## 1. 负采样分布 ∝ 词频^0.75（Mikolov 等 2013）

- 实现：`gensim/models/word2vec.py:245` 默认 `ns_exponent=0.75`；`:297` 注明「0.75 取自原论文」；`:825` `make_cum_table()` 按 `count ** ns_exponent` 建累积表；采样在 `word2vec_inner.pyx:156`。
- 实测：合成语料词频 [88.0, 60.0, 20.0, 6.0, 2.0]，累积表增量与 f^0.75 理论分布的最大偏差 **2.38e-10**（理论 [0.4402, 0.3303, 0.1449, 0.0587, 0.0258]，实测 [0.4402, 0.3303, 0.1449, 0.0587, 0.0258]）。
- 结论：✅ 一致

## 2. 子词 n-gram 与哈希（Bojanowski 等 2017）

- 实现：`fasttext_inner.pyx:644` `compute_ngrams()` 把词包成 `f'<{word}>'` 再取字符 n-gram；`:677` `compute_ngrams_bytes()` 注明移植自 Facebook 实现；`:619` `ft_hash_bytes()` 为 FNV-1a 哈希（逐字节异或按 int8 解释），`:1326` 取模进 `bucket` 个桶。
- 实测：独立复算 `<词>` 的字符 n-gram + FNV-1a（int8 异或）取模，与本机 `ft_ngram_hashes(word, 2, 4, 200000)` 对拍：

| 词 | gensim 条数 | 复算条数 | 哈希集合相同 |
|---|---|---|---|
| 足球 | 6 | 6 | ✅ |
| computer | 24 | 24 | ✅ |
| 中国 | 6 | 6 | ✅ |
| ＡＴＴ | 9 | 9 | ✅ |
| 🚑🚒🚓🚕 | 12 | 12 | ✅ |

- 结论：✅ 全部一致

## 3. 词表内词的向量 = 词行与子词桶行的平均（FB 参考实现口径）

- 实现：`fasttext.py:1191` `adjust_vectors()`：「composes the trained full-word-token vectors with the vectors of the subword ngrams, matching the Facebook reference implementation behavior」，即 `(词行 + Σ 子词桶行) / (桶数 + 1)`。
- 实测：词「足球」子词桶 6 个，`wv.vectors[i]` 与上式复算的最大分量偏差 **1.19e-07**；与只用词行的最大偏差 1.62e+00。
- 结论：✅ 一致（且确实不是只用词行）

## 4. OOV 词向量 = 子词桶向量均值

- 实现：`fasttext.py:1085` `get_vector()`：词不在词表时，取该词全部 n-gram 的哈希，桶向量求和后除以 n-gram 个数；抽不出任何 n-gram 时返回零向量。
- 实测（取语料里频次低于 min_count=3 被丢弃的真实词，即 OOV）：

| OOV 词 | 语料频次 | n-gram 个数 | 与桶向量均值的最大分量偏差 |
|---|---|---|---|
| 中国区 | 2 | 9 | 0.00e+00 |
| 中国娃 | 2 | 9 | 0.00e+00 |
| 中国情 | 1 | 9 | 0.00e+00 |

- 结论：✅ 一致

## 5. 动态窗口（Mikolov 等 2013）

- 实现：`word2vec.py:352` `shrink_windows=True`（默认）：「the effective window size is uniformly sampled from [1, window] for each target word during training, to match the original word2vec algorithm's approximate weighting of context words by distance」；实现位置 `word2vec_inner.pyx:565`（每词一次 randint 缩小窗口）。
- 结论：✅ 本实验用默认值，窗口从 [1, 5] 均匀采样，与论文一致（源码核对，无独立复算）

## 6. 高频词下采样（Mikolov 等 2013）

- 实现：`word2vec.py:722` 当 `sample < 1` 时阈值 `t = sample × 总词数`（本实验 `sample=1e-3` 走这条）；`:734` 保留概率 `(√(v/t)+1)·(t/v)`，与论文公式逐项相同；`:742` 量化成 `sample_int`，`word2vec_inner.pyx:544` 按此丢弃词元。（`sample ≥ 1` 时才用 `t = sample·(3+√5)/2` 的另一种写法。）
- 结论：✅ 保留概率公式与论文一致（源码核对，无独立复算）
