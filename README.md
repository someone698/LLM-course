# LLM-course

《大语言模型与应用实践》课程作业合集。每份作业一个目录，目录内的 README 写清楚内容、复现步骤与结果。

| 目录 | 作业 | 内容 |
|---|---|---|
| [`hw1-ngram/`](hw1-ngram/) | hw1（P125 编程作业） | 人民日报语料清洗，KenLM 训练 1–5 阶 n-gram，困惑度、稀疏度与续写实验 |
| [`hw2-gensim-fasttext/`](hw2-gensim-fasttext/) | hw2 | 人民日报语料训练 Gensim Word2Vec / FastText 词向量，官方 fastText 做 2/4/8 类新闻分类（弱标签），含论文↔实现核对 |

后续作业完成后继续加入。

## 复现约定

- Python 依赖装在仓库根的共用虚拟环境 `.venv/` 里，各目录的脚本用 `../.venv/bin/python` 调用；具体命令见各目录 README。
- 语料与模型体积大，不进仓库，下载与重新生成方式见各目录 README。
