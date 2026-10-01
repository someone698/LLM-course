# hw2：Gensim 表示学习 + FastText 文本分类

两部分：① 用 Gensim 在同一语料上训练 Word2Vec 与 FastText 词向量，比较近邻、词表外的词（OOV）与类别组间相似度；② 把 FastText 词向量导出为 `.vec`，用官方 `fasttext.train_supervised` 做 2/4/8 类新闻分类，对照组不给预训练向量。另有 `06_paper_check.py` 把两篇论文的要点与本机 Gensim 源码逐条核对。

## 语料

- 来源：公开仓库 [`chenhui-bupt/PeopleDaily1998`](https://github.com/chenhui-bupt/PeopleDaily1998) 的 `199801.zip`，内含 **199801–199806 六个月**，解压 67 MB（sha256 `17474bbf2b36…`）
- 与 hw1 的关系：同一批报纸的**两个标注版本**（hw1 用 A 版，本次用 B 版）；随包的 `shengming.doc` 三方声明只免费公开了 1 月，2–6 月来源无据，本次按课程练习使用
- 格式：每行一句，行首为 `日期-版次-版面-序号/词性`，词后跟 `/词性`，多词实体用方括号标注
- 构建后：18,647 篇文章、123,812 句、618 万词元、词表 14.2 万（清洗后）

```bash
mkdir -p data && curl -sL -o data/199801.zip https://github.com/chenhui-bupt/PeopleDaily1998/raw/master/199801.zip
unzip -q data/199801.zip -d data/    # 得到 data/199801/199801.txt … 199806.txt
```

## 流程

| 脚本 | 做什么 | 产物 |
|---|---|---|
| `01_prepare_corpus.py` | 清洗；按文章合并（前 3 段 ID）；标题关键词打弱标签 | `out/sentences.txt`、`articles.tsv`、`labels.tsv`、`corpus_stats.md` |
| `02_train_gensim.py` | 训练 Word2Vec 与 FastText（同参） | `out/models/*.model` |
| `03_report_gensim.py` | 近邻对比、OOV 词、类别组内/组间余弦 | `out/gensim_report.md` |
| `04_export_vec.py` | 导出词向量为官方 `.vec` 格式 | `out/models/fasttext.vec` |
| `05_train_supervised.py` | 官方 fastText 分类：2/4/8 类 × 有/无预训练向量 | `out/supervised_report.md` |
| `06_paper_check.py` | 论文要点 ↔ Gensim 实现的逐条核对 | `out/paper_check.md` |

命令都在本目录下执行，虚拟环境建在仓库根、各次作业共用：

```bash
../.venv/bin/python scripts/01_prepare_corpus.py
PYTHONHASHSEED=0 ../.venv/bin/python scripts/02_train_gensim.py
../.venv/bin/python scripts/03_report_gensim.py
../.venv/bin/python scripts/04_export_vec.py
../.venv/bin/python scripts/05_train_supervised.py
../.venv/bin/python scripts/06_paper_check.py
```

## 训练配置

两模型同参：skip-gram（`sg=1`）、100 维、窗口 5、`min_count=3`、5 轮、`seed=42`、`workers=1`、负采样 5、高频词下采样 1e-3。`workers=1` 加 `PYTHONHASHSEED=0` 是为了结果可复现。

FastText 另设字符子词 `min_n=2, max_n=4`、`bucket=200000`（桶矩阵约 137 MB，211,000 行 × 100 维）。子词取 2–4（论文取 3–6）：中文词多为 1–2 字，2–4 能覆盖整词，也不浪费在跨词片段上。

分类器（官方 fastText）两组同参：`dim=100, epoch=10, lr=0.1, wordNgrams=2, minn=0, maxn=0, loss=softmax, thread=1, seed=42`。`minn=maxn=0` 表示分类器不加字符子词特征，两组只差「词向量初始化」一个变量。

## 结果

- 词向量报告：[`out/gensim_report.md`](out/gensim_report.md)
- 分类报告：[`out/supervised_report.md`](out/supervised_report.md)
- 论文核对：[`out/paper_check.md`](out/paper_check.md)
- 语料统计：[`out/corpus_stats.md`](out/corpus_stats.md)

### 词向量

近邻（括号内为余弦，完整 top-8 见报告）：

| 查询词 | 语料频次 | Word2Vec | FastText |
|---|---|---|---|
| 经济 | 16584 | 国民经济 0.787、经济基础 0.713、非国有经济 0.700 | 非经济 0.836、经济体 0.835、经济域 0.822 |
| 企业 | 15366 | 国有 0.836、中小企业 0.782、破产 0.752 | 国营企业 0.894、非国有企业 0.891、大中企业 0.878 |
| 足球 | 615 | 篮球 0.800、排球 0.777、网球 0.752 | 棒球 0.857、篮球 0.852、排球 0.837 |
| 计算机 | 702 | 软件 0.823、电脑 0.814、数字式 0.789 | 计算机网 0.931、计算机业 0.897、计算机房 0.866 |
| 总统 | 3554 | 克林顿 0.839、叶利钦 0.810、纳扎尔巴耶夫 0.787 | 代总统 0.882、克林顿 0.846、总统令 0.836 |

两列近邻的取材不同：Word2Vec 取上下文同现的词，FastText 取共享字面片段的词（足球→踢球/足球队/足协杯，计算机→计算机网/计算机业）。

OOV（词表外）：

| 词 | 频次 | Word2Vec | FastText |
|---|---|---|---|
| 中国杯 | 2 | 不在词表 | 乒协杯 0.918、超霸杯 0.904、汤姆斯杯 0.903 |
| 中国牌 | 1 | 不在词表 | 健智牌 0.866、棋牌 0.850、中国货 0.832 |
| 足球运动员 | 0 | 不在词表 | 运动员 0.910、教练员 0.824、乒乓球队 0.807 |
| 计算机技术 | 0 | 不在词表 | 计算机网 0.935、计算机业 0.923、计算机 0.910 |
| 经济全球化 | 0 | 不在词表 | 全球化 0.916、经济体 0.870、经济域 0.870 |

前两行是语料里频次低于 `min_count=3` 的真实词，后三行是语料中不存在的构造词，FastText 用子词桶合成向量，近邻都落在语义相近的词上。

类别组内 / 组间余弦（每类取类内高频标题词 top-10）：FastText 组内均值 **0.570**，组间均值 **0.307**；8 个类的对角线值都高于所有非对角线值。

### 分类

| 类别数 | 从零训练 准确率 / 宏 F1 | 预训练向量 准确率 / 宏 F1 |
|---|---|---|
| 2 | 0.944 / 0.924 | 0.987 / 0.984 |
| 4 | 0.675 / 0.395 | 0.882 / 0.854 |
| 8 | 0.597 / 0.320 | 0.779 / 0.703 |

多数类基线（经济）准确率 0.271。类别数越多，从零训练掉得越快：8 类时科技、教育、卫生、军事 4 类的 F1 为 0.000，这些类的测试篇目基本都判成了经济、文化或政治外交；接上预训练向量后 8 类宏 F1 从 0.320 升到 0.703，逐类指标见报告。

## 口径与已知限制

- 弱标签来自标题关键词（8 类，按优先级去重），分类输入只用正文，标题不进特征，避免「标签词直接进特征」的泄漏。单行简讯没有独立标题/正文之分，不进分类数据集。
- 类别不均衡是报纸版面天然如此（体育 737 篇，经济 2020 篇），评价看宏平均 F1 与多数类基线，不看单一准确率。
- `.vec` 只装词向量，装不了子词桶：官方 fastText 的 `pretrainedVectors` 按词查表初始化，子词合成能力不会随 `.vec` 转移。
