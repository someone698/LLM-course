#!/usr/bin/env python3
"""01_prepare_corpus.py — 构建 hw2 的文章级语料与弱标签数据集。

输入：data/199801/199801.txt … 199806.txt
      人民日报 1998 年 1–6 月（B 版标注，UTF-8 / CRLF）。
      每行一句，行首为「日期-版次-版面-序号/词性」；词间空格分隔，词后跟 /词性；
      多词实体用方括号标注（[中国/ns 政府/n]nt）。

输出（out/）：
  sentences.txt   每行一句（已清洗），Gensim LineSentence 的输入
  articles.tsv    文章级：ID \t 标题 \t 正文（正文不含标题行）
  labels.tsv      弱标签：ID \t 类别 \t 划分（标题关键词命中，按优先级取首个类别；
                  划分按类内 ID 排序每 5 篇取 1 篇作测试集，确定性、类别集嵌套时不变）
  corpus_stats.md 统计与抽样（每类前 8 条标题，供人工核对关键词质量）

清洗：去掉 /词性 与实体括号；去掉词性为 w 的标点。

两个实测事实决定了两处设计：
- 多行文章的首行是标题（如「迈向 充满 希望 的 新 世纪 —— 一九九八年 新年 讲话」），
  单行简讯的首行就是全文，没有独立正文——后者不进分类数据集，只进 sentences.txt。
- 分类只用正文、标题只用于打弱标签，避免"标签词直接进特征"的泄漏。
"""

from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "199801"
OUT = ROOT / "out"

MONTHS = [f"1998{m:02d}" for m in range(1, 7)]

# 弱标签关键词（子串匹配标题），按优先级排列：标题同时命中多个类别时取靠前的
LABEL_RULES = [
    ("体育", "足球 篮球 排球 乒乓球 羽毛球 网球 棒球 冰球 手球 曲棍球 球赛 球队 球员 球迷 球场 "
             "赛区 赛场 比赛 联赛 锦标赛 冠军 亚军 季军 运动员 教练 裁判 队员 选手 主教练 俱乐部 "
             "世界杯 奥运会 亚运会 全运会 甲A 围棋 象棋 桥牌 田径 游泳 射击 举重 体操 滑冰 滑雪 "
             "柔道 摔跤 拳击 武术 夺冠 卫冕 进球 出线 小组赛 半决赛 决赛 主场 客场 积分榜"),
    ("卫生", "卫生 医疗 医院 医生 大夫 健康 病人 疾病 医药 药品 防治 疫苗 患者 保健 献血 艾滋病 "
             "传染病 门诊 病房 手术 中药 西药 护士 防疫 疫情 康复 义诊"),
    ("军事", "军事 部队 军人 战士 军区 国防 官兵 士兵 武器 装备 演习 战争 军队 军营 参谋 舰队 "
             "军舰 空军 海军 陆军 导弹 炮兵 坦克 战机 战斗机 军校 复员 转业 拥军 军属"),
    ("文化", "文化 艺术 电影 电视 电视剧 演出 音乐 文艺 图书 出版 文物 展览 戏剧 京剧 博物馆 "
             "作家 小说 诗歌 节目 舞台 画展 美术 摄影 书法 曲艺 杂技 歌舞 乐团 剧院 影片 观众"),
    ("教育", "教育 学校 学生 教师 大学 学院 中学 小学 幼儿园 高考 中考 招生 考试 课程 教学 教材 "
             "培养 人才 学位 毕业 留学 培训 研究生 博士生 硕士 本科 校长 师生 校园 助学 希望工程"),
    ("科技", "科技 技术 科学 计算机 电脑 软件 网络 互联网 信息 卫星 航天 发射 发明 专利 科研 "
             "高新技术 通信 电信 电子 生物 基因 新材料 能源 核电站 院士 研究员 科学家 研究所 "
             "研制 科技成果"),
    ("经济", "经济 企业 市场 金融 银行 财政 税务 税收 贸易 出口 进口 投资 股市 股票 证券 上市公司 "
             "物价 价格 产值 生产 经营 效益 国有资产 资产 贷款 保险 商业 消费 商品 供销 合作社 "
             "人民币 外汇 关税 通货膨胀 亏损 盈利 利润 产权 股份制 私营 个体户 乡镇 工业 农业 农村 "
             "农民 粮食 丰收 春耕 秋收 水利 扶贫 脱贫 国有企业 乡镇企业 经贸 招商 引资"),
    ("政治外交", "外交 访问 会谈 会见 会晤 总统 总理 主席 大使 联合国 议会 条约 双边 友好 外交部 "
                 "发言人 国际 会议 代表 委员 国务院 中共中央 总书记 政协 人大 常委会 首相 外长 "
                 "议长 峰会 声明 谴责 交涉 领土 主权"),
]


def clean_token(raw):
    """'[中央/n' → ('中央', 'n')；'电台/n]nt' → ('电台', 'n')；'发展/v/%' → ('发展', 'v')

    按第一个斜杠分词与词性：语料里有 1334 个双斜杠词元（如 '近年来/l/%'），
    按最后一个斜杠切会把词切成 '近年来/l'，按第一个切才正确。
    """
    tok = raw.lstrip("[")
    if "/" not in tok:
        return tok, ""
    word, pos = tok.split("/", 1)
    if "]" in pos:
        pos = pos.split("]", 1)[0]
    return word, pos.split("/", 1)[0]


def clean_sentence(line):
    """行 → 清洗后的词列表（去掉行首 ID、标点）"""
    toks = []
    for raw in line.split()[1:]:
        word, pos = clean_token(raw)
        if word and pos != "w":
            toks.append(word)
    return toks


def split_of(ids_in_class):
    """类内按 ID 排序，每 5 篇取 1 篇作测试集。确定性；只看类内成员，类别集嵌套时划分不变。"""
    return {aid: ("test" if i % 5 == 0 else "train")
            for i, aid in enumerate(sorted(ids_in_class))}


def main():
    OUT.mkdir(exist_ok=True)

    articles = {}          # aid -> {"title": [...], "body": [...], "month": str}
    n_rows = n_tokens = 0
    month_rows = Counter()

    with (OUT / "sentences.txt").open("w", encoding="utf-8") as fs:
        for month in MONTHS:
            rows_this_month = 0
            with (DATA / f"{month}.txt").open(encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\r\n")
                    if not line.strip():
                        continue
                    aid_full = line.split()[0].split("/")[0]     # 19980101-01-001-001
                    aid = "-".join(aid_full.split("-")[:3])      # 19980101-01-001（文章）
                    toks = clean_sentence(line)
                    if not toks:
                        continue
                    fs.write(" ".join(toks) + "\n")
                    rows_this_month += 1
                    n_tokens += len(toks)
                    art = articles.setdefault(aid, {"title": None, "body": [], "month": month})
                    if art["title"] is None:
                        art["title"] = toks                      # 每篇文章首行作标题
                    else:
                        art["body"].extend(toks)
            month_rows[month] = rows_this_month
            n_rows += rows_this_month

    # 文章表
    with (OUT / "articles.tsv").open("w", encoding="utf-8") as fa:
        for aid in sorted(articles):
            art = articles[aid]
            body = art["body"]
            if not body:
                continue
            fa.write(f"{aid}\t{' '.join(art['title'])}\t{' '.join(body)}\n")

    # 弱标签：标题命中关键词，按优先级取首个类别。
    # 只给「有正文的文章」打标——单行简讯首行即全文、没有独立正文，本就进不了分类数据集。
    labels = {}            # aid -> 类别
    n_conflict = 0
    body_articles = {aid: art for aid, art in articles.items() if art["body"]}
    for aid, art in body_articles.items():
        title = " ".join(art["title"])
        hits = [cls for cls, needles in LABEL_RULES if any(n in title for n in needles.split())]
        if hits:
            labels[aid] = hits[0]
            if len(hits) > 1:
                n_conflict += 1

    by_class = defaultdict(list)
    for aid, cls in labels.items():
        by_class[cls].append(aid)

    with (OUT / "labels.tsv").open("w", encoding="utf-8") as fl:
        for cls, _ in LABEL_RULES:
            sp = split_of(by_class[cls])
            for aid in sorted(by_class[cls]):
                fl.write(f"{aid}\t{cls}\t{sp[aid]}\n")

    # 统计
    with_body = sum(1 for a in articles.values() if a["body"])
    vocab = Counter()
    for art in articles.values():
        vocab.update(art["title"])
        vocab.update(art["body"])
    lines = []
    lines.append("# 语料统计（01_prepare_corpus.py 生成）\n")
    lines.append(f"- 月份文件：{'、'.join(MONTHS)}\n")
    lines.append(f"- 句子（有效行）：{n_rows}，其中按月：" +
                 "、".join(f"{m} {month_rows[m]}" for m in MONTHS) + "\n")
    lines.append(f"- 文章（前 3 段 ID）：{len(articles)}，其中有正文的 {with_body}\n")
    lines.append(f"- 词元总数：{n_tokens}；词表（type）：{len(vocab)}\n")
    lines.append(f"- 弱标签（只统计有正文的文章）：命中 {len(labels)} 篇，未命中 "
                 f"{len(body_articles) - len(labels)} 篇，标题多类冲突（取优先级高者）{n_conflict} 篇\n"
                 f"- 说明：多行文章的首行是标题，正文为其余行；单行简讯首行即全文，不进分类数据集，"
                 f"只进 sentences.txt\n\n")
    lines.append("| 类别 | 文章数 | train | test |\n|---|---|---|---|\n")
    for cls, _ in LABEL_RULES:
        ids = by_class[cls]
        sp = split_of(ids)
        n_test = sum(1 for a in ids if sp[a] == "test")
        lines.append(f"| {cls} | {len(ids)} | {len(ids) - n_test} | {n_test} |\n")
    lines.append("\n## 每类前 8 条标题（人工核对关键词质量）\n")
    for cls, _ in LABEL_RULES:
        lines.append(f"\n### {cls}\n\n")
        for aid in sorted(by_class[cls])[:8]:
            lines.append(f"- {aid} {' '.join(articles[aid]['title'])}\n")
    (OUT / "corpus_stats.md").write_text("".join(lines), encoding="utf-8")

    print(f"文章 {len(articles)}（有正文 {with_body}），句子 {n_rows}，词元 {n_tokens}，词表 {len(vocab)}")
    print(f"弱标签命中 {len(labels)}，未命中 {len(body_articles) - len(labels)}，多类冲突 {n_conflict}")
    for cls, _ in LABEL_RULES:
        print(f"  {cls}: {len(by_class[cls])}")


if __name__ == "__main__":
    main()
