WHU 知识图谱作业：图灵知识图谱构建

## 1. 项目概述

- **实体抽取（Entity Extraction）**
- **关系抽取（Relation Extraction）**（当前以规则/模板方法为主，后续可继续增强）

## 2. 任务范围

本项目面向 Alan Turing（艾伦·图灵）英文语料，抽取结构化知识，为后续构建知识图谱提供三元组数据。

### 实体类型

| 实体类型 | 示例 |
| --- | --- |
| PERSON | Alan Turing, Alonzo Church |
| ORGANIZATION | Princeton University, Bletchley Park |
| LOCATION | London, Wilmslow, England |
| WORK | On Computable Numbers, Computing Machinery and Intelligence |
| CONCEPT | Turing machine, Turing Test |
| DEVICE | Bombe, Automatic Computing Engine, Manchester Mark I |
| EVENT | World War II |
| TIME | 1912, 1936, 1950, 1954 |

### 关系类型

| 关系 | 含义 |
| --- | --- |
| BORN_IN | person -> place |
| BORN_IN_YEAR | person -> year |
| STUDIED_AT | person -> organization |
| STUDIED_UNDER | person -> person |
| PUBLISHED | person -> work |
| PUBLISHED_IN_YEAR | work/concept -> year |
| INTRODUCED | work -> concept |
| WORKED_AT | person -> organization |
| WORKED_FOR | person -> organization |
| DESIGNED | person -> device |
| USED_AGAINST | device -> target |
| PROPOSED | person -> concept/device |
| JOINED | person -> organization |
| CONTRIBUTED_TO | person -> device |
| DESCRIBED_IN | concept -> work |
| DIED_IN | person -> place |
| DIED_IN_YEAR | person -> year |
| LOCATED_IN | place -> place |

## 3. 目录结构

```text
Knowledge_Graph_Homework/
├─ data/
│  └─ raw/
│     └─ turing_corpus.txt
├─ output/
│  ├─ entities.json
│  ├─ relations.json
│  └─ triples.csv
├─ src/
│  └─ extract_kg.py
└─ README.md
```

## 4. 方法说明

### 4.1 实体抽取

实体抽取主要基于以下策略：

1. 人工设计的领域词典
2. 基于正则表达式的别名匹配
3. 基于规则的年份（时间）抽取


### 4.2 关系抽取

关系抽取采用句式模板匹配。例如：

- `Alan Turing was born in X in Y`
- `Alan Turing studied at X`
- `Alan Turing proposed X`
- `Alan Turing died in X, Y, in Z`

脚本会将匹配结果转换为三元组：

```text
(Alan Turing, BORN_IN, London)
(Alan Turing, STUDIED_AT, Princeton University)
(On Computable Numbers, INTRODUCED, Turing machine)
```

## 5. 语料说明

源文本位于：

- [data/raw/turing_corpus.txt](data/raw/turing_corpus.txt),用于抽取字典内容

- [data/raw/turing_corpus.txt](data/raw/alan_turing_wiki.txt)wiki页面
## 6. 运行方式

在项目根目录执行：

```bash
python src/extract_kg.py
```

脚本会生成以下文件：

- [output/entities.json](output/entities.json)
- [output/relations.json](output/relations.json)
- [output/triples.csv](output/triples.csv)

## 7. 当前结果

当前版本运行后可得到约 25 个实体与 22 条关系，包括：

- 实体列表（含类型、提及次数、证据句）；
- 关系列表（含主语、谓词、宾语、证据句）；
- 可用于后续建图的三元组 CSV 文件。
