---
title: "ナレッジグラフのテキスト化、モデルによって変えるべきらしい"
emoji: "✨"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: [LLM, knowledgegraph, NLP, benchmark, Graph]
published: true
published_at: "2026-02-12 07:10"
---

# 論文紹介
[KG-LLM-Bench: A Scalable Benchmark for Evaluating LLM Reasoning on Textualized Knowledge Graphs](https://arxiv.org/abs/2504.07087)
（NAACL 2025 KnowledgeNLP Workshop）という面白い論文があったのでご紹介です。

# ざっくり要約
- GraphRAGなど、ナレッジグラフ（KG）のデータをLLMに入力する時には基本的に**テキストに変換（テキスト化）** して入力する必要がある
- その方法には「エッジをリストにして渡す」「JSON構造で渡す」「YAMLで渡す」「RDF Turtleで渡す」「JSON-LDで渡す」など様々な入力方法がある
- **テキスト化の方法を変えるだけで、ベンチマーク全体の精度が最大17.5%も変わる**
- リストにして渡すのが一般的ではあったが、実はLLMのモデルによって、リストにして渡す方が良い時もあれば、JSONやYAMLで渡す方がいい時もある
- そのため**汎用的なベストプラクティスはなく、モデルとタスクに応じて最適な形式を選ぶ必要がある**

# そもそもナレッジグラフのテキスト化とは？

ナレッジグラフ（KG）は`[主語, 関係, 目的語]` のトリプル（三つ組）でデータを表現するグラフ構造のデータベースです。

例えば、映画『インセプション』に関する知識は以下のように表せます：
```
[Inception, director, Christopher Nolan]
[Inception, genre, Science Fiction]
```

LLMはテキストを入力として受け取るため、このグラフ構造を何らかのテキスト形式に変換する必要があります。この論文では以下の **5つのテキスト化戦略** を比較しています：

### 1. List of Edges（エッジのリスト）
最もシンプルで一般的な形式。各行に`[主語, 関係, 目的語]`を列挙します。
```
[France, borders, Germany]
[France, borders, Spain]
[Germany, capital, Berlin]
```

### 2. Structured JSON（構造化JSON）
主語ごとに関係をグループ化したJSON形式です。
```json
{
  "France": {
    "borders": ["Germany", "Spain"],
    "capital": ["Paris"]
  }
}
```

### 3. Structured YAML（構造化YAML）
JSONと同様の構造をYAMLで表現します。
```yaml
France:
  borders:
    - Germany
    - Spain
  capital:
    - Paris
```

### 4. RDF Turtle
セマンティックWebの標準規格。名前空間やURIを使った厳密な記述です。
```turtle
@prefix ex: <http://example.org/countries#> .
ex:France ex:borders ex:Germany, ex:Spain ;
          ex:capital ex:Paris .
```

### 5. JSON-LD
JSON-LD（Linked Data）はJSON形式でリンクトデータを表現するフォーマットです。
```json
{
  "@context": { "ex": "http://example.org/countries#" },
  "@graph": [
    { "@id": "ex:France", "ex:borders": [{"@id": "ex:Germany"}] }
  ]
}
```

# ベンチマークの5つのタスク

テキスト化の良し悪しを測るために、論文では以下の **5つのタスク** をベンチマークとして設計しています。それぞれ、KG推論の異なる側面を評価します。

| タスク名 | 何を評価するか | 難易度 |
|---|---|---|
| **Triple Retrieval** | 特定のトリプルがグラフに存在するかを判定（Yes/No） | 易 |
| **Shortest Path** | 2つのエンティティ間の最短パスを特定 | 難 |
| **Agg by Relation** | 特定のエンティティの関係数を集計（例: フランスが国境を接する国の数は？） | 中 |
| **Agg Neighbor Property** | 2ホップ先の関係を含む集計（例: フランスに隣接する国で、公用語にドイツ語を持つ国はいくつ？） | 中 |
| **Highest Degree Node** | グラフ全体で最も多くのエッジを持つノードを特定 | 難 |

各タスクについて100問ずつ、WikiDataSetsのCountriesナレッジグラフ（200エッジのサブグラフ）から生成しています。

# 実験設定

## 使用モデル（7つ）
- **Claude 3.5 Sonnet**
- **GPT-4o-Mini**
- **Gemini 1.5 Flash**
- **Llama 3.3-70B**
- **Llama 3.2-1B**
- **Amazon Nova Pro**
- **Amazon Nova Lite**

## 偽名化（Pseudonymization）
モデルが事前学習で覚えた知識に頼らず、提供されたKGの情報だけで回答しているか検証するために、エンティティ名を架空の名前に置き換える実験も実施しています。

# 主要な貢献・発見

## 1. テキスト化形式のグローバルな傾向

全体平均では **Structured JSON（平均0.42）が最も高い精度** で、YAML、List of Edgesが続きます。RDF Turtle（0.35）やJSON-LD（0.34）は精度が低い傾向です。

ただしタスクによって傾向が異なります：
- **集計タスク** → Structured JSONやYAMLが強い（関連するエッジがグループ化されるため）
- **Highest Degree（グローバル集計）** → List of Edgesが有利（最多ノードがリスト中に最も多く出現するため）

## 2. モデルごとに最適な形式が異なる

| モデル | 最適なテキスト化形式 |
|---|---|
| Claude 3.5 Sonnet | RDF Turtle |
| Gemini 1.5 Flash | List of Edges |
| GPT-4o-Mini | List of Edges |
| Llama 3.2-1B | List of Edges |
| Llama 3.3-70B | Structured JSON |
| Nova Lite | Structured JSON |
| Nova Pro | JSON-LD |

**同じタスクでも、モデルによって最適な形式がバラバラ** であることがわかります。これは「とりあえずList of Edgesで渡せばOK」という従来の前提を覆す重要な発見です。

## 3. 特筆すべきモデル性能

- **Nova Pro** がShortest Pathタスクで突出（RDF Turtleで47%、他のモデルは最高でも17%程度）
- **Claude 3.5 Sonnet** がHighest Degreeタスクで突出（RDF Turtleで61.5%、次点のNova Proは16.2%）
- 総合的にはClaude 3.5 SonnetとNova Proがトップ

## 4. 偽名化の影響はほぼゼロ

エンティティ名を架空名に置き換えても、全体の精度差はわずか **0.2%** でした。つまり、200エッジのサブグラフに対する質問では、モデルは事前知識にほぼ頼っていないことが示されています。

## 5. トークン効率の大きな差

| テキスト化形式 | 平均入力トークン数 |
|---|---|
| List of Edges | 2,645 |
| Structured YAML | 2,903 |
| Structured JSON | 4,505 |
| RDF Turtle | 8,171 |
| JSON-LD | 13,503 |

JSON-LDはList of Edgesの約5倍のトークンを消費します。RDF TurtleやJSON-LDはURIや名前空間の記述が増えるためです。コスト面ではList of EdgesやYAMLが有利です。

## 6. 集計方向による性能差

エッジの **出方向の集計は得意** だが、**入方向の集計は苦手** という傾向がありました。これはどのテキスト化形式でも出方向のエッジが隣接して記述されるのに対し、入方向のエッジはテキスト中に散在するためです。

## 7. 集計数が増えると急激に精度低下

AggByRelationタスクでは、集計対象が1エッジなら正答率80%以上ですが、4を超えると急激に低下し、約10%まで落ちます。LLMの数え上げ能力にはまだ大きな改善の余地があることを示しています。

# （おまけ）個人の感想
おそらく、モデルによって良い入力形式が違うのはそれぞれのLLMの学習データや学習方法に起因する部分も大きいのではないかなと思いました。例えばRDF Turtleがよく効くモデルはセマンティックWeb関連のデータを多く学習している可能性がありますし、JSONが効くモデルはコードやAPI関連のデータに強いのかもしれません。

GraphRAGなどでナレッジグラフを活用する際に、**「どのテキスト形式で渡すか」を意識するだけで精度が大きく変わる** というのは実務上も非常に重要な知見だと感じました。特にモデルを切り替える際には、テキスト化の方法も一緒に見直す価値がありそうです。

# 参考文献
Markowitz, Elan, et al. "KG-LLM-Bench: A Scalable Benchmark for Evaluating LLM Reasoning on Textualized Knowledge Graphs." arXiv preprint arXiv:2504.07087 (2025).