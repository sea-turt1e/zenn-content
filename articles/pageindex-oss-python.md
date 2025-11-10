---
title: "OSS版PageIndexと自作Pythonコードで「ツリー検索」を実装する"
emoji: "🗂️"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: [LLM, RAG, Python, OSS, PageIndex]
published: true
---

この記事では、[PageIndex](https://pageindex.ai/)をOSSとPythonで試してみた内容を書いていきます。

# PageIndexとは
[PageIndex](https://pageindex.ai/)は、章立てになっているPDFやMarkdownなどの文書をJSON形式のツリー構造に変換し、そのJSONをもとに情報の検索を行う手法です。
概念としての詳細は[「ベクトルDB不要」なRAG手法「PageIndex」を解説](https://zenn.dev/knowledgesense/articles/2895f9adc8d802)の記事がわかりやすいと思います。

この記事では、実装にフォーカスした、OSSとして公開されているPageIndexと独自のPythonコードを組み合わせた内容を書いていきます。

:::message
PageIndexではAPI版とOSS版があります。
OSS版では2025/11/10時点で、ツリー構造にするまでのコードは公開されていますが、検索部分は含まれていませんので実装は自分で行う必要があります。
今回の記事ではOSS版でツリー構造を作成し、自作のPythonコードで検索部分を実装しています。
:::

# 実装コード
実装に関しては「ツリー構造の出力方法」と「検索部分の実装」に分けて説明します。

## ツリー構造の出力方法
- [PageIndex OSSリポジトリ](https://github.com/VectifyAI/PageIndex/tree/main?tab=readme-ov-file#-package-usage)にある通りの手順でPDFからツリー構造を出力します。

### Python環境のセットアップ
```bash
# リポジトリをクローン
git clone git@github.com:VectifyAI/PageIndex.git
cd PageIndex

# 仮想環境を作成・有効化（自分の環境に合わせてください）
python3 -m venv .venv
source .venv/bin/activate  # zsh/macOS

# 依存パッケージをインストール
pip3 install --upgrade -r requirements.txt
```

### OpenAI APIキーを環境変数に設定
```bash
export CHATGPT_API_KEY="your_openai_api_key"
```

### サンプルドキュメントの配置
ルートディレクトリに実際に検索したいPDFドキュメントを配置します。
（今回はサンプルとして[Attention Is All You Need](https://arxiv.org/pdf/1706.03762)を使用します。）

### ツリー構造の出力
```bash
python3 run_pageindex.py --pdf_path /path/to/your/pdf_of_attention_is_all_you_need.pdf --model gpt-4.1-mini-2025-04-14
```
:::message
modelオプションは使用する言語モデルに応じて変更してください。
現在はOpenAIのGPTのみ対応している模様です。
ちなみにgpt-5はまだ対応していないなど、モデルのバージョンによってはエラーになる場合があります。
デフォルトではgpt-4o-2024-11-20が使用されます。
:::

:::message alert
自分は最初にgpt-4.1-nanoを指定して実行しましたが、どうやらモデルの性能や文書の長さなどによってはうまくいかないようで、エラーになりました。 gpt-4.1-mini-2025-04-14を指定したところ正常に動作しました。
:::

### 出力結果
実際の出力結果の**抜粋**です。
JSON形式でツリー構造が出力されます。ちゃんと章立ての構成をJSONで表現できていることがわかります。

```json
{
  "doc_name": "Attention_is_all_you_need.pdf",
  "structure": [
    {
      "title": "Preface",
      "start_index": 1,
      "end_index": 1,
      "node_id": "0000",
      "summary": "The partial document presents the introduction and abstract of the paper \"Attention Is All You Need,\" which proposes the Transformer, a novel neural network architecture for sequence transduction tasks. Unlike previous models relying on recurrent or convolutional networks, the Transformer is based entirely on attention mechanisms, eliminating the need for recurrence and convolutions. The paper highlights the model's superior performance on machine translation benchmarks (WMT 2014 English-to-German and English-to-French), achieving state-of-the-art BLEU scores with significantly reduced training time and computational resources. Additionally, the Transformer demonstrates strong generalization to other tasks such as English constituency parsing. The document also includes author contributions and affiliations, emphasizing the collaborative effort behind the development and implementation of the Transformer and its associated codebases."
    },
    {
      "title": "Introduction",
      "start_index": 2,
      "end_index": 2,
      "node_id": "0001",
      "summary": "The partial document introduces the limitations of recurrent neural networks (RNNs), particularly their sequential nature that hinders parallelization and efficiency in sequence modeling tasks like language modeling and machine translation. It highlights the rise of attention mechanisms as a way to model dependencies regardless of sequence distance, though typically used alongside recurrent models. The document then proposes the Transformer architecture, which eliminates recurrence entirely and relies solely on self-attention to capture global dependencies, enabling greater parallelization and improved translation quality. It contrasts the Transformer with prior convolutional and recurrent models, emphasizing its constant-time operation for relating sequence positions and introducing Multi-Head Attention to address resolution issues. The background section reviews related models that reduce sequential computation and discusses the use of self-attention in various NLP tasks. Finally, the document outlines the standard encoder-decoder framework for sequence transduction, setting the stage for a detailed description of the Transformer model architecture."
    },
    {
      "title": "Model Architecture",
      "start_index": 2,
      "end_index": 3,
      "nodes": [
        {
          "title": "Encoder and Decoder Stacks",
          "start_index": 3,
          "end_index": 5,
          "node_id": "0004",
          "summary": "The partial document provides a detailed explanation of the Transformer model architecture, focusing on its encoder and decoder stacks, attention mechanisms, feed-forward networks, and embedding strategies. It describes the encoder and decoder as composed of six identical layers, each with sub-layers involving multi-head self-attention and position-wise feed-forward networks, enhanced by residual connections and layer normalization. The decoder includes an additional multi-head attention sub-layer over the encoder output and employs masking to maintain autoregressive properties. The attention section elaborates on the Scaled Dot-Product Attention mechanism, explaining its computation, the rationale for scaling, and its efficiency compared to additive attention. It further introduces Multi-Head Attention, which projects queries, keys, and values into multiple subspaces to capture diverse information, detailing the mathematical formulation and parameter dimensions. The document also outlines the three applications of multi-head attention within the Transformer: encoder-decoder attention, encoder self-attention, and decoder self-attention with masking. Additionally, it covers the position-wise feed-forward networks applied identically at each position with two linear transformations and ReLU activation. Finally, it discusses the use of learned embeddings for input and output tokens, sharing weights between embedding layers and the pre-softmax linear transformation to generate predicted token probabilities."
        },
        {
          "title": "Attention",
          "start_index": 5,
          "end_index": 6,
          "nodes": [
            {
              "title": "Scaled Dot-Product Attention",
              "start_index": 6,
              "end_index": 4,
              "node_id": "0006",
              "summary": "The partial document discusses key concepts or instructions related to a specific topic, focusing on essential points or steps. It likely outlines procedures, definitions, or explanations aimed at providing clarity and understanding of the subject matter."
            },
            {
              "title": "Multi-Head Attention",
              "start_index": 4,
              "end_index": 4,
              "node_id": "0007",
              "summary": "The partial document explains the concepts of Scaled Dot-Product Attention and Multi-Head Attention. It details how Scaled Dot-Product Attention computes attention weights by taking the dot product of queries and keys, scaling by the square root of the key dimension to prevent large magnitude values, and applying a softmax function to obtain weighted values. The document contrasts this method with additive attention, highlighting the efficiency and practical advantages of scaled dot-product attention. It then introduces Multi-Head Attention, which involves linearly projecting queries, keys, and values multiple times into lower-dimensional spaces and performing parallel attention operations. This approach allows the model to jointly attend to information from different representation subspaces, enhancing performance."
            },
            {
              "summary": "The partial document presents the introduction and abstract of the paper \"Attention Is All You Need,\" which proposes the Transformer, a novel neural network architecture for sequence transduction tasks. Unlike previous models relying on recurrent or convolutional networks, the Transformer is based entirely on attention mechanisms, eliminating the need for recurrence and convolutions. The paper highlights the model's superior performance on machine translation benchmarks (WMT 2014 English-to-German and English-to-French), achieving state-of-the-art BLEU scores with significantly reduced training time and computational resources. Additionally, the Transformer demonstrates strong generalization to other tasks such as English constituency parsing. The document also includes author contributions and affiliations, emphasizing the collaborative effort behind the development and implementation of the Transformer and its associated codebases."
            }
          ]
        }
      ]
    }
  ]
}
```

## 検索部分の実装
ここからはOSS版のGitHubリポジトリに実装は載せられていないため、自分で実装した検索部分のコードを紹介します。
ただし元のOSS版リポジトリの[こちら](https://github.com/VectifyAI/PageIndex/tree/main/tutorials/tree-search)にツリー構造からの検索の考え方は載せられていますので、そちらも参考にしてください。

:::message
この記事中のコードは簡略化したものです。
実際の全体のコードは以下をご覧ください。 
実行方法やコードの詳細説明: [tree_search_example_usage.md](https://github.com/sea-turt1e/PageIndex/blob/main/tutorials/tree-search/tree_search_example_usage.md)
実装コード: [tree_search_example.py](https://github.com/sea-turt1e/PageIndex/blob/main/tutorials/tree-search/tree_search_example.py)
:::


### 実装の流れ
1. ツリー構造をJSONからPythonのクラスにマッピングする
2. LLMに評価させる関数を作成する
3. 幅優先探索しつつLLMの指示に従う
4. クエリを与えて実行する

### 1. ツリー構造をJSONからPythonのクラスにマッピングする
まずは最小限の `TreeNode` クラスと、JSON からツリーを読み込む関数を定義します。パス情報とサマリーを持たせると、後続の表示が分かりやすくなります。

```python
from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Dict, List, Sequence, Tuple

@dataclass
class TreeNode:
    node_id: str
    title: str
    summary: str
    depth: int
    path_titles: Tuple[str, ...]
    children: List["TreeNode"] = field(default_factory=list)

    def pretty_path(self) -> str:
        return " > ".join(self.path_titles)

    def short_summary(self, max_chars: int = 160) -> str:
        if len(self.summary) <= max_chars:
            return self.summary
        return self.summary[: max_chars - 3] + "..."


def load_structure(path: Path) -> Tuple[List[TreeNode], Dict[str, TreeNode]]:
    payload = json.loads(path.read_text())
    roots: List[TreeNode] = []
    lookup: Dict[str, TreeNode] = {}

    def build(node_payload: dict, parent_path: Tuple[str, ...], depth: int) -> TreeNode:
        node = TreeNode(
            node_id=node_payload.get("node_id", ""),
            title=node_payload.get("title", ""),
            summary=node_payload.get("summary", ""),
            depth=depth,
            path_titles=parent_path + (node_payload.get("title", ""),),
        )
        lookup[node.node_id] = node
        for child_payload in node_payload.get("nodes", []):
            child = build(child_payload, node.path_titles, depth + 1)
            node.children.append(child)
        return node

    for root_payload in payload.get("structure", []):
        roots.append(build(root_payload, tuple(), 0))

    return roots, lookup
```

---

### 2. 候補ノードを LLM に評価させる
LLM に投げるプロンプトを作り、JSON 形式で返答してもらう最小例です。ここでは OpenAI のクライアントを前提としています。

```python
import openai

PROMPT_TEMPLATE = """
You are given a query and several candidate nodes from a document tree.
Select nodes that are relevant and tell us which node to expand next.

Query: {query}

Candidates:
{candidates}

Return JSON with keys "thinking", "relevant_nodes", and "expand".
"""

def render_candidates(nodes: Sequence[TreeNode]) -> str:
    lines = []
    for node in nodes:
        lines.append(f"[{node.node_id}] depth={node.depth} title={node.title}")
        lines.append(f"summary: {node.short_summary(110)}")
    return "\n".join(lines)


def ask_llm(query: str, nodes: Sequence[TreeNode], *, model: str, api_key: str) -> dict:
    prompt = PROMPT_TEMPLATE.format(query=query, candidates=render_candidates(nodes))
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)
```

---

### 3. 幅優先で探索しつつ LLM の指示に従う
実際の `tree_search_example.py` にあるロジックを簡略化し、LLM の指示に応じて探索を続ける部分を最小化します。失敗時にはキーワードスコアにフォールバックする単純な例です。

```python
from collections import deque
import re


def keyword_score(query: str, node: TreeNode) -> float:
    terms = re.findall(r"\w+", query.lower())
    haystack = f"{node.title} {node.summary}".lower()
    hits = sum(1 for term in terms if term in haystack)
    return hits / max(1, len(terms))


def tree_search(query: str, roots: Sequence[TreeNode], *, model: str, api_key: str, max_turns: int = 3) -> List[TreeNode]:
    queue: deque[TreeNode] = deque(roots)
    seen: set[str] = set()
    selected: List[TreeNode] = []

    for _ in range(max_turns):
        batch: List[TreeNode] = []
        while queue and len(batch) < 4:
            node = queue.popleft()
            if node.node_id in seen:
                continue
            seen.add(node.node_id)
            batch.append(node)

        if not batch:
            break

        decision = ask_llm(query, batch, model=model, api_key=api_key)
        
        for item in decision.get("relevant_nodes", []):
            node = next((n for n in batch if n.node_id == item.get("node_id")), None)
            if node and node not in selected:
                selected.append(node)

        for node_id in decision.get("expand", []):
            node = next((n for n in batch if n.node_id == node_id), None)
            if node:
                queue.extend(node.children)

    return selected
```

### 4. 実行例
実際にAttention Is All You Needの論文PDFを元に生成したツリー構造を使って検索を実行してみます。

### 実行方法
クエリとして「このドキュメントの主要な結論は何ですか？」を与えた場合の実行例です。
```bash
python tutorials/tree-search/tree_search_example.py "What is the main conclusion in this document?" --no-trace 
```

### 出力
> 1. [0000] 序文 (1-1ページ, 深さ 0)
   パス: 序文
   理由: 結論や要約を含む可能性のある、概要や要旨が含まれているため。
   概要: このドキュメントの一部は、シーケンス変換タスクのための新しいニューラルネットワークアーキテクチャであるTransformerを提案する論文「Attention Is All You Need」の導入と概要を紹介しています。 以前のモデルとは異なり...
> 2. [0022] 結論 (10-10ページ, 深さ 0)
   パス: 結論
   理由: Transformerモデルの結論と最終評価が含まれているため。
   概要: このドキュメントの一部は、英語の構成要素解析におけるTransformerモデルのパフォーマンス評価、特にWSJセクション23データセットについて紹介しています。 Transformerを様々な先行モデルと比較しており...

:::message
実際は英語でのクエリを与えたため、出力も英語になっています。上記は英語での出力結果を別アプリで翻訳した文です。
:::

# まとめ
以上がPageIndexのOSS版を利用し、自作のPythonコードで検索部分を実装した例になります。
PageIndex自体はまだまだ新しい概念/技術であり、OSS版も発展途上な部分が多いですが、ツリー構造を利用した検索手法として面白い方法だと思います。
ある程度章立てになっているなど、使える場面やドメインは限られるかもしれませんが、よかったら試してみてください。
