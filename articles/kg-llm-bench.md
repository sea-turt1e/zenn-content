---
title: "ナレッジグラフをLLMへ応用する時はLLMによってやり方を変えた方が良いらしい"
emoji: "✨"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: [scholar, LLM, NLP, KnowledgeGraph, Bench]
published: false
---

# 論文紹介
[KG-LLM-Bench: A Scalable Benchmark for Evaluating LLM Reasoning on Textualized Knowledge Graphs](https://arxiv.org/abs/2504.07087)
という面白い論文があったのでご紹介です。

# ざっくり要約
- GraphRAGなど、ナレッジグラフのデータをLLMに入力する時には基本的にテキストに変換して入力する必要がある。
- その方法には「エッジをリスト化して渡す」や、「JSON構造で渡す」、「YAMLで渡す」など様々な入力方法がある。
- リスト化して渡すのが一般的ではあったが、実はLLMのモデルによって、リスト化して渡す方が良い時もあれば、YAMLで渡す方がいい時もある。
- そのため汎用的なベストプラクティスはない。

# 主要な貢献




# （おまけ）個人の感想
おそらく、モデルによって良い入力形式が違うのはそれぞれのLLMの学習方法に起因する部分も大きのではないかなと思いました。

# 参考文献
Markowitz, Elan, et al. "KG-LLM-Bench: A Scalable Benchmark for Evaluating LLM Reasoning on Textualized Knowledge Graphs." arXiv preprint arXiv:2504.07087 (2025).