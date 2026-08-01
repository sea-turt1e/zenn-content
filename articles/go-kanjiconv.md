---
title: "【go-kanjiconv】漢字を「かな/ローマ字」に変換するGoライブラリを作成しました"
emoji: "㊙️"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["go", "ライブラリ", "自然言語処理", "kagome", "unidic"]
published: true
published_at: "2026-08-03 07:10"
---

![kanjiconv](https://storage.googleapis.com/zenn-user-upload/b29fba0cf6fa-20241026.png)

# はじめに
以前作成したPythonライブラリ[kanjiconv](https://github.com/sea-turt1e/kanjiconv)のGo版として、『go-kanjiconv』を作成しました。
漢字をひらがな、カタカナ、ローマ字に変換するライブラリ＆CLIツールです。

Go版はPure Goの日本語形態素解析器である[kagome](https://github.com/ikawaha/kagome)をベースに開発しています。辞書はバイナリに埋め込まれるため、外部サービスや辞書のダウンロードは不要で、`go get`（または`go install`）するだけで動きます。
※go-kanjiconvのLicenseはApache License 2.0です。

Python版の記事はこちらです。
[【kanjiconv】固有名詞にも対応した「漢字」→「かな/ローマ字」Python変換ライブラリ](https://zenn.dev/sea_turt1e/articles/e7dc022231a86c)

# GitHubページ
https://github.com/sea-turt1e/go-kanjiconv

# Python版との違い
Python版はSudachiPy/SudachiDictをベースにしています。Go版ではSudachiDictを扱えるメンテナンスされたGoバインディングが無いため、代わりにkagomeを使用し、デフォルトの辞書にUniDicを採用しています。

そのため以下の違いがあります。

- 読みはPython版とおおむね近い結果になりますが、トークンの区切り方が微妙に異なる場合があります（例：複合固有名詞が1トークンではなく2トークンに分かれるなど）。
- より軽量なIPA辞書も`WithDict(ipa.Dict())`（ライブラリ）や`--dict ipa`（CLI）で利用できます。読みの精度を多少犠牲にしてバイナリサイズを抑えたい場合はこちらを使ってください。
- SudachiPyのsplit_mode（A/B/C）に相当するものはなく、代わりにkagome自身の`TokenizeMode`（Normal/Search/Extended）を指定します。

# 環境
```
Go >= 1.24
```
同梱のUniDic辞書はバイナリに埋め込まれ、バイナリサイズがおよそ45MB増加します。バイナリサイズを抑えたい場合はIPA辞書を使用してください（増加量は約11MB）。

# インストール
ライブラリとして使う場合:
```bash
go get github.com/sea-turt1e/go-kanjiconv
```

CLIとして使う場合:
```bash
go install github.com/sea-turt1e/go-kanjiconv/cmd/kanjiconv@latest
```

# 使用方法
## CLIの使用方法
デフォルトではローマ字に変換し、各トークンの読みの間に半角スペースを挿入します。
```bash
kanjiconv "幽☆遊☆白書は、最高の漫画デス。"
# 出力: yuuyuu hakusho ha ,  saikou no manga desu .
```

`-m`/`--mode`で出力形式を選択できます。
```bash
# ひらがなに変換
kanjiconv "幽☆遊☆白書は、最高の漫画デス。" --mode hiragana
# 出力: ゆうゆう はくしょ は 、 さいこう の まんが です 。

# カタカナに変換
kanjiconv "幽☆遊☆白書は、最高の漫画デス。" --mode katakana
# 出力: ユウユウ ハクショ ハ 、 サイコウ ノ マンガ デス 。
```

`-s`/`--separator`で区切り文字を変更できます。
```bash
kanjiconv "今日は良い天気です" --mode hiragana --separator "/"
# 出力: きょう/は/よい/てんき/です

kanjiconv "今日は良い天気です" --mode hiragana --separator ""
# 出力: きょうはよいてんきです
```

その他のオプション:
```bash
# デフォルトのUniDicの代わりに、より軽量なIPA辞書を使用
kanjiconv "東京に行く" --mode hiragana --dict ipa

# カスタム読みのフォールバックを無効化
kanjiconv "激を飛ばす" --mode hiragana --no-custom-readings
```

## ライブラリの使用方法
### インポートとインスタンスの生成
Functional Optionsパターンで区切り文字や辞書を指定できます。
```go
import kanjiconv "github.com/sea-turt1e/go-kanjiconv"

// 基本的な使い方（デフォルトで同梱のUniDic辞書を使用）
conv, err := kanjiconv.New(kanjiconv.WithSeparator("/"))
```

### 漢字をひらがなに変換
```go
text := "今日は良い天気です"
fmt.Println(conv.ToHiragana(text))
// 出力: きょう/は/よい/てんき/です
```

### 漢字をカタカナに変換
```go
fmt.Println(conv.ToKatakana(text))
// 出力: キョウ/ハ/ヨイ/テンキ/デス
```

### 漢字をローマ字に変換
```go
fmt.Println(conv.ToRoman(text))
// 出力: kyou/ha/yoi/tenki/desu
```

### 区切り文字を変更
```go
// 区切り文字を'_'に設定
conv, _ = kanjiconv.New(kanjiconv.WithSeparator("_"))
fmt.Println(conv.ToHiragana(text))
// 出力: きょう_は_よい_てんき_です

// 区切り文字をなしに設定
conv, _ = kanjiconv.New(kanjiconv.WithSeparator(""))
fmt.Println(conv.ToHiragana(text))
// 出力: きょうはよいてんきです
```

### 軽量なIPA辞書＋UniDicフォールバック
バイナリサイズを抑えつつ精度も確保したい場合は、IPA辞書をメインにして、解決できない語だけUniDicにフォールバックさせることもできます。
```go
import (
    "github.com/ikawaha/kagome-dict/ipa"
    "github.com/ikawaha/kagome-dict/uni"
)

conv, err := kanjiconv.New(
    kanjiconv.WithSeparator("/"),
    kanjiconv.WithDict(ipa.Dict()),
    kanjiconv.WithFallbackDict(uni.Dict()),
)
```

# カスタム辞書への対応
Python版と同様に、IPA辞書やUniDicで正しく認識されない漢字の読みをカスタム辞書で指定できます。専門用語や固有名詞、文脈によって読みが変わる漢字などに有用です。
```go
conv, err := kanjiconv.New(
    kanjiconv.WithSeparator("/"),
    kanjiconv.WithCustomReadings(kanjiconv.CustomReadings{
        Single: map[string][]string{
            "激": {"げき"},
            "飛": {"と", "ひ"},
        },
        Compound: map[string]string{
            "激を飛ばす": "げきをとばす",
        },
    }),
)

fmt.Println(conv.ToHiragana("激を飛ばす"))
// 出力: げき/を/とばす
```

- `Single`：個々の漢字とその読み候補のマッピングです。最初の読みがデフォルトとして使用されます。
- `Compound`：複数文字の表現とその読みのマッピングです。トークン化前にテキスト置換として適用され、優先されます。

また、角括弧でテキスト中に読みを直接埋め込むこともできます。その部分はトークナイズをバイパスします。
```go
fmt.Println(conv.ToHiragana("[テスト]です"))
// 出力: てすと/です
```

# まとめ
『go-kanjiconv』を使うことで、Goでも簡単に漢字をひらがな、カタカナ、ローマ字に変換できます。辞書埋め込み型なので外部依存なくシングルバイナリで動くのがGo版の強みです。日本語テキストの前処理や検索用の読み生成など、さまざまな用途でぜひ試してみてください。

# その他
## 作成した経緯
意外とGoだと、漢字->かな変換のライブラリがなかったので作成してみました。

## PR、Issueなど
Goはここ最近使ってないため、Claude Codeにほとんどやってもらいました。  
そのため抜け漏れやバグなどがあるかもしれません。その際はPRやIssueをいただけると非常にありがたいです。

# ライセンス
- go-kanjiconv: Apache License 2.0
- [kagome](https://github.com/ikawaha/kagome): MIT License
- [kagome-dict](https://github.com/ikawaha/kagome-dict)（IPA辞書・UniDic辞書）: MIT License
