# Zenn CLI

* [📘 How to use](https://zenn.dev/zenn/articles/zenn-cli-guide)

## 記事作成コマンド

```bash
npx zenn new:article --slug 記事のスラッグ --title タイトル --type idea --emoji ✨
```

## .movを.gifに変更するコマンド
```bash
# パレット画像生成
ffmpeg -i input.mov -vf "palettegen" -y palette.png
# GIF 画像を出力
# 24の値はフレームレート。調整することも可能。
ffmpeg -i input.mov -i palette.png -r 24 -y output.gif
```