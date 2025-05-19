---
title: "【Go】S3にある画像をTextractでOCR"
emoji: "🗒️"
type: "tech"
topics:
  - "aws"
  - "go"
  - "s3"
  - "ocr"
  - "textract"
published: true
published_at: "2025-03-19 17:54"
---

## 概要
- S3に保存してある画像をAmazon Textractを利用してOCRしてみました。
- AWS SDK for Go v2を使用しています。
:::message
Textractは現在英語にしか対応していないのでご注意ください。
:::

## コード全体
コードの全体はこちらのGitHubにあります。
https://github.com/sea-turt1e/aws-sdk-go-v2-example/blob/main/textract/detectDocumentTextWithS3Object

以下は実際のコードです。
```go
import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/textract"
	"github.com/aws/aws-sdk-go-v2/service/textract/types"
	"github.com/joho/godotenv"
)

func init() {
	// Load Environment Variables
	if err := godotenv.Load("../../.env"); err != nil {
		log.Fatal("Error loading .env file")
	}
}

func detectDocumentTextWithS3Object() {
	// Load the SDK's configuration from the environment
	sdkConfig, err := config.LoadDefaultConfig(context.Background())
	if err != nil {
		log.Fatalf("unable to load SDK config, %v", err)
	}

	// Create a Texttract client
	textractClient := textract.NewFromConfig(sdkConfig)

	// Get Document Text from S3 Object
	detectDocumentTextDetectionOutput, err := textractClient.DetectDocumentText(context.Background(), &textract.DetectDocumentTextInput{
		Document: &types.Document{
			S3Object: &types.S3Object{
				Bucket: aws.String(os.Getenv("BUCKET_NAME")),
				Name:   aws.String(os.Getenv("OBJECT_NAME")),
			},
		},
	})
	if err != nil {
		log.Fatalf("unable to detect document text, %v", err)
	}
	detectedText := getTextFromTextractOutput(detectDocumentTextDetectionOutput)
	fmt.Println(detectedText)
}

func getTextFromTextractOutput(output *textract.DetectDocumentTextOutput) string {
	text := ""
	for _, block := range output.Blocks {
		if block.BlockType == types.BlockTypeLine {
			text += *block.Text + "\n"
		}
	}
	return text
}

func main() {
	detectDocumentTextWithS3Object()
}
```

## DetectDocumentTextについて
DetectDocumentTextは文字列のみ抽出する機能です。
特に他の情報が必要ない場合はDetectDocumentTextを使えばいいと思います。

## AnalyzeDocumentについて
TextractにはAnalyzeDocumentという機能もあります。
こちらはDetectDocumentTextよりもリッチな情報（例えばテーブル情報など）が抽出できます。
DetectDocumentTextよりも料金は高くなりますが、詳細な情報が欲しい場合はこちらを使いましょう。
AnalyzeDocumentのサンプルコードはこちらにあります。
https://github.com/sea-turt1e/aws-sdk-go-v2-example/blob/main/textract/analyzeDocumentTextWithS3Object

## AWS SDK for Go v1の場合
今回はAWS SDK for Goのv2を使っていますが、v1の場合は[こちらの記事](https://qiita.com/MasatoraAtarashi/items/c780fbb453e36f17e2b2)が詳しいです。
## まとめ
S3に置いてある画像からTextractでOCRする方法を書きました。
もしもっといい書き方などありましたら教えていただけるとありがたいです。