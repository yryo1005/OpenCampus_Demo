プログラムの書き方やマークダウンの書き方は `@cursor_template/root_prompt.md` を参考にしてください
高校生向けに大学のオープンキャンパスとしてAIについてのデモンストレーションを作ります

要件 
* Google Colab(T4, 14GB VRAM)で動作する 
* Gradioをフロントエンドとして用いて画像や音声，文章の入力を受け取る 
* 動作確認のためにサンプル入力も用意する
* プログラムはColabで実行されるのでipynbノートブックのみで動作する
* 作成したノートブックの名前を'OC_{デモを表す単語}.ipynb'としてください
* 作成したノートブックの使用方法やデモの説明を'OC_{デモを表す単語}.md'に記述してください
* 途中で実行が止まることを考え，下記のようにセルを分けてください
    * ライブラリのインストール
    * ライブラリの読み込み，変数のインスタンス化
    * Gradioの実行
* Hugging FaceのWEBサイトで認証が必要な場合，その旨をユーザに述べてください
* このプログラムは `yryo1005/OpenCampus_Demo` リポジトリの `main` ブランチにPublicとして公開されます
* ipynbノートブックの上部にColabで開くボタンを追加してください
```text
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yryo1005/OpenCampus_Demo/blob/main/{ノートブック名}.ipynb)
```

`@colab_env_info.txt` がGoogle Colabの環境情報なので，これを参考にしてください
GeminiのAPIキーが `tokens.json` の `gemini` キーに保存されているので，必要に応じて使用してください
Geminiは無料の `gemini-2.5-flash` を用いてください

Depth Anythingによる深度推定を実装してください
* ユーザーは写真をカメラから入力します
* ユーザーが写真を撮影しなくて良いようにインターネットからサンプル画像を追加してください