# 属性グループ自動分類スクリプト

このツールは、CSVファイルに含まれる日本語の属性グループ名を自動的に分類してくれるツールです。
AI（OpenAI GPT）を使って、「組織構造」「人事管理」「雇用管理」などのカテゴリに分けてくれます。

## 📋 このツールでできること

- CSVファイルの属性グループ名を自動分類
- 分類の信頼度（どれくらい確実か）も表示
- 結果をCSVファイルとレポートで出力
- 信頼度が低い項目を別ファイルで確認可能

## 🏷️ 分類カテゴリ

属性グループは以下の6つのカテゴリに分類されます：

1. **組織構造** - 部署、チーム、拠点など
2. **人事管理** - 職位、等級、役割など  
3. **雇用管理** - 雇用区分、採用形態、勤務期間など
4. **業務機能** - 職種、専門領域など
5. **個人属性** - 年齢、性別、学歴など
6. **その他・未分類** - 上記に当てはまらないもの

## 🛠️ 必要な準備

### 1. 必要なライブラリのインストール

```bash
# uvを使用する場合
uv sync

# または pipを使用する場合
pip install pandas openai python-dotenv
```

### 2. OpenAI API Keyの設定

#### ステップ1: .envファイルを作成
.env.exampleをコピー

```bash
cp .env.example .env
```

#### ステップ2: API Keyを書き込む
作成した`.env`ファイルに以下の内容を書きます：

```
OPENAI_API_KEY=あなたのAPIキーをここに書く
```

## 📊 使い方

### 1. CSVファイルの準備
分類したいCSVファイルを用意します。ファイルには以下の列が必要です：
- `属性グループ名` - 分類したい属性の名前
- `数` - その属性のデータ件数

**CSVファイルの例：**
```csv
属性グループ名,数
営業部,150
年齢,300
入社年度,200
```

### 2. スクリプトの実行

#### 基本的な使い方
```bash
python classify_attributes.py
```

またはuvを使っている場合

```bash
uv run classify_attributes.py
```

実行すると、CSVファイルを選択する画面が出ます。

#### コマンドラインオプション付きの使い方
```bash
# 入力ファイルを指定
python classify_attributes.py --input データ.csv

# 出力ファイル名も指定
python classify_attributes.py --input データ.csv --output 結果.csv

# API Keyを直接指定
python classify_attributes.py --api-key sk-your-api-key

# バッチサイズを変更（一度に処理する件数）
python classify_attributes.py --batch-size 30
```

### 3. 結果の確認

実行すると、以下のファイルが作成されます：

#### 📁 日付フォルダ
結果は実行日の日付フォルダ（例：`20241225`）に保存されます。

#### 📄 出力ファイル
1. **メインの結果ファイル** - `classified_attributes_with_confidence_YYYYMMDD_HHMMSS.csv`
   - 分類結果と信頼度が含まれます
   
2. **低信頼度項目ファイル** - `classified_attributes_with_confidence_YYYYMMDD_HHMMSS_low_confidence.csv`
   - 信頼度が低い項目（要確認項目）
   
3. **レポートファイル** - `classified_attributes_with_confidence_YYYYMMDD_HHMMSS_report.md`
   - 分類結果のサマリーレポート

## 📋 コマンドラインオプション

| オプション | 説明 | 例 |
|-----------|------|-----|
| `--input` | 入力CSVファイルのパス | `--input data.csv` |
| `--output` | 出力CSVファイルのパス | `--output result.csv` |
| `--api-key` | OpenAI API Key | `--api-key sk-abc123...` |
| `--batch-size` | 一度に処理する件数 | `--batch-size 30` |
| `--no-confidence` | 信頼度を計算しない | `--no-confidence` |
| `--confidence-threshold` | 低信頼度の閾値 | `--confidence-threshold 0.6` |

## 🔍 結果の見方

### 分類結果CSV
| 列名 | 説明 |
|------|------|
| 属性グループ名 | 元の属性名 |
| 数 | データ件数 |
| 分類 | 分類されたカテゴリ |
| 信頼度 | 分類の確実さ（0.0-1.0） |

### 信頼度の目安
- **0.9-1.0**: とても確実
- **0.7-0.9**: まあまあ確実
- **0.5-0.7**: 普通
- **0.3-0.5**: 少し不安
- **0.0-0.3**: かなり不安（要確認）

## ❗ よくある問題と解決方法

### 1. `OPENAI_API_KEY not found` エラー
**原因**: API Keyが設定されていません
**解決**: `.env`ファイルを作成し、正しいAPI Keyを設定してください

### 2. `pandas not found` エラー
**原因**: 必要なライブラリがインストールされていません
**解決**: `pip install pandas openai python-dotenv` を実行してください

### 3. CSVファイルが読み込めない
**原因**: ファイルに必要な列（`属性グループ名`, `数`）がありません
**解決**: CSVファイルの列名を確認してください
