import pandas as pd
import openai
import json
import time
import os
from datetime import datetime
from typing import Dict, List
import argparse
from dotenv import load_dotenv

# Classification categories
CLASSIFICATION_CATEGORIES = {
    "組織構造": [
        "部署・事業部（経理課、マーケティング、海外事業部、本部、部、課、室、係、班、セクション、ユニットなど）",
        "チーム・グループ（データチーム、ディレクションG、WebコンサルティングG、プロジェクトチームなど）",
        "拠点・エリア（新宿店、梅田店、東海エリア、台湾支社、支店、営業所、工場、センター、事業所、店舗など）",
        "地域・エリア（関東、東海、関西、海外など）",
        "組織階層（組織1、組織2、組織階層１など）",
    ],
    "人事管理": [
        "職位階層（マネージャー、課長、部長、リーダー、メンバー、管理職、役職、職位、Positionなど）",
        "人事等級（M1、グレード、クラス、等級、Grade、ジョブグレード、職級、ランクなど）",
        "職責・役割（社員、職責、プレイングマネージャー、役割、職名、職務、職掌、職分など）",
        "配属・所属（所属、配属、所属部署、所属部門、所属チームなど）",
    ],
    "雇用管理": [
        "雇用区分（正社員、契約社員、派遣、アルバイト、内定社員、雇用形態、雇用区分、社員区分、従業員区分など）",
        "採用形態（新卒、中途入社、インターン採用、採用、採用区分、採用形態、採用種別、入社経路、入社経緯など）",
        "勤務期間（入社時期、勤続年数、就業年数、入社年、入社年度、入社年次、入社区分、社歴、在籍年数、在職年数、勤務年数、入社年数、現会社での就業年数、在籍期間、年次など）",
        "勤務形態（勤務形態、勤務区分、就業形態、内外勤、出向区分、出向先など）",
    ],
    "業務機能": [
        "職種分類（技術、営業、事務、コンサルタント、コーディネート、職種、職群、職務区分、職掌・資格など）",
        "専門領域（デジタルマーケティング、システム、コールセンター、プロジェクト、職能資格など）",
    ],
    "個人属性": [
        "基本情報（年齢、性別、年代、年齢層、年齢区分、生年、男女、Gender、Age、婚姻、家族構成、世代など）",
        "地域・エリア（東海エリア、関東、地区、地域、外国籍、国籍など）",
        "学歴（学歴、最終学歴など）",
    ],
    "その他・未分類": ["特殊カテゴリ・null値など"],
}


def create_classification_prompt_with_confidence(attribute_names: List[str]) -> str:
    """Create a prompt for classifying Japanese attribute group names with confidence scores."""

    categories_text = ""
    for main_category, subcategories in CLASSIFICATION_CATEGORIES.items():
        categories_text += f"\n{main_category}:\n"
        for subcategory in subcategories:
            categories_text += f"  - {subcategory}\n"

    # メインカテゴリのリストを明確に提示
    main_categories = list(CLASSIFICATION_CATEGORIES.keys())
    main_categories_text = "、".join(main_categories)

    prompt = f"""あなたは日本の属性グループ名を事前定義されたカテゴリに分類するタスクを担当しています。

## 重要: 必ず以下の6つのメインカテゴリのいずれか1つのみを返してください：
{main_categories_text}

分類カテゴリの詳細:
{categories_text}

## 分類ルール:
1. **必須**: 各属性グループ名を上記の6つのメインカテゴリのいずれか1つに分類してください
2. **禁止**: サブカテゴリや説明文の一部（例：「部署・事業部」「チーム・グループ」など）を返さないでください
3. 日本語の意味と文脈を考慮してください
4. 以下のキーワードマッピングを参考にしてください：
   - 入社、採用、雇用、勤務、就業、契約、正社員、派遣 → 雇用管理
   - 部署、部、課、室、係、チーム、グループ、拠点、支店、店舗、組織 → 組織構造
   - 役職、職位、等級、グレード、クラス、職責、職務、マネージャー → 人事管理
   - 年齢、性別、年代、学歴、地域、国籍 → 個人属性
   - 職種、職群、専門、技術、営業 → 業務機能
   - 上記に当てはまらない場合 → その他・未分類

## 分類例（正しい形式）:
- 入社区分 → 雇用管理 (信頼度: 0.9)
- データU編集部 → 組織構造 (信頼度: 0.8)
- マーケティングチーム → 組織構造 (信頼度: 0.9)
- 職位 → 人事管理 (信頼度: 0.9)
- 所属 → 人事管理 (信頼度: 0.7)
- 勤務地 → 組織構造 (信頼度: 0.8)
- 会社名 → その他・未分類 (信頼度: 0.3)
- 国籍 → 個人属性 (信頼度: 0.8)

## 信頼度スコア基準:
- 0.9-1.0: 非常に確信（明確で曖昧さのない分類）
- 0.7-0.8: 確信（おそらく正しいが若干の曖昧さあり）
- 0.5-0.6: 中程度の確信（妥当な分類だが不確実）
- 0.3-0.4: 低い確信（分類が困難、複数の可能性）
- 0.0-0.2: 非常に低い確信（不明確または曖昧）

## 出力形式:
以下のJSON形式で返してください。categoryには必ず上記6つのメインカテゴリのいずれか1つを使用してください：

{{
  "属性名": {{
    "category": "組織構造",
    "confidence": 0.85
  }},
  ...
}}

分類対象の属性名:
{json.dumps(attribute_names, ensure_ascii=False, indent=2)}

重要: JSON分類結果のみを返し、追加のテキストや説明は含めないでください。必ず6つのメインカテゴリのいずれか1つを使用してください。"""

    return prompt


def classify_with_openai_confidence(
    client: openai.OpenAI, attribute_names: List[str]
) -> Dict[str, Dict[str, float]]:
    """Classify attribute names using OpenAI API with confidence scores."""

    prompt = create_classification_prompt_with_confidence(attribute_names)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=4000,
        )

        text_content = response.choices[0].message.content

        if text_content is None:
            print("APIからの応答が空でした")
            return {}

        # Remove markdown code block formatting if present
        if text_content.startswith("```json"):
            text_content = text_content.replace("```json\n", "").replace("\n```", "")
        elif text_content.startswith("```"):
            text_content = text_content.replace("```\n", "").replace("\n```", "")

        result = json.loads(text_content)
        return result

    except Exception as e:
        print(f"Error with OpenAI API (confidence mode): {str(e)}")
        return {}


def process_csv_with_confidence(
    client: openai.OpenAI,
    df: pd.DataFrame,
    batch_size: int = 50,
    use_confidence: bool = True,
) -> pd.DataFrame:
    """Process CSV data in batches with optional confidence scores."""

    results = {}
    attribute_names = df["属性グループ名"].tolist()

    print(f"Total attributes to classify: {len(attribute_names)}")
    print(f"Processing in batches of {batch_size}...")

    # Process in batches
    for i in range(0, len(attribute_names), batch_size):
        batch = attribute_names[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(attribute_names) + batch_size - 1) // batch_size

        print(f"Processing batch {batch_num}/{total_batches}...")

        batch_results = classify_with_openai_confidence(client, batch)

        results.update(batch_results)

        # Rate limiting - wait between requests
        if i + batch_size < len(attribute_names):
            time.sleep(0.2)

    # Add classification results to dataframe
    if use_confidence:
        df["分類"] = df["属性グループ名"].map(
            lambda x: results.get(x, {}).get("category", "その他・未分類")
        )
        df["信頼度"] = df["属性グループ名"].map(
            lambda x: results.get(x, {}).get("confidence", 0.0)
        )
    else:
        df["分類"] = df["属性グループ名"].map(
            lambda x: results.get(x, "その他・未分類")
        )

    return df


def extract_low_confidence_items(
    df: pd.DataFrame, threshold: float = 0.7
) -> pd.DataFrame:
    """Extract items with low confidence scores for priority validation."""
    if "信頼度" not in df.columns:
        print("信頼度列が見つかりません。")
        return pd.DataFrame()

    low_confidence = df[df["信頼度"] < threshold].copy()
    return low_confidence.sort_values(by=["信頼度"], ascending=[True])  # type: ignore


def analyze_confidence_distribution(df: pd.DataFrame) -> Dict:
    """Analyze and return confidence score distribution as dictionary."""
    if "信頼度" not in df.columns:
        return {}

    stats = {
        "平均信頼度": df["信頼度"].mean(),
        "中央値": df["信頼度"].median(),
        "最小値": df["信頼度"].min(),
        "最大値": df["信頼度"].max(),
    }

    # 信頼度区間別の統計
    confidence_ranges = [
        ("Very High (0.9-1.0)", 0.9, 1.0),
        ("High (0.7-0.9)", 0.7, 0.9),
        ("Medium (0.5-0.7)", 0.5, 0.7),
        ("Low (0.3-0.5)", 0.3, 0.5),
        ("Very Low (0.0-0.3)", 0.0, 0.3),
    ]

    range_stats = []
    for label, min_val, max_val in confidence_ranges:
        count = len(df[(df["信頼度"] >= min_val) & (df["信頼度"] < max_val)])
        percentage = count / len(df) * 100
        range_stats.append({"信頼度区間": label, "件数": count, "割合": percentage})

    return {"stats": stats, "ranges": range_stats}


def generate_markdown_report(
    df: pd.DataFrame,
    output_file: str,
    confidence_analysis: Dict,
    low_confidence_items: pd.DataFrame,
    processing_time: float = 0.0,
) -> str:
    """Generate a comprehensive markdown report."""

    timestamp = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")

    # 分類結果の集計
    classification_counts = df["分類"].value_counts()

    # 信頼度分析
    confidence_stats = confidence_analysis.get("stats", {})
    confidence_ranges = confidence_analysis.get("ranges", [])

    # マークダウンレポートの生成
    report = f"""# 属性グループ分類結果レポート

**生成日時**: {timestamp}  
**処理対象ファイル**: {output_file}  

## 分類結果サマリー

### 分類別件数

| 分類カテゴリ | 件数 | 割合 |
|-------------|------|------|"""

    for category, count in classification_counts.items():
        percentage = count / len(df) * 100
        report += f"\n| {category} | {count:,}件 | {percentage:.1f}% |"

    report += """

## 信頼度分析

### 信頼度統計

| 項目 | 値 |
|------|-----|"""

    for key, value in confidence_stats.items():
        report += f"\n| {key} | {value:.3f} |"

    report += """

### 信頼度区間別統計

| 信頼度区間 | 件数 | 割合 |
|-----------|------|------|"""

    for range_info in confidence_ranges:
        report += f"\n| {range_info['信頼度区間']} | {range_info['件数']:,}件 | {range_info['割合']:.1f}% |"

    report += """

## 低信頼度項目

**閾値**: 0.7未満  

### 信頼度が最も低い項目（上位10件）

| 順位 | 属性グループ名 | 分類 | 信頼度 |
|------|---------------|------|--------|"""

    top_low_confidence = low_confidence_items.head(10)
    for i, (_, row) in enumerate(top_low_confidence.iterrows(), 1):
        report += (
            f"\n| {i} | {row['属性グループ名']} | {row['分類']} | {row['信頼度']:.3f} |"
        )

    report += """

## 詳細データ

### 分類別詳細一覧

各分類カテゴリの詳細な属性グループ一覧は、出力されたCSVファイルをご確認ください。

### データ形式

| 列名 | 説明 |
|------|------|
| 属性グループ名 | 分類対象の属性名 |
| 数 | 該当するデータ件数 |
| 分類 | 自動分類されたカテゴリ |
| 信頼度 | 分類の信頼度スコア（0.0-1.0） |

## 検証推奨項目

以下の項目は信頼度が低いため、手動での検証をお勧めします：

1. **信頼度0.3未満**: 分類が困難な項目
2. **信頼度0.3-0.5**: 複数の分類可能性がある項目
3. **信頼度0.5-0.7**: 基本的な分類は可能だが、詳細な検証が必要な項目

"""

    return report


def create_date_directory() -> str:
    """当日の日付でディレクトリを作成し、そのパスを返す"""
    today = datetime.now().strftime("%Y%m%d")
    directory_path = os.path.join(".", today)

    # ディレクトリが存在しない場合は作成
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"日付ディレクトリを作成しました: {directory_path}")

    return directory_path


def save_markdown_report(report: str, output_file: str, date_dir: str) -> str:
    """Save markdown report to file in the date directory."""
    # ファイル名を取得し、日付ディレクトリに配置
    filename = os.path.basename(output_file).replace(".csv", "_report.md")
    report_file = os.path.join(date_dir, filename)

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
    return report_file


def select_csv_file():
    """Prompt user to select a CSV file."""
    print("分類前のCSVファイルを選択してください:")
    print(
        "ファイル名を入力するか、Enterキーを押して現在のディレクトリのCSVファイルを一覧表示:"
    )

    user_input = input().strip()

    if user_input:
        if os.path.exists(user_input):
            return user_input
        else:
            print(f"ファイルが見つかりません: {user_input}")
            return None

    # List CSV files in current directory
    csv_files = [f for f in os.listdir(".") if f.endswith(".csv")]

    if not csv_files:
        print("現在のディレクトリにCSVファイルが見つかりません。")
        return None

    print("利用可能なCSVファイル:")
    for i, file in enumerate(csv_files, 1):
        print(f"{i}. {file}")

    while True:
        try:
            choice = input("ファイル番号を選択してください: ").strip()
            if choice.isdigit():
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(csv_files):
                    return csv_files[choice_idx]
                else:
                    print("無効な番号です。")
            else:
                print("数字を入力してください。")
        except KeyboardInterrupt:
            print("\n処理を中断しました。")
            return None


def main():
    parser = argparse.ArgumentParser(
        description="属性グループ自動分類ツール（信頼度付き）"
    )
    parser.add_argument("--api-key", help="OpenAI API Key")
    parser.add_argument("--input", help="入力CSVファイルのパス")
    parser.add_argument("--output", help="出力CSVファイルのパス")
    parser.add_argument(
        "--batch-size", type=int, default=50, help="バッチサイズ（デフォルト: 50）"
    )
    parser.add_argument(
        "--no-confidence",
        action="store_true",
        help="信頼度スコアを取得しない（従来モード）",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="低信頼度項目の閾値（デフォルト: 0.7）",
    )

    args = parser.parse_args()

    load_dotenv()

    api_key = args.api_key
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = input("OpenAI API Keyを入力してください: ").strip()
        if not api_key:
            print("API Keyが必要です。")
            print("以下のいずれかの方法でAPI Keyを設定してください:")
            print("1. --api-key オプションで指定")
            print("2. .envファイルに OPENAI_API_KEY=your_key を設定")
            print("3. 対話的に入力")
            return

    # Initialize OpenAI client
    try:
        client = openai.OpenAI(api_key=api_key)
        print("OpenAI APIクライアントを初期化しました。")
    except Exception as e:
        print(f"API Key の設定に失敗しました: {str(e)}")
        return

    input_file = args.input
    if not input_file:
        input_file = select_csv_file()
        if not input_file:
            return

    try:
        print(f"CSVファイルを読み込み中: {input_file}")
        df = pd.read_csv(input_file)

        # Validate columns
        required_columns = ["属性グループ名", "数"]
        if not all(col in df.columns for col in required_columns):
            print(f"CSVファイルに必要な列が含まれていません: {required_columns}")
            return

        print(f"CSVファイルを読み込みました: {len(df)} 行")

        # Classification
        use_confidence = not args.no_confidence
        print("分類を開始します...")

        start_time = time.time()
        classified_df = process_csv_with_confidence(
            client, df.copy(), args.batch_size, use_confidence
        )
        processing_time = time.time() - start_time

        if "分類" in classified_df.columns:
            print("分類が完了しました！")

            # Create date directory
            date_dir = create_date_directory()

            # Generate output filename with date
            if args.output:
                # ユーザー指定のファイル名を使用し、ディレクトリ内に配置
                output_filename = os.path.basename(args.output)
                output_file = os.path.join(date_dir, output_filename)
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if use_confidence:
                    output_filename = (
                        f"classified_attributes_with_confidence_{timestamp}.csv"
                    )
                else:
                    output_filename = f"classified_attributes_{timestamp}.csv"
                output_file = os.path.join(date_dir, output_filename)

            # Save results
            classified_df.to_csv(output_file, index=False)
            print(f"分類結果を保存しました: {output_file}")

            # Show summary
            classification_counts = classified_df["分類"].value_counts()
            print("\n分類結果サマリー:")
            for category, count in classification_counts.items():
                print(f"  {category}: {count}件")

            # Confidence analysis (if available)
            if use_confidence and "信頼度" in classified_df.columns:
                confidence_analysis = analyze_confidence_distribution(classified_df)

                # Display confidence analysis
                if confidence_analysis:
                    stats = confidence_analysis.get("stats", {})
                    ranges = confidence_analysis.get("ranges", [])

                    print("\n信頼度分析:")
                    for key, value in stats.items():
                        print(f"  {key}: {value:.3f}")

                    print("\n信頼度区間別統計:")
                    for range_info in ranges:
                        print(
                            f"  {range_info['信頼度区間']}: {range_info['件数']}件 ({range_info['割合']:.1f}%)"
                        )

                # Extract and save low confidence items
                low_confidence_items = extract_low_confidence_items(
                    classified_df, args.confidence_threshold
                )
                if not low_confidence_items.empty:
                    low_conf_filename = output_filename.replace(
                        ".csv", "_low_confidence.csv"
                    )
                    low_conf_file = os.path.join(date_dir, low_conf_filename)
                    low_confidence_items.to_csv(low_conf_file, index=False)
                    print(
                        f"\n低信頼度項目（< {args.confidence_threshold}）を保存しました: {low_conf_file}"
                    )
                    print(f"検証推奨項目: {len(low_confidence_items)}件")

                    # Show top low confidence items
                    if len(low_confidence_items) > 0:
                        print("\n信頼度が最も低い項目（上位5件）:")
                        top_low_confidence = low_confidence_items.head(5)
                        for _, row in top_low_confidence.iterrows():
                            print(
                                f"  - {row['属性グループ名']} (分類: {row['分類']}, 信頼度: {row['信頼度']:.3f})"
                            )

                # Generate and save markdown report
                if confidence_analysis:
                    report = generate_markdown_report(
                        classified_df,
                        output_file,
                        confidence_analysis,
                        low_confidence_items,
                        processing_time,
                    )
                    report_file = save_markdown_report(report, output_file, date_dir)
                    print(f"\n📊 マークダウンレポートを生成しました: {report_file}")

        else:
            print("分類処理に失敗しました。")

    except Exception as e:
        print(f"ファイル処理中にエラーが発生しました: {str(e)}")


if __name__ == "__main__":
    main()
