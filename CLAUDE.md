# CLAUDE.md

Solar Filament Segmentation Challenge 2026 のリポジトリ。
このファイルはプロジェクトの運用ルールであり、**コミット対象**。
除外するのはローカル設定（`.claude/`, `.mcp.json`）の方。

---

## 1. プロジェクト

GONG の Hα 太陽観測画像（2048×2048, 8bit グレースケール JPEG）から、
ダークフィラメントを **1本ずつインスタンス分割**する。

- コンペ: https://www.kaggle.com/competitions/filament-segmentation-2026
- 評価指標: Panoptic Quality (PQ), IoU 閾値 0.5
- 目標: ローカル PQ 0.40 前後（正攻法の公開最高が 0.37）
- 最終提出: 2026-11-15（Kaggle の submission とは別に、GitHub リポジトリ＋4ページ英語レポートを Google Form で提出）

詳しい背景は `Working/00_survey/` を読むこと（日本語、非コミット）。

## 2. 言語ポリシー（重要）

| 対象 | 言語 |
|---|---|
| 対話・issue・PR・コミットメッセージ本文 | 日本語 |
| **コード内のコメント・docstring** | **英語** |
| **README.md / 公開ドキュメント** | **英語** |
| **ログ出力・例外メッセージ・CLI ヘルプ** | **英語** |
| 技術レポート（4ページ） | 英語 |
| `Working/` 配下の作業メモ | 日本語 |

日本語で指示を受けても、**成果物のコードとドキュメントは英語で書く**。
審査委員は全員アメリカの研究機関（UMSL / GSU / NJIT / NSO）に所属しており、
ルーブリックの30%がコード品質とドキュメントの評価に充てられている。

コミットメッセージは `feat: add PQ evaluator` のように Conventional Commits の
prefix ＋英語 summary とし、詳細な説明が必要なら body に日本語で書いてよい。

## 3. 絶対に守ること

1. **公開版 MAGFiLO（Harvard Dataverse など）の ground-truth を一切使わない。**
   test 180枚の大半が公開版と重複しておりリークになる。主催者が明確に禁止している。
   ダウンロードスクリプトも書かない。参照もしない。
2. **`MAGFiLO_1.0_Kaggle_2026/` をコミットしない。** CC BY-NC 4.0 で再配布不可。
3. **`Working/` をコミットしない。** 日本語資料・実験結果・中間生成物の置き場。
4. **ローカル絶対パス（`C:\work\...`）をコードに埋め込まない。** 設定は `configs/` か環境変数経由。
5. **秘密情報をコミットしない。** `kaggle.json`, `.env`, トークン類。
6. ライセンスは **Apache-2.0**（受賞時に OSI 承認ライセンスでの公開が必須）。

## 4. パス

| 用途 | パス |
|---|---|
| リポジトリルート | このファイルのある場所 |
| データセット | `MAGFiLO_1.0_Kaggle_2026/`（非コミット） |
| 学習画像 | `MAGFiLO_1.0_Kaggle_2026/train/train_images/` (707枚) |
| 学習アノテーション | `MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json` |
| テスト画像 | `MAGFiLO_1.0_Kaggle_2026/test/test_images/` (180枚) |
| 作業領域（非コミット） | `Working/` |
| 調査資料（非コミット） | `Working/00_survey/` |

データのパスは `configs/paths.yaml` に集約し、環境変数 `MAGFILO_ROOT` で上書きできるようにする。

## 5. 評価指標の理解（設計判断の根拠になる）

```
PQ = Σ_{TP} IoU / ( |TP| + 0.5|FP| + 0.5|FN| )   … IoU > 0.5 のペアのみ TP

分解すると  PQ = SQ × RQ
  SQ = TP ペアの平均 IoU           … マスクの形の品質
  RQ = TP / (TP + 0.5FP + 0.5FN)   … 検出の F1
```

設計上の帰結:

- **IoU 0.49 の予測は FP と FN を同時に立てる。** 分母 +1.0、分子 0。
  何も出さない場合（FN のみ、分母 +0.5）より悪い。
- 出すか出さないかの損益分岐は概ね `p·u > 0.5 × 現在のPQ`。
  PQ 0.40、成功時 IoU 0.7 なら「IoU>0.5 を超える確率 29% 以上なら出す」。
  `min_area` と `conf` はこの観点で最適化する。
- Dice を上げる最適化と PQ を上げる最適化は一致しない。
  **実験ログには必ず PQ / SQ / RQ / TP / FP / FN を分けて記録する。**

### 複数アノテータ

学習データは 707 画像 / 1,154 アノテータ×画像エントリ（1人:411, 2人:145, 3人:151）。
`image_id` は `<アノテータバッチ>-<画像名>` 形式。
評価は**アノテータ×画像の単位**でループし、同じ予測が複数回採点される。

→ **train/val の分割は必ず画像ファイル名（stem）単位のグループ分割**にすること。
アノテータ違いの同一画像が train と val に跨ると検証が壊れる。

## 6. 提出フォーマットの制約

- 単一 CSV、列は `filament_id`, `segmentation_rle`
- `filament_id` = `<画像ID>_<連番>`（例 `20150125172714Mh_1`）。画像IDは拡張子なし
- `segmentation_rle` = pycocotools の compressed RLE の **counts のみ**。size は不要、引用符も付けない
- **同一画像内で予測マスクが1画素でも重なると submission が ERROR になる**
  → スコア降順の panoptic painting で非重複を保証し、提出前に必ず検査スクリプトを通す
- Kaggle への提出は 1日5回まで。**ローカル PQ で判断し、提出枠を探索に使わない**

## 7. ディレクトリ構成

```
.
├── CLAUDE.md                  # このファイル（非コミット）
├── README.md                  # 英語。再現手順
├── LICENSE                    # Apache-2.0
├── pyproject.toml
├── requirements.txt           # バージョン固定（規約要件）
├── configs/                   # yaml。パスとハイパーパラメータ
├── src/filament/
│   ├── data/                  # COCO読み込み、split、Dataset
│   ├── models/                # detector / refiner
│   ├── postprocess/           # panoptic painting, 足切り, マージ
│   ├── metrics/               # PQ, 重なり検査
│   └── submit/                # RLE化, CSV出力
├── scripts/                   # train.py / predict.py / evaluate.py
├── docs/
│   └── experiments/           # フェーズ単位の公開記録（英語・凍結）
├── tests/
├── notebooks/
│   ├── 00_eda.ipynb
│   └── 99_full_pipeline.ipynb # 規約要件。これ1本で再現できること
├── MAGFiLO_1.0_Kaggle_2026/   # 非コミット
└── Working/                   # 非コミット
```

## 8. 開発ルール

- Python 3.12（Kaggle のノートブックと同じ系列に揃える）、パッケージ管理は `uv`
- lint/format は `ruff`、型は `mypy`（strict までは求めない）
- テストは `pytest`。**`src/filament/metrics/` は必ずテストを書く**（評価器が壊れると全実験が無意味になる）
- 乱数 seed は固定し、split は一度決めたら変えない
- 1実験 = 1 commit を原則とし、実験日誌に1エントリ書く（第9節）。進行中の実験ログはこの日誌に一本化し、別途サマリ表は作らない。例外は `docs/experiments/` の公開記録で、これはフェーズ終了時に1度だけ書いて凍結する（第9節末）
- 重い処理（学習・全画像推論）は勝手に実行せず、コマンドを提示して確認を取る
- 新しい依存を追加するときは理由を述べてから追加する

## 9. 実験日誌（必須）

**1タスク完了ごとに `Working/20_laboratory_notebook/YYYYMMDD.md` へ追記する。**
書式と規則は `Working/20_laboratory_notebook/00_記録ルール.md` に定義してある。
作業を始める前に一度読むこと。

要点だけ再掲:

- ファイルは日付ごと。1日1ファイル、タスクごとに追記。過去のエントリは書き換えない
- 構成は `文書区分` ラベル → `結論` → `進捗状況` → `計測` → `変更点・根拠` → `詳細` → `別件`
- 結論から書く。思考した順番（調査→発見→仮説→結論）で書かない
- **数値を伴う実験をした回は「計測」を必須とし、PQ 単独ではなく SQ / RQ / TP / FP / FN を分けて書く**
- **失敗した実験こそ記録する。** 効かなかった手法の記録は最終レポートの材料になる
- commit hash と再現コマンドを必ず残す

この日誌は最終提出の4ページ技術レポート（アブレーション）の一次資料になる。

### 公開記録との関係

`docs/experiments/` に、フェーズ単位の公開記録を英語で置く（構成は
やった事 / 結果 / 考察 / 次）。日誌との役割分担は次のとおり。

| | `Working/20_laboratory_notebook/` | `docs/experiments/` |
|---|---|---|
| 読み手 | 自分 | 審査員・外部 |
| 言語 | 日本語 | 英語 |
| 単位 | 日次・タスクごと | フェーズごと |
| 更新 | 進行中に追記 | **終了時に1度書いて凍結** |
| コミット | しない | する |

公開記録は日誌から派生させる。数値は凍結時点のものを写し、以後書き換えない。
後のフェーズが前の結論を覆した場合は、前を書き換えず後の記録で述べる。
**凍結するのは、同じ数値が2か所で更新され続けると片方が腐るため。**

## 10. よく使うコマンド

```bash
uv sync                                    # 依存のインストール
uv run pytest                              # テスト
uv run ruff check . && uv run ruff format .
uv run python scripts/evaluate.py --pred <csv> --gt <csv>
uv run python scripts/check_submission.py --csv submission.csv   # 重なり検査
```

## 11. 公開前チェックリスト（コンペ終了時に実施）

- [ ] `CLAUDE.md` 内の `Working/` への参照を外す（公開リポジトリには Working/ が無いため）
- [ ] `CLAUDE.md` を英語化するか、英語の `CONTRIBUTING.md` を別途用意する
- [ ] `.claude/`, `.mcp.json` が履歴に入っていないか確認
- [ ] コミットトレーラにセッションURLが残っていないか確認
- [ ] ローカル絶対パスの混入がないか `grep -r "C:\\\\work"` で確認
- [ ] README.md が英語で、クローンから submission.csv まで再現できる手順になっているか
- [ ] `requirements.txt` のバージョンが固定されているか
- [ ] `notebooks/99_full_pipeline.ipynb` が Save & Run All 相当で通るか
- [ ] LICENSE が Apache-2.0 で配置されているか
- [ ] `docs/experiments/` が全フェーズ分そろっていて、英語で、凍結後に編集されていないか
