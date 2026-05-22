# Real Data Operation Guide

この手順書は、`relationship_lifelog_agent` を実データに接続して使う前に、安全性を確認しながら小さく進めるための運用ガイドです。

このアプリは relationship evidence review のためのローカルツールです。上流2アプリのデータを判断材料として read-only に参照しますが、上流アプリや上流DBを変更しません。最初から広い期間に対して書き込みを行わず、counts-only の短期間確認から始めてください。

## 1. 前提

- `personal_lifelog_rag` が作成済みで、必要なローカルDBまたはexportが存在する。
- `notes_lifelog_rag` が作成済みで、必要なローカルDBまたはexportが存在する。
- `relationship_lifelog_agent` がローカルで起動できる。
- `pytest` が通る。
- `config.local.yaml` はgit管理しない。
- 外部API、クラウドアップロード、モデル自動ダウンロード、Gradio `share=True` は使わない。
- 実データ本文、LINE全文、メモ全文、正確GPS、顔情報、写真実体、private path はdocsやGitHubに載せない。

## 2. 推奨手順

### Step 1: AGENTS.md確認

作業前に `AGENTS.md` を確認し、以下の前提を守ってください。

- 上流2アプリは編集しない。
- 上流DBは read-only で扱う。
- relationship label と person / LINE speaker の対応は手動設定のみ。
- public mode では個人名、関係ラベル、本文excerpt、正確GPS、顔情報、写真実体、private path を出さない。

### Step 2: config.local.yaml作成

`config.example.yaml` を参考に、ローカル専用の `config.local.yaml` を作成します。実パスはこのファイルだけに書き、gitに追加しません。

```yaml
app:
  host: 127.0.0.1
  allow_external_api: false
  allow_model_auto_download: false
  allow_gradio_share: false

adapter:
  backend: upstream_readonly
  upstream_access_mode: readonly
  copy_raw_upstream_data: false

paths:
  relationship_db: "<relationship_db.sqlite>"
  personal_lifelog_db: "<personal_lifelog_db.sqlite>"
  notes_lifelog_db: "<notes_lifelog_db.sqlite>"

privacy:
  public_mode_enabled: false
  redact_exact_gps: true
  redact_line_full_text: true
  redact_note_full_text: true
  forbid_public_relationship_labels: true
```

パス値は環境ごとに異なるため、このdocsには実パスを書かないでください。

### Step 3: doctor実行

まず doctor で危険設定と未設定項目を確認します。

```bash
python -m relationship_lifelog_agent.cli --config config.local.yaml doctor
python -m relationship_lifelog_agent.cli --config config.local.yaml doctor --backend upstream_readonly
python -m relationship_lifelog_agent.cli --config config.local.yaml doctor --format json
```

`WARN` は profile 未設定や upstream DB 未設定など、追加セットアップが必要な状態です。`ERROR` は `share=True`、外部API許可、モデル自動ダウンロード許可、read-only接続不可など、安全に進めない状態です。

### Step 4: upstream inspect実行

上流DBのschemaと adapter mapping の対応を確認します。出力は table / column / row count / coverage だけで、本文は出しません。

```bash
python -m relationship_lifelog_agent.cli --config config.local.yaml upstream inspect \
  --backend upstream_readonly \
  --format markdown \
  --output data/exports/upstream_schema_inspection.md

python -m relationship_lifelog_agent.cli --config config.local.yaml upstream inspect --format json
```

reportを共有する場合でも、private path や本文が混ざっていないことを確認してください。

### Step 5: upstream smoke実行

次に、小さな期間で counts-only のE2E接続確認をします。relationship DB には書き込みません。

```bash
python -m relationship_lifelog_agent.cli --config config.local.yaml upstream smoke \
  --backend upstream_readonly \
  --date-from 2025-01-01 \
  --date-to 2025-01-07 \
  --profile-id 1 \
  --output data/exports/upstream_smoke_week1.md
```

確認するのは source count、coverage、warnings、redaction status、`write_count: 0` です。本文や写真パスを探す作業には使わないでください。

### Step 6: profile手動作成

relationship profile は必ずユーザーが手動で作成します。AIに relationship label を推定させたり、person と LINE speaker を自動リンクさせたりしません。

```bash
python -m relationship_lifelog_agent.cli --config config.local.yaml profile create \
  --profile-name "<profile_name>" \
  --person-source-id "<person_source_id>" \
  --line-speaker-source-id "<line_speaker_source_id>" \
  --relationship-label "<allowed_relationship_label>" \
  --valid-from 2025-01-01

python -m relationship_lifelog_agent.cli --config config.local.yaml profile list
```

`relationship-label` は `partner`, `ex_partner`, `close_person`, `other_private` のいずれかを手動で選びます。public mode の回答やreportでは関係ラベルを出さないでください。

### Step 7: 1週間だけ counts-only dry-run

最初の analysis は1週間だけに絞り、`counts-only` で実行します。candidate数、source counts、warnings、evidence strength分布だけを確認します。

```bash
python -m relationship_lifelog_agent.cli --config config.local.yaml analyze dry-run \
  --profile-id 1 \
  --date-from 2025-01-01 \
  --date-to 2025-01-07 \
  --backend upstream_readonly \
  --privacy-level counts-only \
  --output data/exports/dry_run_counts_week1.md
```

この段階では本文excerptを出さず、relationship DBにも書き込みません。

### Step 8: 1か月だけ redacted dry-run

1週間の結果に問題がなければ、1か月だけ `redacted` で確認します。

```bash
python -m relationship_lifelog_agent.cli --config config.local.yaml analyze dry-run \
  --profile-id 1 \
  --date-from 2025-01-01 \
  --date-to 2025-01-31 \
  --backend upstream_readonly \
  --privacy-level redacted \
  --output data/exports/dry_run_redacted_month.md
```

redacted report では匿名化されたsummaryと安全化された source pointer のみを確認します。人物名、関係ラベル、LINE本文、メモ本文、正確GPS、path が出ていないことを確認してください。

### Step 9: private dry-runを自分だけ確認

必要な場合だけ、ローカル環境で `private` dry-run を確認します。`private` でも相手の内心や関係状態を断定してはいけません。

```bash
python -m relationship_lifelog_agent.cli --config config.local.yaml analyze dry-run \
  --profile-id 1 \
  --date-from 2025-01-01 \
  --date-to 2025-01-31 \
  --backend upstream_readonly \
  --privacy-level private \
  --output data/exports/dry_run_private_month.md
```

private report は共有しないでください。public mode では `private` privacy level を使えません。

### Step 10: 問題なければ --write

dry-runの内容と安全性を確認できた場合だけ、明示的に `--write` を付けます。デフォルトは常に dry-run です。

```bash
python -m relationship_lifelog_agent.cli --config config.local.yaml analyze dry-run \
  --profile-id 1 \
  --date-from 2025-01-01 \
  --date-to 2025-01-31 \
  --backend upstream_readonly \
  --privacy-level redacted \
  --write
```

`--write` 前には relationship DB のbackupが作成されます。保存対象は candidate summary、source pointer、短い安全なexcerpt、confidence、evidence_strength などに限定されます。上流DBには書き込みません。

### Step 11: Chat UIで質問

Chat UI はローカルホストで起動します。

```bash
python -m relationship_lifelog_agent.app --config config.local.yaml
```

設定Accordionで backend、profile、date range、post-conflict window、private/public mode、debugを選びます。debugはデフォルトoffのままにしてください。

### Step 12: review actionで修正

回答に含まれる candidate は、Chat UI の最小review actionで修正します。

- 確認済みにする。
- これは違う。
- 軽いすれ違いにする。
- 冗談として扱う。
- 仲直り済みにする。
- 再分析対象にする。

review action は `relationship_review_actions` に履歴として保存されます。上流データや元の source record は削除・変更されません。

## 3. 推奨しないこと

- 初回から全期間に対して `--write` する。
- public mode で private 内容を見る。
- profile未設定で関係分析する。
- dry-run report を他人に共有する。
- raw LINE / raw note / GPS / 顔情報 / 写真パスを docs や GitHub に載せる。
- relationship DB のbackupを取らずに手動編集する。
- `config.local.yaml`、DB、exports、backups、cache をgitに追加する。
- AIに恋人ラベル、親密度、関係状態、相手の気持ちを推定させる。

## 4. 安全コマンド例

通常は次の順で進めます。

```bash
python -m relationship_lifelog_agent.cli --config config.local.yaml doctor --backend upstream_readonly
python -m relationship_lifelog_agent.cli --config config.local.yaml upstream inspect --backend upstream_readonly
python -m relationship_lifelog_agent.cli --config config.local.yaml upstream smoke \
  --backend upstream_readonly \
  --date-from 2025-01-01 \
  --date-to 2025-01-07 \
  --profile-id 1
python -m relationship_lifelog_agent.cli --config config.local.yaml analyze dry-run \
  --profile-id 1 \
  --date-from 2025-01-01 \
  --date-to 2025-01-07 \
  --backend upstream_readonly \
  --privacy-level counts-only
```

書き込み前には backup 一覧も確認します。

```bash
python -m relationship_lifelog_agent.cli --config config.local.yaml db backup
python -m relationship_lifelog_agent.cli --config config.local.yaml db backups
```

実データ運用前後にはテストも実行してください。

```bash
pytest
python eval/run_eval.py
```

## 5. トラブルシューティング

### doctor が upstream DB 未設定を WARN にする

`config.local.yaml` の upstream DB 設定を確認してください。docsやissueには実パスを書かず、必要なら `<personal_lifelog_db.sqlite>` のように匿名化して共有します。

### read-only 接続に失敗する

DBファイルの場所、権限、SQLiteファイルであることを確認します。adapterは read-only 接続を使うため、上流DBを更新する必要はありません。

### profile が未設定と表示される

`profile create` で手動profileを作成してください。person_source_id と line_speaker_source_id は手動で選びます。AIによる自動リンクは行いません。

### dry-run report に想定外の内容が出る

まず `counts-only` に戻してください。次に `redacted` で再確認し、本文、正確GPS、顔情報、写真パス、private path が含まれないことを確認します。

### Chat UIで upstream_readonly が使えない

backendを `mock` に戻すとUIは継続利用できます。upstream設定は doctor、inspect、smoke の順に確認してください。

### pytest が失敗する

失敗したテスト名と内容を確認し、安全系テストを優先して直します。privacy guard、answer safety、adapter contract、schema、review-aware ranking の失敗は実データ運用前に解消してください。

## 6. 削除・rollback

`--write` 前には relationship DB backup が作成されます。誤保存があった場合は、明示的に restore します。

```bash
python -m relationship_lifelog_agent.cli --config config.local.yaml db backups
python -m relationship_lifelog_agent.cli --config config.local.yaml db restore \
  --backup-path "<backup_file.sqlite>"
```

rollback対象は relationship DB のみです。上流DBは read-only で扱うため、このアプリの restore では上流DBを変更しません。

不要な report を削除する場合も、内容に実データが含まれていないか確認してください。共有済みの report に private 内容が含まれていた可能性がある場合は、共有先からも削除し、以後は `counts-only` からやり直してください。

## 7. Privacy checklist

実データ運用前に、次を確認してください。

- `config.local.yaml` がgit管理外である。
- `app.host` が `127.0.0.1` である。
- `allow_external_api` が `false` である。
- `allow_model_auto_download` が `false` である。
- `allow_gradio_share` が `false` である。
- `adapter.backend` は目的に応じて `mock` または `upstream_readonly` である。
- `upstream_access_mode` が `readonly` である。
- `copy_raw_upstream_data` が `false` である。
- public mode で人物名、関係ラベル、本文excerpt、正確GPS、顔情報、写真実体、private path が出ない。
- `counts-only` report に本文excerptが含まれない。
- `redacted` report に人物名、関係ラベル、LINE本文、メモ本文、正確GPS、path が含まれない。
- `private` report はローカルの自分だけが確認する。
- `--write` は dry-run 結果と backup を確認した後だけ使う。
- review action は元データを削除せず、履歴として残る。
- 回答は喧嘩を断定せず「喧嘩候補」と表現する。
- 回答は相手の内心、恋人ラベル、親密度、関係状態をAI推定で断定しない。
