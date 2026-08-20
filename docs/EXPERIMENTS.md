# 可續跑實驗流程

`riskaware-eda experiment` 將真實 ABC 實驗拆成可驗證、可續跑的三個階段：

1. `collect`：依 `recipe seed × circuit` 建立獨立 CSV shard。
2. `train`：合併同一 recipe seed 的 shards，逐一進行 leave-one-circuit-out
   (LOCO) 訓練與 calibration。
3. `evaluate`：重播 held-out oracle，執行所有 budget 與 search seed 組合，最後
   產生彙總 CSV/JSON。

## 先跑 pilot

    .venv/bin/riskaware-eda experiment \
      --config configs/pilot_experiment.json \
      --resume

或使用：

    make pilot

Pilot 使用 5 個 EPFL circuits、24 條長度 6 的 recipes、1 個 recipe seed、兩種
budget 與兩個 search seeds。這是環境與完整資料流驗證，不應當作正式研究結論。

只驗證設定及估算工作量，不建立輸出：

    .venv/bin/riskaware-eda experiment \
      --config configs/experiment.json \
      --dry-run

## 續跑語意

輸出根目錄包含 `manifest.json`，記錄設定 fingerprint、Git revision/dirty state、
Python 與套件版本。使用 `--resume` 時：

- collection shard 必須同時存在 CSV 與成功 checkpoint，且檔案 signature 相符才
  會跳過；中斷或失敗的 shard 會重跑。
- 合併資料集、LOCO 模型與單一 simulation 都有各自的 checkpoint。
- 設定 fingerprint 不相容時會停止，避免把不同實驗意外混在同一目錄。
- 所有 metadata 以暫存檔後原子替換，降低中斷時留下半份 JSON 的機會。

可以只執行一個階段：

    .venv/bin/riskaware-eda experiment --config configs/experiment.json --phase collect --resume
    .venv/bin/riskaware-eda experiment --config configs/experiment.json --phase train --resume
    .venv/bin/riskaware-eda experiment --config configs/experiment.json --phase evaluate --resume

## 輸出結構

    artifacts/experiments/<name>/
      manifest.json
      recipes/
      shards/<recipe-seed>/*.csv
      datasets/<recipe-seed>.csv
      checkpoints/collect/<recipe-seed>/*.done.json
      checkpoints/datasets/*.done.json
      models/<recipe-seed>/holdout_*.joblib
      reports/training/<recipe-seed>/holdout_*.json
      simulations/<recipe-seed>/<holdout>/budget_*_search_*.json
      results.csv
      summary.json
      last_run.json

## 效能設定

目前 WSL 記憶體較小，因此 pilot 與正式設定預設：

- `collection.jobs: 2`：同時跑兩個 ABC circuit sessions。
- `training.model_jobs: 2`：限制 random forest 的平行 worker，避免 LOCO 訓練耗盡
  記憶體。
- collection 逐 trajectory 串流寫入 shard，不在記憶體保留整個實驗資料集。
- evaluation 對同一 holdout 只載入一次 model 與 oracle，再跑所有 budget/seed。
- 搜尋開始時將所有 candidate prefixes 合併成矩陣批次預測，避免逐 recipe、逐樹
  的 Python 呼叫成本。

若要調整 ABC 平行度，可在不改設定檔的情況下傳入 `--jobs N`；但同一輸出目錄
會以設定 fingerprint 保護，若要比較不同執行設定，請搭配新的 `--output` 目錄。
