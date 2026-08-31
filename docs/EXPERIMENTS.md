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

- collection 每完成一條 recipe 就將完整 trajectory 寫入 `.csv.partial`，執行
  `flush`/`fsync` 並更新 `.progress.json`。中斷後會從下一條 recipe 繼續，不會
  重跑已確認寫入的 prefix。
- 恢復時會掃描 partial CSV 的最長合法連續 prefix；尾端半條 trajectory 或損壞
  CSV record 會被捨棄，但先前完整 recipes 仍保留。舊版 runner 留下的
  `.csv.tmp` 也可自動匯入。
- 正式 collection shard 必須同時存在 CSV 與成功 checkpoint，且 ABC、circuit、
  recipe 與檔案 signature 相符才會跳過；依賴變動時只重算受影響的工作。
- 合併資料集、LOCO 模型與單一 simulation 都有各自的 checkpoint。
- evaluate 會依 search seed 合併尚未完成的 budgets；若只缺部分 budget，僅補齊
  缺少的 JSON，既有相容結果不會覆寫。
- 新產生的 simulation JSON 另記錄實際 oracle shard 的 path/signature；舊版 JSON
  仍可續用，但新版 artifact 會直接驗證 oracle 來源。
- 設定 fingerprint 不相容時會停止，避免把不同實驗意外混在同一目錄。
- 所有 metadata 以暫存檔後原子替換，降低中斷時留下半份 JSON 的機會。
- `run.lock` 使用作業系統 advisory lock；同一輸出目錄同時只能有一個 runner，
  程序異常結束時鎖會由作業系統自動釋放。

可以只執行一個階段：

    .venv/bin/riskaware-eda experiment --config configs/experiment.json --phase collect --resume
    .venv/bin/riskaware-eda experiment --config configs/experiment.json --phase train --resume
    .venv/bin/riskaware-eda experiment --config configs/experiment.json --phase evaluate --resume

## 輸出結構

    artifacts/experiments/<name>/
      manifest.json
      recipes/
      shards/<recipe-seed>/*.csv
      shards/<recipe-seed>/*.csv.partial        # 執行中才存在
      datasets/<recipe-seed>.csv
      checkpoints/collect/<recipe-seed>/*.done.json
      checkpoints/collect/<recipe-seed>/*.progress.json
      checkpoints/datasets/*.done.json
      models/<recipe-seed>/holdout_*.joblib
      reports/training/<recipe-seed>/holdout_*.json
      simulations/<recipe-seed>/<holdout>/budget_*_search_*.json
      results.csv
      summary.json
      last_run.json
      run.lock

## 效能設定

目前 WSL 記憶體較小，因此 pilot 與正式設定預設：

- `collection.jobs: 2`：同時跑兩個 ABC circuit sessions。
- `training.model_jobs: 2`：限制 random forest 的平行 worker，避免 LOCO 訓練耗盡
  記憶體。
- collection 逐 trajectory 串流寫入並 durable checkpoint，不在記憶體保留整個
  實驗資料集；每 10 條 recipes 輸出一次可見進度。
- evaluation 對同一 holdout 只載入一次 model 與 oracle，並共用一份 prediction
  cache：500 個 start states 只批次預測一次，相同 `(recipe_id, step)` prefix 也只
  呼叫模型一次。
- oracle 直接從該 circuit 的 collection shard 載入，避免每個 holdout 都重新掃描
  同一份 merged seed dataset；dataset checkpoint 仍負責驗證 shard signatures。
- 相同 search seed 的多個 budgets 共用一條搜尋軌跡；runner 在每個 budget 邊界、
  下一輪 elimination 之前保存 snapshot，因此 `candidates_eliminated` 與
  `termination_reason` 仍和各 budget 獨立執行完全一致。
- cache 僅限單一 recipe seed、holdout circuit 與 model，絕不跨 holdout 共用。

可重複執行以下唯讀 benchmark，比較獨立執行、共用 cache，以及 cache 加
multi-budget snapshots 三種路徑；`--verify-artifacts` 會要求所有巢狀結果逐欄
等於既有 simulation JSON：

    .venv/bin/python scripts/benchmark_evaluation.py \
      --config configs/pilot_experiment.json \
      --experiment-dir artifacts/experiments/epfl_pilot \
      --repeats 3 \
      --verify-artifacts

目前完整 model 單份約數百 MiB，請勿以 process pool 同時載入多個 holdout；若要
再平行化，應先對同一模型的 tree predictions 測試 2–4 threads，避免 swap。

若要調整 ABC 平行度，可在不改設定檔的情況下傳入 `--jobs N`；但同一輸出目錄
會以設定 fingerprint 保護，若要比較不同執行設定，請搭配新的 `--output` 目錄。

## 結果稽核與報告

正式 evaluate 完成後，先以唯讀分析器檢查完整格網並產生技術報告：

    .venv/bin/python scripts/analyze_results.py \
      --simulation-root artifacts/experiments/epfl_full/simulations \
      --output-dir artifacts/analysis/epfl_full \
      --config configs/experiment.json \
      --bootstrap-reps 1000

分析器會重新讀取每個 JSON，檢查 method × recipe seed × holdout circuit × budget ×
search seed 的唯一性與完整性，並計算平均值、中位數、P05/P95、標準差與
deterministic percentile bootstrap 95% CI。`validation.md` 會把結構檢查、計算 spot
check、圖表 QA 與必須揭露的 caveat 分開記錄。圖表使用離散 budget 的 grouped bars；
不將少量 budget anchor 誤畫成時間序列。

## Baseline 與 ablation

`configs/ablation_pilot.json` 是可直接執行的六策略 pilot：

    .venv/bin/python scripts/run_ablations.py \
      --settings configs/ablation_pilot.json \
      --resume

正式 20-circuit × 10 recipe-seed × 3 budget × 10 search-seed 的設定已另存為
`configs/ablation_full.json`。它會產生 36,000 個 policy cells；建議先跑 pilot、確認
live validation 與磁碟/時間預算後再啟動，且一定使用獨立的 `output_dir`：

    .venv/bin/python scripts/run_ablations.py \
      --settings configs/ablation_full.json \
      --resume

策略定義如下：

- `risk_aware`：LCB selection + safe elimination + early stopping。
- `random`：隨機 selection，停用兩個 risk control。
- `mean_greedy`：依預測 mean selection，停用兩個 risk control。
- `lcb_only`：LCB selection，停用 elimination 與 early stopping。
- `selection_only`：LCB selection + safe elimination，停用 early stopping。
- `early_stop_only`：隨機 selection + early stopping，停用 safe elimination。

每個 cell 的 artifact 都記錄 settings fingerprint、model/oracle signature 與 method；
若只缺某些 budgets，`--resume` 只補缺少的 JSON。ablation 完成後可用同一分析器：

    .venv/bin/python scripts/analyze_results.py \
      --simulation-root artifacts/ablations/epfl_pilot \
      --output-dir artifacts/analysis/epfl_pilot_ablation \
      --config configs/pilot_experiment.json \
      --methods risk_aware,random,mean_greedy,lcb_only,selection_only,early_stop_only \
      --recipe-seeds 7 --circuits adder,bar,cavlc,ctrl,router \
      --budgets 5,10 --search-seeds 0,1

## Live ABC validation

`configs/live_validation_pilot.json` 會把一個 adder/budget=3/search seed=0 實際送進
ABC，並把 live best QoR 與相同 recipe shard 的 exhaustive oracle 對照：

    .venv/bin/python scripts/validate_live.py \
      --settings configs/live_validation_pilot.json \
      --resume

live output 與 offline output 分開保存，且每個 cell 都驗證 circuit、recipe、model
signature。若 live 與 replay 差異很大，先檢查 ABC binary、檔案 provenance 與量測雜訊，
再決定是否需要重新收集資料或訓練模型。

Live runner 也支援同一個 cell 的策略比較與重複量測。六策略 pilot：

    .venv/bin/python scripts/validate_live.py \
      --settings configs/live_validation_ablation_pilot.json \
      --resume

重複三次的 timing-noise pilot：

    .venv/bin/python scripts/validate_live.py \
      --settings configs/live_validation_repeated_pilot.json \
      --resume

每個新版 artifact 都記錄 `method` 與 `repeat`，路徑為
`<seed>/<circuit>/method_<name>/repeat_<index>/budget_*_search_*.json`；舊版單一
risk-aware artifact 仍可在預設設定下續跑並被安全升級。`summary.json` 的
`expected_runs` 會計入 methods 與 repeats，`results.csv` 則可直接依 method/repeat
分組比較 `relative_gap_pct`、`live_wall_s` 與 early-stop 行為。

使用 `scripts/analyze_live.py` 可稽核 CSV 格網並輸出 `report.json`、`report.md`。
它會針對每個 method × budget 計算 gap/wall-time 的 bootstrap 95% CI，並在有
`repeats >= 2` 時計算每個 live cell 的 wall-time coefficient of variation：

    .venv/bin/python scripts/analyze_live.py \
      --results artifacts/live_validation/epfl_pilot_repeated/results.csv \
      --output-dir artifacts/analysis/live_epfl_pilot_repeated \
      --settings configs/live_validation_repeated_pilot.json

這份報告的 repeat 統計只描述目前硬體、ABC binary 與 circuit 的量測噪聲，不能
直接外推到不同執行環境。
