# Risk-Aware Budgeted Search for Logic Synthesis

這個專案把研究規劃落成一個可執行的 MVP：使用公開 EPFL combinational
benchmarks 與 Berkeley ABC，在有限 synthesis evaluation 預算下尋找接近最佳的
logic-synthesis recipe，並以具校準區間的 surrogate model 同時決定：

1. 下一條最值得執行的 recipe。
2. 一條正在執行的 recipe 是否已不可能打敗 incumbent，因而可以提前停止。

整套流程只需要 CPU，不依賴 OpenROAD、商用 EDA 或大型現成資料集。

## 已實作範圍

- 可重現的隨機 candidate recipe 產生器。
- ABC AIGER/BLIF/Verilog runner；每個 operator 後保存狀態並擷取
  PI、PO、AIG nodes、logic depth 與 CPU wall time。
- Prefix trajectory CSV；每個中間狀態都以完整 recipe 的 final QoR 為 label。
- Cross-circuit random-forest surrogate。
- Trajectory-level split-conformal calibration：每個 calibration trajectory 先取所有
  prefixes 的最大 normalized residual，再校準區間。這比逐列隨機切分更符合
  sequential early stopping。
- Optimistic lower-bound selection、safe elimination 與 risk-aware early stopping。
- Held-out circuit oracle replay，可在不用重新執行 ABC 的情況下快速做 ablation。
- 具 recipe 級 durable checkpoint、設定 fingerprint、single-run lock 與原子
  metadata 的可續跑實驗 runner。
- 評估時以 holdout-scoped cache 共用 start/prefix predictions，並在一次搜尋中
  精確保存多個 budget snapshots；結果與獨立執行各 budget 完全等價。
- Synthetic end-to-end demo 與單元測試。
- 結果稽核器：驗證 simulation 格網、重新計算 bootstrap 95% CI，並輸出含 SVG
  圖表的技術報告與 validation report。
- 可續跑的 baseline/ablation runner：同一批 oracle shards 可比較 random、mean-only
  greedy、LCB-only、selection-only、early-stop-only 與完整 risk-aware policy。
- 獨立的 live-ABC validation runner：使用已完成的 model/recipe artifacts，小規模驗證
  真實 ABC 與 offline replay 是否一致。
- Live validation 支援六種策略與重複量測；每個 method × repeat × circuit × budget ×
  search seed cell 都有獨立 checkpoint，可估計真實 ABC 的時間噪聲與策略差異。

## QoR 定義

目前 scalar loss 為：

    0.5 * final_nodes / initial_nodes
      + 0.5 * final_depth / initial_depth

數值越小越好，原始網路為 1.0。研究時應同時回報 nodes、depth 與 Pareto
結果；scalar loss 主要用於 budgeted selection 的明確決策。

## 安裝

在 WSL/Ubuntu 執行：

    bash scripts/bootstrap_python.sh

目前唯一正式環境與 compiler 版本請見
[docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)。常用入口也可直接使用：

    make env
    make test
    make check

先跑不需 ABC 的完整 smoke test：

    .venv/bin/riskaware-eda demo --output artifacts/demo

輸出包含 recipes、synthetic trajectory dataset、校準後模型、training report 與
held-out circuit simulation。

真實 ABC 小規模 pilot 可直接執行，若中斷後重下同一指令會從最後完成的 recipe、
模型或 simulation 繼續：

    make pilot

完整研究設定先用 dry-run 檢查工作量：

    make experiment-plan

runner 的輸出結構、checkpoint 判定與分階段指令請見
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。

評估效能可用不修改 artifacts 的 benchmark 檢查：

    .venv/bin/python scripts/benchmark_evaluation.py \
      --config configs/pilot_experiment.json \
      --experiment-dir artifacts/experiments/epfl_pilot \
      --verify-artifacts

正式結果完成後，可執行結果稽核與技術報告（不會改寫正式 experiment artifacts）：

    make analyze-full

輸出在 `artifacts/analysis/epfl_full/`，包括 `report.md`、`validation.md`、
`report.json`、`analysis_rows.csv` 與三張帶 bootstrap 誤差線的 SVG 圖。六種策略的
離線 pilot ablation：

    make ablation-pilot

完整 sweep 只需把 `configs/ablation_pilot.json` 的 recipe seeds、circuits、budgets
與 search seeds 換成正式矩陣，並指定新的 `output_dir`；每個 cell 都有簽章與原子
checkpoint，可中斷後以 `--resume` 繼續。小規模 live ABC 驗證：

    make live-pilot

比較六種策略的實際 ABC 行為（預設只跑 adder、budget 3）：

    make live-ablation-pilot

重複同一個 risk-aware cell 三次，以估計 ABC wall-time 變異：

    make live-repeat-pilot

live 結果可再產生策略／重複量測報告：

    make analyze-live-ablation
    make analyze-live-repeat

live 結果獨立寫入 `artifacts/live_validation/`，不會覆蓋 offline simulation。

## 安裝 Berkeley ABC 與取得小型 EPFL 子集

    bash scripts/setup_abc.sh
    bash scripts/fetch_epfl.sh

setup_abc.sh 會從官方 berkeley-abc/abc repository 編譯工具。
fetch_epfl.sh 使用 sparse checkout，只取 arithmetic 與 random/control 的 AIGER
檔案，不包含三個超過一千萬 gates 的 MtM circuits。

來源：

- https://github.com/berkeley-abc/abc
- https://www.epfl.ch/labs/lsi/page-102566-en-html/benchmarks/
- https://github.com/lsils/benchmarks

## 真實資料流程

先建立 300 條、長度 10 的 recipes：

    .venv/bin/riskaware-eda recipes --count 300 --length 10 --seed 7 --output artifacts/recipes.json

以 EPFL 小型 circuits 收集完整 trajectories。初期建議從 8–15 個 circuits、
100–300 recipes 開始：

    .venv/bin/riskaware-eda collect --abc .tools/abc --circuits data/epfl/arithmetic/adder.aig data/epfl/arithmetic/bar.aig data/epfl/random_control/cavlc.aig --recipes artifacts/recipes.json --jobs 2 --output artifacts/trajectories.csv

訓練時必須將最終測試 circuit 完全排除。以下把 adder 留作 unseen test，
其餘 circuits 再做 circuit-level train/calibration split：

    .venv/bin/riskaware-eda train --dataset artifacts/trajectories.csv --exclude-circuit adder --alpha 0.01 --output artifacts/model.joblib --report artifacts/training_report.json

先用已收集資料模擬 budget 20：

    .venv/bin/riskaware-eda simulate --dataset artifacts/trajectories.csv --model artifacts/model.joblib --circuit adder --budget 20 --output artifacts/adder_simulation.json

最後才在一個新 circuit 上做 live search：

    .venv/bin/riskaware-eda search --abc .tools/abc --circuit data/epfl/arithmetic/adder.aig --recipes artifacts/recipes.json --model artifacts/model.joblib --budget 20 --output artifacts/adder_live_search.json

## 演算法決策

對 minimization 問題，模型為每個 recipe prefix 輸出：

    predicted final QoR, lower bound, upper bound

Selection 使用最小 lower bound，優先測試仍可能成為最佳解的候選。

當已有 incumbent 時：

- 若未執行 candidate 的 lower bound 高於 incumbent，直接淘汰。
- recipe 執行至少兩步後，若其 prefix 預測 lower bound 高於 incumbent，提前停止。
- 第一條 recipe 一定完整執行，確保有可比較的 incumbent。

這裡的 alpha 是在 exchangeable calibration trajectories 假設下的有限樣本
conformal 風險目標，不是對任意 distribution shift 的形式化保證。論文實驗應
明確檢查 held-out circuits 的 empirical simultaneous coverage，並對 circuit
family shift 分層報告。

## 建議實驗矩陣

- Split：Leave-One-Circuit-Out，不使用 row-level random split。
- Budget：10、20、50 recipes。
- Baselines：random、full-evaluation random、mean-only greedy、LCB、
  risk-aware selection only、early stopping only、兩者結合。
- 主要結果：best QoR vs. total CPU seconds。
- 次要結果：gap-to-oracle、top-1% hit rate、early-stop rate、錯殺最佳 recipe
  次數、row coverage、simultaneous trajectory coverage。
- Seeds：至少 10 個 recipe/search seeds。
- 額外 ablation：alpha、minimum stop step、recipe length、node/depth 權重。

預設研究設定也保存在 configs/experiment.json。

## 專案結構

    src/riskaware_eda/
      abc_runner.py    ABC 執行與 print_stats 解析
      recipes.py       recipe schema 與生成
      dataset.py       trajectory CSV
      features.py      circuit/recipe/prefix 特徵
      model.py         forest surrogate + conformal calibration
      search.py        selection、elimination、early stopping
      simulation.py    held-out oracle replay
      experiment.py    可續跑的 sharded experiment runner
      synthetic.py     可重現的開發用 oracle
      cli.py           命令列入口
    scripts/
      analyze_results.py  格網稽核、bootstrap 統計、SVG 與 Markdown 報告
      analyze_live.py     live ABC 策略比較與重複 timing-noise 報告
      run_ablations.py    可續跑 baseline/ablation sweep
      validate_live.py    真實 ABC live validation（策略與重複量測）

## 目前邊界

- 這是研究 MVP，不宣稱已得到 DAC/ICCAD 等級實驗結果。
- 目前使用 bag/count、position 與 compact AIG statistics，尚未加入 graph
  embedding；這刻意符合低算力限制。
- ABC 每一步以獨立 batch process 執行並透過 AIGER state 銜接，量測包含 process
  startup overhead。所有方法使用相同 runner 時比較仍公平；正式論文可再加入
  persistent ABC process 量測作為系統優化。
- 真實 EPFL leave-one-circuit-out 結果必須在收集足夠 trajectories 後才能判斷。
