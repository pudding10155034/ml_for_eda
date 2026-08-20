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
- Synthetic end-to-end demo 與單元測試。

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
- Baselines：random、full-evaluation random、mean-only greedy、Thompson/LCB、
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
      synthetic.py     可重現的開發用 oracle
      cli.py           命令列入口

## 目前邊界

- 這是研究 MVP，不宣稱已得到 DAC/ICCAD 等級實驗結果。
- 目前使用 bag/count、position 與 compact AIG statistics，尚未加入 graph
  embedding；這刻意符合低算力限制。
- ABC 每一步以獨立 batch process 執行並透過 AIGER state 銜接，量測包含 process
  startup overhead。所有方法使用相同 runner 時比較仍公平；正式論文可再加入
  persistent ABC process 量測作為系統優化。
- 真實 EPFL leave-one-circuit-out 結果必須在收集足夠 trajectories 後才能判斷。
