# 專案與編譯環境

本專案以 WSL2 Ubuntu 為唯一執行環境，避免 Windows Python、WSL Python 與
臨時 uv runtime 混用。

## Canonical environment

| 項目 | 版本／位置 |
| --- | --- |
| OS | Ubuntu 22.04.2 LTS on WSL2 |
| Kernel | Linux 6.6.87.2-microsoft-standard-WSL2 x86_64 |
| Python | 3.10.12，位於 .venv/bin/python |
| pip | 26.2.1 |
| NumPy | 2.2.6 |
| SciPy | 1.15.3 |
| scikit-learn | 1.7.2 |
| GCC / G++ | 11.4.0 |
| GNU Make | 4.3 |
| Git | 2.34.1 |
| CMake | 未安裝；目前流程不需要 |
| Berkeley ABC | 1.01，位於 .tools/abc |
| ABC source commit | 5ea34643247f9f45baa3bba40738832d7db19c60 |
| EPFL benchmark commit | 82d8cc6910419298e713a46644ed59fd3df53038 |

Python 的可重現套件版本保存在 requirements-dev.lock；專案宣告與相容版本範圍
仍以 pyproject.toml 為準。

## 目錄責任

| 路徑 | 用途 | 是否可重建 |
| --- | --- | --- |
| src/riskaware_eda | 正式 Python source | 否 |
| tests | 單元與 pipeline 測試 | 否 |
| configs | 研究設定 | 否 |
| scripts | 安裝、檢查與整理腳本 | 否 |
| data/epfl | 20 個小型 EPFL AIGER circuits | 是，make epfl |
| .tools/abc-src | ABC source 與 build objects | 是，make abc |
| .tools/abc | ABC 執行檔 symlink | 是，make abc |
| .venv | 唯一 Python virtual environment | 是，make bootstrap |
| artifacts/demo | Synthetic end-to-end 產生物 | 是，make demo |
| artifacts/smoke | 真實 ABC 工程驗證產生物 | 是 |
| artifacts/archive | 整理前的小型備份 | 視內容而定 |

## 日常命令

    make env
    make test
    make check
    make demo

第一次建立或環境損壞時：

    make bootstrap
    make abc
    make epfl

setup_abc.sh 與 fetch_epfl.sh 預設固定在上表 commits；若要刻意升級，可分別設定
ABC_REVISION 或 EPFL_REVISION，驗證後同步更新本文件。

## 整理決策

- 保留 .venv 作為唯一有效 Python 環境。
- 原本的 .venv-linux 與 .uv-python 缺少 Python 3.12 主執行檔，無法啟動；連同
  .uv-cache 封存至 artifacts/archive/obsolete-python-envs-20260821.tar.gz 後移除。
- ABC smoke test 與 EPFL leave-one-design-out smoke test 分別集中至
  artifacts/smoke/abc_runner 與 artifacts/smoke/epfl_lodo。
- pytest、coverage、egg-info、Python bytecode 與 abc.history 都屬可重建快取，
  由 scripts/organize_workspace.sh 清除。

## 注意事項

ABC 目前以 Makefile 直接編譯，因此 CMake 缺失不影響專案。若未來加入需要 CMake
的 native extension，再以系統套件安裝並更新本文件；不要為目前流程額外加入一套
編譯器環境。
