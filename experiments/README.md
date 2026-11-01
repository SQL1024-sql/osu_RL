# 實驗設定檔

每個 `.json` 是一個可以重跑的假設，用 [run.py](../run.py) 執行：

```bash
python run.py experiments/015-tail-miss.json --seed 2
```

## run.py 的參數

| 參數 | 預設 | 意思 |
|---|---|---|
| 設定檔路徑 | 必填 | 要跑哪個假設 |
| `--seed` | 必填 | 隨機種子。換數字就是同一個假設的另一次獨立訓練（隨機譜、網路初始值、探索都不同）。多人一起跑時，每個人各選一個沒跑過的 seed |
| `--device` | `hybrid` | `hybrid` = 收集資料用 CPU、更新用 GPU，需要 CUDA 版 PyTorch；沒有 GPU 用 `cpu` |
| `--steps` | 設定檔的值 | 覆蓋訓練步數，例如 `--steps 520000` 用來快速測試流程 |
| `--every` | `1000000` | 每隔幾步的 checkpoint 列入挑選 |
| `--dry-run` | 關 | 只印出訓練指令，不執行 |

輸出：`results/experiments/<id>/seed<N>/summary.json`（選到的 checkpoint、驗證分數、測試成績、完整設定）和 `model.zip`。

## 設定檔格式

| 欄位 | 意思 |
|---|---|
| `id` | 實驗名稱，也是輸出資料夾的名字 |
| `record` | 對應 [RESEARCH.md](../RESEARCH.md) 裡的哪一個實驗（完整的事先登記寫在那裡） |
| `baseline` | 拿來比較的基準實驗 `id` |
| `hypothesis` | 假設 |
| `train` | 傳給 `train.py` 的參數（見下表） |
| `criteria` | 成功標準：`metric`、`op`（`<` `<=` `>` `>=`）、`value`。用多個 seed 的平均判定，單一 seed 只能當參考 |

## train 的參數

| 參數 | 意思 | 來自 |
|---|---|---|
| `steps` | 訓練步數（20M 約 1 小時，Linux + `hybrid`） | |
| `hz` | 模擬器每秒幾步；120 和 replay 的幀率一致 | Exp 002 |
| `gamma` | 未來獎勵的折扣；0.995 在 120 Hz 下大約看 1.7 秒以內的後果 | Exp 002 |
| `diff` | 隨機譜的難度範圍：`narrow` = CS 3–5.5 / AR 7–10 / OD 5–9，`wide` = CS 2–7 / AR 3–10 / OD 2–10 | Exp 007 |
| `timing` | circle 點得越接近完美時間，最多多拿這麼多獎勵（誤差 25 ms 歸零） | Exp 009 |
| `spin_shaping` | spinner 期間把游標拉向中心 50 px 圓周的距離獎勵 | Exp 010 |
| `spin_free_press` | spinner 期間按鍵不扣空點分 | Exp 011 |
| `pool` | 真實譜庫資料夾（先用 `get_hf_maps.py` 下載） | Exp 012 |
| `pool_frac` | 多少比例的回合用真實譜，其餘用隨機譜 | Exp 012 |
| `tail_miss` | 漏 slider tail 的獎勵；不設定就和其他 slider 部分一樣是 −0.3 | Exp 015 |
| `eval_map` | 訓練中每 25 萬步拿來試打、記錄成績的譜；只用來記錄，不影響訓練 | |

其他 `train.py` 的參數（`--envs`、`--workers`、`--gamma` 等）也可以寫進 `train`，名稱把 `-` 換成 `_`。

## 成功標準可以用的指標

| 指標 | 意思 |
|---|---|
| `real.acc` | 26 張真實測試譜的平均 acc |
| `real.fc` | 26 張裡 FC 的張數 |
| `real.miss` | miss 總數（circle、slider、spinner） |
| `real.n100` | 100 的總數 |
| `real.head_miss` / `real.tail_miss` | slider 頭 / tail 漏掉的次數 |
| `real.spinner_miss` | spinner miss 的個數 |
| `hf.acc` / `hf.fc` / `hf.miss` | 60 張 HF 測試譜的平均 acc / FC 張數 / miss 總數 |
| `val.acc` / `val.miss` | 挑 checkpoint 用的 25 張驗證譜 |

## 新增一個假設

1. 在 RESEARCH.md 先寫好假設、操作變因、控制變因和成功標準。
2. 複製最接近的設定檔，只改一個變因，再把成功標準寫進 `criteria`。
3. 至少跑 2 個 seed，再比較平均。
