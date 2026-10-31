# osu! 強化學習研究紀錄

## 研究問題
只用「物件座標向量」當觀察，PPO 能否在程序生成的隨機譜面上學會 osu!standard 的瞄準與點擊時機，並遷移到真實譜面？

## 實驗設定（Exp 000–002 共通；之後的改變寫在各實驗裡）
- 環境：自製 Python 模擬器（[env.py](env.py)），60 Hz（每步 16.7 ms），playfield 512×384
- 判定：300/100/50 窗口照 stable 公式（OD），提早 400 ms 內點擊算 miss（lazer 規則），只有最早未判定的物件可點（notelock）
- 觀察（17 維）：游標位置、按鍵狀態、圓半徑、50 窗口；接下來 3 個「已出現」物件的相對位置 /256、距離打點時間 /200ms、存在旗標
- 動作：dx、dy（上限 80 px/步）、按鍵（>0 為按住，按下邊緣才算點擊）
- 獎勵：300=+1、100=+0.4、50=+0.1、miss=−1、空點 −0.05，加上朝目標物件靠近的距離 shaping（×1/512）
- 訓練資料：每回合隨機生成 40 個物件（BPM 120–240，1/1 與 1/2 節奏，CS 3–5.5、AR 7–10、OD 5–9）
- 評估：每 25 萬步在真實譜面 BAKENOHANA [What Has Been Lost]（CS4 / AR9.6 / OD9.2，569 物件）完整打一次，deterministic policy
- 已知限制：slider 當 circle（只算頭）、spinner 略過、沒有 stacking 偏移

## 實驗紀錄

### Exp 000 — smoke test（2026-10-31）
- PPO，MLP [256,256]，8 env，n_steps 1024，batch 2048，其餘 SB3 預設，1M 步（約 3 分鐘，~5100 fps）
- 結果（真實譜）：acc 65.41%，300/100/50/miss = 278/279/7/5，max combo 244
- 觀察：瞄準幾乎學會（miss 很少），但時機不準，100 佔一半；OD9.2 的 300 窗口只有 ±24.8 ms，約 1.5 步

### Exp 001 — baseline 5M（2026-10-31）
- 設定同 Exp 000，5M 步，seed 0
- 紀錄：`runs/*-baseline/`（config.json、progress.csv、curves.png、ckpt/、model.zip）
- 訓練時間 993 秒（~5000 fps），訓練回合獎勵從 −43 升到 ~48，約 3M 步後趨於平穩
- 最終模型（真實譜）：**acc 97.89%，300/100/50/miss = 551/18/0/0，max combo 569（FC）**
- 真實譜評估曲線（每 25 萬步）：

| 步數 | acc | 300 | 100 | miss |
|---|---|---|---|---|
| 0.25M | 46.7% | 181 | 213 | 91 |
| 0.5M | 39.7% | 55 | 513 | 0 |
| 1.0M | 83.4% | 427 | 142 | 0 |
| 1.25M | 97.4% | 547 | 22 | 0 |
| 2.5M | 68.7% | 302 | 267 | 0 |
| 3.25M | 97.5% | 548 | 21 | 0 |
| 4.25M | 88.3% | 469 | 100 | 0 |
| 5.0M | 97.9% | 551 | 18 | 0 |

- 觀察：
  1. 瞄準在 0.5M 步就學會：從那之後全程 0 miss、全部 FC，學習難點只剩點擊時機
  2. 時機不穩定：acc 在 69%–98% 之間大幅震盪，訓練獎勵卻已平穩。推測是 60 Hz 下 300 窗口（±24.8 ms）只有約 1.5 步，policy 稍微偏移就會整批從 300 掉到 100
  3. 只在單一譜面、單一 seed 評估，雜訊大，結論需要多譜面、多 seed 驗證
  4. 從隨機生成譜遷移到真實譜完全沒有落差：模型從沒看過真實譜，仍然 FC
- 下一步候選：提高模擬頻率（120/240 Hz）提升時機解析度；learning rate 衰減減少震盪；多譜面評估集；3 個 seed 取平均
- 未驗證：`bakenohana.osr` 尚未在 osu!lazer 實際播放

### Exp 002 — 120 Hz 模擬（2026-10-31）
- 假設：時機震盪來自 60 Hz 時間解析度太粗；120 Hz（每步 8.3 ms）讓 300 窗口變成約 3 步，acc 應更高更穩
- 為了只改「時間解析度」一個變因，其他換算成相同的實際時間：
  - 游標速度上限維持 4.8 px/ms（每步 40 px）
  - gamma 0.99 → 0.995，讓折扣的有效時間範圍維持約 1.67 秒
  - 訓練 10M 步 = 與 baseline 5M 步相同的遊戲時間；5M 步的點也在紀錄裡，兩種比較方式都能看
- 紀錄：`runs/20261031-0055-hz120/`（訓練 2106 秒，~4750 fps）
- 結果比較（真實譜 eval acc）：

| | 60 Hz（Exp 001） | 120 Hz（Exp 002） |
|---|---|---|
| 第一次 acc > 99% | 從未 | 0.5M 步（99.88%） |
| 2.5M–5M 步 mean ± std | 90.5% ± 8.5% | 98.7% ± 1.0% |
| 5M 步 | 97.9%，FC | 97.1%，FC |
| 10M 步（最終） | — | 96.6%，2 miss |
| 最佳 checkpoint | 97.9%（5M） | **100.00% SS（2.5M）** |
| 1M 步後總 miss 數 | 0 | 36 |

- 各 checkpoint 單獨重跑（120 Hz）：0.5M 99.88%、2.5M 100%（569 個全 300）、5M 97.07%、10M 96.63%
- 觀察：
  1. **假設成立**：120 Hz 讓時機學得更快、更穩，震盪幅度（std）降到約 1/8
  2. **新問題：偶發 miss**。60 Hz 從不 miss，120 Hz 在後期會在固定幾個物件 miss（#108、#379、#405 反覆出現），導致斷 combo
  3. miss 的共同點：都是緊接著一個 **超大跳**（下一個物件距離 321 / 255 / 355 px、間隔 167 ms）的前一顆。隨機譜生成器的跳躍上限是 300 px，**真實譜的跳超出訓練分布**；推測 agent 為了趕下一個大跳提早離開而沒點到
  4. **8M 步之後退化**：acc 從 ~98.5% 降到 ~96.6%，miss 變多，訓練獎勵卻沒下降 → 可能是固定 learning rate 造成的 policy 漂移，或對隨機譜過擬合
- 方法注意：「最佳 checkpoint = 100%」是在**同一張測試譜**上挑出來的，有選擇偏差，不能當成泛化能力的證據。之後需要分開「驗證譜（挑 checkpoint）」和「測試譜（回報成績）」
- Replay：`bakenohana_best.osr`（2.5M checkpoint，100%）、`bakenohana.osr`（最終模型）
- 下一步候選：
  1. 隨機譜跳躍上限提高到 ~450 px，覆蓋真實譜的大跳（直接針對 miss 原因）
  2. learning rate 線性衰減，解決後期退化
  3. 建立驗證譜 / 測試譜集，修正選擇偏差

### Exp 003 — 用 osu!lazer 官方判定驗證模擬器（2026-10-31）
- 目的：確認模擬器的判定和真正的 osu! 一致，replay 能被 lazer 正確讀取
- 方法：
  - clone ppy/osu（commit `845336e`，2026-10-30），.NET 10 SDK build
  - 在 `osu.Game.Rulesets.Osu.Tests/TestSceneOsuRLReplay.cs` 加一個 headless 測試：用 lazer 的 `LegacyScoreDecoder` 讀我們的 .osr、檢查 beatmap MD5、用 `ReplayPlayer` 完整播放，輸出每個判定（參考官方的 `LegacyReplayPlaybackTestScene`）
  - [verify.py](verify.py)：模型在模擬器打一次 → 寫 .osr → 丟進 lazer 判定 → 逐物件比對（circle 比 Great/Ok/Meh/Miss；slider 頭在 classic 模式只有 hit/miss）
  - 一張譜在 lazer 跑約 7 秒（headless 不受即時速度限制）
- 第一輪結果：
  - .osr 格式正確，lazer 可讀，MD5 相符，自動帶上 Classic mod（stable replay 行為）
  - 點擊時間誤差：模擬器 vs lazer 平均 0.23 ms，最大 0.33 ms（只差在 replay 時間取整到 ms）
  - **發現模擬器 bug**：120 Hz 最終模型有 29 顆 circle 不一致，全部是模擬器判 300、lazer 判 100，誤差都落在 23.7–24.3 ms
  - 原因：lazer 的判定窗口是 `floor(80 − 6·OD) − 0.5`（`OsuHitWindows.SetDifficulty`），OD 9.2 的 300 窗口是 ±23.5 ms，模擬器原本用 ±24.8 ms
- 修正：[beatmap.py](beatmap.py) 的 `windows` 改成 lazer 的取整方式
- 修正後結果（BAKENOHANA，569 個物件）：

| 模型 | 模擬器 acc（舊窗口 → 新窗口） | circle 一致 | slider 頭一致 |
|---|---|---|---|
| 120 Hz 2.5M checkpoint | 100% → 100% | 218/218 | 351/351 |
| 120 Hz 最終（10M） | 96.51% → **82.51%** | 218/218 | 351/351 |
| 60 Hz baseline 最終（5M） | 97.89% → 97.77% | 218/218 | 350/351 |

- 觀察：
  1. **修正後模擬器和 lazer 逐顆完全一致**（唯一例外見第 3 點）
  2. **120 Hz 最終模型在鑽漏洞**：它有 100 多顆剛好壓在 23.5–24.8 ms 這段「模擬器算 300、lazer 算 100」的區間。代表後期 policy 學會了「盡量晚按、壓在窗口邊緣」，這也可能是 Exp 002 後期退化的一部分原因。Exp 001 / 002 的 acc 都是用舊窗口算的，偏樂觀
  3. baseline 有 1 個 slider 頭不一致（#125）：它跟前一顆在**同一位置、只隔 28 ms**。lazer 會對疊在一起的物件做 stacking 位移，模擬器沒做；60 Hz 下單鍵連按也很勉強。屬於已知限制
  4. lazer 整體 acc 只有 58–68%，因為 slider 只點了頭、沒有跟住，slider tick 和尾端大量 miss，combo 最多 177。**要在真實 osu! 拿到好成績，下一步必須做 slider**
- 未做：在 lazer 遊戲畫面實際觀看 replay（headless 測試用的是同一套判定程式碼）

### Exp 004 — 加入 slider（2026-10-31）
- 模擬器實作（照 lazer 原始碼 + Classic mod 行為，因為我們的 .osr 是 stable 格式，lazer 會自動套 CL）：
  - 路徑：解析 `.osu` 的 L / P / B 曲線（B 依重複點切段，P 用三點求圓弧），裁切或延長到譜面寫的長度（`SliderPath.calculateLength`）
  - 時間：TimingPoints 的 beat length + 綠線 SV（`GetPrecisionAdjustedBeatLength`，clamp 10–1000），速度 = 100 × SliderMultiplier / 調整後 beat length
  - 巢狀物件：tick / repeat / tail 生成照 `SliderEventGenerator`（tick 距離、離尾端 10 ms 內不放 tick）
  - 追蹤：按住鍵 + 游標在 follow 範圍內（沒在追蹤時是 1 倍半徑，追蹤中擴大成 2.4 倍）；tick / repeat 在該時間點依追蹤狀態判 hit / miss；tail 可提早 36 ms 判 hit（`SliderInputManager`）
  - 頭：classic 模式下只有 hit / miss（50 窗口內），晚按時會一併結算已經過去的 tick（`PostProcessHeadJudgement`）
  - 整條 slider 判定（classic）：巢狀物件全中 = 300、≥ 50% = 100、> 0 = 50、全 miss = miss
  - combo：頭 / tick / repeat / 整條判定各 +1，tail 不影響 combo
  - 已知限制：catmull 曲線當直線處理（現代譜幾乎沒有）、沒有 stacking、spinner 仍然略過
- 驗證（oracle = 腳本寫的完美玩家，跟著 slider 球走）：

| | 模擬器 | lazer |
|---|---|---|
| acc（120 Hz） | 100.00% | 100.00% |
| max combo（120 Hz） | 1003 | 1003 |
| 逐項一致 | circle 218/218、slider 351/351、頭 351/351、tick 75/75、repeat 8/8、tail 351/351 | |
| 60 Hz | 99.88%、combo 795 | 99.88%、combo 795，同樣全部一致 |

  → **模擬器的 slider 判定和 lazer 完全一致**（這組驗證只涵蓋「全部追蹤成功」的路徑，追丟的情況要等訓練出的模型才能驗）
- 訓練設定的改變（相對 Exp 002）：
  - 觀察加入：每個即將出現物件的「是不是 slider」旗標；目前 slider 的 [進行中、是否在追蹤、剩餘時間、球在 +1 步 / +50 ms / +150 ms 的相對位置]，共 29 維
  - 獎勵加入：slider 頭 / tick / repeat / tail 中 +0.1、miss −0.3；整條 slider 用和 circle 一樣的 300/100/50/miss 獎勵；slider 進行中 shaping 目標改成球的位置
  - 隨機譜：40% 物件是 slider（L / P / B，長度 40–260 px，1–3 段，SV 0.6–1.6，tick rate 1 或 2）
  - 隨機譜跳躍上限 300 → 450 px（針對 Exp 002 大跳 miss 的問題，**這是第二個改變的變因**）
  - 游標可以超出 playfield 100 px（slider 路徑偶爾會出界，真實游標也不受 playfield 限制）
- 訓練：120 Hz、gamma 0.995、10M 步、seed 0。紀錄 `runs/20261031-0152-slider120/`
  - **中途停止**：決定先把所有遊戲機制做完（Exp 005）再訓練，這個 run 的環境已過時，結果不採用

### Exp 005 — 補齊所有打法 + 大量隨機比對（2026-10-31）
- 新增機制（全部照 lazer 原始碼）：
  - **Stacking**：`OsuBeatmapProcessor.ApplyStacking`（v6+ 新演算法與舊版），疊放物件位移 `stack × scale × −6.4`，slider 整條路徑一起移
  - **Spinner**：轉速用游標繞中心的角度變化（`SpinnerRotationTracker`），反向不累加（`SpinnerSpinHistory`），所需圈數依 OD（`Spinner.SpinsRequired`），完成度 ≥ 1 / > 0.9 / > 0.75 對應 300 / 100 / 50，每轉一圈一個 bonus tick
  - **Catmull 曲線**：`PathApproximator.CatmullToPiecewiseLinear` + lazer 的 `OptimiseCatmull`（包含它對長度的奇怪修正）
  - **雙鍵（K1 / K2）**：動作從 3 維變 4 維；slider 追蹤只認打頭的那個鍵，直到另一鍵放開過（`SliderInputManager.isValidTrackingAction`）
  - **Classic note lock**（`LegacyHitPolicy`）：前面還沒判完的物件會擋住後面的點擊；疊放的上層物件沒判完時點擊會穿到下一個；點擊必須 **嚴格小於** 400 ms
- 用比對找到並修正的細節（每一個都是 fuzz 抓到的）：
  1. 整條 slider 要到結束時間才判完（`DrawableSlider.CheckForResult`），影響 note lock 時機
  2. replay 時間是整數 ms，模擬器時鐘改成同樣的整數 ms（tail 的 −36 ms 邊界會差 0.3 ms）
  3. **lazer 會在 replay 兩幀之間更新遊戲**（`FrameAccuratePlayback` 預設關閉）：用上一幀的按鍵、內插的游標位置。會改變判定的時間點（tick、tail、slider / spinner 結束、circle 逾時）在 replay 裡補一個整數 ms 的固定幀，讓任何客戶端都在同一時間更新
  4. 同一次更新內，slider 的 tick / tail 先於頭判定（`DrawableSlider` 子物件順序），頭逾時 miss 後其他部分要等下一次更新
  5. 點擊時間差剛好 400 ms 時不算點到（`< hittableRange`）
- 驗證方法：[beatmap.py](beatmap.py) 的 `random_osu` 產生真正的 `.osu` 檔（含 stack、stream、spinner、四種 slider 曲線、紅點分段、裁切長度、變速），oracle 分別用無雜訊 / 中雜訊 / 大雜訊（游標亂抖、按鍵亂按）去打，產生大量 miss、斷追蹤、提早點擊，再逐項比對 lazer
- 5 張隨機譜 × 3 種雜訊：circle 645/645、slider 252/252、spinner 12/12、頭 252/252、tick 291/291、repeat 141/141、tail 252/252、spinner tick 12/12、max combo 15/15 **全部一致**
- 大量驗證（每張譜 × 無 / 中 / 大雜訊）：

| | 120 Hz（30 張） | 60 Hz（20 張） |
|---|---|---|
| circle | 3927/3927 | 2613/2616 |
| slider | 1451/1452 | 927/948 |
| spinner | 117/117 | 89/90 |
| 頭 / tick / repeat / tail | 1452/1452、1871/1872、878/879、1450/1452 | 948/948、1212/1218、567/570、932/948 |
| spinner tick 數 | 115/117 | 87/90 |
| max combo | 90/90 | 58/60 |
| 不一致比例 | 7 / 11268 ≈ **0.06%**，全部在大雜訊 | 53 / 7428 ≈ 0.7% |

- 剩下的不一致來自 lazer 在兩幀之間「不固定時間」的更新：追蹤狀態用內插的游標位置判斷，模擬器只在幀和固定插入點判斷。幀距越大（60 Hz 是 16.7 ms）越容易差，120 Hz 幾乎只在游標劇烈亂抖時發生
- 結論：**模擬器在 120 Hz 下和 lazer 判定幾乎完全一致**，之後訓練一律用 120 Hz

#### 補充：真實譜驗證
- 測試集：ppy/osu 測試資源裡的 22 張真實 ranked 譜、Renatus 的 3 個難度，再加上 BAKENOHANA，共 26 張（[get_maps.py](get_maps.py)）
  - 涵蓋 v3–v14 的譜面格式
  - CS 3–9、AR 2–9.6、OD 2–9.2
  - 共 21477 circle、17589 slider、177 spinner
- 方法同上：oracle 在 3 種雜訊下各打一次，逐項比對 lazer
- 比對抓到兩個 fuzz 沒抓到的差異：
  1. **v8 以前的譜，tick 間距不受 SV 影響**（lazer `OsuBeatmapConverter`：tick 距離再乘 1 / SV）。5 張舊譜有 46 個 tick 不一致，全部是這個原因
  2. **頭還沒判定時 slider 就在追蹤**（`SliderInputManager.Update` 從 slider 出現在畫面上就開始更新追蹤狀態，頭還沒判之前任何鍵都算；只有判定 tick / tail 時才要求頭已經判過）。模擬器原本要等頭判定後才開始追蹤
     - 情境：頭被 note lock 擋住、一直沒點到，玩家卻按著鍵跟著球走，最後頭逾時 miss
     - 這時 lazer 判 tail / repeat hit，模擬器判 miss
     - 隨機雜訊幾乎不會造成這種情況，是 Exp 006 訓練出的模型打真實譜時才大量出現（87 個不一致中佔絕大多數）

| 26 張真實譜 × 3 種雜訊 | 修正前 | 修正後 |
|---|---|---|
| circle | 21477/21477 | 21476/21477 |
| slider | 17570/17589 | 17575/17589 |
| spinner | 177/177 | 177/177 |
| 頭 / tick / repeat / tail | 17589、3335/3381、4141/4143、17578 | 17588、3406/3408、4141/4143、17578 |
| spinner tick 數 | 169/177 | 171/177 |
| max combo | 74/78 | 78/78 |
| 不一致 | 113 / 82149 | **37 / 82149 ≈ 0.045%** |

- 修正後新出現的 1 個 circle 和 1 個頭的不一致，用同一個 seed 單獨重跑後，新舊模擬器都和 lazer 完全一致
  - 代表這是 lazer headless 播放本身的不確定性：兩幀之間的更新時間每次不同
  - 不是模擬器的錯，也說明 lazer 自己在幀與幀之間的判定有少量隨機性
- 剩下的不一致幾乎都在大雜訊（游標每步亂抖）下，屬於上面說的兩幀之間更新問題
- 修正後 fuzz（30 張隨機譜 × 3 種雜訊）沒有退步：不一致從 7 個降到 4 個

### Exp 006 — 完整機制 120 Hz 訓練（2026-10-31）
- 環境：Exp 005 的完整模擬器（circle / slider / spinner / stacking / 雙鍵 / note lock），觀察 38 維、動作 4 維（dx、dy、K1、K2）
- 訓練資料：`random_osu` 隨機譜，每回合 40 個物件（4% spinner、6% 疊放、6% 連打、40% slider，其餘 circle；跳躍上限 450 px）
- 獎勵：circle / 整條 slider / spinner 300 = +1、100 = +0.4、50 = +0.1、miss = −1；slider 頭 / tick / repeat / tail 中 +0.1、miss −0.3；spinner 完成度 × 0.5；空點 −0.05；朝目標靠近的距離 shaping
- PPO：MLP [256, 256]、8 env、n_steps 1024、batch 2048、gamma 0.995、其餘 SB3 預設（固定 learning rate，和之前一樣，沒有加衰減）；20M 步、seed 0、約 4300 fps
- 評估（每 25 萬步）：
  - **驗證集**：5 張固定隨機譜（seed 10000–10004，各約 100 物件），用來挑 checkpoint
  - **測試譜**：BAKENOHANA，只用來回報成績，不用來挑 checkpoint（修正 Exp 002 的選擇偏差）
- 紀錄：`runs/20261031-0226-full120/`（約 1.5 小時）；整理後的結果在 [results/20261031-0226-full120/](results/20261031-0226-full120/results.md)
- 評估流程（[evaluate.py](evaluate.py)，跑在修正 pre-head 追蹤之後的模擬器上）：
  1. 41 個 checkpoint 在驗證集上打分，選出最佳 checkpoint：**15.5M 步**（驗證 acc 95.13%）
  2. 用最佳 checkpoint 打測試集，和 autoplay 對照：
     - 26 張真實譜
     - 50 張沒看過的隨機譜（seed 20000–20049，各 100 物件）
  3. 真實譜的 replay 全部丟進 lazer 重新判定
- 結果：

| 測試集 | 模型 acc（平均 / 中位數） | FC | 平均 miss | autoplay acc |
|---|---|---|---|---|
| 真實譜 26 張 | 93.16% / 97.06% | 0/26 | 13.6 | 100.00% |
| 　其中難度在訓練範圍內（16 張） | 97.25% / 97.79% | 0/16 | 9.6 | 100.00% |
| 　其中難度超出訓練範圍（10 張） | 86.60% / 92.07% | 0/10 | 19.9 | 100.00% |
| 隨機譜 50 張 | 94.48% / 94.69% | 0/50 | 3.4 | 99.96% |

- 「訓練範圍」指隨機譜的 CS 3–5.5、AR 7–10、OD 5–9
- 最好的幾張真實譜：
  - 1124896（1044 物件，CS3.8 AR8.7 OD7.5）：99.65%，miss 3
  - renatus_hard：99.52%，miss 2
  - 871815：99.25%，miss 1
  - 75858：99.21%，miss 2
- 最差的幾張：
  - 2768615（**CS9**，圓的半徑只有 CS4 的約 40%）：45.2%
  - BAKENOHANA（**OD9.2**，比訓練範圍嚴）：77.9%，90 miss
  - 20544（**AR2 / OD2**，物件提早 1.6 秒出現）：79.3%
- **模擬器 vs lazer**：模型打的 26 張 replay，27391 / 27409 個判定一致（**99.93%**）
  - lazer 顯示的 acc 和模擬器 acc 相差都在 ±0.5% 以內
  - 修正 pre-head 追蹤之前是 27322 / 27409（99.68%），見 Exp 005 補充
- 學習曲線（驗證集）：
  - 1.5M 步之前幾乎不會打（acc < 50%）
  - 2M 步跳到 89%，4.5M 步到 95%
  - 之後到 20M 步一直在 93.7%–95.1% 之間震盪，沒有再進步
- 觀察：
  1. **泛化到真實譜**：難度在訓練範圍內的真實譜，acc 97.3%，比隨機譜（94.5%）還高。這可能是因為隨機譜刻意塞了比較多的大跳、stream 和疊放
  2. **失敗幾乎都來自訓練分布外的難度**：CS 太大、AR / OD 太低或太高。agent 只看過固定範圍的 CS / AR / OD，觀察雖然有半徑和判定窗口，卻沒學會外插。下一步最直接的改進是擴大隨機譜的 CS / AR / OD 範圍
  3. **沒有任何一張 FC**：幾乎每張都有 miss，平均每 100 物件約 2.7 個（訓練範圍內的譜約 1.5 個）
     - 對照 Exp 001 / 002 只有 circle 時常常 FC，現在多了 slider 追蹤、spinner、雙鍵、note lock，難度高很多
     - BAKENOHANA 從 Exp 002 的 100%（slider 當 circle）掉到 77.9%，但兩者規則不同，不能直接比較
  4. **4.5M 步之後停滯**：固定 learning rate 加上 20M 步，驗證 acc 沒有再提升。之後可以試 LR 衰減、更大的網路，或更長的觀察（K > 3）
  5. **模型行為暴露模擬器 bug**：模型常在頭被 note lock 擋住時繼續追 slider，這是 fuzz 測不到、真實 policy 才會走到的狀態。訓練出來的 agent 本身也是很好的測試工具
- 下一步候選：
  1. 隨機譜擴大到 CS 2–7、AR 3–10.5、OD 2–10
  2. LR 線性衰減
  3. 觀察加入更多即將出現的物件（K = 3 → 5），處理 stream 和低 AR
  4. 用修正後的模擬器重新訓練：這個模型是在修正前的模擬器上訓練的，差異只影響少數 slider，但嚴格來說應該重訓
- 仍然沒做 / 已知限制：
  - HP 與失敗（replay 播放時 lazer 不會中途判失敗，但真的玩會）
  - Mods（HR / DT / HD 等）
  - 分數（只算判定、acc、combo，沒算 ScoreV1 / lazer 分數）
  - 落在 replay 幀前最後 1 ms 內的事件無法用固定幀鎖定，模擬器當作在該幀判定；高 FPS 客戶端可能提早一點判
  - spinner 位置不在 (256, 192) 的譜沒驗證過
