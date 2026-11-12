# 在直播機上架設 osu_rl

目標：直播機跑遊戲（我們的 lazer fork）和學生模型，OBS 也在同一台推流。

兩種分工：

- **直播機只打歌，訓練在原本那台**（`live.stream`）：直播機只要約 2–3 GB 顯示記憶體。新模型由原本那台不定時送過來，丟進資料夾就換上。見「只打歌模式」。
- **全部在直播機**（`live.improve`）：打歌、訓練、換模型都自己來，訓練要 16 GB 的卡才順。

## 需要什麼

- Windows 10 / 11，NVIDIA 顯示卡。只打歌約要 2–3 GB 顯示記憶體；在這台訓練要 16 GB（8–12 GB 要把 `--max-recs` 調小）。
- 螢幕最好 120 Hz 以上（學生每秒決定 120 次）。不接螢幕的話，要插顯示卡欺騙器（HDMI dummy plug），否則擷取畫面只有 30 fps。
- 硬碟約 3 GB，加上錄影：每首歌約 20 MB。

## 步驟

### 1. 在原本那台做包

```bash
python -m live.bundle
```

會產生 `runs/bundle/osu_rl_bundle.zip`，大約 1 GB。裡面有：

- 程式碼
- 譜、skin、老師的錄影、訓練資料
- 目前的模型和迴圈進度
- 打包好的 fork：直播機不用裝 .NET

### 2. 在直播機解壓

例如解壓到 `C:\osu_rl`，資料夾裡要有 `fork\osu!.exe`。

### 3. 裝 Python 3.11 和套件

在 `C:\osu_rl` 開 PowerShell：

```bash
py -3.11 -m venv .venv-gpu
```

```bash
.venv-gpu\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
```

```bash
.venv-gpu\Scripts\python -m pip install -r requirements.txt
```

RTX 50 系列要 cu128 以上的 torch。舊一點的卡也可以用 cu126。

### 4. 裝虛擬繪圖板驅動（VMulti）

只有前景模式要用，背景模式可以跳過。

1. 從 <https://github.com/X9VoiD/vmulti-bin/releases/latest> 下載並解壓。
2. 以系統管理員身分執行 `install_hiddriver.bat`。
3. 重開機。

### 5. 電源設定

設定 → 系統 → 電源：插電時「螢幕關閉」和「睡眠」都改成**永不**。

另外把 Windows Update 的「使用時段」設好，避免它半夜自動重開機。

### 6. 檢查

```bash
.venv-gpu\Scripts\python -m live.doctor
```

會列出缺什麼、怎麼補。全部 `[OK]` 就可以開始。

### 7. 開始

#### 只打歌模式（訓練在原本那台）

```bash
.venv-gpu\Scripts\python -u -m live.stream --foreground
```

一首接一首打，不訓練。一開始用包裡的模型（做包時迴圈正在用的那個）。去掉 `--foreground` 就是背景模式。

**換新模型**：

1. 原本那台執行下面這行。它會把目前上場的模型複製到 `runs\stream\outbox\`，檔名是輪數，例如 `r007.pt`，約 11 MB：

   ```bash
   .venv-gpu\Scripts\python -m live.stream --export
   ```

2. 用 Discord、雲端硬碟或隨身碟傳到直播機。
3. 放進直播機的 `runs\stream\new\` 資料夾。

直播機每首歌開始前會檢查這個資料夾：

- 檔案沒問題：下一首就換上，原檔搬到 `runs\stream\models\`。
- 不是模型、或檔案壞了：搬到 `runs\stream\bad\`，舊模型繼續打。

目前上場的是哪個，看 `runs\stream\state.json`，直播畫面的狀態文字也會寫。直播機重開後會接著用同一個模型。

（選用）加 `--record` 會保存每首的錄影（每首約 130 MB，最多 50 GB），帶回原本那台放進 `runs\stream\plays\` 也能拿來訓練。

#### 全部在直播機

前景模式：遊戲在最前面全螢幕，用虛擬繪圖板打，最像真人在玩。

```bash
.venv-gpu\Scripts\python -u -m live.improve --foreground
```

背景模式：遊戲可以被其他視窗蓋住，輸入直接送進遊戲。

```bash
.venv-gpu\Scripts\python -u -m live.improve
```

第一次會自己開 fork、略過 lazer 的新手設定、匯入 skin 和譜。

### 8. OBS

OBS 28 以上：工具 → WebSocket 伺服器設定 → 啟用（埠 4455）。然後執行：

```bash
.venv-gpu\Scripts\python -m live.obs_setup --switch
```

會建一個 `osu_rl` 場景，裡面有遊戲畫面、學生的視角、狀態文字。只打歌模式要多加 `--stream`，狀態文字才會讀對檔案：

```bash
.venv-gpu\Scripts\python -m live.obs_setup --switch --stream
```

## 注意

- **前景模式**：遊戲視窗一定要在最前面。有東西搶走焦點（通知、更新視窗）時，那首會放棄，迴圈會把遊戲叫回前面再繼續。直播機上不要開其他會跳出來的程式。
- **背景模式**：遊戲視窗不能縮小，被蓋住沒關係。
- 不要同時開正式版 lazer：OBS 和程式都可能抓錯視窗。
- 只打歌模式的紀錄在 `runs/stream/`：`plays/` 是每首的分數，`state.json` 是目前上場的模型。
- 全部在直播機時，進度在 `runs/improve/`：`state.json` 是目前上場的模型，`rounds.jsonl` 是每輪結果。重開會從這裡接著跑。
