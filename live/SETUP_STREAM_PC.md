# 在直播機上架設 osu_rl

目標：直播機自己跑遊戲（我們的 lazer fork）、學生模型和自我改進迴圈，OBS 也在同一台推流。

## 需要什麼

- Windows 10 / 11，NVIDIA 顯示卡。打歌約要 2 GB 顯示記憶體；訓練新模型用 16 GB 卡跑得順，8–12 GB 要把 `--max-recs` 調小。
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

會建一個 `osu_rl` 場景，裡面有遊戲畫面、學生的視角、狀態文字。

## 注意

- **前景模式**：遊戲視窗一定要在最前面。有東西搶走焦點（通知、更新視窗）時，那首會放棄，迴圈會把遊戲叫回前面再繼續。直播機上不要開其他會跳出來的程式。
- **背景模式**：遊戲視窗不能縮小，被蓋住沒關係。
- 不要同時開正式版 lazer：OBS 和程式都可能抓錯視窗。
- 進度在 `runs/improve/`：`state.json` 是目前上場的模型，`rounds.jsonl` 是每輪結果。重開會從這裡接著跑。
