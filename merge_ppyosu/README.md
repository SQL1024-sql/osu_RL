# osu_rl x osu!lazer 直接整合模組 (OsuModRLBot)

本目錄包含將訓練好的強化學習模型（PPO）直接整合進 **ppy/osu (osu!lazer)** 的完整原始碼。

## 特色

1. **後台遊玩，零干擾**：
   - 作為 lazer 內部的 Automation Mod 運作，AI 動作直接注入遊戲引擎輸入管線。
   - **完全不搶佔 Windows 實體滑鼠或鍵盤**，你可以隨時切視窗做其他事、打字、寫程式。
2. **純 C# 原生極速推論（Zero Dependencies）**：
   - 包含自主實作的超輕量 MLP 推論引擎（`NeuralNetwork.cs`），推論單步僅需 ~5 微秒。
   - 不依賴任何額外 NuGet 套件（如 heavy 的 Python 或 C++ ONNX DLL），編譯即用，跨平台相容。
3. **無縫整合遊戲 UI**：
   - 選歌介面的 **Mod 選單（Automation / 自動化分類）** 裡會直接出現 **`RL Bot`**。
   - 點擊開啟後，開始歌曲即可觀賞 AI 遊玩。

---

## 包含檔案

| 檔案 | 說明 |
|---|---|
| `OsuModRLBot.cs` | 自訂 Mod 定義，繼承自 `ModAutoplay`，註冊於 Automation 分類 |
| `OsuRLBotGenerator.cs` | 讀取 lazer 的 `OsuBeatmap`，特徵化後餵入神經網路產生 Replay 幀 |
| `NeuralNetwork.cs` | 純 C# 矩陣相乘與 Tanh 前向傳播引擎 |
| `model_weights.bin` | 從最佳模型（`best_model_h6.zip`）匯出的純權重二進位檔（約 300 KB） |
| `install_to_ppy_osu.ps1` | 一鍵安裝/同步到 `..\ppy-osu` 的 PowerShell 腳本 |

---

## 一鍵安裝與編譯

在 PowerShell 中執行：

```powershell
# 1. 執行安裝腳本（將 Mod 原始碼與權重複製到 ppy-osu 並註冊）
powershell -ExecutionPolicy Bypass -File merge_ppyosu\install_to_ppy_osu.ps1

# 2. 編譯 osu!lazer
dotnet build ..\ppy-osu\osu.Desktop

# 3. 啟動 osu!lazer
& "..\ppy-osu\osu.Desktop\bin\Debug\net10.0\osu!.exe"
```

## 在遊戲中使用

1. 啟動 lazer 後進入選歌畫面。
2. 按 `F1` 或點擊底部的 **Mods** 選單。
3. 切換到 **Automation（自動化）** 分類。
4. 在 Autoplay / Cinema 旁邊會看到 **RL Bot (RL)**，點選它。
5. 點擊開始遊玩，AI 即會以 120Hz 高精度即時打譜！
