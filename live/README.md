# AI 玩家（接 osu! 私人伺服器的 lazer 多人房）

AI 用一個普通帳號，以 lazer 客戶端的協定連上 [Illyaverse](https://github.com/Illyavrs/Illyaverse)（或任何相容的 lazer 伺服器），跟真人在同一個多人房裡打。不用改伺服器，也不用模擬滑鼠鍵盤。

```
Discord 機器人 (TS) --tool call--> live/service.py (HTTP) --> live/player.py --SignalR / REST--> osu! 伺服器 <-- 玩家的 lazer
                                                          \-> live/prepare.py: 模擬器打譜 + headless lazer 算分
```

## 一場怎麼跑

1. 進房後，AI 下載房間目前的譜（`/osu/{id}`，會核對 MD5），用模型在模擬器打一次，再讓 headless lazer（[lazer/TestSceneOsuRLLive.cs](../lazer/TestSceneOsuRLLive.cs)）用房間的 mod 重播並計分，大約 10–30 秒。完成後 AI 自動準備。
2. 房主開始後，AI 跟一般 lazer 客戶端一樣：
   - 拿 score token；
   - 進入 Loaded → ReadyForGameplay；
   - 遊戲開始後，照真實時間推送每 200 ms 一包的游標和分數（lazer 的 `FrameDataBundle`）；
   - 結束時送出成績，切到 FinishedPlay。
3. 房裡其他人的即時排行看得到 AI，也能觀戰 AI 的游標。結算時 AI 的成績也在房間成績裡。
4. 換譜或換難度時，AI 會自動重新準備，每一輪都會自動再準備。

分數和判定都是 lazer 自己的計分程式算出來的，跟真人用 lazer 打出來的格式和數字完全一樣。

## 需要的東西

- **Python**：專案的套件，加上 `aiohttp`（已寫進 `requirements.txt`）。
- **.NET 10 和編好的 ppy-osu**：照主 README 的「驗證」步驟。另外把 `lazer/TestSceneOsuRLLive.cs` 複製到 `ppy-osu/osu.Game.Rulesets.Osu.Tests/`，然後執行 `dotnet build osu.Game.Rulesets.Osu.Tests`。ppy-osu 的位置用 `OSU_REPO` 環境變數指定，預設是 `../ppy-osu`。
- **伺服器上的 AI 帳號**：在網站正常註冊就好。

## 設定（`live/.env` 或環境變數；`live/.env` 不會進 git）

| 變數 | 意思 |
|---|---|
| `OSU_RL_SERVER` | 伺服器網址，例如 `https://shdcloud.top`，本機測試用 `http://127.0.0.1:3000` |
| `OSU_RL_AI_USERNAME` / `OSU_RL_AI_PASSWORD` | AI 帳號 |
| `OSU_RL_CONTROL_TOKEN` | 控制 API 的金鑰（16 字以上），Discord 機器人呼叫時帶 `Authorization: Bearer <token>` |
| `OSU_RL_TESTER_USERNAME` / `OSU_RL_TESTER_PASSWORD` | 只有 `live.e2e` 測試要用 |

## 啟動

```bash
python -m live.service --port 8765
```

只聽 `127.0.0.1`。如果 Discord 機器人在別台電腦，用 `--host 0.0.0.0`，前面再加 HTTPS 反向代理。

## 給 Discord 機器人的接口

[tools.json](tools.json) 是 OpenAI function calling 格式的 tool 定義，TS 那邊直接載入，放進 `tools` 就能用。每個 tool 對應一個 HTTP 端點：

| tool | HTTP | body | 回傳 |
|---|---|---|---|
| `osu_ai_status` | `GET /status` | | 所在房間、難度、是否已準備、是否在打、上一場結果 |
| `osu_ai_list_rooms` | `GET /rooms` | | `[{id, name, host, has_password, participants}]` |
| `osu_ai_join_room` | `POST /join` | `{room_id, password?, level?}` | 同 status |
| `osu_ai_leave_room` | `POST /leave` | | 同 status |
| `osu_ai_set_level` | `POST /level` | `{level}` | 同 status |
| `osu_ai_play_map` | `POST /play` | `{beatmap_id, level?, mods?}` | acc、rank、分數、combo、判定統計、`replay_url` |

`replay_url`（`/replays/<id>.osr`）不用金鑰就能下載，機器人可以直接把檔案貼到 Discord。

TS 範例（tool 名稱轉成 HTTP 呼叫）：

```ts
import tools from "./tools.json";
const ROUTES: Record<string, [string, string]> = {
  osu_ai_status: ["GET", "/status"], osu_ai_list_rooms: ["GET", "/rooms"], osu_ai_join_room: ["POST", "/join"],
  osu_ai_leave_room: ["POST", "/leave"], osu_ai_set_level: ["POST", "/level"], osu_ai_play_map: ["POST", "/play"],
};
async function callOsuTool(name: string, args: object) {
  const [method, path] = ROUTES[name];
  const r = await fetch(`${process.env.OSU_AI_URL}${path}`, {
    method, headers: { Authorization: `Bearer ${process.env.OSU_AI_TOKEN}`, "Content-Type": "application/json" },
    body: method === "POST" ? JSON.stringify(args) : undefined,
  });
  return r.json(); // 把這個當成 tool 的結果交回給模型
}
```

## 難度

| level | 模型 | H6 驗證 acc |
|---|---|---|
| `easy` | Exp 012 的 1M 步 checkpoint | 72% |
| `normal` | 2M 步 | 97.5% |
| `hard` | 5M 步 | 98.2% |
| `expert` | 12M 步（目前最佳） | 99.3% |

`normal` 以上差距不大。要更明顯的分級，之後可以加反應延遲或游標雜訊。

## 測試

```bash
python -m live.e2e <beatmap id> <beatmap md5> [level]
```

測試帳號開一個房間，AI 進房打完一輪，測試帳號在旁邊觀戰。最後檢查即時封包有送到，成績也有送出。

2026-11-02 在本機 Illyaverse（Docker，GHCR 映像）上測過：

- 自動測試：Renatus Hard 和 Insane 兩張都完整跑完。
- 真的 lazer 客戶端：進房後看得到 AI 已準備，比賽中即時排行有 AI，AI 的成績也存進房間。

## 已知限制

- AI 只學過不開 mod 的譜。房間有 HR、DT 時照樣會打，但會比較差。
- 每台跑服務的機器都要有 .NET 和編好的 ppy-osu，因為計分是 headless lazer 算的。
- 跳過前奏（skip intro）時，AI 不會跟著跳。即時排行是照遊戲時間對齊的，所以分數顯示不受影響，但 AI 會等到它自己的時間才結束那一輪。
- AI 的成績會跟一般玩家一樣進排行榜和 pp。要不要排除，得在伺服器那邊加一個 AI 帳號的標記。IllyaBot 用的 `is_bot` 會擋住登入，不能直接拿來用。
