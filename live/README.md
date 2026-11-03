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

## 安裝（第一次）

1. 在伺服器網站註冊 AI 的帳號（開台的話就是 Miyo 的 osu 帳號，名稱和頭像用她的）。
2. 安裝 Python 套件（`pip install -r requirements.txt`）和 .NET 10 SDK（Windows：`winget install Microsoft.DotNet.SDK.10`）。
3. 執行：

```bash
python -m live.setup
```

它會問伺服器網址和 AI 帳號，產生控制金鑰，寫進 `live/.env`；ppy/osu 不在 `OSU_REPO`（預設 `../ppy-osu`）時，會下載測過的版本、放進計分用的 harness 並編譯（第一次要幾分鐘）；最後用 AI 帳號登入確認，並印出 Miyoria 要貼的設定。重跑不會重問，也不會換掉金鑰。

## 設定（`live/.env` 或環境變數；`live/.env` 不會進 git，`live.setup` 會寫好）

| 變數 | 意思 |
|---|---|
| `OSU_RL_SERVER` | 伺服器網址，例如 `https://shdcloud.top`，本機測試用 `http://127.0.0.1:3000` |
| `OSU_RL_AI_USERNAME` / `OSU_RL_AI_PASSWORD` | AI 帳號 |
| `OSU_RL_CONTROL_TOKEN` | 控制 API 的金鑰（16 字以上），Discord 機器人呼叫時帶 `Authorization: Bearer <token>` |
| `OSU_RL_TESTER_USERNAME` / `OSU_RL_TESTER_PASSWORD` | 只有 `live.e2e` 測試要用 |

## 啟動

```bash
python -m live.service
```

預設聽 `127.0.0.1:8765`，跟 Miyoria 在同一台就不用改。Miyoria 在別台電腦時用 `--host 0.0.0.0`，兩台之間走 Tailscale（或 HTTPS 反向代理）。

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

直播用（Miyoria 的 osu game adapter 用這幾個，見 Miyoria 的 `extensions/osu/README.md`）：

| HTTP | body / query | 做什麼 |
|---|---|---|
| `POST /solo` | `{beatmap_id, level?, mods?}` | AI 用自己的帳號單人打一張譜：準備好就照真實時間打，畫面推到伺服器，任何 lazer 客戶端都能觀戰它；打完送出成績。馬上回傳，過程看事件 |
| `POST /stop` | | 停止單人打譜 |
| `GET /events` | `?after=<id>&wait=<秒>` | 事件（long-poll，最多等 60 秒）：`preparing`、`started`、`combo_break`、`finished`、`aborted`、`error`。回傳 `{events, last}` |

`GET /status` 的 `now` 是正在打的這張譜（曲名、進度、combo、準度、miss），照遊戲時間更新。

**直播畫面**：在直播機開 lazer，用另一個帳號登入同一台伺服器，到 AI 帳號的個人頁按觀戰，OBS 擷取這個視窗。

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

單人直播模式：

```bash
python -m live.e2e <beatmap id> - [level] --solo
```

測試帳號觀戰 AI，AI 單人打一張譜，檢查觀戰端收到畫面、事件順序（preparing → started → … → finished）和單人成績。2026-11-03 在本機 Illyaverse 測過 Renatus Hard：expert 打出 SS（1046 包畫面送到觀戰端，成績已存）。

## 已知限制

- AI 只學過不開 mod 的譜。房間有 HR、DT 時照樣會打，但會比較差。
- 每台跑服務的機器都要有 .NET 和編好的 ppy-osu，因為計分是 headless lazer 算的。
- 跳過前奏（skip intro）時，AI 不會跟著跳。即時排行是照遊戲時間對齊的，所以分數顯示不受影響，但 AI 會等到它自己的時間才結束那一輪。
- AI 的成績會跟一般玩家一樣進排行榜和 pp。要不要排除，得在伺服器那邊加一個 AI 帳號的標記。IllyaBot 用的 `is_bot` 會擋住登入，不能直接拿來用。
