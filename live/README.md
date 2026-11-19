# AI 玩家（接 osu! 私人伺服器的 lazer 多人房）

> **2026-11-17 起只用 vision 打。** AI 上場一律是 vision 學生（`live.vision_play.VisionPlayer`，跟 `live.stream` 打的是同一個模型），只看遊戲畫面決定每一步的移動和按鍵，不讀譜面。讀 .osu 算動作的 state 模型不再上場：
> - 多人房、手動開的譜：`live.room_play`，擷取螢幕，用 VMulti 繪圖板打；也能自己列房間、進房、按準備、離開（見「坐在房間裡打」）。
> - 點播、學生的譜：`live.requests` / `live.stream`，擷取 fork 視窗（被蓋住也可以），動作經 bridge 送進去。
> - 下面「一場怎麼跑」到「手感」講的協定打法（`live.service`）沒有畫面、只能用 state 模型，所以停用了，`python -m live.service` 會直接拒絕。`live.humanize` 的手感是加在 state 模型動作上的，也不再有作用。
> - fork 的 bridge `setup` 要帶 `"background": false`（三支程式都會帶）：學生是在沒有背景圖的譜上學的，譜面自己的背景就算 100% dim 也會透出來，那時 DISCO★PRINCE 只打 5%；蓋黑之後 81.6%（r005）、82.7%（r033）。

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

跟大家一起打時，多半不要全力：

```bash
python -m live.service --feel good --view 8771
```

- `--feel`：還沒被指定手感時用哪個（`full` / `pro` / `good` / `casual`，預設 `full`），見下面「手感」。進房、換強度、單人打都可以另外帶 `feel`。
- `--view`：學生的視角（`live.vision_view`，`neuro` 樣式），照房間的遊戲時間畫出地圖和 AI 的游標，給 OBS 的瀏覽器來源用。不要跟 `live.stream` 的 8770 用同一個埠。

## 給 Discord 機器人的接口

[tools.json](tools.json) 是 OpenAI function calling 格式的 tool 定義，TS 那邊直接載入，放進 `tools` 就能用。每個 tool 對應一個 HTTP 端點：

| tool | HTTP | body | 回傳 |
|---|---|---|---|
| `osu_ai_status` | `GET /status` | | 所在房間、難度、是否已準備、是否在打、上一場結果 |
| `osu_ai_list_rooms` | `GET /rooms` | | `[{id, name, host, has_password, participants}]` |
| `osu_ai_join_room` | `POST /join` | `{room_id, password?, level?, feel?}` | 同 status |
| `osu_ai_leave_room` | `POST /leave` | | 同 status |
| `osu_ai_set_level` | `POST /level` | `{level?, feel?}`（至少一個） | 同 status |
| `osu_ai_play_map` | `POST /play` | `{beatmap_id, level?, feel?, mods?}` | acc、rank、分數、combo、判定統計、`replay_url` |

`replay_url`（`/replays/<id>.osr`）不用金鑰就能下載，機器人可以直接把檔案貼到 Discord。

直播用（Miyoria 的 osu game adapter 用這幾個，見 Miyoria 的 `extensions/osu/README.md`）：

| HTTP | body / query | 做什麼 |
|---|---|---|
| `POST /solo` | `{beatmap_id, level?, feel?, mods?}` | AI 用自己的帳號單人打一張譜：準備好就照真實時間打，畫面推到伺服器，任何 lazer 客戶端都能觀戰它；打完送出成績。馬上回傳，過程看事件 |
| `POST /stop` | | 停止單人打譜 |
| `GET /events` | `?after=<id>&wait=<秒>` | 事件（long-poll，最多等 60 秒）：`preparing`、`started`、`combo_break`、`finished`、`aborted`、`error`。回傳 `{events, last}` |

`GET /status` 的 `now` 是正在打的這張譜（曲名、進度、combo、準度、miss），照遊戲時間更新。

`GET /status` 多了 `feel`，`GET /feels` 列出可用的手感。

**直播畫面**：在直播機開 lazer，用另一個帳號登入同一台伺服器，OBS 擷取這個視窗。多人房：進同一個房間、不按準備、改成觀戰，lazer 會把房裡每個人的畫面並排。單人：到 AI 帳號的個人頁按觀戰。

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

`normal` 以上差距不大。要更像人，用「手感」（`feel`，見下面 `live.humanize`）：多人房、單人打和 `/play` 都可以帶。手感是在 headless lazer 算分**之前**加進動作的，所以房裡的分數、即時排行、觀戰的游標和 replay 都是同一場「像人」的打法。它的亂數由譜決定：同一張譜、同樣的 level 和 feel 每次打法都一樣，第二次直接從快取拿。

```bash
python -m live.prepare maps/test/renatus_hard.osu --feel good     # expert / good: 98.94% rank A
```

## 坐在房間裡打（`live.room_play`）

直播要的是「一個人在打」的畫面：AI 坐在一個真的 lazer 前面，跟大家一樣進房、選歌、打譜、看結算。做法是 fork 登入 AI 的帳號、連私人伺服器；進房可以由開台的人在 fork 裡操作，也可以叫 `live.room_play` 進（`POST /join`，Miyoria 的 `/queue osu join` 和 `osu_ai_join_room` 用這個），進去後它自己按準備。房主一開打，`live.room_play` 就讓 vision 學生看著螢幕、用虛擬繪圖板打（bridge 只告訴它現在有沒有在打、遊戲時間，不給譜面內容）。OBS 直接擷取這個 fork 視窗，成績由這個 lazer 自己送給伺服器，跟真人一樣。

```
房主選譜 --> fork（AI 帳號，私人伺服器）--bridge--> live.room_play --VMulti 繪圖板--> 同一個視窗 <-- OBS
```

**第一次**（在直播機）：

1. 重新 build fork（`dotnet build ../ppy-osu/osu.Desktop -c Release`，或重做 `live.bundle`）。新版的 fork 有設 `OSU_RL_LAZER_SERVER` 時連那台伺服器（以前只有 Debug build 會），bridge 的 `status` 也會回報它連的伺服器和登入的帳號。**舊版 Release fork 一律連官方伺服器**，`live.room_play` 會拒絕。
2. 裝好 VMulti（`SETUP_STREAM_PC.md` 第 4 步）。不用 tosu：tosu 只有官方 lazer 版本的記憶體位置，讀不了我們自己 build 的 fork（`Failed to fetch offsets for 0.0`）。改由 fork 的 `OsuRLWatcher` 把任何一場遊戲（多人房的也算）的時鐘、mod、速度和成績送到 bridge。
3. 用一個 fork 副本開起來，連私人伺服器。同一個副本也拿來打點播和學生的譜（見下面「同一個 fork 做全部」）：

```powershell
$env:OSU_RL_INSTANCE = "9"; $env:OSU_RL_LAZER_SERVER = "https://shdcloud.top"; .venv-gpu\Scripts\python -m live.bridge launch
```

4. 在那個視窗裡**手動**登入 AI 的帳號（只能是 AI 的帳號；密碼自己打）。

**每次開台**（同一個 PowerShell，`OSU_RL_INSTANCE` 一樣是 9）：

```powershell
.venv-gpu\Scripts\python -m live.room_play                 # 學生預設是 live.stream 正在用的模型；--ckpt 可換
```

- 它先問 fork 的 bridge 登入的是誰、連的是哪台伺服器：帳號要是 `OSU_RL_AI_USERNAME`（或 `--account`），不能是 `--blocked`（或 `OSU_RL_BLOCKED_ACCOUNTS`，逗號分隔）裡的帳號，伺服器要是 `OSU_RL_SERVER`。不對就印出原因，一下都不打（每次開打前都會再檢查）。
- 學生不用事先準備：選了譜、檢查過帳號，直播文字 `runs/room_play/overlay.txt` 就寫「AI 準備好了」。開始時它先設好學生的樣子（學生的皮膚、不顯示 HUD、背景蓋黑），游標照樣顯示。
- 什麼時候算打完：fork 回報玩家離開，或遊戲時間過了譜的最後一個物件 1 秒（只用來決定停手，不拿來打）。
- **只要開打就自動打**：多人房的譜，或有人在 fork 的選歌畫面自己按開始的單人譜都算。只有 `live.stream`、`live.requests` 經由 bridge 開的譜（畫面是 `OsuRLPlayer`）它不碰，那些由它們自己在背景打。手動開的單人譜跟真人一樣會把成績送給伺服器（2026-11-17 在 shdcloud.top 看到 `SubmitSoloScoreRequest` 成功）。
- mod 就是它看到的畫面（DT 變快、HR 翻轉都是看出來的）；它只練過沒有 mod 的 No Fail 譜，開 mod 會打得比較差。
- 繪圖板移動的是真的游標，所以打譜時 fork 必須在最前面，不要碰滑鼠。譜在載入時、開打時它會自己把 fork 叫到最前面（先按一下 Alt 再切：Windows 只讓「最後送出輸入的程式」搶前景，fork 自己的 `focus` 在 Chrome 擋著時切不過去），中途被擋住就放開按鍵，每 2 秒再叫一次。lazer 的單人模式在視窗不在最前面時會自己暫停，所以這一步單人也要。有人在 fork 裡按 Esc 暫停後切到別的視窗，它也會把 fork 拉回來（分不出是故意的）。
- 學生的視角（`--view 8770`，`neuro` 樣式）照遊戲時間畫出譜和 AI 的游標；`--view 0` 關掉。
- API（`127.0.0.1:8767`，金鑰同 `OSU_RL_CONTROL_TOKEN`）：
  - `GET /status`：多了 `model`、`room_id`、`multi`（房號、自己的狀態、譜下載了沒），沒有 `feel`、`level` 了。`room` 是 fork 在多人畫面時為 true，`live.requests` 會在這時把點播排著。
  - `GET /rooms`：伺服器上開著的多人房（fork 自己問伺服器）。
  - `POST /join {room_id, password?}`：fork 用房號進房（跟點房間連結一樣），等它真的進去才回。正在打譜、已經在別的房間時拒絕。
  - `POST /leave`：離開房間和多人畫面，回主選單。
  - `POST /feel` 一律回 400，因為 vision 學生沒有手感可換。
  - 列房間和進房前都先過帳號檢查（跟開打前一樣）。
- 在房間裡，自己的狀態是閒置、譜也下載好了（lazer 會自己下載缺的譜），它就按準備，每一輪都會；`--no-auto-ready` 關掉，交給開房的人。
- 這些是 fork bridge 的 `rooms`、`join`、`ready`、`leave` 指令（2026-11-17 加的），編譯過，還沒在真房間實測。

注意：

- 遊戲時鐘來自 fork 自己（同一個 QPC 時間戳），2026-11-16 在這台實測：DT 時跟遊戲的取樣差平均 −0.2 ms（最大 0.76 ms），動作在到時間後 0.11 ms 內送出（中位數）。VMulti 繪圖板也實測過（單人、沒登入）：Renatus Normal 打完 0 miss、98.43%（headless 算同一組動作 98.75%），遊戲看到的游標跟送出的位置中位數差 0.2 px，按鍵比預定晚約 4.4 ms。2026-11-17 在 shdcloud.top 的真多人房裡實打 Renatus Normal（good）：96.82%、2 miss，跟單人實測一致。
- 不要同時用 `live.service` 登入同一個 AI 帳號（同一個帳號連兩次，伺服器會踢掉其中一個）。
- stock lazer 沒有讓程式操作房間的方法，進房、準備、離開是 fork 自己加的 bridge 指令。

### 同一個 fork 做全部

多人房、點播、學生的譜都在同一個 fork 副本（例如 9，登入 AI 帳號、連私人伺服器）裡打，`live.room_play` 和 `live.stream --requests 8766`（或 `live.requests`）同時開著，都用同一個 `OSU_RL_INSTANCE`。bridge 同時收好幾個連線，事件每個都收得到。

- **單人的譜（點播、學生）只在 fork 有空時開**：每一首開之前先問 fork（`live.room_play.solo_problem`）。fork 在多人畫面、或有人手動開的譜正在打時先等；fork 沒登入可以打，有登入就要是 `OSU_RL_AI_USERNAME`、不在 `OSU_RL_BLOCKED_ACCOUNTS`、連的是 `OSU_RL_SERVER`，不然不打。
- **fork 自己也擋**：在多人或 playlists 的畫面收到 bridge 的 `play`、`menu` 會拒絕（不然會把它拉出房間），多人畫面也照樣顯示游標，讓開房的人按得到準備。2026-11-17 實測：fork 9 在 `Multiplayer` 時兩個都被拒絕，畫面沒動。
- **經 bridge 打的譜成績不會送出**（`OsuRLPlayer` 不要 token），有登入也一樣：2026-11-17 fork 9 登入 `sql1024` 時打了兩首點播（Renatus Normal 98.27%、DISCO★PRINCE Normal 93.88%），網路紀錄裡沒有送成績的請求。伺服器還是看得到 AI 在打（觀戰）。
- 學生的譜是一首接一首打，中間只隔幾秒，人很難趁空檔進房；目前要先停掉 `live.stream`，進房後再開。

## 直播點譜（`live.requests`）

觀眾用 Miyoria 的 `/queue osu request <譜面連結>` 點譜，AI 照順序在直播機的 fork 視窗裡打，OBS 直接擷取那個視窗（`live.obs_setup` 的「osu_rl」場景）。點播由 vision 學生打（跟 `live.stream` 同一個模型，`live.requests` 單獨開時用 `--ckpt` 可換）：它擷取 fork 的視窗（被別的視窗蓋住也可以），動作經 bridge 送進遊戲，所以視窗在不在最前面都可以。點播帶的 `feel` 還會收，但不再影響打法。2026-11-17 實測 DISCO★PRINCE Normal：r005 81.6%（28 miss）、r033 82.7%（22 miss）。

```bash
python -m live.stream --requests 8766          # 有點播先打點播，沒有就讓學生照舊打它的譜
python -m live.requests --port 8766            # 只打點播（不用學生模型）；學生的視角一樣開在 8770
```

- **譜從哪來**：點播給的是 osu.ppy.sh 上某個難度的 id。程式先問譜面鏡像站（預設 catboy.best；`OSU_RL_MIRROR_LOOKUP`、`OSU_RL_MIRROR_DOWNLOAD` 可換）這張譜屬於哪一組，下載整組（有音樂和背景，不含影片），第一次打時匯入 fork。抓過的譜放在 `runs/requests/maps/`。只收 osu!（standard）的譜。
- **佇列**：存在 `runs/requests/queue.json`，重開也還在；打到一半被中斷的那首會排回第一首。每個人同時最多 2 首（`--per-person`），整個佇列最多 30 首，同一張譜不能重複排。學生正在打的那首會先打完，才輪到點播。
- **直播文字**（`runs/stream/overlay.txt`）多兩行：正在打的點播和誰點的，還有後面排了幾首。
- **AI 在多人房時先排著**：`live.room_play`（`127.0.0.1:8767`）或 `live.service`（`127.0.0.1:8765`）說 AI 在房間裡時（最多每 5 秒問一次 `/status`；`OSU_RL_ROOM_URLS` 可換），點播照樣收，但不會開始打，`GET /queue` 的 `held` 是那個房間（`live.room_play` 時是 `true`）。兩個都沒開就當作不在房間。
- **AI 坐在房間裡時也先排著**：同一個 fork 在多人畫面時，下一首點播不會開始（開了會把 fork 拉出房間）。
- **成績不會送到任何 osu! 帳號**：fork 的 `OsuRLPlayer` 不會要成績送出的 token（不管有沒有登入）。fork 有登入時，只在登入的是 AI 帳號、連的是私人伺服器時才打（見上面「同一個 fork 做全部」），不然那首點播會回報為什麼沒打。
- 2026-11-17 實打一次完整流程（HTTP 點 beatmap 75，DISCO★PRINCE Normal，good）：從 catboy.best 下載整組譜、匯入 fork 9、自動打完，93.88%、2 miss，從點譜到打完約 3 分鐘（算打法 11 s）。

### 手感（`live.humanize`）

> 2026-11-17 起不再使用：手感是加在 state 模型動作上的，現在上場的只有 vision 學生。下面留著當紀錄。

state 模型每顆都瞄正中間、每下都踩在拍子上，直播看起來就像 Autoplay。所以打點播之前，會把算好的動作扭一下，不用重新訓練：

- **時間**：每個物件早一點或晚一點按（常態分布，單位 ms；UR = 10 × σ），物件越密越不準；
- **瞄準**：每個物件瞄偏一點（常態分布，以圈的半徑為單位），跳得越快偏越多，所以小圈和大跳偶爾會 miss；
- **手滑**：偶爾（大跳後比較常）瞄到圈外面，一定 miss；
- **抖動**：手不會完全靜止：整條游標路線疊上快的顫動（10 Hz 左右）和慢的晃動（2 Hz 以下），轉盤前後 150 ms 淡出。

兩個物件之間的偏移是平滑過去的，游標不會瞬移，動作的順序也不會變。轉盤不動。

| feel | 時間 σ | 瞄準 σ | 手滑機率 | 顫動 / 晃動（osu! px，每軸 RMS） | Renatus Normal / Hard / Insane |
|---|---|---|---|---|---|
| `full` | | | | | 98.8% / 100% / 99.7% |
| `pro` | 12 ms | 0.15 r | 0.15% | 0.5 / 1.2 | 98.8% / 99.8% / 97.1% |
| `good`（預設） | 18 ms | 0.25 r | 0.4% | 0.8 / 2.0 | 98.5% / 98.4% / 90.9% |
| `casual` | 28 ms | 0.36 r | 1.0% | 1.2 / 3.0 | 96.1% / 92.2% / 80.2% |

準度是 headless lazer 算的（每種打兩次的平均）。抖動對準度幾乎沒影響：加之前是 98.6 / 98.3 / 90.8（good）。直播用的 1080p 視窗一個 osu! px 約 2.25 個螢幕 px，所以 good 的游標大約晃 ±5 個螢幕 px。自己算一張譜：

```bash
python -m live.humanize maps/test/renatus_hard.osu
```

### 控制 API

金鑰同 `live.service`（`Authorization: Bearer <OSU_RL_CONTROL_TOKEN>`），預設只聽 `127.0.0.1:8766`。Miyoria 在別台時加 `--requests-host 0.0.0.0`，走 Tailscale。

| HTTP | body | 回傳 |
|---|---|---|
| `GET /queue` | | `{now, queue, recent, feels, held}` |
| `POST /request` | `{beatmap_id, by, by_id?, feel?}` | `{item, position}`（會先抓譜，要幾秒） |
| `POST /skip` | | `{skipped}` |
| `POST /clear` | `{all?}` | `{cleared, skipped}` |

錯誤回 400 和 `{"error": "<給觀眾看的原因>"}`，例如「找不到這張譜」、「每個人最多同時點 2 首」。

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
