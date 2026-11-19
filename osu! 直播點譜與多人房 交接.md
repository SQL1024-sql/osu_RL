# osu! 直播點譜與多人房 交接

Nov 16, 2026 · @SQL1024

觀眾用 Miyoria 指令點譜、AI 在直播上像人一樣打（點播和多人房）的程式都寫好了，離線和本機 fork 上都測過；VMulti 繪圖板也在這台實際按進 fork 打完一首（2026-11-16）。2026-11-17 在 shdcloud.top 的真多人房裡打完一首（96.82%），之後改成一個 fork（9）同時打多人房、點播和學生的譜，點譜在 osu\_rl 這邊從下載到打完實測過；還差專屬的 AI 帳號、Discord 那一半（Miyoria 部署）和學生／多人房的開關，三個 repo 的改動都還沒 commit。

## 2026-11-17 重大變更：只用 vision 打

AI 上場一律用 vision 學生（看畫面打，不讀譜面）；state 模型和手感不再上場。下面有提到 state 模型、手感、協定打法的地方都是改之前的紀錄。

| 路 | 現在 | 實測 |
| --- | --- | --- |
| 點播、學生的譜（`live.requests` / `live.stream`） | vision 學生擷取 fork 視窗（被蓋住也行），動作經 bridge 送 | DISCO★PRINCE Normal：r005 81.6%、r033 82.7%（之前 state 模型 93.9%） |
| 多人房、手動開的譜（`live.room_play`） | vision 學生擷取螢幕，用 VMulti 繪圖板打；能自己列房間、進房、按準備、離開（fork 加了 bridge 指令 `rooms`、`join`、`ready`、`leave`） | 列房間在 shdcloud 上實測可用；進房到打完沒實測（使用者說照 state 時的流程就可以，不用測） |
| 協定打法（`live.service`） | 停用（沒有畫面，只能用 state 模型），啟動直接拒絕 | — |
| 手感（`feel`） | 不再有作用；room\_play 的 `POST /feel` 回 400 | — |

- **fork 加了「背景蓋黑」**（bridge `setup` 的 `"background": false`，三支程式都會帶）：學生是在沒有背景圖的譜上學的，譜面自己的背景就算 100% dim 也透出來，第一次點播只打 5%。改完要重新 build fork。
- **Miyoria 跟著改了（`feat/osu-queue`，擴充 0.2.0，還沒 commit）**：功能都留著，全部改成 vision 打。`osu_ai_status`、`list_rooms`、`join_room`、`leave_room` 和 `/queue osu status | join | leave` 改走 `live.room_play`；`osu_ai_play_map` 改成排進點播佇列（回排第幾首，不再直接回成績）。拿掉的只有強度（`level`）、手感（`feel`、`/queue osu feel`）和只給 `live.service` 用的設定（`url`、`public_url`、`account`、`blocked_accounts`）。8 個測試、lint、型別檢查、build 都過。
- 預設學生是 `runs/stream/state.json` 的模型（目前 r005）；`live.improve` 現在最好的是 r033，要換就把它放進 `runs/stream/new/`，或 `--ckpt`。

## 現況總覽

改動分在三個 repo，全部都在工作目錄裡、**還沒 commit**。

| 部分 | 位置 | 分支 | 狀態 |
| --- | --- | --- | --- |
| 點譜佇列 `live.requests`、手感 `live.humanize` | osu\_rl `C:\SQL1024\Code\osu_rl` | `requests-queue` | 寫好，離線測過 |
| 多人房坐著打 `live.room_play` | osu\_rl | `requests-queue` | 寫好，在 shdcloud.top 的真多人房裡實打過（用 `sql1024` 測，不是專屬 AI 帳號） |
| 協定打法加手感（`live.service`、`live.prepare`、`live.player`） | osu\_rl | `requests-queue` | 寫好，離線和 headless lazer 測過 |
| fork：不送點播成績、Release 也能連私人伺服器、`OsuRLWatcher` | ppy-osu `C:\SQL1024\Code\ppy-osu` | `feature/rl-bot` | 編好 Release，在本機跑過 |
| `/queue osu` 指令 | Miyoria worktree `C:\SQL1024\Code\Miyoria-osu-queue` | `feat/osu-queue`（從 `origin/main` 開） | 15 個測試、typecheck、biome、build 都過 |

Miyoria 主 checkout（`perf/sing-pipeline`）沒動：那時有人正在改 `workers/sing/engine.py`，所以另開 worktree。

## 怎麼運作

兩條路都用 state 模型（`expert`）打，再用「手感」把動作扭得像人（時間、瞄準、手滑，加上游標的顫動和晃動）；學生的視角（`live.vision_view`，neuro 樣式）照遊戲狀態畫，所以是誰在打都有畫面。

**觀眾點譜**（單人，直播機的學生 fork 視窗）：

1. 觀眾下 `/queue osu request <譜面連結> [手感]`，Miyoria 的 `extensions/osu` 解析出難度 id，轉給 `live.requests`（8766）。
2. `live.requests` 從 catboy.best 查到整組譜並下載（有音樂和背景），排進佇列（存成 `runs/requests/queue.json`）。
3. `live.stream --requests 8766` 每打完一張學生的譜就看佇列：有點播就用 state 打，動作經 bridge 送進 fork；沒有就讓學生照舊打。
4. AI 在多人房時，點播照樣收但先排著。

**多人房坐著打**（直播用，OBS 拍真的遊戲視窗）：

1. 另開一個 fork 副本（instance 9），設 `OSU_RL_LAZER_SERVER` 連私人伺服器，人手動登入 AI 帳號、進房、按準備。
2. `live.room_play` 從 fork 的 bridge 知道選了哪張譜，從遊戲自己的檔案庫讀 .osu，幾秒內算好打法；直播文字寫「AI 準備好了」。
3. 開打後，fork 的 `OsuRLWatcher` 把時鐘、mod、速度送過來，`live.room_play` 用 VMulti 虛擬繪圖板照時間打；DT/NC/HT/DC 照速度、HR 上下翻轉。
4. 成績由這個 lazer 自己送給伺服器，跟真人一樣。

這條路不用 tosu：tosu 只有官方 lazer 版本的記憶體位置，讀不了我們自己 build 的 fork（實測：`Failed to fetch offsets for 0.0`）。

**協定打法**（沒有遊戲視窗的備案）：`live.service` 直接用 lazer 客戶端的協定進房，手感在 headless lazer 算分前加進去；直播要另開 lazer 觀戰。Miyoria 沒設 `seat_url` 時用 `/queue osu join`、`leave` 走這條。

## 已驗證

每一段都有實測或離線測試；真多人房只用 `sql1024` 打過一首。「本機 fork」= 2026-11-16 在這台開的 Release fork（instance 9）。

| 項目 | 怎麼測 | 結果 |
| --- | --- | --- |
| Release fork 有設 `OSU_RL_LAZER_SERVER` | 本機 fork，bridge `status` | 回報 `http://127.0.0.1:3000` |
| Release fork 沒設這個變數 | 本機 fork（instance 8） | 回報 `https://osu.ppy.sh`，room\_play 拒絕 |
| 沒登入（Guest / Offline） | 本機 fork | room\_play 拒絕，原因「not logged in」 |
| 選譜後自動準備 | 本機 fork | 從檔案庫讀到 .osu（MD5 吻合），1 秒內算好 |
| 遊戲時鐘（DT 1.5 倍速） | 本機 fork，23 次比對 | 跟遊戲取樣差平均 −0.2 ms，最大 0.76 ms |
| 打譜迴圈 | 本機 fork，繪圖板換成只記錄的替身 | 20 秒送出 3113 個動作，中位數比該送的時間晚 0.11 ms |
| VMulti 繪圖板真的按進遊戲 | 本機 fork 單人、沒登入、伺服器指向沒人聽的 `127.0.0.1:3000`；bridge 開 Renatus Normal（NF），用跟 room\_play 一樣的 `Playfield` + `Tablet` 打，手感 full | 打完 0 miss、98.43%（headless 算同一組動作 98.75%）；送出 26365 個動作，中位數晚 0.06 ms、最多 1.32 ms，沒失焦 |
| 繪圖板對 fork 視窗的座標 | 同一場，遊戲自己記的游標 vs 送出的位置 | 中位數差 0.2 px，游標晚約 2 ms 到；按鍵比預定晚 4.4 ms（p95 4.9 ms，其中 3 ms 是 `PRESS_GAP`）。9 個 Ok 是模型自己就打晚（預定 +59～+76 ms），不是繪圖板 |
| `OsuRLWatcher` 事件 | 本機 fork | seat（mod、rate 1.5）、live、played 都收到 |
| 手感（Renatus Hard，OD 6） | headless lazer 算分（含抖動） | 全力 100%、職業 99.8%、熟練 98.4%、休閒 92.2% |
| 手感（Renatus Insane，OD 8） | headless lazer 算分（含抖動） | 全力 99.7%、職業 97.1%、熟練 90.9%、休閒 80.2% |
| 真的多人房（shdcloud.top） | 2026-11-17：fork 7 登入 `SQL` 開房選 Renatus Normal，fork 9 登入 `sql1024`（room\_play `--account sql1024`，只限這次）加入、準備；good 手感 | 96.82%、2 miss、最大 combo 411，打的時候沒失焦；跟單人離線實測（96.66%）一致。成績照常送進伺服器 |
| 房間裡的畫面名稱 | fork 9 的 bridge `status.screen` | 房間外層是 `Multiplayer`，打譜和結算是 `MultiplayerPlayer`、`MultiplayerResultsScreen`；room\_play 改成看 `Multiplayer` / `MultiSpectator` 開頭，結算時 `room` 是 true |
| 手感的抖動真的打進遊戲 | 本機 fork 單人離線，繪圖板打 Renatus Normal，手感 good | 96.66%、2 miss；headless 算同一組動作再晚 4.4 ms 是 96.64%、判定數一樣。遊戲記的游標跟送出的（含抖動）中位數差 0.2 px |
| 多人房手感在算分前套上 | `live.prepare --feel` | 熟練 98.94% rank A，第二次從快取拿 |
| 點譜佇列 API、每人上限、重開保留、跳過 | 假的 bridge 和假的鏡像站 | 通過 |
| 房間中排著點播 | 假的 room\_play / live.service | 通過 |
| Miyoria `/queue osu` | vitest | 15 個測試通過 |
| 鏡像站 | catboy.best 實際查詢 | `/api/v2/b/<id>` 和 `/d/<set>n` 可用（要帶 User-Agent） |
| 點譜完整流程（osu\_rl 那一半） | 2026-11-17：fork 9 登入 `sql1024`，`live.requests` 跟 `live.room_play` 同時開，HTTP 點 beatmap 75 | 下載、匯入、自動打完 DISCO★PRINCE Normal 93.88%、2 miss；room\_play 沒插手；沒送成績 |
| 登入的 fork 打點播 | 同上，先用 `.env` 的 `osu_rl_AI` 再用 `sql1024` | 前者拒絕（登入的不是 AI 帳號），後者打完 Renatus Normal 98.27% |
| fork 在房間時擋 `play`、`menu` | fork 9 在 `Multiplayer` 畫面送兩個指令 | 都被拒絕，畫面沒動 |
| 手動單人自動接手、自動切前景 | fork 9 選歌畫面自己按開始後切去 Chrome | room\_play 把 fork 叫到前面打完，97.69%、1 miss；成績有送進 shdcloud |

## 還沒做完、卡住的地方

最大的缺口是「真的多人房」還沒測，卡在這台沒有能跑的伺服器。繪圖板已經裝好也實測過了（見上面「已驗證」）。

- **伺服器改用 `https://shdcloud.top`（線上的 Illyaverse）**：2026-11-16 `live/.env` 的 `OSU_RL_SERVER` 已改成它；`/api/v2` 和三個 SignalR hub 都有回應（沒登入時 401）。fork 9 用 `OSU_RL_LAZER_SERVER=https://shdcloud.top` 開起來，bridge 回報這個伺服器，room\_play 的檢查目前停在「not logged in」。還沒有人用 AI 帳號登入過。站上寫多人和觀戰「仍在驗證中」。
- **開房的真人那邊也用 fork（instance 7）**：站上推薦的 Kaleidoscope 只認 lazer 2026.1005.1（Transfiguration 比對 `osu.Game.dll` 雜湊），這台裝的是 2026.1115.0，啟動會停在 `client_assembly_unverified`。伺服器不檢查客戶端版本，所以另開 fork 副本 7（`OSU_RL_LAZER_SERVER=https://shdcloud.top`）給真人登入開房，AI 用副本 9。Kaleidoscope 開著時不能有其他 lazer（fork 也算）在跑。
- **進房、按準備、離開**：2026-11-17 在 fork 的 bridge 加了指令（`rooms`、`join`、`ready`、`leave`），`live.room_play` 和 Miyoria 都接上了；還沒在真房間實測。
- **沒實測的細節**：實際下載一組譜並匯入 fork；繪圖板在 DT/HR 的房間裡（單人實測只打了沒有 mod 的譜）。
- **點播打完不會主動回報 Discord**：要用 `/queue osu list` 看成績。學生正在打的那首會先打完才輪到點播。
- **開 mod 的房間**：AI 只練過沒有 mod 的譜；手感照沒有 mod 的譜算，會打得更差。
- **改成一個 fork 做全部（2026-11-17）**：多人房、點播、學生的譜都在 fork 9（登入 AI 帳號）裡打，`live.room_play` 和 `live.stream` / `live.requests` 同時開。單人的譜只在 fork 有空時開（不在多人畫面、沒有手動開的譜在打）；fork 在多人畫面時拒絕 bridge 的 `play`、`menu`。不要同時用 `live.service` 登入同一個 AI 帳號。
- **學生的譜一首接一首**，人很難趁空檔進房：目前要先停 `live.stream`。想加一個開關（例如 `/queue osu join` 讓它打完這首就停、`leave` 再接著打）。
- **最近兩場房間成績被伺服器拒收**：fork 9 的紀錄裡 room 20、21 的 `SubmitRoomScoreRequest` 回 `room_score_token_revoked`（第一次 96.82% 那場有收下）。原因還不知道，要再打一場乾淨的房間看。

## 安全規則

bot 絕對不能在官方伺服器打，也不能用任何人的帳號打；下面是現在擋住這兩件事的機制，不要拿掉。

- **Release fork 預設連官方伺服器**。只有設了 `OSU_RL_LAZER_SERVER` 才會連私人伺服器（這次加的 `UseDevelopmentServer` override）。讀這個變數的那一行在 `DevelopmentEndpointConfiguration.cs`，是你原本標「not for commit」的改動，B 要靠它，不能丟。
- **`live.room_play` 每次開打前問 bridge**：伺服器要是 `OSU_RL_SERVER`（先檢查）、要已登入、帳號要是 `OSU_RL_AI_USERNAME`、不能在 `OSU_RL_BLOCKED_ACCOUNTS`；任一不對就一下都不送。
- **點播和學生的譜成績不送**：`OsuRLPlayer` 不要成績 token（有登入也一樣，2026-11-17 實測網路紀錄沒有送出）。fork 有登入時，每一首開之前檢查：帳號要是 `OSU_RL_AI_USERNAME`、不在 `OSU_RL_BLOCKED_ACCOUNTS`、伺服器要是 `OSU_RL_SERVER`（`live.room_play.solo_problem`），不對就不打。以前「`game.ini` 有登入資料就拒絕」改成這個。
- **手動開的單人譜會送成績**：有人在 fork 9 的選歌畫面自己按開始，`live.room_play` 會接手用繪圖板打，成績跟真人一樣送給伺服器（記在 fork 登入的帳號）。
- **Miyoria 的工具和進房指令**照舊先過帳號保護（`account`、`blocked_accounts`）。
- AI 的帳號密碼要人自己在 fork 裡打，不要讓程式代打。

## 下一步

照順序做：先在直播機把完整流程實測一次，沒問題再 commit。

- [x] 直播機裝 VMulti（管理員權限 + 重開機）：2026-11-16 已裝好，單人離線實測按得進 fork
- [x] 準備伺服器：`OSU_RL_SERVER` 改成 `https://shdcloud.top`（2026-11-16）
- [ ] 直播機重新 build fork，或重做 `live.bundle`
- [x] 開 fork instance 9（設 `OSU_RL_LAZER_SERVER`），手動登入：2026-11-17 用 `sql1024` 先測
- [x] 跑 `live.room_play`，確認印出「ready to play」
- [x] 用另一個帳號開房、選一張簡單的譜，等「AI 準備好了」再按開始：fork 7 的 `SQL` 開房，96.82%
- [x] 確認 `status.screen`：打譜和結算不是 `Multiplayer`，已改成看前綴
- [ ] 註冊專屬的 `osu_rl_AI`，在 fork 9 改登入它，不帶 `--account` 再打一次；`OSU_RL_BLOCKED_ACCOUNTS` 加上 `SQL`、`sql1024`（要先問）
- [x] 開打時自動把 fork 9 叫到最前面：2026-11-17 改成先按 Alt 再切，Chrome 在前面時實測切得過去；手動單人實打 97.69%（沒人切視窗）
- [ ] 房間裡也驗證一次自動切到前面，順便看成績會不會又被拒（`room_score_token_revoked`）
- [x] 點譜的 osu\_rl 那一半：2026-11-17 用 HTTP 點 beatmap 75，下載 → 匯入 → 自動打完（93.88%）
- [ ] 點譜的 Discord 那一半：Miyoria 的 `feat/osu-queue` 要進 main、部署到跑機器人的那台，設 `queue_url`、`seat_url`；不在同一台時 osu\_rl 要 `--requests-host 0.0.0.0` 走 Tailscale
- [ ] 學生和多人房之間的開關（見上面「學生的譜一首接一首」）
- [ ] commit：osu\_rl（`requests-queue`）、ppy-osu（只拿這次的檔案）、Miyoria（`feat/osu-queue`，Conventional Commits，無署名）
- [ ] Miyoria 開 PR 時加 repo label（feature request）

## 怎麼跑

指令都在 `C:\SQL1024\Code\osu_rl` 執行；金鑰一律是 `live/.env` 的 `OSU_RL_CONTROL_TOKEN`。詳細步驟在 osu\_rl 的 `live/README.md`「坐在房間裡打」和「直播點譜」。

```powershell
# 多人房：fork 副本 9 連私人伺服器，然後在視窗裡登入 AI 帳號
$env:OSU_RL_INSTANCE = "9"; $env:OSU_RL_LAZER_SERVER = "<私人伺服器網址>"; .venv-gpu\Scripts\python -m live.bridge launch
# 同一個 PowerShell
.venv-gpu\Scripts\python -m live.room_play --feel good

# 點譜（另一個 PowerShell，不要跟 room_play 同時開）
.venv-gpu\Scripts\python -m live.stream --requests 8766

# 協定打法（備案）
.venv-gpu\Scripts\python -m live.service --feel good --view 8771

# 算一張譜在各手感下的準度（要 .NET 和編好的 ppy-osu）
.venv-gpu\Scripts\python -m live.humanize maps/test/renatus_hard.osu
```

| 服務 | 埠 | Miyoria 設定 |
| --- | --- | --- |
| `live.service` | 8765 | `url` |
| `live.requests`（在 `live.stream` 裡） | 8766 | `queue_url` |
| `live.room_play` | 8767 | `seat_url` |
| 學生的視角 | 8770（`live.service --view` 用 8771） | OBS 瀏覽器來源 |

Miyoria `miyoria.toml` 的 `[extensions."miyoria.osu"]` 要加 `queue_url`、`seat_url`；設了 `seat_url` 就有 `/queue osu status`、`feel`，沒有 `join`、`leave`。`call_timeout_ms` 要 120000（點一張沒抓過的譜要先下載）。OBS 的直播文字：點播用 `runs/stream/overlay.txt`，多人房用 `runs/room_play/overlay.txt`。

## 檔案與未提交的改動

commit 時只拿「這次的」檔案；「不是這次的」是原本就在工作目錄的改動，沒碰過。

| repo | 這次的 | 不是這次的 |
| --- | --- | --- |
| osu\_rl | 新檔 `live/humanize.py`、`live/requests.py`、`live/room_play.py`；改 `live/prepare.py`、`live/player.py`、`live/service.py`、`live/stream.py`、`live/tools.json`、`live/README.md` | `live/improve.py`、`live/finetune_real.py`、`vision/student.py`、`FABLE_*.md`、`live/lazer_ctl.py`、`results/experiments/*` |
| ppy-osu | 新檔 `osu.Game/OsuRL/OsuRLWatcher.cs`；改 `osu.Game/OsuGame.OsuRL.cs`、`osu.Game/OsuRL/OsuRLPlayer.cs` | `DevelopmentEndpointConfiguration.cs`（B 要靠它）、`TestSceneOsuRLReplay.cs`、`TestSceneOsuRLLive.cs` |
| Miyoria（worktree） | 新檔 `extensions/osu/src/queue.ts`；改 `src/index.ts`、`test/extension.test.ts`、`config.schema.json`、`miyoria.extension.toml`、`README.md` | 無 |

測試留下的東西：`%APPDATA%\osu-rl-8`、`osu-rl-9`（測試用的 fork 資料夾，只有設定和 Renatus Normal）；`runs/live_cache`、`runs/room_play/maps` 是快取；`runs/tablet_test/` 是繪圖板實測那場遊戲自己的紀錄。離線測試腳本在 Claude 的暫存資料夾，沒進 repo。
