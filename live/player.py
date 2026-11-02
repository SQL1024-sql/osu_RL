"""The AI as a lazer player in a multiplayer room: it joins like a normal client, gets ready once its play is
prepared, streams its live score and cursor while the others play, and submits its score at the end.

Room flow (server's MultiplayerUserState): Idle 0 -> Ready 1 -> WaitingForLoad 2 -> Loaded 3 -> ReadyForGameplay 4
-> Playing 5 -> FinishedPlay 6 -> Results 7.
"""
import asyncio
import hashlib
import logging
import time

from live.hub import Api, HubError
from live.prepare import LEVELS, prepare

log = logging.getLogger("live")
IDLE, READY, LOADED, READY_FOR_GAMEPLAY, FINISHED_PLAY = 0, 1, 3, 4, 6
PASSED = 3  # SpectatedUserState.Passed
SEND_EVERY_MS = 200  # lazer's SpectatorClient.TIME_BETWEEN_SENDS
LEAD_MS = 1500  # frames go out this far ahead of the room's clock; viewers line them up by gameplay time
VERSION_HASH = "osu_rl-ai"


def bundles(play):
    """The play as lazer spectator frame bundles: (gameplay time of the first frame, bundle)."""
    headers, h = play["headers"], 0
    frames = play["frames"]
    i = 0
    while i < len(frames):
        start = frames[i][0]
        j = i
        while j < len(frames) and frames[j][0] < start + SEND_EVERY_MS:
            j += 1
        chunk = frames[i:j]
        while h < len(headers) and headers[h]["time"] <= chunk[-1][0]:
            h += 1
        header = dict(headers[h - 1]) if h else dict(totalScore=0, accuracy=1.0, combo=0, maxCombo=0, statistics={},
                                                       scoreProcessorStatistics=dict(baseScore=0.0, maximumBaseScore=0.0, accuracyJudgementCount=0, comboPortion=0.0, bonusPortion=0.0))
        header.pop("time", None)
        header["mods"] = play["mods"]
        yield start, {"header": header, "frames": [{"time": t, "mouseX": x, "mouseY": y, "buttonState": b} for t, x, y, b in chunk]}
        i = j


class AiPlayer:
    def __init__(self, server, username, password, level="expert"):
        self.api = Api(server)
        self.username, self.password = username, password
        self.level = level
        self.me = None
        self.mp = self.spectator = None
        self.room = None  # latest room snapshot
        self.play = None  # prepared play for the current playlist item
        self.preparing = None
        self.token = None
        self.playing = None
        self.last_result = None

    # --- control (the Discord bot's tools go through these) ---

    async def start(self):
        self.me = await self.api.login(self.username, self.password)
        self.mp = await self.api.hub("multiplayer")
        self.spectator = await self.api.hub("spectator")
        for name in ("LoadRequested", "GameplayStarted", "ResultsReady", "SettingsChanged", "PlaylistItemChanged", "PlaylistItemAdded",
                     "UserKicked", "RoomStateChanged", "GameplayAborted", "UserJoined", "UserLeft", "UserStateChanged"):
            self.mp.on(name, getattr(self, "_on_" + name, self._on_room_event))
        try:
            await self.mp.invoke("LeaveRoom")  # a membership left over from a previous run blocks joining
        except HubError:
            pass

    async def join(self, room_id, password="", level=None):
        if level:
            self.set_level(level)
        if self.room:
            await self.leave()
        self.room = await self.mp.invoke("JoinRoomWithPassword", int(room_id), password)
        log.info("joined room %s", room_id)
        self._prepare_current()
        return self.status()

    async def leave(self):
        if self.room:
            await self.mp.invoke("LeaveRoom")
        self.room = self.play = None
        if self.playing:
            self.playing.cancel()
        return self.status()

    def set_level(self, level):
        if level not in LEVELS:
            raise ValueError(f"level must be one of {list(LEVELS)}")
        changed, self.level = level != self.level, level
        if changed and self.room:
            self._prepare_current()

    def status(self):
        item = self._item()
        return dict(user=self.me and self.me["username"], room=self.room and self.room.get("roomID"), level=self.level,
                    beatmap=item and item.get("beatmapID"), ready=bool(self.play), playing=bool(self.playing and not self.playing.done()),
                    last_result=self.last_result)

    # --- room events ---

    def _item(self):
        if not self.room:
            return None
        current = self.room["settings"].get("playlistItemId")
        return next((i for i in self.room.get("playlist", []) if i.get("id") == current), None)

    def _prepare_current(self):
        item = self._item()
        self.play = None
        if self.preparing:
            self.preparing.cancel()
        if item and not item.get("expired"):
            self.preparing = asyncio.create_task(self._prepare(item))

    async def _prepare(self, item):
        await self.mp.invoke("ChangeBeatmapAvailability", {"state": 1, "downloadProgress": None})  # NotDownloaded -> working on it
        osu = await self.api.request("GET", f"/osu/{item['beatmapID']}", raw=True)
        if hashlib.md5(osu).hexdigest() != item["beatmapChecksum"].lower():
            raise HubError("downloaded .osu does not match the room's checksum")
        mods = item.get("requiredMods") or []
        play = await asyncio.to_thread(prepare, osu, mods, self.level)
        play.update(mods=mods, item=item)
        self.play = play
        await self.mp.invoke("ChangeBeatmapAvailability", {"state": 4, "downloadProgress": None})  # LocallyAvailable
        await self._ready()
        s = play["submission"]
        log.info("prepared %s at %s: %.2f%% %s", item["beatmapID"], self.level, 100 * s["accuracy"], s["rank"])

    async def _ready(self):
        if self.play and self.room and self.room.get("state", 0) == 0:
            try:
                await self.mp.invoke("ChangeState", READY)
            except HubError as e:
                log.info("not ready yet: %s", e)

    async def _on_room_event(self, *args):
        pass

    async def _on_SettingsChanged(self, settings):
        before = self.room["settings"].get("playlistItemId")
        self.room["settings"] = settings
        if settings.get("playlistItemId") != before:
            self._prepare_current()

    async def _on_PlaylistItemChanged(self, item):
        playlist = self.room.setdefault("playlist", [])
        old = next((i for i in playlist if i["id"] == item["id"]), None)
        if old:
            playlist[playlist.index(old)] = item
        else:
            playlist.append(item)
        if item["id"] == self.room["settings"].get("playlistItemId") and (not old or old.get("beatmapID") != item.get("beatmapID") or old.get("requiredMods") != item.get("requiredMods")):
            self._prepare_current()

    _on_PlaylistItemAdded = _on_PlaylistItemChanged

    async def _on_RoomStateChanged(self, state):
        self.room["state"] = state
        if state == 0:
            await self._ready()

    async def _on_UserKicked(self, user):
        if user.get("userID", user.get("user_id")) == self.me["id"]:
            self.room = self.play = None

    async def _on_LoadRequested(self):
        item = self.play["item"]
        room = self.room["roomID"]
        r = await self.api.request("POST", f"/api/v2/rooms/{room}/playlist/{item['id']}/scores", data=dict(
            version_hash=VERSION_HASH, beatmap_id=str(item["beatmapID"]), beatmap_hash=item["beatmapChecksum"], ruleset_id=str(item.get("rulesetID", 0))))
        self.token = r["id"]
        await self.mp.invoke("ChangeState", LOADED)
        await self.mp.invoke("ChangeState", READY_FOR_GAMEPLAY)

    async def _on_GameplayStarted(self):
        self.playing = asyncio.create_task(self._play(self.play, self.token, self.room["roomID"]))

    async def _on_GameplayAborted(self, *args):
        if self.playing:
            self.playing.cancel()

    async def _on_ResultsReady(self):
        await self._ready()

    async def _play(self, play, token, room):
        item = play["item"]
        state = {"beatmapID": item["beatmapID"], "rulesetID": item.get("rulesetID", 0), "mods": play["mods"], "state": 1,
                 "maximumStatistics": play["maximumStatistics"]}
        await self.spectator.invoke("BeginPlaySessionV2", token, state)
        start_ms = min(0, play["frames"][0][0])  # the room's gameplay clock starts here (lead-in before the first object)
        t0 = time.monotonic()
        for at, bundle in bundles(play):
            delay = (at - start_ms - LEAD_MS) / 1000 - (time.monotonic() - t0)
            if delay > 0:
                await asyncio.sleep(delay)
            await self.spectator.send("SendFrameDataV2", token, bundle)
        end = (play["frames"][-1][0] - start_ms) / 1000 - (time.monotonic() - t0)
        await asyncio.sleep(max(0, end))
        await self.spectator.invoke("EndPlaySessionV2", token, PASSED)
        body = dict(play["submission"], mods=play["mods"])
        result = await self.api.request("PUT", f"/api/v2/rooms/{room}/playlist/{item['id']}/scores/{token}", json=body)
        s = play["submission"]
        self.last_result = dict(beatmap=item["beatmapID"], level=play["level"], accuracy=s["accuracy"], rank=s["rank"], total_score=s["total_score"],
                                max_combo=s["max_combo"], statistics=s["statistics"], score_id=result.get("id"))
        log.info("finished: %s", self.last_result)
        await self.mp.invoke("ChangeState", FINISHED_PLAY)  # the round's results open once everyone is here

    async def close(self):
        for h in (self.mp, self.spectator):
            if h:
                await h.close()
        await self.api.close()
