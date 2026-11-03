"""The AI as a lazer player: in a multiplayer room it joins like a normal client, gets ready once its play is
prepared, streams its live score and cursor while the others play, and submits its score at the end. On its own
(solo) it plays one beatmap the same way, so anyone can spectate it from a lazer client (e.g. for a stream).
While it plays, it narrates game events (started, combo breaks, finished) for the Discord bot.

Room flow (server's MultiplayerUserState): Idle 0 -> Ready 1 -> WaitingForLoad 2 -> Loaded 3 -> ReadyForGameplay 4
-> Playing 5 -> FinishedPlay 6 -> Results 7.
"""
import asyncio
import hashlib
import logging
import re
import time

from live.hub import Api, HubError
from live.prepare import LEVELS, prepare

log = logging.getLogger("live")
IDLE, READY, LOADED, READY_FOR_GAMEPLAY, FINISHED_PLAY = 0, 1, 3, 4, 6
PASSED, QUIT = 3, 5  # SpectatedUserState
SEND_EVERY_MS = 200  # lazer's SpectatorClient.TIME_BETWEEN_SENDS
LEAD_MS = 1500  # frames go out this far ahead of the room's clock; viewers line them up by gameplay time
VERSION_HASH = "osu_rl-ai"
MISS = "1"  # HitResult.Miss in spectator headers


def meta(osu):
    """Artist, title, difficulty and mapper from the .osu bytes."""
    found = dict(re.findall(r"^(Artist|Title|Version|Creator):(.*?)\s*$", osu.decode("utf-8", "replace"), re.M))
    return dict(artist=found.get("Artist", ""), title=found.get("Title", ""), version=found.get("Version", ""), creator=found.get("Creator", ""))


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
        self.live = None  # what is happening in the current play, see _narrate
        self.events, self.event_id = [], 0
        self.new_event = asyncio.Event()

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

    async def solo(self, beatmap_id, level=None, mods=()):
        """Plays one beatmap on its own (not in a room), spectatable like any lazer player. Returns at once;
        the play reports through events."""
        if self.room:
            raise ValueError("leave the room first")
        if self.playing and not self.playing.done():
            raise ValueError("already playing")
        if level and level not in LEVELS:
            raise ValueError(f"level must be one of {list(LEVELS)}")
        self.playing = asyncio.create_task(self._solo(int(beatmap_id), level or self.level, list(mods)))
        return self.status()

    def stop(self):
        if self.playing and not self.room:
            self.playing.cancel()
        return self.status()

    async def events_after(self, after, wait=25.0):
        """Events with id > after; waits up to `wait` seconds for one."""
        if not any(e["id"] > after for e in self.events):
            try:
                await asyncio.wait_for(self.new_event.wait(), wait)
            except TimeoutError:
                pass
        return [e for e in self.events if e["id"] > after]

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
                    now=self.live, last_result=self.last_result)

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
        play.update(mods=mods, item=item, meta=meta(osu))
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
        await self._stream(play, token)
        body = dict(play["submission"], mods=play["mods"])
        result = await self.api.request("PUT", f"/api/v2/rooms/{room}/playlist/{item['id']}/scores/{token}", json=body)
        self._finish(play, result)
        await self.mp.invoke("ChangeState", FINISHED_PLAY)  # the round's results open once everyone is here

    async def _solo(self, beatmap_id, level, mods):
        try:
            osu = await self.api.request("GET", f"/osu/{beatmap_id}", raw=True)
            md5 = hashlib.md5(osu).hexdigest()
            self._emit("preparing", beatmap_id=beatmap_id, level=level, **meta(osu))
            play = await asyncio.to_thread(prepare, osu, mods, level)
            play.update(mods=mods, item=dict(beatmapID=beatmap_id, rulesetID=0, beatmapChecksum=md5), meta=meta(osu))
            r = await self.api.request("POST", f"/api/v2/beatmaps/{beatmap_id}/solo/scores", data=dict(
                version_hash=VERSION_HASH, beatmap_hash=md5, ruleset_id="0"))
            await self._stream(play, r["id"])
            result = await self.api.request("PUT", f"/api/v2/beatmaps/{beatmap_id}/solo/scores/{r['id']}", json=dict(play["submission"], mods=mods))
            self._finish(play, result)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.exception("solo play failed")
            self._emit("error", beatmap_id=beatmap_id, message=str(e))

    async def _stream(self, play, token):
        """Sends the play as a spectator stream in real time (lazer's play session), narrating as it goes."""
        item = play["item"]
        state = {"beatmapID": item["beatmapID"], "rulesetID": item.get("rulesetID", 0), "mods": play["mods"], "state": 1,
                 "maximumStatistics": play["maximumStatistics"]}
        await self.spectator.invoke("BeginPlaySessionV2", token, state)
        start_ms = min(0, play["frames"][0][0])  # the gameplay clock starts here (lead-in before the first object)
        t0 = time.monotonic()
        narrator = asyncio.create_task(self._narrate(play, t0, start_ms))
        try:
            for at, bundle in bundles(play):
                delay = (at - start_ms - LEAD_MS) / 1000 - (time.monotonic() - t0)
                if delay > 0:
                    await asyncio.sleep(delay)
                await self.spectator.send("SendFrameDataV2", token, bundle)
            end = (play["frames"][-1][0] - start_ms) / 1000 - (time.monotonic() - t0)
            await asyncio.sleep(max(0, end))
            await narrator
            await self.spectator.invoke("EndPlaySessionV2", token, PASSED)
        except asyncio.CancelledError:
            narrator.cancel()
            self.live = None
            self._emit("aborted", beatmap_id=item["beatmapID"])
            try:
                await self.spectator.invoke("EndPlaySessionV2", token, QUIT)
            except HubError:
                pass
            raise

    async def _narrate(self, play, t0, start_ms):
        """Follows the play's scoring timeline in gameplay time: keeps self.live current and emits events."""
        info = dict(beatmap_id=play["item"]["beatmapID"], level=play["level"], mods=[m["acronym"] for m in play["mods"]], **play.get("meta", {}))
        end = play["frames"][-1][0]
        self.live = dict(info, progress=0.0, combo=0, max_combo=0, accuracy=1.0, misses=0)
        self._emit("started", **info)
        combo = 0
        for h in play["headers"]:
            await asyncio.sleep(max(0.0, t0 + (h["time"] - start_ms) / 1000 - time.monotonic()))
            misses = h["statistics"].get(MISS, 0)
            if h["combo"] < combo:
                self._emit("combo_break", combo=combo, misses=misses, accuracy=h["accuracy"])
            combo = h["combo"]
            self.live.update(progress=round(min(1.0, max(0.0, h["time"] / end)), 3), combo=combo, max_combo=h["maxCombo"],
                             accuracy=h["accuracy"], misses=misses)

    def _finish(self, play, result):
        s, item = play["submission"], play["item"]
        stats = s["statistics"]
        self.last_result = dict(beatmap=item["beatmapID"], level=play["level"], accuracy=s["accuracy"], rank=s["rank"], total_score=s["total_score"],
                                max_combo=s["max_combo"], statistics=stats, score_id=result.get("id"))
        self.live = None
        self._emit("finished", **play.get("meta", {}), **{k: v for k, v in self.last_result.items() if k != "statistics"},
                   misses=stats.get("miss", 0), full_combo=not stats.get("miss") and not stats.get("large_tick_miss"))
        log.info("finished: %s", self.last_result)

    def _emit(self, type, **data):
        self.event_id += 1
        self.events = [*self.events[-199:], dict(id=self.event_id, type=type, time=time.time(), **data)]
        self.new_event.set()
        self.new_event = asyncio.Event()

    async def close(self):
        for h in (self.mp, self.spectator):
            if h:
                await h.close()
        await self.api.close()
