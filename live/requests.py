"""Viewers' map requests for the stream: a queue that Miyoria (or anything with the control token) fills over HTTP,
played one after another in our lazer fork by the vision student, the same one live.stream plays (vision only since
2026-11-17: it sees nothing but the fork's window, captured even while covered, and its moves go over the bridge; the
state model, which reads the .osu, no longer plays). The student's view for OBS (live.vision_view, style "neuro")
shows the map and its cursor. A request's feel is kept but no longer changes the play: live.humanize worked on the
state model's moves.

    python -m live.stream --requests 8766        # requests come first; between them the student plays its set
    python -m live.requests --port 8766          # requests only (no student model needed); the view on 8770

A request names a beatmap (one difficulty) by its id on osu.ppy.sh. Its set is fetched from a beatmap mirror
(OSU_RL_MIRROR_LOOKUP / OSU_RL_MIRROR_DOWNLOAD, catboy.best by default) with its music and background, imported into
the fork once, and kept in runs/requests/maps/. Only osu!standard maps.

HTTP (Authorization: Bearer <OSU_RL_CONTROL_TOKEN>, as live.service):
    GET  /queue                                   -> {now, queue, recent, feels, held}
    POST /request {beatmap_id, by, by_id?, feel?} -> {item, position}   (fetches the map first: a few seconds)
    POST /skip                                    -> {skipped}           (the request being played now)
    POST /clear   {all?}                          -> {cleared, skipped}  (the waiting ones; all: the one playing too)
Errors are {"error": "<why, in words for viewers>"} with status 400 (or 401 without the token).

While the AI is in a multiplayer room, requests wait: the stream is on the room then. They are still taken, and
/queue says so (held). Asked at most every 5 s of live.room_play (the fork at a room, 127.0.0.1:8767) and
live.service (127.0.0.1:8765); OSU_RL_ROOM_URLS (comma separated) changes where. The fork's own screen counts too:
the same fork sits in rooms (live.room_play), and starting a map there would take it out of the room.

The fork plays a request only when it is not logged in, or logged into the AI's account (OSU_RL_AI_USERNAME, none of
OSU_RL_BLOCKED_ACCOUNTS) on the private server (OSU_RL_SERVER): live.room_play.solo_problem, asked before every
request. The fork is built not to submit these plays (OsuRLPlayer takes no score token); logged in, the server still
sees them being played (spectating).
"""
import argparse
import hashlib
import http.server
import json
import os
import re
import shutil
import socketserver
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zipfile

from live.humanize import DEFAULT_FEEL, FEELS

DIR = os.path.join("runs", "requests")
MAPS = os.path.join(DIR, "maps")
STATE = os.path.join(DIR, "queue.json")
LOOKUP = os.environ.get("OSU_RL_MIRROR_LOOKUP", "https://catboy.best/api/v2/b/{id}")
DOWNLOAD = os.environ.get("OSU_RL_MIRROR_DOWNLOAD", "https://catboy.best/d/{set}n")
AGENT = "osu_rl-requests/1.0"  # the mirror refuses clients that do not name themselves
START_TIMEOUT = 90  # s from "play" until the game clock runs (loading a map with its music)
RECENT = 10
ROOM_URLS = os.environ.get("OSU_RL_ROOM_URLS", "http://127.0.0.1:8767,http://127.0.0.1:8765")


class RequestError(Exception):
    """A request refused, with why in words for the person who asked."""


def fetch(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise RequestError("找不到這張譜") from e
        raise RequestError(f"譜面鏡像站回應 {e.code}") from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise RequestError(f"連不到譜面鏡像站（{e}）") from e


def lookup(beatmap_id):
    """The mirror's record of one difficulty: {id, set, md5, stars, length}."""
    try:
        b = json.loads(fetch(LOOKUP.format(id=int(beatmap_id)), timeout=20))
    except ValueError as e:
        raise RequestError("譜面鏡像站的回應看不懂") from e
    if not isinstance(b, dict) or not b.get("beatmapset_id"):
        raise RequestError("找不到這張譜")
    if b.get("mode_int", 0) != 0:
        raise RequestError("只能點 osu!（standard）的譜")
    return dict(id=int(b["id"]), set=int(b["beatmapset_id"]), md5=str(b.get("checksum") or ""),
                stars=float(b.get("difficulty_rating") or 0), length=int(b.get("total_length") or 0))


def osu_meta(data):
    found = dict(re.findall(rb"^(Artist|Title|Version|BeatmapID|Mode):(.*?)\s*$", data, re.M))
    return {k.decode(): v.decode("utf-8", "replace") for k, v in found.items()}


def title_of(data):
    m = osu_meta(data)
    return f"{m.get('Artist', '?')} - {m.get('Title', '?')} [{m.get('Version', '?')}]"


def fetch_map(info):
    """Downloads the set (once) and keeps this difficulty's .osu: -> (osz path, osu path)."""
    os.makedirs(MAPS, exist_ok=True)
    osz = os.path.join(MAPS, f"{info['set']}.osz")
    if not os.path.exists(osz):
        data = fetch(DOWNLOAD.format(set=info["set"]), timeout=120)
        if not data.startswith(b"PK"):
            raise RequestError("下載到的不是譜面檔")
        tmp = osz + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, osz)
    with zipfile.ZipFile(osz) as z:
        for name in z.namelist():
            if not name.lower().endswith(".osu"):
                continue
            data = z.read(name)
            md5 = hashlib.md5(data).hexdigest()
            if md5 == info["md5"] or (not info["md5"] and osu_meta(data).get("BeatmapID") == str(info["id"])):
                path = os.path.join(MAPS, f"{info['id']}.osu")
                with open(path, "wb") as f:
                    f.write(data)
                return osz, path
    raise RequestError("譜面檔裡找不到這個難度（可能是舊版本）")


def fork_problem(clock):
    """Why the fork may not play a request now (live.room_play.solo_problem on its bridge status), or None."""
    from live.room_play import guard_settings, solo_problem
    return solo_problem(clock.call(cmd="status", timeout=10) or {}, *guard_settings())


def fork_in_room(clock):
    from live.room_play import in_multiplayer
    try:
        return in_multiplayer((clock.call(cmd="status", timeout=10) or {}).get("screen"))
    except Exception:  # no answer: the play itself will find out
        return False


class RoomWatch:
    """The multiplayer room the AI is in (live.service: its id; live.room_play: true), or None, asked of each url's
    /status at most every `every` seconds; one that does not answer counts as not in a room."""

    def __init__(self, urls=ROOM_URLS, token="", every=5.0):
        if isinstance(urls, str):
            urls = urls.split(",")
        self.urls, self.token, self.every = [u.strip().rstrip("/") for u in urls if u.strip()], token, every
        self.at, self.room = 0.0, None

    def ask(self, url):
        req = urllib.request.Request(url + "/status", headers={"Authorization": f"Bearer {self.token}"})
        try:
            with urllib.request.urlopen(req, timeout=3) as r:
                return json.loads(r.read()).get("room") or None
        except (OSError, ValueError):
            return None

    def __call__(self):
        if time.time() - self.at >= self.every:
            self.at = time.time()
            self.room = next((room for room in map(self.ask, self.urls) if room), None)
        return self.room


class Requests:
    """The queue: waiting requests, the one playing now and the last few played, kept in runs/requests/queue.json."""

    def __init__(self, path=STATE, per_person=2, limit=30, feel=DEFAULT_FEEL, hold=lambda: None):
        self.path, self.per_person, self.limit, self.feel = path, per_person, limit, feel
        self.hold = hold  # () -> why requests wait now (live.service's room), or None
        self.lock = threading.Lock()
        self.queue, self.now, self.recent, self.next_no = [], None, [], 1
        self.skip = threading.Event()
        self.arrived = threading.Event()
        try:
            s = json.load(open(path, encoding="utf-8"))
            self.queue, self.recent, self.next_no = s["queue"], s["recent"], s["next_no"]
            if s.get("now"):  # cut off by a restart: played again first
                self.queue.insert(0, dict(s["now"], state="waiting"))
        except (OSError, KeyError, ValueError):
            pass

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(dict(queue=self.queue, now=self.now, recent=self.recent, next_no=self.next_no), f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)

    def snapshot(self):
        room = self.hold()
        with self.lock:
            return dict(now=self.now, queue=list(self.queue), recent=list(self.recent), feels=list(FEELS), held=room)

    def add(self, beatmap_id, by, by_id=None, feel=None):
        """Fetches the map and queues it: -> (item, 1-based position)."""
        try:
            beatmap_id = int(beatmap_id)
        except (TypeError, ValueError):
            raise RequestError("譜面 id 要是數字")
        feel = feel or self.feel
        if feel not in FEELS:
            raise RequestError(f"手感只能是 {', '.join(FEELS)}")
        who = str(by_id or by or "?")
        with self.lock:
            if len(self.queue) >= self.limit:
                raise RequestError(f"點播已經排了 {self.limit} 首，等一下再點")
            if sum(1 for q in self.queue if q["by_id"] == who) >= self.per_person:
                raise RequestError(f"每個人最多同時點 {self.per_person} 首")
        info = lookup(beatmap_id)
        osz, path = fetch_map(info)  # outside the lock: a download takes seconds
        data = open(path, "rb").read()
        with self.lock:
            if any(q["beatmap_id"] == beatmap_id for q in self.queue) or (self.now and self.now["beatmap_id"] == beatmap_id):
                raise RequestError("這張譜已經在點播裡了")
            item = dict(no=self.next_no, beatmap_id=beatmap_id, set_id=info["set"], md5=hashlib.md5(data).hexdigest(),
                        title=title_of(data), stars=round(info["stars"], 2), length=info["length"], by=str(by or "?"),
                        by_id=who, feel=feel, osu=path, osz=osz, state="waiting", at=time.time(), result=None)
            self.next_no += 1
            self.queue.append(item)
            self.save()
            position = len(self.queue) + (1 if self.now else 0)
        self.arrived.set()
        return item, position

    def take(self):
        """The next request, now marked as playing, or None (none waiting, or held while the AI is in a room)."""
        if self.hold() is not None:
            return None
        with self.lock:
            if not self.queue:
                self.arrived.clear()
                return None
            self.now = dict(self.queue.pop(0), state="playing", started=time.time())
            self.skip.clear()
            self.save()
            return dict(self.now)

    def finish(self, result):
        with self.lock:
            if self.now is None:
                return
            done = dict(self.now, state="skipped" if result.get("skipped") else ("done" if result.get("accuracy") is not None else "failed"),
                        result=result, ended=time.time())
            self.recent = (self.recent + [done])[-RECENT:]
            self.now = None
            self.save()

    def skip_now(self):
        with self.lock:
            if self.now is None:
                return False
        self.skip.set()
        return True

    def clear(self, everything=False):
        with self.lock:
            n = len(self.queue)
            self.queue = []
            self.save()
        return n, self.skip_now() if everything else False


def serve(requests, port, token, host="127.0.0.1"):
    """The control API on its own threads."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def reply(self, code, data):
            body = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def authorized(self):
            if self.headers.get("Authorization") == f"Bearer {token}":
                return True
            self.reply(401, {"error": "unauthorized"})
            return False

        def do_GET(self):
            if not self.authorized():
                return
            if self.path.split("?")[0] == "/queue":
                return self.reply(200, requests.snapshot())
            self.reply(404, {"error": "not found"})

        def do_POST(self):
            if not self.authorized():
                return
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}") if n else {}
                if not isinstance(body, dict):
                    raise ValueError
            except ValueError:
                return self.reply(400, {"error": "看不懂的請求"})
            path = self.path.split("?")[0]
            try:
                if path == "/request":
                    item, position = requests.add(body.get("beatmap_id"), body.get("by"), body.get("by_id"), body.get("feel"))
                    return self.reply(200, dict(item=item, position=position))
                if path == "/skip":
                    return self.reply(200, dict(skipped=requests.skip_now()))
                if path == "/clear":
                    cleared, skipped = requests.clear(bool(body.get("all")))
                    return self.reply(200, dict(cleared=cleared, skipped=skipped))
            except RequestError as e:
                return self.reply(400, {"error": str(e)})
            except Exception as e:  # never take the stream down over one request
                return self.reply(500, {"error": f"{type(e).__name__}: {e}"})
            self.reply(404, {"error": "not found"})

    class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    server = Server((host, port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def import_set(clock, osz):
    """lazer deletes what it imported: it gets a copy of the set."""
    tmp = tempfile.mkdtemp(prefix="osu_rl_import_")
    copy = os.path.join(tmp, os.path.basename(osz))
    shutil.copy(osz, copy)
    r = clock.call(cmd="import", paths=[copy], timeout=300)
    time.sleep(2)
    return r


def play_request(clock, item, player, say=print, skip=None, out_dir=os.path.join(DIR, "plays")):
    """Plays one request in the fork with the vision student (player: a live.vision_play.VisionPlayer with
    background=True): it sees only the fork's window (captured even while covered) and its moves go over the bridge.
    -> result dict (accuracy, statistics, misses; error / skipped)."""
    import live.tablet
    problem = fork_problem(clock)
    if problem:
        return dict(accuracy=None, error=f"not played: {problem}")
    live.tablet.FORK_PID = (clock.call(cmd="status", timeout=10) or {}).get("pid")  # capture this copy's window
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{time.strftime('%Y%m%d-%H%M%S')}-{item['beatmap_id']}-lazer.json")
    inp = "bridge" if player.background else None  # live.stream --foreground: the tablet, the fork in front
    try:
        clock.play(item["osu"], ["NF"], out, input=inp)
    except RuntimeError as e:
        if "no beatmap" not in str(e):
            raise
        say(f"importing set {item['set_id']}: {import_set(clock, item['osz'])}")
        clock.play(item["osu"], ["NF"], out, input=inp)
    say(f"request #{item['no']} {item['title']} by {item['by']}: the vision student plays it")
    player.attach(wait=False)  # the playfield of the window as it is now
    if player.view is not None:
        player.view.set_map(item["osu"])  # the "neuro" view draws the map (only the picture for OBS)
    length = float(item.get("length") or 600)
    result = player.play(clock, start_timeout=START_TIMEOUT, max_seconds=length + 30,
                         stop=(lambda: skip.is_set()) if skip is not None else None)
    if result is None:
        clock.call(cmd="menu")
        return dict(accuracy=None, error=f"the map did not start within {START_TIMEOUT} s")
    skipped = result["stopped"]
    if skipped or result["timed_out"]:
        clock.call(cmd="menu")
    clock.done_event.wait(30 if not skipped else 10)
    done = clock.done
    if skipped:
        return dict(accuracy=None, skipped=True)
    if not done:
        clock.call(cmd="menu")
        return dict(accuracy=None, error="the game did not report the end of the play")
    if done.get("quit"):
        return dict(accuracy=None, error="the play was left before the end")
    stats = done.get("statistics") or {}
    return dict(accuracy=done["accuracy"], statistics=stats, misses=int(stats.get("Miss", stats.get("miss", 0)) or 0))


def student_player(ckpt="", view=8770, background=True):
    """The vision student as live.stream plays it (same model and key settings): a live.vision_play.VisionPlayer."""
    from live.stream import student_ckpt
    from live.vision_play import VisionPlayer
    ckpt = ckpt or student_ckpt()
    player = VisionPlayer(ckpt, click_th=-0.75, min_hold=5, press="first", view=view or None, view_style="neuro",
                          background=background)
    player.hold_press, player.smooth = 8.0, 0.5  # live.stream's defaults (--hold-press, --smooth)
    player.hold_th = -1.5  # let go of a held key under this hold logit (live.vision_play VisionPlayer.hold_th)
    print(f"vision student: {ckpt}", flush=True)
    return player


def settings():
    """live/.env and OSU_RL_* environment variables, as live.service reads them."""
    env = {}
    if os.path.exists("live/.env"):
        env = dict(line.strip().split("=", 1) for line in open("live/.env", encoding="utf-8") if "=" in line)
    env.update({k: v for k, v in os.environ.items() if k.startswith("OSU_RL_")})
    return env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8766, help="the control API")
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 when Miyoria runs on another machine (over Tailscale)")
    p.add_argument("--view", type=int, default=8770, metavar="PORT", help="the student's view for OBS; 0: off")
    p.add_argument("--feel", default=DEFAULT_FEEL, choices=list(FEELS), help="the feel of a request that names none")
    p.add_argument("--per-person", type=int, default=2)
    p.add_argument("--ckpt", default="", help="the vision student (default: the one live.stream plays, live.stream.student_ckpt)")
    a = p.parse_args()
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    token = settings().get("OSU_RL_CONTROL_TOKEN", "")
    if len(token) < 16:
        raise SystemExit("set OSU_RL_CONTROL_TOKEN (16+ characters) in the environment or live/.env")
    from live.bridge import SKIN_NAME, BridgeClock, launch, import_skin
    requests = Requests(per_person=a.per_person, feel=a.feel, hold=RoomWatch(token=token))
    serve(requests, a.port, token, a.host)
    player = student_player(a.ckpt, a.view)
    print(f"requests on http://{a.host}:{a.port}/ ({len(requests.queue)} waiting)", flush=True)
    clock = None
    while True:
        if clock is not None and clock.connected and fork_in_room(clock):
            time.sleep(5)  # the fork is in a room (live.room_play plays there): requests wait
            continue
        item = requests.take()
        if item is None:
            if requests.hold() is not None:
                time.sleep(5)  # the AI is in a room: requests wait (arrived stays set)
            else:
                requests.arrived.wait(5)
            continue
        if clock is None or not clock.connected:
            try:
                clock = BridgeClock(connect_timeout=3)
            except SystemExit:
                print("the fork is not running: starting it", flush=True)
                launch()
                clock = BridgeClock(connect_timeout=180)
                time.sleep(10)
            setup = dict(cmd="setup", skin=SKIN_NAME, window="borderless", hud=False, cursor=False, background=False)
            r = clock.call(**setup)
            if not r.get("ok") and "no skin" in str(r.get("error")):
                import_skin(clock)
                clock.call(**setup)
        try:
            result = play_request(clock, item, player, skip=requests.skip)
        except Exception as e:
            result = dict(accuracy=None, error=f"{type(e).__name__}: {e}")
            clock = None if clock is None or not clock.connected else clock
        requests.finish(result)
        print(f"request #{item['no']}: {result}", flush=True)


if __name__ == "__main__":
    main()
