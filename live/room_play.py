"""The AI in a multiplayer room like a person: our fork, logged into the AI's account on the private server, sits in a
lazer room (or song select) like any player, and the vision student plays it from the screen through the virtual
tablet (live/tablet.py). The stream shows that window: song select, the room, the play itself and the results, all
real.

    python -m live.room_play                      # the fork must be up: OSU_RL_LAZER_SERVER=<server> python -m live.bridge launch
    python -m live.room_play --view 8770 --ckpt runs/improve/r033_aim6/best_cal.pt

Vision only (2026-11-17): the student (live.vision_play.VisionPlayer, the model live.stream plays) sees nothing but
the fork's picture and decides every move and key from it, like a person; nothing is read from the map. The state
model, which played the .osu it read, no longer plays anywhere live. So there is nothing to prepare per map, mods
are simply what it sees (it learnt on No Fail maps without mods; DT / HR and the like play worse), and there is no
feel any more (live.humanize worked on the state model's moves).

Everything else it knows comes from the fork's bridge (live.bridge; the fork reports any play, OsuRLWatcher): which
screen and map, the play's state and its clock (only so the student knows when a map is under way). tosu cannot read
a local build of lazer (it has memory offsets only for official versions).

Joining the room, readying up and leaving are left to whoever runs it, in that lazer: stock lazer has no way in for
that. The fork must be in front while it plays: the screen is captured where the window is, and the tablet moves the
real cursor and stops (lets go of the buttons) the moment anything else is in front. So it puts the fork in front
(live.tablet.bring_front) while a play loads, as the play starts, and again every FOCUS_EVERY seconds while something
else is (lazer also pauses a solo play whose window is behind others).

Any play in the fork counts: a room's, or a map somebody started by hand in song select. Both are submitted to the
private server like a person's (a solo one too: SubmitSoloScoreRequest, seen on shdcloud.top 2026-11-17); only the
fork's own plays (OsuRLPlayer) never are. The same fork also plays maps of its own between rooms (live.stream,
live.requests: the bridge starts them, the student plays them in the background); those are left alone.

Never on someone's account and never on the official servers: before every play it asks the bridge which server and
account the fork is on, and plays only when the server is the private one (OSU_RL_SERVER), the account is the AI's
(OSU_RL_AI_USERNAME, or --account) and none of --blocked. The fork uses the private server when it was started with
OSU_RL_LAZER_SERVER set; a release build without it is on the official servers, and is refused.

HTTP on 127.0.0.1:8767 (Authorization: Bearer <OSU_RL_CONTROL_TOKEN>):
    GET  /status                      -> {user, server, problem, screen, room, room_id, multi, map, ready, playing, model, last}
    GET  /rooms                       -> {rooms: [{id, name, host, has_password, participants, status}]}
    POST /join {room_id, password?}   -> the status once the fork is in that room (it readies up by itself)
    POST /leave                       -> the status, out of the room and multiplayer
    POST /feel                        -> 400: there is no feel with the vision student
`room` is true while the fork is on its multiplayer screens: live.requests holds viewers' requests meanwhile. Rooms
and joining go through the same account check as playing; once in a room it readies up whenever it is idle and the
map is downloaded (lazer fetches a missing one), every round (--no-auto-ready: whoever runs the room readies).
"""
import argparse
import http.server
import json
import os
import socketserver
import threading
import time
from collections import deque

import numpy as np

DIR = os.path.join("runs", "room_play")
# the fork's top screen while in lazer's multiplayer: Multiplayer (lounge, room, song select), then its own
# MultiplayerPlayerLoader / MultiplayerPlayer / MultiplayerResultsScreen (MultiplayerTeamResultsScreen) and, watching a
# room, MultiSpectator*; seen on shdcloud.top 2026-11-17
MULTI_PREFIXES = ("Multiplayer", "MultiSpectator")
FOCUS_EVERY = 2.0  # s between asking the fork back in front while something else is
SPEED = {"DT": 1.5, "NC": 1.5, "HT": 0.75, "DC": 0.75}
NO_FEEL = "vision only: the student plays from the screen as it plays, there is no feel to change"


def in_multiplayer(screen):
    return bool(screen) and screen.startswith(MULTI_PREFIXES)


def bridge_play(screen):
    """A play the bridge started (live.stream, live.requests: OsuRLPlayerLoader / OsuRLPlayer), driven over the bridge."""
    return bool(screen) and screen.startswith("OsuRL")


def someones_play(screen):
    """A play the bridge did not start, loading or under way: a room's, or one somebody started in the fork by hand
    (SoloPlayerLoader / SoloPlayer); live.room_play plays it."""
    return bool(screen) and not bridge_play(screen) and screen.endswith(("Player", "PlayerLoader"))


def rate_of(mods):
    return next((SPEED[m] for m in mods if m in SPEED), 1.0)


def same_name(a, b):
    return (a or "").strip().lower() == (b or "").strip().lower()


def same_server(a, b):
    norm = lambda u: (u or "").strip().rstrip("/").lower()
    return bool(norm(a)) and norm(a) == norm(b)


def seat_problem(status, account, blocked, server):
    """Why the fork (its bridge status) may not be played, or None."""
    if not account.strip():
        return "no AI account set (OSU_RL_AI_USERNAME or --account)"
    if any(same_name(b, account) for b in blocked):
        return "the AI account is one of --blocked"
    if not server.strip():
        return "no private server set (OSU_RL_SERVER or --server)"
    if not same_server(status.get("server"), server):  # first: the official servers are the worst place to be
        return f"the fork is on {status.get('server') or 'an unknown server'}, not the private server (start it with OSU_RL_LAZER_SERVER)"
    user = status.get("user")
    if not user or status.get("online", "Online") != "Online":  # lazer calls nobody "Guest" while offline
        return f"the fork is not logged in ({status.get('online') or 'no user'})"
    if any(same_name(b, user) for b in blocked):
        return f"the fork is logged in as a blocked account ({user})"
    if not same_name(user, account):
        return f"the fork is logged in as {user}, not the AI account"
    return None


def solo_problem(status, account, blocked, server):
    """Why the fork may not play a map of its own (live.stream, live.requests), or None. Not logged in, nothing reaches
    any account; logged in, only the AI's account on the private server, as in a room (seat_problem). Never while it
    is in a room: starting a map there would take it out of the room."""
    if in_multiplayer(status.get("screen")):
        return "the fork is in a multiplayer room"
    if someones_play(status.get("screen")):  # started by hand: live.room_play plays it, a map now would end it
        return f"a play started in the fork is on ({status.get('screen')})"
    if status.get("online") == "Offline":  # logged out (still connecting or failing with a saved login is not)
        return None
    return seat_problem(status, account, blocked, server)


def guard_settings():
    """(account, blocked, server) for the guards, from live/.env and OSU_RL_* (an OSU_RL_AI_USERNAME set in the
    environment wins over live/.env)."""
    from live.requests import settings
    env = settings()
    return (env.get("OSU_RL_AI_USERNAME", ""), [b for b in env.get("OSU_RL_BLOCKED_ACCOUNTS", "").split(",") if b.strip()],
            env.get("OSU_RL_SERVER", ""))


def map_end(path):
    """The map's last object end (gameplay ms) from its .osu in the game's file store, or None."""
    if not path:
        return None
    try:
        import vision.student as S
        return float(S.scene(path).end.max())
    except Exception:
        return None


class GameClock:
    """Wall clock (time.perf_counter, the same QPC the game stamps its samples with) -> gameplay seconds, from the
    bridge's (qpc, gameplay ms) samples at a known rate."""

    def __init__(self):
        self.samples, self.rate = deque(maxlen=64), 1.0

    def reset(self, rate=1.0):
        self.samples.clear()
        self.rate = rate or 1.0

    def add(self, qpc, t_ms):
        self.samples.append((qpc, t_ms))

    @property
    def model(self):
        """(intercept, rate): gameplay seconds now = (perf_counter() - intercept) * rate; None without samples."""
        if len(self.samples) < 3:
            return None
        s = np.array(self.samples)
        return float(np.median(s[:, 0] - s[:, 1] / 1000 / self.rate)), self.rate


class SeatBridge:
    """The fork's bridge for room_play: commands (call) and the events of any play (OsuRLWatcher): seat (state, map,
    mods, rate), clock, live and played."""

    def __init__(self, connect_timeout=10):
        self.seat, self.live, self.played, self.play_no = None, None, deque(maxlen=20), 0
        self.clock = GameClock()
        self.conn = self.connect(connect_timeout)

    def connect(self, connect_timeout):
        from live.bridge import BridgeClock
        bridge = self

        class Clock(BridgeClock):
            def run(inner):
                try:
                    for line in inner.file:
                        m = json.loads(line)
                        ev = m.get("ev")
                        if ev is None:
                            with inner.reply_ready:
                                inner.replies[m.get("id")] = m
                                inner.reply_ready.notify_all()
                        else:
                            bridge.on_event(ev, m)
                except OSError:  # the fork quit or crashed: reconnect() makes a new connection
                    pass
                inner.state = None

        return Clock(connect_timeout=connect_timeout)

    def reconnect(self, connect_timeout=3):
        """A new connection once the fork is back (raises SystemExit while it is not)."""
        self.seat, self.live = None, None
        self.conn = self.connect(connect_timeout)

    def call(self, **cmd):
        return self.conn.call(**cmd)

    def status(self):
        return self.conn.call(cmd="status", timeout=10)

    @property
    def connected(self):
        return self.conn.connected

    def on_event(self, ev, m):
        if ev == "seat":
            if self.seat is None or self.seat.get("md5") != m.get("md5"):
                # a new play (its first event, "paused" before the clock starts or "playing"); pausing and resuming
                # the same play keeps the number
                self.play_no += 1
                self.clock.reset(m.get("rate") or rate_of(m.get("mods") or []))
            elif m.get("state") == "playing":
                self.clock.rate = m.get("rate") or self.clock.rate
            self.seat = m
        elif ev == "clock":
            self.clock.add(m["qpc"], m["t"])
        elif ev == "live":
            self.live = m
        elif ev == "played":
            self.played.append(m)
            self.seat, self.live = None, None


class SeatClock:
    """live.vision_play's clock interface (playing, model) over one play the fork reports (SeatBridge). end_ms: the
    map's last object (gameplay ms; only for when to stop, not how to play): past it the play counts as over, so the
    student does not go on clicking while the results come up."""

    def __init__(self, bridge, md5, end_ms=None):
        self.bridge, self.md5, self.end_ms = bridge, md5, end_ms

    def past_end(self):
        model = self.bridge.clock.model
        if self.end_ms is None or model is None:
            return False
        icept, rate = model
        return (time.perf_counter() - icept) * rate * 1000 > self.end_ms + 1000

    @property
    def on(self):
        """The play is still there (loading, paused or playing) and not past the map's end."""
        seat = self.bridge.seat
        return seat is not None and seat.get("md5") == self.md5 and not self.past_end()

    @property
    def playing(self):
        return self.on and self.bridge.seat.get("state") == "playing"

    @property
    def model(self):
        return self.bridge.clock.model


class Seat:
    def __init__(self, bridge, player, account, blocked=(), server="", say=print, overlay=os.path.join(DIR, "overlay.txt"),
                 auto_ready=True):
        self.bridge = bridge  # SeatBridge (or anything with status(), call(), seat, clock, played)
        self.player = player  # live.vision_play.VisionPlayer, background=False: screen capture and the tablet
        self.account, self.blocked, self.server = account, list(blocked), server
        self.say, self.overlay, self.auto_ready = say, overlay, auto_ready
        self.playing, self.last, self.problem, self.fork = None, None, None, {}
        self.done = set()  # (md5, play number) of plays already played or refused
        self.asked_front = self.asked_ready = 0.0

    # --- rooms (Miyoria's /queue osu join | leave and tools) --------------------------------------------------
    def rooms(self):
        """The open multiplayer rooms on the server, as the fork's lounge lists them."""
        if self.refresh():
            raise ValueError(self.problem)
        r = self.bridge.call(cmd="rooms", timeout=20)
        if not r.get("ok"):
            raise ValueError(r.get("error") or "no rooms")
        return r["rooms"]

    def join(self, room_id, password=None, wait=20.0):
        """Into room room_id (only on the AI's account on the private server). -> status once in it."""
        if self.refresh():
            raise ValueError(self.problem)
        if self.playing or someones_play(self.fork.get("screen")) or bridge_play(self.fork.get("screen")):
            raise ValueError("the AI is playing a map now")
        multi = self.fork.get("multi") or {}
        if multi.get("room_id") == int(room_id):
            return self.status()
        if multi:
            raise ValueError(f"already in room {multi.get('room_id')}: leave it first")
        r = self.bridge.call(cmd="join", room_id=int(room_id), password=password or "", timeout=20)
        if not r.get("ok"):
            raise ValueError(r.get("error") or "could not join")
        t0 = time.time()
        while time.time() - t0 < wait:
            time.sleep(0.5)
            self.refresh()
            if (self.fork.get("multi") or {}).get("room_id") == int(room_id):
                self.say(f"joined room {room_id} ({r.get('room')})")
                return self.status()
        raise ValueError(f"did not get into room {room_id} within {wait:.0f} s (wrong password?)")

    def leave(self):
        r = self.bridge.call(cmd="leave", timeout=10)
        if not r.get("ok"):
            raise ValueError(r.get("error") or "could not leave")
        if r.get("left"):
            self.say("left the room")
        time.sleep(1.0)
        self.refresh()
        return self.status()

    def ready_up(self):
        """In a room, idle, its map downloaded, the account fine: ready (every FOCUS_EVERY s at most). lazer sets the
        AI back to idle after each round, so it readies again for the next one."""
        multi = self.fork.get("multi") or {}
        if not self.auto_ready or self.problem or not multi or multi.get("state") != "Idle" \
                or multi.get("availability") != "LocallyAvailable" or time.time() - self.asked_ready < FOCUS_EVERY:
            return
        self.asked_ready = time.time()
        r = self.bridge.call(cmd="ready", timeout=10)
        if r.get("ok"):
            self.say(f"ready in room {multi.get('room_id')}")
        else:
            self.say(f"could not ready up: {r.get('error')}")

    # --- what the fork says -----------------------------------------------------------------------------------
    def status(self):
        f = self.fork
        return dict(user=f.get("user"), server=f.get("server"), problem=self.problem, screen=f.get("screen"),
                    room=in_multiplayer(f.get("screen")), map=f.get("map"),
                    ready=bool(f.get("md5")) and self.problem is None, preparing=False, playing=self.playing,
                    room_id=(f.get("multi") or {}).get("room_id"), multi=f.get("multi"),
                    model=os.path.basename(os.path.dirname(self.player.ckpt)) + "/" + os.path.basename(self.player.ckpt),
                    last=self.last)

    def refresh(self):
        """The fork's status now (map picked, screen, server, account) and the guard's verdict on it."""
        try:
            if not getattr(self.bridge, "connected", True):
                self.bridge.reconnect()
            self.fork = self.bridge.status() or {}
            self.problem = seat_problem(self.fork, self.account, self.blocked, self.server)
        except (Exception, SystemExit) as e:  # SystemExit: BridgeClock found no fork to connect to
            self.fork = {}
            self.problem = f"no answer from the fork's bridge ({type(e).__name__}: {e})"
        return self.problem

    # --- the window in front ----------------------------------------------------------------------------------
    def focus(self):
        """Puts the fork in front (live.tablet.bring_front). -> bool"""
        from live.tablet import bring_front
        try:
            if bring_front():
                return True
            self.say("the fork could not come in front")
        except (Exception, SystemExit) as e:  # SystemExit: no fork window
            self.say(f"bringing the fork in front failed: {type(e).__name__}: {e}")
        return False

    def front_early(self):
        """A play the fork will play is loading (or paused): the fork in front before it starts, every FOCUS_EVERY s
        at most. lazer pauses a solo play whose window is not in front, and the tablet sends nothing to a window
        behind others."""
        if self.problem or not someones_play(self.fork.get("screen")) or time.time() - self.asked_front < FOCUS_EVERY:
            return
        self.asked_front = time.time()
        if not self.fork.get("active"):
            self.focus()

    # --- playing ----------------------------------------------------------------------------------------------
    def play(self, seat):
        """The vision student plays the fork's play now, until it is over (finished, failed, quit or left): through
        pauses too (a solo play), each stretch of playing a VisionPlayer.play of its own."""
        md5, mods = seat.get("md5"), list(seat.get("mods") or [])
        title = self.fork.get("map") or md5
        clock = SeatClock(self.bridge, md5, map_end(self.fork.get("osu")))
        self.playing = dict(map=title, mods=mods, started=time.time())
        self.say(f"playing {title} {'+' + ''.join(mods) if mods else ''} (vision, {self.status()['model']})")
        self.focus()
        try:
            self.player.attach(wait=False)  # the playfield where the window is now; capture and the tablet
        except (Exception, SystemExit) as e:  # no window, no VMulti: this play is sat out, the loop goes on
            self.playing = None
            raise RuntimeError(f"could not attach to the fork: {e}") from None
        if self.player.view is not None and self.fork.get("osu"):
            try:
                self.player.view.set_map(self.fork["osu"])  # the "neuro" view draws the map (a picture for OBS only)
            except Exception as e:
                self.say(f"no map in the student's view: {type(e).__name__}: {e}")
        played_before = len(self.bridge.played)

        def lost_focus():
            """Something else came in front mid-play: ask the fork back until it is, or the play is over."""
            self.say("the fork is not in front: asking it back (the tablet let go of the buttons)")
            while clock.on:
                if self.focus():
                    return True
                time.sleep(FOCUS_EVERY)
            return False

        steps = 0
        try:
            while clock.on:
                r = self.player.play(clock, start_timeout=1.0, on_lost_focus=lost_focus)
                if r is not None:
                    steps += len(r["log"])
        finally:
            self.playing = None
        self.last = self.result(title, md5, played_before)
        self.say(f"played {title}: {self.last} ({steps} steps)")

    def result(self, title, md5, before, wait=15.0):
        """The play's numbers from the fork when its player leaves (results screen), else the last live numbers."""
        t0 = time.time()
        while time.time() - t0 < wait:
            for m in list(self.bridge.played)[before:]:
                if m.get("md5") == md5:
                    acc = m.get("accuracy")
                    return dict(map=title, accuracy=None if acc is None else round(acc * 100, 2), misses=m.get("misses"),
                                max_combo=m.get("max_combo"))
            time.sleep(0.2)
        live = self.bridge.live or {}
        acc = live.get("accuracy")
        return dict(map=title, accuracy=None if acc is None else round(acc * 100, 2), misses=live.get("misses"))

    # --- the loop ---------------------------------------------------------------------------------------------
    def write_overlay(self):
        st = self.status()
        lines = []
        if self.problem:
            lines.append("AI 不能上場（看終端機）")
        elif st["playing"]:
            lines.append(f"AI 正在打：{st['playing']['map']}")
        elif st["map"]:
            lines.append(f"AI 準備好了：{st['map']}")
        if st["last"] and st["last"].get("accuracy") is not None:
            lines.append(f"上一首：{float(st['last']['accuracy']):.2f}%")
        os.makedirs(os.path.dirname(self.overlay), exist_ok=True)
        tmp = self.overlay + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        os.replace(tmp, self.overlay)

    def step(self):
        seat = self.bridge.seat
        if seat is None:
            return
        if seat.get("state") != "playing":
            self.front_early()
            return
        play_id = (seat.get("md5"), self.bridge.play_no)
        if play_id in self.done:
            return
        self.done.add(play_id)  # tried once: not again until the next play
        if self.refresh():
            self.say(f"not playing: {self.problem}")
            return
        if bridge_play(self.fork.get("screen")):  # live.stream / live.requests play it over the bridge
            self.say(f"{self.fork.get('map') or seat.get('md5')} is played over the bridge: left alone")
            return
        self.play(seat)

    def run(self, poll=0.3):
        last_poll = last_overlay = 0.0
        while True:
            try:
                if self.playing is None and time.time() - last_poll > poll:
                    self.refresh()
                    self.ready_up()
                    last_poll = time.time()
                self.step()
            except Exception as e:  # never stop the stream over one map
                self.say(f"room_play: {type(e).__name__}: {e}")
                self.playing = None
                time.sleep(1)
            if time.time() - last_overlay > 0.5:
                self.write_overlay()
                last_overlay = time.time()
            time.sleep(0.02)


def serve(seat, port, token, host="127.0.0.1"):
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

        def ok(self):
            if self.headers.get("Authorization") == f"Bearer {token}":
                return True
            self.reply(401, {"error": "unauthorized"})
            return False

        def do_GET(self):
            if not self.ok():
                return
            path = self.path.split("?")[0]
            if path == "/status":
                return self.reply(200, seat.status())
            if path == "/rooms":
                try:
                    return self.reply(200, {"rooms": seat.rooms()})
                except Exception as e:
                    return self.reply(400, {"error": str(e)})
            self.reply(404, {"error": "not found"})

        def do_POST(self):
            if not self.ok():
                return
            path = self.path.split("?")[0]
            if path == "/feel":
                return self.reply(400, {"error": NO_FEEL})
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}") if n else {}
                if path == "/join":
                    return self.reply(200, seat.join(body.get("room_id"), body.get("password")))
                if path == "/leave":
                    return self.reply(200, seat.leave())
                if path == "/play":
                    return self.reply(400, {"error": "a single map goes into the request queue (live.requests)"})
            except (ValueError, TypeError) as e:
                return self.reply(400, {"error": str(e)})
            except Exception as e:  # never take the seat down over one call
                return self.reply(500, {"error": f"{type(e).__name__}: {e}"})
            self.reply(404, {"error": "not found"})

    class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    server = Server((host, port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def setup_fork(bridge, say):
    """The look the student learnt on: its skin, no HUD and no beatmap background (live.bridge launch sets the rest: no
    storyboard, hit lighting...). The menu cursor stays (someone runs the room)."""
    from live.bridge import SKIN_NAME, import_skin
    r = bridge.call(cmd="setup", skin=SKIN_NAME, hud=False, background=False)
    if not r.get("ok") and "no skin" in str(r.get("error")):
        say(f"importing the student skin: {import_skin(bridge)}")
        r = bridge.call(cmd="setup", skin=SKIN_NAME, hud=False, background=False)
    if not r.get("ok"):
        say(f"setup failed: {r}")


def main():
    import sys
    from live.requests import settings, student_player
    sys.stdout.reconfigure(encoding="utf-8")
    env = settings()
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="", help="the vision student (default: the one live.stream plays)")
    p.add_argument("--feel", help="ignored: vision only, there is no feel any more")
    p.add_argument("--account", default=env.get("OSU_RL_AI_USERNAME", ""), help="the AI's account (default OSU_RL_AI_USERNAME)")
    p.add_argument("--blocked", default=env.get("OSU_RL_BLOCKED_ACCOUNTS", ""), help="accounts never to play on, comma separated")
    p.add_argument("--server", default=env.get("OSU_RL_SERVER", ""), help="the private server (default OSU_RL_SERVER)")
    p.add_argument("--port", type=int, default=8767, help="the status API; 0: off")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--view", type=int, default=8770, metavar="PORT", help="the student's view for OBS; 0: off")
    p.add_argument("--no-auto-ready", action="store_true", help="in a room, leave readying up to whoever runs it")
    a = p.parse_args()
    say = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)
    if a.feel:
        say(f"--feel {a.feel} ignored: {NO_FEEL}")

    from live import tablet
    bridge = SeatBridge()
    first = bridge.status()
    tablet.FORK_PID = first.get("pid")  # the capture, the tablet and the playfield follow this copy's window
    setup_fork(bridge, say)
    player = student_player(a.ckpt, a.view, background=False)
    seat = Seat(bridge, player, a.account, [b for b in a.blocked.split(",") if b.strip()], a.server, say=say,
                auto_ready=not a.no_auto_ready)
    problem = seat.refresh()
    print(f"the fork: {{screen: {first.get('screen')}, server: {first.get('server')}, user: {first.get('user')}, "
          f"online: {first.get('online')}}}", flush=True)
    print("ready to play" if problem is None else f"WILL NOT PLAY until fixed: {problem}", flush=True)
    token = env.get("OSU_RL_CONTROL_TOKEN", "")
    if a.port and len(token) >= 16:
        serve(seat, a.port, token, a.host)
        print(f"status on http://{a.host}:{a.port}/status", flush=True)
    seat.run()


if __name__ == "__main__":
    main()
