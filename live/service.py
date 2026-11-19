"""HTTP control API for the AI player, for the Discord bot's tool calls (live/tools.json describes each one).

    python -m live.service --port 8765
    python -m live.service --feel good --view 8771     # plays like a person; the student's view for OBS on 8771
Settings come from the environment or live/.env: OSU_RL_SERVER, OSU_RL_AI_USERNAME, OSU_RL_AI_PASSWORD and
OSU_RL_CONTROL_TOKEN (callers send it as "Authorization: Bearer <token>").

Retired 2026-11-17: the AI plays with the vision student only, from the game's picture (live.room_play in rooms,
live.requests / live.stream for maps of its own). This plays the state model's moves (it reads the .osu) over the
lazer protocol with no picture at all, so it refuses to start.
"""
import argparse
import asyncio
import hashlib
import logging
import os

from aiohttp import web

from live.humanize import FEELS
from live.player import AiPlayer
from live.prepare import CACHE, LEVELS, prepare


def settings():
    env = {}
    if os.path.exists("live/.env"):
        env = dict(line.strip().split("=", 1) for line in open("live/.env", encoding="utf-8") if "=" in line)
    env.update({k: v for k, v in os.environ.items() if k.startswith("OSU_RL_")})
    return env


def summary(play):
    s = play["submission"]
    return dict(level=play["level"], feel=play.get("feel", "full"), accuracy=s["accuracy"], rank=s["rank"], total_score=s["total_score"], max_combo=s["max_combo"],
                statistics=s["statistics"], replay_url=f"/replays/{play['replay']}")


def app(player, token):
    routes = web.RouteTableDef()

    @web.middleware
    async def auth(request, handler):
        if request.path.startswith("/replays/") or request.headers.get("Authorization") == f"Bearer {token}":
            try:
                return await handler(request)
            except (ValueError, KeyError) as e:
                return web.json_response({"error": str(e)}, status=400)
            except Exception as e:
                logging.exception("request failed")
                return web.json_response({"error": str(e)}, status=502)
        return web.json_response({"error": "unauthorized"}, status=401)

    @routes.get("/status")
    async def status(_):
        return web.json_response(player.status())

    @routes.get("/levels")
    async def levels(_):
        return web.json_response(list(LEVELS))

    @routes.get("/feels")
    async def feels(_):
        return web.json_response(list(FEELS))

    @routes.get("/rooms")
    async def rooms(_):
        rooms = await player.api.request("GET", "/api/v2/rooms")
        return web.json_response([dict(id=r.get("id"), name=r.get("name"), host=(r.get("host") or {}).get("username"), has_password=r.get("has_password"),
                                       participants=r.get("participant_count")) for r in rooms])

    @routes.post("/join")
    async def join(request):
        body = await request.json()
        return web.json_response(await player.join(body["room_id"], body.get("password", ""), body.get("level"), body.get("feel")))

    @routes.post("/leave")
    async def leave(_):
        return web.json_response(await player.leave())

    @routes.post("/level")
    async def level(request):
        body = await request.json()
        player.set_level(body.get("level"), body.get("feel"))
        return web.json_response(player.status())

    @routes.post("/play")
    async def play(request):
        body = await request.json()
        osu = await player.api.request("GET", f"/osu/{int(body['beatmap_id'])}", raw=True)
        result = await asyncio.to_thread(prepare, osu, body.get("mods", []), body.get("level", player.level), body.get("feel", player.feel))
        return web.json_response(dict(summary(result), beatmap_id=body["beatmap_id"], beatmap_md5=hashlib.md5(osu).hexdigest()))

    @routes.post("/solo")
    async def solo(request):
        body = await request.json()
        return web.json_response(await player.solo(body["beatmap_id"], body.get("level"), body.get("mods", []), body.get("feel")))

    @routes.post("/stop")
    async def stop(_):
        return web.json_response(player.stop())

    @routes.get("/events")
    async def events(request):
        after, wait = int(request.query.get("after", 0)), min(float(request.query.get("wait", 25)), 60)
        return web.json_response(dict(events=await player.events_after(after, wait), last=player.event_id))

    @routes.get("/replays/{name}")
    async def replay(request):
        name = request.match_info["name"]
        path = os.path.join(CACHE, name)
        if not name.endswith(".osr") or os.path.basename(name) != name or not os.path.exists(path):
            raise web.HTTPNotFound()
        return web.FileResponse(path, headers={"Content-Disposition": f'attachment; filename="osu_rl-{name}"'})

    a = web.Application(middlewares=[auth])
    a.add_routes(routes)

    async def lifecycle(_):
        await player.start()
        yield
        await player.close()

    a.cleanup_ctx.append(lifecycle)
    return a


if __name__ == "__main__":
    raise SystemExit("live.service is retired: the AI plays with the vision student only (live.room_play, "
                     "live.requests, live.stream). This one plays the state model's moves without a picture.")
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--feel", default="full", choices=list(FEELS), help="how human it plays until told otherwise (live.humanize)")
    p.add_argument("--view", type=int, default=0, metavar="PORT",
                   help="serve the student's view of its plays for OBS (live.vision_view); not the port live.stream uses")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    env = settings()
    if len(env.get("OSU_RL_CONTROL_TOKEN", "")) < 16:
        raise SystemExit("set OSU_RL_CONTROL_TOKEN (16+ characters) in the environment or live/.env")
    view = None
    if args.view:
        from live.vision_view import View
        view = View(args.view, style="neuro")
        logging.info("student's view for OBS: %s", view.url)
    player = AiPlayer(env["OSU_RL_SERVER"], env["OSU_RL_AI_USERNAME"], env["OSU_RL_AI_PASSWORD"], feel=args.feel, view=view)
    web.run_app(app(player, env["OSU_RL_CONTROL_TOKEN"]), host=args.host, port=args.port)
