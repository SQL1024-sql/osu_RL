"""HTTP control API for the AI player, for the Discord bot's tool calls (live/tools.json describes each one).

    python -m live.service --port 8765
Settings come from the environment or live/.env: OSU_RL_SERVER, OSU_RL_AI_USERNAME, OSU_RL_AI_PASSWORD and
OSU_RL_CONTROL_TOKEN (callers send it as "Authorization: Bearer <token>").
"""
import argparse
import asyncio
import hashlib
import logging
import os

from aiohttp import web

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
    return dict(level=play["level"], accuracy=s["accuracy"], rank=s["rank"], total_score=s["total_score"], max_combo=s["max_combo"],
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

    @routes.get("/rooms")
    async def rooms(_):
        rooms = await player.api.request("GET", "/api/v2/rooms")
        return web.json_response([dict(id=r.get("id"), name=r.get("name"), host=(r.get("host") or {}).get("username"), has_password=r.get("has_password"),
                                       participants=r.get("participant_count")) for r in rooms])

    @routes.post("/join")
    async def join(request):
        body = await request.json()
        return web.json_response(await player.join(body["room_id"], body.get("password", ""), body.get("level")))

    @routes.post("/leave")
    async def leave(_):
        return web.json_response(await player.leave())

    @routes.post("/level")
    async def level(request):
        player.set_level((await request.json())["level"])
        return web.json_response(player.status())

    @routes.post("/play")
    async def play(request):
        body = await request.json()
        osu = await player.api.request("GET", f"/osu/{int(body['beatmap_id'])}", raw=True)
        result = await asyncio.to_thread(prepare, osu, body.get("mods", []), body.get("level", player.level))
        return web.json_response(dict(summary(result), beatmap_id=body["beatmap_id"], beatmap_md5=hashlib.md5(osu).hexdigest()))

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
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    env = settings()
    if len(env.get("OSU_RL_CONTROL_TOKEN", "")) < 16:
        raise SystemExit("set OSU_RL_CONTROL_TOKEN (16+ characters) in the environment or live/.env")
    player = AiPlayer(env["OSU_RL_SERVER"], env["OSU_RL_AI_USERNAME"], env["OSU_RL_AI_PASSWORD"])
    web.run_app(app(player, env["OSU_RL_CONTROL_TOKEN"]), host=args.host, port=args.port)
