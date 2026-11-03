"""End-to-end check against a server: a tester account opens a lazer room, the AI joins and plays a round while the
tester spectates, and the AI's live frames and room score are checked. With --solo the AI plays the beatmap on its
own while the tester watches it (as a stream would), and its events and solo score are checked.

    python -m live.e2e <beatmap id> <beatmap md5> [level]
    python -m live.e2e <beatmap id> <beatmap md5> [level] --solo
Needs OSU_RL_TESTER_USERNAME / OSU_RL_TESTER_PASSWORD besides the AI's settings (see live/README.md).
"""
import asyncio
import json
import logging
import sys

from live.hub import Api
from live.player import AiPlayer
from live.service import settings


async def main(beatmap, md5, level):
    env = settings()
    tester = Api(env["OSU_RL_SERVER"])
    await tester.login(env["OSU_RL_TESTER_USERNAME"], env["OSU_RL_TESTER_PASSWORD"])
    mp, spec = await tester.hub("multiplayer"), await tester.hub("spectator")
    results, frames = asyncio.Event(), []
    mp.on("ResultsReady", results.set)
    spec.on("UserSentFrames", lambda user, bundle: frames.append(bundle["header"]))
    try:
        await mp.invoke("LeaveRoom")  # left over from an interrupted run
    except Exception:
        pass

    room = await mp.invoke("CreateRoom", {"settings": {"name": "osu_rl e2e", "password": "", "matchType": 1, "queueMode": 0, "autoStartDuration": 0},
                                          "playlist": [{"beatmapID": beatmap, "beatmapChecksum": md5, "rulesetID": 0, "requiredMods": [], "allowedMods": []}]})
    ai = AiPlayer(env["OSU_RL_SERVER"], env["OSU_RL_AI_USERNAME"], env["OSU_RL_AI_PASSWORD"], level)
    try:
        await ai.start()
        await spec.invoke("StartWatchingUser", ai.me["id"])
        await ai.join(room["roomID"])
        while not ai.play:
            await asyncio.sleep(1)
        await asyncio.sleep(2)  # let the ready state land
        await mp.invoke("ChangeState", 8)  # the tester only spectates
        await mp.invoke("StartMatch")
        await asyncio.wait_for(results.wait(), 900)
        assert frames, "no live frames reached the spectator"
        assert ai.last_result and ai.last_result["score_id"], "the AI's score was not submitted"
        print(f"ok: {len(frames)} frame bundles, final live header {json.dumps(frames[-1]['totalScore'])}, result {ai.last_result}")
    finally:
        await ai.close()
        await mp.invoke("LeaveRoom")
        for h in (mp, spec):
            await h.close()
        await tester.close()


async def main_solo(beatmap, level):
    env = settings()
    tester = Api(env["OSU_RL_SERVER"])
    await tester.login(env["OSU_RL_TESTER_USERNAME"], env["OSU_RL_TESTER_PASSWORD"])
    spec = await tester.hub("spectator")
    frames = []
    spec.on("UserSentFrames", lambda user, bundle: frames.append(bundle["header"]))
    ai = AiPlayer(env["OSU_RL_SERVER"], env["OSU_RL_AI_USERNAME"], env["OSU_RL_AI_PASSWORD"], level)
    try:
        await ai.start()
        await spec.invoke("StartWatchingUser", ai.me["id"])
        await ai.solo(beatmap)
        seen = []
        while not any(e["type"] in ("finished", "error") for e in seen):
            seen += await ai.events_after(seen[-1]["id"] if seen else 0, 60)
            if ai.live:
                print(f"live: {ai.live['progress']:.0%} combo {ai.live['combo']} acc {ai.live['accuracy']:.2%}", flush=True)
        types = [e["type"] for e in seen]
        assert types[0] == "preparing" and "started" in types and types[-1] == "finished", types
        assert frames, "no live frames reached the spectator"
        assert ai.last_result and ai.last_result["score_id"], "the AI's solo score was not submitted"
        print(f"ok: {len(frames)} frame bundles to the watcher, events {types.count('combo_break')} combo breaks, result {ai.last_result}")
    finally:
        await ai.close()
        await spec.close()
        await tester.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = [a for a in sys.argv[1:] if a != "--solo"]
    level = args[2] if len(args) > 2 else "expert"
    asyncio.run(main_solo(int(args[0]), level) if "--solo" in sys.argv else main(int(args[0]), args[1], level))
