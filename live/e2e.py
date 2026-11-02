"""End-to-end check against a server: a tester account opens a lazer room, the AI joins and plays a round while the
tester spectates, and the AI's live frames and room score are checked.

    python -m live.e2e <beatmap id> <beatmap md5> [level]
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    asyncio.run(main(int(sys.argv[1]), sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "expert"))
