"""Sets OBS up for the stream over obs-websocket (built into OBS 28+, Tools -> WebSocket Server Settings): a scene
"osu_rl" with the fork's window, the student's view (live.vision_view) in a corner, and the improve loop's overlay text.

    python -m live.obs_setup              # create what is missing (existing sources are left as they are)
    python -m live.obs_setup --switch     # also make it the program scene (never while streaming or recording)
    python -m live.obs_setup --password X # if the WebSocket server asks for one

Only adds to OBS: no existing scene or source is changed or removed.
"""
import argparse
import asyncio
import base64
import hashlib
import itertools
import os

import aiohttp

SCENE = "osu_rl"
GAME, VIEW, TEXT = "osu_rl 遊戲畫面", "osu_rl 學生的視角", "osu_rl 狀態"
OVERLAY = os.path.abspath("runs/improve/overlay.txt")


class OBS:
    def __init__(self, ws):
        self.ws, self.ids = ws, itertools.count(1)

    @classmethod
    async def connect(cls, session, url, password):
        ws = await session.ws_connect(url, protocols=("obswebsocket.json",))
        hello = (await ws.receive_json())["d"]
        ident = {"rpcVersion": 1}
        if "authentication" in hello:
            if not password:
                raise SystemExit("OBS's WebSocket server wants a password: --password")
            a = hello["authentication"]
            secret = base64.b64encode(hashlib.sha256((password + a["salt"]).encode()).digest()).decode()
            ident["authentication"] = base64.b64encode(hashlib.sha256((secret + a["challenge"]).encode()).digest()).decode()
        await ws.send_json({"op": 1, "d": ident})
        while (await ws.receive_json())["op"] != 2:
            pass
        return cls(ws)

    async def call(self, kind, **data):
        rid = str(next(self.ids))
        await self.ws.send_json({"op": 6, "d": {"requestType": kind, "requestId": rid, "requestData": data}})
        while True:
            m = await self.ws.receive_json()
            if m["op"] == 7 and m["d"]["requestId"] == rid:
                st = m["d"]["requestStatus"]
                if not st["result"]:
                    raise RuntimeError(f"{kind}: {st.get('code')} {st.get('comment')}")
                return m["d"].get("responseData", {})


async def main(a):
    global OVERLAY
    if a.stream:  # live.stream (the stream machine only plays) writes its own overlay
        OVERLAY = os.path.abspath("runs/stream/overlay.txt")
    if not os.path.exists(OVERLAY):  # the text source reads it; the loop rewrites it as it plays
        os.makedirs(os.path.dirname(OVERLAY), exist_ok=True)
        open(OVERLAY, "w", encoding="utf-8").write("osu_rl 準備中\n")
    async with aiohttp.ClientSession() as s:
        obs = await OBS.connect(s, f"ws://127.0.0.1:{a.port}", a.password)
        v = await obs.call("GetVersion")
        video = await obs.call("GetVideoSettings")
        W, H = video["baseWidth"], video["baseHeight"]
        print(f"OBS {v['obsVersion']}, canvas {W}x{H}")

        scenes = [x["sceneName"] for x in (await obs.call("GetSceneList"))["scenes"]]
        if SCENE not in scenes:
            await obs.call("CreateScene", sceneName=SCENE)
            print(f"scene {SCENE} created")
        inputs = {x["inputName"] for x in (await obs.call("GetInputList"))["inputs"]}
        kinds = (await obs.call("GetInputKindList"))["inputKinds"]
        text_kind = next((k for k in sorted(kinds, reverse=True) if k.startswith("text_gdiplus")), None)

        async def add(name, kind, settings, transform):
            if name in inputs:
                print(f"{name}: already there, left as it is")
                return
            r = await obs.call("CreateInput", sceneName=SCENE, inputName=name, inputKind=kind, inputSettings=settings)
            await obs.call("SetSceneItemTransform", sceneName=SCENE, sceneItemId=r["sceneItemId"], sceneItemTransform=transform)
            print(f"{name}: added")

        # the fork's window: Windows Graphics Capture works while other windows cover it (not while it is minimized)
        await add(GAME, "window_capture",
                  dict(window="osu!:SDL_app:osu!.exe", method=2, priority=2, cursor=False, client_area=True),
                  dict(positionX=0, positionY=0, boundsType="OBS_BOUNDS_SCALE_INNER", boundsWidth=W, boundsHeight=H,
                       boundsAlignment=0))
        # what the student sees (4:3), bottom right, a quarter of the width
        vw = W // 4
        vh = vw * 3 // 4
        await add(VIEW, "browser_source", dict(url="http://127.0.0.1:8770/", width=640, height=480, shutdown=False),
                  dict(positionX=W - vw - W // 64, positionY=H - vh - W // 64, scaleX=vw / 640, scaleY=vh / 480))
        if text_kind:
            await add(TEXT, text_kind,
                      dict(read_from_file=True, file=OVERLAY, font=dict(face="Microsoft JhengHei", size=H // 24, style="Bold", flags=1),
                           outline=True, outline_size=4, outline_color=0xFF000000, color=0xFFFFFFFF),
                      dict(positionX=W // 64, positionY=W // 64))
        else:
            print("no GDI+ text source in this OBS: the overlay text was not added")

        if a.switch:
            live = (await obs.call("GetStreamStatus"))["outputActive"] or (await obs.call("GetRecordStatus"))["outputActive"]
            if live:
                print("streaming or recording: the program scene was not switched")
            else:
                await obs.call("SetCurrentProgramScene", sceneName=SCENE)
                print(f"program scene is now {SCENE}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=4455)
    p.add_argument("--password")
    p.add_argument("--switch", action="store_true")
    p.add_argument("--stream", action="store_true", help="the status text of live.stream instead of live.improve")
    asyncio.run(main(p.parse_args()))
