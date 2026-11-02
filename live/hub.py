"""Minimal osu! server client for the AI player: OAuth password login, REST calls and SignalR hubs (JSON protocol)."""
import asyncio
import itertools
import json
import logging

import aiohttp

RS = "\x1e"  # SignalR record separator
LAZER_CLIENT = dict(client_id="5", client_secret="FGc9GAtyHzeQDshWP5Ah7dega8hJACAJpQtw6OXk")  # stock lazer's public client
log = logging.getLogger("live")


class HubError(Exception):
    pass


class Api:
    def __init__(self, server):
        self.server = server.rstrip("/")
        self.session = None  # created inside the running event loop
        self.token = None

    async def login(self, username, password):
        self.session = self.session or aiohttp.ClientSession()
        async with self.session.post(f"{self.server}/oauth/token", data=dict(LAZER_CLIENT, grant_type="password", username=username, password=password, scope="*")) as r:
            r.raise_for_status()
            self.token = (await r.json())["access_token"]
        return await self.request("GET", "/api/v2/me")

    @property
    def auth(self):
        return {"Authorization": f"Bearer {self.token}"}

    async def request(self, method, path, raw=False, **kw):
        async with self.session.request(method, self.server + path, headers=self.auth, **kw) as r:
            body = await r.read()
            if r.status >= 400:
                raise HubError(f"{method} {path}: {r.status} {body[:200]!r}")
            return body if raw else json.loads(body)

    async def hub(self, name):
        h = Hub(self, name)
        await h.connect()
        return h

    async def close(self):
        if self.session:
            await self.session.close()


class Hub:
    def __init__(self, api, name):
        self.api, self.name = api, name
        self.handlers = {}
        self.pending = {}
        self.ids = itertools.count(1)
        self.ws = None
        self.closed = asyncio.Event()

    def on(self, target, handler):
        self.handlers[target] = handler

    async def connect(self):
        base = f"/signalr/{self.name}"
        neg = await self.api.request("POST", f"{base}/negotiate?negotiateVersion=1")
        url = self.api.server.replace("http", "ws", 1) + f"{base}?id={neg['connectionToken']}"
        self.ws = await self.api.session.ws_connect(url, headers=self.api.auth, heartbeat=15)
        await self.ws.send_str(json.dumps({"protocol": "json", "version": 1}) + RS)
        reply = (await self.ws.receive_str()).split(RS)[0]
        if json.loads(reply).get("error"):
            raise HubError(reply)
        asyncio.create_task(self._read())

    async def _read(self):
        try:
            async for msg in self.ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    break
                for record in filter(None, msg.data.split(RS)):
                    self._dispatch(json.loads(record))
        finally:
            self.closed.set()
            for f in self.pending.values():
                if not f.done():
                    f.set_exception(HubError(f"{self.name} hub closed"))

    def _dispatch(self, m):
        if m["type"] == 1:  # invocation from the server
            handler = self.handlers.get(m["target"])
            if handler:
                asyncio.create_task(_guard(handler, m.get("arguments", []), m["target"]))
        elif m["type"] == 3:  # completion
            f = self.pending.pop(m["invocationId"], None)
            if f and not f.done():
                f.set_exception(HubError(m["error"])) if m.get("error") else f.set_result(m.get("result"))
        elif m["type"] == 6:  # ping
            asyncio.create_task(self.ws.send_str(json.dumps({"type": 6}) + RS))
        elif m["type"] == 7:
            log.warning("%s hub closed by server: %s", self.name, m.get("error"))

    async def invoke(self, target, *args):
        i = str(next(self.ids))
        self.pending[i] = asyncio.get_running_loop().create_future()
        await self.ws.send_str(json.dumps({"type": 1, "invocationId": i, "target": target, "arguments": list(args)}) + RS)
        return await asyncio.wait_for(self.pending[i], 30)

    async def send(self, target, *args):
        await self.ws.send_str(json.dumps({"type": 1, "target": target, "arguments": list(args)}) + RS)

    async def close(self):
        if self.ws:
            await self.ws.close()


async def _guard(handler, args, target):
    try:
        r = handler(*args)
        if asyncio.iscoroutine(r):
            await r
    except Exception:
        log.exception("handler for %s failed", target)
