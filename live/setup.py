"""One-time setup of the AI player, then a check of everything it needs. Safe to run again.

    python -m live.setup

1. asks for the osu! server and the AI's account (once), makes the control token, writes live/.env
2. gets lazer (ppy/osu at the tested commit) into OSU_REPO if it is not there, adds the scoring harness, builds it
3. logs in with the AI's account
4. prints the settings for Miyoria's osu extension
"""
import asyncio
import getpass
import os
import secrets
import shutil
import subprocess
import sys

from live.hub import Api
from verify import DOTNET, DOTNET_ROOT, OSU_REPO

ENV = os.path.join("live", ".env")
LAZER_COMMIT = "845336e3c1235f16b24bc4380fa7a6dc0cf7cd91"  # the ppy/osu commit the harness is tested on
HARNESS = ["TestSceneOsuRLLive.cs", "TestSceneOsuRLReplay.cs"]
TESTS = "osu.Game.Rulesets.Osu.Tests"


def fail(msg):
    sys.exit(f"\n[x] {msg}")


def read_env():
    if not os.path.exists(ENV):
        return {}
    with open(ENV, encoding="utf-8") as f:
        return dict(line.strip().split("=", 1) for line in f if "=" in line)


def ask(env, key, prompt, default=None, secret=False):
    if env.get(key):
        return
    hint = f" [{default}]" if default else ""
    value = (getpass.getpass if secret else input)(f"{prompt}{hint}: ").strip() or default
    if not value:
        fail(f"{prompt} is needed")
    env[key] = value


def settings():
    env = read_env()
    ask(env, "OSU_RL_SERVER", "osu! server URL", "https://shdcloud.top")
    ask(env, "OSU_RL_AI_USERNAME", "AI account username (register it on the server's website first)")
    ask(env, "OSU_RL_AI_PASSWORD", "AI account password", secret=True)
    env.setdefault("OSU_RL_CONTROL_TOKEN", secrets.token_urlsafe(24))
    env["OSU_RL_SERVER"] = env["OSU_RL_SERVER"].rstrip("/")
    with open(ENV, "w", encoding="utf-8") as f:
        f.writelines(f"{k}={v}\n" for k, v in env.items())
    print(f"[ok] settings in {ENV}")
    return env


def lazer():
    if not os.path.exists(DOTNET):
        fail(".NET 10 SDK not found. Install it, then run this again:\n"
             "    Windows: winget install Microsoft.DotNet.SDK.10\n"
             "    Linux:   https://learn.microsoft.com/dotnet/core/install/linux")
    if not os.path.isdir(OSU_REPO):
        print(f"getting lazer (ppy/osu {LAZER_COMMIT[:7]}) into {OSU_REPO} ...")
        os.makedirs(OSU_REPO)
        for cmd in (["init", "-q"], ["remote", "add", "origin", "https://github.com/ppy/osu.git"],
                    ["fetch", "-q", "--depth", "1", "origin", LAZER_COMMIT], ["checkout", "-q", "FETCH_HEAD"]):
            subprocess.run(["git", *cmd], cwd=OSU_REPO, check=True)
    for name in HARNESS:
        shutil.copy(os.path.join("lazer", name), os.path.join(OSU_REPO, TESTS, name))
    print("building the lazer scoring harness (the first build takes a few minutes) ...")
    r = subprocess.run([DOTNET, "build", TESTS], cwd=OSU_REPO, env=dict(os.environ, DOTNET_ROOT=DOTNET_ROOT), capture_output=True, text=True)
    if r.returncode:
        fail("lazer build failed:\n" + r.stdout[-3000:])
    print("[ok] lazer harness built")


async def login(env):
    api = Api(env["OSU_RL_SERVER"])
    try:
        me = await api.login(env["OSU_RL_AI_USERNAME"], env["OSU_RL_AI_PASSWORD"])
    except Exception as e:
        fail(f"cannot log in to {env['OSU_RL_SERVER']} as {env['OSU_RL_AI_USERNAME']}: {e}")
    finally:
        await api.close()
    print(f"[ok] logged in as {me['username']} (id {me['id']})")


if __name__ == "__main__":
    env = settings()
    lazer()
    asyncio.run(login(env))
    print(f"""
All set. Start the AI player with:
    python -m live.service

Miyoria (miyoria.toml), when it runs on this machine:
    [extensions."miyoria.osu"]
    account = "{env['OSU_RL_AI_USERNAME']}"
    blocked_accounts = []   # your own osu! accounts, which the AI must never use
    env = {{ MIYORIA_OSU_TOKEN = "{env['OSU_RL_CONTROL_TOKEN']}" }}
(on another machine also set url = "http://<this machine>:8765" and start the service with --host 0.0.0.0)
""")
