"""Turns a beatmap into everything the AI needs to play it in an online room: the simulated play (replay frames)
and lazer's own scoring of it (a spectator header after every judgement, the maximum statistics and the score
submission body), from the headless lazer harness lazer/TestSceneOsuRLLive.cs.

    python -m live.prepare map.osu --level expert
"""
import argparse
import hashlib
import json
import os
import subprocess
import tempfile

from stable_baselines3 import PPO

from beatmap import parse_osu
from state.env import OsuEnv, k_from_obs
from state.play import run
from replay import KEY_BITS, write_osr
from state.verify import DOTNET, DOTNET_ROOT, OSU_REPO

# checkpoints of the Exp 012 run as difficulty levels (H6 validation acc in brackets)
LEVELS = {
    "easy": "results/levels/easy.zip",  # 1M steps (72%)
    "normal": "results/levels/normal.zip",  # 2M steps (97.5%)
    "hard": "results/levels/hard.zip",  # 5M steps (98.2%)
    "expert": "results/20261101-0021-pool120/best_model_h6.zip",  # 12M steps (99.3%)
}
HZ = 120
CACHE = os.path.join("runs", "live_cache")
_models = {}


def model(level):
    if level not in _models:
        _models[level] = PPO.load(LEVELS[level], device="cpu")
    return _models[level]


def lazer_score(osu_path, osr_path, mods, out_path):
    """Plays the replay in headless lazer with the room's mods; returns its scoring timeline."""
    job = out_path + ".job"
    with open(job, "w", encoding="utf-8") as f:
        f.write("|".join([os.path.abspath(osu_path), os.path.abspath(osr_path), os.path.abspath(out_path), json.dumps(mods)]))
    subprocess.run([DOTNET, "test", "osu.Game.Rulesets.Osu.Tests", "--no-build",
                    "--filter", "FullyQualifiedName~TestSceneOsuRLLive"],
                   cwd=OSU_REPO, env=dict(os.environ, DOTNET_ROOT=DOTNET_ROOT, OSU_RL_LIVE_JOB=os.path.abspath(job)), check=True, capture_output=True)
    with open(out_path, encoding="utf-8") as f:
        return json.load(f)


def prepare(osu_bytes, mods=(), level="expert"):
    """-> dict(frames=[(time, x, y, buttons)], headers=[...], maximumStatistics, submission), cached on disk."""
    mods = list(mods)
    key = hashlib.md5(osu_bytes + json.dumps([mods, level, LEVELS[level]], sort_keys=True).encode()).hexdigest()
    path = os.path.join(CACHE, key + ".json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    os.makedirs(CACHE, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        osu_path, osr_path = os.path.join(tmp, "map.osu"), os.path.join(tmp, "play.osr")
        with open(osu_path, "wb") as f:
            f.write(osu_bytes)
        m = model(level)
        env = OsuEnv(parse_osu(osu_path), hz=HZ, k=k_from_obs(m.observation_space.shape[0]))
        frames, counts, _ = run(env, lambda o: m.predict(o, deterministic=True)[0])
        write_osr(osr_path, osu_path, frames, counts, env.max_combo)
        play = lazer_score(osu_path, osr_path, mods, os.path.join(tmp, "score.json"))
        os.replace(osr_path, os.path.join(CACHE, key + ".osr"))
    play["frames"] = [(round(t), round(float(x), 2), round(float(y), 2), sum(b for b, k in zip(KEY_BITS, keys) if k)) for t, x, y, keys in frames]
    play["level"] = level
    play["replay"] = key + ".osr"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=CACHE, suffix=".tmp", delete=False) as f:
        json.dump(play, f)
    os.replace(f.name, path)  # a second request for the same map may be preparing it at the same time
    return play


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("map")
    p.add_argument("--level", default="expert", choices=LEVELS)
    p.add_argument("--mods", default="[]", help='room mods as APIMod JSON, e.g. [{"acronym":"HD"}]')
    args = p.parse_args()
    play = prepare(open(args.map, "rb").read(), json.loads(args.mods), args.level)
    s = play["submission"]
    print(f"{args.level}: {s['accuracy']:.2%} rank {s['rank']} score {s['total_score']} combo {s['max_combo']}, "
          f"{len(play['frames'])} frames, {len(play['headers'])} judgements")
