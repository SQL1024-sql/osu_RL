import os
import struct
import tempfile

import numpy as np
from gymnasium.utils.env_checker import check_env

from beatmap import SLIDER, SPINNER, parse_lines, parse_osu, random_osu
from env import OsuEnv
from play import oracle, run
from replay import write_osr

MAP = "maps/bakenohana.osu"

check_env(OsuEnv(), skip_render_check=True)

# random maps contain every mechanic
kinds = np.concatenate([parse_lines(random_osu(np.random.default_rng(s))).kinds for s in range(20)])
stacks = np.concatenate([parse_lines(random_osu(np.random.default_rng(s))).stack for s in range(20)])
assert (kinds == SLIDER).any() and (kinds == SPINNER).any() and (stacks != 0).any()

for hz in (60, 120):
    env = OsuEnv(hz=hz)
    for seed in range(20):
        env.reset(seed=seed)
        _, _, acc = run(env, lambda o: oracle(env))
        assert acc > 0.95, (hz, seed, acc)
        assert len(env.judgements) == len(env.objs) and not env.active and not env.spinning

env = OsuEnv()
_, counts, acc = run(env, lambda o: env.action_space.sample())
assert acc < 0.3, acc

bm = parse_osu(MAP)
assert len(bm.objects) == 569 and len(bm.sliders) == 351 and not bm.spinners  # same as lazer
nested = [k for s in bm.sliders.values() for _, k in s.nested]
assert (nested.count("tick"), nested.count("repeat"), nested.count("tail")) == (75, 8, 351)  # same as lazer
env = OsuEnv(bm, hz=120)
frames, counts, acc = run(env, lambda o: oracle(env))
assert acc > 0.9, acc

path = os.path.join(tempfile.mkdtemp(), "t.osr")
write_osr(path, MAP, frames, counts, env.max_combo)
raw = open(path, "rb").read()
mode, version = struct.unpack_from("<bi", raw)
assert (mode, version) == (0, 20240101) and raw[5] == 0x0B
print(f"ok: oracle acc on {bm.title} = {acc:.2%}, {dict(counts)}, max combo {env.max_combo}")
