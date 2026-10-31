import sys

import matplotlib.pyplot as plt
import pandas as pd

run_dir = sys.argv[1]
df = pd.read_csv(f"{run_dir}/progress.csv")
ev = df.dropna(subset=["eval/acc"])

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(df["time/total_timesteps"], df["rollout/ep_rew_mean"])
ax[0].set(title="train episode reward", xlabel="steps")
ax[1].plot(ev["time/total_timesteps"], ev["eval/acc"] * 100)
ax[1].set(title="real map accuracy (%)", xlabel="steps")
for k in ("n300", "n100", "n50", "miss"):
    ax[2].plot(ev["time/total_timesteps"], ev[f"eval/{k}"], label=k)
ax[2].set(title="real map judgements", xlabel="steps")
ax[2].legend()
fig.tight_layout()
fig.savefig(f"{run_dir}/curves.png", dpi=120)
print(ev[["time/total_timesteps", "eval/acc", "eval/n300", "eval/n100", "eval/n50", "eval/miss", "eval/max_combo"]].to_string(index=False))
