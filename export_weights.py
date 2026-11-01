import os
import struct
import json
import torch
import numpy as np
from stable_baselines3 import PPO

def export_weights(model_zip_path, out_bin_path, out_json_path=None):
    os.makedirs(os.path.dirname(out_bin_path), exist_ok=True)
    model = PPO.load(model_zip_path, device="cpu")
    policy = model.policy

    # Extract weights
    # policy_net:
    # 0: Linear(38, 256)
    # 1: Tanh
    # 2: Linear(256, 256)
    # 3: Tanh
    # action_net: Linear(256, 4)
    w0 = policy.mlp_extractor.policy_net[0].weight.detach().numpy() # (256, 38)
    b0 = policy.mlp_extractor.policy_net[0].bias.detach().numpy()   # (256,)
    w1 = policy.mlp_extractor.policy_net[2].weight.detach().numpy() # (256, 256)
    b1 = policy.mlp_extractor.policy_net[2].bias.detach().numpy()   # (256,)
    w2 = policy.action_net.weight.detach().numpy()                   # (4, 256)
    b2 = policy.action_net.bias.detach().numpy()                     # (4,)

    in_dim = w0.shape[1]
    hidden_dim = w0.shape[0]
    out_dim = w2.shape[0]

    # Verify forward pass in pure numpy
    dummy = np.random.randn(in_dim).astype(np.float32)
    h0 = np.tanh(w0 @ dummy + b0)
    h1 = np.tanh(w1 @ h0 + b1)
    act = np.clip(w2 @ h1 + b2, -1.0, 1.0)

    sb3_act = model.predict(dummy, deterministic=True)[0]
    np.testing.assert_allclose(act, sb3_act, atol=1e-5)
    print("Verification passed! Pure numpy forward matches SB3 predict.")

    # Write compact binary format:
    # Header: magic 'OSURL', int32 in_dim, int32 hidden_dim, int32 out_dim
    # Data: float32 arrays: w0, b0, w1, b1, w2, b2 (row-major)
    with open(out_bin_path, "wb") as f:
        f.write(b"OSURL")
        f.write(struct.pack("<iii", in_dim, hidden_dim, out_dim))
        f.write(w0.astype(np.float32).tobytes())
        f.write(b0.astype(np.float32).tobytes())
        f.write(w1.astype(np.float32).tobytes())
        f.write(b1.astype(np.float32).tobytes())
        f.write(w2.astype(np.float32).tobytes())
        f.write(b2.astype(np.float32).tobytes())

    print(f"Exported model weights to {out_bin_path} ({os.path.getsize(out_bin_path) / 1024:.1f} KB)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export SB3 PPO weights to osu_rl binary format for osu!lazer C# mod.")
    parser.add_argument("model", nargs="?", default="results/20261101-0021-pool120/best_model_h6.zip", help="Path to SB3 model zip")
    parser.add_argument("--out", "-o", default="merge_ppyosu/model_weights.bin", help="Output binary file path")
    args = parser.parse_args()
    export_weights(args.model, args.out)
