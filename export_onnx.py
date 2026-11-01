import os
import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO

class DeterministicActor(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy_net = policy.mlp_extractor.policy_net
        self.action_net = policy.action_net

    def forward(self, obs):
        latent = self.policy_net(obs)
        action = self.action_net(latent)
        return torch.clamp(action, -1.0, 1.0)

def export_onnx(model_zip_path, out_onnx_path):
    os.makedirs(os.path.dirname(out_onnx_path), exist_ok=True)
    model = PPO.load(model_zip_path, device="cpu")
    actor = DeterministicActor(model.policy)
    actor.eval()

    obs_dim = model.observation_space.shape[0]
    dummy_input = torch.randn(1, obs_dim, dtype=torch.float32)

    with torch.no_grad():
        expected_action = actor(dummy_input).numpy()
        sb3_action = model.predict(dummy_input.numpy()[0], deterministic=True)[0]
        np.testing.assert_allclose(expected_action[0], sb3_action, atol=1e-5)

    torch.onnx.export(
        actor,
        dummy_input,
        out_onnx_path,
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        opset_version=14,
    )
    print(f"Exported ONNX model ({obs_dim}-dim) to {out_onnx_path}")
    print(f"File size: {os.path.getsize(out_onnx_path) / 1024:.2f} KB")

if __name__ == "__main__":
    export_onnx("results/20261101-0021-pool120/best_model_h6.zip", "merge_ppyosu/osu_rl.onnx")
