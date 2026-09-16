from pathlib import Path
import argparse
import json
import time

import numpy as np
import torch
from metaworld.policies import SawyerPegInsertionSideV2Policy

from dual_env import DualViewPegEnv


parser = argparse.ArgumentParser()
parser.add_argument("--shard", type=int, required=True)
parser.add_argument("--count", type=int, default=25)
parser.add_argument("--gain", type=float, default=20)
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--dest", type=str, default=None)
args = parser.parse_args()

root = Path(__file__).parent
dest = Path(args.dest) if args.dest else root / "data" / f"dual_gain{int(args.gain)}_shard{args.shard}"
dest.mkdir(parents=True, exist_ok=True)
torch.manual_seed(7100 + args.shard)
np.random.seed(7100 + args.shard)
env = DualViewPegEnv(device="cuda:0", num_points=512, rgb_size=84, use_point_crop=True)
expert = SawyerPegInsertionSideV2Policy()
accepted = 0
attempts = []
started = time.time()

for attempt in range(args.count * 5):
    seed = 10000 + args.shard * 10000 + attempt
    env.seed(seed)
    obs = env.reset()
    rows = []
    success_trace = []
    initial = {
        "qpos": env.env.sim.data.qpos.copy(),
        "qvel": env.env.sim.data.qvel.copy(),
        "target": env.env._target_pos.copy(),
        "rand_vec": env.env._last_rand_vec.copy(),
    }
    for _ in range(200):
        action = expert.get_action(obs["full_state"])
        action[:3] *= args.gain / 25
        action = np.clip(action, -1, 1).astype(np.float32)
        nxt, reward, done, info = env.step(action)
        success = bool(info["success"])
        success_trace.append(success)
        rows.append({
            "image_corner2": obs["image_corner2"].transpose(1, 2, 0),
            "next_image_corner2": nxt["image_corner2"].transpose(1, 2, 0),
            "image_behindGripper": obs["image_behindGripper"].transpose(1, 2, 0),
            "next_image_behindGripper": nxt["image_behindGripper"].transpose(1, 2, 0),
            "state": obs["agent_pos"],
            "next_state": nxt["agent_pos"],
            "full_state": obs["full_state"],
            "next_full_state": nxt["full_state"],
            "action": action,
            "reward": np.array([reward], np.float32),
            "done": np.array([done and not info["TimeLimit.truncated"]], bool),
            "timeout": np.array([info["TimeLimit.truncated"]], bool),
            "success": np.array([success], bool),
        })
        obs = nxt
        if done:
            break

    passed = any(success_trace)
    attempts.append({"seed": seed, "success": passed, "success_steps": sum(success_trace), "steps": len(rows)})
    if passed:
        data = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
        last_action = expert.get_action(obs["full_state"])
        last_action[:3] *= args.gain / 25
        last_action = np.clip(last_action, -1, 1).astype(np.float32)
        data["next_action"] = np.concatenate([data["action"][1:], last_action[None]])
        returns = np.zeros_like(data["reward"])
        value = 0.0
        for i in reversed(range(len(rows))):
            value = float(data["reward"][i, 0]) + 0.99 * value
            returns[i, 0] = value
        data["return"] = returns
        out = dest / f"episode_{accepted:03d}.npz"
        assert not out.exists(), out
        np.savez_compressed(out, **data, seed=seed, **{"initial_" + k: v for k, v in initial.items()})
        accepted += 1

    status = {"accepted": accepted, "attempted": len(attempts), "attempts": attempts, "elapsed": time.time() - started}
    (dest / "status.json").write_text(json.dumps(status, indent=2))
    print(json.dumps({"shard": args.shard, "accepted": accepted, "attempted": len(attempts), "seed": seed, "success": passed, "elapsed": status["elapsed"]}), flush=True)
    if args.smoke or accepted >= args.count:
        break

assert accepted > 0, "expert produced no successful trajectories"
env.close()
