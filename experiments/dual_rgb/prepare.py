from pathlib import Path
import hashlib
import json

import numpy as np
import zarr


root = Path(__file__).parent
paths = sorted((root / "data").glob("dual_gain20_shard*/episode_*.npz"))
assert len(paths) == 100, len(paths)
statuses = [json.loads((root / "data" / f"dual_gain20_shard{i}" / "status.json").read_text()) for i in range(4)]
assert sum(x["accepted"] for x in statuses) == 100
attempted = sum(x["attempted"] for x in statuses)
out = root / "data" / "metaworld_peg_insert_dual_rgb_gain20.zarr"
assert not out.exists(), out
zroot = zarr.open_group(str(out), mode="w")
data = zroot.create_group("data")
meta = zroot.create_group("meta")
ends, seeds = [], []
total = 0
for path in paths:
    episode = np.load(path)
    n = len(episode["action"])
    seeds.append(int(episode["seed"]))
    assert episode["success"].any()
    assert episode["image_corner2"].shape == episode["image_behindGripper"].shape == (n, 84, 84, 3)
    assert episode["state"].shape == (n, 9) and episode["action"].shape == (n, 4)
    for key in ("image_corner2", "image_behindGripper", "state", "full_state"):
        assert np.array_equal(episode["next_" + key][:-1], episode[key][1:]), (path, key)
    assert not episode["done"].any() and episode["timeout"][-1, 0] and not episode["timeout"][:-1].any()
    for key in episode.files:
        if key == "seed" or key.startswith("initial_"):
            continue
        value = episode[key]
        assert np.isfinite(value).all(), (path, key)
        if key not in data:
            data.create_dataset(key, shape=(0,) + value.shape[1:], chunks=(100,) + value.shape[1:], dtype=value.dtype)
        data[key].append(value)
    total += n
    ends.append(total)

assert len(set(seeds)) == 100
meta.create_dataset("episode_ends", data=np.asarray(ends, dtype=np.int64))
meta.create_dataset("episode_seeds", data=np.asarray(seeds, dtype=np.int64))
val = np.random.default_rng(42).choice(100, 10, replace=False)
train = np.setdiff1d(np.arange(100), val)
meta.create_dataset("train_episode_indices", data=train)
meta.create_dataset("val_episode_indices", data=val)
zroot.attrs.update({
    "gamma": 0.99,
    "reward_definition": "native MetaWorld v2 dense",
    "return_definition": "finite recorded episode discounted dense return, not bootstrapped at truncation",
    "rgb_orientation": "upright",
    "cameras": ["corner2", "behindGripper"],
    "expert_attempts": attempted,
    "expert_successes": 100,
    "source": "synchronized dual-view collection from one simulator state",
})
validation = {
    "episodes": 100,
    "transitions": total,
    "attempted": attempted,
    "expert_success_rate": 100 / attempted,
    "train": train.tolist(),
    "validation": val.tolist(),
    "temporal_pairing": True,
    "shapes": {k: list(data[k].shape) for k in data},
    "sha256_shared": {k: hashlib.sha256(data[k][:].tobytes()).hexdigest() for k in ("action", "state", "reward", "success", "done", "timeout", "return")},
}
(root / "dataset_validation.json").write_text(json.dumps(validation, indent=2))
print("DATASET_VALIDATED", total, attempted, flush=True)
