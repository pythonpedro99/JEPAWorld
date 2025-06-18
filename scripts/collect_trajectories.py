# ────────────────────────────────────────────────────────────────────────
# Imports
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
import miniworld.envs                               # noqa: F401  (needed for Gym registration)
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.envs.registration import register
from shapely.geometry import Polygon, Point, LineString, box
from shapely import affinity
from policies.rearrange import HumanLikeRearrangePolicy
#from JEPAWORLD.policies.rearrange import HumanLikeRearrangePolicy

import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import sys, os

# Calculate project root (two levels up from scripts/)
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
# Ensure it's searched *first*
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class CollectTrajectories:
    def __init__(
        self,
        env_id: str = "JEPAENV-v0",
        n_samples: int = 1000,
        save_images: bool = False,
        output_dir: str = "./saved_data"
    ) -> None:
        # parameters
        self.env_id = env_id
        self.n_samples = n_samples
        self.save_images = save_images
        self.output_dir = output_dir

        # master seed and RNG
        self.master_seed = random.randint(0, 2**31 - 1)
        self.rng = np.random.default_rng(self.master_seed)

        # create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        if self.save_images:
            images_dir = os.path.join(self.output_dir, "images")
            os.makedirs(images_dir, exist_ok=True)

        # storage
        self.observations = []  # list of np.ndarray
        self.actions = []       # list of ints
        self.metadata = {
            "env_id": self.env_id,
            "master_seed": self.master_seed,
            "n_samples": self.n_samples,
            "episodes": []
        }

        # begin collection loop
        self._collect_loop()

        # save non-image data and metadata
        if not self.save_images:
            np.save(os.path.join(self.output_dir, "obs.npy"), np.array(self.observations, dtype=object))
            np.save(os.path.join(self.output_dir, "actions.npy"), np.array(self.actions, dtype=np.int32))
        with open(os.path.join(self.output_dir, "metadata.json"), "w") as f:
            json.dump(self.metadata, f, indent=2)

    def _collect_loop(self):
        total = 0
        episode_idx = 0
        while total < self.n_samples:
            ep_seed = episode_idx #int(self.rng.integers(0, 2**31 - 1))
            episode_idx += 1
            # per-episode seed for fresh placement, colors, and agent start
            

            # recreate environment each episode with domain randomization
            env = gym.make(
                self.env_id,
                seed=ep_seed,
                obs_width=224,
                obs_height=224,
                domain_rand=True
            )

            # run policy on this fresh env
            policy = HumanLikeRearrangePolicy(env=env, seed=ep_seed)
            # attempt rearrangement; skip episode if it fails
            success = policy.rearrange()
            if not success:
                print(f"Episode {episode_idx} failed rearrangement; skipping.")
                continue
            # retrieve actions and observations from policy
            actions = policy.actions
            observations = policy.observations

            # record episode metadata
            ep_meta = {
                "episode": episode_idx,
                "seed": ep_seed,
                "n_actions": len(actions),
                "n_observations": len(observations)
            }
            self.metadata["episodes"].append(ep_meta)

            # collect up to n_samples
            for cmd, obs in zip(actions, observations):
                if total >= self.n_samples:
                    break
                self.actions.append(cmd)
                self.observations.append(obs)
                # optionally save each image
                if self.save_images:
                    img_path = os.path.join(
                        self.output_dir,
                        "images",
                        f"obs_{total+1:04d}.png"
                    )
                    plt.imsave(img_path, obs)
                total += 1
                # simple percentage display
                percent = total / self.n_samples * 100
                print(f"Progress: {percent:.1f}% ({total}/{self.n_samples})", end='\r')

        print(f"\nCollected {total} samples over {episode_idx} episodes.")


if __name__ == "__main__":


    register(
    id="JEPAENV-v0",
    entry_point="miniworld.envs.jeparoom:JEPAENV",
    kwargs={"size": 12, "seed": random.randint(0, 2**31 - 1)},
    max_episode_steps=500,
    )
    CollectTrajectories(save_images=True, n_samples=10000)