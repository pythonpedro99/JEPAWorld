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
            os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)

        # storage: actions in memory, observations on disk if save_images=False
        self.actions: list[int] = []
        if not self.save_images:
            # memmap dtype assumed uint8 for image pixels
            self.obs_memmap = np.memmap(
                os.path.join(self.output_dir, "obs.dat"),
                dtype=np.uint8,
                mode='w+',
                shape=(self.n_samples, 224, 224, 3)
            )

        # metadata
        self.metadata = {
            "env_id": self.env_id,
            "master_seed": self.master_seed,
            "n_samples": self.n_samples,
            "episodes": []
        }

        # begin collection
        self._collect_loop()

        # save actions + metadata
        np.save(
            os.path.join(self.output_dir, "actions.npy"),
            np.array(self.actions, dtype=np.int32)
        )
        with open(os.path.join(self.output_dir, "metadata.json"), "w") as f:
            json.dump(self.metadata, f, indent=2)

        print("Collection complete.")
        if not self.save_images:
            print(f"Observations saved to {os.path.join(self.output_dir, 'obs.dat')} (memmap file)")

    def _collect_loop(self):
        total = 0
        episode_idx = 0
        ep_seed = 0

        while total < self.n_samples:
            ep_seed += 1
            episode_idx += 1

            # recreate environment with domain randomization
            env = gym.make(
                self.env_id,
                seed=ep_seed,
                obs_width=224,
                obs_height=224,
                domain_rand=True
            )

            # run policy
            policy = HumanLikeRearrangePolicy(env=env, seed=ep_seed)
            if not policy.rearrange():
                print(f"Episode {episode_idx} failed; skipping.")
                episode_idx -= 1
                #ep_seed -= 1
                continue

            actions = policy.actions
            observations = policy.observations

            # record episode metadata
            self.metadata["episodes"].append({
                "episode": episode_idx,
                "seed": ep_seed,
                "n_actions": len(actions),
                "n_observations": len(observations)
            })

            for cmd, obs in zip(actions, observations):
                
                # write action
                self.actions.append(cmd)

                if self.save_images:
                    # save image file
                    img_path = os.path.join(
                        self.output_dir, "images", f"obs_{total+1:06d}.png"
                    )
                    plt.imsave(img_path, obs)
                else:
                    # write to memmap and drop in-memory obs
                    self.obs_memmap[total] = obs.astype(np.uint8)

                total += 1
                # progress print
                pct = total / self.n_samples * 100
                print(f"Progress: {pct:.2f}% ({total}/{self.n_samples})", end='\r')

        print(f"\nCollected {total} samples in {episode_idx} episodes.")


if __name__ == "__main__":
    register(
        id="JEPAENV-v0",
        entry_point="miniworld.envs.jeparoom:JEPAENV",
        kwargs={"size": 12, "seed": random.randint(0, 2**31 - 1)},
        max_episode_steps=500,
    )
    CollectTrajectories(save_images=True, n_samples=1000)