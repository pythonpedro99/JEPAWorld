import os
import pyglet
os.environ['PYGLET_HEADLESS'] = 'True'
# pyglet.options['headless'] = True
# pyglet.options['shadow_window'] = False
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.envs.registration import register
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from policies.rearrange import HumanLikeRearrangePolicy

from Miniworld.miniworld.envs.jeparoom import RearrangeOneRoom

def _register_environment(env_id: str) -> None:
    register(
        id=env_id,
        entry_point=RearrangeOneRoom,  # ✅ pass class directly
        kwargs={"size": 12, "seed": random.randint(0, 2**31 - 1)},
        max_episode_steps=250,
    )


class CollectTrajectories:
    def __init__(
        self,
        env_id: str = "RearrangeOneRoom-v0",
        n_samples: int = 1000,
        save_images: bool = False,
        output_dir: str = "/Users/julianquast/Documents/Bachelor Thesis/Datasets/rearrange_1k"
    ) -> None:
        self.env_id = env_id
        self.save_images = save_images
        self.output_dir = output_dir

        self.master_seed = random.randint(0, 2**31 - 1)
        self.rng = np.random.default_rng(self.master_seed)

        os.makedirs(self.output_dir, exist_ok=True)
        if self.save_images:
            os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)

        self.actions: List[int] = []
        self.obs_shape: Tuple[int, int, int] = (224, 224, 3)
        self.obs_memmap: Optional[np.memmap] = None

        actions_path = os.path.join(self.output_dir, "actions.npy")
        metadata_path = os.path.join(self.output_dir, "metadata.json")
        obs_path = os.path.join(self.output_dir, "obs_trimmed.dat")

        self.offset = 0
        if os.path.exists(obs_path):
            obs_size = np.prod(self.obs_shape)
            statinfo = os.stat(obs_path)
            self.offset = statinfo.st_size // (obs_size * np.dtype(np.uint8).itemsize)
            print(f"Resuming from {self.offset} existing observations.")

        self.total = self.offset
        self.n_samples_new = n_samples
        self.n_samples_total = self.total + self.n_samples_new
        self.max_samples = self.n_samples_total + 200

        if not self.save_images:
            self.obs_memmap = np.memmap(
                obs_path,
                dtype=np.uint8,
                mode='r+' if os.path.exists(obs_path) else 'w+',
                shape=(self.max_samples, *self.obs_shape)
            )

        if os.path.exists(actions_path):
            self.actions = np.load(actions_path).tolist()
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                self.metadata = json.load(f)
        else:
            self.metadata: Dict[str, Any] = {
                "env_id": self.env_id,
                "master_seed": self.master_seed,
                "episodes": []
            }

        self._collect_loop()

        np.save(
            actions_path,
            np.array(self.actions, dtype=np.int32)
        )
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

        if not self.save_images:
            self._trim_memmap_file()

        print("Collection complete.")
        if not self.save_images:
            print(
                f"Observations stored in {os.path.join(self.output_dir, 'obs_trimmed.dat')} (memmap file)"
            )

    def _collect_loop(self) -> None:
        episode_idx = len(self.metadata["episodes"])
        ep_seed = 340 + episode_idx

        while self.total < self.n_samples_total:
            ep_seed += 1
            episode_idx += 1

            env = gym.make(
                self.env_id,
                seed=ep_seed,
                obs_width=224,
                obs_height=224,
                domain_rand=True
            )

            try:
                policy = HumanLikeRearrangePolicy(env=env, seed=ep_seed)
                success = policy.rearrange()
                if not success:
                    print(f"Episode {episode_idx} failed; skipping.")
                    episode_idx -= 1
                    continue

                actions = policy.actions
                observations = policy.observations

                self.metadata["episodes"].append({
                    "episode": episode_idx,
                    "seed": ep_seed,
                    "n_actions": len(actions),
                    "n_observations": len(observations)
                })

                for cmd, obs in zip(actions, observations):
                    if self.total >= self.max_samples:
                        print("Maximum sample buffer reached.")
                        return

                    self.actions.append(cmd)
                    if self.save_images:
                        img_path = os.path.join(
                            self.output_dir,
                            "images",
                            f"obs_{self.total+1:06d}.png"
                        )
                        plt.imsave(img_path, obs)
                    else:
                        assert self.obs_memmap is not None
                        self.obs_memmap[self.total] = obs.astype(np.uint8)

                    self.total += 1
                    pct = (self.total - self.offset) / self.n_samples_new * 100
                    print(f"Progress: {pct:.2f}% ({self.total - self.offset}/{self.n_samples_new})", end='\r')
            finally:
                env.close()

        print(f"\nCollected: {self.n_samples_new} new samples (total: {self.total})")

    def _trim_memmap_file(self) -> None:
        if self.obs_memmap is not None:
            self.obs_memmap.flush()
            del self.obs_memmap

        trimmed_path = os.path.join(self.output_dir, "obs_trimmed.dat")
        trimmed_memmap = np.memmap(
            trimmed_path,
            dtype=np.uint8,
            mode='r+',
            shape=(self.total, *self.obs_shape)
        )
        trimmed_memmap.flush()
        del trimmed_memmap


if __name__ == "__main__":
    _register_environment("RearrangeOneRoom-v0")
    CollectTrajectories(save_images=False, n_samples=150000)
