import os
import json
import random
import sys
import shutil
from pathlib import Path
from typing import Any, Dict, Tuple

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
        entry_point=RearrangeOneRoom,
        kwargs={"size": 12, "seed": random.randint(0, 2**31 - 1)},
        max_episode_steps=250,
    )


class CollectTrajectories:
    OBS_SHAPE: Tuple[int, int, int] = (224, 224, 3)

    def __init__(
        self,
        env_id: str = "RearrangeOneRoom-v0",
        n_episodes: int = 1_000,
        save_images: bool = False,
        output_dir: str = (
            "data/test_episodes"
        ),
        overwrite: bool = False,
        base_seed: int = 0,
    ) -> None:

        self.env_id = env_id
        self.save_images = save_images
        self.output_dir = Path(output_dir)
        self.episodes_dir = self.output_dir / "episodes"

        # Prepare directories
        if overwrite and self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        if self.save_images:
            (self.output_dir / "images").mkdir(exist_ok=True)

        # Load or init metadata
        self.metadata_path = self.output_dir / "metadata.json"
        if self.metadata_path.exists() and not overwrite:
            with open(self.metadata_path) as f:
                self.metadata: Dict[str, Any] = json.load(f)
        else:
            self.metadata = {
                "env_id": self.env_id,
                "master_seed": random.randint(0, 2**31 - 1),
                "episodes": [],
            }
            self._save_metadata()

        # Determine starting counters
        if self.metadata["episodes"]:
            last = self.metadata["episodes"][-1]
            self.episode_counter = last["episode"] + 1
            self.seed_counter = last["seed"] + 1
        else:
            self.episode_counter = 1
            self.seed_counter = base_seed

        print(f"▶️  Starting at Episode {self.episode_counter}, Seed {self.seed_counter}")

        # Collect trajectories
        self._collect_loop(n_episodes)

        print(f"✅ Done. New episodes in {self.episodes_dir}")

    def _save_metadata(self) -> None:
        """
        Persist current metadata to disk.
        """
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def _collect_loop(self, n_episodes: int) -> None:
        """Collect exactly n_episodes successful trajectories."""
        success_count = 0
        while success_count < n_episodes:
            ep_num = self.episode_counter
            ep_seed = self.seed_counter

            ep_folder = self.episodes_dir / f"ep_{ep_num:04d}"
            if ep_folder.exists():
                raise RuntimeError(
                    f"Episode folder {ep_folder} already exists. "
                    "Use overwrite=True if you want to start fresh."
                )

            env = gym.make(
                self.env_id,
                seed=ep_seed,
                obs_width=self.OBS_SHAPE[1],
                obs_height=self.OBS_SHAPE[0],
                domain_rand=True,
            )

            try:
                policy = HumanLikeRearrangePolicy(env=env, seed=ep_seed)
                if not policy.rearrange():
                    print(f"Episode {ep_num:04d} failed, skipping.")
                    self.seed_counter += 1
                    continue

                actions = np.asarray(policy.actions, dtype=np.int32)
                observations = np.asarray(policy.observations, dtype=np.uint8)

                # Save episode data
                ep_folder.mkdir()
                np.save(ep_folder / "actions.npy", actions)
                if self.save_images:
                    for i, img in enumerate(observations):
                        plt.imsave(ep_folder / f"obs_{i:04d}.png", img)
                else:
                    np.save(ep_folder / "obs.npy", observations)

                # Append metadata
                self.metadata["episodes"].append({
                    "episode": ep_num,
                    "seed": ep_seed,
                    "n_actions": int(actions.shape[0]),
                    "n_observations": int(observations.shape[0]),
                })
                # Persist metadata after each episode
                self._save_metadata()

                print(
                    f"Ep {ep_num:04d} | acts {actions.shape[0]:3d} | "
                    f"obs {observations.shape[0]:3d}"
                )

                # Increment counters on success
                self.episode_counter += 1
                self.seed_counter += 1
                success_count += 1
            finally:
                env.close()


if __name__ == "__main__":
    _register_environment("RearrangeOneRoom-v0")
    CollectTrajectories(
        n_episodes=2,
        save_images=True,
        overwrite=False,
        base_seed=0,
    )
