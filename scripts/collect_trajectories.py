import os
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



def _register_environment(env_id: str) -> None:
    """
    Register the MiniWorld JEPAENV environment.

    Parameters
    ----------
    env_id : str
        Identifier for the environment to register.

    Returns
    -------
    None
    """
    register(
        id=env_id,
        entry_point="miniworld.envs.jeparoom:JEPAENV",
        kwargs={"size": 12, "seed": random.randint(0, 2**31 - 1)},
        max_episode_steps=500,
    )


class CollectTrajectories:
    """
    Collect trajectories from a MiniWorld environment using a human-like rearrangement policy.

    Attributes
    ----------
    env_id : str
        Gym environment identifier.
    n_samples : int
        Number of samples to collect.
    save_images : bool
        Flag to save observations as image files.
    output_dir : str
        Directory to save collected data.
    master_seed : int
        Random seed for reproducibility.
    rng : np.random.Generator
        NumPy random number generator.
    actions : List[int]
        Recorded actions from all trajectories.
    obs_shape : Tuple[int, int, int]
        Shape of observation images (height, width, channels).
    max_samples : int
        Maximum buffer size for observations.
    obs_memmap : Optional[np.memmap]
        Memory-mapped array for storing observations when save_images is False.
    metadata : Dict[str, Any]
        Metadata for collected episodes.
    """

    def __init__(
        self,
        env_id: str = "JEPAENV-v0",
        n_samples: int = 1000,
        save_images: bool = False,
        output_dir: str = "./saved_data"
    ) -> None:
        """
        Initialize trajectory collection parameters and start the collection loop.

        Parameters
        ----------
        env_id : str, optional
            Gym environment identifier, by default "JEPAENV-v0".
        n_samples : int, optional
            Number of samples to collect, by default 1000.
        save_images : bool, optional
            Whether to save observations as image files, by default False.
        output_dir : str, optional
            Directory to save actions, observations, and metadata, by default "./saved_data".

        Returns
        -------
        None
        """
        self.env_id = env_id
        self.n_samples = n_samples
        self.save_images = save_images
        self.output_dir = output_dir

        self.master_seed = random.randint(0, 2**31 - 1)
        self.rng = np.random.default_rng(self.master_seed)

        os.makedirs(self.output_dir, exist_ok=True)
        if self.save_images:
            os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)

        self.actions: List[int] = []
        self.obs_shape: Tuple[int, int, int] = (128, 224, 3)
        self.max_samples = self.n_samples + 200
        self.obs_memmap: Optional[np.memmap] = None

        if not self.save_images:
            self.obs_memmap = np.memmap(
                os.path.join(self.output_dir, "obs.dat"),
                dtype=np.uint8,
                mode='w+',
                shape=(self.max_samples, *self.obs_shape)
            )

        self.metadata: Dict[str, Any] = {
            "env_id": self.env_id,
            "master_seed": self.master_seed,
            "n_samples": self.n_samples,
            "episodes": []
        }

        self._collect_loop()

        np.save(
            os.path.join(self.output_dir, "actions.npy"),
            np.array(self.actions, dtype=np.int32)
        )
        with open(os.path.join(self.output_dir, "metadata.json"), "w") as f:
            json.dump(self.metadata, f, indent=2)

        if not self.save_images:
            self._trim_memmap_file()

        print("Collection complete.")
        if not self.save_images:
            print(
                f"Observations stored in {os.path.join(self.output_dir, 'obs_trimmed.dat')} (memmap file)"
            )

    def _collect_loop(self) -> None:
        """
        Execute environment episodes until the desired number of samples is collected.

        Returns
        -------
        None
        """
        total = 0
        episode_idx = 0
        ep_seed = 0

        while total < self.n_samples:
            ep_seed += 1
            episode_idx += 1
            env = gym.make(
                self.env_id,
                seed=ep_seed,
                obs_width=1420,
                obs_height=580,
                domain_rand=True
            )
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
                if total >= self.max_samples:
                    print("Maximum sample buffer reached.")
                    return

                self.actions.append(cmd)
                if self.save_images:
                    img_path = os.path.join(
                        self.output_dir,
                        "images",
                        f"obs_{total+1:06d}.png"
                    )
                    plt.imsave(img_path, obs)
                else:
                    assert self.obs_memmap is not None
                    self.obs_memmap[total] = obs.astype(np.uint8)

                total += 1
                pct = total / self.n_samples * 100
                print(f"Progress: {pct:.2f}% ({total}/{self.n_samples})", end='\r')

        print(f"\nCollected: {total} samples in {episode_idx} episodes.")

    def _trim_memmap_file(self) -> None:
        """
        Trim the memory-mapped observation file to the exact number of samples collected.

        Returns
        -------
        None
        """
        if self.obs_memmap is not None:
            self.obs_memmap.flush()
            del self.obs_memmap

        trimmed_path = os.path.join(self.output_dir, "obs_trimmed.dat")
        trimmed_memmap = np.memmap(
            trimmed_path,
            dtype=np.uint8,
            mode='w+',
            shape=(self.n_samples, *self.obs_shape)
        )

        original_memmap = np.memmap(
            os.path.join(self.output_dir, "obs.dat"),
            dtype=np.uint8,
            mode='r',
            shape=(self.max_samples, *self.obs_shape)
        )
        trimmed_memmap[:] = original_memmap[:self.n_samples]
        trimmed_memmap.flush()
        del trimmed_memmap
        del original_memmap
        os.remove(os.path.join(self.output_dir, "obs.dat"))


if __name__ == "__main__":
    _register_environment("JEPAENV-v0")
    CollectTrajectories(save_images=True, n_samples=1000)
