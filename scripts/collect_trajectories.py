import os
import sys
import json
import random
import shutil
import subprocess
import argparse
from pathlib import Path
from typing import Any, Dict, Tuple, List

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.envs.registration import register

# Ensure your project root is on PYTHONPATH so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from policies.rearrange import HumanLikeRearrangePolicy
from policies.helpers import get_graph_data
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
        env_id: str,
        n_episodes: int,
        save_images: bool,
        output_dir: str,
        overwrite: bool,
        base_seed: int,
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

        # Starting counters
        if self.metadata["episodes"]:
            last = self.metadata["episodes"][-1]
            self.episode_counter = last["episode"] + 1
            self.seed_counter = last["seed"] + 1
        else:
            self.episode_counter = 1
            self.seed_counter = base_seed

        print(f"▶️  Starting batch: episodes {self.episode_counter}→"
              f"{self.episode_counter + n_episodes - 1}, seeds {self.seed_counter}→…")

        # Register and collect
        _register_environment(self.env_id)
        self._collect_loop(n_episodes)
        print(f"✅ Batch complete. Episodes now up to {self.episode_counter - 1}")

    def _save_metadata(self) -> None:
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def _collect_loop(self, n_episodes: int) -> None:
        success = 0
        while success < n_episodes:
            ep = self.episode_counter
            seed = self.seed_counter

            try:
                # Create 4 environments with the same seed
                
                
                probe_env = gym.make(
                        self.env_id,
                        seed=seed,
                        obs_width=self.OBS_SHAPE[1],
                        obs_height=self.OBS_SHAPE[0],
                        domain_rand=True,
                        render_mode="rgb_array",
                    )
                graph_data = get_graph_data(probe_env)
                object_nodes: List[str] = [
                    o.node_name
                    for o in graph_data.obstacles
                    if o.type in ("Box", "Ball", "Key")]
                envs = [
                    gym.make(
                        self.env_id,
                        seed=seed,
                        obs_width=self.OBS_SHAPE[1],
                        obs_height=self.OBS_SHAPE[0],
                        domain_rand=True,
                        render_mode="rgb_array",
                    ) for _ in range(len(object_nodes))
                ]

                # Create policies
                policies = [HumanLikeRearrangePolicy(env=env, seed=seed,object_node=target) for env, target in zip(envs,object_nodes)]

                # Execute each policy and store results
                for i, (env, policy) in enumerate(zip(envs, policies)):
                    ep_folder = self.episodes_dir / f"ep_{ep:04d}_{i}"
                    if ep_folder.exists():
                        raise RuntimeError(f"{ep_folder} already exists")

                    ok = policy.rearrange()
                    env.close()

                    if not ok:
                        raise RuntimeError(f"Policy {i} returned False")

                    actions = np.asarray(policy.actions, dtype=np.int32)
                    obs = np.asarray(policy.observations, dtype=np.uint8)

                    ep_folder.mkdir()
                    np.save(ep_folder / "actions.npy", actions)

                    if self.save_images:
                        for j, img in enumerate(obs):
                            plt.imsave(ep_folder / f"obs_{j:04d}.png", img)
                    else:
                        np.save(ep_folder / "obs.npy", obs)

                    self.metadata["episodes"].append({
                        "episode": f"{ep}_{i}",
                        "seed": seed,
                        "n_actions": int(actions.shape[0]),
                        "n_observations": int(obs.shape[0]),
                    })

                    print(f"✅ Ep {ep:04d}_{i} | acts {actions.shape[0]:3d} | obs {obs.shape[0]:3d}")

                self._save_metadata()
                success += 1
                self.episode_counter += 1
                self.seed_counter += 1

            except Exception as e:
                print(f"❌ Ep {ep:04d}, Seed {seed} failed: {e}")
                self.seed_counter += 1



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="RearrangeOneRoom-v0")
    parser.add_argument("--n_episodes", type=int, required=True,
                        help="Number of episodes (or total if using --batch_size).")
    parser.add_argument("--batch_size", type=int,
                        help="If set, run in driver mode: spawn subprocesses collecting batches of this size until total is reached.")
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--output_dir", default="data/test_episodes")
    parser.add_argument("--overwrite", action="store_true",
                        help="Only applies to the very first batch (clears existing data).")
    parser.add_argument("--base_seed", type=int, default=0)
    args = parser.parse_args()

    # Driver mode: spawn subprocesses in chunks of --batch_size
    if args.batch_size:
        total = args.n_episodes
        batch = args.batch_size
        collected = 0
        current_seed = args.base_seed
        first = True

        # Ensure output dir & metadata are fresh if first batch
        if args.overwrite and Path(args.output_dir).exists():
            shutil.rmtree(args.output_dir)

        while collected < total:
            this_batch = min(batch, total - collected)
            cmd = [
                sys.executable, __file__,
                "--env_id", args.env_id,
                "--n_episodes", str(this_batch),
                "--output_dir", args.output_dir,
                "--base_seed", str(current_seed),
            ]
            if args.save_images:
                cmd.append("--save_images")
            if first and args.overwrite:
                cmd.append("--overwrite")
            print(f"▶️  Launching subprocess for {this_batch} eps from seed {current_seed}")
            subprocess.run(cmd, check=True)
            first = False

            # Reload metadata to get how many we have and last seed
            md_path = Path(args.output_dir) / "metadata.json"
            with open(md_path) as f:
                md = json.load(f)
            collected = len(md["episodes"])
            last_seed = md["episodes"][-1]["seed"]
            current_seed = last_seed + 1
            print(f"🔁 Collected {collected}/{total}, next seed {current_seed}")

        print("✅ All done!")

    else:
        # Single-run mode: collect exactly n_episodes then exit
        CollectTrajectories(
            env_id=args.env_id,
            n_episodes=args.n_episodes,
            save_images=args.save_images,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            base_seed=args.base_seed,
        )
