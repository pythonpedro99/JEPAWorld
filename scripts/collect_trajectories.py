###############################################################################
# CollectTrajectories  (resume-fähig + saubere Zählung)
#
#  • Verwendet ZWEI unabhängige Zähler:
#      - seed_counter    → wird bei JEDER Policy-Ausführung erhöht
#      - episode_counter → nur bei erfolgreicher Episode erhöht
#
#  • Beim Resume liest die Klasse metadata.json ein und setzt
#      episode_counter   = letzte_episode + 1
#      seed_counter      = letzter_seed    + 1
#
#  • Liefert GENAU n_episodes neue, erfolgreiche Trajektorien.
###############################################################################
import os, json, random, sys, shutil
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

    # ------------------------------------------------------------------ #
    def __init__(
        self,
        env_id: str = "RearrangeOneRoom-v0",
        n_episodes: int = 1_000,             # wie viele NEUE Episoden sammeln?
        save_images: bool = False,           # PNG-Frames statt obs.npy speichern
        output_dir: str = (
            "/Users/julianquast/Documents/Bachelor Thesis/Datasets/rearrange_1k"
        ),
        overwrite: bool = False,             # True → Sammlung komplett löschen
        base_seed: int = 0,                  # Startseed beim allerersten Lauf
    ) -> None:

        self.env_id = env_id
        self.save_images = save_images
        self.output_dir = Path(output_dir)
        self.episodes_dir = self.output_dir / "episodes"

        # ---------------- Ordner vorbereiten ----------------
        if overwrite and self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        if self.save_images:
            (self.output_dir / "images").mkdir(exist_ok=True)

        # -------------- Metadata laden / anlegen ------------
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

        # -------- Resume-Info EINMALIG beim Start bestimmen --
        if self.metadata["episodes"]:
            last_ep   = self.metadata["episodes"][-1]["episode"]
            last_seed = self.metadata["episodes"][-1]["seed"]
            self.episode_counter = last_ep + 1
            self.seed_counter    = last_seed + 1
        else:
            self.episode_counter = 1
            self.seed_counter    = base_seed

        print(f"▶️  Starte bei Episode {self.episode_counter}, Seed {self.seed_counter}")

        # -------------- Sammeln -----------------------------
        self._collect_loop(n_episodes)

        # -------------- Metadata speichern ------------------
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

        print(f"✅ Fertig. Neue Episoden liegen in {self.episodes_dir}")

    # ------------------------------------------------------------------ #
    def _collect_loop(self, n_episodes: int) -> None:
        """Sammelt genau *n_episodes* erfolgreiche Trajektorien."""
        success_count = 0
        while success_count < n_episodes:
            ep_num  = self.episode_counter
            ep_seed = self.seed_counter

            ep_folder = self.episodes_dir / f"ep_{ep_num:04d}"
            if ep_folder.exists():
                raise RuntimeError(
                    f"Episode-Ordner {ep_folder} existiert bereits. "
                    "Prüfe deine Sammlung oder nutze overwrite=True."
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
                    print(f"Episode {ep_num:04d} fehlgeschlagen – übersprungen.")
                    # Nur der Seed-Zähler wird erhöht
                    self.seed_counter += 1
                    continue

                actions = np.asarray(policy.actions, dtype=np.int32)
                observations = np.asarray(policy.observations, dtype=np.uint8)

                # ---------- Episode speichern ----------
                ep_folder.mkdir()
                np.save(ep_folder / "actions.npy", actions)

                if self.save_images:
                    for i, img in enumerate(observations):
                        plt.imsave(ep_folder / f"obs_{i:04d}.png", img)
                else:
                    np.save(ep_folder / "obs.npy", observations)

                # ---------- Metadata erweitern ----------
                self.metadata["episodes"].append(
                    {
                        "episode": ep_num,
                        "seed": ep_seed,
                        "n_actions": int(actions.shape[0]),
                        "n_observations": int(observations.shape[0]),
                    }
                )

                print(f"Ep {ep_num:04d} | act {actions.shape[0]:3d} | "
                      f"obs {observations.shape[0]:3d}")

                # Zähler nur bei Erfolg erhöhen
                self.episode_counter += 1
                self.seed_counter    += 1
                success_count        += 1

            finally:
                env.close()


# ------------------------------------------------------------------ #
if __name__ == "__main__":
    _register_environment("RearrangeOneRoom-v0")
    CollectTrajectories(
        n_episodes=50,         # sammelt 200 weitere erfolgreiche Episoden
        save_images=False,
        overwrite=False,        # False = an bestehende Sammlung anhängen
        base_seed=0           # Startseed beim allerersten Lauf
    )
