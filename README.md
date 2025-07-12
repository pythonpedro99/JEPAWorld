# JEPAWorld Usage Guide

This repository contains the **MiniWorld-based `RearrangeOneRoom` environment** along with helper utilities and a policy for collecting trajectories. The most relevant code can be found in:

- `Miniworld/miniworld/envs/jeparoom.py` – defines the `RearrangeOneRoom-v0` environment used for data collection.
- `policies/helpers.py` – helper dataclasses and functions for constructing a PRM and saving datasets.
- `policies/rearrange.py` – contains `HumanLikeRearrangePolicy` used to generate actions.
- `scripts/collect_trajectories.py` – script that instantiates the environment and policy to record trajectories.
- `notebooks/01_data_inspection.ipynb` - jupyter notebook for data inspection, sanity checks and training preparation.

---

## Environment

The environment is implemented in `Miniworld/miniworld/envs/jeparoom.py`, where the class `RearrangeOneRoom-v0` inherits from `MiniWorldEnv`. It creates a single rectangular room with randomized elements:

```python
class RearrangeOneRoom-v0(MiniWorldEnv, utils.EzPickle):
    """
    Single-room environment with randomized wall, floor, ceiling colors,
    and randomly placed objects (balls, boxes, keys) that are visually distinguishable.
    """
```

The `_gen_world()` method uses `add_rect_room` to build the layout and place objects. Example visuals:

![RearrangeOneRoom](assets/trajectory_expert_short.jpg)

---

## Rearrange Datasets

- The rearrange_1k dataset (20GiB) can be downloaded here: [rearrange_1k](https://drive.google.com/file/d/1bFiwR0jXowX9YP7Wk7H042glMMlRA7CJ/view?usp=sharing)
---

## Helpers and Policy

The file `policies/helpers.py` defines lightweight dataclasses:

- `Room`, `Obstacle`, `GraphData`
- Functions like `get_graph_data` and `build_prm_graph_single_room` to construct Probabilistic Roadmaps (PRMs)

The core policy in `policies/rearrange.py` is `HumanLikeRearrangePolicy`, which plans and performs a series of **pick-and-place** operations, relying on PRM-based path planning.

---

## Collecting Trajectories

The trajectory collection is controlled by `scripts/collect_trajectories.py`, where the
`CollectTrajectories` class exposes a few parameters:

- `env_id` – gym environment id (defaults to `RearrangeOneRoom-v0`)
- `n_episodes` – number of episodes to record
- `batch_size` – if supplied, run in *driver mode* that spawns subprocesses to
  collect episodes in batches
- `save_images` – store observations as PNG files instead of `obs.npy`
- `overwrite` – remove any existing dataset when starting the first batch
- `base_seed` – seed for the first episode (defaults to `0`)

**Installation Steps**:

```bash
python -m venv .venv
source .venv/bin/activate  
pip install -r requirements.txt
cd Miniworld
pip install -e .
cd ..
```

**Start data collection** by activating the environment and running the
`collect_trajectories.py` script. Below are examples for both single-run and
driver modes.

```bash
# single run collecting 1000 episodes
source .venv/bin/activate
python scripts/collect_trajectories.py \
  --env_id RearrangeOneRoom-v0 \
  --n_episodes 1000 \
  --output_dir data/test_episodes \
  --overwrite

# driver mode: collect 2000 episodes in batches of 30
python scripts/collect_trajectories.py \
  --env_id RearrangeOneRoom-v0 \
  --n_episodes 2000 \
  --batch_size 30 \
  --output_dir data/test_episodes
```

Customize trajectory collection by editing the `CollectTrajectories` instantiation (e.g., `n_episodes`, `save_images`, etc.).

> 💡 On **macOS**, use driver mode and set `--batch_size <= 30` to avoid pyglet issues. The script supports resumable collection — appending new episodes automatically.

---

## Manual Inspection (Keyboard Control)

Use keyboard input to manually explore the environment:

```bash
source .venv/bin/activate  
python Miniworld/scripts/manual_control.py --env-name RearrangeOneRoom-v0 --domain-rand
```

---

## License and Contact

This project is released under the **MIT License**.

For questions, suggestions, or collaboration inquiries, please contact:

📧 **Julian Quast**  
`julian.quast@campus.tu-berlin.de`
