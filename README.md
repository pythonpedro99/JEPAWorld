# JEPAWorld Usage Guide

This repository contains a simple MiniWorld environment along with helper utilities and a policy for collecting
trajectories. The most relevant code can be found in the following files:

- `Miniworld/miniworld/envs/jeparoom.py` – defines the `RearrangeOneRoom-v0` environment used for data collection.
- `policies/helpers.py` – helper dataclasses and functions for constructing a PRM and saving datasets.
- `policies/rearrange.py` – contains `HumanLikeRearrangePolicy` used to generate actions.
- `scripts/collect_trajectories.py` – script that instantiates the environment and policy to record trajectories.

## Environment

The environment is implemented in `Miniworld/miniworld/envs/jeparoom.py`. The file defines a `RearrangeOneRoom-v0` class
which inherits from `MiniWorldEnv` and creates a single rectangular room with randomised colours and
objects:

```python
class RearrangeOneRoom-v0(MiniWorldEnv, utils.EzPickle):
    """
    Single-room environment with randomized wall, floor, ceiling colors,
    and randomly placed objects (balls, boxes, keys) that are visually distinguishable.
    """
```

The `_gen_world` method constructs the room and places entities using `add_rect_room`.

## Helpers and Policy

`policies/helpers.py` defines lightweight dataclasses (`Room`, `Obstacle`, `GraphData`) and utilities such as
`get_graph_data` and `build_prm_graph_single_room` to build a PRM for navigation. It also provides
`save_data_batch` for writing observations and actions to disk.

The policy located in `policies/rearrange.py` is `HumanLikeRearrangePolicy`. It uses the helper functions to
plan and execute a sequence of pick-and-place actions inside the room.

## Collecting Trajectories

The script `scripts/collect_trajectories.py` registers the environment and runs the policy to gather data. The
`CollectTrajectories` class takes parameters such as the environment id, number of samples to collect and the
output directory for images or memmap files.

To install the required dependencies, create a Python virtual environment and run:

```bash
python -m venv .venv
source .venv/bin/activate  
```
then
```bash
pip install -r requirements.txt
cd Miniworld
pip install -e .
cd ..
```

After installation you can launch dataset collection with:

```bash
source .venv/bin/activate  
python scripts/collect_trajectories.py
```

Additional options can be passed by editing the `CollectTrajectories` instantiation at the bottom of the
script, e.g. adjusting `n_samples` or enabling `save_images`.

On macOS, we recommend setting n_samples <= 7000 per run due to memory constraints. You can run the script repeatedly — it will automatically detect the last used seed and episode, and append new data to the existing file. 

## Inspect the env in manual control mode via the keyboard 

```bash
source .venv/bin/activate  
python Miniworld/scripts/manual_control.py --env-name RearrangeOneRoom-v0 --domain-rand
```