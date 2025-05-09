import os
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Any, List, Optional, Tuple
from Miniworld.miniworld.entity import Agent as MWAgent 
try:
    import torch
except ImportError:
    torch = None

# Fixed directory where every call will drop its PNG
_OBS_SAVE_DIR = Path("saved_obs")
_OBS_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Simple counter so filenames don’t collide
_obs_counter = 0

def save_observation(obs) -> None:
    """
    Save one env-observation as a PNG in `saved_obs/` and return the filepath.

    Args:
        obs: whatever env.step() or env.reset() returned as the image—
             can be a dict, torch.Tensor, float array, CHW or HWC, etc.

    Returns:
        The full path (string) to the written PNG.
    """
    global _obs_counter

    # 1) Pull out the pixel array if it’s a dict
    if isinstance(obs, dict):
        for key in ("pixels", "image", "frame", "rgb"):
            if key in obs:
                obs = obs[key]
                break
        else:
            raise ValueError(f"save_observation: no image key in obs dict, got keys={list(obs)}")

    # 2) Tensor → NumPy
    if torch is not None and isinstance(obs, torch.Tensor):
        obs = obs.cpu().numpy()

    # 3) Float [0,1] → uint8
    if np.issubdtype(getattr(obs, "dtype", None), np.floating):
        obs = (obs * 255).clip(0,255).astype(np.uint8)

    # 4) CHW → HWC
    if obs.ndim == 3 and obs.shape[0] in (1,3) and obs.shape[1] != obs.shape[2]:
        obs = np.transpose(obs, (1,2,0))

    # 5) Greyscale → RGB
    if obs.ndim == 3 and obs.shape[2] == 1:
        obs = np.repeat(obs, 3, axis=2)

    # 6) Sanity check
    if obs.ndim != 3 or obs.shape[2] not in (3,):
        raise ValueError(f"save_observation: unsupported obs shape {obs.shape}")

    # 7) Write out
    filename = _OBS_SAVE_DIR / f"obs_{_obs_counter:05d}.png"
    Image.fromarray(obs).save(filename)
    _obs_counter += 1




def get_agent_info(entities):
    """
    Return (pos, yaw) for the first object in `entities`
    that has both .pos and .yaw attributes.
    """
    for ent in entities:
        if hasattr(ent, "pos") and hasattr(ent, "dir") and hasattr(ent, "cam_dir"):
            return ent.pos, ent.dir
    return None


