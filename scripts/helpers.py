import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from typing import Union, Sequence
from PIL import Image
import numpy as np
import networkx as nx
from shapely.geometry import Point, Polygon, box
from shapely import affinity
from pathlib import Path
from typing import Union, Tuple
from datetime import datetime

import numpy as np
from PIL import Image


def plot_prm_graph(
    data,  # GraphData returned by get_graph_data()
    G: nx.Graph,
    node_positions: Dict[str, Tuple[float, float]],
    obstacle_nodes: List[str],
    highlight_path: Optional[List[str]] = None,
    smoothed_curve: Optional[List[Tuple[float, float]]] = None
) -> None:
    """
    Plot the PRM roadmap over the environment, using only the three values
    returned by build_prm_graph plus the original GraphData. Optionally plot
    a smoothed curve of agent waypoints.

    Args:
        data:          GraphData (rooms, agent, obstacles).
        G:             The PRM networkx graph.
        node_positions: Mapping node_id -> (x, z) coords.
        obstacle_nodes: List of node_ids corresponding to obstacles.
        highlight_path: Optional list of node_ids to draw in red.
        smoothed_curve: Optional list of (x, z) points showing the smoothed path,
                        or a list of such lists (history).
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Reconstruct room polygons
    room_polys = {r.id: Polygon(r.vertices) for r in data.rooms}
    for poly in room_polys.values():
        x, z = poly.exterior.xy
        ax.add_patch(MplPolygon(
            list(zip(x, z)),
            closed=True, facecolor="lightgray", edgecolor="black", alpha=0.5
        ))

    # Reconstruct obstacle buffers as rectangles
    obstacle_buffers = []
    for obs in data.obstacles:
        w, d = obs.size
        w = max(w, 0.5)
        d = max(d, 0.5)
        rect = box(-w / 2, -d / 2, w / 2, d / 2)
        rect = affinity.rotate(rect, obs.yaw, use_radians=True)
        rect = affinity.translate(rect, obs.pos[0], obs.pos[1])
        obstacle_buffers.append(rect)

    for buf in obstacle_buffers:
        if buf.is_empty:
            continue
        x, z = buf.exterior.xy
        ax.add_patch(
            MplPolygon(
                list(zip(x, z)),
                closed=True,
                facecolor="red",
                edgecolor="red",
                alpha=0.3,
            )
        )

    # Draw PRM edges
    for u, v in G.edges():
        x1, z1 = node_positions[u]
        x2, z2 = node_positions[v]
        ax.plot([x1, x2], [z1, z2], color="skyblue", linewidth=0.5, alpha=0.7)

    # Draw nodes
    for nid, (x, z) in node_positions.items():
        if nid == "agent":
            ax.scatter(x, z, color="gold", s=100, label="Agent", zorder=5)
        elif "_p" in nid:
            ax.scatter(x, z, color="green", s=50, label="Portal", zorder=5)
        elif nid in obstacle_nodes:
            ax.scatter(x, z, color="purple", s=80, label="Obstacle", zorder=5)
        elif "_s" in nid:
            ax.scatter(x, z, color="blue", s=20, label="Sample", zorder=4)

    # Highlight a path if provided
    if highlight_path:
        for u, v in zip(highlight_path, highlight_path[1:]):
            x1, z1 = node_positions[u]
            x2, z2 = node_positions[v]
            ax.plot([x1, x2], [z1, z2], color="red", linewidth=2.5, zorder=6)

    # Plot smoothed curve if provided
    if smoothed_curve:
        # Handle nested list-of-curves
        if isinstance(smoothed_curve[0], list):
            curve = smoothed_curve[-1]
        else:
            curve = smoothed_curve
        try:
            sx, sz = zip(*curve)
            ax.plot(sx, sz, color="orange", linewidth=2.0, label="Smoothed Path", zorder=7)
        except Exception:
            pass

    ax.set_aspect("equal", "box")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title("PRM Navigation Graph")

    # clean up duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    plt.show()


def save_data_batch(
    obs_list: Sequence[np.ndarray],
    action_list: Sequence,
    base_dir: Union[str, Path]
) -> None:
    """
    Save a batch of RGB images and all their corresponding actions in one CSV.

    Images are written as PNGs under `base_dir/images/`
    All actions are written together in one CSV under `base_dir/actions/`

    Args:
        obs_list:    A sequence of H×W×3 uint8 RGB arrays (or floats in [0,1]).
        action_list: A sequence of array‐like or scalar actions (one per obs).
        base_dir:    Directory where `images/` and `actions/` folders will be created.

    Returns:
        A tuple of (list_of_image_paths, action_csv_path).
    """
    if len(obs_list) != len(action_list):
        raise ValueError(f"Expected same length for obs_list and action_list, "
                         f"got {len(obs_list)} vs {len(action_list)}")

    base = Path(base_dir)
    img_dir = base / "images"
    act_dir = base / "actions"
    img_dir.mkdir(parents=True, exist_ok=True)
    act_dir.mkdir(parents=True, exist_ok=True)

    # Common timestamp stem
    stem_base = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    image_paths: List[str] = []

    # 1) Save all images
    for idx, obs in enumerate(obs_list, start=1):
        # Convert floats → uint8
        if np.issubdtype(obs.dtype, np.floating):
            obs = (obs * 255).clip(0, 255).astype(np.uint8)

        # Sanity check
        if obs.ndim != 3 or obs.shape[2] != 3:
            raise ValueError(f"Expected H×W×3 array for obs #{idx}, got shape {obs.shape}")

        stem = f"{stem_base}_{idx:03d}"
        img_path = img_dir / f"{stem}.png"
        Image.fromarray(obs).save(img_path)
        image_paths.append(str(img_path))

    # 2) Stack all actions into one 2D array
    #    Each action is flattened to one row
    flat_actions = []
    for idx, action in enumerate(action_list, start=1):
        arr = np.atleast_1d(action).reshape(-1)
        flat_actions.append(arr)
    # Ensure all rows have same length
    lengths = [a.size for a in flat_actions]
    if len(set(lengths)) != 1:
        raise ValueError(f"All actions must have the same size; got sizes {set(lengths)}")
    action_matrix = np.vstack(flat_actions)

    # 3) Save the single CSV
    action_csv = act_dir / f"{stem_base}_actions.csv"
    # one row per action, comma-delimited
    np.savetxt(action_csv, action_matrix, delimiter=",", fmt="%s")

    return None


