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
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString
from matplotlib.patches import Polygon as MplPolygon, Rectangle
from dataclasses import dataclass
import numpy as np
from PIL import Image
from typing import Optional, Sequence, Union
from pathlib import Path
import random

@dataclass
class Room:
    id: int
    vertices: List[Tuple[float, float]]


@dataclass
class Agent:
    pos: Tuple[float, float]
    yaw: float
    radius: float


@dataclass
class Obstacle:
    type: str
    pos: Tuple[float, float]
    radius: float
    node_name: str
    yaw: float
    size: Tuple[float, float]

@dataclass
class GraphData:
    """
    Aggregates rooms, agent, and obstacle data for graph building.
    """

    room: List[Room]
    obstacles: List[Obstacle]


def get_graph_data(env) -> GraphData:
    """
    Converts MiniWorld environment entities into lightweight dataclasses for later use.
    """
    unwrapped = env.unwrapped

    # MiniWorld has exactly one room in JEPAENV
    rm = unwrapped.rooms[0]
    room_polygon = [(p[0], p[2]) for p in rm.outline]          # (x, z)
    room = Room(id=0, vertices=room_polygon)

    obstacles: List[Obstacle] = []
    for idx, ent in enumerate(unwrapped.entities):
        # orientation
        yaw = getattr(ent, "dir", 0.0)

        # size handling
        if hasattr(ent, "size"):
            sx, _, sz = ent.size
            width, depth = sx, sz
        elif hasattr(ent, "mesh") and hasattr(ent.mesh, "min_coords"):
            width = (ent.mesh.max_coords[0] - ent.mesh.min_coords[0]) * ent.scale
            depth = (ent.mesh.max_coords[2] - ent.mesh.min_coords[2]) * ent.scale
        else:  # sphere / cylinder
            width = depth = getattr(ent, "radius", 0.0) * 2

        obstacles.append(
            Obstacle(
                type=ent.__class__.__name__,
                pos=(ent.pos[0], ent.pos[2]),
                radius=getattr(ent, "radius", 0.0),
                node_name=f"{ent.__class__.__name__}_{idx}",
                yaw=yaw,
                size=(width, depth),
            )
        )

    return GraphData(room=room, obstacles=obstacles)


def build_prm_graph_single_room(
    graph_data: GraphData,
    sample_density: float = 0.3,
    k_neighbors: int = 15,
    jitter_ratio: float = 0.3,
    min_samples: int = 5,
    min_dist: float = 0.2,
    agent_radius: float = 0.2,
) -> Tuple[nx.Graph, Dict[str, Tuple[float, float]]]:
    """
    Builds a probabilistic‑roadmap graph (nodes = obstacles + random samples).
    """
    graph = nx.Graph()
    room_poly = Polygon(graph_data.room.vertices)

    # Inflate each obstacle by the agent radius so the PRM stays collision‑free
    inflated: Dict[str, Polygon] = {}
    for obs in graph_data.obstacles:
        w, d = max(obs.size[0], 0.5), max(obs.size[1], 0.5)
        rect = box(
            -w / 2 - agent_radius,
            -d / 2 - agent_radius,
            w / 2 + agent_radius,
            d / 2 + agent_radius,
        )
        rect = affinity.rotate(rect, obs.yaw, use_radians=True)
        rect = affinity.translate(rect, obs.pos[0], obs.pos[1])
        inflated[obs.node_name] = rect

    # Node positions dict
    node_pos: Dict[str, Tuple[float, float]] = {}

    # Add obstacle centres as graph nodes
    for obs in graph_data.obstacles:
        node_pos[obs.node_name] = obs.pos
        graph.add_node(obs.node_name)

    # Random interior samples
    inner = room_poly.buffer(-min_dist) or room_poly
    n_samples = max(min_samples, int(room_poly.area * sample_density))
    grid = max(1, int(np.sqrt(n_samples)))
    minx, miny, maxx, maxy = inner.bounds
    dx, dy = (maxx - minx) / grid, (maxy - miny) / grid
    counter = 0

    for i in range(grid):
        for j in range(grid):
            if counter >= n_samples:
                break
            cx, cy = minx + (i + 0.5) * dx, miny + (j + 0.5) * dy
            x = cx + (random.random() - 0.5) * dx * jitter_ratio
            y = cy + (random.random() - 0.5) * dy * jitter_ratio
            pt = Point(x, y)
            if not inner.covers(pt):
                continue
            if any(poly.contains(pt) for poly in inflated.values()):
                continue
            node_name = f"s{counter}"
            node_pos[node_name] = (x, y)
            graph.add_node(node_name)
            counter += 1

    # k‑nearest connections among all nodes
    nodes = list(node_pos)
    for i, n in enumerate(nodes):
        p_n = np.asarray(node_pos[n])
        dists = [
            (m, np.sum((p_n - np.asarray(node_pos[m])) ** 2))
            for m in nodes
            if m != n
        ]
        dists.sort(key=lambda t: t[1])

        for m, _ in dists[:k_neighbors]:
            seg = LineString([node_pos[n], node_pos[m]])
            if not room_poly.covers(seg):
                continue

            # ✅ Allow edge if it only intersects its own node's inflated obstacle
            skip = False
            for obs_name, poly in inflated.items():
                if obs_name in (n, m):
                    continue  # allow entry/exit to own center
                if poly.intersects(seg):
                    skip = True
                    break
            if skip:
                continue

            graph.add_edge(n, m, weight=seg.length)

    return graph, node_pos


def plot_room_with_obstacles_and_path(
    room_polygon: Polygon,
    obstacles: List[Obstacle],
    node_positions: Dict[str, Tuple[float, float]],
    graph: nx.Graph,
    path: Optional[List[str]] = None,
    agent_radius: float = 0.25,
    title: str = "Room with Obstacles and Path",
) -> None:
    """
    Visualise the room, obstacles, full PRM graph and an optional path.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.set_title(title)

    # --- Room boundary -----------------------------------------------------
    rx, ry = room_polygon.exterior.xy
    ax.plot(rx, ry, color="black", linewidth=2, label="Room")

    # --- Obstacles (inflated by agent radius) ------------------------------
    for obs in obstacles:
        w, d = max(obs.size[0], 0.5), max(obs.size[1], 0.5)
        rect = box(
            -w / 2 - agent_radius,
            -d / 2 - agent_radius,
            w / 2 + agent_radius,
            d / 2 + agent_radius,
        )
        rect = affinity.rotate(rect, obs.yaw, use_radians=True)
        rect = affinity.translate(rect, obs.pos[0], obs.pos[1])
        ox, oy = rect.exterior.xy
        ax.fill(ox, oy, color="red", alpha=0.5)
        ax.text(
            obs.pos[0], obs.pos[1], obs.node_name,
            ha="center", va="center", fontsize=7, color="white"
        )

    # --- All PRM edges -----------------------------------------------------
    for u, v in graph.edges:
        x0, y0 = node_positions[u]
        x1, y1 = node_positions[v]
        ax.plot([x0, x1], [y0, y1], color="lightgray", linewidth=1, zorder=0)

    # --- Nodes -------------------------------------------------------------
    for name, (px, py) in node_positions.items():
        if name.startswith("s"):                      # sample node
            ax.plot(px, py, "bo", markersize=3)
        else:                                         # obstacle centre
            ax.plot(px, py, "ko", markersize=4)
        ax.text(px, py + 0.04, name, fontsize=6, ha="center")

    # --- Path overlay ------------------------------------------------------
    if path and len(path) >= 2:
        coords = [node_positions[n] for n in path]
        px, py = zip(*coords)
        ax.plot(px, py, color="green", linewidth=2.5, label="Path", zorder=3)
        ax.plot(px[0], py[0], "go", markersize=8, label="Start", zorder=4)
        ax.plot(px[-1], py[-1], "ro", markersize=8, label="Goal",  zorder=4)

    # --- Cosmetics ---------------------------------------------------------
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def save_data_batch(
    obs_list: Sequence[np.ndarray],
    action_list: Sequence,
    base_dir: Union[str, Path],
    *,
    csv_name: Optional[str] = None,
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
    # if len(obs_list) != len(action_list):
    #     raise ValueError(f"Expected same length for obs_list and action_list, "
    #                      f"got {len(obs_list)} vs {len(action_list)}")

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
     # 3) Save the actions CSV. When csv_name is provided, append to that file
    if csv_name is None:
        action_csv = act_dir / f"{stem_base}_actions.csv"
        mode = "wb"
    else:
        action_csv = act_dir / csv_name
        mode = "ab" if action_csv.exists() else "wb"

    with open(action_csv, mode) as fh:
        np.savetxt(fh, action_matrix, delimiter=",", fmt="%s")

    return None


