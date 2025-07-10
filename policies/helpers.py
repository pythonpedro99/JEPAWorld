import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Rectangle
import networkx as nx
import numpy as np
from PIL import Image
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box


@dataclass
class Room:
    """
    Dataclass representing a room.

    Attributes
    ----------
    id : int
        Unique identifier for the room.
    vertices : List[Tuple[float, float]]
        List of (x, z) vertex coordinates defining the room polygon.
    """
    id: int
    vertices: List[Tuple[float, float]]


@dataclass
class Agent:
    """
    Dataclass representing an agent in the environment.

    Attributes
    ----------
    pos : Tuple[float, float]
        Current (x, z) position of the agent.
    yaw : float
        Current orientation of the agent in radians.
    radius : float
        Radius of the agent for collision checking.
    """
    pos: Tuple[float, float]
    yaw: float
    radius: float


@dataclass
class Obstacle:
    """
    Dataclass representing an obstacle in the environment.

    Attributes
    ----------
    type : str
        Type of the obstacle (e.g., 'Box', 'Ball', 'Key').
    pos : Tuple[float, float]
        (x, z) position of the obstacle.
    radius : float
        Radius of the obstacle.
    node_name : str
        Name of the PRM graph node corresponding to the obstacle.
    yaw : float
        Orientation of the obstacle in radians.
    size : Tuple[float, float]
        Width and depth of the obstacle.
    """
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
    room: Room
    obstacles: List[Obstacle]


def get_graph_data(env) -> GraphData:
    """
    Convert MiniWorld environment entities into lightweight dataclasses.

    Parameters
    ----------
    env : gym.Env
        The MiniWorld environment.

    Returns
    -------
    GraphData
        Object containing room and obstacle data.
    """
    unwrapped = env.unwrapped
    rm = unwrapped.rooms[0]
    room_polygon = [(p[0], p[2]) for p in rm.outline]
    room = Room(id=0, vertices=room_polygon)

    obstacles: List[Obstacle] = []
    for idx, ent in enumerate(unwrapped.entities):
        yaw = getattr(ent, "dir", 0.0)
        if hasattr(ent, "size"):
            sx, _, sz = ent.size
            width, depth = sx, sz
        elif hasattr(ent, "mesh") and hasattr(ent.mesh, "min_coords"):
            width = (ent.mesh.max_coords[0] - ent.mesh.min_coords[0]) * ent.scale
            depth = (ent.mesh.max_coords[2] - ent.mesh.min_coords[2]) * ent.scale
        else:
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
    agent_radius: float = 0.3,
) -> Tuple[nx.Graph, Dict[str, Tuple[float, float]]]:
    """
    Build a probabilistic-roadmap graph for a single room environment.

    Parameters
    ----------
    graph_data : GraphData
        Aggregated room and obstacle data.
    sample_density : float, optional
        Density of random samples, by default 0.3.
    k_neighbors : int, optional
        Number of nearest neighbors for connections, by default 15.
    jitter_ratio : float, optional
        Fraction for jittering sample positions, by default 0.3.
    min_samples : int, optional
        Minimum number of samples, by default 5.
    min_dist : float, optional
        Minimum distance from walls for sampling, by default 0.2.
    agent_radius : float, optional
        Radius of the agent for collision checking, by default 0.3.

    Returns
    -------
    Tuple[nx.Graph, Dict[str, Tuple[float, float]]]
        PRM graph and dictionary mapping node names to positions.
    """
    graph = nx.Graph()
    room_poly = Polygon(graph_data.room.vertices)

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

    node_pos: Dict[str, Tuple[float, float]] = {}
    for obs in graph_data.obstacles:
        node_pos[obs.node_name] = obs.pos
        graph.add_node(obs.node_name)

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

    nodes = list(node_pos)
    for n in nodes:
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
            skip = False
            for obs_name, poly in inflated.items():
                if obs_name in (n, m):
                    continue
                if poly.intersects(seg):
                    skip = True
                    break
            if skip:
                continue
            graph.add_edge(n, m, weight=seg.length)

    return graph, node_pos


def plot_room_with_obstacles_and_path(
    room: Room,
    obstacles: List[Obstacle],
    node_positions: Dict[str, Tuple[float, float]],
    graph: nx.Graph,
    path: Optional[List[str]] = None,
    agent_radius: float = 0.25,
    title: str = "Room with Obstacles and Path",
) -> None:
    """
    Visualize the room, obstacles, PRM graph, and an optional path.

    Parameters
    ----------
    room : Room
        Dataclass instance containing .vertices for the room polygon.
    obstacles : List[Obstacle]
        List of obstacles in the environment.
    node_positions : Dict[str, Tuple[float, float]]
        Mapping of node names to their coordinates.
    graph : nx.Graph
        PRM graph.
    path : Optional[List[str]], optional
        Sequence of node names defining a path, by default None.
    agent_radius : float, optional
        Radius for inflating obstacles, by default 0.25.
    title : str, optional
        Title for the plot, by default "Room with Obstacles and Path".
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.set_title(title)

    # build the polygon from the Room dataclass
    room_poly = Polygon(room.vertices)
    rx, ry = room_poly.exterior.xy
    ax.plot(rx, ry, color="black", linewidth=2, label="Room")

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

    for u, v in graph.edges:
        x0, y0 = node_positions[u]
        x1, y1 = node_positions[v]
        ax.plot([x0, x1], [y0, y1], color="lightgray", linewidth=1, zorder=0)

    for name, (px, py) in node_positions.items():
        if name.startswith("s"):
            ax.plot(px, py, "bo", markersize=3)
        else:
            ax.plot(px, py, "ko", markersize=4)
        ax.text(px, py + 0.04, name, fontsize=6, ha="center")

    if path and len(path) >= 2:
        coords = [node_positions[n] for n in path]
        px, py = zip(*coords)
        ax.plot(px, py, color="green", linewidth=2.5, label="Path", zorder=3)
        ax.plot(px[0], py[0], "go", markersize=8, label="Start", zorder=4)
        ax.plot(px[-1], py[-1], "ro", markersize=8, label="Goal", zorder=4)

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()
