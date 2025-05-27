
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import networkx as nx
import numpy as np

@dataclass
class Portal:
    """
    Represents a doorway between rooms.
    """

    edge_index: int
    start: Tuple[float, float]
    end: Tuple[float, float]
    midpoint: Tuple[float, float]


@dataclass
class Room:
    """
    Represents a room with its polygon outline and portals.
    """

    id: int
    vertices: List[Tuple[float, float]]
    portals: List[Portal]


@dataclass
class Agent:
    """
    Represents the navigating agent.
    """

    pos: Tuple[float, float]
    yaw: float
    radius: float


@dataclass
class Obstacle:
    """
    Represents a static obstacle in the environment.
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

    rooms: List[Room]
    agent: Agent
    obstacles: List[Obstacle]


def closest_node(node_positions: Dict[str, Tuple[float, float]], pos: Tuple[float, float]) -> str:
    """Return the node id closest to the given position."""
    px, py = pos
    best = None
    best_dist = float('inf')
    for name, (nx_pos, ny_pos) in node_positions.items():
        dist = (nx_pos - px) ** 2 + (ny_pos - py) ** 2
        if dist < best_dist:
            best_dist = dist
            best = name
    return best


def find_path(
    graph: nx.Graph,
    nodes: List[str],
    start: str | Tuple[float, float],
    goal: str,
    node_positions: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Optional[List[str]]:

    """Return the shortest path from ``start`` to ``goal``.

    ``start`` may be either a node id or an ``(x, y)`` coordinate. In the latter
    case ``node_positions`` must be provided and the nearest node to the
    coordinate will be used as the starting node.
    """

    if isinstance(start, (tuple, list)):
        if node_positions is None:
            raise ValueError("node_positions required when start is coordinates")
        start = closest_node(node_positions, tuple(start))

    if not graph.has_node(start) or not graph.has_node(goal):
        return None

    to_remove = set(nodes) - {start, goal}
    G = graph.copy()
    G.remove_nodes_from(to_remove)
    if not G.has_node(start) or not G.has_node(goal):
        return None

    try:
        return nx.dijkstra_path(G, start, goal, weight="weight")
    except nx.NetworkXNoPath:
        return None


def smooth_path(pts, iterations=3):
        """
        Apply Chaikin's corner-cutting algorithm to generate a smoother polyline.
        Each iteration replaces each segment [P0, P1] with two points:
          Q = 0.75*P0 + 0.25*P1
          R = 0.25*P0 + 0.75*P1
        This rounds off sharp turns gradually.
        """
        path = pts.copy()
        for _ in range(iterations):
            new_path = [path[0]]
            for i in range(len(path)-1):
                p0 = np.array(path[i])
                p1 = np.array(path[i+1])
                q = tuple(0.75*p0 + 0.25*p1)
                r = tuple(0.25*p0 + 0.75*p1)
                new_path.extend([q, r])
            new_path.append(path[-1])
            path = new_path
        return path

def get_lookahead_point(path_pts, pos, L):
    cum = 0
    last = np.array(pos)
    for pt in path_pts:
        nxt = np.array(pt)
        cum += np.linalg.norm(nxt - last)
        if cum >= L:
            return nxt
        last = nxt
    return path_pts[-1]