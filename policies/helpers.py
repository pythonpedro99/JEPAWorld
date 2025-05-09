
import numpy as np
from dataclasses import dataclass
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import networkx as nx
import numpy as np
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


@dataclass
class GraphData:
    """
    Aggregates rooms, agent, and obstacle data for graph building.
    """

    rooms: List[Room]
    agent: Agent
    obstacles: List[Obstacle]


def find_path(
    graph: nx.Graph,
    nodes: List[str],
    start: str,
    goal: str
) -> Optional[List[str]]:
        
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