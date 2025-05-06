#!/usr/bin/env python3
"""
Module for PRM-based path planning in MiniWorld environments,
including B-spline smoothing and conversion to discrete actions.
"""
import os
import sys
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import imageio
import gymnasium as gym
import miniworld.envs
import networkx as nx
import numpy as np
from scipy.interpolate import splprep, splev
from shapely.geometry import LineString, Point, Polygon
from utils.policy import plot_prm_graph

MINIWORLD_PATH = os.path.expanduser("../MiniWorld")
if MINIWORLD_PATH not in sys.path:
    sys.path.append(MINIWORLD_PATH)

ENV_NAME = "MiniWorld-ThreeRooms-v0"
DEFAULT_FORWARD_STEP = 0.15
DEFAULT_TURN_STEP = 15


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

class ExpertPolicy(env,mission):
    """
    Expert policy for path planning in MiniWorld environments.
    """
    def __init__(self):
        self.env = env
        self.env.reset()
        self.mission = mission
        self.goal = goal

        self.graph_data = get_graph_data(env)
        self.graph = build_prm_graph(
            self.graph_data,
            sample_density=0.05,
            k_neighbors=15,
            jitter_ratio=0.3,
            min_samples=5
        )
        self.node_positions = self.graph[1]
        self.obstacle_nodes = self.graph[2]


def get_graph_data(env: gym.Env) -> GraphData:
    """
    Extracts topology and entity information from a MiniWorld environment.

    Args:
        env: A MiniWorld environment instance.

    Returns:
        GraphData containing rooms, agent, and obstacles.
    """
    unwrapped = env.unwrapped
    rooms: List[Room] = []
    for rid, room in enumerate(unwrapped.rooms):
        verts = [(pt[0], pt[2]) for pt in room.outline]
        portals: List[Portal] = []
        for edge_idx, edge_list in enumerate(getattr(room, "portals", [])):
            a, b = verts[edge_idx], verts[(edge_idx + 1) % len(verts)]
            dx, dz = b[0] - a[0], b[1] - a[1]
            length = np.hypot(dx, dz)
            ux, uz = dx / length, dz / length
            for p in edge_list:
                s, e = p["start_pos"], p["end_pos"]
                start = (a[0] + ux * s, a[1] + uz * s)
                end   = (a[0] + ux * e, a[1] + uz * e)
                mid   = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
                portals.append(Portal(edge_idx, start, end, mid))
        rooms.append(Room(rid, verts, portals))

    agent_obj = unwrapped.agent
    agent = Agent(
        pos=(agent_obj.pos[0], agent_obj.pos[2]),
        yaw=agent_obj.dir,
        radius=agent_obj.radius
    )

    obstacles = [
        Obstacle(
            type=e.__class__.__name__,
            pos=(e.pos[0], e.pos[2]),
            radius=getattr(e, "radius", 0.0),
            node_name=f"{e.__class__.__name__}_{i}"
        )
        for i, e in enumerate(unwrapped.entities)
    ]

    return GraphData(rooms, agent, obstacles)


def build_prm_graph(
    data: GraphData,
    sample_density: float = 0.05,
    k_neighbors: int = 15,
    jitter_ratio: float = 0.3,
    min_samples: int = 5
) -> Tuple[nx.Graph, Dict[str, Tuple[float, float]], List[str]]:
    """
    Constructs a probabilistic roadmap graph from environment data.

    Args:
        data: GraphData with rooms, agent, and obstacles.
        sample_density: Number of samples per unit area.
        k_neighbors: Neighbors per sample for connections.
        jitter_ratio: Random offset ratio within grid cells.
        min_samples: Minimum samples per room.

    Returns:
        graph: The constructed NetworkX graph.
        node_positions: Mapping from node IDs to 2D coordinates.
        obstacle_nodes: List of obstacle node names.
    """
    graph = nx.Graph()
    room_polygons = {r.id: Polygon(r.vertices) for r in data.rooms}
    obstacle_buffers = {
        obs.node_name: Point(*obs.pos).buffer(obs.radius)
        for obs in data.obstacles
    }
    node_positions: Dict[str, Tuple[float, float]] = {}

    # for room in data.rooms:
    #     for idx, p in enumerate(room.portals):
    #         node_positions[f"r{room.id}_p{idx}"] = p.midpoint
    node_positions["agent"] = data.agent.pos
    for obs in data.obstacles:
        node_positions[obs.node_name] = obs.pos

    for room in data.rooms:
        poly = room_polygons[room.id]
        n_samples = max(min_samples, int(poly.area * sample_density))
        grid = max(1, int(np.sqrt(n_samples)))
        minx, miny, maxx, maxy = poly.bounds
        dx, dy = (maxx - minx) / grid, (maxy - miny) / grid
        count = 0
        for i in range(grid):
            for j in range(grid):
                if count >= n_samples:
                    break
                cx = minx + (i + 0.5) * dx
                cy = miny + (j + 0.5) * dy
                x = cx + (random.random() - 0.5) * dx * jitter_ratio
                y = cy + (random.random() - 0.5) * dy * jitter_ratio
                pt = Point(x, y)
                if not poly.covers(pt):
                    continue
                if any(buf.contains(pt) for buf in obstacle_buffers.values()):
                    continue
                node_positions[f"r{room.id}_s{count}"] = (x, y)
                count += 1

    def room_of(point: Tuple[float, float]) -> Optional[int]:
        """
        Determines which room contains a given point.
        """
        for rid, poly in room_polygons.items():
            if poly.covers(Point(point)):
                return rid
        return None

    agent_room = room_of(data.agent.pos)

    for rid, poly in room_polygons.items():
        nodes_in_room = [
            name for name, pos in node_positions.items()
            if poly.covers(Point(pos)) and not (name == "agent" and agent_room != rid)
        ]
        for n in nodes_in_room:
            x1, y1 = node_positions[n]
            # ——— Start of fix ———
            pos_n = np.array((x1, y1))
            dists: List[Tuple[str, float]] = []
            for m in nodes_in_room:
                if m == n:
                    continue 
                # skip direct portal→portal connections
                if n.startswith(f"r{rid}_p") and m.startswith(f"r{rid}_p"):
                    continue
                pos_m = np.array(node_positions[m])
                dist_sq = float(np.sum((pos_n - pos_m) ** 2))
                dists.append((m, dist_sq))
            dists.sort(key=lambda t: t[1])
            for m, _ in dists[:k_neighbors]:
            # ———  End of fix  ———
                seg = LineString([node_positions[n], node_positions[m]])
                if not poly.covers(seg):
                    continue
                if any(
                    name not in {n, m} and buf.intersects(seg)
                    for name, buf in obstacle_buffers.items()
                ):
                    continue
                graph.add_edge(n, m, weight=seg.length)

    return graph, node_positions, [obs.node_name for obs in data.obstacles]

def find_path(
    graph: nx.Graph, start: str, goal: str
) -> Optional[List[str]]:
    """
    Computes the shortest path between two graph nodes using Dijkstra's algorithm.

    Args:
        graph: The roadmap graph.
        start: Starting node identifier.
        goal: Goal node identifier.

    Returns:
        Sequence of node IDs along the shortest path, or None if unreachable.
    """
    if not graph.has_node(start) or not graph.has_node(goal):
        return None
    try:
        return nx.dijkstra_path(graph, start, goal, weight="weight")
    except nx.NetworkXNoPath:
        return None


def smooth_path(
     waypoints: List[Tuple[float, float]],
     degree: int = 3,
     smoothing: float = 0.0,
     num_points: int = 100
    ) -> List[Tuple[float, float]]:
     """
     Fits a B-spline through waypoints and returns uniformly sampled points.
     """
     print(f"Waypoints: {waypoints}")
     if len(waypoints) < 2:
         return waypoints

     xs, ys = zip(*waypoints)
     # spline order must be at least 1 and strictly less than number of points
     k = min(degree, len(xs) - 1)
     if k < 1:
         return waypoints

     try:
         tck, _ = splprep([xs, ys], k=k, s=smoothing)
     except ValueError:
        # fall back if inputs are invalid for splprep
        return waypoints
     u = np.linspace(0, 1, num_points)
     x_s, y_s = splev(u, tck)
     return list(zip(x_s, y_s))


def curve_to_actions(
    curve: List[Tuple[float, float]],
    start_yaw: float,
    forward_step: float = DEFAULT_FORWARD_STEP,
    turn_step: int = DEFAULT_TURN_STEP,
    dist_tol: Optional[float] = None,
    angle_tol: Optional[float] = None
) -> List[str]:
    """
    Converts a smoothed path into discrete navigation actions.

    Args:
        curve: List of (x, y) points along the smoothed path.
        start_yaw: Agent's initial heading in degrees.
        forward_step: Distance per forward action.
        turn_step: Degrees per turn action.
        dist_tol: Minimum movement threshold.
        angle_tol: Minimum angular threshold.

    Returns:
        List of action strings: "move_forward", "turn_left", "turn_right".
    """
    dist_tol = dist_tol or forward_step / 2
    angle_tol = angle_tol or turn_step / 2
    actions: List[str] = []
    yaw = start_yaw
    pos = np.array(curve[0])

    def normalize(angle: float) -> float:
        return ((angle + 180) % 360) - 180

    for pt in curve[1:]:
        target = np.array(pt)
        delta = target - pos
        dist = np.linalg.norm(delta)
        if dist < dist_tol:
            continue
        desired = np.degrees(np.arctan2(delta[1], delta[0]))
        d_yaw = normalize(desired - yaw)
        turns = int(round(abs(d_yaw) / turn_step))
        cmd = 0 if d_yaw > 0 else 1
        actions += [cmd] * turns
        yaw = normalize(yaw + np.sign(d_yaw) * turns * turn_step)
        steps = int(round(dist / forward_step))
        actions += [2] * steps
        pos = target

    return actions


def go_to(self,target_node: str) -> Optional[List[str]]:
    """
    Plans and converts a path to a sequence of MiniWorld navigation actions.

    Args:
        env: A MiniWorld environment.
        target_node: Identifier of the destination node.
        sample_density: PRM sampling density.
        k_neighbors: PRM neighbor count.
        jitter_ratio: Sampling jitter ratio.
        min_samples: Minimum PRM samples per room.
        spline_degree: B-spline degree.
        smoothing: B-spline smoothing factor.
        num_points: Number of spline samples.
        forward_step: Step size per forward action.
        turn_step: Degrees per turn action.

    Returns:
        List of actions or None if pathfinding fails.
    """
    forbidden = set(self.obstacle_nodes) - {target_node}
    graph_reduced = self.graph.copy()
    graph_reduced.remove_nodes_from(forbidden)

    if target_node not in graph_reduced or "agent" not in graph_reduced:
        print(f"Missing agent or target '{target_node}' in graph.")
        return None

    try:
        path = nx.dijkstra_path(graph_reduced, "agent", target_node, weight="weight")
    except nx.NetworkXNoPath:
        print("No path found.")
        return None
    waypoints = [self.node_positions[n] for n in path]
    print(waypoints)


def main():
    
    env = gym.make(ENV_NAME, render_mode="rgb_array",)
    env.reset()

    data = get_graph_data(env)
    print(data.obstacles)
    G, node_positions, obstacle_nodes = build_prm_graph(data)

    #path = find_path(G, "agent", "Box_0")
    plot_prm_graph(data, G, node_positions, obstacle_nodes, highlight_path=path)

    # actions = go_to(
    #     env,
    #     target_node="r0_p0",
    #     sample_density=0.05,
    #     k_neighbors=15,
    #     jitter_ratio=0.3,
    #     min_samples=5,
    #     spline_degree=3,
    #     smoothing=0.0,
    #     num_points=100,
    #     forward_step=DEFAULT_FORWARD_STEP,
    #     turn_step=DEFAULT_TURN_STEP
    # )
    # print(f"Actions: {actions}")


if __name__ == "__main__":
    main()
