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
from Miniworld.miniworld.manual_control import ManualControl

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


class ExpertPolicy:
    """
    Expert policy for path planning in MiniWorld environments.
    """

    def __init__(self, env: gym.Env, mission: list[str]):
        self.env = env
        self.env.reset()
        self.mission = mission

        self.graph_data = self.get_graph_data()
        self.graph, self.node_positions, self.obstacle_nodes = self.build_prm_graph(
            sample_density=0.5, k_neighbors=15, jitter_ratio=0.3, min_samples=5
        )
        self.path = self.find_path(self.mission[0],self.mission[1])
        self.waypoints = [self.node_positions[n] for n in self.path]
        self.forward_step = DEFAULT_FORWARD_STEP   # 0.15
        self.turn_step    = DEFAULT_TURN_STEP      # 15°
        self.agent_yaw    = env.unwrapped.agent.dir
        self.actions =[]
        self.waypoints_smoothed = []

    def get_graph_data(self) -> GraphData:
        """
        Extracts topology and entity information from a MiniWorld environment.

        Args:
            env: A MiniWorld environment instance.

        Returns:
            GraphData containing rooms, agent, and obstacles.
        """
        unwrapped = self.env.unwrapped
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
                    end = (a[0] + ux * e, a[1] + uz * e)
                    mid = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
                    portals.append(Portal(edge_idx, start, end, mid))
            rooms.append(Room(rid, verts, portals))

        agent_obj = unwrapped.agent
        agent = Agent(
            pos=(agent_obj.pos[0], agent_obj.pos[2]),
            yaw=agent_obj.dir,
            radius=agent_obj.radius,
        )

        obstacles = [
            Obstacle(
                type=e.__class__.__name__,
                pos=(e.pos[0], e.pos[2]),
                radius=getattr(e, "radius", 0.0),
                node_name=f"{e.__class__.__name__}_{i}",
            )
            for i, e in enumerate(unwrapped.entities)
        ]

        return GraphData(rooms, agent, obstacles)

    def build_prm_graph(
        self,
        sample_density: float = 0.05,
        k_neighbors: int = 15,
        jitter_ratio: float = 0.3,
        min_samples: int = 5,
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
        room_polygons = {r.id: Polygon(r.vertices) for r in self.graph_data.rooms}
        obstacle_buffers = {
            obs.node_name: Point(*obs.pos).buffer(obs.radius)
            for obs in self.graph_data.obstacles
        }
        node_positions: Dict[str, Tuple[float, float]] = {}

        for room in self.graph_data.rooms:
            for idx, p in enumerate(room.portals):
                node_positions[f"r{room.id}_p{idx}"] = p.midpoint
        node_positions["agent"] = self.graph_data.agent.pos
        for obs in self.graph_data.obstacles:
            node_positions[obs.node_name] = obs.pos

        for room in self.graph_data.rooms:
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

        agent_room = room_of(self.graph_data.agent.pos)

        for rid, poly in room_polygons.items():
            nodes_in_room = [
                name
                for name, pos in node_positions.items()
                if poly.covers(Point(pos))
                and not (name == "agent" and agent_room != rid)
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

        return (
            graph,
            node_positions,
            [obs.node_name for obs in self.graph_data.obstacles],
        )

    def find_path(self, start: str, goal: str) -> Optional[List[str]]:
        """
        Computes the shortest path between two graph nodes using Dijkstra's algorithm.

        Args:
            graph: The roadmap graph.
            start: Starting node identifier.
            goal: Goal node identifier.

        Returns:
            Sequence of node IDs along the shortest path, or None if unreachable.
        """
        if not self.graph.has_node(start) or not self.graph.has_node(goal):
            return None
        try:
            return nx.dijkstra_path(self.graph, start, goal, weight="weight")
        except nx.NetworkXNoPath:
            return None
    
    def waypoints_to_actions(
        self,
        degree: int = 3,
        smoothing: float = 0.0,
        oversample: int = 1000
    ) -> None:
        """
        Robustly fit & re-sample a B-spline through self.waypoints,
        then convert to discrete actions. Returns (actions, smoothed_curve).
        """
        # 1) Dedupe exact repeats
        pts = []
        for p in self.waypoints:
            if not pts or p != pts[-1]:
                pts.append(p)
        if len(pts) < 2:
            return None

        xs, ys = zip(*pts)

        # 2) Try splprep with decreasing k
        curve = None
        for k in range(min(degree, len(xs)-1), 0, -1):
            try:
                tck, _ = splprep([xs, ys], k=k, s=smoothing)
                # oversample + arc-length re-sample
                u_fine = np.linspace(0.0, 1.0, oversample)
                xf, yf = splev(u_fine, tck)
                coords = np.stack([xf, yf], axis=1)

                deltas = np.diff(coords, axis=0)
                seg_lens = np.linalg.norm(deltas, axis=1)
                cumlen = np.concatenate([[0.0], np.cumsum(seg_lens)])
                total_L = cumlen[-1]
                n_steps = int(np.floor(total_L / self.forward_step))
                target_L = np.linspace(0.0, n_steps*self.forward_step, n_steps+1)

                ui = np.interp(target_L, cumlen, u_fine)
                xs_s, ys_s = splev(ui, tck)
                curve = list(zip(xs_s, ys_s))
                break
            except ValueError:
                # try with k-1
                continue

        # 3) If we never got a spline, fall back to linear re-sample
        if curve is None:
            # Just linearly walk and re-sample at forward_step
            # build cumulative length on pts:
            coords = np.array(pts)
            deltas = np.diff(coords, axis=0)
            seg_lens = np.linalg.norm(deltas, axis=1)
            cumlen = np.concatenate([[0.0], np.cumsum(seg_lens)])
            total_L = cumlen[-1]
            n_steps = int(np.floor(total_L / self.forward_step))
            target_L = np.linspace(0.0, n_steps*self.forward_step, n_steps+1)
            # find segment for each target_L
            curve = []
            for L in target_L:
                idx = np.searchsorted(cumlen, L) - 1
                idx = max(0, min(idx, len(pts)-2))
                t = (L - cumlen[idx]) / (cumlen[idx+1] - cumlen[idx] + 1e-8)
                p0, p1 = np.array(pts[idx]), np.array(pts[idx+1])
                curve.append(tuple((1-t)*p0 + t*p1))

        # 4) Discretize into actions
        actions: List[int] = []
        yaw = self.agent_yaw
        pos = np.array(curve[0])
        def norm(a): return ((a+180)%360)-180

        for pt in curve[1:]:
            tgt = np.array(pt)
            delta = tgt - pos
            dist = np.linalg.norm(delta)
            if dist < self.forward_step/2:
                continue

            # turn
            desired = np.degrees(np.arctan2(delta[1], delta[0]))
            dy = norm(desired - yaw)
            n_turns = int(round(abs(dy)/self.turn_step))
            turn_cmd = 0 if dy > 0 else 1
            actions += [turn_cmd]*n_turns
            yaw = norm(yaw + np.sign(dy)*n_turns*self.turn_step)

            # forward
            n_fwd = int(round(dist/self.forward_step))
            actions += [2]*n_fwd

            pos = tgt

        # store & return
        self.actions.append(actions)
        self.waypoints_smoothed.append(curve)


def main():

    mission = ["agent", "Box_0"]
    env = gym.make(
        ENV_NAME,
        render_mode="human",
    )
    expert = ExpertPolicy(env, mission)
    print(expert.waypoints)
    print(expert.waypoints_smoothed)
    expert.waypoints_to_actions(
        degree=2,
        smoothing=0.0,
        oversample=1000
    )
    print(expert.actions)
    plot_prm_graph(
        expert.graph_data,
        expert.graph,
        expert.node_positions,
        expert.obstacle_nodes,
        highlight_path=expert.path,
        smoothed_curve=expert.waypoints_smoothed
    )
    

   

if __name__ == "__main__":
    main()
