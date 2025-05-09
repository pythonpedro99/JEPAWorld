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
from utils.plot_prm_graph import plot_prm_graph
from utils.run_actions_in_env import save_observation, get_agent_info


MINIWORLD_PATH = os.path.expanduser("../MiniWorld")
if MINIWORLD_PATH not in sys.path:
    sys.path.append(MINIWORLD_PATH)

ENV_NAME = "MiniWorld-ThreeRooms-v0"
DEFAULT_FORWARD_STEP = 0.15
DEFAULT_TURN_STEP = np.deg2rad(15)


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
            sample_density=1.0, k_neighbors=10, jitter_ratio=0.0, min_samples=5
        )
        self.path = self.find_path(self.mission[0], self.mission[1])
        self.waypoints = [self.node_positions[n] for n in self.path]
        self.forward_step = DEFAULT_FORWARD_STEP  # 0.15
        self.turn_step = DEFAULT_TURN_STEP  # 15°
        self.agent_yaw = env.unwrapped.agent.dir
        self.actions = []
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
        Compute the shortest path from start to goal, but treat every node
        in self.obstacle_nodes as off‐limits *except* the start and goal.
        """
        # 1) Ensure start/goal exist
        if not self.graph.has_node(start) or not self.graph.has_node(goal):
            return None

        # 2) Build the set of obstacles *to remove*, excluding start/goal
        to_remove = set(self.obstacle_nodes) - {start, goal}

        # 3) Copy & prune
        G = self.graph.copy()
        G.remove_nodes_from(to_remove)
        print(f"Removed {to_remove} nodes from the graph")

        # 4) Double‐check start/goal survived
        if not G.has_node(start) or not G.has_node(goal):
            return None

        # 5) Run Dijkstra on the pruned graph
        try:
            return nx.dijkstra_path(G, start, goal, weight="weight")
        except nx.NetworkXNoPath:
            return None


    

    def go_to(self, pos_tol=0.2):
        """
        Sequentially drive to each waypoint in self.waypoints (or a provided list),
        by: (1) turning in place until facing the waypoint within ±turn_tol,
            (2) stepping forward one fixed step,
            (3) repeating until within pos_tol meters, then advancing to the next.
        """
        import numpy as np

        unwrapped = self.env.unwrapped
        pts = self.waypoints
        if not pts:
            return

        forward  = 0.15
        turn_ang = np.deg2rad(15)
        turn_tol = turn_ang / 1.2

        all_actions = []
        print(pts[-1])

        for goal in pts:
            print(f"\n=== New goal: ({goal[0]:.2f}, {goal[1]:.2f}) ===")
            while True:
                # current pose
                x, y    = unwrapped.agent.pos[0], unwrapped.agent.pos[2]
                yaw_rad = unwrapped.agent.dir
                # normalize yaw into [0,360)
                yaw_deg = (np.degrees(yaw_rad) + 360) % 360

                # vector to goal
                dx, dy = goal[0] - x, goal[1] - y
                dist   = np.hypot(dx, dy)
                if dist < pos_tol:
                    print(f"[DEBUG] Reached goal: pos=({x:.2f},{y:.2f}) dist={dist:.3f}m\n")
                    break

                # compute desired heading & error
                desired_rad = np.arctan2(-dy, dx)
                desired_deg = (np.degrees(desired_rad) + 360) % 360

                # shortest error in [−180,180)
                err_rad = ((desired_rad - yaw_rad + np.pi) % (2*np.pi)) - np.pi
                err_deg = ((desired_deg - yaw_deg + 180) % 360) - 180

                # debug print
                print(f"[DEBUG] pos=({x:.2f},{y:.2f})  yaw={yaw_deg:.1f}°  "
                    f"target=({goal[0]:.2f},{goal[1]:.2f})  "
                    f"dist={dist:.3f}m\n"
                    f"        desired={desired_deg:.1f}°  err={err_deg:.1f}°  "
                    f"(tol={np.degrees(turn_tol):.1f}°)")

                # turn-in-place
                if abs(err_rad) > turn_tol:
                    cmd = 0 if err_rad > 0 else 1
                    obs, _, term, trunc, _ = self.env.step(cmd)
                    save_observation(obs)
                    all_actions.append(cmd)
                    if term or trunc:
                        print("[DEBUG] Episode ended during turn.")
                        return
                    continue  # re-read pose & re-evaluate

                # move forward
                obs, _, term, trunc, _ = self.env.step(2)
                save_observation(obs)
                all_actions.append(2)
                if term or trunc:
                    print("[DEBUG] Episode ended during forward.")
                    return
                # loop back to re-fetch pose

        self.actions.append(all_actions)
        print(f"[DEBUG] All actions: {all_actions}") 






def main():

    mission = ["Agent_6", "Box_0"]
    env = gym.make(
        ENV_NAME,
        render_mode="human",
    )
    expert = ExpertPolicy(env, mission)
    print(expert.obstacle_nodes)
    #print(expert.waypoints_smoothed)
    expert.go_to()
    #print(expert.actions)

    # run_actions_and_save_images(
    #     env, expert.actions, save_dir="test_images_01", start_idx=0
    # )
    plot_prm_graph(
        expert.graph_data,
        expert.graph,
        expert.node_positions,
        expert.obstacle_nodes,
        highlight_path=expert.path,
        smoothed_curve=expert.waypoints_smoothed,
    )


if __name__ == "__main__":
    main()
