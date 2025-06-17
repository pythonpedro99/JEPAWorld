# ────────────────────────────────────────────────────────────────────────
# Imports
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
import miniworld.envs                               # noqa: F401  (needed for Gym registration)
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.envs.registration import register
from shapely.geometry import Polygon, Point, LineString, box
from shapely import affinity
from policies.rearrange import HumanLikeRearrangePolicy
# ────────────────────────────────────────────────────────────────────────
# Environment registration
# ────────────────────────────────────────────────────────────────────────
register(
    id="JEPAENV-v0",
    entry_point="miniworld.envs.jeparoom:JEPAENV",
    kwargs={"size": 12, "seed": random.randint(0, 2**31 - 1)},
    max_episode_steps=500,
)

# ────────────────────────────────────────────────────────────────────────
# Data classes
# ────────────────────────────────────────────────────────────────────────
@dataclass
class Obstacle:
    type: str
    pos: Tuple[float, float]          # (x, z) in MiniWorld’s ground‑plane coords
    radius: float
    node_name: str
    yaw: float                        # radians
    size: Tuple[float, float]         # (width, depth)


@dataclass
class Room:
    id: int
    vertices: List[Tuple[float, float]]   # 2‑D vertices on (x, z) plane


@dataclass
class GraphData:
    room: Room
    obstacles: List[Obstacle]


# ────────────────────────────────────────────────────────────────────────
# Main helper class
# ────────────────────────────────────────────────────────────────────────
class CollectTrajectories:
    """
    1. Starts the MiniWorld env
    2. Extracts room & obstacle geometry
    3. Builds a PRM graph
    4. (Optionally) plots the scene and a path
    """

    # ───────────────────────────────────────
    # Init
    # ───────────────────────────────────────
    def __init__(self) -> None:
        self.env = gym.make("JEPAENV-v0", seed=random.randint(0, 2**31 - 1),obs_width=224, obs_height=224)
    
        obs, _ = self.env.reset()

        # extract geometry →
        self.graph_data = self._get_graph_data()

        # build PRM →
        self.graph, self.node_positions = self._build_prm_graph_single_room(
            sample_density=0.6,
            k_neighbors=15,
            jitter_ratio=0.1,
            min_samples=20,
            min_dist=0.2,
        )
        output_dir = "/Users/julianquast/Documents/Bachelor Thesis/Code/JEPAWorld/saved_img"
        os.makedirs(output_dir, exist_ok=True)  # This creates the directory if it doesn't exist

        # print(self.node_positions)
        # print(list(self.graph.nodes))
        policy = HumanLikeRearrangePolicy(self.env, self.graph, self.node_positions)
        #done = False
        #obs, _ = self.env.reset()
        i = 0          # initial frame

        # while not done:
        #     i += 1

        #     act = policy.next_action(obs)
        #     obs, _, done, trunc, info = self.env.step(act)
        #     plt.imsave(os.path.join(output_dir, f"obs_{i:03d}.png"), obs)
  
    #     policy = HumanLikeRearrangePolicy(
    #     env= self.env,
    #     prm_graph= self.graph,
    #     node_positions= self.node_positions,
    #     obstacles= self.graph_data.obstacles,
    # )   
        policy.perform_full_scan()  # rotate to survey the room
        policy.go_to(list(self.node_positions.keys())[2])  # go to agent node
        obs, _, term, trunc, _ = self.env.step(4)
        policy.observations.append(obs)  # save the observation after the action
        #print(policy.path)
        # Define the directory
       
        #Save observations
        for i, obs in enumerate(policy.observations):
            plt.imsave(os.path.join(output_dir, f"obs_{i:03d}.png"), obs)
        # # obs, _ = self.env.reset()
        # done = False
        # while not done:
        #     action = policy.next_action(obs)
        #     obs, _, done, trunc, info = self.env.step(action)
        #     self.env.render()                         # comment out if running head‑less


        # # shortest‑path example (remove if you don’t need it)
        # # Pick two random non‑obstacle nodes if available
        # free_nodes = [n for n in self.node_positions if not n.startswith("Obstacle")]
        # if len(free_nodes) >= 2:
        #     src, dst = free_nodes[0], free_nodes[-1]
        #     try:
        #         path = nx.shortest_path(self.graph, source=src, target=dst, weight="weight")
        #     except nx.NetworkXNoPath:
        #         path = None
        # else:
        #     path = None

        # visualisation (optional)
        self._plot_room_with_obstacles_and_path(
            room_polygon=Polygon(self.graph_data.room.vertices),
            obstacles=self.graph_data.obstacles,
            node_positions=self.node_positions,
            graph=self.graph,
            path=policy.path,
        )

    # ───────────────────────────────────────
    # Geometry extraction
    # ───────────────────────────────────────
    def _get_graph_data(self) -> GraphData:
        """
        Converts MiniWorld entities into lightweight dataclasses for later use.
        """
        unwrapped = self.env.unwrapped

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
            else:                                   # sphere / cylinder
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

    # ───────────────────────────────────────
    # PRM builder
    # ───────────────────────────────────────
    def _build_prm_graph_single_room(
    self,
    sample_density: float = 0.3,
    k_neighbors: int = 15,
    jitter_ratio: float = 0.3,
    min_samples: int = 5,
    min_dist: float = 0.2,
) -> Tuple[nx.Graph, Dict[str, Tuple[float, float]]]:
        """
        Builds a probabilistic‑roadmap graph (nodes = obstacles + random samples).
        """
        graph = nx.Graph()
        room_poly = Polygon(self.graph_data.room.vertices)

        # Inflate each obstacle by the agent radius so the PRM stays collision‑free
        agent_radius = 0.2
        inflated: Dict[str, Polygon] = {}
        for obs in self.graph_data.obstacles:
            w, d = max(obs.size[0], 0.5), max(obs.size[1], 0.5)
            rect = box(-w / 2 - agent_radius, -d / 2 - agent_radius,
                    w / 2 + agent_radius,  d / 2 + agent_radius)
            rect = affinity.rotate(rect, obs.yaw, use_radians=True)
            rect = affinity.translate(rect, obs.pos[0], obs.pos[1])
            inflated[obs.node_name] = rect

        # Node positions dict
        node_pos: Dict[str, Tuple[float, float]] = {}

        # Add obstacle centres as graph nodes
        for obs in self.graph_data.obstacles:
            node_pos[obs.node_name] = obs.pos
            graph.add_node(obs.node_name)

        # Random interior samples
        inner = room_poly.buffer(-min_dist) or room_poly  # defensive fallback
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
                for m in nodes if m != n
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



    # ───────────────────────────────────────
    # Visualisation helper
    # ───────────────────────────────────────
    def _plot_room_with_obstacles_and_path(
    self,
    room_polygon: Polygon,
    obstacles: List[Obstacle],
    node_positions: Dict[str, Tuple[float, float]],
    graph: nx.Graph,                 # ⬅️  NEW: pass the graph in
    path: List[str] | None = None,
    agent_radius: float = 0.25,
    title: str = "Room with Obstacles and Path",
) -> None:
        """Visualise the room, obstacles, full PRM graph and an optional path."""
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect("equal")
        ax.set_title(title)

        # --- Room boundary -----------------------------------------------------
        rx, ry = room_polygon.exterior.xy
        ax.plot(rx, ry, color="black", linewidth=2, label="Room")

        # --- Obstacles (inflated by agent radius) ------------------------------
        for obs in obstacles:
            w, d = max(obs.size[0], 0.5), max(obs.size[1], 0.5)
            rect = box(-w / 2 - agent_radius, -d / 2 - agent_radius,
                    w / 2 + agent_radius,  d / 2 + agent_radius)
            rect = affinity.rotate(rect, obs.yaw, use_radians=True)
            rect = affinity.translate(rect, obs.pos[0], obs.pos[1])
            ox, oy = rect.exterior.xy
            ax.fill(ox, oy, color="red", alpha=0.5)
            ax.text(obs.pos[0], obs.pos[1], obs.node_name,
                    ha="center", va="center", fontsize=7, color="white")

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



# ────────────────────────────────────────────────────────────────────────
# Run once for testing
# ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    CollectTrajectories()


    














#         self.graph_data = self.get_graph_data()
#         self.graph, self.node_positions, self.nodes = self.build_prm_graph(
#             sample_density=3.0, k_neighbors=9, jitter_ratio=0.0, min_samples=0, min_dist=0.1
#         )
#         agent_node = next(
#          (n for n in self.nodes if n.startswith("agent")),
#             None
#             )
#         if agent_node is None:
#             raise RuntimeError("No Agent node found in self.nodes")
#         self.agent = agent_node
#         self.mission = self._select_random_movable()
#         print("Mission:", self.mission)
#         try:
#           expert_policy = ExpertPolicy( self.env, self.graph, self.nodes, self.node_positions, self.graph_data.obstacles,self.mission,self.agent)
#           expert_policy.solve_mission()
  
#           plot_prm_graph(
#             self.graph_data,
#             self.graph,
#             self.node_positions,
#             self.nodes,
#             highlight_path= expert_policy.path,
#             smoothed_curve= expert_policy.smoothed_waypoints
#            )
#         except Exception as e:
#             print(f"An error occurred while solving the mission: {e}")
#             plot_prm_graph(
#             self.graph_data,
#             self.graph,
#             self.node_positions,
#             self.nodes
#            )
#             return
        
#         print("Mission completed successfully!")

#     # Functions

#     
    
#     def build_prm_graph(
#         self,
#         sample_density: float = 0.05,
#         k_neighbors: int = 15,
#         jitter_ratio: float = 0.3,
#         min_samples: int = 5,
#         min_dist: float = 0.2,
#     ) -> Tuple[nx.Graph, Dict[str, Tuple[float, float]], List[str]]:
#         """
#         Constructs a probabilistic roadmap graph from environment data,
#         ensuring that all samples lie at least `min_dist` away from room boundaries.
#         """
#         graph = nx.Graph()
#         # Precompute room polygons
#         room_polygons = {r.id: Polygon(r.vertices) for r in self.graph_data.rooms}

#         # Build obstacle buffers (inflated by agent radius)
#         agent_radius = 0.25
#         obstacle_buffers: Dict[str, Polygon] = {}
#         for obs in self.graph_data.obstacles:
#             w, d = obs.size
#             w = max(w, 0.5)
#             d = max(d, 0.5)
#             rect = box(
#                 -w / 2 - agent_radius,
#                 -d / 2 - agent_radius,
#                 w / 2 + agent_radius,
#                 d / 2 + agent_radius,
#             )
#             rect = affinity.rotate(rect, obs.yaw, use_radians=True)
#             rect = affinity.translate(
#                 rect,
#                 obs.pos[0] + agent_radius,
#                 obs.pos[1] + agent_radius,
#             )
#             obstacle_buffers[obs.node_name] = rect

#         # Node position map
#         node_positions: Dict[str, Tuple[float, float]] = {}

#         portal_offset = 0.3
#         portal_nodes: Dict[Tuple[float, float], str] = {}
#         portal_idx = 0
#         for room in self.graph_data.rooms:
#             poly = room_polygons[room.id]
#             for p in room.portals:
#                 mid_pt = (float(p.midpoint[0]), float(p.midpoint[1]))
#                 key = (round(mid_pt[0], 3), round(mid_pt[1], 3))
#                 if key not in portal_nodes:
#                     # compute perpendicular to portal edge
#                     start = np.array(p.start)
#                     end = np.array(p.end)
#                     dir_vec = end - start
#                     norm = np.linalg.norm(dir_vec)
#                     unit = dir_vec / norm if norm > 0 else np.array([1.0, 0.0])
#                     perp = np.array([-unit[1], unit[0]])

#                     # offset candidates
#                     cand1 = np.array(mid_pt) + perp * portal_offset
#                     cand2 = np.array(mid_pt) - perp * portal_offset
#                     # choose the point inside the room polygon
#                     pos = cand1 if poly.contains(Point(*cand1)) else cand2

#                     name = f"_portal_{portal_idx}"
#                     portal_nodes[key] = name
#                     node_positions[name] = (float(pos[0]), float(pos[1]))
#                     portal_idx += 1

#         # 2) Add agent and obstacle nodes
#         #node_positions["agent"] = tuple(self.graph_data.agent.pos)
#         for obs in self.graph_data.obstacles:
#             node_positions[obs.node_name] = tuple(obs.pos)

#         # 3) Sample interior points in each room, shrunk by min_dist
#         for room in self.graph_data.rooms:
#             poly = room_polygons[room.id]
#             inner_poly = poly.buffer(-min_dist)
#             if inner_poly.is_empty:
#                 inner_poly = poly

#             n_samples = max(min_samples, int(poly.area * sample_density))
#             grid = max(1, int(np.sqrt(n_samples)))
#             minx, miny, maxx, maxy = inner_poly.bounds
#             dx, dy = (maxx - minx) / grid, (maxy - miny) / grid
#             count = 0

#             for i in range(grid):
#                 for j in range(grid):
#                     if count >= n_samples:
#                         break
#                     cx = minx + (i + 0.5) * dx
#                     cy = miny + (j + 0.5) * dy
#                     x = cx + (random.random() - 0.5) * dx * jitter_ratio
#                     y = cy + (random.random() - 0.5) * dy * jitter_ratio
#                     pt = Point(x, y)

#                     if not inner_poly.covers(pt):
#                         continue
#                     if any(buf.contains(pt) for buf in obstacle_buffers.values()):
#                         continue

#                     node_positions[f"r{room.id}_s{count}"] = (x, y)
#                     count += 1

#         # 4) Helper to find which room a point lies in
#         def room_of(point: Tuple[float, float]) -> Optional[int]:
#             for rid, poly in room_polygons.items():
#                 if poly.covers(Point(point)):
#                     return rid
#             return None

#         agent_room = room_of(self.graph_data.agent.pos)

#         # 5) Connect k-nearest neighbors within each room
#         for rid, poly in room_polygons.items():
#             nodes_in_room = [
#                 name
#                 for name, pos in node_positions.items()
#                 if poly.covers(Point(pos))
#                 and not (name == "agent" and agent_room != rid)
#             ]
#             for n in nodes_in_room:
#                 pos_n = np.array(node_positions[n])
#                 dists: List[Tuple[str, float]] = []
#                 for m in nodes_in_room:
#                     if m == n:
#                         continue
#                     pos_m = np.array(node_positions[m])
#                     dist_sq = float(np.sum((pos_n - pos_m) ** 2))
#                     dists.append((m, dist_sq))
#                 dists.sort(key=lambda t: t[1])
#                 for m, _ in dists[:k_neighbors]:
#                     seg = LineString([node_positions[n], node_positions[m]])
#                     if not poly.covers(seg):
#                         continue
#                     if any(
#                         other_name not in {n, m} and buf.intersects(seg)
#                         for other_name, buf in obstacle_buffers.items()
#                     ):
#                         continue
#                     graph.add_edge(n, m, weight=seg.length)
#         # 7) Additionally connect portal nodes sharing the same X or Y coordinate
#         portal_list = [n for n in node_positions if n.startswith("_portal_")]
#         # group by rounded X
#         x_groups: Dict[float, List[str]] = {}
#         for n in portal_list:
#             x = round(node_positions[n][0], 3)
#             x_groups.setdefault(x, []).append(n)
#         for group in x_groups.values():
#             if len(group) > 1:
#                 for i in range(len(group)):
#                     for j in range(i+1, len(group)):
#                         n1, n2 = group[i], group[j]
#                         p1 = np.array(node_positions[n1])
#                         p2 = np.array(node_positions[n2])
#                         seg = LineString([p1, p2])
#                         if any(buf.intersects(seg) for buf in obstacle_buffers.values()):
#                             continue
#                         dist = float(np.linalg.norm(p1 - p2))
#                         graph.add_edge(n1, n2, weight=dist)
#         # group by rounded Y
#         y_groups: Dict[float, List[str]] = {}
#         for n in portal_list:
#             y = round(node_positions[n][1], 3)
#             y_groups.setdefault(y, []).append(n)
#         for group in y_groups.values():
#             if len(group) > 1:
#                 for i in range(len(group)):
#                     for j in range(i+1, len(group)):
#                         n1, n2 = group[i], group[j]
#                         p1 = np.array(node_positions[n1])
#                         p2 = np.array(node_positions[n2])
#                         seg = LineString([p1, p2])
#                         if any(buf.intersects(seg) for buf in obstacle_buffers.values()):
#                             continue
#                         dist = float(np.linalg.norm(p1 - p2))
#                         graph.add_edge(n1, n2, weight=dist)
        

#         # 6) Return graph, all node positions, and obstacle names
#         obstacle_nodes = [obs.node_name for obs in self.graph_data.obstacles]
#         return (
#             graph,
#             node_positions,
#             obstacle_nodes,
#         )




if __name__ == "__main__":
    CollectTrajectories()