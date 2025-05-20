
import random
from typing import Dict, List, Optional, Tuple
import gymnasium as gym
import miniworld.envs
import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point, Polygon, box
from shapely import affinity
from scripts.helpers import plot_prm_graph
from policies.expert import ExpertPolicy
from policies.helpers import GraphData,Agent, Obstacle, Room, Portal
from gymnasium.envs.registration import register
register(
    id="JEPAWorld-v0",
    entry_point="miniworld.envs.jepaworld:JEPAWorld",
    max_episode_steps=500,
    kwargs={"seed":6},   # any default kwargs your ctor needs;  random.randint(0, 2**31 - 1)
)

class CollectTrajectories:
    
    def __init__(self):
        
        # Basics 
        self.env = gym.make("JEPAWorld-v0", seed=6)
        self.env.reset()
        self.mission = [("go_to","duckie")]
        self.graph_data = self.get_graph_data()
        self.graph, self.node_positions, self.nodes = self.build_prm_graph(
            sample_density=2.0, k_neighbors=10, jitter_ratio=0.0, min_samples=4, min_dist=0.7
        )
        print(self.nodes)

        expert_policy = ExpertPolicy( self.env, self.graph, self.nodes, self.node_positions, self.graph_data.obstacles,self.mission)
        expert_policy.solve_mission()
  
        plot_prm_graph(
            self.graph_data,
            self.graph,
            self.node_positions,
            self.nodes,
            highlight_path= expert_policy.path,
            smoothed_curve= expert_policy.smoothed_waypoints
        )

    # Functions 

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

        obstacles = []
        for i, e in enumerate(unwrapped.entities):
            yaw = getattr(e, "dir", 0.0)
            if hasattr(e, "size"):
                sx, _, sz = e.size
                width, depth = sx, sz
            elif hasattr(e, "mesh") and hasattr(e.mesh, "min_coords"):
                width = (e.mesh.max_coords[0] - e.mesh.min_coords[0]) * e.scale
                depth = (e.mesh.max_coords[2] - e.mesh.min_coords[2]) * e.scale
            else:
                width = depth = getattr(e, "radius", 0.0) * 2

            obstacles.append(
                Obstacle(
                    type=e.__class__.__name__,
                    pos=(e.pos[0], e.pos[2]),
                    radius=getattr(e, "radius", 0.0),
                    node_name=f"{e.__class__.__name__}_{i}",
                    yaw=yaw,
                    size=(width, depth),
                )
            )

        return GraphData(rooms, agent, obstacles)
    
    def build_prm_graph(
    self,
    sample_density: float = 0.05,
    k_neighbors: int = 15,
    jitter_ratio: float = 0.3,
    min_samples: int = 5,
    min_dist: float = 0.2,          # <— neuer Parameter: Mindestabstand zu Wänden/Polygonrand
) -> Tuple[nx.Graph, Dict[str, Tuple[float, float]], List[str]]:
        """
        Constructs a probabilistic roadmap graph from environment data,
        ensuring that all samples lie at least `min_dist` away from room boundaries.
        """
        graph = nx.Graph()
        room_polygons = {r.id: Polygon(r.vertices) for r in self.graph_data.rooms}

        agent_radius = 0.2
        obstacle_buffers = {}
        for obs in self.graph_data.obstacles:
            w, d = obs.size
            w = max(w, 0.5)
            d = max(d, 0.5)
            rect = box(
                -w / 2 - agent_radius,
                -d / 2 - agent_radius,
                w / 2 + agent_radius,
                d / 2 + agent_radius,
            )
            rect = affinity.rotate(rect, obs.yaw, use_radians=True)
            rect = affinity.translate(
                rect,
                obs.pos[0] + agent_radius,
                obs.pos[1] + agent_radius,
            )
            obstacle_buffers[obs.node_name] = rect
        node_positions: Dict[str, Tuple[float, float]] = {}

        # Portale und Agent/Obstacles wie bisher
        for room in self.graph_data.rooms:
            for idx, p in enumerate(room.portals):
                node_positions[f"r{room.id}_p{idx}"] = p.midpoint
        node_positions["agent"] = self.graph_data.agent.pos
        for obs in self.graph_data.obstacles:
            node_positions[obs.node_name] = obs.pos

        # ——— Hier beginnt das Sampling, angepasst um den Shrink ———
        for room in self.graph_data.rooms:
            poly = room_polygons[room.id]
            # 1) Schrumpfe das Polygon um min_dist
            inner_poly = poly.buffer(-min_dist)
            if inner_poly.is_empty:
                # Fallback: wenn zu stark geschrumpft, nutze das Original
                inner_poly = poly

            # 2) Wie viele Samples?
            n_samples = max(min_samples, int(poly.area * sample_density))
            grid = max(1, int(np.sqrt(n_samples)))
            minx, miny, maxx, maxy = inner_poly.bounds
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

                    # 3) Prüfe gegen das geschrumpfte Polygon
                    if not inner_poly.covers(pt):
                        continue
                    # 4) Prüfe gegen Hindernisse wie bisher
                    if any(buf.contains(pt) for buf in obstacle_buffers.values()):
                        continue

                    node_positions[f"r{room.id}_s{count}"] = (x, y)
                    count += 1
        # ——— Sampling Ende ———

        def room_of(point: Tuple[float, float]) -> Optional[int]:
            for rid, poly in room_polygons.items():
                if poly.covers(Point(point)):
                    return rid
            return None

        agent_room = room_of(self.graph_data.agent.pos)

        # Kanten verbinden wie bisher
        for rid, poly in room_polygons.items():
            nodes_in_room = [
                name
                for name, pos in node_positions.items()
                if poly.covers(Point(pos))
                and not (name == "agent" and agent_room != rid)
            ]
            for n in nodes_in_room:
                pos_n = np.array(node_positions[n])
                dists: List[Tuple[str, float]] = []
                for m in nodes_in_room:
                    if m == n:
                        continue
                    if n.startswith(f"r{rid}_p") and m.startswith(f"r{rid}_p"):
                        continue
                    pos_m = np.array(node_positions[m])
                    dist_sq = float(np.sum((pos_n - pos_m) ** 2))
                    dists.append((m, dist_sq))
                dists.sort(key=lambda t: t[1])
                for m, _ in dists[:k_neighbors]:
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



if __name__ == "__main__":
    CollectTrajectories()