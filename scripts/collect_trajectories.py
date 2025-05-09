
import random
from typing import Dict, List, Optional, Tuple
import gymnasium as gym
import miniworld.envs
import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from helpers import plot_prm_graph
from policies.expert import ExpertPolicy
from policies.helpers import GraphData,Agent, Obstacle, Room, Portal


class CollectTrajectories:
    
    def __init__(self):
        
        # Basics 
        self.env = gym.make("MiniWorld-ThreeRooms-v0")
        self.env.reset()
        self.mission = ["Box_0"]
        self.graph_data = self.get_graph_data()
        self.graph, self.node_positions, self.obstacle_nodes = self.build_prm_graph(
            sample_density=1.0, k_neighbors=10, jitter_ratio=0.0, min_samples=5
        )

        policy = ExpertPolicy( self.env, self.graph, self.nodes, self.node_positions, self.graph_data.obsticals).solve_mission
  
        plot_prm_graph(
            self.graph_data,
            self.graph,
            self.node_positions,
            self.obstacle_nodes,
            highlight_path= policy.path,
            smoothed_curve= policy.smoothed_path
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


