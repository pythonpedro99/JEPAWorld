
import random
from typing import Dict, List, Optional, Tuple
import gymnasium as gym
import miniworld.envs
import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point, Polygon, box
from shapely import affinity
from policies.expert import ExpertPolicy
from policies.helpers import GraphData,Agent, Obstacle, Room, Portal
from gymnasium.envs.registration import register
register(
    id="JEPAWorld-v0",
    entry_point="miniworld.envs.jepaworld:JEPAWorld",
    max_episode_steps=500,
    kwargs={"seed":0},   # any default kwargs your ctor needs;  random.randint(0, 2**31 - 1)
)

class CollectTrajectories:

    def __init__(self, env_name: str = "JEPAWorld-v0"):

        # Basics
        self.env = gym.make(env_name, seed=random.randint(0, 2**31 - 1))
        self.env.reset()
        self.graph_data = self.get_graph_data()
        self.graph, self.node_positions, self.nodes = self.build_prm_graph(
            sample_density=1.5, k_neighbors=7, jitter_ratio=0.0, min_samples=0, min_dist=0.5
        )
        agent_node = next(
         (n for n in self.nodes if n.startswith("agent")),
            None
            )
        if agent_node is None:
            raise RuntimeError("No Agent node found in self.nodes")
        self.agent = agent_node

    # Functions

    def _select_random_movable(self) -> List[Tuple[str, str]]:
        """Select a mission to go to a random movable object."""
        movables = [
            "duckie",
            "chips",
            "handy",
            "keys",
            "dish",
            "towl",
        ]

        candidates = [
            node for node in self.nodes if any(node.startswith(m) for m in movables)
        ]
        if not candidates:
            raise ValueError("No movable objects found in nodes")

        target_movable = random.choice(candidates)

        # Pick a random node to drop the movable at.  Exclude the movable
        # itself and the agent start position to avoid degenerate missions.
        drop_candidates = [
            n
            for n in self.node_positions.keys()
            if n not in {target_movable, self.agent}
        ]
        if not drop_candidates:
            raise ValueError("No valid drop location found in nodes")
        drop_target = random.choice(drop_candidates)

        return [
            ("go_to", target_movable),
            ("pick_up", ""),
            ("go_to", drop_target),
            ("drop", ""),
        ]

    def random_node(self) -> str:
        """Return a random node from the roadmap."""
        return random.choice(list(self.node_positions.keys()))

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
                    node_name=f"{e.name}_{i}",
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
        min_dist: float = 0.2,
    ) -> Tuple[nx.Graph, Dict[str, Tuple[float, float]], List[str]]:
        """
        Constructs a probabilistic roadmap graph from environment data,
        ensuring that all samples lie at least `min_dist` away from room boundaries.
        """
        graph = nx.Graph()
        # Precompute room polygons
        room_polygons = {r.id: Polygon(r.vertices) for r in self.graph_data.rooms}

        # Build obstacle buffers (inflated by agent radius)
        agent_radius = 0.4
        obstacle_buffers: Dict[str, Polygon] = {}
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

        # Node position map
        node_positions: Dict[str, Tuple[float, float]] = {}

        portal_offset = 0.3
        portal_nodes: Dict[Tuple[float, float], str] = {}
        portal_idx = 0
        for room in self.graph_data.rooms:
            poly = room_polygons[room.id]
            for p in room.portals:
                mid_pt = (float(p.midpoint[0]), float(p.midpoint[1]))
                key = (round(mid_pt[0], 3), round(mid_pt[1], 3))
                if key not in portal_nodes:
                    # compute perpendicular to portal edge
                    start = np.array(p.start)
                    end = np.array(p.end)
                    dir_vec = end - start
                    norm = np.linalg.norm(dir_vec)
                    unit = dir_vec / norm if norm > 0 else np.array([1.0, 0.0])
                    perp = np.array([-unit[1], unit[0]])

                    # offset candidates
                    cand1 = np.array(mid_pt) + perp * portal_offset
                    cand2 = np.array(mid_pt) - perp * portal_offset
                    # choose the point inside the room polygon
                    pos = cand1 if poly.contains(Point(*cand1)) else cand2

                    name = f"_portal_{portal_idx}"
                    portal_nodes[key] = name
                    node_positions[name] = (float(pos[0]), float(pos[1]))
                    portal_idx += 1

        # 2) Add agent and obstacle nodes
        #node_positions["agent"] = tuple(self.graph_data.agent.pos)
        for obs in self.graph_data.obstacles:
            node_positions[obs.node_name] = tuple(obs.pos)

        # 3) Sample interior points in each room, shrunk by min_dist
        for room in self.graph_data.rooms:
            poly = room_polygons[room.id]
            inner_poly = poly.buffer(-min_dist)
            if inner_poly.is_empty:
                inner_poly = poly

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

                    if not inner_poly.covers(pt):
                        continue
                    if any(buf.contains(pt) for buf in obstacle_buffers.values()):
                        continue

                    node_positions[f"r{room.id}_s{count}"] = (x, y)
                    count += 1

        # 4) Helper to find which room a point lies in
        def room_of(point: Tuple[float, float]) -> Optional[int]:
            for rid, poly in room_polygons.items():
                if poly.covers(Point(point)):
                    return rid
            return None

        agent_room = room_of(self.graph_data.agent.pos)

        # 5) Connect k-nearest neighbors within each room
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
                    pos_m = np.array(node_positions[m])
                    dist_sq = float(np.sum((pos_n - pos_m) ** 2))
                    dists.append((m, dist_sq))
                dists.sort(key=lambda t: t[1])
                for m, _ in dists[:k_neighbors]:
                    seg = LineString([node_positions[n], node_positions[m]])
                    if not poly.covers(seg):
                        continue
                    if any(
                        other_name not in {n, m} and buf.intersects(seg)
                        for other_name, buf in obstacle_buffers.items()
                    ):
                        continue
                    graph.add_edge(n, m, weight=seg.length)
        # 7) Additionally connect portal nodes sharing the same X or Y coordinate
        portal_list = [n for n in node_positions if n.startswith("_portal_")]
        # group by rounded X
        x_groups: Dict[float, List[str]] = {}
        for n in portal_list:
            x = round(node_positions[n][0], 3)
            x_groups.setdefault(x, []).append(n)
        for group in x_groups.values():
            if len(group) > 1:
                for i in range(len(group)):
                    for j in range(i+1, len(group)):
                        n1, n2 = group[i], group[j]
                        p1 = np.array(node_positions[n1])
                        p2 = np.array(node_positions[n2])
                        seg = LineString([p1, p2])
                        if any(buf.intersects(seg) for buf in obstacle_buffers.values()):
                            continue
                        dist = float(np.linalg.norm(p1 - p2))
                        graph.add_edge(n1, n2, weight=dist)
        # group by rounded Y
        y_groups: Dict[float, List[str]] = {}
        for n in portal_list:
            y = round(node_positions[n][1], 3)
            y_groups.setdefault(y, []).append(n)
        for group in y_groups.values():
            if len(group) > 1:
                for i in range(len(group)):
                    for j in range(i+1, len(group)):
                        n1, n2 = group[i], group[j]
                        p1 = np.array(node_positions[n1])
                        p2 = np.array(node_positions[n2])
                        seg = LineString([p1, p2])
                        if any(buf.intersects(seg) for buf in obstacle_buffers.values()):
                            continue
                        dist = float(np.linalg.norm(p1 - p2))
                        graph.add_edge(n1, n2, weight=dist)
        

        # 6) Return graph, all node positions, and obstacle names
        obstacle_nodes = [obs.node_name for obs in self.graph_data.obstacles]
        return (
            graph,
            node_positions,
            obstacle_nodes,
        )




def create_dataset(
    expert: bool = True,
    num_frames: int = 1000,
    base_dir: str = "datasets",
    fail_rate: float = 0.1,
) -> None:
    """Collect a dataset of expert trajectories in JEPAWorld."""

    if not expert:
        raise NotImplementedError("Only expert dataset creation is supported")

    collector = CollectTrajectories("JEPAWorld-v0")

    collected = 0

    while collected < num_frames:
        mission = collector._select_random_movable()

        # 50% chance to add a second object mission for variety (long missions)
        if random.random() < 0.5:
            mission += collector._select_random_movable()

        # Inject occasional failures
        if random.random() < fail_rate:
            fail_type = random.choice(["random_node", "target_pickup"])
            if fail_type == "random_node":
                wrong = collector.random_node()
                # ensure not using the correct nodes when possible
                avoid = {mission[0][1], mission[2][1]}
                while wrong in avoid:
                    wrong = collector.random_node()
                mission[0] = ("go_to", wrong)
            else:  # target_pickup
                mission[0] = ("go_to", mission[2][1])

        # move to a random point before starting next mission
        mission.append(("go_to", collector.random_node()))

        policy = ExpertPolicy(
            collector.env,
            collector.graph,
            collector.nodes,
            collector.node_positions,
            collector.graph_data.obstacles,
            mission,
            collector.agent,
            base_dir=base_dir,
        )

        frames = policy.solve_mission()

        if frames == 0:
            # infeasible, try another mission
            continue

        collected += frames

    collector.env.close()


if __name__ == "__main__":
    create_dataset(True, 100)
