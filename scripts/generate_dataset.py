"""Dataset generation script for JEPAWorld."""

import random
from typing import Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point, Polygon, box
from shapely import affinity

from gymnasium.envs.registration import register

from policies.expert import ExpertPolicy
from policies.helpers import Agent, GraphData, Obstacle, Portal, Room
from scripts.helpers import plot_prm_graph
from Miniworld.miniworld.entity import MeshEnt


register(
    id="JEPAWorld-v0",
    entry_point="miniworld.envs.jepaworld:JEPAWorld",
    max_episode_steps=500,
    kwargs={"seed": 0},
)


class DatasetGenerator:
    """Generate expert demonstration dataset."""

    def __init__(self, num_frames: int, output_dir: str, missions_per_env: int = 5):
        self.num_frames = num_frames
        self.output_dir = output_dir
        self.missions_per_env = missions_per_env

        self.frames_collected = 0

        self.env: gym.Env | None = None
        self.graph_data: GraphData | None = None
        self.graph: nx.Graph | None = None
        self.node_positions: Dict[str, Tuple[float, float]] | None = None
        self.nodes: List[str] | None = None
        self.agent_node: str | None = None
        self.current_movable = None
        self.mission_defs = [
            ("towl_01/towl_01", "bedroom", ("obstacle", "washer")),
            ("duckie", "living_room", ("room", "children_room")),
            ("keys_01/keys_01", "living_room", ("room", "hallway")),
            ("chips_01/chips_01", "kitchen", ("room", "living_room")),
            ("dish_01/dish_01", "living_room", ("obstacle", "sink_01")),
        ]

        self._new_env()

    # ------------------------------------------------------------------
    # Environment & Graph utilities
    # ------------------------------------------------------------------
    def _new_env(self) -> None:
        if self.env is not None:
            self.env.close()
        self.env = gym.make("JEPAWorld-v0", seed=random.randint(0, 2**31 - 1))
        self.env.reset()
        print (f"New environment created: ")
        self.current_movable = None
        self._update_graph()


    def _update_graph(self) -> None:
        self.graph_data = self.get_graph_data()
        self.graph, self.node_positions, self.nodes = self.build_prm_graph(
            sample_density=3.0,
            k_neighbors=9,
            jitter_ratio=0.0,
            min_samples=0,
            min_dist=0.1,
        )
        self.agent_node = next(n for n in self.nodes if n.startswith("agent"))

    # ------------------------------------------------------------------
    # Mission helpers
    # ------------------------------------------------------------------
    


    def _remove_current_movable(self) -> None:
        if self.current_movable is None:
            return
        if self.env.unwrapped.agent.carrying is self.current_movable:
            self.env.unwrapped.agent.carrying = None
        if self.current_movable in self.env.unwrapped.entities:
            self.env.unwrapped.entities.remove(self.current_movable)
        self.current_movable = None

    def _select_mission(self) -> Tuple[str, str]:
        """Set up and return a mission based on predefined definitions."""
        mesh, room_name, dest = random.choice(self.mission_defs)
        # Place the movable in the specified room
        height = self.env.unwrapped.SCALE_FACTORS.get(mesh, 0.1)
        ent = MeshEnt(mesh_name=mesh, height=height, static=False)
        room = self.env.unwrapped.named_rooms.get(room_name)
        if room is None:
            raise ValueError(f"Unknown room '{room_name}' in mission definition")
        self.env.unwrapped.place_entity(ent, room=room)
        self.current_movable = ent
        # Rebuild the graph now that the movable is placed
        self._update_graph()

        # Determine pick-up node for the placed movable
        prefix = mesh.split("/")[0]
        movable_candidates = [n for n in self.nodes if n.startswith(prefix)]
        if not movable_candidates:
            raise ValueError(f"No node found for movable '{mesh}'")
        pick_node = movable_candidates[0]

        # Determine drop target
        dest_type, dest_name = dest
        if dest_type == "room":
            rid = self.env.unwrapped.room_name_to_id.get(dest_name)
            if rid is None:
                raise ValueError(f"Unknown room '{dest_name}' in mission definition")
            drop_candidates = [n for n in self.node_positions if n.startswith(f"r{rid}_s")]
        else:
            drop_candidates = [n for n in self.node_positions if n.startswith(dest_name)]

        if not drop_candidates:
            raise ValueError("No valid drop location found for mission")
        drop_node = random.choice(drop_candidates)

        return pick_node, drop_node

    def _random_drop_target(self, exclude: Sequence[str]) -> str:
        candidates = [n for n in self.node_positions if n not in set(exclude)]
        return random.choice(candidates)

    # ------------------------------------------------------------------
    def get_graph_data(self) -> GraphData:
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
        graph = nx.Graph()
        room_polygons = {r.id: Polygon(r.vertices) for r in self.graph_data.rooms}

        agent_radius = 0.25
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
                    start = np.array(p.start)
                    end = np.array(p.end)
                    dir_vec = end - start
                    norm = np.linalg.norm(dir_vec)
                    unit = dir_vec / norm if norm > 0 else np.array([1.0, 0.0])
                    perp = np.array([-unit[1], unit[0]])

                    cand1 = np.array(mid_pt) + perp * portal_offset
                    cand2 = np.array(mid_pt) - perp * portal_offset
                    pos = cand1 if poly.contains(Point(*cand1)) else cand2

                    name = f"_portal_{portal_idx}"
                    portal_nodes[key] = name
                    node_positions[name] = (float(pos[0]), float(pos[1]))
                    portal_idx += 1

        for obs in self.graph_data.obstacles:
            node_positions[obs.node_name] = tuple(obs.pos)

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

        def room_of(point: Tuple[float, float]) -> Optional[int]:
            for rid, poly in room_polygons.items():
                if poly.covers(Point(point)):
                    return rid
            return None

        agent_room = room_of(self.graph_data.agent.pos)

        for rid, poly in room_polygons.items():
            nodes_in_room = [
                name
                for name, pos in node_positions.items()
                if poly.covers(Point(pos)) and not (name == "agent" and agent_room != rid)
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

        portal_list = [n for n in node_positions if n.startswith("_portal_")]
        x_groups: Dict[float, List[str]] = {}
        for n in portal_list:
            x = round(node_positions[n][0], 3)
            x_groups.setdefault(x, []).append(n)
        for group in x_groups.values():
            if len(group) > 1:
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        n1, n2 = group[i], group[j]
                        p1 = np.array(node_positions[n1])
                        p2 = np.array(node_positions[n2])
                        seg = LineString([p1, p2])
                        if any(buf.intersects(seg) for buf in obstacle_buffers.values()):
                            continue
                        dist = float(np.linalg.norm(p1 - p2))
                        graph.add_edge(n1, n2, weight=dist)

        y_groups: Dict[float, List[str]] = {}
        for n in portal_list:
            y = round(node_positions[n][1], 3)
            y_groups.setdefault(y, []).append(n)
        for group in y_groups.values():
            if len(group) > 1:
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        n1, n2 = group[i], group[j]
                        p1 = np.array(node_positions[n1])
                        p2 = np.array(node_positions[n2])
                        seg = LineString([p1, p2])
                        if any(buf.intersects(seg) for buf in obstacle_buffers.values()):
                            continue
                        dist = float(np.linalg.norm(p1 - p2))
                        graph.add_edge(n1, n2, weight=dist)

        obstacle_nodes = [obs.node_name for obs in self.graph_data.obstacles]
        return graph, node_positions, obstacle_nodes

    # ------------------------------------------------------------------
    def _run_mission(self) -> bool:
        pick, drop_target = self._select_mission()

        policy = ExpertPolicy(
            self.env,
            self.graph,
            self.nodes,
            self.node_positions,
            self.graph_data.obstacles,
            [],
            dataset_dir=self.output_dir,
        )

        attempts = 0
        success = policy.go_to(pick)
        while not success and attempts < 3:
            policy.obs = []
            policy.actions = []
            pick, drop_target = self._select_mission()
            success = policy.go_to(pick)
            attempts += 1
        if not success:
            self.env.unwrapped.place_agent()
            self._remove_current_movable()
            return False

        success = policy.pick_up()
        if not success or not self.env.unwrapped.agent.carrying:
            policy.obs = []
            policy.actions = []
            self.env.unwrapped.place_agent()
            self._remove_current_movable()
            return False

        attempts = 0
        drop_success = False
        target = drop_target
        while attempts < 3 and not drop_success:
            if policy.go_to(target) and policy.drop() and not self.env.unwrapped.agent.carrying:
                drop_success = True
            else:
                policy.obs = []
                policy.actions = []
                target = self._random_drop_target(exclude=[pick, self.agent_node])
                attempts += 1

        if not drop_success:
            self.env.unwrapped.place_agent()
            self._remove_current_movable()
            return False

        policy.actions.append(-1)
        self.frames_collected += policy._save_episode()
        self.env.unwrapped.place_agent()
        self._remove_current_movable()
        plot_prm_graph(
                self.graph_data,
                self.graph,
                self.node_positions,
                self.nodes,
                highlight_path=policy.path,
                smoothed_curve=None,
            )
        return True

    # ------------------------------------------------------------------
    def generate(self) -> None:
        missions_in_env = 0
        while self.frames_collected < self.num_frames:
            if missions_in_env >= self.missions_per_env:
                self._new_env()
                missions_in_env = 0

            self._run_mission()
            missions_in_env += 1
            
        


def main() -> None:
    generator = DatasetGenerator(num_frames=50, output_dir="/Users/julianquast/Documents/Bachelor Thesis/Datasets")
    generator.generate()


if __name__ == "__main__":
    main()

