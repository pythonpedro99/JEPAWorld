# ─────────────────────────────────────────────────────────────
# Expert rearrangement policy (refactored)
# ─────────────────────────────────────────────────────────────
import math
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np
import gymnasium as gym
import networkx as nx
from scripts.helpers import build_prm_graph_single_room, get_graph_data

import random


# ─── simple dataclasses (unchanged) ─────────────────────────────────
@dataclass
class Room:
    id: int
    vertices: List[Tuple[float, float]]


@dataclass
class Agent:
    pos: Tuple[float, float]
    yaw: float
    radius: float


@dataclass
class Obstacle:
    type: str
    pos: Tuple[float, float]
    radius: float
    node_name: str
    yaw: float
    size: Tuple[float, float]


# ─── helper ─────────────────────────────────────────────────────────
def _angle_diff(target: float, current: float) -> float:
    """Smallest signed difference between two angles (-π … +π)."""
    return (target - current + math.pi) % (2 * math.pi) - math.pi


# ─── main policy ────────────────────────────────────────────────────
class HumanLikeRearrangePolicy:
    """Only stage 1 implemented: rotate until every obstacle has been seen once."""

    # tunables
    TURN_TOL = 10.0  # how precisely to face a target

    def __init__(
        self,
        env: gym.Env,
        seed: int = 0,
    ) -> None:
        self.observations: List[np.ndarray] = []
        self.actions: List[int] = []
        self.env = env
        self.rng = np.random.default_rng(seed)  # fixed seed for reproducibility
        obs, _ = self.env.reset()
        self.agent_start_pos = (
            self.env.unwrapped.agent.pos[0],
            self.env.unwrapped.agent.pos[2],
        )
        print(self.env.unwrapped.actions)
        self.observations.append(obs)
        self.graph_data = get_graph_data(env)
        self.prm_graph, self.node_pos2d = build_prm_graph_single_room(
            self.graph_data,
            sample_density=0.3,
            k_neighbors=15,
            jitter_ratio=0.3,
            min_samples=5,
            min_dist=0.2,
            agent_radius=0.2,
        )
       
        self.max_arrangements = 4
        self.view_distance = 4.0
        self._TURN_LEFT = 0  # env.actions.turn_left
        self._TURN_RIGHT = 1  # env.actions.turn_right

        self.path = []  # list of (x, z) tuples for the agent to follow

    def turn_towards(self, target_pos, yaw_tol_deg: float = 10.0):
        """
        Rotate in place until facing the target position.
        Appends each turn action and resulting observation.
        """
       
        agent = self.env.unwrapped.agent
        dx = target_pos[0] - agent.pos[0]
        dz = target_pos[1] - agent.pos[2]
        # desired yaw toward target
        desired = math.atan2(-dz, dx)
        yaw_tol = math.radians(yaw_tol_deg)

        while True:
            current = agent.dir
            error = (desired - current + math.pi) % (2 * math.pi) - math.pi
            if abs(error) <= yaw_tol:
                break
            cmd = self._TURN_LEFT if error > 0 else self._TURN_RIGHT
            obs, _, term, trunc, _ = self.env.step(cmd)
            self.actions.append(cmd)
            self.observations.append(obs)
    
    def wiggle(self, n: int):
        """
        Rotate in place:
        1) n steps to the left,
        2) n steps back to the original yaw,
        3) n steps to the right.
        Records each action+observation in self.actions/self.observations.
        """
        # shorthand for left/right commands
        CMD_LEFT = 0
        CMD_RIGHT = 1

        # 1) n steps left
        for _ in range(n):
            cmd = CMD_LEFT
            obs, _, term, trunc, _ = self.env.step(cmd)
            self.actions.append(cmd)
            self.observations.append(obs)

        # 2) n steps back (undo the left turns)
        for _ in range(2*n):
            cmd = CMD_RIGHT
            obs, _, term, trunc, _ = self.env.step(cmd)
            self.actions.append(cmd)
            self.observations.append(obs)


    def rearrange(self):
        # collect object node IDs by prefix
        object_nodes = [
            nid
            for nid in self.node_pos2d.keys()
            if any(str(nid).startswith(pref) for pref in ("Box", "Ball", "Key"))
        ]
        # choose how many to rearrange (allow zero)
        n = self.rng.integers(1, min(self.max_arrangements, len(object_nodes)))
        targets = list(self.rng.choice(object_nodes, size=n, replace=False))

        # sample goal nodes from samples
        sample_nodes = [nid for nid in self.node_pos2d if str(nid).startswith("s")]
        goal_nodes = self.rng.choice(sample_nodes, size=n, replace=False).tolist()

        # initial orientation: full ordered scan of obstacles
        self.wiggle(3)  # wiggle to randomize initial yaw

        # execute pick-and-place sequence
        for obj_node, goal in zip(targets, goal_nodes):
            # pick up
            if not self.go_to(obj_node):
                return False
            obs, _, term, trunc, _ = self.env.step(4)
            self.actions.append(4)
            self.observations.append(obs)
            if not self.env.unwrapped.agent.carrying:
                return False

            # drop
            if not self.go_to(goal):
                return False
            obs, _, term, trunc, _ = self.env.step(5)
            self.actions.append(5)
            self.observations.append(obs)
            if self.env.unwrapped.agent.carrying:
                return False

            self.graph_data = get_graph_data(self.env)
            self.prm_graph, self.node_pos2d = build_prm_graph_single_room(
                self.graph_data,
                sample_density=0.3,
                k_neighbors=15,
                jitter_ratio=0.3,
                min_samples=5,
                min_dist=0.2,
                agent_radius=0.2,
            )

        start_2d = np.array([
            self.agent_start_pos[0],
            self.agent_start_pos[1]
        ])

        # --- 3) Compute room bounds from Room vertices ---
        room: Room = self.graph_data.room
        xs, zs = zip(*room.vertices)
        min_x, max_x = min(xs), max(xs)
        min_z, max_z = min(zs), max(zs)

        # optional inset from walls
        offset = 0.2
        min_x += offset; max_x -= offset
        min_z += offset; max_z -= offset

        # --- 4) Find sampling/navigation nodes ---
        sample_nodes = [nid for nid in self.node_pos2d if str(nid).startswith("s")]
        if not sample_nodes:
            raise RuntimeError("No sample nodes available for navigation.")

        # --- 5) Choose node closest to start ---
        closest_start_node = min(
            sample_nodes,
            key=lambda nid: np.linalg.norm(
                np.array(self.node_pos2d[nid]) - start_2d
            )
        )

        # --- 6) Navigate back to start-like location ---
        if not self.go_to(closest_start_node):
            return False

        # --- 7) Compute room center and yaw ---
        room_center = np.array([
            (min_x + max_x) / 2,
            (min_z + max_z) / 2
        ])
        self.turn_towards(room_center)

        # --- 8) End episode ---
        end_cmd = 7
        obs, _, term, trunc, _ = self.env.step(end_cmd)
        self.actions.append(end_cmd)
        self.observations.append(obs)

        return self.actions, self.observations

        # # compute cluster centroid
        # pts = [self.node_pos2d[n] for n in goal_nodes]
        # centroid = np.mean(pts, axis=0)
        # start = np.array(
        #     [self.env.unwrapped.agent.pos[0], self.env.unwrapped.agent.pos[2]]
        # )

        # vec = centroid - start
        # norm = np.linalg.norm(vec)
        # unit = vec / norm if norm > 1e-3 else np.array([1.0, 0.0])
        # view_point = centroid + unit * self.view_distance

        # # view configuration
        # sample_nodes = [nid for nid in self.node_pos2d if str(nid).startswith("s")]
        # view_node = min(
        #     sample_nodes,
        #     key=lambda nid: np.linalg.norm(np.array(self.node_pos2d[nid]) - view_point),
        # )
        # self.go_to(view_node)
        # self.turn_towards(centroid)

        # # end episode
        # obs, _, term, trunc, _ = self.env.step(7)
        # self.actions.append(7)
        # self.observations.append(obs)
        # return self.actions, self.observations

    def go_to(self, goal: str) -> bool:
        """
        Navigate to the PRM node called `goal`, verbosely.

        Prints:
            • path-planning details (start, goal, full path)
            • per-waypoint status (agent pose, target, error, chosen action)
            • stuck / timeout diagnostics
        """
        TURN_LEFT, TURN_RIGHT, FORWARD = 0, 1, 2
        POS_TOL = 0.10
        LAST_TOL = 1.0
        TURN_TOL = math.radians(10.0)

        # ── 1 / A* path planning ─────────────────────────────────────────
        ax, az = self.env.unwrapped.agent.pos[0], self.env.unwrapped.agent.pos[2]
        agent_xy = (float(ax), float(az))
        _euclid = lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1])

        start_node = min(
            self.node_pos2d, key=lambda n: _euclid(self.node_pos2d[n], agent_xy)
        )
        walkable_graph = self.prm_graph.copy()
        for n in list(walkable_graph.nodes):
            if not ((n in (start_node, goal)) or n.startswith("s")):
                walkable_graph.remove_node(n)

        try:
            self.path = nx.astar_path(
                walkable_graph,
                start_node,
                goal,
                heuristic=lambda n1, n2: _euclid(
                    self.node_pos2d[n1], self.node_pos2d[n2]
                ),
                weight="weight",
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
            print(f"[go_to] A* failed from {start_node} to {goal}: {e}")
            return False

        waypoints = [self.node_pos2d[n] for n in self.path]
        print(f"[go_to] start={start_node}  goal={goal}")
        print(f"[go_to] A* path: {'  ->  '.join(self.path)}")
        print(f"[go_to] waypoints (x,z): {waypoints}")

        # ── 2 / Walk the waypoints ───────────────────────────────────────
        for idx, (wx, wz) in enumerate(waypoints):
            target_tol = LAST_TOL if idx == len(waypoints) - 1 else POS_TOL
            print(
                f"\n[waypoint {idx+1}/{len(waypoints)}] target=({wx:.2f},{wz:.2f}) "
                f"tol={target_tol:.2f} m"
            )
            no_move = 0

            while True:
                # current pose
                ax, az = (
                    self.env.unwrapped.agent.pos[0],
                    self.env.unwrapped.agent.pos[2],
                )
                ayaw = self.env.unwrapped.agent.dir
                dx, dz = wx - ax, wz - az
                dist = math.hypot(dx, dz)

                if dist <= target_tol:
                    print(f"[reached] dist={dist:.3f} m  pos=({ax:.2f},{az:.2f})")
                    break

                desired = math.atan2(-dz, dx)
                err = (desired - ayaw + math.pi) % (2 * math.pi) - math.pi

                if abs(err) > TURN_TOL:
                    cmd = TURN_LEFT if err > 0 else TURN_RIGHT
                    act_name = "LEFT" if err > 0 else "RIGHT"
                else:
                    cmd = FORWARD
                    act_name = "FWD"

                print(
                    f"[step] pos=({ax:.2f},{az:.2f}) yaw={math.degrees(ayaw):6.1f}°  "
                    f"→ tgt=({wx:.2f},{wz:.2f})  dist={dist:.3f} m target_tol={target_tol:.2f} m  "
                    f"err={math.degrees(err):6.1f}°  action={act_name}"
                )

                obs, _, term, trunc, _ = self.env.step(cmd)
                self.observations.append(obs)
                self.actions.append(cmd)

                if term or trunc:
                    print("[go_to] episode ended prematurely")
                    return False

                # forward-movement check
                if cmd == FORWARD:
                    moved = math.hypot(
                        self.env.unwrapped.agent.pos[0] - ax,
                        self.env.unwrapped.agent.pos[2] - az,
                    )
                    no_move = no_move + 1 if moved < 1e-3 else 0
                    if no_move >= 2:
                        print("[go_to] stuck (no progress) — aborting")
                        return False
                else:
                    no_move = 0  # reset on a turn

        print("[go_to] all waypoints reached ✓")
        return True
