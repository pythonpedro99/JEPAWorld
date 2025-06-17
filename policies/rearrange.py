# ─────────────────────────────────────────────────────────────
# Expert rearrangement policy (refactored)
# ─────────────────────────────────────────────────────────────
import math
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np
import gymnasium as gym
import networkx as nx


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
    TURN_TOL = 15.0                   # how precisely to face a target

    def __init__(
        self,
        env: gym.Env,
        prm_graph: nx.Graph,
        node_positions: Dict[str, Tuple[float, float]],
    ) -> None:
        self.env          = env
        self.prm_graph    = prm_graph
        self.node_pos2d   = node_positions
        

        # action ids (MiniWorld)
        self._TURN_LEFT  = 0 #env.actions.turn_left
        self._TURN_RIGHT = 1 #env.actions.turn_right
        self._IDLE       = -1 #getattr(env.actions, "no_op", -1)  # fallback

        # public running log
        self.observations: List[np.ndarray] = []
        self.actions_taken: List[int]       = []
        self.path = []  # list of (x, z) tuples for the agent to follow


    def perform_full_scan(self, yaw_tol_deg: float = 10.0) -> None:
        """
        Rotate CCW (turn_left) through ≤360° so that every obstacle is faced
        exactly once.  Observations and actions are appended to the public lists:
            self.observations   list[np.ndarray]
            self.actions_taken  list[int]
        """
        agent   = self.env.unwrapped.agent
        yaw0    = agent.dir
        wrapCCW = lambda a: (a - yaw0) % (2 * math.pi)  # 0..2π from current yaw

        # ── collect bearings of all relevant nodes ────────────────────
        bearings: List[float] = []
        for name, (x, z) in self.node_pos2d.items():
            if name.startswith(("Agent", "TextFrame","s")):
                continue
            alpha = math.atan2(x - agent.pos[0], z - agent.pos[2])
            bearings.append(wrapCCW(alpha))
            print(bearings)            # in [0, 2π)

        bearings.sort()                                # CCW order
        yaw_tol = math.radians(yaw_tol_deg)
        TURN_LEFT = self._TURN_LEFT                   # shorthand

        # ── iterate once through the sorted list ──────────────────────
        for target in bearings:
            # convert target back to global yaw
            desired = (yaw0 + target) % (2 * math.pi)
            while True:
                error = _angle_diff(desired, agent.dir)
                if abs(error) <= yaw_tol:
                    break                              # next obstacle
                obs, _, _, _, _ = self.env.step(TURN_LEFT)
                self.observations.append(obs)
                self.actions_taken.append(TURN_LEFT)
    
        # ── inside HumanLikeRearrangePolicy ─────────────────────────────────
    def go_to(self, goal: str) -> bool:
        """
        Navigate to the PRM node called `goal`, verbosely.

        Prints:
            • path-planning details (start, goal, full path)
            • per-waypoint status (agent pose, target, error, chosen action)
            • stuck / timeout diagnostics
        """
        TURN_LEFT, TURN_RIGHT, FORWARD = 0, 1, 2
        POS_TOL   = 0.10
        LAST_TOL  = 0.95
        TURN_TOL  = math.radians(10.0)

        # ── 1 / A* path planning ─────────────────────────────────────────
        ax, az = self.env.unwrapped.agent.pos[0], self.env.unwrapped.agent.pos[2]
        agent_xy = (float(ax), float(az))
        _euclid = lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1])

        start_node = min(self.node_pos2d, key=lambda n: _euclid(self.node_pos2d[n], agent_xy))
        walkable_graph = self.prm_graph.copy()
        for n in list(walkable_graph.nodes):
            if not ((n in (start_node, goal)) or n.startswith("s")):
                walkable_graph.remove_node(n)

        try:
            self.path = nx.astar_path(
                walkable_graph, start_node, goal,
                heuristic=lambda n1, n2: _euclid(self.node_pos2d[n1], self.node_pos2d[n2]),
                weight="weight",
            )
        except nx.NetworkXNoPath:
            print(f"[go_to] no path from {start_node} to {goal}")
            return False

        waypoints = [self.node_pos2d[n] for n in self.path]
        print(f"[go_to] start={start_node}  goal={goal}")
        print(f"[go_to] A* path: {'  ->  '.join(self.path)}")
        print(f"[go_to] waypoints (x,z): {waypoints}")

        # ── 2 / Walk the waypoints ───────────────────────────────────────
        for idx, (wx, wz) in enumerate(waypoints):
            target_tol = LAST_TOL if idx == len(waypoints) - 1 else POS_TOL
            print(f"\n[waypoint {idx+1}/{len(waypoints)}] target=({wx:.2f},{wz:.2f}) "
                f"tol={target_tol:.2f} m")
            no_move = 0

            while True:
                # current pose
                ax, az = self.env.unwrapped.agent.pos[0], self.env.unwrapped.agent.pos[2]
                ayaw   = self.env.unwrapped.agent.dir
                dx, dz = wx - ax, wz - az
                dist   = math.hypot(dx, dz)

                if dist <= target_tol:
                    print(f"[reached] dist={dist:.3f} m  pos=({ax:.2f},{az:.2f})")
                    break

                desired = math.atan2(-dz, dx) 
                err     = (desired - ayaw + math.pi) % (2*math.pi) - math.pi

                if abs(err) > TURN_TOL:
                    cmd = TURN_LEFT if err > 0 else TURN_RIGHT
                    act_name = "LEFT" if err > 0 else "RIGHT"
                else:
                    cmd = FORWARD
                    act_name = "FWD"

                print(f"[step] pos=({ax:.2f},{az:.2f}) yaw={math.degrees(ayaw):6.1f}°  "
                    f"→ tgt=({wx:.2f},{wz:.2f})  dist={dist:.3f} m target_tol={target_tol:.2f} m  "
                    f"err={math.degrees(err):6.1f}°  action={act_name}")

                obs, _, term, trunc, _ = self.env.step(cmd)
                self.observations.append(obs)
                self.actions_taken.append(cmd)

                if term or trunc:
                    print("[go_to] episode ended prematurely")
                    return False

                # forward-movement check
                if cmd == FORWARD:
                    moved = math.hypot(
                        self.env.unwrapped.agent.pos[0] - ax,
                        self.env.unwrapped.agent.pos[2] - az
                    )
                    no_move = no_move + 1 if moved < 1e-3 else 0
                    if no_move >= 5:
                        print("[go_to] stuck (no progress) — aborting")
                        return False
                else:
                    no_move = 0  # reset on a turn

        print("[go_to] all waypoints reached ✓")
        return True




































        # # Mutable task state
        # self._seen_names: set[str] = set()        # obstacles already seen in scan
        # self._carry_idx: int | None = None
        # self._targets: deque[Tuple[str, Tuple[float, float]]] = deque()
        # self._state = "scan"                      # scan → carry → final_view
        # self._subplan: deque[int] = deque()       # env actions queued

        # # Pre‑compute neat row of drop positions along far wall (+Z)
        # self._drop_sites = self._compute_drop_sites()








