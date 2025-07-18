import math
from typing import List, Tuple, Union
import numpy as np
import gymnasium as gym
import networkx as nx
from policies.helpers import build_prm_graph_single_room, get_graph_data, Room, plot_room_with_obstacles_and_path
from shapely.geometry import Point, Polygon


class HumanLikeRearrangePolicy:
    '''
    Policy for human-like object rearrangement in a single room.

    Stage 1: rotate until every obstacle has been seen once, then pick and place objects.
    '''
    TURN_TOL: float = 10.0

    def __init__(
        self,
        env: gym.Env,
        seed: int = 0,
        target: List[str]= [],
    ) -> None:
        '''
        Initialize the rearrangement policy.

        Parameters
        ----------
        env : gym.Env
            The Gymnasium environment to interact with.
        seed : int, optional
            Seed for reproducible randomness, by default 0.
        '''
        self.observations: List[np.ndarray] = []
        self.actions: List[int] = []
        self.env = env
        self.target = target
        self.rng = np.random.default_rng(seed)
        obs, _ = self.env.reset(seed=seed)
        self.agent_start_pos = (
            self.env.unwrapped.agent.pos[0],
            self.env.unwrapped.agent.pos[2],
        )
        self.observations.append(obs)
        self.graph_data = get_graph_data(env)
        self.prm_graph, self.node_pos2d = build_prm_graph_single_room(
            self.graph_data,
            sample_density=0.7,
            k_neighbors=10,
            jitter_ratio=0.0,
            min_samples=30,
            min_dist=0.4,
            agent_radius=0.35,
        )
        self.max_arrangements: int = 1
        self._TURN_LEFT: int = 0
        self._TURN_RIGHT: int = 1
        self.path: List[Tuple[float, float]] = []

    def turn_towards(
        self,
        target_pos: Tuple[float, float],
        yaw_tol_deg: float = 10.0,
    ) -> None:
        '''
        Rotate in place until the agent faces the target position within a yaw tolerance.

        Parameters
        ----------
        target_pos : Tuple[float, float]
            Target position as (x, z) coordinates.
        yaw_tol_deg : float, optional
            Yaw tolerance in degrees, by default 10.0.
        '''
        agent = self.env.unwrapped.agent
        dx = target_pos[0] - agent.pos[0]
        dz = target_pos[1] - agent.pos[2]
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

    def wiggle(self, n: int) -> None:
        '''
        Perform a wiggle turn: n steps left, back, then right.

        Parameters
        ----------
        n : int
            Number of steps for each leg of the wiggle.
        '''
        CMD_LEFT, CMD_RIGHT = 0, 1

        for _ in range(n):
            obs, _, term, trunc, _ = self.env.step(CMD_LEFT)
            self.actions.append(CMD_LEFT)
            self.observations.append(obs)

        for _ in range(2 * n):
            obs, _, term, trunc, _ = self.env.step(CMD_RIGHT)
            self.actions.append(CMD_RIGHT)
            self.observations.append(obs)

        for _ in range(n):
            obs, _, term, trunc, _ = self.env.step(CMD_LEFT)
            self.actions.append(CMD_LEFT)
            self.observations.append(obs)

    def rearrange(self) -> Union[bool, Tuple[List[int], List[np.ndarray]]]:
        '''
        Execute the rearrangement sequence: pick and place objects, then return to start and end.

        Returns
        -------
        Union[bool, Tuple[List[int], List[np.ndarray]]]
            False on failure, otherwise (actions, observations).
        '''
        self.wiggle(6)
        object_nodes = [
            nid
            for nid in self.node_pos2d.keys()
            if any(str(nid).startswith(pref) for pref in ('Box', 'Ball', 'Key'))
        ]
        n = 1
        #targets = list(self.rng.choice(object_nodes, size=n, replace=False))
        targets = self.target

        room_verts = self.graph_data.room.vertices
        xs, zs = zip(*room_verts)
        min_x, max_x = min(xs), max(xs)
        min_z, max_z = min(zs), max(zs)

        inner_rect = Polygon([
            (min_x + 1.0, min_z + 3.2),
            (max_x - 1.0, min_z + 3.2),
            (max_x - 1.0, max_z - 2.5),
            (min_x + 1.0, max_z - 2.5),
        ])

        DIST_T: float = 2.0
        target0_pos = np.array(self.node_pos2d[targets[0]])

        filtered_sample_nodes = [
            nid
            for nid in self.node_pos2d
            if (
                str(nid).startswith('s')
                and Point(self.node_pos2d[nid]).within(inner_rect)
                and np.linalg.norm(
                    np.array(self.node_pos2d[nid]) - target0_pos
                ) > DIST_T
            )
        ]
        if not filtered_sample_nodes:
            return False

        goal_nodes = self.rng.choice(filtered_sample_nodes, size=n, replace=False).tolist()

        for obj_node, goal in zip(targets, goal_nodes):
            if not self.go_to(obj_node):
                return False
            obs, _, term, trunc, _ = self.env.step(4)
            self.actions.append(4)
            self.observations.append(obs)
            if not self.env.unwrapped.agent.carrying:
                return False

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
                sample_density=0.7,
                k_neighbors=10,
                jitter_ratio=0.0,
                min_samples=30,
                min_dist=0.4,
                agent_radius=0.35,
            )

        start_2d = np.array([
            self.agent_start_pos[0],
            self.agent_start_pos[1],
        ])

        room: Room = self.graph_data.room
        xs, zs = zip(*room.vertices)
        min_x, max_x = min(xs), max(xs)
        min_z, max_z = min(zs), max(zs)

        offset: float = 0.2
        min_x += offset
        max_x -= offset
        min_z += offset
        max_z -= offset

        sample_nodes = [nid for nid in self.node_pos2d if str(nid).startswith('s')]
        if not sample_nodes:
            raise RuntimeError('No sample nodes available for navigation.')

        closest_start_node = min(
            sample_nodes,
            key=lambda nid: math.hypot(
                self.node_pos2d[nid][0] - start_2d[0],
                self.node_pos2d[nid][1] - start_2d[1],
            ),
        )

        if not self.go_to(closest_start_node):
            return False
        
        end_cmd: int = 2
        obs, _, term, trunc, _ = self.env.step(end_cmd)
        self.actions.append(end_cmd)
        self.observations.append(obs)
        obs, _, term, trunc, _ = self.env.step(end_cmd)
        self.actions.append(end_cmd)
        self.observations.append(obs)
        obs, _, term, trunc, _ = self.env.step(end_cmd)
        self.actions.append(end_cmd)
        self.observations.append(obs)
        obs, _, term, trunc, _ = self.env.step(end_cmd)
        self.actions.append(end_cmd)
        self.observations.append(obs)
        obs, _, term, trunc, _ = self.env.step(end_cmd)
        self.actions.append(end_cmd)
        self.observations.append(obs)

        self.turn_towards((3.5, 10.0))
        # end_cmd: int = 7
        # obs, _, term, trunc, _ = self.env.step(end_cmd)
        # self.actions.append(end_cmd)
        # self.observations.append(obs)

        return self.actions, self.observations

    def go_to(self, goal: str) -> bool:
        '''
        Navigate to the specified PRM node using A* path planning and step execution.

        Parameters
        ----------
        goal : str
            The name of the PRM node to navigate to.

        Returns
        -------
        bool
            True if navigation succeeded, False otherwise.
        '''
        TURN_LEFT, TURN_RIGHT, FORWARD = 0, 1, 2
        POS_TOL: float = 0.10
        LAST_TOL: float = 1.2
        TURN_TOL_RAD: float = math.radians(10.0)

        ax, az = self.env.unwrapped.agent.pos[0], self.env.unwrapped.agent.pos[2]
        _euclid = lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1])

        start_node = min(
            self.node_pos2d,
            key=lambda n: _euclid(self.node_pos2d[n], (ax, az)),
        )
        walkable_graph = self.prm_graph.copy()
        for n in list(walkable_graph.nodes):
            if not ((n == start_node or n == goal) or n.startswith('s')):
                walkable_graph.remove_node(n)

        try:
            self.path = nx.astar_path(
                walkable_graph,
                start_node,
                goal,
                heuristic=lambda n1, n2: _euclid(self.node_pos2d[n1], self.node_pos2d[n2]),
                weight='weight',
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return False
        
        # plot_room_with_obstacles_and_path(
        #     self.graph_data.room,
        #     self.graph_data.obstacles,
        #     self.node_pos2d,
        #     self.prm_graph,
        #     self.path,
        #     title="",
        # )

        waypoints = [self.node_pos2d[n] for n in self.path]

        for idx, (wx, wz) in enumerate(waypoints):
            target_tol = LAST_TOL if idx == len(waypoints) - 1 else POS_TOL
            no_move_count = 0

            while True:
                ax, az = self.env.unwrapped.agent.pos[0], self.env.unwrapped.agent.pos[2]
                ayaw = self.env.unwrapped.agent.dir
                dx, dz = wx - ax, wz - az
                dist = math.hypot(dx, dz)

                if dist <= target_tol:
                    break

                desired = math.atan2(-dz, dx)
                err = (desired - ayaw + math.pi) % (2 * math.pi) - math.pi

                if abs(err) > TURN_TOL_RAD:
                    cmd = TURN_LEFT if err > 0 else TURN_RIGHT
                else:
                    cmd = FORWARD

                obs, _, term, trunc, _ = self.env.step(cmd)
                self.actions.append(cmd)
                self.observations.append(obs)

                if term or trunc:
                    return False

                if cmd == FORWARD:
                    moved = math.hypot(
                        self.env.unwrapped.agent.pos[0] - ax,
                        self.env.unwrapped.agent.pos[2] - az,
                    )
                    no_move_count = no_move_count + 1 if moved < 1e-3 else 0
                    if no_move_count >= 3:
                        return False
                else:
                    no_move_count = 0

        return True
