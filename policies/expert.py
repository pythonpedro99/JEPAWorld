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
from policies.helpers import find_path, smooth_path
from shapely.geometry import Point
import numpy as np
import random
import time
from scripts.helpers import save_data_batch


MINIWORLD_PATH = os.path.expanduser("../MiniWorld")
if MINIWORLD_PATH not in sys.path:
    sys.path.append(MINIWORLD_PATH)


class ExpertPolicy:
    """
    Expert policy for path planning in MiniWorld environments.
    """

    def __init__(
        self,
        env: gym.Env,
        graph: nx.Graph,
        nodes,
        nodes_positions,
        obstacles,
        mission: list[str],
    ):
        # Basics
        self.env = env
        self.graph = graph
        self.mission = mission
        self.node_positions = nodes_positions
        self.nodes = nodes
        self.obstacles = obstacles
        self.obs = []
        self.actions = []
        self.path = []

        # Debugging
        self.waypoints = []
        self.smoothed_waypoints = []
        # Default values
        self.forward_step = 0.15
        self.turn_step = np.deg2rad(15)
        self.turn_tol = self.turn_step / 2
        self.chaikins_iterations = 3
        self.min_forward_prob = 0.1  # minimum forward-if-misaligned probability
        self.max_forward_prob = 0.5  # maximum forward-if-misaligned probability
        self.CMD_TURN_RIGHT = 0
        self.CMD_TURN_LEFT = 1
        self.CMD_FORWARD = 2
        self.pause_prob = 0.1  # probability of pausing at each step

    def solve_mission(self) -> None:
        self.go_to(self.mission[0])
        save_data_batch(self.obs, self.actions, "test_data")
        return None
        # TODO: Implement the rest of the mission logic

    def go_to(self, goal: str) -> None:
        # 1) Look up obstacle (if any) to get its radius; waypoints get 0.0
        target_obs = next((o for o in self.obstacles if o.node_name == goal), None)
        target_buffer = target_obs.radius if target_obs else 0.0

        # 2) Plan & smooth
        path = find_path(self.graph, self.nodes, "Agent_6", goal)
        self.path += path
        waypoints = [self.node_positions[node] for node in path]
        self.waypoints += waypoints
        smoothed_waypoints = smooth_path(waypoints, self.chaikins_iterations)
        self.smoothed_waypoints += smoothed_waypoints
        if not smoothed_waypoints:
            return

        # 3) Precompute goal point
        goal_pt = Point(smoothed_waypoints[-1][0], smoothed_waypoints[-1][1])

        # 4) Walk the smoothed path
        for gx, gy in smoothed_waypoints:
            while True:
                # current agent position & heading
                x, y = self.env.unwrapped.agent.pos[0], self.env.unwrapped.agent.pos[2]
                yaw = self.env.unwrapped.agent.dir
                curr_pt = Point(x, y)

                # a) If within buffer (or exactly at the point when buffer=0), we’re done with this waypoint
                if curr_pt.distance(goal_pt) <= target_buffer:
                    break

                # b) Maybe pause
                if random.random() < self.pause_prob:
                    time.sleep(random.uniform(0.1, 0.3))

                # c) Compute control
                desired = np.arctan2(-(gy - y), (gx - x))
                err = ((desired - yaw + np.pi) % (2 * np.pi)) - np.pi
                align = 1 - min(abs(err) / np.pi, 1)
                fprob = (
                    self.min_forward_prob
                    + (self.max_forward_prob - self.min_forward_prob) * align
                )
                if abs(err) > self.turn_tol and random.random() > fprob:
                    action = self.CMD_TURN_RIGHT if err > 0 else self.CMD_TURN_LEFT
                else:
                    action = self.CMD_FORWARD

                # d) Step & save
                obs, _, term, trunc, _ = self.env.step(action)
                self.actions.append(action)
                self.obs.append(obs)
                if term or trunc:
                    print("[DEBUG] Episode terminated.")
                    return

        print("[DEBUG] Completed smoothed path.")

    def pick_up():
        """
        Pick up an object at the current agent position.
        """
        pass

    def drop():
        """
        Drop an object at the current agent position.
        """
        pass

    def toogle():
        """
        Toggle the state of an object at the current agent position.
        """
        pass

    def check_surroundings():
        """
        Turn around the agent.
        """
        pass

    def clear_way():
        """
        Clear the way for the agent.
        """
        pass
