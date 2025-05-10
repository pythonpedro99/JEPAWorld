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
from policies.helpers import find_path, smooth_path, get_lookahead_point
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
        self.turn_step = np.deg2rad(5)
        self.turn_tol = self.turn_step / 1.5
        self.chaikins_iterations = 3
        self.min_forward_prob = 0.1  # minimum forward-if-misaligned probability
        self.max_forward_prob = 0.5  # maximum forward-if-misaligned probability
        self.CMD_TURN_RIGHT = 1
        self.CMD_TURN_LEFT = 0
        self.CMD_FORWARD = 2
        self.pause_prob = 0.1  # probability of pausing at each step
        self.waypoint_tolerance = 0.2  # tolerance for reaching waypoints
        self.lookahead_distance = 0.5  # distance to look ahead for the next waypoint

    def solve_mission(self) -> None:
        self.go_to(self.mission[0])
        self.pick_up()
        save_data_batch(self.obs, self.actions, "/Users/julianquast/Documents/Bachelor Thesis/Datasets")
        return None
        # TODO: Implement the rest of the mission logic
    
   

    def go_to(self, goal: str) -> None:

        # 2) Plan path (no smoothing)
        path = find_path(self.graph, self.nodes, "Agent_6", goal)
        self.path += path
        waypoints = [self.node_positions[node] for node in path]
        self.waypoints += waypoints

        if not waypoints:
            print(f"[DEBUG] No waypoints for goal '{goal}'.")
            return

        n_wp = len(waypoints)
        #print(f"[DEBUG] Moving to goal '{goal}' via {n_wp} waypoints (final buffer={target_buffer}).")
        target_obs = next((o for o in self.obstacles if o.node_name == goal), None)
        target_buffer = target_obs.radius +0.6
        print(f"[DEBUG] Target buffer: {target_buffer:.2f}m")

        for idx, (wx,wy)  in enumerate(waypoints, start=1):
            print(f"\n=== New goal: ({wx:.2f}, {wy:.2f}) ===")
            buf = target_buffer if idx == len(waypoints) else self.waypoint_tolerance 
            while True:
                # current pose
                x, y    = self.env.unwrapped.agent.pos[0], self.env.unwrapped.agent.pos[2]
                yaw_rad = self.env.unwrapped.agent.dir
                # normalize yaw into [0,360)
                yaw_deg = (np.degrees(yaw_rad) + 360) % 360

                # vector to goal
                dx, dy = wx - x, wy - y
                dist   = np.hypot(dx, dy)
                if dist < buf:
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
                    f"target=({wx:.2f},{wy:.2f})  "
                    f"dist={dist:.3f}m\n"
                    f"        desired={desired_deg:.1f}°  err={err_deg:.1f}°  "
                    f"(tol={np.degrees(self.turn_tol):.1f}°)")

                # turn-in-place
                if abs(err_rad) > self.turn_tol:
                    cmd = 0 if err_rad > 0 else 1
                    obs, _, term, trunc, _ = self.env.step(cmd)
                    self.obs.append(obs)
                    self.actions.append(cmd)
                    if term or trunc:
                        print("[DEBUG] Episode timeout.")
                        return
                    continue  # re-read pose & re-evaluate

                # move forward
                obs, _, term, trunc, _ = self.env.step(2)
                self.obs.append(obs)
                self.actions.append(cmd)
                if term or trunc:
                    print("[DEBUG] Episode timeout.")
                    return
                # loop back to re-fetch pose

        
        print("reached goal")





    def pick_up(self)-> None:
        """
        Pick up an object at the current agent position.
        """
        obs, _, term, trunc, _ = self.env.step(4)
        self.obs.append(obs)
        self.actions.append(4)
        if term or trunc:
            print("[DEBUG] Episode timeout.")
        return

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
