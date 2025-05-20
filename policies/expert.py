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
from shapely.geometry import LineString, Point, Polygon, box
from shapely import affinity
from policies.helpers import find_path, smooth_path, get_lookahead_point
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
        agent_name
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
        self.agent_name = agent_name

        # Debugging
        self.waypoints = []
        self.smoothed_waypoints = []
        # Default values
        self.forward_step = 0.3
        self.turn_step = np.deg2rad(20)
        self.turn_tol = self.turn_step *0.75
        self.chaikins_iterations = 3
        self.min_forward_prob = 0.1  # minimum forward-if-misaligned probability
        self.max_forward_prob = 0.5  # maximum forward-if-misaligned probability
        self.CMD_TURN_RIGHT = 1
        self.CMD_TURN_LEFT = 0
        self.CMD_FORWARD = 2
        self.pause_prob = 0.1  # probability of pausing at each step
        self.waypoint_tolerance = 0.2  # tolerance for reaching waypoints
        self.lookahead_distance = 0.5  # distance to look ahead for the next waypoint

    def _obstacle_polygon(self, obs) -> Polygon:
        """Return shapely polygon of the obstacle."""
        w, d = obs.size
        poly = box(-w / 2, -d / 2, w / 2, d / 2)
        poly = affinity.rotate(poly, obs.yaw, use_radians=True)
        return affinity.translate(poly, obs.pos[0], obs.pos[1])

    def _save_episode(self) -> None:
        """Persist collected observations and actions."""
        if not self.obs:
            return
        save_data_batch(
            self.obs,
            self.actions,
            "/Users/julianquast/Documents/Bachelor Thesis/Datasets",
        )

    def solve_mission(self) -> None:
        """
        Execute self.mission, which should be a list of (action, target) tuples.
        - action: one of "go_to", "pick_up", "put_down"/"drop", "toggle"
        - target: for "go_to" either a node‐name (str) or an (x,y) tuple;
                    ignored (None) for other actions.
        """
        for action, target in self.mission:
            if action == "go_to":
                # target can be a node‐name or a raw (x,y) coordinate
                self.go_to(target)
            elif action == "pick_up":
                self.pick_up()
            elif action in ("put_down", "drop"):
                self.drop()
            elif action == "toggle":
                self.toggle()
            else:
                raise ValueError(f"[solve_mission] unknown action '{action}'")

        # finally, dump the collected observations & actions
        self._save_episode()
    
   

    def go_to(self, goal: str) -> None:

        # 2) Plan path (no smoothing)
        path = find_path(self.graph, self.nodes, self.agent_name, goal)
        self.path += path
        waypoints = [self.node_positions[node] for node in path]
        self.waypoints += waypoints

        if not waypoints:
            print(f"[DEBUG] No waypoints for goal '{goal}'.")
            return

        target_obs = next((o for o in self.obstacles if o.node_name == goal), None)
        agent_radius = getattr(self.env.unwrapped.agent, "radius", 0.2)
        target_poly = self._obstacle_polygon(target_obs) if target_obs else None
        target_buffer = agent_radius + 0.5
        if target_obs:
            print(
                f"[DEBUG] Target polygon size=({target_obs.size[0]:.2f},{target_obs.size[1]:.2f})"
            )

        for idx, (wx, wy) in enumerate(waypoints, start=1):
            print(f"\n=== New goal: ({wx:.2f}, {wy:.2f}) ===")
            is_last = idx == len(waypoints)
            buf = target_buffer if is_last else self.waypoint_tolerance
            while True:
                # current pose
                x, y    = self.env.unwrapped.agent.pos[0], self.env.unwrapped.agent.pos[2]
                yaw_rad = self.env.unwrapped.agent.dir
                # normalize yaw into [0,360)
                yaw_deg = (np.degrees(yaw_rad) + 360) % 360

                # vector to goal
                dx, dy = wx - x, wy - y
                dist = np.hypot(dx, dy)

                reached = False
                if is_last and target_poly is not None:
                    reached = target_poly.distance(Point(x, y)) <= target_buffer
                else:
                    reached = dist < buf

                if reached:
                    print(
                        f"[DEBUG] Reached goal: pos=({x:.2f},{y:.2f}) dist={dist:.3f}m\n"
                    )
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
                        self._save_episode()
                        return
                    continue  # re-read pose & re-evaluate

                # move forward
                obs, _, term, trunc, _ = self.env.step(2)
                self.obs.append(obs)
                self.actions.append(2)
                if term or trunc:
                    print("[DEBUG] Episode timeout.")
                    self._save_episode()
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
            self._save_episode()
        return

    def drop(self):
        """
        Drop an object at the current agent position.
        """
        obs, _, term, trunc, _ = self.env.step(5)
        self.obs.append(obs)
        self.actions.append(5)
        if term or trunc:
            print("[DEBUG] Episode timeout.")
            self._save_episode()
        return

    def toogle(self):
        """
        Toggle the state of an object at the current agent position.
        """
        obs, _, term, trunc, _ = self.env.step(6)
        self.obs.append(obs)
        self.actions.append(6)
        if term or trunc:
            print("[DEBUG] Episode timeout.")
            self._save_episode()
        return

    def check_surroundings(self):
        """
        Turn around the agent.
        """
        pass

    def clear_way(self):
        """
        Clear the way for the agent.
        """
        pass
