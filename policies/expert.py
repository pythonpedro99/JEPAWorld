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
        dataset_dir: str | None = None,
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
        self.dataset_dir = dataset_dir

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

    def _save_episode(self) -> int:
        """Persist collected observations and actions and reset buffers."""
        if not self.obs:
            return 0
        out_dir = self.dataset_dir or "/tmp/jepa_dataset"
        save_data_batch(
            self.obs,
            self.actions,
            out_dir,
            csv_name="actions.csv",
        )
        count = len(self.obs)
        self.obs = []
        self.actions = []
        return count

    def solve_mission(self) -> bool:
        """
        Execute self.mission, which should be a list of (action, target) tuples.
        - action: one of "go_to", "pick_up", "put_down"/"drop", "toggle"
        - target: for "go_to" either a node‐name (str) or an (x,y) tuple;
                    ignored (None) for other actions.
        """
        success = True
        for action, target in self.mission:
            if action == "go_to":
                ok = self.go_to(target)
            elif action == "pick_up":
                ok = self.pick_up()
            elif action in ("drop"):
                ok = self.drop()
            elif action == "toggle":
                ok = self.toggle()
            else:
                raise ValueError(f"[solve_mission] unknown action '{action}'")
            if not ok:
                success = False
                break

        # finally, dump the collected observations & actions
        self._save_episode()
        return success
   

    def go_to(self, goal: str) -> bool:
        # 1) Plan path (no smoothing)
        # Determine the nearest graph node to the agent's current position
        agent_pos = (
            float(self.env.unwrapped.agent.pos[0]),
            float(self.env.unwrapped.agent.pos[2]),
        )
        path = find_path(
            self.graph,
            self.nodes,
            agent_pos,
            goal,
            node_positions=self.node_positions,
        )

        if path is None:
            print(f"[DEBUG] No path found for goal '{goal}'.")
            return False

        self.path += path
        waypoints = [self.node_positions[node] for node in path]
        self.waypoints += waypoints

        # 2) Early exit if no path
        if not waypoints:
            print(f"[DEBUG] No waypoints for goal '{goal}'.")
            return False

        # 3) Compute stopping buffer for the final waypoint
        agent_radius = getattr(self.env.unwrapped.agent, "radius", 0.2)
        target_buffer = agent_radius + 0.8

        # 4) Iterate through each waypoint
        for i, (wx, wy) in enumerate(waypoints):
            print(f"\n=== New goal: ({wx:.2f}, {wy:.2f}) ===")
            is_last = (i == len(waypoints) - 1)
            buf = target_buffer if is_last else self.waypoint_tolerance

            # Track consecutive no-move attempts
            no_move_count = 0
            #self.scan_room(sweep_steps=3)

            while True:
                # 5) Read current pose
                x, y = self.env.unwrapped.agent.pos[0], self.env.unwrapped.agent.pos[2]
                yaw_rad = self.env.unwrapped.agent.dir
                yaw_deg = (np.degrees(yaw_rad) + 360) % 360

                # 6) Compute distance to current waypoint
                dx, dy = wx - x, wy - y
                dist = np.hypot(dx, dy)

                # 7) Debug print
                print(f"[DEBUG] pos=({x:.2f},{y:.2f})  yaw={yaw_deg:.1f}°  "
                    f"target=({wx:.2f},{wy:.2f})  dist={dist:.3f}m  buf={buf:.3f}m")

                # 8) Check if reached
                if dist <= buf:
                    print(f"[DEBUG] Reached goal: pos=({x:.2f},{y:.2f}) dist={dist:.3f}m\n")
                    break

                # 9) Compute desired heading & error
                desired_rad = np.arctan2(-dy, dx)
                desired_deg = (np.degrees(desired_rad) + 360) % 360
                err_rad = ((desired_rad - yaw_rad + np.pi) % (2*np.pi)) - np.pi
                err_deg = ((desired_deg - yaw_deg + 180) % 360) - 180

                # 10) Turn-in-place if needed
                if abs(err_rad) > self.turn_tol:
                    cmd = 0 if err_rad > 0 else 1
                    obs, _, term, trunc, _ = self.env.step(cmd)
                    self.obs.append(obs)
                    self.actions.append(cmd)
                    if term or trunc:
                        print("[DEBUG] Episode timeout.")
                        self._save_episode()
                        return False
                    continue

                # 11) Move forward with no-move detection
                old_x, old_y = x, y
                obs, _, term, trunc, _ = self.env.step(2)
                self.obs.append(obs)
                self.actions.append(2)
                if term or trunc:
                    print("[DEBUG] Episode timeout.")
                    self._save_episode()
                    return False

                # 12) Check if movement occurred
                new_x, new_y = self.env.unwrapped.agent.pos[0], self.env.unwrapped.agent.pos[2]
                if np.hypot(new_x - old_x, new_y - old_y) < 1e-3:
                    no_move_count += 1
                    print(f"[DEBUG] No movement detected (count={no_move_count}).")
                    if no_move_count >= 5:
                        print("[DEBUG] Stuck: aborting go_to.")
                        return False
                else:
                    no_move_count = 0
                # loop back to re-fetch pose

        print("reached goal")
        return True
    def scan_room(self, sweep_steps: int = 5) -> None:
        """
        Perform a left-right-left rotation to survey the room interior.
        sweep_steps: number of turn-in-place actions per half-sweep
        """
        import numpy as np

        def yaw_deg():
            return (np.degrees(self.env.unwrapped.agent.dir) + 360) % 360

        print("[DEBUG] Starting room scan: current yaw=%.1f°" % yaw_deg())

        # 1) Sweep left from center
        for i in range(sweep_steps):
            obs, _, term, trunc, _ = self.env.step(0)  # cmd=0: turn left
            self.obs.append(obs)
            self.actions.append(0)
            print(f"[SCAN] Left step {i+1}/{sweep_steps}, yaw={yaw_deg():.1f}°")
            if term or trunc:
                print("[DEBUG] Episode terminated during scan.")
                return

        # 2) Sweep right through center to rightmost
        for i in range(2 * sweep_steps):
            obs, _, term, trunc, _ = self.env.step(1)  # cmd=1: turn right
            self.obs.append(obs)
            self.actions.append(1)
            print(f"[SCAN] Right step {i+1}/{2*sweep_steps}, yaw={yaw_deg():.1f}°")
            if term or trunc:
                print("[DEBUG] Episode terminated during scan.")
                return

        # 3) Return to center
        for i in range(sweep_steps):
            obs, _, term, trunc, _ = self.env.step(0)  # cmd=0: turn left
            self.obs.append(obs)
            self.actions.append(0)
            print(f"[SCAN] Return left step {i+1}/{sweep_steps}, yaw={yaw_deg():.1f}°")
            if term or trunc:
                print("[DEBUG] Episode terminated during scan.")
                return

        print("[DEBUG] Room scan complete: final yaw=%.1f°" % yaw_deg())




    def pick_up(self) -> bool:
        """
        Pick up an object at the current agent position.
        """
        obs, _, term, trunc, _ = self.env.step(4)
        print("picked up moveable")
        self.obs.append(obs)
        self.actions.append(4)
        if term or trunc:
            print("[DEBUG] Episode timeout.")
            self._save_episode()
            return False
        return True  #self.env.unwrapped.agent.carrying is not None

    def drop(self) -> bool:
        """
        Drop an object at the current agent position.
        """
        obs, _, term, trunc, _ = self.env.step(5)
        self.obs.append(obs)
        self.actions.append(5)
        if term or trunc:
            print("[DEBUG] Episode timeout.")
            self._save_episode()
            return False
        return self.env.unwrapped.agent.carrying is None

    def toogle(self) -> bool:
        """
        Toggle the state of an object at the current agent position.
        """
        obs, _, term, trunc, _ = self.env.step(6)
        self.obs.append(obs)
        self.actions.append(6)
        if term or trunc:
            print("[DEBUG] Episode timeout.")
            self._save_episode()
            return False
        return True

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
