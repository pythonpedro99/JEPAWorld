import random
from typing import Dict, List, Optional, Tuple
import gymnasium as gym
import miniworld.envs
import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point, Polygon, box
from shapely import affinity
from scripts.helpers import plot_prm_graph
from policies.expert import ExpertPolicy
from policies.helpers import GraphData,Agent, Obstacle, Room, Portal
from gymnasium.envs.registration import register
register(
    id="OneRoomTest-v0",
    entry_point="miniworld.envs.oneroom_test:SeededOneRoom",
    max_episode_steps=500,
    #kwargs={"seed":0},   # any default kwargs your ctor needs;  random.randint(0, 2**31 - 1)
)


env = gym.make("OneRoomTest-v0", seed= random.randint(0, 2**31 - 1))
env.reset()
