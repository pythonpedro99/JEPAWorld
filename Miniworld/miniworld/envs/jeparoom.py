from gymnasium import utils
import random
import colorsys
from miniworld.entity import Box, Ball, Key, COLOR_NAMES
from miniworld.miniworld import MiniWorldEnv, Agent
from miniworld.entity import TextFrame, ImageFrame
import math
from typing import Any, Dict, Optional, Tuple
from gymnasium.core import ObsType
import numpy as np 
import string


class RearrangeOneRoom(MiniWorldEnv, utils.EzPickle):
    """
    ## Description

    Single-room environment with randomized wall, floor, ceiling colors,
    and randomly placed objects (balls, boxes, keys) that are visually distinguishable.
    """

    def __init__(self, size=2, seed=0, max_entities=4, **kwargs):
        assert size >= 2
        self.size_a = 6
        self.size_b = 8
        self.max_entities = max_entities
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        MiniWorldEnv.__init__(self, max_episode_steps=500, **kwargs)
        utils.EzPickle.__init__(self, size, seed, max_entities, **kwargs)

    def random_contrasting_rgb_triplet(self):
        h = self.rng.random()
        s = 0.2 + self.rng.random() * 0.4
        v = 0.4 + self.rng.random() * 0.3

        floor_hsv = (h, s, v)
        wall_hsv = ((h + 0.08) % 1.0, s, min(v + 0.1, 0.9))
        ceil_hsv = ((h + 0.16) % 1.0, s * 0.8, max(v + 0.15, 0.75))

        floor_rgb = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(*floor_hsv))
        wall_rgb = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(*wall_hsv))
        ceil_rgb = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(*ceil_hsv))

        return wall_rgb, floor_rgb, ceil_rgb

    def _gen_world(self):
        wall_color, floor_color, ceil_color = self.random_contrasting_rgb_triplet()

        self.add_rect_room(
            min_x=0,
            max_x=self.size_a,
            min_z=0,
            max_z=self.size_b,
            wall_tex=None,
            floor_tex=None,
            ceil_tex=None,
            wall_color=wall_color,
            floor_color=floor_color,
            ceil_color=ceil_color
        )

        offset_from_wall = 0.7
        min_x, max_x = 0.0, self.size_a
        min_z, max_z = 0.0, self.size_b

        agent_x = (min_x + max_x) / 2
        agent_z = min_z + offset_from_wall

        center_x = (min_x + max_x) / 2
        center_z = (min_z + max_z) / 2
        dx = center_x - agent_x
        dz = center_z - agent_z
        yaw = math.atan2(-dz, dx)

        self.place_agent(
            pos=(agent_x, 0, agent_z),
            dir=yaw,
            min_x=min_x, max_x=max_x,
            min_z=min_z, max_z=max_z,
        )

        entity_classes = [Box, Ball, Key]
        min_x, max_x = 0.5, self.size_a - 0.5
        min_z, max_z = 0.5, self.size_b - 0.5
        fixed_radius = 0.25
        min_sep = 2 * fixed_radius + 0.1

        positions = []
        while len(positions) < self.max_entities:
            x = self.rng.uniform(min_x, max_x)
            z = self.rng.uniform(min_z, max_z)
            if all(math.hypot(x - px, z - pz) >= min_sep for px, pz in positions):
                positions.append((x, z))

        for (x, z) in positions:
            yaw = self.rng.uniform(0.0, 2 * math.pi)
            entity_cls = self.rng.choice(entity_classes)
            size = self.rng.uniform(0.2, 0.5)
            named_color = self.rng.choice(COLOR_NAMES)

            if entity_cls is Key:
                entity = Key(color=named_color)
            else:
                entity = entity_cls(color=named_color, size=size)

            self.place_entity(entity, pos=(x, 0, z), dir=yaw)

        def rand_label():
            length = self.rng.randint(1, 5)
            return ''.join(self.rng.choices(string.ascii_uppercase, k=length))

        TEXT_HEIGHT, TEXT_DEPTH, TEXT_Y = 1.0, 0.75, 1.20
        PIC_WIDTH, PIC_Y = 1.8, 1.35

        walls_all = [
            ((self.size_a / 2, TEXT_Y, self.size_b - 0.05), math.pi / 2),
            ((self.size_a / 2, TEXT_Y, 0.05), -math.pi / 2),
            ((0.05, TEXT_Y, self.size_b / 2), 0),
            ((self.size_a - 0.05, TEXT_Y, self.size_b / 2), -math.pi),
        ]

        picture_walls = set(self.rng.sample(range(len(walls_all)), k=2))
        pictures = ["alessandro_allori", "edmund_blair_leighton", "logo_mila"]

        for idx, (base_pos, dir_ang) in enumerate(walls_all):
            if idx in picture_walls:
                pos = (base_pos[0], PIC_Y, base_pos[2])
                tex = self.rng.choice(pictures)
                ent = ImageFrame(pos=pos, dir=dir_ang, width=PIC_WIDTH, tex_name=tex)
            else:
                ent = TextFrame(
                    pos=base_pos,
                    dir=dir_ang,
                    str=rand_label(),
                    height=TEXT_HEIGHT,
                    depth=TEXT_DEPTH
                )

            self.entities.append(ent)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[ObsType, Dict[str, Any]]:
        # Call parent reset to set Gym and MiniWorld seeds properly
        super().reset(seed=seed)

        # Manual reseed of local RNGs
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self.step_count = 0
        self.agent = Agent()
        self.entities = []
        self.rooms = []
        self.wall_segs = []

        self._gen_world()

        rand = self.np_rng if self.domain_rand else None
        for ent in self.entities:
            ent.randomize(self.params, rand)

        self.min_x = min(r.min_x for r in self.rooms)
        self.max_x = max(r.max_x for r in self.rooms)
        self.min_z = min(r.min_z for r in self.rooms)
        self.max_z = max(r.max_z for r in self.rooms)

        if len(self.wall_segs) == 0:
            self._gen_static_data()

        self._render_static()

        obs = self.render_obs()
        return obs, {}