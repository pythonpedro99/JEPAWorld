from gymnasium import utils
import random
import colorsys
from miniworld.entity import Box, Ball, Key, COLOR_NAMES
from miniworld.miniworld import MiniWorldEnv
from miniworld.entity import TextFrame
import math
import numpy as np 


class JEPAENV(MiniWorldEnv, utils.EzPickle):
    """
    ## Description

    Single-room environment with randomized wall, floor, ceiling colors,
    and randomly placed objects (balls, boxes, keys) that are visually distinguishable.
    """

    def __init__(self, size=10, seed=0, max_entities=5, **kwargs):
        assert size >= 2
        self.size = size
        self.max_entities = max_entities
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        MiniWorldEnv.__init__(self, max_episode_steps=500, **kwargs)
        utils.EzPickle.__init__(self, size, seed, max_entities, **kwargs)

    def random_contrasting_rgb_triplet(self):
        h = self.rng.random()
        s = 0.2 + self.rng.random() * 0.4  # [0.2 – 0.6]
        v = 0.4 + self.rng.random() * 0.3  # [0.4 – 0.7]

        floor_hsv = (h, s, v)
        wall_hsv = ((h + 0.08) % 1.0, s, min(v + 0.1, 0.9))
        ceil_hsv = ((h + 0.16) % 1.0, s * 0.8, max(v + 0.15, 0.75))

        floor_rgb = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(*floor_hsv))
        wall_rgb = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(*wall_hsv))
        ceil_rgb = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(*ceil_hsv))

        return wall_rgb, floor_rgb, ceil_rgb


    def _gen_world(self):
        # Generate random colors for walls, floor, and ceiling
        wall_color, floor_color, ceil_color = self.random_contrasting_rgb_triplet()

        # Create a rectangular room with the specified colors
        self.add_rect_room(
            min_x=0,
            max_x=self.size,
            min_z=0,
            max_z=self.size,
            wall_tex=None,
            floor_tex=None,
            ceil_tex=None,
            wall_color=wall_color,
            floor_color=floor_color,
            ceil_color=ceil_color
        )

        # Place the agent inside the room
        offset = 0.5
        min_x, max_x = 0.0, self.size
        min_z, max_z = 0.0, self.size
        corners = [
            (min_x + offset, min_z + offset),
            (min_x + offset, max_z - offset),
            (max_x - offset, min_z + offset),
            (max_x - offset, max_z - offset),
        ]

        # pick one at random
        cx, cz = self.rng.choice(corners)

        # compute yaw so the agent faces the room center
        center_x = (min_x + max_x) / 2
        center_z = (min_z + max_z) / 2
        dx = center_x - cx
        dz = center_z - cz
        yaw = math.atan2(-dz, dx)

        # place the agent at (cx, cz), keeping its default height, and set its direction
        self.place_agent(
            pos=(cx, 0, cz),
            dir=yaw,
            min_x=min_x,
            max_x=max_x,
            min_z=min_z,
            max_z=max_z,
        )

        # Possible entity classes
        entity_classes = [Box, Ball, Key]

        # Generate distinct entity colors (RGB) with contrast to background
        

        # Place entities
        for i in range(self.max_entities):
            entity_cls = self.rng.choice(entity_classes)
            size = self.rng.uniform(0.4, 0.85)
            named_color = self.rng.choice(COLOR_NAMES)

            if entity_cls == Key:
                # Keys only accept named colors
                
                entity = Key(color=named_color)
            else:
                entity = entity_cls(color=named_color, size=size)

            self.place_entity(entity)

        from miniworld.entity import TextFrame

        # After room creation and agent placement
        frame_height = 1.0
        frame_depth = 0.05
        text_y = 1.2  # vertical placement on the wall

        wall_labels = [
    ("North", (self.size / 2, text_y, self.size - 0.05), math.pi / 2),    # Blick +x
    ("South", (self.size / 2, text_y, 0.05), -math.pi / 2),               # Blick -x
    ("West",  (0.05, text_y, self.size / 2), 0),                          # Blick +z
    ("East",  (self.size - 0.05, text_y, self.size / 2), -math.pi),       # Blick -z
]



        for label, pos, dir_angle in wall_labels:
            frame = TextFrame(pos=pos, dir=dir_angle, str=label, height=frame_height, depth=frame_depth)
            frame.randomize(self.params, self.np_rng)  # or just: frame.randomize(None, self.rng)
            self.entities.append(frame)


    

