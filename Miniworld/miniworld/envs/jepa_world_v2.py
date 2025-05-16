import random
from gymnasium import spaces, utils
from miniworld.entity import Box, MeshEnt
from miniworld.miniworld import MiniWorldEnv, Texture
#from OpenGL.GL import glBegin, glEnd, glNormal3f, glTexCoord2f, glVertex3f, glColor3f, glDisable, glEnable, GL_POLYGON, GL_QUADS, GL_TEXTURE_2D
import numpy as np
from miniworld.miniworld import Y_VEC, gen_texcs_floor, gen_texcs_wall
from miniworld.miniworld import Y_VEC, gen_texcs_floor, gen_texcs_wall
from pyglet.gl import (
    GL_POLYGON,
    GL_QUADS,
    GL_TEXTURE_2D,
    glBegin,
    glColor3f,
    glDisable,
    glEnable,
    glEnd,
    glNormal3f,
    glTexCoord2f,
    glVertex3f,
)



class JEPAWorld(MiniWorldEnv, utils.EzPickle):
    """
    A procedurally-generated flat with living room, hallway(s),
    kitchen, office, bedroom, and bathroom—all rooms 3m deep
    and rendered as solid matte colors from COLOR_TEXTURES.
    """

    ROOM_SIZE_RANGES = {
        "living_room": (10, 14),
        "kitchen":     (8, 12),
        "bedroom":     (8, 14),
        "office":      (8, 16),
        "bathroom":    (10, 12),
        "hallway":     (15, 20),
    }

    COLOR_TEXTURES = {
        "matte_white":  (1.0, 1.0, 1.0),
        "matte_grey":   (0.6, 0.6, 0.6),
        "light_grey":   (0.8, 0.8, 0.8),
        "warm_white":   (1.0, 0.98, 0.94),
        "off_white":    (0.95, 0.95, 0.95),
        "light_wood":   (0.87, 0.72, 0.53),
        "dark_grey":    (0.4, 0.4, 0.4),
        "sage_green":   (0.74, 0.82, 0.76),
        "dusty_blue":   (0.65, 0.76, 0.89),
        "soft_beige":   (0.96, 0.89, 0.76),
        "terracotta":   (0.89, 0.52, 0.36),
        "muted_olive":  (0.72, 0.75, 0.58),
        "powder_pink":  (0.96, 0.80, 0.82),
        "midnight_blue":(0.30, 0.34, 0.48),
    }

    def __init__(self, seed=50, **kwargs):
        self.seed_value = seed
        self.rng = random.Random(seed)
        super().__init__(max_episode_steps=500, **kwargs)
        utils.EzPickle.__init__(self, seed=seed, **kwargs)
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)

    def _random_dim(self, min_size, max_size):
        """
        Returns a random (width, depth) on the ground plane,
        both uniformly drawn between min_size and max_size.
        """
        w = self.rng.uniform(min_size, max_size)
        d = self.rng.uniform(min_size, max_size)
        return w, d

    def _add_room(self, name, x, z, w, h, rooms, room_positions):
        # choose solid-color keys
        wall_key  = self.rng.choice(list(self.COLOR_TEXTURES))
        floor_key = self.rng.choice(list(self.COLOR_TEXTURES))
        ceil_key  = self.rng.choice(list(self.COLOR_TEXTURES))

        room = self.add_rect_room(
            min_x=x, max_x=x + w,
            min_z=z, max_z=z + h,
            wall_tex = wall_key,
            floor_tex= floor_key,
            ceil_tex = ceil_key
        )

        rooms[name] = room
        room_positions.append((name, x, z, w, h))
        return room
    
    def _attach_hallway(self, living, wall_idx, rooms, room_positions):
        """
        Attach a 2 m‐wide hallway to `living` on wall `wall_idx`,
        carving a 2 m‐wide portal centered on that hallway.
        """
        # Hallway footprint: width fixed 2 m, depth randomized
        hw = 2.0
        _, hd = self._random_dim(*self.ROOM_SIZE_RANGES["hallway"])

        # Living‐room bounds
        lx0, lx1 = living.min_x, living.max_x
        lz0, lz1 = living.min_z, living.max_z

        # Portal half‐width
        p_hw = 0.75

        if wall_idx == 0:   # east wall
            hx = lx1 + 1.0
            hz = lz0 + (lz1 - lz0 - hd) / 2
            hallway = self._add_room("hallway", hx, hz, hw, hd, rooms, room_positions)

            # center portal along Z
            center_z = hz + hd/2
            self.connect_rooms(
                living, hallway,
                min_z=center_z - p_hw, max_z=center_z + p_hw,
                max_y=2.2
            )

        elif wall_idx == 2: # west wall
            hx = lx0 - hw - 1.0
            hz = lz0 + (lz1 - lz0 - hd) / 2
            hallway = self._add_room("hallway", hx, hz, hw, hd, rooms, room_positions)

            center_z = hz + hd/2
            self.connect_rooms(
                living, hallway,
                min_z=center_z - p_hw, max_z=center_z + p_hw,
                max_y=2.2
            )

        elif wall_idx == 1: # south wall
            hx = lx0 + (lx1 - lx0 - hw) / 2
            hz = lz0 - hd - 1.0
            hallway = self._add_room("hallway", hx, hz, hw, hd, rooms, room_positions)

            # center portal along X
            center_x = hx + hw/2
            self.connect_rooms(
                living, hallway,
                min_x=center_x - p_hw, max_x=center_x + p_hw,
                max_y=2.2
            )

        else:               # north wall
            hx = lx0 + (lx1 - lx0 - hw) / 2
            hz = lz1 + 1.0
            hallway = self._add_room("hallway", hx, hz, hw, hd, rooms, room_positions)

            center_x = hx + hw/2
            self.connect_rooms(
                living, hallway,
                min_x=center_x - p_hw, max_x=center_x + p_hw,
                max_y=2.2
            )

        return hallway



    def create_flat(self):
        rooms, room_positions = {}, []

        # 1) living
        lw, ld = self._random_dim(*self.ROOM_SIZE_RANGES["living_room"])
        living = self._add_room("living_room", 0, 0, lw, ld, rooms, room_positions)

        # 2) hallway on random living-room wall
        wall_idx = self.rng.choice(range(living.num_walls))
        hallway = self._attach_hallway(living, wall_idx, rooms, room_positions)

        # (…then attach K/B/O and bedroom similarly, passing rooms & room_positions…)
        self.place_entity(
            MeshEnt(mesh_name="armchair_01/armchair_01", height=1.0),
            room=living,
            min_x=living.min_x + 0.2,
            max_x=living.min_x + 0.8,
            min_z=4.0,
            max_z=5.0,
        )

        return rooms, room_positions


    def _render(self):
        """
        Detect solid-color vs. texture and draw accordingly.
        """
        # --- Floor ---
        if isinstance(self.floor_tex, tuple):
            glDisable(GL_TEXTURE_2D)
            glColor3f(*self.floor_tex)
        else:
            glEnable(GL_TEXTURE_2D)
            self.floor_tex.bind()

        glBegin(GL_POLYGON)
        glNormal3f(0,1,0)
        for v, tc in zip(self.floor_verts, self.floor_texcs):
            glTexCoord2f(*tc)
            glVertex3f(*v)
        glEnd()

        # --- Ceiling ---
        if not self.no_ceiling:
            if isinstance(self.ceil_tex, tuple):
                glDisable(GL_TEXTURE_2D)
                glColor3f(*self.ceil_tex)
            else:
                glEnable(GL_TEXTURE_2D)
                self.ceil_tex.bind()

            glBegin(GL_POLYGON)
            glNormal3f(0,-1,0)
            for v, tc in zip(self.ceil_verts, self.ceil_texcs):
                glTexCoord2f(*tc)
                glVertex3f(*v)
            glEnd()

        # --- Walls ---
        if isinstance(self.wall_tex, tuple):
            glDisable(GL_TEXTURE_2D)
            glColor3f(*self.wall_tex)
        else:
            glEnable(GL_TEXTURE_2D)
            self.wall_tex.bind()

        glBegin(GL_QUADS)
        for vert, norm, tc in zip(self.wall_verts, self.wall_norms, self.wall_texcs):
            glNormal3f(*norm)
            glTexCoord2f(*tc)
            glVertex3f(*vert)
        glEnd()

        # restore default
        glEnable(GL_TEXTURE_2D)
        glColor3f(1.0,1.0,1.0)

    def _gen_world(self):
        # build rooms + portals
        self.create_flat()
        # place goal and agent
        self.box = self.place_entity(Box(color="red"))
        self.place_agent()
