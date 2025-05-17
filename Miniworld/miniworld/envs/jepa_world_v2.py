import random
import math
from gymnasium import spaces, utils
from miniworld.entity import Box, MeshEnt
from miniworld.miniworld import MiniWorldEnv, Texture
import pprint
# from OpenGL.GL import glBegin, glEnd, glNormal3f, glTexCoord2f, glVertex3f, glColor3f, glDisable, glEnable, GL_POLYGON, GL_QUADS, GL_TEXTURE_2D
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

    COLOR_TEXTURES = {
        "matte_white": (1.0, 1.0, 1.0),
        "matte_grey": (0.6, 0.6, 0.6),
        "light_grey": (0.8, 0.8, 0.8),
        "warm_white": (1.0, 0.98, 0.94),
        "off_white": (0.95, 0.95, 0.95),
        "light_wood": (0.87, 0.72, 0.53),
        "dark_grey": (0.4, 0.4, 0.4),
        "sage_green": (0.74, 0.82, 0.76),
        "dusty_blue": (0.65, 0.76, 0.89),
        "soft_beige": (0.96, 0.89, 0.76),
        "terracotta": (0.89, 0.52, 0.36),
        "muted_olive": (0.72, 0.75, 0.58),
        "powder_pink": (0.96, 0.80, 0.82),
        "midnight_blue": (0.30, 0.34, 0.48),
    }

    ROOM_SIZE_RANGES = {
        "living_room": ((4.66, 5.49), (7.71, 9.53)),
        "kitchen": ((1.52, 3.04), (3.65, 6.09)),
        "bedroom": ((3.04, 3.04), (4.27, 5.49)),
        "study_room": ((3.04, 3.04), (4.27, 4.87)),
        "children_room": ((3.05, 3.05), (3.05, 4.57)),
        "office": ((3.04, 3.65), (4.27, 4.87)),
        "bathroom": ((0.90, 1.50), (4.00, 3.70)),
        "hallway": ((1.00, 4.60), (1.20, 6.10)),
    }

    FURNITURE = {
        "living_room": {
            "couches": [
                "sofa_01/sofa_01",
                "sofa_02/sofa_02",
            ],
            "lamps": [
                "lamp_01/lamp_01"],
            "chairs": [
                "armchair_01/armchair_01",
                "armchair_02/armchair_02",
            ],
            "tables": [
                "table_living_01/table_living_01",
                "table_living_02/table_living_02",
            ],
            "consoles": [
                "console_tv_01/console_tv_01",
                "console_tv_02/console_tv_02",
            ],
            "sideboards": [
                "sideboard_01/sideboard_01",
                "sideboard_02/sideboard_02",
            ],
            "coffee_tables": ["table_01/table_01"],
            "tvs": [
                "tv_01/tv_01",
            ],
            "movables": ["duckie","chips_01/chips_01","handy_01/handy_01","keys_01/keys_01","dish_01/dish_01","towl_01/towl_01"
            ]
        }
    }
    COUCH_DIR_OFFSETS = {
    "armchair_01/armchair_01": np.deg2rad(0),
     }

    def __init__(self, seed=50, **kwargs):
        self.seed_value = seed
        self.rng = random.Random(seed)
        super().__init__(max_episode_steps=500, **kwargs)
        utils.EzPickle.__init__(self, seed=seed, **kwargs)
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)

    def _random_dim(self, size_range):
        """
        Returns a random (width, depth) on the ground plane,
        drawn independently:
        width  ∈ [min_w, max_w]
        depth  ∈ [min_d, max_d]
        where size_range == ((min_w, min_d), (max_w, max_d)).
        """
        (min_w, min_d), (max_w, max_d) = size_range
        w = self.rng.uniform(min_w, max_w)
        d = self.rng.uniform(min_d, max_d)
        return w, d

    def _add_room(self, name, x, z, w, h, rooms, room_positions):
        # choose solid-color keys
        wall_key  = self.rng.choice(list(self.COLOR_TEXTURES.values()))
        floor_key = self.rng.choice(list(self.COLOR_TEXTURES.values()))
        ceil_key  = self.rng.choice(list(self.COLOR_TEXTURES.values()))

        room = self.add_rect_room(
            min_x=x,
            max_x=x + w,
            min_z=z,
            max_z=z + h,
            wall_tex=wall_key,
            floor_tex=floor_key,
            ceil_tex=ceil_key,
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
        _, hd = self._random_dim(self.ROOM_SIZE_RANGES["hallway"])
        lx0, lx1 = living.min_x, living.max_x
        lz0, lz1 = living.min_z, living.max_z
        p_hw = 0.75  # portal half-width

        if wall_idx == 0:  # east wall
            hx = lx1 + 1.0
            hz = lz0 + (lz1 - lz0 - hd) / 2
            hallway = self._add_room("hallway", hx, hz, hw, hd, rooms, room_positions)
            center_z = hz + hd / 2
            self.connect_rooms(
                living, hallway, min_z=center_z - p_hw, max_z=center_z + p_hw, max_y=2.2
            )

        elif wall_idx == 2:  # west wall
            hx = lx0 - hw - 1.0
            hz = lz0 + (lz1 - lz0 - hd) / 2
            hallway = self._add_room("hallway", hx, hz, hw, hd, rooms, room_positions)
            center_z = hz + hd / 2
            self.connect_rooms(
                living, hallway, min_z=center_z - p_hw, max_z=center_z + p_hw, max_y=2.2
            )

        elif wall_idx == 1:  # south wall
            hx = lx0 + (lx1 - lx0 - hw) / 2
            hz = lz0 - hd - 1.0
            hallway = self._add_room("hallway", hx, hz, hw, hd, rooms, room_positions)
            center_x = hx + hw / 2
            self.connect_rooms(
                living, hallway, min_x=center_x - p_hw, max_x=center_x + p_hw, max_y=2.2
            )

        else:  # north wall
            hx = lx0 + (lx1 - lx0 - hw) / 2
            hz = lz1 + 1.0
            hallway = self._add_room("hallway", hx, hz, hw, hd, rooms, room_positions)
            center_x = hx + hw / 2
            self.connect_rooms(
                living, hallway, min_x=center_x - p_hw, max_x=center_x + p_hw, max_y=2.2
            )

        return hallway

    def _attach_room(self, base_room, wall_idx, room_name, rooms, room_positions):
        """
        Attach a generic room of type `room_name` to `base_room` on `wall_idx`,
        carving a 1.5 m‐wide portal centered on that connection.
        """
        # random footprint
        w, d = self._random_dim(self.ROOM_SIZE_RANGES[room_name])
        lx0, lx1 = base_room.min_x, base_room.max_x
        lz0, lz1 = base_room.min_z, base_room.max_z
        p_hw = 0.6  # portal half-width

        if wall_idx == 0:  # east
            hx = lx1 + 1.0
            hz = lz0 + (lz1 - lz0 - d) / 2
        elif wall_idx == 2:  # west
            hx = lx0 - w - 1.0
            hz = lz0 + (lz1 - lz0 - d) / 2
        elif wall_idx == 1:  # south
            hx = lx0 + (lx1 - lx0 - w) / 2
            hz = lz0 - d - 1.0
        else:  # north
            hx = lx0 + (lx1 - lx0 - w) / 2
            hz = lz1 + 1.0

        room = self._add_room(room_name, hx, hz, w, d, rooms, room_positions)

        # carve portal
        if wall_idx in (0, 2):  # east or west
            center_z = hz + d / 2
            self.connect_rooms(
                base_room, room, min_z=center_z - p_hw, max_z=center_z + p_hw, max_y=2.2
            )
        else:  # south or north
            center_x = hx + w / 2
            self.connect_rooms(
                base_room, room, min_x=center_x - p_hw, max_x=center_x + p_hw, max_y=2.2
            )

        return room

    def _furnish_living_room(self, living, living_wall, second_wall):
        """Place couch, lamp, sideboard+TV, two chairs and coffee table in `living`, using center-of-wall positioning."""
        cfg = self.FURNITURE["living_room"]

        # determine free walls
        used_walls = {living_wall, second_wall}
        free_walls = [w for w in range(living.num_walls) if w not in used_walls]

        # compute room center
        cx = 0.5 * (living.min_x + living.max_x)
        cz = 0.5 * (living.min_z + living.max_z)

        # 1) Couch
        couch_wall = self.rng.choice(free_walls)
        free_walls.remove(couch_wall)
        couch_mesh = self.rng.choice(cfg["couches"])
        couch_ent  = MeshEnt(mesh_name=couch_mesh, height=1.0)

        # compute orientation and center-of-wall position
        if couch_wall == 0:  # east wall → face west
            dir_couch = -math.pi/2
            pos_x     = living.max_x - 0.7
            pos_z     = cz
        elif couch_wall == 2:  # west wall → face east
            dir_couch = math.pi/2
            pos_x     = living.min_x + 0.7
            pos_z     = cz
        elif couch_wall == 1:  # south wall → face north
            dir_couch = 0.0
            pos_x     = cx
            pos_z     = living.min_z + 0.7
        else:  # north wall → face south
            dir_couch = math.pi
            pos_x     = cx
            pos_z     = living.max_z - 0.7
        
        # place couch at computed pos
        self.place_entity(
            couch_ent,
            room=living,
            pos=(pos_x, 0.0, pos_z),
            dir=dir_couch
        )


        # 2) Coffee table flush against same wall as couch, centered along wall
        table_mesh   = self.rng.choice(cfg["coffee_tables"])
        table_ent    = MeshEnt(mesh_name=table_mesh, height=0.5)
        table_offset = 2.5  # distance from wall

        # if couch against east/west, shift in x and center z
        if couch_wall in (0, 2):
            pos_x = (living.max_x - table_offset) if couch_wall == 0 else (living.min_x + table_offset)
            pos_z = cz
        else:
            # if couch against south/north, shift in z and center x
            pos_x = cx
            pos_z = (living.min_z + table_offset) if couch_wall == 1 else (living.max_z - table_offset)

        self.place_entity(
            table_ent,
            room=living,
            pos=(pos_x, 0.0, pos_z),
            dir=dir_couch
        )

        

                        
                
        # 3) Two chairs with individual wall, lateral and rotation offsets, placing randomly one or both
        chair_meshes = cfg["chairs"]
        # define unique offsets and directions for left (-1) and right (+1) chairs
        offsets = {
            -1: {"wall_offset": 2.1, "lateral_shift": 1.8, "direction": np.deg2rad(-85)},  # left chair
            1: {"wall_offset": 2.2, "lateral_shift": 1.4, "direction": np.deg2rad(80)}   # right chair
        }
        # determine wall coordinate and anchor axis
        if couch_wall == 0:      # east wall
            wall_coord = living.max_x
            axis = 'z'
        elif couch_wall == 2:    # west wall
            wall_coord = living.min_x
            axis = 'z'
        elif couch_wall == 1:    # south wall
            wall_coord = living.min_z
            axis = 'x'
        else:                    # north wall
            wall_coord = living.max_z
            axis = 'x'

        # randomly decide to place left, right, or both chairs
        options = [[-1], [1], [-1, 1]]
        selected = self.rng.choice(options)

        for factor in selected:  # place chairs based on selection
            chair_mesh = self.rng.choice(chair_meshes)
            chair_ent  = MeshEnt(mesh_name=chair_mesh, height=1.0)
            offs       = offsets[factor]
            wall_off   = offs['wall_offset']
            lat_off    = offs['lateral_shift']
            dir_chair = dir_couch - factor * (math.pi/2)
            # compute position for each chair
            if axis == 'z':
                pos_x = wall_coord - wall_off if couch_wall == 0 else wall_coord + wall_off
                pos_z = cz + factor * lat_off
                # if factor == -1:
                #     # left chair
                #     dir_chair = dir_couch + math.pi/2
                # else:
                #     # right chair
                #     dir_chair = dir_couch - math.pi/2


            else:
                pos_z = wall_coord + wall_off if couch_wall == 1 else wall_coord - wall_off
                pos_x = cx + factor * lat_off
                # to get “90° counter-clockwise” (i.e. –90°):
                # if factor == -1:
                #     # left chair
                #     dir_chair = dir_couch - math.pi/2
                # else:
                #     # right chair
                #     dir_chair = dir_couch + math.pi/2

            # apply individual rotation offset from couch direction
            #dir_chair = dir_couch + rot_offset

            # place chair
            self.place_entity(
                chair_ent,
                room=living,
                pos=(pos_x, 0.0, pos_z),
                dir=dir_chair
            )

            # 4) Lamp in a random corner of the couch wall
        lamp_mesh    = self.rng.choice(cfg["lamps"])
        lamp_ent     = MeshEnt(mesh_name=lamp_mesh, height=1.5)
        lamp_offset  = 0.5
        corner_off   = 0.9
        corner_factor = self.rng.choice([-1, 1])  # -1 = lower/left, +1 = upper/right

        if couch_wall in (0, 2):
            lamp_x = living.max_x - lamp_offset if couch_wall == 0 else living.min_x + lamp_offset
            lamp_z = living.min_z + corner_off if corner_factor == -1 else living.max_z - corner_off
        else:
            lamp_z = living.min_z + lamp_offset if couch_wall == 1 else living.max_z - lamp_offset
            lamp_x = living.min_x + corner_off if corner_factor == -1 else living.max_x - corner_off

        self.place_entity(
            lamp_ent,
            room=living,
            pos=(lamp_x, 0.0, lamp_z),
            dir= 0.0
        )

                    # 5) TV in the left corner on the wall opposite the couch
        tv_wall = (couch_wall + 2) % living.num_walls
        tv_mesh = self.rng.choice(cfg.get("tvs", cfg.get("sideboards", [])))
        tv_ent  = MeshEnt(mesh_name=tv_mesh, height=1.0)
        tv_offset = 0.5
        corner_off = 1.25  # offset from wall corner
        corner_factor = 1  # left corner

        # compute TV position based on tv_wall (same as before)
        if tv_wall == 0:  # east wall
            tv_x = living.max_x - tv_offset
            tv_z = living.min_z + (corner_off if corner_factor == -1 else living.max_z - corner_off)
        elif tv_wall == 2:  # west wall
            tv_x = living.min_x + tv_offset
            tv_z = living.min_z + (corner_off if corner_factor == -1 else living.max_z - corner_off)
        elif tv_wall == 1:  # south wall
            tv_x = living.min_x + (corner_off if corner_factor == -1 else living.max_x - corner_off)
            tv_z = living.min_z + tv_offset
        else:  # north wall
            tv_x = living.min_x + (corner_off if corner_factor == -1 else living.max_x - corner_off)
            tv_z = living.max_z - tv_offset

        # place TV facing opposite direction of couch
        tv_dir = dir_couch + math.pi
        self.place_entity(
            tv_ent,
            room=living,
            pos=(tv_x, 0.0, tv_z),
            dir=tv_dir
        )
        # movables ents 
        def spot_is_free(x, z, r, entities):
            """
            Return True if no existing entity in `entities`
            overlaps the circle at (x,z) with radius r.
            """
            for ent in entities:
                ex, _, ez = ent.pos
                # sum of radii
                min_dist = r + ent.radius
                # squared distance
                if (x - ex)**2 + (z - ez)**2 < min_dist**2:
                    return False
            return True

        # … inside your room-gen method …

        movables  = cfg["movables"]
        count     = min(3, len(movables))
        chosen    = self.rng.sample(movables, count)
        margin    = 0.5
        max_tries = 20

        for mesh_name in chosen:
            # create the entity to place
            ent    = MeshEnt(mesh_name=mesh_name, height=0.2, static=False)
            radius = ent.radius  # use its computed bounding radius

            for attempt in range(max_tries):
                # sample a random spot
                x = self.rng.uniform(living.min_x + margin, living.max_x - margin)
                z = self.rng.uniform(living.min_z + margin, living.max_z - margin)

                # check against every placed entity
                if not spot_is_free(x, z, radius, self.entities):
                    continue  # collision, retry

                # free spot found → place it
                dir_movable = self.rng.uniform(0, 2 * math.pi)
                self.place_entity(
                    ent,
                    room=living,
                    pos=(x, 0.0, z),
                    dir=dir_movable
                )
                break
            else:
                # fallback after too many tries:
                # either skip or place anyway with a warning
                print(f"Warning: no free spot for {mesh_name}, placing anyway")
                x = max(min(x, living.max_x - margin), living.min_x + margin)
                z = max(min(z, living.max_z - margin), living.min_z + margin)
                self.place_entity(
                    ent,
                    room=living,
                    pos=(x, 0.0, z),
                    dir=self.rng.uniform(0, 2 * math.pi)
                )




    

    def create_flat(self):
        rooms, room_positions = {}, []

        # 1) living room
        lw, ld = self._random_dim(self.ROOM_SIZE_RANGES["living_room"])
        living = self._add_room("living_room", 0, 0, lw, ld, rooms, room_positions)

        # 2) hallway on a random living‐room wall
        living_wall = self.rng.choice(range(living.num_walls))
        hallway = self._attach_hallway(living, living_wall, rooms, room_positions)

        # 3) attach second room to another wall
        to_place = [
            r for r in self.ROOM_SIZE_RANGES if r not in ("living_room", "hallway")
        ]
        second_wall = self.rng.choice(
            [w for w in range(living.num_walls) if w != living_wall]
        )
        first_name = self.rng.choice(to_place)
        self._attach_room(living, second_wall, first_name, rooms, room_positions)
        to_place.remove(first_name)

        # 4) attach remaining to hallway walls
        hall_used_wall = (living_wall + 2) % hallway.num_walls
        free_hall_walls = [w for w in range(hallway.num_walls) if w != hall_used_wall]
        self.rng.shuffle(to_place)
        for w_idx, name in zip(free_hall_walls, to_place):
            self._attach_room(hallway, w_idx, name, rooms, room_positions)

        # 5) furnish living room
        self._furnish_living_room(living, living_wall, second_wall)

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
        glNormal3f(0, 1, 0)
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
            glNormal3f(0, -1, 0)
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
        glColor3f(1.0, 1.0, 1.0)

    def _gen_world(self):
        # build rooms + portals
        self.create_flat()
        # place goal and agent
        #self.box = self.place_entity(Box(color="red"))
        self.place_agent()
