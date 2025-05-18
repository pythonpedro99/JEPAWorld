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
        "kitchen": ((4.52, 5.04), (4.65, 6.09)),
        "bedroom": ((4.04, 3.04), (4.27, 5.49)),
        "children_room": ((4.05, 4.05), (4.05, 4.57)),
        "bathroom": ((4.90, 4.50), (4.00, 4.70)),
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
        },
        "kitchen": {
            "kitchen": [
                "kitchen_01/kitchen_01"
            ],
            "fridge": [
                "fridge_01/fridge_01",
            ],
            "stove": [
                "stove_01/stove_01",
            ],
            "sink": [
                "sink_01/sink_01",
            ],
            "trash": [
                "trash_01/trash_01",
            ],
        },
        "bathroom": {
            "bath": [
                "bath_01/bath_01",
            ],
            "sink": ["sink_02/sink_02"],
            "toilet": [
                "toilet_01/toilet_01",
            ],
            "washer": ["washer_01/washer_01"],    
        },
        "bedroom": {"bed": ["bed_01/bed_01"]},
        
        }
    SCALE_FACTORS = {
    # living room
    "sofa_01/sofa_01":            1.5,
    "sofa_02/sofa_02":            1.25,
    "lamp_01/lamp_01":            0.8,
    "armchair_01/armchair_01":    0.9,
    "armchair_02/armchair_02":    1.1,
    "table_living_01/table_living_01": 1.0,
    "table_living_02/table_living_02": 1.0,
    "console_tv_01/console_tv_01":     1.0,
    "console_tv_02/console_tv_02":     1.0,
    "sideboard_01/sideboard_01":       1.0,
    "sideboard_02/sideboard_02":       1.0,
    "table_01/table_01":               0.5,
    "tv_01/tv_01":                     1.0,
    "duckie":                          0.2,
    "chips_01/chips_01":               0.2,
    "handy_01/handy_01":               0.5,
    "keys_01/keys_01":                 0.1,
    "dish_01/dish_01":                 0.06,
    "towl_01/towl_01":                 0.01,

    # kitchen
    "kitchen_01/kitchen_01":    1.0,
    "fridge_01/fridge_01":      1.9,
    "stove_01/stove_01":        0.9,
    "sink_01/sink_01":          0.6,
    "trash_01/trash_01":        0.5,
    "washer_01/wash_01":      0.8,
    "toilet_01/toilet_01":    1.2,

    #bathroom
    "bath_01/bath_01":        0.8,
    "sink_02/sink_02":        1.0,
    # bedroom   
    "bed_01/bed_01":          1.0,
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
        hw = 2.0  # hallway cross‐section width
        _, hd = self._random_dim(self.ROOM_SIZE_RANGES["hallway"])
        lx0, lx1 = living.min_x, living.max_x
        lz0, lz1 = living.min_z, living.max_z
        p_hw = 0.5  # portal half‐width

        if wall_idx == 0:  # east wall
            # 1) portal center on living’s east wall (z‐axis midpoint)
            center_z = (lz0 + lz1) / 2
            # 2) hallway spans z from center_z–hw/2 to center_z+hw/2
            hz = center_z - hw / 2
            #    and shoots out in +x for hd meters
            hx = lx1 + 0.3
            hallway = self._add_room("hallway", hx, hz, hd, hw, rooms, room_positions)
            # 3) carve portal exactly ±0.75 m around that living‐wall center
            self.connect_rooms(
                living, hallway,
                min_z = center_z - p_hw,
                max_z = center_z + p_hw,
                max_y = 2.2
            )

        elif wall_idx == 2:  # west wall
            center_z = (lz0 + lz1) / 2
            hz = center_z - hw / 2
            # shoots out in –x for hd meters
            hx = lx0 - hd - 0.3
            hallway = self._add_room("hallway", hx, hz, hd, hw, rooms, room_positions)
            self.connect_rooms(
                living, hallway,
                min_z = center_z - p_hw,
                max_z = center_z + p_hw,
                max_y = 2.2
            )

        elif wall_idx == 1:  # south wall (already perpendicular)
            center_x = (lx0 + lx1) / 2
            hx = center_x - hw / 2
            hz = lz0 - hd - 0.3
            hallway = self._add_room("hallway", hx, hz, hw, hd, rooms, room_positions)
            self.connect_rooms(
                living, hallway,
                min_x = center_x - p_hw,
                max_x = center_x + p_hw,
                max_y = 2.2
            )

        else:  # north wall
            center_x = (lx0 + lx1) / 2
            hx = center_x - hw / 2
            hz = lz1 + 0.3
            hallway = self._add_room("hallway", hx, hz, hw, hd, rooms, room_positions)
            self.connect_rooms(
                living, hallway,
                min_x = center_x - p_hw,
                max_x = center_x + p_hw,
                max_y = 2.2
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
        p_hw = 0.5  # portal half-width

        if wall_idx == 0:  # east
            hx = lx1 + 0.3
            hz = lz0 + (lz1 - lz0 - d) / 2
        elif wall_idx == 2:  # west
            hx = lx0 - w - 0.3
            hz = lz0 + (lz1 - lz0 - d) / 2
        elif wall_idx == 1:  # south
            hx = lx0 + (lx1 - lx0 - w) / 2
            hz = lz0 - d - 0.3
        else:  # north
            hx = lx0 + (lx1 - lx0 - w) / 2
            hz = lz1 + 0.3

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
        table_ent    = MeshEnt(mesh_name=table_mesh, height=self.SCALE_FACTORS[table_mesh])
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
            dir_chair = dir_couch - factor * ( -math.pi /2)  # rotate chair to face couch
            # if it's the left chair (factor = -1), rotate another 180°
            # if factor == -1:
            #     dir_chair += math.pi
            #compute position for each chair
            if axis == 'z':
                pos_x = wall_coord - wall_off if couch_wall == 0 else wall_coord + wall_off
                pos_z = cz + factor * lat_off
            else:
                pos_z = wall_coord + wall_off if couch_wall == 1 else wall_coord - wall_off
                pos_x = cx + factor * lat_off
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
        corner_off = 1.0  # offset from wall corner
        corner_factor = 1.0  # left corner

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
        # # movables ents 
        # def spot_is_free(x, z, r, entities):
        #     """
        #     Return True if no existing entity in `entities`
        #     overlaps the circle at (x,z) with radius r.
        #     """
        #     for ent in entities:
        #         ex, _, ez = ent.pos
        #         # sum of radii
        #         min_dist = r + ent.radius
        #         # squared distance
        #         if (x - ex)**2 + (z - ez)**2 < min_dist**2:
        #             return False
        #     return True

        # # … inside your room-gen method …

        # movables  = cfg["movables"]
        # count     = min(1, len(movables))
        # chosen    = self.rng.sample(movables, count)
        # margin    = 0.5
        # max_tries = 20

        # for mesh_name in chosen:
        #     # create the entity to place
        #     ent    = MeshEnt(mesh_name=mesh_name, height= self.SCALE_FACTORS[mesh_name], static=False)
        #     radius = ent.radius  # use its computed bounding radius

        #     for attempt in range(max_tries):
        #         # sample a random spot
        #         x = self.rng.uniform(living.min_x + margin, living.max_x - margin)
        #         z = self.rng.uniform(living.min_z + margin, living.max_z - margin)

        #         # check against every placed entity
        #         if not spot_is_free(x, z, radius, self.entities):
        #             continue  # collision, retry

        #         # free spot found → place it
        #         dir_movable = self.rng.uniform(0, 2 * math.pi)
        #         self.place_entity(
        #             ent,
        #             room=living,
        #             pos=(x, 0.0, z),
        #             dir=dir_movable
        #         )
        #         break
        #     else:
        #         # fallback after too many tries:
        #         # either skip or place anyway with a warning
        #         print(f"Warning: no free spot for {mesh_name} skipping")
        #         # x = max(min(x, living.max_x - margin), living.min_x + margin)
        #         # z = max(min(z, living.max_z - margin), living.min_z + margin)
        #         # self.place_entity(
        #         #     ent,
        #         #     room=living,
        #         #     pos=(x, 0.0, z),
        #         #     dir=self.rng.uniform(0, 2 * math.pi)
        #         # )

    def _furnish_kitchen(self, rooms, attach_walls):
            """
            Pick one non–living/non–hallway room, choose a random wall
            (not the portal wall), and place kitchen parts:
            fridge, stove, sink along that wall from the lower corner,
            then on the opposite wall place a trash bin in the lower corner
            and a cupboard at the midpoint. For parts whose config is a nested list,
            flatten the lists and choose randomly among all entries (e.g. stove).
            """
            cfg = self.FURNITURE["kitchen"]
            OFF = 0.7       # offset from wall
            SPACING = 1.5   # spacing between units along wall
            room = rooms.get("kitchen")
            if not room:
                return
            portal_wall = attach_walls["kitchen"]

            # 3) choose a free wall (not the portal back to parent)
            free_walls = [w for w in range(room.num_walls) if w != portal_wall]
            wall = self.rng.choice(free_walls)

            # 4) compute base position, orientation, and step vector
            x0, x1 = room.min_x, room.max_x
            z0, z1 = room.min_z, room.max_z
            cx, cz = 0.5*(x0 + x1), 0.5*(z0 + z1)

            if wall == 0:      # east wall → face west, runs alongside +z
                base_px, base_pz, base_pd = x1 - OFF, z0 + OFF, -math.pi/2
                step = (0.0, 0.0, SPACING)
            elif wall == 2:    # west wall → face east, runs alongside +z
                base_px, base_pz, base_pd = x0 + OFF, z0 + OFF, math.pi/2
                step = (0.0, 0.0, SPACING)
            elif wall == 1:    # south wall → face north, runs alongside +x
                base_px, base_pz, base_pd = x0 + OFF, z0 + OFF, 0.0
                step = (SPACING, 0.0, 0.0)
            else:              # north wall → face south, runs alongside +x
                base_px, base_pz, base_pd = x0 + OFF, z1 - OFF, math.pi
                step = (SPACING, 0.0, 0.0)

            # helper to flatten nested lists
            def _flatten(items):
                flat = []
                for it in items:
                    if isinstance(it, (list, tuple)):
                        flat.extend(it)
                    else:
                        flat.append(it)
                return flat

           # 5) place fridge & stove along the wall, then sink separately
            sequence = ["fridge", "stove"]
            for i, key in enumerate(sequence):
                meshes = cfg.get(key, [])
                meshes = _flatten(meshes)
                if not meshes:
                    continue
                mesh = self.rng.choice(meshes)
                ent  = MeshEnt(mesh_name=mesh,
                            height=self.SCALE_FACTORS.get(mesh, 1.0),
                            static=True)
                # position fridge & stove spaced by `step`
                px = base_px + step[0] * i
                pz = base_pz + step[2] * i
                self.place_entity(ent,
                                room=room,
                                pos=(px, 0.0, pz),
                                dir=base_pd)

            # now place the sink in its own spot—e.g. next to the stove but inset,
            # or on the adjacent wall, or even as an island in the room.
            sink_meshes = _flatten(cfg.get("sink", []))
            if sink_meshes:
                sink_mesh = self.rng.choice(sink_meshes)
                sink_ent   = MeshEnt(mesh_name=sink_mesh,
                                    height=self.SCALE_FACTORS.get(sink_mesh, 1.0))

                # OPTION A: place sink just after the stove with a corner offset
                sink_px = base_px + step[0] * len(sequence) 
                sink_pz = base_pz + step[2] * len(sequence) 


                self.place_entity(sink_ent,
                                room=room,
                                pos=(sink_px, 0.6, sink_pz),
                                dir=base_pd # you can re-use base_pd or rotate 90°
                               )

            # 6) place trash and cupboard on opposite wall
            opp = (wall + 2) % room.num_walls
            if opp == 0:
                opp_px, opp_pz, opp_pd = x1 - OFF, z0 + OFF, -math.pi/2
                mid_px, mid_pz = x1 - OFF, cz
            elif opp == 2:
                opp_px, opp_pz, opp_pd = x0 + OFF, z0 + OFF, math.pi/2
                mid_px, mid_pz = x0 + OFF, cz
            elif opp == 1:
                opp_px, opp_pz, opp_pd = x0 + OFF, z0 + OFF, 0.0
                mid_px, mid_pz = cx, z0 + OFF
            else:
                opp_px, opp_pz, opp_pd = x0 + OFF, z1 - OFF, math.pi
                mid_px, mid_pz = cx, z1 - OFF

            # trash bin in the corner
            trash_list = _flatten(cfg.get("trash", []))
            if trash_list:
                mesh_trash = self.rng.choice(trash_list)
                trash = MeshEnt(mesh_name=mesh_trash, height=self.SCALE_FACTORS.get(mesh_trash, 1.0))
                self.place_entity(trash, room=room, pos=(opp_px, 0.0, opp_pz), dir=opp_pd)

    def _furnish_bathroom(self, rooms, attach_walls):
        """
        Furnish the bathroom:
        1) Pick the bathroom room and a random wall (not the portal wall).
        2) Along that wall, place the sink then the toilet with fixed spacing.
        3) On the opposite wall, place the bath starting from one corner.
        4) On the last free wall, place the washer.
        """
        cfg = self.FURNITURE["bathroom"]
        OFF     = 0.4    # offset from wall
        SPACING = 2.0    # spacing between sink & toilet
        print(rooms)

        # 1) pick the bathroom
        if "bathroom" not in rooms:
            print("No bathroom room found")
            return
        room        = rooms["bathroom"]
        portal_wall = attach_walls.get("bathroom", None)
        print(portal_wall)

        # 2) choose a free wall
        free_walls = [w for w in range(room.num_walls) if w != portal_wall]
        if not free_walls:
            return
        wall = self.rng.choice(free_walls)

        # 3) compute coordinates for that wall
        x0, x1 = room.min_x, room.max_x
        z0, z1 = room.min_z, room.max_z
        cx, cz = 0.5*(x0 + x1), 0.5*(z0 + z1)

        if wall == 0:      # east wall → face west, runs +z
            base_px, base_pz, base_pd = x1 - OFF, z0 + 1.5, -math.pi/2
            step = (0.0, 0.0, SPACING)
        elif wall == 2:    # west wall → face east, runs +z
            base_px, base_pz, base_pd = x0 + OFF, z0 + 1.5, math.pi/2
            step = (0.0, 0.0, SPACING)
        elif wall == 1:    # south wall → face north, runs +x
            base_px, base_pz, base_pd = x0 + 1.5, z0 + OFF, 0.0
            step = (SPACING, 0.0, 0.0)
        else:              # north wall → face south, runs +x
            base_px, base_pz, base_pd = x0 + 1.5, z1 - OFF, math.pi
            step = (SPACING, 0.0, 0.0)

        # helper to flatten nested lists
        def _flatten(items):
            flat = []
            for it in items:
                if isinstance(it, (list, tuple)):
                    flat.extend(it)
                else:
                    flat.append(it)
            return flat

        # 4) place sink then toilet along selected wall
        for i, key in enumerate(["sink", "toilet"]):
            meshes = _flatten(cfg.get(key, []))
            if not meshes:
                print(f"No meshes found for {key}")
                continue
            mesh = self.rng.choice(meshes)
            ent  = MeshEnt(
                mesh_name=mesh,
                height=self.SCALE_FACTORS.get(mesh, 1.0)
            )
            px = base_px + step[0] * i
            pz = base_pz + step[2] * i
            self.place_entity(
                ent,
                room=room,
                pos=(px, 0.0, pz),
                dir=base_pd
            )

        # 5) place bath on the opposite wall, starting from the same corner
        opp_wall = (wall + 2) % room.num_walls
        if opp_wall == 0:
            bath_px, bath_pz, bath_pd = x1 - 0.75, z0 + 1.5, -math.pi/2
        elif opp_wall == 2:
            bath_px, bath_pz, bath_pd = x0 + 0.75, z0 + 1.5, math.pi/2
        elif opp_wall == 1:
            bath_px, bath_pz, bath_pd = x0 + 1.5, z0 + 0.75, 0.0
        else:
            bath_px, bath_pz, bath_pd = x0 + 1.5, z1 - 0.75, math.pi

        bath_meshes = _flatten(cfg.get("bath", []))
        if bath_meshes:
            mesh = self.rng.choice(bath_meshes)
            ent  = MeshEnt(
                mesh_name=mesh,
                height=self.SCALE_FACTORS.get(mesh, 1.0)
            )
            # single bath, no extra offset beyond starting corner
            self.place_entity(
                ent,
                room=room,
                pos=(bath_px, 0.0, bath_pz),
                dir=bath_pd 
            )

        # 6) on the last free wall, place the washer
        remaining = set(range(room.num_walls)) - {portal_wall, wall, opp_wall}
        if remaining:
            wash_wall = remaining.pop()
            if wash_wall == 0:
                wx, wz, wpd = x1 - OFF, cz, -math.pi/2
            elif wash_wall == 2:
                wx, wz, wpd = x0 + OFF, cz, math.pi/2
            elif wash_wall == 1:
                wx, wz, wpd = cx, z0 + OFF, 0.0
            else:
                wx, wz, wpd = cx, z1 - OFF, math.pi

            wash_meshes = _flatten(cfg.get("washer", []))
            if wash_meshes:
                mesh = self.rng.choice(wash_meshes)
                ent  = MeshEnt(
                    mesh_name=mesh,
                    height=self.SCALE_FACTORS.get(mesh, 1.0)
                )
                self.place_entity(
                    ent,
                    room=room,
                    pos=(wx, 0.0, wz),
                    dir=wpd
                )
    
    def _furnish_bedroom(self, rooms, attach_walls):
        """
        Furnish the bedroom:
        1) Pick the bedroom room by name.
        2) Compute the wall opposite the portal wall.
        3) Place a single bed centered on that opposite wall with a fixed inset.
        """
        cfg    = self.FURNITURE["bedroom"]
        OFF    = 1.5    # inset from the wall

        # 1) grab the bedroom
        room = rooms.get("bedroom")
        if not room:
            return
        portal_wall = attach_walls.get("bedroom", 0)

        # 2) opposite wall
        wall = (portal_wall + 2) % room.num_walls

        # 3) compute the wall-center coordinate and facing direction
        x0, x1 = room.min_x, room.max_x
        z0, z1 = room.min_z, room.max_z
        cx, cz = 0.5*(x0 + x1), 0.5*(z0 + z1)

        if wall == 0:       # east wall → face west
            bed_x, bed_z, bed_dir = x1 - OFF, cz, -math.pi/2
        elif wall == 2:     # west wall → face east
            bed_x, bed_z, bed_dir = x0 + OFF, cz,  math.pi/2
        elif wall == 1:     # south wall → face north
            bed_x, bed_z, bed_dir = cx,      z0 + OFF, 0.0
        else:               # north wall → face south
            bed_x, bed_z, bed_dir = cx,      z1 - OFF, math.pi

        # helper to flatten nested lists/tuples
        def _flatten(items):
            flat = []
            for it in items:
                if isinstance(it, (list, tuple)):
                    flat.extend(it)
                else:
                    flat.append(it)
            return flat

        # 4) pick and place the bed
        bed_list = _flatten(cfg.get("bed", []))
        if not bed_list:
            return
        bed_mesh = self.rng.choice(bed_list)
        ent = MeshEnt(
            mesh_name=bed_mesh,
            height=self.SCALE_FACTORS.get(bed_mesh, 1.0)
        )
        self.place_entity(
            ent,
            room=room,
            pos=(bed_x, 0.0, bed_z),
            dir=bed_dir
    )





    def create_flat(self):
        rooms, room_positions = {}, []
        attach_walls = {}  # record which wall each room was attached on

        # 1) living room
        lw, ld = self._random_dim(self.ROOM_SIZE_RANGES["living_room"])
        living = self._add_room("living_room", 0, 0, lw, ld, rooms, room_positions)

        # 2) hallway
        living_wall = self.rng.choice(range(living.num_walls))
        hallway = self._attach_hallway(living, living_wall, rooms, room_positions)
        attach_walls["hallway"] = (living_wall + 2) % hallway.num_walls

        # 3) second room on living
        to_place = [r for r in self.ROOM_SIZE_RANGES if r not in ("living_room","hallway")]
        second_wall = self.rng.choice([w for w in range(living.num_walls) if w != living_wall])
        first_name = self.rng.choice(to_place)
        room2 = self._attach_room(living, second_wall, first_name, rooms, room_positions)
        attach_walls[first_name] = (second_wall + 2) % room2.num_walls
        to_place.remove(first_name)

        # 4) remaining rooms on hallway
        hall_used = attach_walls["hallway"]
        free_hall = [w for w in range(hallway.num_walls) if w != hall_used]
        self.rng.shuffle(to_place)
        for w_idx, name in zip(free_hall, to_place):
            roomN = self._attach_room(hallway, w_idx, name, rooms, room_positions)
            attach_walls[name] = (w_idx + 2) % roomN.num_walls

        # 5) furnish living
        self._furnish_living_room(living, living_wall, second_wall)

        # 6) furnish one kitchen unit
        self._furnish_kitchen(rooms, attach_walls)
        self._furnish_bathroom(rooms, attach_walls)
        self._furnish_bedroom(rooms, attach_walls)


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
