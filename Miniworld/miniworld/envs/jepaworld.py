import random
import math
from gymnasium import spaces, utils
from miniworld.entity import MeshEnt
from miniworld.miniworld import MiniWorldEnv
import numpy as np
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
    WALL_DIRS = (-math.pi/2, 0.0, math.pi/2, math.pi)
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
    "tv_01/tv_01":                     0.7,
    "duckie":                          0.1,
    "chips_01/chips_01":               0.1,
    "handy_01/handy_01":               0.3,
    "keys_01/keys_01":                 0.09,
    "dish_01/dish_01":                 0.05,
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

    def _flatten(self, items):
        out = []
        for it in items:
            if isinstance(it, (list, tuple)):
                out.extend(self._flatten(it))
            else:
                out.append(it)
        return out

    def _room_center(self, room):
        return 0.5 * (room.min_x + room.max_x), 0.5 * (room.min_z + room.max_z)

    def _wall_setup(self, room, wall, off_wall=0.0, off_along=0.0, spacing=0.0):
        x0, x1 = room.min_x, room.max_x
        z0, z1 = room.min_z, room.max_z
        if wall == 0:
            base_px, base_pz = x1 - off_wall, z0 + off_along
            step = (0.0, 0.0, spacing)
        elif wall == 2:
            base_px, base_pz = x0 + off_wall, z0 + off_along
            step = (0.0, 0.0, spacing)
        elif wall == 1:
            base_px, base_pz = x0 + off_along, z0 + off_wall
            step = (spacing, 0.0, 0.0)
        else:
            base_px, base_pz = x0 + off_along, z1 - off_wall
            step = (spacing, 0.0, 0.0)
        return base_px, base_pz, self.WALL_DIRS[wall], step

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
        cx, cz = self._room_center(living)
        p_hw = 0.75  # portal half‐width

        if wall_idx == 0:  # east wall
            # 1) portal center on living’s east wall (z‐axis midpoint)
            center_z = cz
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
            center_z = cz
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
            center_x = cx
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
            center_x = cx
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
        p_hw = 0.75  # portal half-width

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
        cx, cz = self._room_center(living)

        # 1) Couch
        couch_wall = self.rng.choice(free_walls)
        free_walls.remove(couch_wall)
        couch_mesh = self.rng.choice(cfg["couches"])
        couch_ent  = MeshEnt(mesh_name=couch_mesh, height=1.0)

        # compute orientation and center-of-wall position
        if couch_wall in (0, 2):
            along = cz - living.min_z
        else:
            along = cx - living.min_x
        pos_x, pos_z, dir_couch, _ = self._wall_setup(living, couch_wall, 0.7, along)
        
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

    def _furnish_kitchen(self, rooms, attach_walls):
        """
        Place kitchen units in the kitchen:
        fridge and stove on one wall, the sink nearby, and trash on the opposite wall.
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
        base_px, base_pz, base_pd, step = self._wall_setup(
            room, wall, OFF, OFF, SPACING
        )

        # 5) place fridge & stove along the wall, then sink separately
        sequence = ["fridge", "stove"]
        for i, key in enumerate(sequence):
            meshes = self._flatten(cfg.get(key, []))
            if not meshes:
                continue
            mesh = self.rng.choice(meshes)
            ent  = MeshEnt(mesh_name=mesh,
                        height=self.SCALE_FACTORS.get(mesh, 1.0),
                        static=True)
            # position fridge & stove spaced by `step`
            px = base_px + step[0] * i
            pz = base_pz + step[2] * i
            self.place_entity(
                ent,
                room=room,
                pos=(px, 0.0, pz),
                dir=base_pd,
            )

        # now place the sink in its own spot
        sink_meshes = self._flatten(cfg.get("sink", []))
        if sink_meshes:
            sink_mesh = self.rng.choice(sink_meshes)
            sink_ent   = MeshEnt(
                mesh_name=sink_mesh,
                height=self.SCALE_FACTORS.get(sink_mesh, 1.0),
            )

            sink_px = base_px + step[0] * len(sequence)
            sink_pz = base_pz + step[2] * len(sequence)

            self.place_entity(
                sink_ent,
                room=room,
                pos=(sink_px, 0.6, sink_pz),
                dir=base_pd,
            )

        # 6) place trash and cupboard on opposite wall
        opp = (wall + 2) % room.num_walls
        opp_px, opp_pz, opp_pd, _ = self._wall_setup(room, opp, OFF, OFF)

        # trash bin in the corner
        trash_list = self._flatten(cfg.get("trash", []))
        if trash_list:
            mesh_trash = self.rng.choice(trash_list)
            trash = MeshEnt(
                mesh_name=mesh_trash,
                height=self.SCALE_FACTORS.get(mesh_trash, 1.0),
            )
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

        # 1) pick the bathroom
        if "bathroom" not in rooms:
            return
        room        = rooms["bathroom"]
        portal_wall = attach_walls.get("bathroom", None)

        # 2) choose a free wall
        excluded = {portal_wall, (portal_wall + 2) % room.num_walls}
        candidate_walls = [w for w in range(room.num_walls) if w not in excluded]
        if not candidate_walls:
            return
        wall = self.rng.choice(candidate_walls)

        # 3) compute coordinates for that wall
        x0, x1 = room.min_x, room.max_x
        z0, z1 = room.min_z, room.max_z
        cx, cz = self._room_center(room)

        base_px, base_pz, base_pd, step = self._wall_setup(room, wall, OFF, 1.5, SPACING)

        # helper to flatten nested lists
        # 4) place sink then toilet along selected wall
        for i, key in enumerate(["sink", "toilet"]):
            meshes = self._flatten(cfg.get(key, []))
            if not meshes:
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
        bath_px, bath_pz, bath_pd, _ = self._wall_setup(room, opp_wall, 0.75, 1.5)

        bath_meshes = self._flatten(cfg.get("bath", []))
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
            if wash_wall in (0, 2):
                along = cz - z0
            else:
                along = cx - x0
            wx, wz, wpd, _ = self._wall_setup(room, wash_wall, OFF, along)

            wash_meshes = self._flatten(cfg.get("washer", []))
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
        cx, cz = self._room_center(room)

        if wall in (0, 2):
            offset = cz - z0
        else:
            offset = cx - x0
        bed_x, bed_z, bed_dir, _ = self._wall_setup(room, wall, OFF, offset)

        # 4) pick and place the bed
        bed_list = self._flatten(cfg.get("bed", []))
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
        
    def _place_random_movables(self, rooms, n):
        """
        Pick n distinct movables, then for each:
        • choose a random room from `rooms`
        • sample up to max_tries random (x,z) positions inside its bounds
        • test each candidate against all existing entities for overlap
        • place it static=False as soon as you find a free spot—or warn and skip
        """
        cfg       = self.FURNITURE["living_room"]
        movables  = cfg.get("movables", [])
        n         = min(n, len(movables))
        chosen    = self.rng.sample(movables, n)
        margin    = 0.5
        max_tries = 20

        def spot_is_free(x, z, r):
            for ent in self.entities:
                ex, _, ez = ent.pos
                if (x - ex)**2 + (z - ez)**2 < (r + ent.radius)**2:
                    return False
            return True

        for mesh_name in chosen:
            # prepare the entity
            ent    = MeshEnt(
                mesh_name=mesh_name,
                height=self.SCALE_FACTORS.get(mesh_name, 1.0),
                static=False
            )
            radius = ent.radius

            # pick a random room by (name, object)
            room_name, room = self.rng.choice(list(rooms.items()))

            # precompute bounds with margin
            x_min = room.min_x + margin
            x_max = room.max_x - margin
            z_min = room.min_z + margin
            z_max = room.max_z - margin

            # try to find a free spot
            for _ in range(max_tries):
                x = self.rng.uniform(x_min, x_max)
                z = self.rng.uniform(z_min, z_max)
                if not spot_is_free(x, z, radius):
                    continue
                dir_movable = self.rng.uniform(0, 2 * math.pi)
                self.place_entity(
                    ent,
                    room=room,
                    pos=(x, 0.0, z),
                    dir=dir_movable
                )
                break
            else:
                # if we exhaust our tries, skip with a clear warning
                print(f"Warning: could not place '{mesh_name}' in any free spot of room '{room_name}'")

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
        self._place_random_movables(rooms, n=1)


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
