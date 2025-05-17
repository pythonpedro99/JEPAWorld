
from miniworld.entity import Box, MeshEnt

FURNITURE = {
    "living_room": {
        "couches": [
            "armchair_01/armchair_01",
        ],
        "lamps": [
            "lamp_standing_01/lamp_standing_01",
            "lamp_standing_02/lamp_standing_02",
        ],
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
          }
       }

def _furnish_living_room(self, living):
        """Place couch, lamp, sideboard+TV, two chairs and coffee table in `living`."""
        cfg = FURNITURE["living_room"]

        # figure out which living-room walls are occupied
        # (living_wall & second_wall were picked earlier and stored on the room if you like,
        # or you can recompute / pass them in; here we assume attributes)
        used = getattr(living, "occupied_walls", set())
        free_walls = [w for w in range(living.num_walls) if w not in used]

        # 1) Couch
        couch_wall  = self.rng.choice(free_walls); free_walls.remove(couch_wall)
        couch_mesh  = self.rng.choice(cfg["couches"])
        # place couch flush against couch_wall
        if couch_wall == 0:   # east
            min_x, max_x = living.max_x - 1.2, living.max_x - 0.2
            min_z, max_z = living.min_z + 0.2, living.max_z - 0.2
        elif couch_wall == 2: # west
            min_x, max_x = living.min_x + 0.2, living.min_x + 1.2
            min_z, max_z = living.min_z + 0.2, living.max_z - 0.2
        elif couch_wall == 1: # south
            min_x, max_x = living.min_x + 0.2, living.max_x - 0.2
            min_z, max_z = living.min_z + 0.2, living.min_z + 1.2
        else:                 # north
            min_x, max_x = living.min_x + 0.2, living.max_x - 0.2
            min_z, max_z = living.max_z - 1.2, living.max_z - 0.2

        self.place_entity(
            MeshEnt(mesh_name=couch_mesh, height=1.0),
            room=living, min_x=min_x, max_x=max_x, min_z=min_z, max_z=max_z
        )

        # # 2) Lamp next to couch
        # lamp_mesh = self.rng.choice(cfg["lamps"])
        # # offset slightly into the room corner nearest the couch
        # if couch_wall in (0,2):
        #     lamp_min_x = max_x - 0.4 if couch_wall==0 else min_x + 0.2
        #     lamp_max_x = lamp_min_x + 0.2
        #     lamp_min_z = living.max_z - 1.0
        #     lamp_max_z = living.max_z - 0.2
        # else:
        #     lamp_min_x = living.max_x - 1.0
        #     lamp_max_x = living.max_x - 0.2
        #     lamp_min_z = max_z - 0.4 if couch_wall==1 else min_z + 0.2
        #     lamp_max_z = lamp_min_z + 0.2

        # self.place_entity(
        #     MeshEnt(mesh_name=lamp_mesh, height=1.5),
        #     room=living,
        #     min_x=lamp_min_x, max_x=lamp_max_x,
        #     min_z=lamp_min_z, max_z=lamp_max_z,
        # )

        # # 3) Sideboard + TV on another free wall
        # sb_wall  = self.rng.choice(free_walls); free_walls.remove(sb_wall)
        # sb_mesh   = self.rng.choice(cfg["sideboards"])
        # tv_mesh   = self.rng.choice(cfg["consoles"])
        # # compute placement flush to sb_wall
        # if sb_wall == 0:   # east
        #     sb_min_x, sb_max_x = living.max_x - 1.5, living.max_x - 0.2
        #     sb_min_z, sb_max_z = living.min_z + 0.2, living.min_z + 1.0
        # elif sb_wall == 2: # west
        #     sb_min_x, sb_max_x = living.min_x + 0.2, living.min_x + 1.5
        #     sb_min_z, sb_max_z = living.min_z + 0.2, living.min_z + 1.0
        # elif sb_wall == 1: # south
        #     sb_min_x, sb_max_x = living.min_x + 0.2, living.min_x + 1.0
        #     sb_min_z, sb_max_z = living.min_z + 0.2, living.min_z + 1.5
        # else:              # north
        #     sb_min_x, sb_max_x = living.max_x - 1.0, living.max_x - 0.2
        #     sb_min_z, sb_max_z = living.max_z - 1.5, living.max_z - 0.2

        # # sideboard
        # self.place_entity(
        #     MeshEnt(mesh_name=sb_mesh, height=1.0),
        #     room=living,
        #     min_x=sb_min_x, max_x=sb_max_x,
        #     min_z=sb_min_z, max_z=sb_max_z,
        # )
        # # TV
        # self.place_entity(
        #     MeshEnt(mesh_name=tv_mesh, height=1.0),
        #     room=living,
        #     min_x=sb_min_x + 0.1, max_x=sb_max_x - 0.1,
        #     min_z=sb_min_z + 0.1, max_z=sb_max_z - 0.1,
        # )

        # # 4) Two chairs + coffee table in center
        # table_mesh   = self.rng.choice(cfg["tables"])
        # chair1, chair2 = self.rng.choice(cfg["chairs"]), self.rng.choice(cfg["chairs"])
        # cx = (living.min_x + living.max_x) / 2
        # cz = (living.min_z + living.max_z) / 2

        # # coffee table
        # self.place_entity(
        #     MeshEnt(mesh_name=table_mesh, height=0.5),
        #     room=living,
        #     min_x=cx - 0.5, max_x=cx + 0.5,
        #     min_z=cz - 0.5, max_z=cz + 0.5,
        # )
        # # chair 1 (west of table)
        # self.place_entity(
        #     MeshEnt(mesh_name=chair1, height=1.0),
        #     room=living,
        #     min_x=cx - 1.2, max_x=cx - 0.6,
        #     min_z=cz - 0.4, max_z=cz + 0.4,
        # )
        # # chair 2 (east of table)
        # self.place_entity(
        #     MeshEnt(mesh_name=chair2, height=1.0),
        #     room=living,
        #     min_x=cx + 0.6, max_x=cx + 1.2,
        #     min_z=cz - 0.4, max_z=cz + 0.4,
        # )