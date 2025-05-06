#!/usr/bin/env python3
import os
import sys
import random

import gymnasium as gym
import miniworld.envs
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from shapely.geometry import Polygon, Point, LineString
from matplotlib.patches import Polygon as MplPolygon, Circle

# Optional: Add local MiniWorld path
MINIWORLD_PATH = os.path.expanduser("../MiniWorld")
if MINIWORLD_PATH not in sys.path:
    sys.path.append(MINIWORLD_PATH)

ENV_NAME = "MiniWorld-ThreeRooms-v0"


def print_env_info(env):
    """
    Print room outlines, portals, entities, and agent info.
    """
    unwrapped = env.unwrapped
    rooms = unwrapped.rooms
    entities = unwrapped.entities
    agent = unwrapped.agent

    print("\n=== Room Outlines and Portals ===")
    for i, room in enumerate(rooms):
        pts2d = [(float(pt[0]), float(pt[2])) for pt in room.outline]
        poly = Polygon(pts2d)
        print(f"\nRoom {i}:")
        print(f"  Polygon points (x, z): {pts2d}")
        print(f"  Bounds: {tuple(poly.bounds)}")
        print("  Portals:")
        if not getattr(room, "portals", None):
            print("    (none)")
            continue
        for edge_idx, edge_list in enumerate(room.portals):
            if not edge_list:
                continue
            a = pts2d[edge_idx]
            b = pts2d[(edge_idx + 1) % len(pts2d)]
            dx, dz = b[0] - a[0], b[1] - a[1]
            length = np.hypot(dx, dz)
            ux, uz = dx/length, dz/length
            for portal in edge_list:
                s = float(portal["start_pos"])
                e = float(portal["end_pos"])
                p_start = (a[0] + ux*s, a[1] + uz*s)
                p_end   = (a[0] + ux*e, a[1] + uz*e)
                print(f"    - edge {edge_idx}: start={p_start}, end={p_end}")

    print("\n=== Entities ===")
    for idx, e in enumerate(entities):
        name = getattr(e, "name", e.__class__.__name__)
        x, _, z = e.pos
        r = getattr(e, "radius", None)
        print(f"  {e.__class__.__name__} '{name}': "
              f"(x, z)=({x:.2f}, {z:.2f}), "
              f"radius={float(r) if r is not None else 'Unknown'}")

    print("\n=== Agent Info ===")
    ax, _, az = agent.pos
    print(f"  Position (x, z): ({ax:.2f}, {az:.2f})")
    print(f"  Direction (yaw): {float(agent.dir):.2f} rad")
    print(f"  Radius: {float(agent.radius):.2f}")


def get_graph_data(env):
    """
    Extract rooms, agent, and obstacles from the environment.
    """
    unwrapped = env.unwrapped
    rooms = unwrapped.rooms
    entities = unwrapped.entities
    agent = unwrapped.agent

    # Rooms + portals
    room_list = []
    for i, room in enumerate(rooms):
        pts2d = [(float(pt[0]), float(pt[2])) for pt in room.outline]
        portals = []
        if getattr(room, "portals", None):
            for edge_idx, edge_list in enumerate(room.portals):
                a = pts2d[edge_idx]
                b = pts2d[(edge_idx + 1) % len(pts2d)]
                dx, dz = b[0] - a[0], b[1] - a[1]
                length = np.hypot(dx, dz)
                ux, uz = dx/length, dz/length
                for p in edge_list:
                    s = float(p["start_pos"])
                    e = float(p["end_pos"])
                    p_start = (a[0] + ux*s, a[1] + uz*s)
                    p_end   = (a[0] + ux*e, a[1] + uz*e)
                    mid = ((p_start[0]+p_end[0]) / 2, (p_start[1]+p_end[1]) / 2)
                    portals.append({
                        "edge": edge_idx,
                        "start": p_start,
                        "end": p_end,
                        "midpoint": mid
                    })
        room_list.append({"id": i, "vertices": pts2d, "portals": portals})

    # Agent
    agent_data = {
        "pos": (float(agent.pos[0]), float(agent.pos[2])),
        "yaw": float(agent.dir),
        "radius": float(agent.radius)
    }

    # Obstacles (all entities)
    obstacles = []
    for idx, e in enumerate(entities):
        x, _, z = e.pos
        obstacles.append({
            "type": e.__class__.__name__,
            "pos": (float(x), float(z)),
            "radius": float(getattr(e, "radius", 0.0)),
            "node_name": f"{e.__class__.__name__}_{idx}"
        })

    return {"rooms": room_list, "agent": agent_data, "obstacles": obstacles}


def build_prm_graph(data, sample_density=0.05, k_neighbors=15, jitter_ratio=0.3, min_samples=5):
    """
    Build a PRM-style navigation graph:
      - portal midpoints
      - agent node
      - each entity as a node
      - evenly distributed samples per room
    Uses Polygon.covers() to include boundary nodes.
    """
    G = nx.Graph()

    # Room polygons
    room_polys = {r["id"]: Polygon(r["vertices"]) for r in data["rooms"]}

    # Obstacle buffers
    obstacle_buffers = [
        (o["node_name"], Point(*o["pos"]).buffer(o["radius"]))
        for o in data["obstacles"]
    ]

    node_positions = {}

    # Portal nodes
    for room in data["rooms"]:
        rid = room["id"]
        for idx, p in enumerate(room["portals"]):
            nid = f"r{rid}_p{idx}"
            G.add_node(nid)
            node_positions[nid] = p["midpoint"]

    # Agent node
    G.add_node("agent")
    node_positions["agent"] = tuple(data["agent"]["pos"])

    # Entity nodes
    for o in data["obstacles"]:
        nid = o["node_name"]
        G.add_node(nid)
        node_positions[nid] = o["pos"]

    # Even grid + jitter sampling
    for room in data["rooms"]:
        rid = room["id"]
        poly = room_polys[rid]
        area = poly.area
        num_samples = max(min_samples, int(area * sample_density))
        grid_n = max(1, int(np.sqrt(num_samples)))
        minx, miny, maxx, maxy = poly.bounds
        cell_w = (maxx - minx) / grid_n
        cell_h = (maxy - miny) / grid_n
        count = 0
        for i in range(grid_n):
            for j in range(grid_n):
                if count >= num_samples:
                    break
                cx = minx + (i + 0.5) * cell_w
                cy = miny + (j + 0.5) * cell_h
                dx = (random.random() - 0.5) * cell_w * jitter_ratio
                dy = (random.random() - 0.5) * cell_h * jitter_ratio
                x, y = cx + dx, cy + dy
                pt = Point(x, y)
                if not poly.covers(pt):
                    continue
                if any(buf.contains(pt) for _, buf in obstacle_buffers):
                    continue
                sid = f"r{rid}_s{count}"
                G.add_node(sid)
                node_positions[sid] = (x, y)
                count += 1

    # Helper: find room of a point
    def find_room(pt):
        for rid, poly in room_polys.items():
            if poly.covers(Point(pt)):
                return rid
        return None

    agent_room = find_room(node_positions["agent"])

    # Connect nodes via k-nearest within each room
    for rid, poly in room_polys.items():
        nodes_in_room = [
            nid for nid, pos in node_positions.items()
            if poly.covers(Point(pos))
               and not (nid == "agent" and rid != agent_room)
        ]
        for nid in nodes_in_room:
            x1, y1 = node_positions[nid]
            # k nearest
            dists = sorted(
                (
                    (m, (x1 - node_positions[m][0])**2 + (y1 - node_positions[m][1])**2)
                    for m in nodes_in_room if m != nid
                ),
                key=lambda t: t[1]
            )[:k_neighbors]
            for m, _ in dists:
                x2, y2 = node_positions[m]
                seg = LineString([(x1, y1), (x2, y2)])
                if not poly.covers(seg):
                    continue
                blocked = False
                for name, buf in obstacle_buffers:
                    if name in (nid, m):
                        continue
                    if buf.intersects(seg):
                        blocked = True
                        break
                if blocked:
                    continue
                dist = np.hypot(x1 - x2, y1 - y2)
                G.add_edge(nid, m, weight=dist)

    obstacle_nodes = [o["node_name"] for o in data["obstacles"]]
    return G, node_positions, room_polys, [buf for _, buf in obstacle_buffers], obstacle_nodes


def plot_prm_graph(G, node_positions, room_polys, obstacle_buffers, obstacle_nodes, highlight_path=None):
    """
    Plot the PRM graph with various node types and an optional path.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw rooms
    for poly in room_polys.values():
        xs, ys = poly.exterior.xy
        ax.add_patch(MplPolygon(
            list(zip(xs, ys)), closed=True,
            facecolor='lightgray', edgecolor='black', alpha=0.5
        ))

    # Draw obstacles
    for buf in obstacle_buffers:
        if buf.is_empty:
            continue
        c = buf.centroid.coords[0]
        r = buf.exterior.distance(buf.centroid)
        ax.add_patch(Circle(c, r, facecolor='red', alpha=0.3))

    # Draw edges
    for u, v in G.edges():
        x1, y1 = node_positions[u]
        x2, y2 = node_positions[v]
        ax.plot([x1, x2], [y1, y2], color='skyblue', lw=0.5, alpha=0.7)

    # Draw nodes
    for nid, (x, y) in node_positions.items():
        if nid == 'agent':
            ax.scatter(x, y, color='gold', s=100, label='Agent', zorder=5)
        elif nid.startswith('r') and '_p' in nid:
            ax.scatter(x, y, color='green', s=50, label='Portal', zorder=5)
        elif nid in obstacle_nodes:
            ax.scatter(x, y, color='purple', s=80, label='Entity', zorder=5)
        elif '_s' in nid:
            ax.scatter(x, y, color='blue', s=20, label='Sample', zorder=4)

    # Highlight path
    if highlight_path:
        for u, v in zip(highlight_path, highlight_path[1:]):
            x1, y1 = node_positions[u]
            x2, y2 = node_positions[v]
            ax.plot([x1, x2], [y1, y2], color='red', lw=3, zorder=6)

    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_title('PRM Navigation Graph')
    plt.show()


if __name__ == "__main__":
    env = gym.make(ENV_NAME, render_mode='rgb_array', obs_width=224, obs_height=224)
    env.reset()

    # 1) Print environment info
    print_env_info(env)

    # 2) Build compact data
    data = get_graph_data(env)

    # 3) Build PRM graph
    G, node_positions, room_polys, obstacle_buffers, obstacle_nodes = build_prm_graph(
        data,
        sample_density=0.50,
        k_neighbors=15,
        jitter_ratio=0.09,
        min_samples=5
    )

    # 4) Shortest path to MeshEnt
    target = next((n for n in obstacle_nodes if n.startswith("MeshEnt")), None)
    path = None
    if target and nx.has_path(G, "agent", target):
        path = nx.dijkstra_path(G, "agent", target, weight="weight")
        print("\nPath to MeshEnt:")
        print(" → ".join(path))
    else:
        print("\nNo path to MeshEnt found.")

    # 5) Plot
    plot_prm_graph(G, node_positions, room_polys, obstacle_buffers, obstacle_nodes, highlight_path=path)
