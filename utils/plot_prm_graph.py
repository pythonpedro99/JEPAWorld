import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Circle
from typing import Dict, List, Optional, Tuple

import networkx as nx
from shapely.geometry import Point, Polygon


def plot_prm_graph(
    data,  # GraphData returned by get_graph_data()
    G: nx.Graph,
    node_positions: Dict[str, Tuple[float, float]],
    obstacle_nodes: List[str],
    highlight_path: Optional[List[str]] = None,
    smoothed_curve: Optional[List[Tuple[float, float]]] = None
) -> None:
    """
    Plot the PRM roadmap over the environment, using only the three values
    returned by build_prm_graph plus the original GraphData. Optionally plot
    a smoothed curve of agent waypoints.

    Args:
        data:          GraphData (rooms, agent, obstacles).
        G:             The PRM networkx graph.
        node_positions: Mapping node_id -> (x, z) coords.
        obstacle_nodes: List of node_ids corresponding to obstacles.
        highlight_path: Optional list of node_ids to draw in red.
        smoothed_curve: Optional list of (x, z) points showing the smoothed path,
                        or a list of such lists (history).
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Reconstruct room polygons
    room_polys = {r.id: Polygon(r.vertices) for r in data.rooms}
    for poly in room_polys.values():
        x, z = poly.exterior.xy
        ax.add_patch(MplPolygon(
            list(zip(x, z)),
            closed=True, facecolor="lightgray", edgecolor="black", alpha=0.5
        ))

    # Reconstruct obstacle buffers
    obstacle_buffers = [
        Point(*obs.pos).buffer(obs.radius)
        for obs in data.obstacles
    ]
    for buf in obstacle_buffers:
        if buf.is_empty:
            continue
        cx, cz = buf.centroid.xy
        r = buf.exterior.distance(buf.centroid)
        ax.add_patch(Circle((cx[0], cz[0]), r, facecolor="red", alpha=0.3))

    # Draw PRM edges
    for u, v in G.edges():
        x1, z1 = node_positions[u]
        x2, z2 = node_positions[v]
        ax.plot([x1, x2], [z1, z2], color="skyblue", linewidth=0.5, alpha=0.7)

    # Draw nodes
    for nid, (x, z) in node_positions.items():
        if nid == "agent":
            ax.scatter(x, z, color="gold", s=100, label="Agent", zorder=5)
        elif "_p" in nid:
            ax.scatter(x, z, color="green", s=50, label="Portal", zorder=5)
        elif nid in obstacle_nodes:
            ax.scatter(x, z, color="purple", s=80, label="Obstacle", zorder=5)
        elif "_s" in nid:
            ax.scatter(x, z, color="blue", s=20, label="Sample", zorder=4)

    # Highlight a path if provided
    if highlight_path:
        for u, v in zip(highlight_path, highlight_path[1:]):
            x1, z1 = node_positions[u]
            x2, z2 = node_positions[v]
            ax.plot([x1, x2], [z1, z2], color="red", linewidth=2.5, zorder=6)

    # Plot smoothed curve if provided
    if smoothed_curve:
        # Handle nested list-of-curves
        if isinstance(smoothed_curve[0], list):
            curve = smoothed_curve[-1]
        else:
            curve = smoothed_curve
        try:
            sx, sz = zip(*curve)
            ax.plot(sx, sz, color="orange", linewidth=2.0, label="Smoothed Path", zorder=7)
        except Exception:
            pass

    ax.set_aspect("equal", "box")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title("PRM Navigation Graph")

    # clean up duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    plt.show()