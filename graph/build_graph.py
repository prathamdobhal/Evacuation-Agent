"""
graph/build_graph.py
Downloads the Sendai road network using OSMnx and exports it as a
simplified JSON file for the backend A* routing service.

Output: graph/sendai_graph.json  (also copies to backend/data/)

Run:
  python graph/build_graph.py
"""

import json
import os
import sys

try:
    import osmnx as ox
except ImportError:
    print("ERROR: osmnx not installed. Run: pip install osmnx")
    sys.exit(1)

# Greater Sendai Area — wide coverage for regional evacuation
PLACE = [
    "Sendai, Miyagi, Japan",
    "Shichigahama, Miyagi, Japan",
    "Tagajo, Miyagi, Japan",
    "Natori, Miyagi, Japan",
    "Iwanuma, Miyagi, Japan",
    "Rifu, Miyagi, Japan",
    "Shiogama, Miyagi, Japan",
    "Watari, Miyagi, Japan",
    "Yamamoto, Miyagi, Japan"
]
NETWORK_TYPE = "walk"  # pedestrian network for evacuation

OUTPUT_GRAPH = os.path.join(os.path.dirname(__file__), "sendai_graph.json")
OUTPUT_BACKEND = os.path.join(os.path.dirname(__file__), "..", "backend", "data", "sendai_graph.json")


def build():
    print(f"Downloading '{NETWORK_TYPE}' network for {PLACE}...")
    print("(This may take 1-3 minutes depending on connection)")

    G = ox.graph_from_place(PLACE, network_type=NETWORK_TYPE, simplify=True)
    print(f"  Raw graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    # Convert to undirected for simpler routing
    G = ox.convert.to_undirected(G)
    print(f"  Undirected: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    # Build JSON structure
    nodes = []
    for node_id, data in G.nodes(data=True):
        nodes.append({
            "id": int(node_id),
            "lat": data.get("y", 0),
            "lng": data.get("x", 0),
            "elevation": data.get("elevation", None),
        })

    edges = []
    for u, v, data in G.edges(data=True):
        length = data.get("length", 100)
        highway = data.get("highway", "")
        if isinstance(highway, list):
            highway = highway[0]
        name = data.get("name", "")
        if isinstance(name, list):
            name = name[0]

        # Detect bridges from highway tag or name
        is_bridge = bool(data.get("bridge", False))
        if not is_bridge and isinstance(name, str):
            is_bridge = "bridge" in name.lower() or "橋" in name

        edges.append({
            "u": int(u),
            "v": int(v),
            "length": round(float(length), 1),
            "highway": highway,
            "name": name if isinstance(name, str) else "",
            "is_bridge": is_bridge,
        })

    graph_json = {
        "place": PLACE,
        "network_type": NETWORK_TYPE,
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "nodes": nodes,
        "edges": edges,
    }

    # Save
    with open(OUTPUT_GRAPH, "w") as f:
        json.dump(graph_json, f)
    print(f"  Saved: {OUTPUT_GRAPH} ({os.path.getsize(OUTPUT_GRAPH) / 1024 / 1024:.1f} MB)")

    # Copy to backend
    os.makedirs(os.path.dirname(OUTPUT_BACKEND), exist_ok=True)
    with open(OUTPUT_BACKEND, "w") as f:
        json.dump(graph_json, f)
    print(f"  Copied: {OUTPUT_BACKEND}")

    print(f"\n✓ Done. {len(nodes):,} nodes, {len(edges):,} edges")


if __name__ == "__main__":
    build()
