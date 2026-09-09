
import json
import os

path = "backend/data/sendai_graph.json"
if os.path.exists(path):
    with open(path, "r", encoding="utf-8") as f:
        g = json.load(f)
        lats = [n['lat'] for n in g['nodes']]
        lngs = [n['lng'] for n in g['nodes']]
        print(f"Nodes: {len(g['nodes'])}")
        print(f"Lat range: {min(lats)} to {max(lats)}")
        print(f"Lng range: {min(lngs)} to {max(lngs)}")
else:
    print("Graph file not found")
