"""
backend/services/graph_service.py

FIXES vs uploaded version:
  1. Singleton pattern — camps.py and route.py both did GraphService() at module
     level, loading the 85k-node graph TWICE on startup (~2s each). Now both
     import `get_graph_service()` which returns a single shared instance.
  2. Elevation = None for ALL nodes in sendai_graph.json (0% have data).
     Tsunami exclusion is silently a no-op. Added a startup warning so you
     know why tsunami routing looks identical to normal routing. The fix-open
     behaviour (keep node if elevation unknown) is intentional and safe.
  3. Fixed path import to use absolute imports (works with uvicorn from backend/).
"""

import json
import math
from pathlib import Path
from typing import List, Optional

from services.astar import build_adjacency, astar, haversine_m
from services.ucs import ucs
from services.bfs import bfs
from services.dfs import dfs

TSUNAMI_ELEVATION_THRESHOLD = 10.0   # meters


class GraphService:
    def __init__(
        self,
        graph_path: str = "data/sendai_graph.json",
        camps_path: str = "data/camps.json",
    ):
        graph_file = Path(graph_path)
        if not graph_file.exists():
            raise FileNotFoundError(
                f"Graph file not found: {graph_path}\n"
                "Run graph/build_graph.py to generate sendai_graph.json"
            )

        print(f"[GraphService] Loading graph from {graph_path} ...")
        with open(graph_file, "r") as f:
            self.graph = json.load(f)

        self.nodes_dict = {int(n["id"]): n for n in self.graph["nodes"]}
        print(f"[GraphService] {len(self.nodes_dict):,} nodes, "
              f"{len(self.graph['edges']):,} edges")

        # Warn if elevation data is missing (tsunami exclusion will be no-op)
        nodes_with_elev = sum(
            1 for n in self.graph["nodes"]
            if n.get("elevation") is not None
        )
        if nodes_with_elev == 0:
            print("[GraphService] ⚠  No elevation data in graph nodes. "
                  "Tsunami low-ground exclusion is disabled. "
                  "Re-run build_graph.py with elevation=True to enable it.")

        with open(camps_path, "r") as f:
            self.camps = json.load(f)
        print(f"[GraphService] {len(self.camps)} camps loaded")

    # ── Utilities ─────────────────────────────────────────────────────────────
    
    def romanize_name(self, name: str) -> str:
        """Helper to convert common Japanese road terms to English."""
        if not name: return ""
        subs = {
            "橋": " Bridge",
            "国道": "Route ",
            "通": " St",
            "線": " Line",
            "大橋": " Great Bridge",
            "北": "North ",
            "南": "South ",
            "東": "East ",
            "西": "West ",
        }
        for jp, en in subs.items():
            name = name.replace(jp, en)
        return name.strip().replace("  ", " ")

    def nearest_node(self, lat: float, lng: float) -> int:
        best_id, best_dist = None, float("inf")
        for node_id, node in self.nodes_dict.items():
            d = haversine_m(lat, lng, node["lat"], node["lng"])
            if d < best_dist:
                best_dist, best_id = d, node_id
        return best_id

    def get_valid_camps(self, calamity: str) -> List[dict]:
        return [
            c for c in self.camps
            if calamity in c.get("valid_for", []) or calamity == "none"
        ]

    def get_camp_nodes(self, valid_camps: List[dict]) -> List[int]:
        return [self.nearest_node(c["lat"], c["lng"]) for c in valid_camps]

    def get_excluded_nodes(self, calamity: str) -> set:
        if calamity != "tsunami":
            return set()
        excluded = set()
        for node_id, node in self.nodes_dict.items():
            elev = node.get("elevation")
            if elev is not None and float(elev) < TSUNAMI_ELEVATION_THRESHOLD:
                excluded.add(node_id)
        return excluded

    def path_to_coords(self, path: List[int]) -> List[List[float]]:
        coords = []
        for node_id in path:
            node = self.nodes_dict.get(int(node_id))
            if node:
                coords.append([node["lat"], node["lng"]])
        return coords

    def find_blocked_roads(self, calamity: str, severity: float) -> List[str]:
        blocked = set()
        for edge in self.graph["edges"]:
            name = edge.get("name", "")
            if not name:
                continue
            is_bridge = bool(edge.get("is_bridge", False))
            if calamity == "tsunami" and is_bridge:
                blocked.add(self.romanize_name(str(name)))
            elif calamity == "earthquake" and is_bridge and severity > 0.5:
                # Use the same Romanization for earthquake bridge blocks
                blocked.add(self.romanize_name(str(name)))
        return list(blocked)[:10]

    def get_camp_for_goal(
        self, goal_node: int, valid_camps: List[dict]
    ) -> Optional[dict]:
        camp_nodes = self.get_camp_nodes(valid_camps)
        for camp, nid in zip(valid_camps, camp_nodes):
            if nid == goal_node:
                return camp
        # Fallback: nearest camp by straight-line distance
        gn = self.nodes_dict.get(goal_node, {})
        if not gn:
            return valid_camps[0] if valid_camps else None
        best, best_d = None, float("inf")
        for camp in valid_camps:
            d = haversine_m(gn["lat"], gn["lng"], camp["lat"], camp["lng"])
            if d < best_d:
                best_d, best = d, camp
        return best

    # ── Main routing ──────────────────────────────────────────────────────────

    def find_route(
        self,
        lat: float,
        lng: float,
        calamity: str,
        severity: float,
        algorithm: str = "astar",
    ) -> dict:
        start_node  = self.nearest_node(lat, lng)
        valid_camps = self.get_valid_camps(calamity)

        if not valid_camps:
            # Fallback: use ALL camps if none match the calamity
            valid_camps = self.camps
        if not valid_camps:
            return {"error": "No camps configured."}

        camp_nodes = self.get_camp_nodes(valid_camps)
        excluded   = self.get_excluded_nodes(calamity)
        adj        = build_adjacency(self.graph, calamity, severity, excluded)

        algo_results = {}
        for fn, name in [(astar, "astar"), (ucs, "ucs"), (bfs, "bfs"), (dfs, "dfs")]:
            algo_results[name] = fn(self.nodes_dict, adj, start_node, camp_nodes)

        primary = algo_results.get(algorithm) or algo_results["astar"]
        if not primary.get("path"):
            return {"error": "No route found. All paths may be blocked."}

        path_coords = self.path_to_coords(primary["path"])
        goal_node   = primary.get("goal", camp_nodes[0])
        camp        = self.get_camp_for_goal(goal_node, valid_camps)

        dist_m        = primary["cost"]
        walk_minutes  = (dist_m / 1.2) / 60   # 1.2 m/s walking speed
        blocked_roads = self.find_blocked_roads(calamity, severity)

        algo_comparison = [
            {
                "algorithm":      name.upper(),
                "nodes_expanded": r["nodes_expanded"],
                "cost_meters":    round(r["cost"], 1) if r["cost"] != float("inf") else -1,
                "time_ms":        r["time_ms"],
                "path_length":    len(r["path"]),
            }
            for name, r in algo_results.items()
        ]

        return {
            "calamity":          calamity,
            "severity":          severity,
            "camp_name":         camp["name"]     if camp else "Unknown",
            "camp_lat":          camp["lat"]      if camp else 0.0,
            "camp_lng":          camp["lng"]      if camp else 0.0,
            "camp_capacity":     camp.get("capacity", 0) if camp else 0,
            "distance_km":       round(dist_m / 1000, 2),
            "walk_time_minutes": round(walk_minutes, 1),
            "path_coords":       path_coords,
            "blocked_roads":     blocked_roads,
            "algo_comparison":   algo_comparison,
        }


# ── Singleton accessor (import this instead of GraphService()) ─────────────────

_instance: Optional[GraphService] = None

def get_graph_service() -> GraphService:
    """Returns the single shared GraphService instance."""
    global _instance
    if _instance is None:
        _instance = GraphService()
    return _instance