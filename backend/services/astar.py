"""
backend/services/astar.py
A* search algorithm + shared graph utilities (haversine, edge weights, adjacency builder).
"""

import heapq
import time
from typing import Dict, List, Tuple, Optional
from math import radians, sin, cos, sqrt, atan2


# ── Haversine distance (meters) ─────────────────────────────────────────────

def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6_371_000
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lng2 - lng1)
    a = sin(dphi/2)**2 + cos(phi1) * cos(phi2) * sin(dlambda/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


# ── Edge weight modulation based on calamity + severity ─────────────────────

def compute_edge_weight(edge: dict, calamity: str, severity: float) -> float:
    """
    Returns the effective traversal cost for an edge given current disaster.
    severity ∈ [0.0, 1.0] — output of the ML severity scaler.
    """
    base = float(edge.get("length", 100))
    highway = edge.get("highway", "")
    if isinstance(highway, list):
        highway = highway[0] if highway else ""
    is_bridge = bool(edge.get("is_bridge", False))

    if calamity == "tsunami":
        # Tsunami blocks bridges entirely due to wave impact / structural risk
        if is_bridge:
            return float("inf")
        return base

    elif calamity == "earthquake":
        if is_bridge:
            # Soften the block: only absolute block at extreme severity
            if severity > 0.95:
                return float("inf")
            # Otherwise, apply a heavy penalty (fear of collapse/slow inspection)
            # Penalty scales from 5x to 50x
            penalty_factor = 5 + (severity * 45)
            return base * penalty_factor
        
        # Roads are just slower due to debris
        slow_factor = 1 + (severity * 1.5)
        return base * slow_factor

    elif calamity == "typhoon":
        open_highways = {"motorway", "motorway_link", "trunk", "trunk_link", "primary"}
        if any(h in highway for h in open_highways):
            return base * (1 + severity * 2)
        if highway in {"residential", "living_street", "unclassified"}:
            return base * 0.8
        return base

    return base


# ── Build adjacency list from raw graph JSON ─────────────────────────────────

def build_adjacency(
    graph: dict,
    calamity: str,
    severity: float,
    excluded_nodes: Optional[set] = None
) -> Dict[int, List[Tuple[int, float]]]:
    """
    Pre-computes weighted adjacency list.
    """
    excluded = excluded_nodes or set()
    adj: Dict[int, List[Tuple[int, float]]] = {}

    for edge in graph["edges"]:
        u, v = int(edge["u"]), int(edge["v"])
        if u in excluded or v in excluded:
            continue
        w = compute_edge_weight(edge, calamity, severity)
        if w == float("inf"):
            continue
        adj.setdefault(u, []).append((v, w))
        adj.setdefault(v, []).append((u, w))

    return adj


# ── A* ───────────────────────────────────────────────────────────────────────

def astar(
    nodes_dict: Dict[int, dict],
    adj: Dict[int, List[Tuple[int, float]]],
    start: int,
    goals: List[int]
) -> dict:
    """
    Single-source, Multi-goal A* search.
    Finds the shortest path to the NEAREST reachable goal.
    """
    t0 = time.perf_counter()
    nodes_expanded = 0
    
    if start not in nodes_dict:
        return {"path": [], "cost": float("inf"), "nodes_expanded": 0, "time_ms": 0, "algorithm": "astar"}

    goal_set = set(goals)
    
    def get_h(nid: int) -> float:
        n = nodes_dict[nid]
        min_d = float("inf")
        for gid in goals:
            gn = nodes_dict[gid]
            d = haversine_m(n["lat"], n["lng"], gn["lat"], gn["lng"])
            if d < min_d:
                min_d = d
        return min_d

    open_heap = [(get_h(start), 0.0, start, [start])]
    g_score: Dict[int, float] = {start: 0.0}
    visited = set()

    while open_heap:
        f, g, cur, path = heapq.heappop(open_heap)
        
        if cur in visited: continue
        visited.add(cur)
        nodes_expanded += 1

        if cur in goal_set:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return {
                "path": path, 
                "cost": g, 
                "goal": cur,
                "nodes_expanded": nodes_expanded, 
                "time_ms": round(elapsed_ms, 2),
                "algorithm": "astar"
            }

        if g > g_score.get(cur, float("inf")):
            continue

        for nb, w in adj.get(cur, []):
            tg = g + w
            if tg < g_score.get(nb, float("inf")):
                g_score[nb] = tg
                heapq.heappush(open_heap, (tg + get_h(nb), tg, nb, path + [nb]))

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return {
        "path": [], 
        "cost": float("inf"),
        "nodes_expanded": nodes_expanded, 
        "time_ms": round(elapsed_ms, 2),
        "algorithm": "astar"
    }