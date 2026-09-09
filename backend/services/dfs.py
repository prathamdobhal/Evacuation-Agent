"""
backend/services/dfs.py
Depth-First Search — explores as deep as possible before backtracking.
Does NOT guarantee an optimal path; useful as a baseline comparison.
"""

import time
from typing import Dict, List, Tuple


def dfs(
    nodes_dict: Dict[int, dict],
    adj: Dict[int, List[Tuple[int, float]]],
    start: int,
    goals: List[int],
    depth_limit: int = 500
) -> dict:
    """
    Iterative DFS with a depth limit to prevent stack explosion on city-scale graphs.
    Records the first goal found (non-optimal).
    """
    t0 = time.perf_counter()
    goal_set = set(goals)
    nodes_expanded = 0
    best = None

    stack = [(start, [start], 0.0, 0)]
    visited = set()

    while stack:
        cur, path, cost, depth = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        nodes_expanded += 1

        if cur in goal_set:
            if best is None:
                best = {"path": path, "cost": cost, "goal": cur}
            continue

        if depth >= depth_limit:
            continue

        for nb, w in adj.get(cur, []):
            if nb not in visited:
                stack.append((nb, path + [nb], cost + w, depth + 1))

    elapsed_ms = (time.perf_counter() - t0) * 1000
    if best is None:
        return {"path": [], "cost": float("inf"), "nodes_expanded": nodes_expanded,
                "time_ms": round(elapsed_ms, 2), "algorithm": "dfs"}
    return {
        "path": best["path"], "cost": best["cost"], "goal": best["goal"],
        "nodes_expanded": nodes_expanded,
        "time_ms": round(elapsed_ms, 2), "algorithm": "dfs"
    }
