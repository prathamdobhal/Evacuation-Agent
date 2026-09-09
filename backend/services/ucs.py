"""
backend/services/ucs.py
Uniform Cost Search — optimal, uninformed graph search.
Expands nodes in order of cumulative path cost.
"""

import heapq
import time
from typing import Dict, List, Tuple


def ucs(
    nodes_dict: Dict[int, dict],
    adj: Dict[int, List[Tuple[int, float]]],
    start: int,
    goals: List[int]
) -> dict:
    """
    Finds the least-cost path from start to the nearest goal.
    Equivalent to Dijkstra's algorithm with early termination.
    """
    t0 = time.perf_counter()
    goal_set = set(goals)
    nodes_expanded = 0

    # (cost, node_id, path)
    heap = [(0.0, start, [start])]
    visited: Dict[int, float] = {}

    while heap:
        cost, cur, path = heapq.heappop(heap)
        nodes_expanded += 1

        if cur in visited:
            continue
        visited[cur] = cost

        if cur in goal_set:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return {
                "path": path, "cost": cost, "goal": cur,
                "nodes_expanded": nodes_expanded,
                "time_ms": round(elapsed_ms, 2), "algorithm": "ucs"
            }

        for nb, w in adj.get(cur, []):
            if nb not in visited:
                heapq.heappush(heap, (cost + w, nb, path + [nb]))

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return {"path": [], "cost": float("inf"), "nodes_expanded": nodes_expanded,
            "time_ms": round(elapsed_ms, 2), "algorithm": "ucs"}
