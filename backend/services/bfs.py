"""
backend/services/bfs.py
Breadth-First Search — finds the path with the fewest edges (hops).
Does NOT minimise total distance; useful as a baseline comparison.
"""

import time
from collections import deque
from typing import Dict, List, Tuple


def bfs(
    nodes_dict: Dict[int, dict],
    adj: Dict[int, List[Tuple[int, float]]],
    start: int,
    goals: List[int]
) -> dict:
    """
    Explores nodes layer by layer (FIFO queue).
    Returns the first goal reached — fewest hops, not shortest distance.
    """
    t0 = time.perf_counter()
    goal_set = set(goals)
    nodes_expanded = 0

    queue = deque([(start, [start], 0.0)])
    visited = {start}

    while queue:
        cur, path, cost = queue.popleft()
        nodes_expanded += 1

        if cur in goal_set:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return {
                "path": path, "cost": cost, "goal": cur,
                "nodes_expanded": nodes_expanded,
                "time_ms": round(elapsed_ms, 2), "algorithm": "bfs"
            }

        for nb, w in adj.get(cur, []):
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, path + [nb], cost + w))

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return {"path": [], "cost": float("inf"), "nodes_expanded": nodes_expanded,
            "time_ms": round(elapsed_ms, 2), "algorithm": "bfs"}
