import heapq
from typing import List, Tuple, Optional
import numpy as np

# 8-connected moves + stay at index 8 (we ignore stay for planning)
MOVES = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

def _blocked(x, y, grid_occ: np.ndarray, nofly: Optional[np.ndarray]):
    H, W = grid_occ.shape
    if not (0 <= x < W and 0 <= y < H):
        return True
    if grid_occ[y, x] == 1:
        return True
    if nofly is not None and nofly[y, x] == 1:
        return True
    return False

def astar(occ: np.ndarray,
          start: Tuple[int,int],
          goal: Tuple[int,int],
          nofly: Optional[np.ndarray] = None,
          moving_positions: Optional[list] = None):
    """
    A* on a copy of occupancy that also marks moving obstacles as blocked.
    moving_positions: list of (x,y) for current time step.
    """
    H, W = occ.shape
    occ_dyn = occ.copy()
    if moving_positions:
        for (mx, my) in moving_positions:
            if 0 <= mx < W and 0 <= my < H:
                occ_dyn[my, mx] = 1

    def h(x,y):  # Chebyshev
        return max(abs(goal[0]-x), abs(goal[1]-y))

    open_heap = []
    g = {start: 0}
    came = {}
    heapq.heappush(open_heap, (h(*start), 0, start))
    while open_heap:
        f, gc, cur = heapq.heappop(open_heap)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            path.reverse()
            return path

        x, y = cur
        for dx, dy in MOVES:
            nx, ny = x + dx, y + dy
            if _blocked(nx, ny, occ_dyn, nofly):
                continue
            ng = gc + 1
            if (nx, ny) not in g or ng < g[(nx, ny)]:
                g[(nx, ny)] = ng
                came[(nx, ny)] = (x, y)
                heapq.heappush(open_heap, (ng + h(nx, ny), ng, (nx, ny)))
    return None

def next_action_from_path(start, path):
    if not path or len(path) < 2:
        return 8  # stay
    x0, y0 = path[0]
    x1, y1 = path[1]
    dx, dy = x1 - x0, y1 - y0
    deltas = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    try:
        return deltas.index((dx, dy))
    except ValueError:
        return 8
