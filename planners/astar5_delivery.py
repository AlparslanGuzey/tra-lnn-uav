# planners/astar5_delivery.py
import heapq
import numpy as np

MOVES5 = [(0,0),(0,-1),(0,1),(-1,0),(1,0)]

def _blocked(x,y, occ, nofly, dyn):
    H,W = occ.shape
    if not (0<=x<W and 0<=y<H): return True
    if occ[y,x]==1: return True
    if nofly is not None and nofly[y,x]==1: return True
    if (x,y) in dyn: return True
    return False

def astar5(occ, start, goal, nofly=None, dyn_positions=None, occupied=None):
    dyn_positions = set(dyn_positions or [])
    occupied = set(occupied or [])
    occ_dyn = occ.copy()
    for (ox,oy) in dyn_positions | occupied:
        if 0<=ox<occ.shape[1] and 0<=oy<occ.shape[0]:
            occ_dyn[oy,ox] = 1

    def h(p): return abs(p[0]-goal[0]) + abs(p[1]-goal[1])
    g = {start:0}; pq=[(h(start),0,start)]; came={}
    seen=set()
    while pq:
        f, gc, cur = heapq.heappop(pq)
        if cur in seen: continue
        seen.add(cur)
        if cur==goal:
            path=[cur]
            while cur in came:
                cur=came[cur]; path.append(cur)
            return list(reversed(path))
        x,y=cur
        for dx,dy in MOVES5[1:]:
            nx,ny=x+dx,y+dy
            if _blocked(nx,ny,occ_dyn,nofly,dyn_positions): continue
            if (nx,ny) in occupied and (nx,ny)!=goal: continue
            ng = gc+1
            if (nx,ny) not in g or ng<g[(nx,ny)]:
                g[(nx,ny)] = ng
                came[(nx,ny)] = (x,y)
                heapq.heappush(pq, (ng+h((nx,ny)), ng, (nx,ny)))
    return None

def next_action5(start, path):
    if not path or len(path)<2: return 0
    (x0,y0),(x1,y1) = path[0], path[1]
    dx,dy = x1-x0, y1-y0
    mapping={(0,0):0,(0,-1):1,(0,1):2,(-1,0):3,(1,0):4}
    return mapping.get((dx,dy),0)
