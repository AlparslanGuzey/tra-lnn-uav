# planners/astar5_coord.py
from planners.astar5_delivery import astar5, next_action5

def policy_astar_coordinated(env):
    """
    Compute joint actions: sequentially plan with reservation of next cells
    to reduce collisions. Tie-break by Manhattan distance-to-goal.
    """
    # order by distance to current target (closer first to finish deliveries)
    def tgt_for(i):
        opts = [p for p,done in zip(env.deliveries[i], env.delivered[i]) if not done]
        if not opts: return env.depot if env.require_rtb else env.uav_xy[i]
        ax,ay = env.uav_xy[i]
        return min(opts, key=lambda g: abs(g[0]-ax)+abs(g[1]-ay))
    order = sorted(range(env.n_uav),
                   key=lambda i: abs(tgt_for(i)[0]-env.uav_xy[i][0]) + abs(tgt_for(i)[1]-env.uav_xy[i][1]))
    dyn = [(o.x,o.y) for o in env.dyn]
    nf  = env.nofly if (env.use_nofly and env.nf_on) else None
    occupied = set(env.uav_xy)
    acts = [0]*env.n_uav
    for i in order:
        start = env.uav_xy[i]; goal = tgt_for(i)
        others = occupied - {start}
        path = astar5(env.occ, start, goal, nf, dyn_positions=dyn, occupied=others)
        a = next_action5(start, path) if path else 0
        # reserve target if movement
        if a != 0:
            dxdy = [(0,0),(0,-1),(0,1),(-1,0),(1,0)][a]
            nx, ny = start[0]+dxdy[0], start[1]+dxdy[1]
            occupied.add((nx,ny)); occupied.discard(start)
        acts[i] = a
    return acts