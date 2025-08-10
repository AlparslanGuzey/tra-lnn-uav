# viz/fig2_scenario_snapshot.py
import os, heapq
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText
import matplotlib.patheffects as pe

from envs.grid_env_multi_delivery import GridMultiUAVDeliveryEnv

# --- publication style ---
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "savefig.dpi": 300,
})

OUT_PDF = "results/figs/fig2_scenario_snapshot.pdf"
OUT_PNG = "results/figs/fig2_scenario_snapshot.png"

# ---------- helpers -----------------------------------------------------------
def rect_cells(x0, y0, w, h):
    for x in range(x0, x0 + w):
        for y in range(y0, y0 + h):
            yield (x, y)

def build_blocked(env):
    blocked = set()
    for (x0, y0, w, h) in env.nf_rects:
        blocked.update(rect_cells(x0, y0, w, h))
    for o in env.dyn:
        blocked.add((int(o.x), int(o.y)))
    return blocked

def nearest_free(cell, blocked, W, H):
    if cell not in blocked:
        return cell
    from collections import deque
    q, seen = deque([cell]), {cell}
    while q:
        x, y = q.popleft()
        for nx, ny in [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]:
            if 0 <= nx < W and 0 <= ny < H and (nx, ny) not in seen:
                if (nx, ny) not in blocked:
                    return (nx, ny)
                seen.add((nx, ny)); q.append((nx, ny))
    return cell

def astar(start, goal, blocked, W, H):
    if start == goal: return [start]
    def h(a,b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
    N = [(1,0),(-1,0),(0,1),(0,-1)]
    openq = []
    heapq.heappush(openq, (h(start,goal), 0, start))
    came, g = {}, {start: 0}
    closed = set()
    while openq:
        _, gc, u = heapq.heappop(openq)
        if u in closed: continue
        if u == goal:
            path = [u]
            while u in came:
                u = came[u]
                path.append(u)
            return path[::-1]
        closed.add(u)
        for dx, dy in N:
            v = (u[0]+dx, u[1]+dy)
            if not (0 <= v[0] < W and 0 <= v[1] < H): continue
            if v in blocked: continue
            nv = gc + 1
            if nv < g.get(v, 1e9):
                g[v] = nv; came[v] = u
                heapq.heappush(openq, (nv + h(v,goal), nv, v))
    return [start]

def place_label(ax, path, text, color, z=6):
    """Place a small label near the 60% point of the path with a white halo."""
    if len(path) < 2:
        return
    idx = int(0.6 * (len(path) - 1))
    x, y = path[idx]
    # small offset so the text doesn’t sit on the line
    ax.text(x + 0.15, y + 0.18, text, color=color, fontsize=9,
            zorder=z, path_effects=[pe.withStroke(linewidth=3, foreground="white", alpha=0.8)])

# ---------- main --------------------------------------------------------------
def main():
    os.makedirs("results/figs", exist_ok=True)

    env = GridMultiUAVDeliveryEnv(
        W=10, H=10, n_uav=3, n_dyn=3, k_deliveries=2,
        use_nofly=True, require_rtb=False, seed=7
    )
    env.reset(seed=7)

    # --- fixed, non-overlapping scenario -------------------------------------
    W, H = env.W, env.H
    blocked = build_blocked(env)

    env.depot = (5, 4)  # central depot

    fixed_deliveries = [
        [(5, 2), (6, 0)],   # UAV1: down then right
        [(2, 0), (9, 0)],   # UAV2: left-down then long right sweep
        [(0, 1), (7, 6)],   # UAV3: left-up then top-right
    ]

    env.deliveries = [
        [nearest_free(tuple(p), blocked, W, H) for p in pts]
        for pts in fixed_deliveries
    ]
    env.uav_xy = [env.depot for _ in range(env.n_uav)]

    # Compute trajectories (A* avoids NFZ + dynamic obstacles)
    traj_colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]  # blue, green, orange
    trajs = []
    for i in range(env.n_uav):
        start = tuple(env.depot)
        path = [start]
        for goal in env.deliveries[i]:
            p = astar(start, tuple(goal), blocked, W, H)
            if len(p) > 1:
                path.extend(p[1:])
            start = tuple(goal)
        trajs.append(path)

    # --- plotting -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.9))
    fig.subplots_adjust(right=0.80, top=0.90, bottom=0.18)

    ax.set_xlim([-0.5, W - 0.5]); ax.set_ylim([-0.5, H - 0.5])
    ax.set_xticks(range(W)); ax.set_yticks(range(H))
    ax.grid(True, ls=":", alpha=0.30); ax.set_aspect("equal")
    ax.tick_params(length=0)
    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # No-fly zone (behind) with crisper border
    for (x0, y0, w, h) in env.nf_rects:
        ax.add_patch(Rectangle((x0-0.5, y0-0.5), w, h,
                               facecolor="tab:red", alpha=0.12,
                               edgecolor="firebrick", hatch="///", lw=1.2, zorder=0))

    # Dynamic obstacles & dashed waypoint segments (kept)
    for o in env.dyn:
        ax.scatter([o.x], [o.y], marker="x", s=60, color="black", zorder=3)
        if getattr(o, "wps", None):
            for wp in o.wps:
                ax.plot([o.x, wp[0]], [o.y, wp[1]],
                        ls="--", lw=1.2, color="0.5", alpha=0.70, zorder=1)

    # Depot and delivery points
    ax.scatter([env.depot[0]], [env.depot[1]], marker="P", s=140, color="purple", zorder=4)
    for i, pts in enumerate(env.deliveries):
        for gx, gy in pts:
            ax.scatter([gx], [gy], marker="*", s=140, color=traj_colors[i], zorder=5)

    # UAV trajectories: solid lines + markers + larger arrowheads + labels
    for i, path in enumerate(trajs):
        xs, ys = zip(*path)
        ax.plot(xs, ys, lw=2.2, color=traj_colors[i],
                marker='o', markevery=3, markersize=3.2, zorder=2 + i)

        # Direction arrows (up to 3 evenly spaced)
        if len(path) > 2:
            num_arrows = min(3, max(1, len(path)//4))
            arrow_positions = np.linspace(0, len(path)-2, num_arrows, dtype=int)
            for pos in arrow_positions:
                ax.annotate(
                    '', xy=path[pos+1], xytext=path[pos],
                    arrowprops=dict(
                        arrowstyle='-|>,head_length=0.60,head_width=0.40',
                        color=traj_colors[i], lw=1.2, shrinkA=0, shrinkB=0
                    ),
                    zorder=3 + i
                )
        # Label along the path
        place_label(ax, path, f"UAV{i+1}", traj_colors[i])

    # Title
    ax.set_title("Scenario snapshot (multi-UAV delivery)")

    # Legend (grouped, outside right)
    legend_handles = [
        # Map constraints
        Rectangle((0, 0), 1, 1, facecolor="tab:red", alpha=0.12,
                  edgecolor="firebrick", hatch="///", lw=1.2, label="No-fly zone"),
        # Key assets
        Line2D([0], [0], marker="P", ls="none", color="purple", markersize=9, label="Depot"),
        Line2D([0], [0], marker="*", ls="none", color="0.25", markersize=10, label="Delivery point"),
        # Obstacles
        Line2D([0], [0], marker="x", ls="none", color="black", markersize=8, label="Dynamic obstacle"),
        Line2D([0], [0], ls="--", color="0.5", lw=1.2, label="Dynamic obstacle waypoints"),
        # Trajectories
        Line2D([0], [0], color=traj_colors[0], lw=2.2, label="UAV1 trajectory"),
        Line2D([0], [0], color=traj_colors[1], lw=2.2, label="UAV2 trajectory"),
        Line2D([0], [0], color=traj_colors[2], lw=2.2, label="UAV3 trajectory"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              framealpha=0.95, borderaxespad=0.0)



    # Save
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {OUT_PDF} and {OUT_PNG}")

if __name__ == "__main__":
    main()