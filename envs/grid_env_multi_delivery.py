# envs/grid_env_multi_delivery.py
import math, random
import numpy as np

ACTIONS_5 = [(0,0),(0,-1),(0,1),(-1,0),(1,0)]
STAY5, UP, DOWN, LEFT, RIGHT = 0,1,2,3,4

class MovingObstacle:
    def __init__(self, x, y, waypoints):
        self.x, self.y = x, y
        self.wps = waypoints[:] if waypoints else [(x,y)]
        self.i = 0
    def step(self, W, H, occ):
        if not self.wps: return
        tx, ty = self.wps[self.i]
        dx = int(np.sign(tx - self.x))
        dy = int(np.sign(ty - self.y))
        nx, ny = max(0, min(W-1, self.x+dx)), max(0, min(H-1, self.y+dy))
        if occ[ny, nx] == 0:
            self.x, self.y = nx, ny
        if (self.x, self.y) == (tx, ty):
            self.i = (self.i + 1) % len(self.wps)

class GridMultiUAVDeliveryEnv:
    """
    10x10. 3 UAVs. Single depot. Dynamic obstacles. Toggling no-fly.
    Each UAV has K delivery points; success when all delivered (and optional RTB).
    Actions: 0 stay, 1 up, 2 down, 3 left, 4 right.
    """
    def __init__(
        self,
        W=10, H=10,
        n_uav=3,
        n_dyn=3,              # dynamic obstacles
        k_deliveries=2,       # per-UAV deliveries
        seed=0,
        use_nofly=True,
        require_rtb=False,
        max_steps=200
    ):
        self.W, self.H = W, H
        self.n_uav = n_uav
        self.n_dyn = n_dyn
        self.k_deliveries = k_deliveries
        self.use_nofly = use_nofly
        self.require_rtb = require_rtb
        self.max_steps = max_steps

        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        # no static walls in this scenario
        self.occ = np.zeros((self.H, self.W), dtype=np.int8)

        self._make_depot()
        self._make_nofly()
        self._make_dynamics()
        self.reset(seed=seed)

    # ---------- world init ----------
    def _make_depot(self):
        self.depot = (self.W//2, self.H//2)

    def _make_nofly(self):
        self.nofly = np.zeros((self.H, self.W), dtype=np.int8)
        if not self.use_nofly:
            self.nf_on, self.nf_timer, self.nf_period = False, 0, 999999
            self.nf_rects = []
            return
        self.nf_rects = []
        w, h = self.rng.randint(3,5), self.rng.randint(3,5)
        x0 = self.rng.randint(0, max(0, self.W - w - 1))
        y0 = self.rng.randint(0, max(0, self.H - h - 1))
        self.nf_rects.append((x0,y0,w,h))
        self.nf_on = False
        self.nf_timer = 0
        self.nf_period = self.rng.randint(6, 12)

    def _make_dynamics(self):
        self.dyn = []
        forbid = {self.depot}
        for _ in range(self.n_dyn):
            while True:
                x, y = self.rng.randint(0, self.W-1), self.rng.randint(0, self.H-1)
                if (x, y) not in forbid and self.occ[y, x] == 0:
                    break
            wps = [(self.rng.randint(0,self.W-1), self.rng.randint(0,self.H-1)) for __ in range(2)]
            self.dyn.append(MovingObstacle(x,y,wps))
            forbid.add((x,y))

    def _sample_free(self, forbid=None):
        forbid = set(forbid or [])
        while True:
            x, y = self.rng.randint(0,self.W-1), self.rng.randint(0,self.H-1)
            if self.occ[y,x] == 0 and (x,y) not in forbid:
                return x, y

    # ---------- episode ----------
    def reset(self, seed=None):
        if seed is not None:
            self.rng.seed(seed)
            self.np_rng = np.random.default_rng(seed)
        self.uav_xy = [self.depot for _ in range(self.n_uav)]
        self.batt   = [1.0 for _ in range(self.n_uav)]

        # deliveries per UAV (distinct across all UAVs)
        self.deliveries = []
        used = {self.depot} | {(o.x,o.y) for o in self.dyn}
        for _ in range(self.n_uav):
            pts = set()
            while len(pts) < self.k_deliveries:
                pts.add(self._sample_free(used))
                used |= pts
            self.deliveries.append(list(pts))
        self.delivered = [np.zeros(self.k_deliveries, dtype=np.bool_) for _ in range(self.n_uav)]

        self.t = 0
        return self.obs()

    # ---------- helpers ----------
    def _in_bounds(self,x,y): return 0 <= x < self.W and 0 <= y < self.H

    def _blocked(self,x,y):
        if not self._in_bounds(x,y): return True
        if self.occ[y,x] == 1: return True
        if self.use_nofly and self.nf_on and self.nofly[y,x] == 1: return True
        if any((o.x,o.y) == (x,y) for o in self.dyn): return True
        return False

    def _toggle_nofly_if_needed(self):
        if not self.use_nofly: return
        self.nf_timer += 1
        if self.nf_timer % self.nf_period == 0:
            self.nf_on = not self.nf_on
            self.nofly[:] = 0
            if self.nf_on:
                for (x0,y0,w,h) in self.nf_rects:
                    self.nofly[y0:y0+h, x0:x0+w] = 1

    def _dyn_step(self):
        for o in self.dyn:
            o.step(self.W, self.H, self.occ)

    def legal_moves_mask(self, i):
        mask = np.zeros(5, dtype=bool)
        ax, ay = self.uav_xy[i]
        occupied = set(self.uav_xy)
        for a_idx, (dx,dy) in enumerate(ACTIONS_5):
            nx, ny = ax + dx, ay + dy
            if a_idx == STAY5:
                mask[a_idx] = True
            else:
                if self._blocked(nx,ny): continue
                if (nx,ny) in occupied:  continue
                mask[a_idx] = True
        return mask

    def _nearest(self, src, points):
        if not points: return None, 1e9
        best, bestd = None, 1e9
        for p in points:
            d = abs(p[0]-src[0]) + abs(p[1]-src[1])
            if d < bestd: bestd, best = d, p
        return best, bestd

    def _next_delivery(self, i):
        opts = [p for p,done in zip(self.deliveries[i], self.delivered[i]) if not done]
        tgt,_ = self._nearest(self.uav_xy[i], opts)
        return tgt

    # ---------- observations ----------
    def _rays4(self, x, y, max_r=3):
        out = []
        for dx,dy in [(0,-1),(0,1),(-1,0),(1,0)]:
            r=0; cx,cy=x,y
            while r<max_r:
                cx+=dx; cy+=dy; r+=1
                if not self._in_bounds(cx,cy) or self._blocked(cx,cy):
                    break
            out.append(r/max_r)
        return np.array(out, dtype=np.float32)

    def _nearest_mate_rel(self, i):
        ax,ay = self.uav_xy[i]
        dmin, rel = 1e9, (0.0,0.0)
        for j,(x,y) in enumerate(self.uav_xy):
            if j==i: continue
            d = math.hypot(x-ax, y-ay)
            if d < dmin: dmin, rel = d, (x-ax, y-ay)
        s = max(1.0, min(5.0, dmin))
        return np.array([rel[0]/s, rel[1]/s], dtype=np.float32)

    def _nearest_dyn_rel(self, i):
        ax,ay = self.uav_xy[i]
        dmin, rel = 1e9, (0.0,0.0)
        for o in self.dyn:
            d = math.hypot(o.x-ax, o.y-ay)
            if d < dmin: dmin, rel = d, (o.x-ax, o.y-ay)
        s = max(1.0, min(5.0, dmin))
        return np.array([rel[0]/s, rel[1]/s], dtype=np.float32)

    def _obs_agent(self, i):
        ax, ay = self.uav_xy[i]
        dx, dy = self.depot
        tgt = self._next_delivery(i)
        if tgt is None:
            tgt = self.depot if self.require_rtb else self.uav_xy[i]
        gx, gy = tgt

        ego = np.array([
            ax/(self.W-1), ay/(self.H-1),
            (dx-ax)/max(1,self.W-1), (dy-ay)/max(1,self.H-1),
            self.batt[i],
            (gx-ax)/max(1,self.W-1), (gy-ay)/max(1,self.H-1),
        ], dtype=np.float32)
        rays = self._rays4(ax, ay, max_r=3)
        mate = self._nearest_mate_rel(i)
        dyn  = self._nearest_dyn_rel(i)
        nf_local = int(self.use_nofly and self.nf_on and self.nofly[ay,ax]==1)
        nf_on = int(self.nf_on)
        phase = (self.nf_timer % max(1,self.nf_period))/max(1,self.nf_period)

        return np.concatenate([ego, rays, mate, dyn, [nf_local, nf_on, phase]], axis=0).astype(np.float32)

    def obs(self):
        return [self._obs_agent(i) for i in range(self.n_uav)]

    # ---------- step ----------
    def step(self, actions):
        self.t += 1
        self._dyn_step()
        self._toggle_nofly_if_needed()

        rewards = [-0.05]*self.n_uav
        info = {"success_all": False}

        targets = list(self.uav_xy)
        occupied = set(self.uav_xy)
        for i in range(self.n_uav):
            a = int(actions[i])
            mask = self.legal_moves_mask(i)
            if not mask[a]:
                rewards[i] -= 0.5
                continue
            ax, ay = self.uav_xy[i]
            dx, dy = ACTIONS_5[a]
            nx, ny = ax+dx, ay+dy
            if a != STAY5 and (nx,ny) not in occupied:
                targets[i] = (nx,ny)
                occupied.add((nx,ny))
                occupied.discard((ax,ay))

        new_xy = []
        for i in range(self.n_uav):
            old = self.uav_xy[i]
            newp = targets[i]
            moved = (newp != old)
            if moved:
                self.batt[i] = max(0.0, self.batt[i] - 0.03)
            else:
                if newp == self.depot:
                    self.batt[i] = min(1.0, self.batt[i] + 0.03)
            new_xy.append(newp)
        self.uav_xy = new_xy

        # deliveries
        for i in range(self.n_uav):
            for k,pt in enumerate(self.deliveries[i]):
                if not self.delivered[i][k] and self.uav_xy[i] == pt:
                    self.delivered[i][k] = True
                    rewards[i] += 1.0

        all_done = all(bool(d.all()) for d in self.delivered)
        if self.require_rtb:
            all_done = all_done and all(self.uav_xy[i] == self.depot for i in range(self.n_uav))
        any_empty = any(b <= 0.0 for b in self.batt)
        done = False
        if all_done:
            info["success_all"] = True
            done = True
        elif any_empty or self.t >= self.max_steps:
            done = True

        return self.obs(), rewards, done, info
