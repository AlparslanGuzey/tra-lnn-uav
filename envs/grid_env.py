import math, random
import numpy as np

ACTION_DELTAS_8 = [  # 8-neighbors; use 8 as STAY
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
]
STAY_IDX = 8

class MovingObstacle:
    def __init__(self, x, y, waypoints, speed=1):
        self.x, self.y = x, y
        self.waypoints = waypoints[:] if waypoints else [(x, y)]
        self.idx = 0
        self.speed = speed

    def step(self, grid_w, grid_h, occ):
        if not self.waypoints:
            return
        tx, ty = self.waypoints[self.idx]
        dx = np.sign(tx - self.x)
        dy = np.sign(ty - self.y)
        nx, ny = int(self.x + dx), int(self.y + dy)
        nx = max(0, min(grid_w - 1, nx))
        ny = max(0, min(grid_h - 1, ny))
        if occ[ny, nx] == 0:
            self.x, self.y = nx, ny
        if (self.x, self.y) == (tx, ty):
            self.idx = (self.idx + 1) % len(self.waypoints)

class GridUAVEnv:
    """2D grid with static obstacles, moving obstacles, no-fly toggles, and wind."""
    def __init__(self, W=40, H=40, n_moving=5, seed=0):
        self.W, self.H   = W, H
        self.n_moving    = n_moving
        self.rng         = random.Random(seed)
        self.np_rng      = np.random.default_rng(seed)

        # Build static world once
        self._make_static_obstacles()
        # Start an episode
        self._episode_init()

    # ---------- Map/episode generation ----------
    def _make_static_obstacles(self):
        self.occ = np.zeros((self.H, self.W), dtype=np.int8)
        for _ in range(6):
            w, h = self.rng.randint(4, 8), self.rng.randint(3, 7)
            x0   = self.rng.randint(0, self.W - w - 1)
            y0   = self.rng.randint(0, self.H - h - 1)
            self.occ[y0:y0 + h, x0:x0 + w] = 1

    def _make_nofly_zones(self):
        self.nofly = np.zeros((self.H, self.W), dtype=np.int8)
        self.nf_rects = []
        for _ in range(2):
            w, h = self.rng.randint(6, 10), self.rng.randint(6, 10)
            x0   = self.rng.randint(0, self.W - w - 1)
            y0   = self.rng.randint(0, self.H - h - 1)
            self.nf_rects.append((x0, y0, w, h))
        self.nf_on    = False
        self.nf_timer = 0
        self.nf_period = self.rng.randint(8, 14)

    def _make_moving_obstacles(self):
        self.mov_obs = []
        for _ in range(self.n_moving):
            while True:
                x, y = self.rng.randint(0, self.W - 1), self.rng.randint(0, self.H - 1)
                if self.occ[y, x] == 0:
                    break
            waypoints = []
            for __ in range(3):
                wx, wy = self.rng.randint(0, self.W - 1), self.rng.randint(0, self.H - 1)
                waypoints.append((wx, wy))
            self.mov_obs.append(MovingObstacle(x, y, waypoints))

    def _sample_free_cell(self):
        while True:
            x, y = self.rng.randint(0, self.W - 1), self.rng.randint(0, self.H - 1)
            if self.occ[y, x] == 0:
                return x, y

    def _reset_agent(self):
        self.ax, self.ay = self._sample_free_cell()
        self.gx, self.gy = self._sample_free_cell()
        self.energy = 1.0

    def _reset_wind(self):
        self.wind        = np.zeros(2, dtype=np.float32)
        self.wind_timer  = 0
        self.wind_period = self.rng.randint(6, 12)
        self._update_wind()

    def _update_wind(self):
        # piecewise constant wind in [-1,1]
        self.wind = np.asarray(
            [self.np_rng.uniform(-1, 1), self.np_rng.uniform(-1, 1)],
            dtype=np.float32
        )

    def _episode_init(self):
        """Reinitialize per-episode state (no recursion)."""
        self._make_nofly_zones()
        self._make_moving_obstacles()
        self._reset_agent()
        self._reset_wind()
        self.t = 0

    # ---------- Env API ----------
    def reset(self, seed=None, hard=False):
        """
        Reset episode.
        - seed: reseed RNGs for reproducibility.
        - hard=True: also rebuild static obstacles.
        """
        if seed is not None:
            self.rng.seed(seed)
            self.np_rng = np.random.default_rng(seed)
        if hard or not hasattr(self, "occ"):
            self._make_static_obstacles()
        self._episode_init()
        return self._obs()

    def _toggle_nofly_if_needed(self):
        self.nf_timer += 1
        if self.nf_timer % self.nf_period == 0:
            self.nf_on = not self.nf_on
            self.nofly[:] = 0
            if self.nf_on:
                for (x0, y0, w, h) in self.nf_rects:
                    self.nofly[y0:y0 + h, x0:x0 + w] = 1

    def _step_moving_obstacles(self):
        for mo in self.mov_obs:
            mo.step(self.W, self.H, self.occ)

    def _in_bounds(self, x, y):
        return 0 <= x < self.W and 0 <= y < self.H

    def _blocked(self, x, y):
        if not self._in_bounds(x, y):
            return True
        if self.occ[y, x] == 1:
            return True
        if self.nofly[y, x] == 1:
            return True
        # treat occupying the same cell as a moving obstacle as blocked
        if any((mo.x, mo.y) == (x, y) for mo in self.mov_obs):
            return True
        return False

    def _goal_reached(self):
        return (self.ax, self.ay) == (self.gx, self.gy)

    def _ray_distances(self, max_r=8):
        # 8 directions, normalized [0,1] capped at max_r
        rays = []
        dirs = ACTION_DELTAS_8
        for dx, dy in dirs:
            r = 0
            x, y = self.ax, self.ay
            while r < max_r:
                x += dx; y += dy; r += 1
                if not self._in_bounds(x, y) or self._blocked(x, y):
                    break
            rays.append(r / max_r)
        return np.asarray(rays, dtype=np.float32)

    def _nearest_moving_rel(self, max_d=10.0):
        if not self.mov_obs:
            return np.array([0.0, 0.0], dtype=np.float32)
        dmin, best = 1e9, (0.0, 0.0)
        for mo in self.mov_obs:
            dx = mo.x - self.ax
            dy = mo.y - self.ay
            d  = math.hypot(dx, dy)
            if d < dmin:
                dmin = d; best = (dx, dy)
        s = max(1.0, min(max_d, dmin))
        return np.array([best[0] / s, best[1] / s], dtype=np.float32)

    def _obs(self):
        gdx = (self.gx - self.ax) / max(1, self.W - 1)
        gdy = (self.gy - self.ay) / max(1, self.H - 1)
        ego = np.array([
            self.ax / (self.W - 1),
            self.ay / (self.H - 1),
            0.0, 0.0,                # vx, vy placeholders for now
            self.energy,
            gdx, gdy
        ], dtype=np.float32)
        rays = self._ray_distances()
        mo   = self._nearest_moving_rel()
        nf   = int(self.nofly[self.ay, self.ax] == 1)
        return {
            "ego": ego,
            "rays": rays,
            "move_obs": mo,
            "wind": self.wind.copy(),
            "nofly_local": nf
        }

    def step(self, action_idx: int):
        self.t += 1
        # world updates
        self._step_moving_obstacles()
        self._toggle_nofly_if_needed()
        self.wind_timer += 1
        if self.wind_timer % self.wind_period == 0:
            self._update_wind()

        # action → tentative move
        if action_idx == STAY_IDX:
            nx, ny = self.ax, self.ay
            move = np.array([0.0, 0.0], dtype=np.float32)
        else:
            dx, dy = ACTION_DELTAS_8[action_idx]
            nx, ny = self.ax + dx, self.ay + dy
            move = np.array([float(dx), float(dy)], dtype=np.float32)

        reward = -0.1
        done   = False
        info   = {}

        if not self._in_bounds(nx, ny) or self._blocked(nx, ny):
            reward -= 1.0  # illegal
            nx, ny = self.ax, self.ay
        else:
            # energy model: base move + headwind penalty
            base     = 0.02 * np.linalg.norm(move, ord=1)
            headwind = 0.01 * max(0.0, float(np.dot(self.wind, move)))
            self.energy = max(0.0, self.energy - (base + headwind))
            reward -= (base + headwind)

        self.ax, self.ay = nx, ny

        if self._goal_reached():
            reward += 5.0
            done = True
            info["success"] = True
        elif self.energy <= 0.0 or self.t >= 400:
            done = True
            info["success"] = False

        return self._obs(), reward, done, info
