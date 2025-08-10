import argparse, json, os, numpy as np, torch
from collections import deque
from itertools import permutations
from tqdm import trange

from envs.grid_env_multi_delivery import GridMultiUAVDeliveryEnv
from train.train_bc_multi_delivery import LiquidPolicy5
from train.train_bc_gru_multi import GRUPolicy5
from planners.astar5_coord import policy_astar_coordinated

# ---- helpers ----
def legal_mask(env, i):
    return env.legal_moves_mask(i)

def masked_argmax(logits: torch.Tensor, mask_np, device):
    mask = torch.tensor(mask_np, dtype=torch.bool, device=device).unsqueeze(0)
    if not mask.any():
        mask[:, 0] = True
    logits = logits.masked_fill(~mask, -1e9)
    return int(torch.argmax(logits, dim=1).item())

def apply_perm(logits: torch.Tensor, perm):
    """Reorder columns of logits according to perm (tuple/list of length 5)."""
    idx = torch.tensor(perm, dtype=torch.long, device=logits.device)
    return logits.index_select(dim=1, index=idx)

def policy_greedy(env):
    acts=[]
    for i in range(env.n_uav):
        ax, ay = env.uav_xy[i]
        opts = [p for p, done in zip(env.deliveries[i], env.delivered[i]) if not done]
        if opts:
            gx, gy = min(opts, key=lambda g: abs(g[0]-ax)+abs(g[1]-ay))
        else:
            gx, gy = env.depot if env.require_rtb else (ax, ay)
        best, bestd = 0, 1e9
        mask = legal_mask(env, i)
        for a,(dx,dy) in enumerate([(0,0),(0,-1),(0,1),(-1,0),(1,0)]):
            if not mask[a]:
                continue
            nx, ny = ax+dx, ay+dy
            d = abs(nx-gx)+abs(ny-gy)
            if d < bestd:
                best, bestd = a, d
        acts.append(best)
    return acts

# ---- policies ----
def policy_lnn(env, model, device):
    acts=[]
    for i in range(env.n_uav):
        x = torch.from_numpy(env._obs_agent(i)).to(device).unsqueeze(0)
        with torch.no_grad():
            a = masked_argmax(model(x), legal_mask(env,i), device)
        acts.append(a)
    return acts

class GRUStateful:
    """Rolling T-window per UAV to match sequence training exactly."""
    def __init__(self, model: GRUPolicy5, feat_dim: int, n_uav: int, T: int = 8, device='cpu', perm=None):
        self.model = model
        self.T = T
        self.device = device
        self.buffers = [deque(maxlen=T) for _ in range(n_uav)]
        self.feat_dim = feat_dim
        self.n_uav = n_uav
        self.perm = perm if perm is not None else (0,1,2,3,4)

    def reset(self, env: GridMultiUAVDeliveryEnv):
        for i in range(self.n_uav):
            v = env._obs_agent(i).astype(np.float32)
            self.buffers[i].clear()
            for _ in range(self.T):
                self.buffers[i].append(v.copy())

    def act(self, env: GridMultiUAVDeliveryEnv):
        acts=[]
        for i in range(self.n_uav):
            obs = env._obs_agent(i).astype(np.float32)
            if len(self.buffers[i]) < self.T:
                while len(self.buffers[i]) < self.T:
                    self.buffers[i].append(obs.copy())
            else:
                self.buffers[i].append(obs.copy())

            xseq = np.stack(list(self.buffers[i]), axis=0)[None, ...]
            xseq = torch.from_numpy(xseq).to(self.device)
            with torch.no_grad():
                logits = self.model(xseq)                # (1,5)
                logits = apply_perm(logits, self.perm)   # align to env action order
                a = masked_argmax(logits, legal_mask(env,i), self.device)
            acts.append(a)
        return acts

# ---- calibration for GRU action order ----
def candidate_perms(full=False):
    # env order: [stay, up, down, left, right]
    I = (0,1,2,3,4)
    cands = {
        "identity": I,
        "swap_up_down": (0,2,1,3,4),
        "swap_left_right": (0,1,2,4,3),
        "up_left_right_down": (0,1,3,4,2),
        "up_right_left_down": (0,1,4,3,2),
        "left_right_up_down": (0,3,4,1,2),
        "right_left_up_down": (0,4,3,1,2),
    }
    if full:
        # try all 5! permutations (can be slow)
        return list(permutations([0,1,2,3,4]))
    return list(cands.values())

def calibrate_gru_perm(model, env_seed=999, episodes=60, T=8, device='cpu', full=False):
    print("[GRU] Calibrating action order…")
    best_sr, best_perm = -1.0, (0,1,2,3,4)
    perms = candidate_perms(full=full)
    for perm in perms:
        env = GridMultiUAVDeliveryEnv(seed=env_seed)
        agent = GRUStateful(model, feat_dim=len(env._obs_agent(0)), n_uav=env.n_uav, T=T, device=device, perm=perm)
        sr = 0
        for _ in range(episodes):
            env.reset(); agent.reset(env); steps=0
            while True:
                acts = agent.act(env)
                _, _, done, info = env.step(acts)
                steps += 1
                if done or steps >= 600:
                    sr += int(info.get('success_all', False))
                    break
        sr = sr / episodes
        if sr > best_sr:
            best_sr, best_perm = sr, perm
    print(f"[GRU] Calibration picked perm={best_perm} with success={best_sr:.3f}")
    return best_perm

# ---- main eval loop ----
def run(policy='lnn', episodes=300, seed=123, tag='', full_perm=False):
    env = GridMultiUAVDeliveryEnv(seed=seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    T = 8

    model = None
    gru_agent = None
    if policy == 'lnn':
        in_dim = len(env._obs_agent(0))
        model = LiquidPolicy5(in_dim=in_dim).to(device)
        sd = torch.load('results/multi_delivery_lnn.pt', map_location=device)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print("[LNN] missing:", missing, "unexpected:", unexpected)
        model.eval()
    elif policy == 'gru':
        in_dim = len(env._obs_agent(0))
        model = GRUPolicy5(in_dim=in_dim).to(device)
        path = 'results/multi_delivery_gru_dagger.pt' if os.path.exists(
            'results/multi_delivery_gru_dagger.pt') else 'results/multi_delivery_gru.pt'
        sd = torch.load(path, map_location=device)
        print(f"[GRU] loaded {os.path.basename(path)}")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print("[GRU] missing:", missing, "unexpected:", unexpected)
        model.eval()

        # load cached perm if exists, else calibrate and save
        perm_path = 'results/gru_perm.json'
        if os.path.exists(perm_path):
            with open(perm_path,'r') as f: perm = tuple(json.load(f)['perm'])
            print(f"[GRU] Using cached perm: {perm}")
        else:
            perm = calibrate_gru_perm(model, env_seed=seed+777, episodes=60, T=T, device=device, full=full_perm)
            os.makedirs('results', exist_ok=True)
            with open(perm_path,'w') as f: json.dump({'perm': list(perm)}, f)
        gru_agent = GRUStateful(model, feat_dim=in_dim, n_uav=env.n_uav, T=T, device=device, perm=perm)

    metrics = {'success':0, 'steps':[], 'batt_fail':0}
    max_steps_failsafe = 600

    for ep in trange(episodes, desc=f"Evaluating {policy}"):
        env.reset(); steps=0
        if policy == 'gru':
            gru_agent.reset(env)

        while True:
            if policy == 'lnn':
                acts = policy_lnn(env, model, device)
            elif policy == 'gru':
                acts = gru_agent.act(env)
            elif policy == 'greedy':
                acts = policy_greedy(env)
            elif policy in ('astar_coord','astar'):
                acts = policy_astar_coordinated(env)
            else:
                raise ValueError(policy)

            _, _, done, info = env.step(acts)
            steps += 1
            if steps >= max_steps_failsafe:
                done = True
                info = info or {}
                info['success_all'] = False

            if done:
                metrics['success'] += int(info.get('success_all', False))
                metrics['batt_fail'] += int(any(b <= 0.0 for b in env.batt))
                metrics['steps'].append(steps)
                break

        if (ep+1) % 100 == 0:
            sr = metrics['success']/(ep+1)
            avg = float(np.mean(metrics['steps']))
            print(f"[{policy}] {ep+1}/{episodes} interim: success={sr:.3f}, avg_steps={avg:.2f}")

    out = {
        'policy': policy,
        'episodes': episodes,
        'success_rate': metrics['success'] / episodes,
        'avg_steps': float(np.mean(metrics['steps'])),
        'batt_fail_rate': metrics['batt_fail'] / episodes
    }
    os.makedirs('results', exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    with open(f'results/multi_delivery_metrics_{policy}{suffix}.json','w') as f:
        json.dump(out, f, indent=2)
    print(out)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', choices=['lnn','gru','greedy','astar_coord','astar'], default='lnn')
    ap.add_argument('--episodes', type=int, default=500)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--tag', type=str, default='')
    ap.add_argument('--full_perm', action='store_true', help='Try all 5! permutations (slow)')
    args = ap.parse_args()
    run(args.policy, args.episodes, args.seed, args.tag, args.full_perm)