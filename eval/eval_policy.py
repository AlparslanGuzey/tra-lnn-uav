import argparse, json, os
import numpy as np
import torch

from envs.grid_env import GridUAVEnv, ACTION_DELTAS_8
from planners.astar import astar, next_action_from_path
from lnn.liquid import LiquidPolicyStateful
from lnn.encoders import encode_state


def policy_astar(env):
    """A* that treats moving obstacles as blocked."""
    movs = [(mo.x, mo.y) for mo in env.mov_obs]
    nf   = env.nofly if env.nf_on else None
    path = astar(env.occ, (env.ax, env.ay), (env.gx, env.gy), nf, moving_positions=movs)
    return 8 if path is None else next_action_from_path((env.ax, env.ay), path)


def get_stateful_model(in_dim: int) -> LiquidPolicyStateful:
    """Load stateful LNN; prefer sequence-trained weights if available."""
    mdl = LiquidPolicyStateful(in_dim=in_dim)
    # prefer seq model
    try:
        sd = torch.load('results/bc_lnn_seq.pt', map_location='cpu')
        mdl.load_state_dict(sd, strict=False)
    except Exception:
        # fallback to single-step model if present
        try:
            sd = torch.load('results/bc_lnn.pt', map_location='cpu')
            mdl.load_state_dict(sd, strict=False)
        except Exception:
            pass
    mdl.eval()
    return mdl


# ---------- Action masking helpers ----------
def legal_moves_mask(env):
    """Return a (9,) bool mask of which actions are legal in the current state."""
    mask = np.zeros(9, dtype=bool)

    def inb(x, y): return 0 <= x < env.W and 0 <= y < env.H

    def blocked(x, y):
        if not inb(x, y): return True
        if env.occ[y, x] == 1: return True
        if env.nofly[y, x] == 1: return True
        if any((mo.x, mo.y) == (x, y) for mo in env.mov_obs): return True
        return False

    # stay is always legal
    mask[8] = True
    for i, (dx, dy) in enumerate(ACTION_DELTAS_8):
        nx, ny = env.ax + dx, env.ay + dy
        mask[i] = not blocked(nx, ny)
    return mask


def masked_argmax(logits, mask):
    """Argmax over allowed indices; if none allowed except stay, pick stay."""
    t = logits.clone()
    illegal = ~torch.tensor(mask, dtype=torch.bool, device=t.device).unsqueeze(0)
    t[illegal] = -1e9
    return int(torch.argmax(t, dim=1).item())


# ---------- Fallback legalizer (kept, rarely needed with masking) ----------
def legalize_action(env, a: int) -> int:
    """
    If chosen action is illegal, pick the closest legal neighbor (or stay).
    Heuristic: prefer moves that reduce Chebyshev distance to goal.
    """
    def inb(x, y): return 0 <= x < env.W and 0 <= y < env.H

    def blocked(x, y):
        if not inb(x, y): return True
        if env.occ[y, x] == 1: return True
        if env.nofly[y, x] == 1: return True
        if any((mo.x, mo.y) == (x, y) for mo in env.mov_obs): return True
        return False

    # check current action
    if a == 8:
        nx, ny = env.ax, env.ay
    else:
        dx, dy = ACTION_DELTAS_8[a]
        nx, ny = env.ax + dx, env.ay + dy
    if not blocked(nx, ny):
        return a

    # try alternatives ordered by goal heuristic
    cand_idxs = list(range(8)) + [8]  # 8 moves then stay
    def next_xy(ai):
        if ai == 8: return env.ax, env.ay
        ddx, ddy = ACTION_DELTAS_8[ai]
        return env.ax + ddx, env.ay + ddy

    cand_idxs.sort(key=lambda ai: (
        abs(next_xy(ai)[0] - env.gx),
        abs(next_xy(ai)[1] - env.gy)
    ))
    for ai in cand_idxs:
        nx, ny = next_xy(ai)
        if not blocked(nx, ny):
            return ai
    return 8  # nothing legal, stay


def run_eval(policy_name, n_episodes=200, seed=123):
    env = GridUAVEnv(seed=seed)
    in_dim = len(encode_state(env._obs()))
    mdl = get_stateful_model(in_dim) if policy_name == 'lnn' else None

    metrics = {'success': 0, 'path_len': [], 'energy_end': [], 'violations': 0}

    for _ in range(n_episodes):
        obs = env.reset()
        if policy_name == 'lnn':
            mdl.reset_state(batch_size=1)  # reset hidden state per episode

        steps = 0
        while True:
            if policy_name == 'astar':
                a = policy_astar(env)
            else:
                x = torch.from_numpy(encode_state(obs)).unsqueeze(0)
                with torch.no_grad():
                    logits = mdl.step(x)
                # choose best legal action directly
                mask = legal_moves_mask(env)
                a = masked_argmax(logits, mask)

            _obs, _r, done, info = env.step(a)

            # simple violation proxy: attempted move but position unchanged
            if a != 8 and (_obs['ego'][0] == obs['ego'][0] and _obs['ego'][1] == obs['ego'][1]):
                metrics['violations'] += 1

            obs = _obs
            steps += 1
            if done:
                metrics['success'] += int(info.get('success', False))
                metrics['path_len'].append(steps)
                metrics['energy_end'].append(float(obs['ego'][4]))
                break

    out = {
        'policy': policy_name,
        'episodes': n_episodes,
        'success_rate': metrics['success'] / n_episodes,
        'avg_path_len': float(np.mean(metrics['path_len'])),
        'avg_energy_end': float(np.mean(metrics['energy_end'])),
        'violations_per_ep': metrics['violations'] / n_episodes
    }
    os.makedirs('results', exist_ok=True)
    with open(f'results/metrics_{policy_name}.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', type=str, default='lnn', choices=['lnn', 'astar'])
    ap.add_argument('--n_episodes', type=int, default=200)
    ap.add_argument('--seed', type=int, default=123)
    args = ap.parse_args()
    run_eval(args.policy, args.n_episodes, args.seed)


if __name__ == '__main__':
    main()
