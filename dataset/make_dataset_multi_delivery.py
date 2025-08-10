# dataset/make_dataset_multi_delivery.py
import argparse, os, json, numpy as np
from tqdm import trange
from envs.grid_env_multi_delivery import GridMultiUAVDeliveryEnv, STAY5
from planners.astar5_delivery import astar5, next_action5

def legal_mask(env, i): return env.legal_moves_mask(i)
def encode_agent(env, i): return env._obs_agent(i)

def nearest_undelivered(env, i):
    ax, ay = env.uav_xy[i]
    opts = [p for p, done in zip(env.deliveries[i], env.delivered[i]) if not done]
    if not opts:
        return env.depot if env.require_rtb else (ax, ay)
    return min(opts, key=lambda p: abs(p[0]-ax) + abs(p[1]-ay))

def roll_episode(env):
    env.reset()
    Xs, ys, Ms = [], [], []
    while True:
        dyn = [(o.x, o.y) for o in env.dyn]
        occupied = set(env.uav_xy)
        acts = []
        # expert for all agents at once (greedy A*5 with collision avoidance via occupied set)
        for i in range(env.n_uav):
            start = env.uav_xy[i]
            goal  = nearest_undelivered(env, i)
            nf    = env.nofly if (env.use_nofly and env.nf_on) else None
            others = occupied - {start}
            path = astar5(env.occ, start, goal, nf, dyn_positions=dyn, occupied=others)
            a = next_action5(start, path) if path else STAY5
            mask = legal_mask(env, i)
            if not mask[a]:
                cand = [k for k in range(5) if mask[k]]
                a = cand[0] if cand else STAY5
            acts.append(a)

        for i in range(env.n_uav):
            Xs.append(encode_agent(env, i))
            ys.append(acts[i])
            Ms.append(legal_mask(env, i).astype(np.uint8))

        _, _, done, info = env.step(acts)
        if done: break
    return np.asarray(Xs, np.float32), np.asarray(ys, np.int64), np.asarray(Ms, np.uint8), bool(info.get("success_all", False))

def main():
    ap = argparse.ArgumentParser()
    # generic
    ap.add_argument('--episodes', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out',  type=str, default='results/multi_delivery.npz')
    # scenario
    ap.add_argument('--W', type=int, default=10)
    ap.add_argument('--H', type=int, default=10)
    ap.add_argument('--n_uav', type=int, default=3)
    ap.add_argument('--n_dyn', type=int, default=3)
    ap.add_argument('--k_deliveries', type=int, default=2)
    ap.add_argument('--use_nofly', action='store_true', default=True)
    ap.add_argument('--no_nofly',  action='store_true', help='override to disable no-fly')
    ap.add_argument('--require_rtb', action='store_true', default=False)
    args = ap.parse_args()
    if args.no_nofly: args.use_nofly = False

    env = GridMultiUAVDeliveryEnv(W=args.W, H=args.H, n_uav=args.n_uav,
                                  n_dyn=args.n_dyn, k_deliveries=args.k_deliveries,
                                  use_nofly=args.use_nofly, require_rtb=args.require_rtb,
                                  seed=args.seed)
    Xs, ys, Ms = [], [], []; succ = 0
    for _ in trange(args.episodes):
        X, y, M, ok = roll_episode(env)
        Xs.append(X); ys.append(y); Ms.append(M); succ += int(ok)

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    M = np.concatenate(Ms, axis=0)
    os.makedirs('results', exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y, M=M)
    meta = {
        'episodes': args.episodes,
        'success_eps': succ,
        'feat_dim': int(X.shape[1]),
        'W': args.W, 'H': args.H,
        'n_uav': args.n_uav, 'n_dyn': args.n_dyn,
        'k_deliveries': args.k_deliveries,
        'use_nofly': args.use_nofly, 'require_rtb': args.require_rtb,
        'seed': args.seed
    }
    with open(os.path.splitext(args.out)[0] + '_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {X.shape}  success episodes: {succ} / {args.episodes}")

if __name__ == '__main__':
    main()