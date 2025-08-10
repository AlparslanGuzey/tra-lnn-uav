# dataset/make_dataset_seq.py
import argparse, os, json
import numpy as np
from tqdm import trange
from envs.grid_env import GridUAVEnv
from planners.astar import astar, next_action_from_path
from lnn.encoders import encode_state

def roll_episode(env: GridUAVEnv, max_steps=400):
    obs = env.reset()
    frames = []
    acts = []
    for _ in range(max_steps):
        movs = [(mo.x, mo.y) for mo in env.mov_obs]
        nf   = env.nofly if env.nf_on else None
        path = astar(env.occ, (env.ax, env.ay), (env.gx, env.gy), nf, moving_positions=movs)
        a    = 8 if path is None else next_action_from_path((env.ax, env.ay), path)
        frames.append(encode_state(obs))
        acts.append(a)
        obs, _, done, _ = env.step(a)
        if done: break
    return np.stack(frames), np.asarray(acts, dtype=np.int64)

def make_windows(frames, acts, T):
    if len(frames) < T: return [], []
    Xs, ys = [], []
    for i in range(len(frames) - T + 1):
        Xs.append(frames[i:i+T])
        ys.append(acts[i+T-1])      # supervise last action
    return Xs, ys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_episodes', type=int, default=500)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--T', type=int, default=8)
    ap.add_argument('--out', type=str, default='results/imitation_seq_T8.npz')
    args = ap.parse_args()

    env = GridUAVEnv(seed=args.seed)
    Xs, ys = [], []
    succ = 0
    for _ in trange(args.n_episodes):
        frames, acts = roll_episode(env)
        if (env.ax, env.ay) == (env.gx, env.gy): succ += 1
        ws, wa = make_windows(frames, acts, args.T)
        Xs += ws; ys += wa

    X = np.asarray(Xs, dtype=np.float32)   # (N, T, F)
    y = np.asarray(ys, dtype=np.int64)     # (N,)
    os.makedirs('results', exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y)
    with open('results/dataset_seq_meta.json','w') as f:
        json.dump({'episodes': args.n_episodes, 'seed': args.seed, 'samples': int(X.shape[0]), 'T': args.T}, f, indent=2)
    print('Saved', args.out, 'with', X.shape, ' success episodes:', succ, '/', args.n_episodes)

if __name__ == '__main__':
    main()
