import argparse, os, json
import numpy as np
from tqdm import trange
from envs.grid_env import GridUAVEnv
from planners.astar import astar, next_action_from_path
from lnn.encoders import encode_state

def roll_episode(env: GridUAVEnv, max_steps=400):
    obs = env.reset()
    X, y = [], []
    for t in range(max_steps):
        # include moving obstacles as blocked cells
        movs = [(mo.x, mo.y) for mo in env.mov_obs]
        nf   = env.nofly if env.nf_on else None
        path = astar(env.occ, (env.ax, env.ay), (env.gx, env.gy), nf, moving_positions=movs)
        action = next_action_from_path((env.ax, env.ay), path) if path is not None else 8

        X.append(encode_state(obs))
        y.append(action)

        obs, _, done, _ = env.step(action)
        if done:
            break
    return np.stack(X), np.asarray(y, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_episodes', type=int, default=500)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', type=str, default='results/imitation_dataset.npz')
    args = ap.parse_args()

    env = GridUAVEnv(seed=args.seed)
    Xs, ys = [], []
    succ = 0
    for _ in trange(args.n_episodes):
        X, y = roll_episode(env)
        Xs.append(X); ys.append(y)
        succ += int((env.ax, env.ay) == (env.gx, env.gy))
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    os.makedirs('results', exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y)
    with open('results/dataset_meta.json','w') as f:
        json.dump({'episodes': args.n_episodes, 'seed': args.seed, 'samples': int(X.shape[0])}, f, indent=2)
    print('Saved', args.out, 'with', X.shape, ' Success episodes:', succ, '/', args.n_episodes)

if __name__ == '__main__':
    main()
