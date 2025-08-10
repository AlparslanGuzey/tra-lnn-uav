import os, json, random
import numpy as np
import matplotlib.pyplot as plt
from envs.grid_env import GridUAVEnv, ACTION_DELTAS_8, STAY_IDX
from planners.astar import astar, next_action_from_path
from lnn.liquid import LiquidPolicyStateful
from lnn.encoders import encode_state
import torch

# reuse helpers from eval
from eval.eval_policy import legalize_action, legal_moves_mask, masked_argmax


def run_episode_collect(env, policy='lnn', seed=0, max_steps=400):
    random.seed(seed); np.random.seed(seed)
    obs = env.reset()
    xs, ys = [env.ax], [env.ay]

    model = None
    if policy == 'lnn':
        model = LiquidPolicyStateful(in_dim=len(encode_state(obs)))
        # prefer seq model if exists
        try:
            sd = torch.load('results/bc_lnn_seq.pt', map_location='cpu')
            model.load_state_dict(sd, strict=False)
        except Exception:
            try:
                sd = torch.load('results/bc_lnn.pt', map_location='cpu')
                model.load_state_dict(sd, strict=False)
            except Exception:
                pass
        model.eval()
        model.reset_state(batch_size=1)

    while True:
        if policy == 'astar':
            movs = [(mo.x, mo.y) for mo in env.mov_obs]
            nf   = env.nofly if env.nf_on else None
            path = astar(env.occ, (env.ax, env.ay), (env.gx, env.gy), nf, moving_positions=movs)
            a = STAY_IDX if path is None else next_action_from_path((env.ax, env.ay), path)
        else:
            x = torch.from_numpy(encode_state(obs)).unsqueeze(0)
            with torch.no_grad():
                logits = model.step(x)
            mask = legal_moves_mask(env)
            a = masked_argmax(logits, mask)
            # optional extra safety:
            a = legalize_action(env, a)

        obs, _, done, _ = env.step(a)
        xs.append(env.ax); ys.append(env.ay)
        if done or len(xs) >= max_steps:
            break
    return xs, ys, env


def plot_episode(xs, ys, env, title, out_path):
    H, W = env.H, env.W
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(env.occ + env.nofly*0.5, cmap='gray_r', origin='lower')
    mos = np.zeros_like(env.occ, dtype=float)
    for mo in env.mov_obs:
        mos[mo.y, mo.x] = 1.0
    ax.imshow(mos, cmap='Reds', alpha=0.5, origin='lower')
    ax.plot(xs, ys, linewidth=2)
    ax.scatter([xs[0]],[ys[0]], marker='s', label='start')
    ax.scatter([env.gx],[env.gy], marker='*', s=140, label='goal')
    ax.set_title(title)
    ax.set_xlim([-0.5, W-0.5]); ax.set_ylim([-0.5, H-0.5])
    ax.legend()
    os.makedirs('results/figs', exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    env = GridUAVEnv(seed=7)
    xs, ys, env = run_episode_collect(env, policy='lnn', seed=7)
    plot_episode(xs, ys, env, 'LNN (stateful + masked) Trajectory', 'results/figs/lnn_traj.png')

    env = GridUAVEnv(seed=7)
    xs, ys, env = run_episode_collect(env, policy='astar', seed=7)
    plot_episode(xs, ys, env, 'A* Trajectory', 'results/figs/astar_traj.png')

    print('Saved to results/figs/*.png')


if __name__ == '__main__':
    main()
