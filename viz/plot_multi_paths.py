# viz/plot_multi_paths.py
import os, numpy as np, matplotlib.pyplot as plt, torch
from envs.grid_env_multi_delivery import GridMultiUAVDeliveryEnv, ACTIONS_5
from train.train_bc_multi_delivery import LiquidPolicy5

def legal_mask(env, i): return env.legal_moves_mask(i)

def step_actions(env, model):
    acts=[]
    for i in range(env.n_uav):
        x = torch.from_numpy(env._obs_agent(i)).unsqueeze(0)
        with torch.no_grad():
            logits = model(x).masked_fill(~torch.tensor(legal_mask(env,i)).unsqueeze(0), -1e9)
            a = int(torch.argmax(logits, dim=1).item())
        acts.append(a)
    return acts

def run_episode(env, model, max_steps=200):
    trails = [[env.uav_xy[i]] for i in range(env.n_uav)]
    while True and max_steps>0:
        acts = step_actions(env, model)
        _, _, done, _ = env.step(acts)
        for i in range(env.n_uav):
            trails[i].append(env.uav_xy[i])
        if done: break
        max_steps -= 1
    return trails

def plot_episode(env, trails, out_path):
    H,W = env.H, env.W
    fig, ax = plt.subplots(figsize=(5,5))
    base = env.nofly*0.7
    ax.imshow(base, cmap='Reds', origin='lower', vmin=0, vmax=1)
    for o in env.dyn:
        ax.scatter([o.x],[o.y], c='k', s=40, marker='x', label='_dyn')
    colors = ['tab:blue','tab:green','tab:orange']
    for i,tr in enumerate(trails):
        xs = [p[0] for p in tr]; ys = [p[1] for p in tr]
        ax.plot(xs, ys, '-', lw=2, color=colors[i%len(colors)], label=f'UAV {i}')
        ax.scatter([xs[0]],[ys[0]], marker='s', color=colors[i%len(colors)])
        for (gx,gy) in env.deliveries[i]:
            ax.scatter([gx],[gy], marker='*', s=120, color=colors[i%len(colors)])
    ax.scatter([env.depot[0]],[env.depot[1]], marker='P', s=140, c='purple', label='Depot')
    ax.set_xlim([-0.5,W-0.5]); ax.set_ylim([-0.5,H-0.5]); ax.set_title('Multi-UAV LNN (delivery)')
    ax.legend(loc='upper right', fontsize=8)
    os.makedirs('results/figs', exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches='tight'); plt.close(fig)

def main():
    env = GridMultiUAVDeliveryEnv(seed=7)
    model = LiquidPolicy5(in_dim=len(env._obs_agent(0)))
    sd = torch.load('results/multi_delivery_lnn.pt', map_location='cpu')
    model.load_state_dict(sd, strict=False); model.eval()
    trails = run_episode(env, model)
    plot_episode(env, trails, 'results/figs/multi_delivery_traj.png')
    print('Saved results/figs/multi_delivery_traj.png')

if __name__ == '__main__':
    main()
