import argparse, os, json
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from envs.grid_env import GridUAVEnv
from planners.astar import astar, next_action_from_path
from lnn.liquid import LiquidPolicyStateful
from lnn.encoders import encode_state

def legal_moves_mask(env):
    from envs.grid_env import ACTION_DELTAS_8
    import numpy as np
    def inb(x,y): return 0 <= x < env.W and 0 <= y < env.H
    def blocked(x,y):
        if not inb(x,y): return True
        if env.occ[y,x] == 1: return True
        if env.nofly[y,x] == 1: return True
        if any((mo.x,mo.y)==(x,y) for mo in env.mov_obs): return True
        return False
    mask = np.zeros(9, dtype=bool); mask[8] = True
    for i,(dx,dy) in enumerate(ACTION_DELTAS_8):
        nx, ny = env.ax + dx, env.ay + dy
        mask[i] = not blocked(nx,ny)
    return mask

def expert_action(env):
    movs = [(mo.x, mo.y) for mo in env.mov_obs]
    nf   = env.nofly if env.nf_on else None
    path = astar(env.occ, (env.ax, env.ay), (env.gx, env.gy), nf, moving_positions=movs)
    return 8 if path is None else next_action_from_path((env.ax, env.ay), path)

def collect_rollouts(model, episodes=200, seed=0, max_steps=400):
    env = GridUAVEnv(seed=seed)
    X, y = [], []
    for _ in range(episodes):
        obs = env.reset()
        model.reset_state(batch_size=1)
        steps = 0
        while True:
            x = torch.from_numpy(encode_state(obs)).unsqueeze(0)
            with torch.no_grad():
                logits = model.step(x)
                a = int(torch.argmax(logits, dim=1).item())
            # ask expert label at *visited* state
            y_star = expert_action(env)

            X.append(encode_state(obs))
            y.append(y_star)

            # step with masked action to keep it legal-ish
            mask = legal_moves_mask(env)
            logits = logits.clone()
            illegal = ~torch.tensor(mask, dtype=torch.bool).unsqueeze(0)
            logits[illegal] = -1e9
            a = int(torch.argmax(logits, dim=1).item())

            obs, _, done, _ = env.step(a)
            steps += 1
            if done or steps >= max_steps:
                break
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    return X, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=400)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch_size', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=3e-4)
    args = ap.parse_args()

    # load stateful model (weights from seq BC)
    env = GridUAVEnv(seed=args.seed)
    in_dim = len(encode_state(env._obs()))
    model = LiquidPolicyStateful(in_dim=in_dim)
    sd = torch.load('results/bc_lnn_seq.pt', map_location='cpu')
    model.load_state_dict(sd, strict=False)
    model.eval()

    # collect DAgger dataset
    X, y = collect_rollouts(model, episodes=args.episodes, seed=args.seed)
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    # train a classifier head on single steps (fast), updating the whole net
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(1, args.epochs+1):
        loss_sum, acc_sum, n = 0.0, 0.0, 0
        for xb, yb in dl:
            opt.zero_grad()
            # reuse the step() path by resetting state each batch element
            model.reset_state(batch_size=xb.size(0))
            logits = model.step(xb)  # one-step teacher forcing
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * xb.size(0)
            acc_sum  += (logits.argmax(1) == yb).float().sum().item()
            n        += xb.size(0)
        print(f"DAgger Epoch {ep}: loss {loss_sum/n:.4f} acc {acc_sum/n:.3f}")

    os.makedirs('results', exist_ok=True)
    torch.save(model.state_dict(), 'results/bc_lnn_seq_dagger.pt')
    with open('results/dagger_meta.json','w') as f:
        json.dump({'episodes': args.episodes, 'seed': args.seed}, f, indent=2)
    print("Saved fine-tuned model to results/bc_lnn_seq_dagger.pt")

if __name__ == '__main__':
    main()
