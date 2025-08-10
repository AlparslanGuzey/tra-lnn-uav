# train/train_dagger_gru_multi.py
import argparse, os, json, numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import trange
from envs.grid_env_multi_delivery import GridMultiUAVDeliveryEnv
from planners.astar5_coord import policy_astar_coordinated
from train.train_bc_gru_multi import GRUPolicy5  # uses same arch as BC
torch.set_float32_matmul_precision('high')

def collect_rollouts(model, T, episodes=400, seed=7, device='cpu', max_steps=600):
    env = GridMultiUAVDeliveryEnv(seed=seed)
    model.eval()
    Xs, ys, Ms = [], [], []
    feat_dim = len(env._obs_agent(0))

    for _ in trange(episodes, desc="DAgger collect"):
        env.reset()
        # per-agent rolling windows
        windows = [np.zeros((T, feat_dim), dtype=np.float32) for _ in range(env.n_uav)]
        filled  = [False]*env.n_uav
        steps = 0
        while True:
            # agent actions (GRU on-policy)
            acts=[]
            for i in range(env.n_uav):
                obs = env._obs_agent(i).astype(np.float32)
                if not filled[i]:
                    for t in range(T): windows[i][t] = obs
                    filled[i] = True
                else:
                    windows[i][:-1] = windows[i][1:]
                    windows[i][-1]  = obs
                xseq = torch.from_numpy(windows[i][None, ...]).to(device)
                with torch.no_grad():
                    logits = model(xseq)  # (1,5)
                    a = int(torch.argmax(logits, dim=1).item())
                acts.append(a)

            # query expert for labels (coordinated A*)
            expert_acts = policy_astar_coordinated(env)

            # save (seq, label, mask) for each agent and this step
            for i in range(env.n_uav):
                Xs.append(windows[i].copy())                     # (T,F)
                ys.append(expert_acts[i])                        # int
                Ms.append(env.legal_moves_mask(i).astype(np.uint8))  # (5,)

            # step
            _, _, done, _ = env.step(acts)
            steps += 1
            if done or steps >= max_steps:
                break

    X = np.stack(Xs, axis=0).astype(np.float32)   # (N,T,F)
    y = np.array(ys, dtype=np.int64)              # (N,)
    M = np.stack(Ms, axis=0).astype(np.uint8)     # (N,5)
    return X, y, M

def train(model, X, y, M, epochs=5, batch_size=2048, lr=3e-4, device='cpu'):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(M))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    ce  = nn.CrossEntropyLoss()
    model.train()
    for ep in range(1, epochs+1):
        tl=ta=ns=0
        for xb,yb,mb in dl:
            xb = xb.to(device); yb = yb.to(device); mb = mb.bool().to(device)
            opt.zero_grad()
            logits = model(xb).masked_fill(~mb, -1e9)
            loss = ce(logits, yb)
            loss.backward()
            opt.step()
            tl += loss.item()*xb.size(0)
            ta += (logits.argmax(1)==yb).float().sum().item()
            ns += xb.size(0)
        print(f"DAgger Epoch {ep}: loss {tl/ns:.4f} acc {ta/ns:.3f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=600)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--T', type=int, default=8)
    ap.add_argument('--batch_size', type=int, default=2048)
    ap.add_argument('--lr', type=float, default=3e-4)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # init from BC weights
    # we expect results/multi_delivery_gru.pt from BC training
    # (this speeds up DAgger convergence)
    env = GridMultiUAVDeliveryEnv(seed=args.seed)
    model = GRUPolicy5(in_dim=len(env._obs_agent(0))).to(device)
    sd = torch.load('results/multi_delivery_gru.pt', map_location=device)
    model.load_state_dict(sd, strict=False)
    print("Loaded BC GRU weights.")

    # collect on-policy rollouts labeled by expert
    X, y, M = collect_rollouts(model, T=args.T, episodes=args.episodes, seed=args.seed, device=device)

    # fine-tune
    train(model, X, y, M, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=device)

    os.makedirs('results', exist_ok=True)
    torch.save(model.state_dict(), 'results/multi_delivery_gru_dagger.pt')
    with open('results/gru_dagger_meta.json','w') as f:
        json.dump({'episodes': args.episodes, 'epochs': args.epochs, 'T': args.T}, f, indent=2)
    print("Saved fine-tuned model to results/multi_delivery_gru_dagger.pt")

if __name__ == '__main__':
    main()