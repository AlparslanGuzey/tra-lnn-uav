import argparse, json, os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from lnn.liquid import LiquidPolicyClassifier

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, default='results/imitation_dataset.npz')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--save', type=str, default='results/bc_lnn.pt')
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    data = np.load(args.data)
    X = data['X'].astype(np.float32)
    y = data['y'].astype(np.int64)
    # simple split
    n = X.shape[0]
    idx = np.arange(n); np.random.shuffle(idx)
    n_tr = int(0.9*n)
    tr_idx, va_idx = idx[:n_tr], idx[n_tr:]
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xva, yva = X[va_idx], y[va_idx]

    tr_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    va_ds = TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva))
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False)

    model = LiquidPolicyClassifier(in_dim=X.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_va = 0.0
    os.makedirs('results', exist_ok=True)

    for ep in range(1, args.epochs+1):
        model.train()
        tr_loss, tr_acc, n_trs = 0.0, 0.0, 0
        for xb, yb in tr_dl:
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)
            tr_acc  += (logits.argmax(1) == yb).float().sum().item()
            n_trs   += xb.size(0)
        tr_loss /= n_trs; tr_acc /= n_trs

        # val
        model.eval()
        va_acc, n_vas = 0.0, 0
        with torch.no_grad():
            for xb, yb in va_dl:
                logits = model(xb)
                va_acc += (logits.argmax(1) == yb).float().sum().item()
                n_vas  += xb.size(0)
        va_acc /= n_vas
        print(f"Epoch {ep:02d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | val acc {va_acc:.3f}")
        if va_acc > best_va:
            best_va = va_acc
            torch.save(model.state_dict(), args.save)

    with open('results/train_summary.json','w') as f:
        json.dump({'best_val_acc': best_va}, f, indent=2)
    print('Saved best model to', args.save)

if __name__ == '__main__':
    main()
