# train/train_bc_multi_delivery.py
import argparse, os, json, csv
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from lnn.liquid import LiquidLayer

class LiquidPolicy5(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 64):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model)
        self.l1 = LiquidLayer(d_model, d_model)
        self.l2 = LiquidLayer(d_model, d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 5))
    def forward(self, x):
        x = F.relu(self.in_proj(x)).unsqueeze(1)
        x = self.l1(x, steps=1); x = self.l2(x, steps=1)
        return self.head(x.squeeze(1))

def set_seed(s): torch.manual_seed(s); np.random.seed(s)

def build_loaders(X, y, M, batch_size, val_split=0.1):
    n = len(X); idx = np.arange(n); np.random.shuffle(idx)
    tr = int((1.0 - val_split) * n)
    tr_idx, va_idx = idx[:tr], idx[tr:]
    mk = lambda a: torch.from_numpy(a)
    tr_dl = DataLoader(TensorDataset(mk(X[tr_idx]), mk(y[tr_idx]), mk(M[tr_idx])),
                       batch_size=batch_size, shuffle=True)
    va_dl = DataLoader(TensorDataset(mk(X[va_idx]), mk(y[va_idx]), mk(M[va_idx])),
                       batch_size=batch_size, shuffle=False)
    return tr_dl, va_dl

def masked_logits(logits, mask):
    mask = mask.bool()
    dead = ~mask.any(dim=1)
    if dead.any(): mask[dead, 0] = True
    return logits.masked_fill(~mask, -1e9)

def epoch_pass(model, dl, opt=None, grad_clip=1.0, device='cpu'):
    train = opt is not None
    model.train() if train else model.eval()
    tot_loss = tot_acc = n = 0.0
    ce = nn.CrossEntropyLoss()
    for xb, yb, mb in dl:
        xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
        if train: opt.zero_grad(set_to_none=True)
        lg = masked_logits(model(xb), mb)
        loss = ce(lg, yb)
        if train:
            loss.backward()
            if grad_clip and grad_clip > 0: nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
        with torch.no_grad():
            tot_acc += (lg.argmax(1) == yb).float().sum().item()
            bs = xb.size(0); n += bs; tot_loss += loss.item() * bs
    return tot_loss / max(1, n), tot_acc / max(1, n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='results/multi_delivery.npz')
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--batch_size', type=int, default=4096)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--d_model', type=int, default=64)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--early_stop', type=int, default=8)
    ap.add_argument('--val_split', type=float, default=0.1)
    ap.add_argument('--grad_clip', type=float, default=1.0)
    ap.add_argument('--out_prefix', type=str, default='results/multi_delivery_lnn')
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = np.load(args.data)
    X = data['X'].astype(np.float32); y = data['y'].astype(np.int64); M = data['M'].astype(np.uint8)

    tr_dl, va_dl = build_loaders(X, y, M, args.batch_size, args.val_split)
    model = LiquidPolicy5(in_dim=X.shape[1], d_model=args.d_model).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr*0.1)

    os.makedirs('results', exist_ok=True)
    curve_csv = 'results/train_curve_multi_delivery.csv'
    with open(curve_csv, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch','train_loss','train_acc','val_loss','val_acc','lr'])

    best = -1.0; bad = 0
    for ep in range(1, args.epochs+1):
        tr_loss, tr_acc = epoch_pass(model, tr_dl, opt, args.grad_clip, device)
        with torch.no_grad():
            va_loss, va_acc = epoch_pass(model, va_dl, None, args.grad_clip, device)
        sched.step()

        print(f"Epoch {ep:02d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.3f}")
        with open(curve_csv, 'a', newline='') as f:
            csv.writer(f).writerow([ep, f"{tr_loss:.6f}", f"{tr_acc:.6f}",
                                    f"{va_loss:.6f}", f"{va_acc:.6f}",
                                    f"{sched.get_last_lr()[0]:.8f}"])

        torch.save(model.state_dict(), f'{args.out_prefix}_last.pt')
        if va_acc > best:
            best = float(va_acc); bad = 0
            torch.save(model.state_dict(), f'{args.out_prefix}.pt')
        else:
            bad += 1
            if args.early_stop and bad >= args.early_stop:
                print(f"Early stopping at epoch {ep}."); break

    with open('results/train_multi_delivery.json', 'w') as f:
        json.dump({'best_val_acc': best,
                   'best_checkpoint': os.path.basename(f'{args.out_prefix}.pt'),
                   'last_checkpoint': os.path.basename(f'{args.out_prefix}_last.pt'),
                   'epochs_run': ep}, f, indent=2)
    print(f"Saved best to {args.out_prefix}.pt")

if __name__ == '__main__':
    main()