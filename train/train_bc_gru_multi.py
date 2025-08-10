# train/train_bc_gru_multi.py
import argparse, os, json, csv, numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class GRUPolicy5(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 64, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(input_size=in_dim, hidden_size=d_model,
                          num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 5))
    def forward(self, x_seq):  # (B, T, F)
        out, _ = self.gru(x_seq)          # (B, T, d)
        last = out[:, -1, :]              # (B, d)
        return self.head(last)            # (B, 5)

def set_seed(s): torch.manual_seed(s); np.random.seed(s)
def masked_logits(logits, mask):
    mask = mask.bool()
    dead = ~mask.any(dim=1)
    if dead.any(): mask[dead, 0] = True
    return logits.masked_fill(~mask, -1e9)

def build_loaders(X, y, M, batch):
    n = len(X); idx = np.arange(n); np.random.shuffle(idx)
    tr = int(0.9*n); tr_idx, va_idx = idx[:tr], idx[tr:]
    mk = lambda a: torch.from_numpy(a)
    tr_dl = DataLoader(TensorDataset(mk(X[tr_idx]), mk(y[tr_idx]), mk(M[tr_idx])),
                       batch_size=batch, shuffle=True)
    va_dl = DataLoader(TensorDataset(mk(X[va_idx]), mk(y[va_idx]), mk(M[va_idx])),
                       batch_size=batch, shuffle=False)
    return tr_dl, va_dl

def epoch_pass(model, dl, opt=None, device='cpu', grad_clip=1.0):
    ce = nn.CrossEntropyLoss()
    train = opt is not None
    model.train() if train else model.eval()
    tot_loss = tot_acc = n = 0.0
    for xs, yb, mb in dl:
        xs, yb, mb = xs.to(device), yb.to(device), mb.to(device)
        if train: opt.zero_grad(set_to_none=True)
        lg = masked_logits(model(xs), mb)
        loss = ce(lg, yb)
        if train:
            loss.backward()
            if grad_clip: nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
        with torch.no_grad():
            tot_acc += (lg.argmax(1)==yb).float().sum().item()
            bs = xs.size(0); n += bs; tot_loss += loss.item()*bs
    return tot_loss/max(1,n), tot_acc/max(1,n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='results/multi_delivery_seq.npz')
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--batch_size', type=int, default=2048)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--d_model', type=int, default=64)
    ap.add_argument('--layers', type=int, default=1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--early_stop', type=int, default=8)
    ap.add_argument('--out_prefix', type=str, default='results/multi_delivery_gru')
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = np.load(args.data)
    X = data['X'].astype(np.float32)   # (N, T, F)
    y = data['y'].astype(np.int64)
    M = data['M'].astype(np.uint8)
    tr_dl, va_dl = build_loaders(X, y, M, args.batch_size)

    model = GRUPolicy5(in_dim=X.shape[-1], d_model=args.d_model, num_layers=args.layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs('results', exist_ok=True)
    curve_csv = 'results/train_curve_multi_delivery_gru.csv'
    with open(curve_csv, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch','train_loss','train_acc','val_loss','val_acc'])

    best = -1.0; bad = 0
    for ep in range(1, args.epochs+1):
        tr_loss, tr_acc = epoch_pass(model, tr_dl, opt, device)
        with torch.no_grad():
            va_loss, va_acc = epoch_pass(model, va_dl, None, device)
        print(f"Epoch {ep:02d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f}")
        with open(curve_csv, 'a', newline='') as f:
            csv.writer(f).writerow([ep, tr_loss, tr_acc, va_loss, va_acc])
        torch.save(model.state_dict(), f'{args.out_prefix}_last.pt')
        if va_acc > best:
            best = float(va_acc); bad = 0
            torch.save(model.state_dict(), f'{args.out_prefix}.pt')
        else:
            bad += 1
            if args.early_stop and bad >= args.early_stop:
                print(f"Early stopping at epoch {ep}."); break
    with open('results/train_multi_delivery_gru.json', 'w') as f:
        json.dump({'best_val_acc': best}, f, indent=2)
    print(f"Saved best to {args.out_prefix}.pt")

if __name__ == '__main__':
    main()