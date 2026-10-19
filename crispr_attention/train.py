"""Training loop with early stopping on a held out validation set."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def fit(model, X_train, y_train, X_val, y_val, *, epochs=30, batch_size=128,
        lr=1e-3, weight_decay=0.0, patience=5, class_weight=True, device=None,
        seed=0, verbose=True):
    """Train and return the best weights by validation macro F1."""
    from sklearn.metrics import f1_score

    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if class_weight:
        counts = np.bincount(y_train, minlength=int(y_train.max()) + 1).astype(float)
        w = torch.tensor(len(y_train) / (len(counts) * np.maximum(counts, 1)),
                         dtype=torch.float32, device=device)
    else:
        w = None
    criterion = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
                        batch_size=batch_size, shuffle=True)
    Xv = torch.tensor(X_val).to(device)
    yv = np.asarray(y_val)

    best_f1, best_state, bad, history = -1.0, None, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        train_loss = total / len(loader.dataset)

        model.eval()
        with torch.no_grad():
            logits = model(Xv)
            # Same criterion as training, so the two losses are comparable.
            val_loss = criterion(logits, torch.tensor(yv).to(device)).item()
            val_pred = logits.argmax(1).cpu().numpy()
        val_f1 = f1_score(yv, val_pred, average="macro", zero_division=0)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "val_macro_f1": val_f1})
        if verbose:
            print(f"epoch {epoch:3d}  train loss {train_loss:.4f}  "
                  f"val loss {val_loss:.4f}  val macro F1 {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1, bad = val_f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"early stopping at epoch {epoch}, best val macro F1 {best_f1:.4f}")
                break

    model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def predict(model, X, batch_size=256, device=None, return_attention=False):
    device = device or next(model.parameters()).device
    model.eval()
    preds, attn = [], []
    for i in range(0, len(X), batch_size):
        xb = torch.tensor(X[i:i + batch_size]).to(device)
        if return_attention:
            logits, w = model(xb, return_attention=True)
            attn.append(w.cpu().numpy())
        else:
            logits = model(xb)
        preds.append(logits.argmax(1).cpu().numpy())
    preds = np.concatenate(preds)
    return (preds, np.concatenate(attn)) if return_attention else preds


@torch.no_grad()
def predict_proba(model, X, batch_size=256, device=None):
    """Class probabilities, shape [N, n_classes]."""
    device = device or next(model.parameters()).device
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.tensor(X[i:i + batch_size]).to(device)
        out.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(out)
