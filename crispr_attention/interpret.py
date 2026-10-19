"""Attribution methods, so the model is not used as a black box.

Three complementary views of which positions matter:

- **Attention weights** say where the pooling layer looks. They are cheap
  and come free with a forward pass, but attention is not by itself an
  explanation (Jain and Wallace, 2019, *NAACL*).
- **Saliency** is the gradient of the predicted class with respect to the
  input embedding (Simonyan et al., 2014). It says how sensitive the
  prediction is to a small change at each position.
- **Integrated gradients** accumulate gradients along a straight path from
  a baseline to the input (Sundararajan et al., 2017, *ICML*). It satisfies
  completeness, so attributions sum to the difference in model output
  between the input and the baseline.

Reporting all three, and checking them against a known answer, is what
turns an attention plot into evidence.
"""
from __future__ import annotations

import numpy as np
import torch


def _batches(n, size):
    for i in range(0, n, size):
        yield slice(i, i + size)


@torch.no_grad()
def attention_weights(model, X, batch_size=256, device=None):
    """Per position attention, shape [N, L]."""
    device = device or next(model.parameters()).device
    model.eval()
    out = []
    for sl in _batches(len(X), batch_size):
        xb = torch.tensor(X[sl]).to(device)
        _, w = model(xb, return_attention=True)
        out.append(w.cpu().numpy())
    return np.concatenate(out)


def saliency(model, X, targets, batch_size=128, device=None):
    """Gradient magnitude at each position, shape [N, L]."""
    device = device or next(model.parameters()).device
    model.eval()
    out = []
    for sl in _batches(len(X), batch_size):
        xb = torch.tensor(X[sl]).to(device)
        tb = torch.tensor(np.asarray(targets)[sl]).to(device)
        mask = xb != 0
        h = model.embed(xb).detach().requires_grad_(True)
        logits = model.forward_embedded(h, mask)
        logits.gather(1, tb.view(-1, 1)).sum().backward()
        g = h.grad.abs().sum(-1)                     # sum over embedding dimension
        out.append((g * mask).cpu().numpy())
    return np.concatenate(out)


def integrated_gradients(model, X, targets, steps=32, batch_size=64, device=None):
    """Integrated gradients against a zero embedding baseline, shape [N, L]."""
    device = device or next(model.parameters()).device
    model.eval()
    out = []
    for sl in _batches(len(X), batch_size):
        xb = torch.tensor(X[sl]).to(device)
        tb = torch.tensor(np.asarray(targets)[sl]).to(device)
        mask = xb != 0
        h = model.embed(xb).detach()
        baseline = torch.zeros_like(h)
        total = torch.zeros_like(h)
        for a in torch.linspace(1.0 / steps, 1.0, steps, device=device):
            point = (baseline + a * (h - baseline)).requires_grad_(True)
            logits = model.forward_embedded(point, mask)
            logits.gather(1, tb.view(-1, 1)).sum().backward()
            total += point.grad
        attr = ((h - baseline) * total / steps).sum(-1)   # sum over embedding dimension
        out.append((attr.abs() * mask).detach().cpu().numpy())
    return np.concatenate(out)


def per_class_profile(scores, y, n_classes, normalise=True):
    """Average a [N, L] attribution per class, giving [n_classes, L]."""
    prof = np.zeros((n_classes, scores.shape[1]))
    for c in range(n_classes):
        m = np.asarray(y) == c
        if m.any():
            prof[c] = scores[m].mean(0)
    if normalise:
        s = prof.sum(1, keepdims=True)
        prof = np.divide(prof, s, out=np.zeros_like(prof), where=s > 0)
    return prof


def enrichment(profile, window):
    """Mean attribution per position inside a window, over the mean outside it.

    ``window`` is ``(start, end)`` with ``end`` excluded, the same convention
    as Python slicing and as the ``highlight`` argument of the plotting
    helpers, so one window variable can be passed to both.

    A value of 1.0 means the profile is flat with respect to the window.
    10.0 means ten times as much attribution per position inside it as
    outside. Below 1.0 means the window is attended to *less* than the rest
    of the sequence, which is worth knowing and easy to miss on a plot.
    """
    profile = np.asarray(profile, dtype=float)
    start, end = int(window[0]), int(window[1])
    if not 0 <= start < end <= len(profile):
        raise ValueError(f"window {window} outside a profile of length {len(profile)}")
    mask = np.zeros(len(profile), dtype=bool)
    mask[start:end] = True
    inside, outside = profile[mask].mean(), profile[~mask].mean()
    return float(inside / outside) if outside > 0 else float("inf")
