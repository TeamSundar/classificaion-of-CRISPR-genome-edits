"""Figures for evaluation and interpretation.

Every plotting function returns the matplotlib figure, so notebooks can
save it at publication resolution without re-running the analysis.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score,
                             precision_recall_curve)


def confusion(cm, labels, normalise=True, figsize=(6, 5), title="Confusion matrix"):
    m = cm.astype(float)
    if normalise:
        s = m.sum(1, keepdims=True)
        m = np.divide(m, s, out=np.zeros_like(m), where=s > 0)
    fig, ax = plt.subplots(figsize=figsize)
    ConfusionMatrixDisplay(m, display_labels=labels).plot(
        ax=ax, cmap="Blues", values_format=".2f" if normalise else "d", colorbar=False)
    ax.set_title(title + (" (row normalised)" if normalise else ""))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    return fig


def pr_curves(y_true, y_proba, labels, figsize=(6, 5)):
    """One precision recall curve per class, one versus rest.

    Preferred over ROC under heavy imbalance, because precision reacts to
    the number of false positives relative to true positives rather than
    to the large negative class (Saito and Rehmsmeier, 2015, *PLoS ONE*).
    """
    y_true = np.asarray(y_true)
    fig, ax = plt.subplots(figsize=figsize)
    rows = []
    for c, name in enumerate(labels):
        pos = (y_true == c).astype(int)
        if pos.sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(pos, y_proba[:, c])
        ap = average_precision_score(pos, y_proba[:, c])
        base = pos.mean()                       # precision of a random ranker
        ax.plot(rec, prec, label=f"{name}  AP={ap:.3f}  (chance {base:.3f})")
        rows.append({"class": name, "n_positive": int(pos.sum()),
                     "average_precision": ap, "chance": base, "lift": ap / base})
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_title("Precision recall, one versus rest")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig, rows


def position_heatmap(profiles, class_labels, highlight=None, figsize=(10, 3.4),
                     title="Positional importance by class", cbar_label="share of attribution"):
    """Classes by positions heatmap of any per position attribution."""
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(profiles, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(class_labels)), class_labels)
    ax.set_xlabel("position in sequence"); ax.set_title(title)
    if highlight is not None:
        lo, hi = highlight
        ax.axvline(lo - 0.5, color="red", lw=1.2)
        ax.axvline(hi - 0.5, color="red", lw=1.2)
    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.tight_layout()
    return fig


def profile_comparison(profiles: dict, highlight=None, figsize=(10, 3.4),
                       title="Mean attribution by position"):
    """Overlay several per position attributions, for example attention,
    saliency and integrated gradients, each scaled to sum to one."""
    fig, ax = plt.subplots(figsize=figsize)
    for name, p in profiles.items():
        p = np.asarray(p, dtype=float)
        ax.plot(p / p.sum(), label=name, lw=1.4)
    if highlight is not None:
        ax.axvspan(highlight[0], highlight[1] - 1, color="orange", alpha=0.25, label="informative window")
    ax.set_xlabel("position in sequence"); ax.set_ylabel("share of attribution")
    ax.set_title(title); ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def nucleotide_importance(X, scores, labels=("A", "C", "G", "T"), figsize=(10, 2.6),
                          title="Mean attribution by base and position"):
    """Average attribution split by which base occupies each position."""
    grid = np.zeros((len(labels), X.shape[1]))
    for b, _ in enumerate(labels):
        m = X == (b + 1)                       # ALPHABET: A=1, C=2, G=3, T=4
        with np.errstate(invalid="ignore"):
            grid[b] = np.where(m.sum(0) > 0, (scores * m).sum(0) / np.maximum(m.sum(0), 1), 0)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(grid, aspect="auto", cmap="magma")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("position in sequence"); ax.set_title(title)
    fig.colorbar(im, ax=ax, label="mean attribution")
    fig.tight_layout()
    return fig


def class_distribution(y, labels, figsize=(6, 3)):
    counts = np.bincount(np.asarray(y), minlength=len(labels))
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(labels, counts, color="#2c7fb8")
    for i, c in enumerate(counts):
        ax.text(c, i, f" {c:,} ({c/counts.sum():.1%})", va="center", fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("sequences (log scale)")
    ax.set_title("Class distribution")
    fig.tight_layout()
    return fig


def predicted_vs_observed(pred, obs, labels, figsize=(10, 2.8)):
    """Predicted against observed outcome frequency, one panel per class.

    This is the site level view. Each point is one target site. A model that
    has learned the sequence dependence of repair falls on the diagonal; a
    model that has only learned the overall class frequencies collapses to a
    horizontal band at that frequency.
    """
    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=figsize, sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for c, ax in enumerate(axes):
        r = np.corrcoef(pred[:, c], obs[:, c])[0, 1] if obs[:, c].std() > 0 else np.nan
        ax.scatter(obs[:, c], pred[:, c], s=5, alpha=0.3, edgecolors="none")
        ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="grey")
        ax.set(title=f"{labels[c]}\nr = {r:.2f}", xlim=(-0.03, 1.03), ylim=(-0.03, 1.03),
               xlabel="observed")
        ax.set_aspect("equal")
    axes[0].set_ylabel("predicted")
    fig.tight_layout()
    return fig


def genotype_vocabulary(y, classes, top_n=30, figsize=(10, 3.2)):
    """Genotype frequencies, most common first, on a log scale.

    Repair genotype vocabularies are long tailed: a few outcomes dominate and
    most are rare. The shape matters, because a model can score well on
    accuracy while never predicting anything outside the head.
    """
    counts = np.bincount(np.asarray(y), minlength=len(classes))
    order = np.argsort(-counts)[:top_n]
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(order)), counts[order], color="tab:blue")
    ax.set_yscale("log")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([classes[i] for i in order], rotation=90, fontsize=7)
    ax.set_ylabel("reads (log)")
    ax.set_title(f"Genotype vocabulary, top {len(order)} of {len(classes)} classes")
    fig.tight_layout()
    return fig


def topk_curve(topk: dict, baseline_top1=None, figsize=(5, 3.2)):
    """Top k accuracy against k, with the majority baseline for reference."""
    ks = sorted(topk)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(ks, [topk[k] for k in ks], marker="o", color="tab:blue", label="model")
    if baseline_top1 is not None:
        ax.axhline(baseline_top1, ls="--", lw=1, color="grey",
                   label=f"majority baseline ({baseline_top1:.3f})")
    ax.set(xlabel="k", ylabel="accuracy within top k",
           title="Is the right genotype near the front of the ranking",
           ylim=(0, 1.02))
    ax.set_xticks(ks)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig
