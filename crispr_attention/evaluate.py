"""Metrics that stay honest under heavy class imbalance.

Accuracy and weighted averages are dominated by the largest class. When
one class holds 99% of the rows, a model that always predicts it scores
0.99 while learning nothing. Every report here is therefore shown next
to the majority class baseline, and macro averages are included.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score)


def majority_baseline(y_train, y_test) -> dict:
    """Score of always predicting the most common training class."""
    values, counts = np.unique(y_train, return_counts=True)
    majority = values[np.argmax(counts)]
    pred = np.full_like(np.asarray(y_test), majority)
    return {"class": majority,
            "accuracy": accuracy_score(y_test, pred),
            "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_test, pred)}


def evaluate(y_true, y_pred, y_train=None, target_names=None, digits=3) -> dict:
    """Read level metrics, with the majority baseline alongside.

    ``labels`` is pinned to the full class list rather than inferred from the
    data. With a large vocabulary some classes are absent from a given test
    split, and letting scikit-learn infer the set then silently renumbers the
    classes, misaligns ``target_names`` and changes what macro F1 averages
    over.
    """
    labels = range(len(target_names)) if target_names is not None else None
    macro_kw = {} if labels is None else {"labels": list(labels)}
    out = {"accuracy": accuracy_score(y_true, y_pred),
           "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0,
                                **macro_kw),
           "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0,
                                   **macro_kw),
           "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
           "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels)}
    out["report"] = classification_report(y_true, y_pred, labels=labels,
                                          target_names=target_names,
                                          zero_division=0, digits=digits)
    if y_train is not None:
        out["baseline"] = majority_baseline(y_train, y_true)
        out["beats_baseline_accuracy"] = out["accuracy"] > out["baseline"]["accuracy"]
        out["beats_baseline_macro_f1"] = out["macro_f1"] > out["baseline"]["macro_f1"]
    return out


def site_evaluate(proba, y_true, groups, n_classes=None) -> dict:
    """Evaluate at the target site, not the read.

    Published repair outcome models predict the **distribution** of outcomes
    at a site rather than a class for a single read (Shen et al., 2018,
    *Nature*, inDelphi; Allen et al., 2019, *Nature Biotechnology*,
    FORECasT). The reason is structural. Reads from one site share a
    sequence but not an outcome, so a per read label is only a draw from the
    site's distribution and much of the per read error is irreducible.

    Predicted class probabilities are averaged over a site's reads and
    compared with the observed outcome frequencies at that site.

    Returns the modal class accuracy, the correlation pooled over all site
    and class pairs, the mean correlation computed within a site, and the
    two matrices so they can be plotted.
    """
    proba, y_true, groups = np.asarray(proba), np.asarray(y_true), np.asarray(groups)
    n_classes = n_classes or proba.shape[1]
    sites = np.unique(groups)

    pred_mat = np.zeros((len(sites), n_classes))
    obs_mat = np.zeros((len(sites), n_classes))
    n_reads = np.zeros(len(sites), dtype=int)
    for i, g in enumerate(sites):
        m = groups == g
        pred_mat[i] = proba[m].mean(0)
        obs_mat[i] = np.bincount(y_true[m], minlength=n_classes) / m.sum()
        n_reads[i] = m.sum()

    within = [np.corrcoef(p, o)[0, 1] for p, o in zip(pred_mat, obs_mat) if o.std() > 0]
    return {"n_sites": len(sites),
            "median_reads_per_site": int(np.median(n_reads)),
            "modal_accuracy": float(np.mean(pred_mat.argmax(1) == obs_mat.argmax(1))),
            "pooled_pearson_r": float(np.corrcoef(pred_mat.ravel(), obs_mat.ravel())[0, 1]),
            "mean_within_site_r": float(np.mean(within)),
            "mean_absolute_error": float(np.abs(pred_mat - obs_mat).mean()),
            "predicted": pred_mat, "observed": obs_mat}


def site_summary(res: dict) -> str:
    return "\n".join([
        f"target sites               {res['n_sites']:,}"
        f"  (median {res['median_reads_per_site']} reads each)",
        f"modal outcome accuracy     {res['modal_accuracy']:.4f}",
        f"pooled Pearson r           {res['pooled_pearson_r']:.4f}",
        f"mean within site Pearson r {res['mean_within_site_r']:.4f}",
        f"mean absolute error        {res['mean_absolute_error']:.4f}",
    ])


def summary(res: dict) -> str:
    lines = [f"accuracy           {res['accuracy']:.4f}",
             f"balanced accuracy  {res['balanced_accuracy']:.4f}",
             f"macro F1           {res['macro_f1']:.4f}",
             f"weighted F1        {res['weighted_f1']:.4f}"]
    if "baseline" in res:
        b = res["baseline"]
        lines += ["", f"majority baseline ('{b['class']}')",
                  f"  accuracy         {b['accuracy']:.4f}",
                  f"  macro F1         {b['macro_f1']:.4f}", "",
                  f"beats baseline on accuracy : {res['beats_baseline_accuracy']}",
                  f"beats baseline on macro F1 : {res['beats_baseline_macro_f1']}"]
    return "\n".join(lines)


def topk_accuracy(y_true, proba, ks=(1, 3, 5, 10)) -> dict:
    """Fraction of reads whose true class is in the model's top k.

    With a genotype vocabulary of hundreds of classes, top 1 accuracy alone
    understates a model that consistently places the right genotype near the
    front of its ranking. Lindel and FORECasT are judged the same way.
    """
    y_true = np.asarray(y_true)
    order = np.argsort(-np.asarray(proba), axis=1)
    return {k: float(np.mean((order[:, :k] == y_true[:, None]).any(axis=1)))
            for k in ks if k <= proba.shape[1]}


def prior_baseline(y_train, y_true, groups, n_classes) -> dict:
    """The null model for distribution prediction.

    Predicts the overall training genotype frequencies for every read,
    ignoring sequence entirely. A sequence model has to beat this on the site
    level correlation, not just on accuracy, or it has learned the vocabulary
    rather than the sequence dependence. This is the comparison that says
    whether a genotype model is doing anything.
    """
    prior = np.bincount(np.asarray(y_train), minlength=n_classes).astype(float)
    prior /= prior.sum()
    proba = np.repeat(prior[None, :], len(np.asarray(y_true)), axis=0)
    res = site_evaluate(proba, y_true, groups, n_classes=n_classes)
    return {"accuracy": float(np.mean(np.asarray(y_true) == int(prior.argmax()))),
            "site_pooled_pearson_r": res["pooled_pearson_r"],
            "site_modal_accuracy": res["modal_accuracy"],
            "site_mean_absolute_error": res["mean_absolute_error"]}
