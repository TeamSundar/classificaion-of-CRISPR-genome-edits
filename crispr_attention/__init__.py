"""Attention based sequence model for CRISPR repair outcome classification."""
from .model import AttentionBiLSTM
from .data import (ALPHABET, encode, decode, make_synthetic, group_split,
                   make_deletion_task, make_insertion_task, make_genotype_task,
                   bayes_ceiling, DELETION_CLASSES, INSERTION_CLASSES)
from .evaluate import (evaluate, majority_baseline, summary,
                       site_evaluate, site_summary, topk_accuracy,
                       prior_baseline)
from .train import fit, predict, predict_proba
from . import interpret, plots

__all__ = [
    # model
    "AttentionBiLSTM",
    # data
    "ALPHABET", "encode", "decode", "group_split",
    "make_synthetic", "make_deletion_task", "make_insertion_task",
    "make_genotype_task",
    "bayes_ceiling", "DELETION_CLASSES", "INSERTION_CLASSES",
    # training and inference
    "fit", "predict", "predict_proba",
    # evaluation and interpretation
    "evaluate", "majority_baseline", "summary",
    "site_evaluate", "site_summary", "topk_accuracy", "prior_baseline",
    "interpret", "plots",
]
__version__ = "0.1.0"
