#!/usr/bin/env python
"""Score new sequences with a trained model.

This is the deployment entry point. It loads a checkpoint written by the
notebooks and writes one row per input sequence, with the predicted class,
the probability of every class, and the position the model attended to
most. It accepts CSV or FASTA.

    python predict.py --model models/deletion_model.pt \
                      --input examples/sequences.csv --seq-col sequence \
                      --output predictions.csv

    python predict.py --model models/insertion_model.pt \
                      --input examples/sequences.fa --output predictions.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from crispr_attention import AttentionBiLSTM, encode, interpret


def read_fasta(path):
    names, seqs, name, buf = [], [], None, []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith(">"):
            if name is not None:
                names.append(name); seqs.append("".join(buf))
            name, buf = line[1:].split()[0], []
        elif line:
            buf.append(line)
    if name is not None:
        names.append(name); seqs.append("".join(buf))
    return names, seqs


def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = AttentionBiLSTM(num_classes=len(ckpt["classes"]), **ckpt["model_kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt["classes"], ckpt["seq_length"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="checkpoint written by a notebook")
    ap.add_argument("--input", required=True, help="CSV or FASTA of sequences")
    ap.add_argument("--seq-col", default="sequence", help="CSV column holding the sequence")
    ap.add_argument("--id-col", default=None, help="CSV column holding an identifier")
    ap.add_argument("--output", default="predictions.csv")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, classes, seq_length = load_model(args.model, device)

    if args.input.lower().endswith((".fa", ".fasta", ".fna")):
        ids, seqs = read_fasta(args.input)
    else:
        df = pd.read_csv(args.input)
        if args.seq_col not in df.columns:
            sys.exit(f"column '{args.seq_col}' not found. Columns: {list(df.columns)}")
        seqs = df[args.seq_col].astype(str).tolist()
        ids = df[args.id_col].astype(str).tolist() if args.id_col else [f"seq_{i}" for i in range(len(seqs))]

    lengths = {len(s) for s in seqs}
    if any(l != seq_length for l in lengths):
        print(f"note: model expects {seq_length} bp; inputs are {sorted(lengths)}. "
              "Longer sequences are truncated and shorter ones padded.", file=sys.stderr)

    X = np.stack([encode(s, seq_length) for s in seqs])
    with torch.no_grad():
        probs = []
        for i in range(0, len(X), args.batch_size):
            xb = torch.tensor(X[i:i + args.batch_size]).to(device)
            probs.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    probs = np.concatenate(probs)
    attn = interpret.attention_weights(model, X, batch_size=args.batch_size, device=device)

    summary = pd.DataFrame({"id": ids,
                            "predicted_class": [classes[i] for i in probs.argmax(1)],
                            "confidence": probs.max(1).round(4),
                            "top_attended_position": attn.argmax(1)})
    # Built in one go: a genotype vocabulary can run to hundreds of columns,
    # and inserting them one at a time fragments the frame.
    per_class = pd.DataFrame(probs.round(4), columns=[f"p_{n}" for n in classes])
    out = pd.concat([summary, per_class], axis=1)
    out.to_csv(args.output, index=False)
    print(f"wrote {len(out)} predictions to {args.output}")
    print(out.head().to_string(index=False))


if __name__ == "__main__":
    main()
