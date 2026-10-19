"""Sequence encoding, a synthetic benchmark, and group aware splitting."""
from __future__ import annotations

import numpy as np

# Index 0 is reserved for padding, so every real base has a non zero index.
ALPHABET = {"P": 0, "A": 1, "C": 2, "G": 3, "T": 4, "N": 5}
INVERSE = {v: k for k, v in ALPHABET.items()}
VOCAB_SIZE = len(ALPHABET)


def encode(seq: str, length: int | None = None) -> np.ndarray:
    """Integer encode a DNA string. Unknown characters map to N."""
    seq = seq.upper()
    out = [ALPHABET.get(b, ALPHABET["N"]) for b in seq]
    if length is not None:
        out = out[:length] + [ALPHABET["P"]] * max(0, length - len(out))
    return np.asarray(out, dtype=np.int64)


def decode(codes) -> str:
    return "".join(INVERSE[int(c)] for c in codes)


def make_synthetic(n_groups: int = 400, per_group: int = 25, length: int = 60,
                   signal_start: int = 28, signal_width: int = 4,
                   label_noise: float = 0.10, seed: int = 0):
    """A benchmark where the correct answer is known by construction.

    The class of every sequence is decided by a short window at a fixed
    position, chosen to imitate a cut site. Everything else is random.
    A model that works must concentrate its attention on that window,
    which makes this a test of the method rather than a claim about
    biology.

    Returns sequences, labels, group ids and the signal window.
    """
    rng = np.random.default_rng(seed)
    bases = np.array(list("ACGT"))
    sig_end = signal_start + signal_width

    X, y, groups = [], [], []
    for g in range(n_groups):
        # One "site" shares a backbone, as reads from one amplicon would.
        backbone = rng.choice(bases, length)
        motif = rng.choice(bases, signal_width)
        label = int("".join(motif).count("A") + "".join(motif).count("T") >= 3)
        for _ in range(per_group):
            s = backbone.copy()
            s[signal_start:sig_end] = motif
            # Sequencing noise away from the signal window.
            for _ in range(rng.poisson(1.0)):
                i = int(rng.integers(0, length))
                if not (signal_start <= i < sig_end):
                    s[i] = rng.choice(bases)
            X.append(encode("".join(s), length))
            y.append(label if rng.random() > label_noise else 1 - label)
            groups.append(g)
    return np.stack(X), np.asarray(y), np.asarray(groups), (signal_start, sig_end)


def group_split(groups, train=0.7, val=0.15, seed=0):
    """Split by group, never by row.

    Reads from one amplicon are near copies of each other. Splitting rows
    at random puts copies in both train and test and inflates every
    metric. Splitting by group removes that leak.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    n_tr, n_va = int(len(uniq) * train), int(len(uniq) * val)
    sets = (set(uniq[:n_tr]), set(uniq[n_tr:n_tr + n_va]), set(uniq[n_tr + n_va:]))
    idx = [np.array([i for i, g in enumerate(groups) if g in s]) for s in sets]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    return idx  # train, validation, test row indices


# ---------------------------------------------------------------------------
# Benchmarks shaped like real repair outcome tasks.
#
# The rules below are simplified versions of published observations. They
# are used so that the pipeline can be tested against a known answer. They
# are not a model of repair biology, and no biological claim is made here.
# ---------------------------------------------------------------------------

DELETION_CLASSES = ["no_indel", "del_1", "del_2", "del_3", "del_4plus"]
INSERTION_CLASSES = ["no_indel", "ins_A", "ins_C", "ins_G", "ins_T"]


_BASES = np.array(list("ACGT"))


def _backbone(rng, length, cut, shared=None, variable_radius=12):
    """One site's sequence.

    With ``shared`` given, every site is built on the same flanking sequence
    and only the window of ``variable_radius`` bases either side of the cut
    differs between sites. That is what an amplicon library looks like: the
    primer and vector regions are common to every read, and only the
    protospacer region varies. It also stops a model from identifying a site
    by its flanks, which is the easiest way for one of these models to score
    well without learning anything.
    """
    if shared is None:
        return rng.choice(_BASES, length)
    s = shared.copy()
    lo, hi = max(0, cut - variable_radius), min(length, cut + variable_radius)
    s[lo:hi] = rng.choice(_BASES, hi - lo)
    return s


def _add_read_noise(rng, s, length, cut, protect=6, rate=0.8):
    """Sequencing noise, kept away from the informative positions."""
    r = s.copy()
    for _ in range(rng.poisson(rate)):
        i = int(rng.integers(0, length))
        if abs(i - cut) > protect:
            r[i] = rng.choice(_BASES)
    return r


def make_deletion_task(n_groups=500, per_group=30, length=60, cut=30,
                       base_rate=0.15, rate_per_mh=0.18, shared_flank=True,
                       variable_radius=12, microhomology="homopolymer", seed=0):
    """Microhomology at the cut site sets both deletion size and frequency.

    Motivated by the role of microhomology in end joining deletions
    (Bae et al., 2014, *Nature Methods*; Shen et al., 2018, *Nature*).
    Two things are tied to the length of the matching arms flanking the cut:
    the deletion size, and the probability that a read carries a deletion at
    all. The second part matters. If deletion frequency were constant across
    sites, the best possible per read prediction would be the majority class
    for every read, and no model could beat that baseline. Coupling frequency
    to local sequence is what makes the task learnable, and it is also what
    is observed at real cut sites.

    Two ways of building the microhomology are available, and they differ
    sharply in difficulty.

    ``microhomology="homopolymer"`` (default)
        The arms are a run of one base spanning the cut, so ``mh`` bases of
        one nucleotide sit on each side. This is the simplest microhomology
        there is, and the deletion size can be read off the length of the run
        at a fixed position. A recurrent model can learn it.

    ``microhomology="shifted"``
        The ``mh`` bases to the right of the cut are a copy of the ``mh``
        bases to its left, which is the general case. Recovering ``mh`` then
        means comparing two blocks at a variable offset. No single position
        carries the answer. This variant is kept because it is the honest
        version of the problem, and because the model does **not** solve it
        at this sample size, which is worth showing rather than hiding.

    Informative positions are the microhomology arms, roughly
    ``cut - 4`` to ``cut + 4``.

    Returns sequences, labels, group ids, class names and the cut position.
    """
    if microhomology not in ("homopolymer", "shifted"):
        raise ValueError("microhomology must be 'homopolymer' or 'shifted'")
    rng = np.random.default_rng(seed)
    shared = rng.choice(_BASES, length) if shared_flank else None
    X, y, groups = [], [], []
    for g in range(n_groups):
        s = _backbone(rng, length, cut, shared, variable_radius)
        # Force a controlled amount of microhomology across the cut.
        mh = int(rng.integers(0, 5))
        if microhomology == "homopolymer":
            if mh:
                base = rng.choice(_BASES)
                s[cut - mh:cut + mh] = base
                # Close the run off, so the length is unambiguous.
                for i in (cut - mh - 1, cut + mh):
                    if 0 <= i < length:
                        s[i] = rng.choice(_BASES[_BASES != base])
            else:
                s[cut] = rng.choice(_BASES[_BASES != s[cut - 1]])
        elif mh:
            s[cut:cut + mh] = s[cut - mh:cut]
        cls = max(mh, 1)                               # index into DELETION_CLASSES
        p_indel = base_rate + rate_per_mh * mh         # longer arms delete more often
        for _ in range(per_group):
            r = _add_read_noise(rng, s, length, cut)
            label = cls if rng.random() < p_indel else 0
            X.append(encode("".join(r), length))
            y.append(label)
            groups.append(g)
    return np.stack(X), np.asarray(y), np.asarray(groups), DELETION_CLASSES, cut


def make_insertion_task(n_groups=500, per_group=30, length=60, cut=30,
                        p_prone=0.75, p_background=0.04, shared_flank=True,
                        variable_radius=12, seed=0):
    """A one base insertion is templated by the base at position ``cut - 4``.

    Motivated by reports that single base insertions copy the nucleotide
    at the fourth position upstream of the cut (van Overbeek et al., 2016,
    *Molecular Cell*; Lemos et al., 2018, *PNAS*).

    Two separate things have to be learned, which is the point of the task.
    Which base is inserted is set by the identity of position ``cut - 4``.
    How often a site inserts at all is set by local context: sites where
    ``cut - 4`` and ``cut - 3`` carry the same base insert at a high rate,
    the rest at a low background rate. Class imbalance is severe by
    design, as it is in real insertion data.

    Informative positions are ``cut - 4`` and ``cut - 3``.

    Returns sequences, labels, group ids, class names and ``cut - 4``.
    """
    rng = np.random.default_rng(seed)
    shared = rng.choice(_BASES, length) if shared_flank else None
    template_pos = cut - 4
    X, y, groups = [], [], []
    for g in range(n_groups):
        s = _backbone(rng, length, cut, shared, variable_radius)
        cls = INSERTION_CLASSES.index(f"ins_{s[template_pos]}")
        prone = s[template_pos] == s[template_pos + 1]
        p_indel = p_prone if prone else p_background
        for _ in range(per_group):
            r = _add_read_noise(rng, s, length, cut)
            label = cls if rng.random() < p_indel else 0
            X.append(encode("".join(r), length))
            y.append(label)
            groups.append(g)
    return np.stack(X), np.asarray(y), np.asarray(groups), INSERTION_CLASSES, template_pos


def bayes_ceiling(y, groups):
    """Best accuracy any per read classifier can reach on these data.

    Reads in a group share a sequence but not a label, so the most a model
    can do is predict each group's most frequent label. Reporting this
    alongside the majority class baseline shows how much of the remaining
    gap is reachable at all.
    """
    y, groups = np.asarray(y), np.asarray(groups)
    correct = 0
    for g in np.unique(groups):
        counts = np.bincount(y[groups == g])
        correct += counts.max()
    return correct / len(y)


# ---------------------------------------------------------------------------
# Genotype level task.
#
# The two tasks above predict a coarse class: a deletion size, or the identity
# of one inserted base. Published repair outcome models predict the genotype
# itself, meaning the actual sequence inserted or deleted, over a vocabulary
# of hundreds of outcomes (Chen et al., 2019, *Nucleic Acids Research*, 557
# classes in Lindel; Allen et al., 2019, *Nature Biotechnology*, of the order
# of a thousand in FORECasT).
#
# This benchmark reproduces that shape. Genotypes up to `max_unit` bases are
# kept as distinct classes and anything longer falls into one of two bins,
# which is the same vocabulary plus tail arrangement those papers use.
# ---------------------------------------------------------------------------

def _n_purines(seq) -> int:
    return int(sum(b in ("A", "G") for b in seq))


def make_genotype_task(n_groups=6000, per_group=10, length=40, cut=20,
                       max_unit=3, shared_flank=True, variable_radius=12,
                       seed=0):
    """Predict the repair genotype, covering deletions and insertions together.

    Each site gets an outcome distribution decided by the sequence flanking
    the cut, and both the length and the content of the event come from the
    same local sequence, which is what makes the genotype predictable rather
    than arbitrary.

    Length and content come from different places, which is the realistic
    arrangement. Which bases are deleted is not a free choice, it is simply
    whatever sits at the cut. How many are deleted is what depends on wider
    sequence context.

    Deletions
        Length is set by the purine count of a three base context block 5' of
        the cut, and the deleted sequence is the bases at the cut itself.
        Deletions longer than ``max_unit`` collapse into ``del_large``.

    Insertions
        Length is set by a three base context block 3' of the cut, and the
        inserted sequence is a tandem duplication of the bases immediately 5'
        of the cut, following the templated insertion pattern reported at
        Cas9 and Cas12a breaks (van Overbeek et al., 2016, *Molecular Cell*).
        Insertions longer than ``max_unit`` collapse into ``ins_large``.

    A model therefore has to read two separate regions, one that says how long
    the event is and one that says which bases it involves, and combine them
    into a single genotype label.

    Informative positions run from ``cut - 2 * max_unit`` to
    ``cut + 2 * max_unit - 1``.

    Returns sequences, integer labels, group ids, the class vocabulary and
    the cut position.
    """
    rng = np.random.default_rng(seed)
    shared = rng.choice(_BASES, length) if shared_flank else None

    X, labels, groups = [], [], []
    for g in range(n_groups):
        s = _backbone(rng, length, cut, shared, variable_radius)
        # Context blocks that set the length, kept clear of the bases that
        # supply the content, so genotype length and genotype sequence are
        # independent and the full vocabulary can occur.
        del_ctx = s[cut - 2 * max_unit:cut - max_unit]
        ins_ctx = s[cut + max_unit:cut + 2 * max_unit]

        del_len = _n_purines(del_ctx) + 1           # 1 .. max_unit + 1
        ins_len = _n_purines(ins_ctx) + 1
        del_gt = ("del_large" if del_len > max_unit
                  else "del_" + "".join(s[cut:cut + del_len]))
        ins_gt = ("ins_large" if ins_len > max_unit
                  else "ins_" + "".join(s[cut - ins_len:cut]))

        p_del, p_ins = 0.10 + 0.12 * del_len, 0.08 + 0.08 * ins_len
        if p_del + p_ins > 0.85:                   # leave room for no_indel
            scale = 0.85 / (p_del + p_ins)
            p_del, p_ins = p_del * scale, p_ins * scale

        for _ in range(per_group):
            r = _add_read_noise(rng, s, length, cut)
            u = rng.random()
            labels.append(del_gt if u < p_del
                          else ins_gt if u < p_del + p_ins
                          else "no_indel")
            X.append(encode("".join(r), length))
            groups.append(g)

    # Stable vocabulary: no_indel, then deletions, then insertions, then bins.
    seen = set(labels)
    classes = ["no_indel"]
    classes += sorted(c for c in seen if c.startswith("del_") and c != "del_large")
    classes += sorted(c for c in seen if c.startswith("ins_") and c != "ins_large")
    classes += [c for c in ("del_large", "ins_large") if c in seen]
    index = {c: i for i, c in enumerate(classes)}

    y = np.asarray([index[l] for l in labels])
    return np.stack(X), y, np.asarray(groups), classes, cut
