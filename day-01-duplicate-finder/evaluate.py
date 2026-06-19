"""
evaluate.py — measuring whether the duplicate finder actually works
===================================================================

The app has a threshold slider, but "move it until it looks right" isn't an
answer I'd want to defend. This script does the honest version: I hand-label a
small set where I *know* which texts are duplicates, then measure how well the
tool agrees with me across a range of thresholds — and pick a number with
evidence behind it.

What it measures, per threshold:
  - Precision: of the pairs we *called* duplicates, how many really were?
               (high precision = few false alarms)
  - Recall:    of the real duplicate pairs, how many did we *catch*?
               (high recall = few misses)
  - F1:        the balance of the two (one number to compare thresholds by)

Run with:  python evaluate.py
"""

import numpy as np
from itertools import combinations
from sentence_transformers import SentenceTransformer


# ---- The labelled set: my ground truth ----
# Each inner list is a group of texts that MEAN the same thing.
# Single-item lists are unique (they should match nothing).
GROUPS = [
    ["App crashes when I tap save", "Saving a draft force-closes the app", "The save button kills the app"],
    ["I was charged twice", "Why is there a double payment on my card?", "Got billed two times for one order"],
    ["Login screen spins forever", "Can't sign in, it just keeps loading"],
    ["Please add a dark mode"],            # unique
    ["Can you support CSV export?"],       # unique
    ["The app is really slow on older phones"],  # unique
]


def build_truth(groups):
    """Flatten into a list of texts + the set of index-pairs that are true duplicates."""
    texts, owner = [], []
    for gi, group in enumerate(groups):
        for t in group:
            texts.append(t)
            owner.append(gi)
    true_pairs = set()
    for a, b in combinations(range(len(texts)), 2):
        if owner[a] == owner[b]:
            true_pairs.add((a, b))
    return texts, true_pairs


def score(pred_pairs, true_pairs):
    tp = len(pred_pairs & true_pairs)
    fp = len(pred_pairs - true_pairs)
    fn = len(true_pairs - pred_pairs)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def main():
    texts, true_pairs = build_truth(GROUPS)
    print(f"{len(texts)} texts, {len(true_pairs)} true duplicate pairs\n")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    vecs = model.encode(texts, normalize_embeddings=True)

    # Precompute cosine similarity for every pair (normalised -> dot product).
    pairs = list(combinations(range(len(texts)), 2))
    sims = {(a, b): float(np.dot(vecs[a], vecs[b])) for a, b in pairs}

    print(f"{'thresh':>7} {'precision':>10} {'recall':>8} {'F1':>6}")
    print("-" * 34)
    best = None
    for t in [round(x, 2) for x in np.arange(0.45, 0.91, 0.05)]:
        pred = {p for p, s in sims.items() if s >= t}
        precision, recall, f1 = score(pred, true_pairs)
        print(f"{t:>7} {precision:>10.2f} {recall:>8.2f} {f1:>6.2f}")
        if best is None or f1 > best[1]:
            best = (t, f1)

    print(f"\nBest threshold by F1: {best[0]} (F1={best[1]:.2f})")
    print(
        "\nRead the curve, don't just grab the top F1: low thresholds catch\n"
        "everything but raise false alarms (precision drops); high thresholds\n"
        "are safe but miss real duplicates (recall drops). The right pick\n"
        "depends on which mistake costs more for your use case."
    )


if __name__ == "__main__":
    main()
