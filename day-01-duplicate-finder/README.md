# Day 01 — Near-Duplicate Finder

A little tool that finds texts saying the same thing, even when they don't share any of the same words. Behind the scenes it's using sentence embeddings, cosine similarity, and a bit of clustering to pull it off.

**Real-world use:** tidying up a support queue, review pile, or bug tracker where the same issue keeps getting reported in different wording.

**What I learned today:** vector embeddings, semantic similarity, unsupervised clustering — and how to actually *evaluate* a system instead of eyeballing it.

## How the data flows

```
texts → embed each → vectors → compare every pair (cosine) → cluster by threshold → duplicate groups
```

Full reasoning behind each step is in `DECISIONS.md`.

## Running it

```bash
pip install streamlit sentence-transformers scikit-learn
streamlit run app.py
```

The first run downloads the embedding model (~80 MB), then everything's local after that.

To see the evaluation (precision/recall across thresholds on a labelled set):

```bash
python evaluate.py
```

## Having a play

Paste in a few short texts — support tickets, reviews, bug reports, whatever — one per line, and use the slider to decide how close two things have to be before they count as duplicates. The sample data has three hidden pairs in it plus a few odd ones out to keep it honest.

## What's in here

- `app.py` — the tool itself
- `evaluate.py` — measures whether it actually works, on a labelled set
- `EXPLAINER.md` — how it works, in plain English
- `DECISIONS.md` — the choices I made, what I rejected, data flow, latency, edge cases
- `EVALUATION.md` — what I measured and how to read it
