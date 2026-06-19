# Day 01 — Near-Duplicate Finder

Groups short texts that mean the same thing, even when they share no words.
Built on sentence embeddings + cosine similarity + agglomerative clustering.

**Skill showcased:** vector embeddings, semantic similarity, unsupervised clustering.

## Run it

```bash
pip install streamlit sentence-transformers scikit-learn
streamlit run app.py
```

The first run downloads the embedding model (~80 MB) once, then it's local.

## Try it
Paste support tickets, reviews, or bug reports — one per line — and move the
slider to control how strict "duplicate" means. The default sample data has
three hidden duplicate pairs (a save-crash, a double-billing, a login issue)
plus distractors.

## Files
- `app.py` — the tool
- `EXPLAINER.md` — how it works, in plain English
- `DEFEND.md` — interview questions + answers
