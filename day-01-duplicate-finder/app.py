"""
Day 1 — Duplicate / Near-Duplicate Finder
==========================================

Paste in a list of short texts (support tickets, reviews, bug reports).
The tool groups the ones that *mean* the same thing, even when they share
no words. This is semantic clustering built on top of embeddings.

The whole pipeline:
  1. Embed every text into a vector (meaning -> coordinates).
  2. Measure how close every pair of vectors is (cosine similarity).
  3. Group texts whose similarity crosses a threshold you control.

Run with:  streamlit run app.py
"""

import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering


# ---- 1. Load the embedding model (cached so it loads once) ----
@st.cache_resource
def load_model():
    # A small, fast, well-respected general-purpose embedding model.
    # 384-dimensional vectors — plenty for this job, quick on a laptop.
    return SentenceTransformer("all-MiniLM-L6-v2")


def embed(texts, model):
    """Turn a list of strings into a matrix of vectors (one row each)."""
    # normalize_embeddings=True makes every vector length 1, which means
    # a plain dot product == cosine similarity. Small trick, saves a step.
    return model.encode(texts, normalize_embeddings=True)


def cluster(vectors, threshold):
    """
    Group vectors that are close together.

    We feed the model a *distance* threshold. With normalized vectors,
    cosine distance = 1 - cosine similarity. So a similarity threshold of
    0.75 becomes a distance threshold of 0.25.
    """
    distance_threshold = 1 - threshold
    model = AgglomerativeClustering(
        n_clusters=None,                 # let the threshold decide how many groups
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",
    )
    labels = model.fit_predict(vectors)
    return labels


# ---- UI ----
st.set_page_config(page_title="Duplicate Finder", page_icon="◆", layout="centered")

st.title("Near-Duplicate Finder")
st.caption(
    "Groups texts that mean the same thing — even when they share no words. "
    "Built on sentence embeddings + cosine similarity."
)

DEFAULT = """App crashes when I tap the save button
Payment failed but I was still charged
The save icon does nothing on iOS
Login screen keeps spinning forever
I got billed twice for one order
Can't sign in, it just loads endlessly
Saving a draft force-closes the app
Why was my card charged two times?"""

text = st.text_area(
    "One text per line",
    value=DEFAULT,
    height=220,
    help="Paste support tickets, reviews, bug reports — anything short.",
)

threshold = st.slider(
    "How similar counts as a duplicate?",
    min_value=0.40,
    max_value=0.95,
    value=0.70,
    step=0.01,
    help="Higher = stricter (only very close matches group together).",
)

if st.button("Find duplicates", type="primary"):
    items = [line.strip() for line in text.splitlines() if line.strip()]

    if len(items) < 2:
        st.warning("Add at least two lines so there's something to compare.")
    else:
        model = load_model()
        with st.spinner("Embedding and grouping…"):
            vectors = embed(items, model)
            labels = cluster(vectors, threshold)

        # Organise items by their assigned group.
        groups = {}
        for item, label in zip(items, labels):
            groups.setdefault(label, []).append(item)

        # Show multi-item groups first (those are the actual duplicates).
        ordered = sorted(groups.values(), key=len, reverse=True)
        dupe_groups = [g for g in ordered if len(g) > 1]
        singles = [g[0] for g in ordered if len(g) == 1]

        st.subheader(f"{len(dupe_groups)} duplicate group(s) found")

        for i, group in enumerate(dupe_groups, 1):
            with st.container(border=True):
                st.markdown(f"**Group {i}** — {len(group)} items mean the same thing")
                for member in group:
                    st.markdown(f"- {member}")

        if singles:
            with st.expander(f"{len(singles)} unique item(s) — no match found"):
                for s in singles:
                    st.markdown(f"- {s}")
