# Decisions & tradeoffs

The choices I made building this, what I turned down, and why. The point of
writing these down is partly for me — it forces me to actually have a reason —
and partly so anyone reading can see the thinking, not just the result.

### Embeddings, not keyword matching
The whole job is spotting texts that mean the same thing while using different
words. Keyword/fuzzy matching can't do that — "charged twice" and "double
payment" share nothing. Embeddings work in meaning-space, so this was the
obvious call. The cost is needing a model and losing the ability to point at
*why* two things matched as plainly as "they share these words."

### Agglomerative clustering, not k-means
K-means makes you say how many groups exist up front. For duplicate-finding I
genuinely don't know — could be 2, could be 40. Agglomerative lets me set a
*similarity threshold* instead ("group anything closer than X"), which is the
actual question. Tradeoff: it's O(n²) on distances, so it doesn't scale to huge
inputs without help (see "what's next").

### A small fast model (all-MiniLM-L6-v2)
It runs on a laptop, embeds quickly, and is good enough for short texts. I'd
rather the demo be instant than slightly more accurate and sluggish. If
accuracy mattered more than speed I'd swap in a bigger model and accept the
cost — it's a one-line change.

### Normalising vectors to length 1
Lets me use a plain dot product as cosine similarity, which keeps the code
simple and the clustering metric consistent. No real downside for this use.

## How data flows
```
texts (one per line)
      │  embed each one
      ▼
vectors  (384 numbers each, normalised)
      │  cosine similarity between every pair
      ▼
distance matrix
      │  agglomerative clustering at chosen threshold
      ▼
groups  →  shown as duplicate sets + leftover uniques
```

## Latency
Embedding is the slow-ish step but still fast for a few hundred texts. The real
bottleneck is the pairwise comparison: it grows with the *square* of the number
of texts, so a few hundred is fine and a few hundred thousand is not. For this
demo's scale it's comfortably instant.

## Edge cases & failure modes
- **0 or 1 input line** — nothing to compare; the app asks for at least two.
- **Everything identical** — collapses into one big group, which is correct.
- **Nothing similar** — every item comes back as unique, also correct.
- **Opposite sentiment, same topic** — "love this app" / "hate this app" can
  land close, because embeddings track topic more than tone. Known blind spot.
- **Very long pasted texts** — the model truncates beyond its limit, so very
  long inputs lose their tail. Fine for short tickets, worth noting for essays.

## What I'd do next
- Swap the brute-force comparison for an approximate-nearest-neighbour index
  (FAISS or similar) so it scales past a few thousand texts.
- Show *why* two items grouped (their similarity score) for a bit of trust.
- Let people upload a CSV instead of pasting, for real-world sized batches.
