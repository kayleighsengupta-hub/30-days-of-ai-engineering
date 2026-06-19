# How this works — in plain English

## The problem
You have a pile of short texts — support tickets, reviews, bug reports — and
lots of them are *secretly the same issue* worded differently. "I was charged
twice" and "Why is there a double payment on my card?" are the same complaint
but share almost no words. Keyword matching can't catch that. You need
something that understands **meaning**, not spelling.

## The three steps

### 1. Embedding — turn meaning into numbers
Every text gets fed through an **embedding model** (here, `all-MiniLM-L6-v2`),
which outputs a list of 384 numbers — a *vector*. Think of it as coordinates in
a 384-dimensional space. The model was trained so that texts with similar
meaning land near each other in that space, regardless of the exact words.

So "charged twice" and "double payment" end up as near-neighbours, while
"login keeps spinning" lands somewhere far away.

### 2. Cosine similarity — measure closeness
To decide how related two texts are, we measure the **angle** between their
vectors using cosine similarity:
- Same direction → score near **1** (same meaning)
- Right angle → score near **0** (unrelated)
- Opposite → score near **-1**

Trick used in the code: if you *normalise* every vector to length 1 first, the
cosine similarity is just their dot product. One less step to compute.

### 3. Clustering — group the close ones
We hand all the vectors to **agglomerative clustering**. It starts with every
text in its own group, then repeatedly merges the two closest groups, stopping
when nothing left is closer than the threshold you set with the slider.

The slider is a *similarity* threshold (e.g. 0.70). The clustering algorithm
wants a *distance* threshold, and since these are normalised vectors,
`distance = 1 - similarity`. That's the `1 - threshold` line in the code.

## Why agglomerative and not k-means?
K-means makes you say *how many* groups you want up front. With duplicate
detection you have no idea — could be 2 groups, could be 40. Agglomerative lets
you instead say "group anything closer than X," which is exactly the question
we're actually asking.

## The honest limitation
Embeddings capture *topical/semantic* similarity, not logic or sentiment.
"This app is great" and "This app is terrible" can sit suspiciously close
because they're structurally and topically near-identical. For true duplicate
*detection* that's usually fine; just know the tool's edge.
