# How this works — in plain English

## The problem I'm solving
You've got a pile of short texts — support tickets, reviews, bug reports — and a load of them are secretly the same thing, just worded differently. "I was charged twice" and "Why is there a double payment on my card?" are the same complaint but barely share a word between them. Keyword matching falls flat here, because what I actually care about is **meaning**, not spelling.

## The three steps

### 1. Embedding — turning meaning into numbers
Every text gets run through an **embedding model** (here, `all-MiniLM-L6-v2`), which spits out a list of 384 numbers — a *vector*. Think of it as coordinates in a 384-dimensional space. The model's been trained so that texts meaning similar things land near each other in that space, no matter which exact words they use.

So "charged twice" and "double payment" end up as close neighbours, while "login keeps spinning" sits somewhere completely different.

### 2. Cosine similarity — measuring how close
To work out how related two texts are, I measure the **angle** between their vectors using cosine similarity:
- Pointing the same way → score near **1** (same meaning)
- At right angles → score near **0** (unrelated)
- Opposite → score near **-1**

Little trick in the code: if you *normalise* every vector to length 1 first, the cosine similarity is just their dot product, one less thing to calculate.

### 3. Clustering — grouping the close ones
I hand all the vectors over to **agglomerative clustering**. It starts with every text in its own little group, then keeps merging the two closest groups together, stopping once nothing left is closer than the threshold I set with the slider.

The slider is a *similarity* threshold (say 0.70). The clustering wants a *distance* threshold instead, and because these are normalised vectors, `distance = 1 - similarity`. That's what the `1 - threshold` line in the code is doing.

## Why agglomerative and not k-means?
K-means makes you say *how many* groups you want up front. With duplicate-finding I've got no idea — could be 2 groups, could be 40. Agglomerative lets me say "group anything closer than X" instead, which is exactly the question I'm actually asking.

## The honest limitation
Embeddings catch *topical* similarity, not logic or tone. "This app is great" and "This app is terrible" can land surprisingly close because they're so alike topically and structurally, even though they mean opposite things. For finding duplicates that's usually fine, but it's good to know where the tool runs out of road.
