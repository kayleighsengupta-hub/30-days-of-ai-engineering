# Evaluation — does it actually work?

The app has a slider, but "drag it until the groups look right" isn't something
I'd want to defend. So I did the proper version: I built a small labelled set
where I *know* which texts are genuine duplicates, then measured how closely the
tool agrees with me across a range of thresholds. Run it yourself with
`python evaluate.py`.

## What I'm measuring

For each threshold, I look at every possible pair of texts and check whether the
tool called them duplicates, then compare against the truth:

- **Precision** — of the pairs I *flagged* as duplicates, how many really were?
  Low precision means false alarms (grouping things that aren't the same).
- **Recall** — of the *real* duplicate pairs, how many did I catch?
  Low recall means misses (letting real duplicates slip through).
- **F1** — the balance of the two, so I can compare thresholds with one number.

## How to read the result

The two pull against each other, and that tension is the whole point:

- **Low threshold** → catches almost everything (high recall) but starts
  grouping things that only loosely relate (precision drops).
- **High threshold** → only groups very close matches (high precision) but
  quietly misses real duplicates worded differently (recall drops).

So there isn't one "correct" threshold — there's the right one *for what the
mistake costs you*. Cleaning a support queue? Lean toward recall; a few extra
groupings to glance past beats missing duplicate complaints. Auto-merging
records where a wrong merge is painful to undo? Lean toward precision.

## What I take from it

I pick the threshold using the curve, not just the single best F1 — then I can
say exactly why I chose it, which is the part that actually matters. The
labelled set here is deliberately tiny so it's easy to read; a real version
would use more examples so the numbers are less jumpy and more trustworthy.
