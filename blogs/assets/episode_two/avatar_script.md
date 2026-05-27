# Avatar Script — Episode 02: Trust the Batch

> Spoken script for avatar model. Each chunk is a distinct segment.
> Keep delivery natural and conversational. Pause between chunks.
>
> **Pronunciation guide** (write phonetically in TTS input as shown):
> - FeynRL → "Fine R L"
> - P3O → "P three O"
> - GRPO → "G R P O"
> - DAPO → "D A P O"
> - GSPO → "G S P O"
> - ESS → "E S S" (or say "Effective Sample Size" in full on first use)
> - FP8 → "F P eight"
> - BF16 → "B F sixteen"
> - Qwen3-4B → "Kwen three, four B"

---

## Chunk 1 — The Problem with Off-Policy Data

When you train a large language model with reinforcement learning, the data your optimizer sees is almost never perfectly fresh.

Here's why: you take multiple training steps on the same batch of responses. Your model generating the responses and your model being trained are running at the same time — so by the time a batch arrives, the model has already moved on. Sometimes rollouts are even generated in a lower-precision format to save compute, which shifts the numbers further.

All of this is called "off-policy mismatch." Current methods deal with it by adding more knobs — more hyperparameters you have to tune before training even starts.

We take the opposite approach.

---

## Chunk 2 — The Fix: Measure, Don't Guess

Instead of picking a fixed safety margin before training and hoping it holds, our method measures how stale the current batch is — and uses that measurement to automatically adjust the update.

The measurement is a single number we call the Effective Sample Size, or E S S. Think of it as a freshness score for your data. When the batch is fresh and on-policy, E S S is close to one and training proceeds normally. When the batch is stale or mismatched, E S S drops, and the update automatically tightens — no manual retuning needed.

---

## Chunk 3 — What P Three O Does

Our new algorithm, P three O, uses this freshness score to do two things at once.

First, it caps how much weight any single response can have on the gradient — but the cap is set by the batch freshness, not a fixed number. Second, it adds a stabilizing penalty that grows stronger the staler the batch is, and disappears entirely when the batch is fresh.

Every response contributes to training. No data is thrown away.

Compare this to existing methods — G R P O, D A P O, G S P O — which all require you to set a fixed clip range before training. P three O replaces that entirely with the adaptive batch score.

---

## Chunk 4 — Fine R L Made This Possible

All experiments in our paper were run on Fine R L, the post-training framework we introduced in Episode One.

Because Fine R L cleanly separates the algorithm from the rollout engine and distributed training infrastructure, going from G R P O to P three O was a single-file change. We only touched the loss function. Rollouts, orchestration, distributed training — all untouched.

Fine R L version zero point one is now out of beta with full support for both synchronous and asynchronous training, low-precision rollout engines, and the P three O objective.

---

## Chunk 5 — Result: Clip Sensitivity

Here's the first result. We swept different clip values for G R P O on Kwen three, four B Thinking and compared against a single P three O run.

The spread is dramatic. With a small clip value, G R P O collapses by step 35. With a larger one, it stays above 0.5. P three O matches or exceeds the best clip setting — without any clip parameter at all.

You shouldn't need to tune your optimizer to get stable training. P three O doesn't require it.

---

## Chunk 6 — Result: Temperature Shift

When you sample rollouts at a non-standard temperature — say, 1.2 instead of 1.0 — every token's probability is shifted by a constant factor. For G R P O, this offset either fits inside or falls outside the fixed clip window. It's a bias that can't be corrected without retuning.

P three O's freshness score directly detects this shift and adapts on the fly.

At temperature 1.2, G R P O rises to around 0.70 reward and then collapses to nearly zero by step 32. P three O reaches the same peak and stays stable through the full run.

---

## Chunk 7 — Result: F P Eight Mixed Precision

This is the most practically important result.

A common production setup is to generate rollouts in low-precision F P eight for speed, while training in the standard higher-precision format. The problem is that these two produce different token probabilities. It's off-policy data created at the systems level — and you can't avoid it without giving up the throughput gains.

Under F P eight rollouts, G R P O's reward collapses from 0.54 to 0.02. That's a 96% drop. P three O holds steady at 0.52. The only difference is the objective.

This collapse shows up on held-out benchmarks too. By iteration 30, G R P O is near zero on every math benchmark we tested. P three O at iteration 30 actually improves over iteration 15 — continued training is beneficial rather than destructive.

---

## Chunk 8 — What This Means

The same story plays out across all three experiments: fixed clipping is brittle, and the failure mode depends on which direction things go wrong.

P three O handles all three regimes — clip sensitivity, temperature shift, and F P eight precision mismatch — with the same objective, the same code, and zero hyperparameter changes.

For practitioners: if you're using F P eight rollouts, async pipelines, or reusing rollouts across epochs, you don't need to retune anything. P three O handles the mismatch automatically. Off-policy data becomes a feature rather than a problem.

---

## Chunk 9 — Try It Yourself

P three O is available in Fine R L today. It's a single file, the loss is a single function, and switching from G R P O is a one-line config change.

Links to the paper, the P three O algorithm documentation, and the Fine R L repository are all in the episode description. Give it a try.

---
