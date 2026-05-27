# Post-Edit Script — Episode 02: Trust the Batch

> Use this alongside the avatar recording to know when to cut in visuals,
> overlays, and B-roll. Times are approximate — sync to avatar speech cues.

---

## Segment 1 — The Problem with Off-Policy Data

**Avatar says:** "When you train a large language model with reinforcement learning…"

- At **"you take multiple training steps"** → cut in `offpolicy_sources.png`
  - Hold for ~5 seconds, then return to avatar
- At **"sometimes rollouts are generated in a lower-precision format"** → add text overlay:
  `FP8 rollouts ≠ BF16 training policy`

---

## Segment 2 — The Fix: Measure, Don't Guess

**Avatar says:** "Instead of picking a fixed safety margin…"

- At **"a single number we call the Effective Sample Size"** → insert animated callout:

  ```
  ESS = freshness score of the batch
        • ESS ≈ 1  →  batch is on-policy  →  full update
        • ESS ↓    →  batch is stale      →  tighter update
  ```

- Return to avatar. No image cutaway needed here — the concept is verbal.

---

## Segment 3 — What P3O Does

**Avatar says:** "Our new algorithm, P three O, uses this freshness score…"

- At **"caps how much weight any single response can have"** → insert comparison table overlay:

  | Method | Clip Range | Adaptive? | New Hyperparameters |
  |--------|-----------|-----------|---------------------|
  | GRPO   | fixed ε   | No        | ε                   |
  | DAPO   | fixed εₗ, εₕ | No     | εₗ, εₕ             |
  | GSPO   | sequence-level ε | No | ε^seq              |
  | **P3O** | **batch ESS** | **Yes** | **None**        |

- Hold table for ~6 seconds, then return to avatar.

---

## Segment 4 — FeynRL Made This Possible

**Avatar says:** "All experiments in our paper were run on Fine R L…"

- At **"single-file change"** → cut in the split diff graphic from the blog:
  - Left panel: GRPO loss (clip + clamp)
  - Right panel: P3O loss (ESS + adaptive KL)
  - Highlight the added lines in green on the right side
  - Hold for ~6 seconds

- At **"Fine R L version zero point one is now out of beta"** → add text overlay:
  `FeynRL v0.1 — out of beta`
  `Sync · Async · FP8 · P3O`

---

## Segment 5 — Result: Clip Sensitivity

**Avatar says:** "We swept different clip values for G R P O…"

- **Immediately** cut in `clip_sensitivity_reward.png`
- Add annotation overlays on the chart:
  - Arrow pointing to GRPO ε=0.2 collapse: `"Collapses to 0.13 by step 35"`
  - Arrow pointing to P3O line: `"P3O — no clip parameter"`
- Hold chart through remainder of segment
- Return to avatar for transition

---

## Segment 6 — Result: Temperature Shift

**Avatar says:** "When you sample rollouts at a non-standard temperature…"

- At **"At temperature 1.2"** → cut in `temp_robustness_reward.png`
- Add annotation overlays:
  - Arrow to GRPO collapse: `"GRPO collapses to ~0.09 by step 32"`
  - Arrow to P3O line: `"P3O stays stable"`
- Hold chart through rest of segment
- Return to avatar for transition

---

## Segment 7 — Result: FP8 Mixed Precision

**Avatar says:** "This is the most practically important result…"

- At **"Under F P eight rollouts"** → cut in `fp8_reward_curves.png`
- Add large text callout overlay:

  ```
  ┌─────────────────────────────────┐
  │  FP8 Rollouts                   │
  │  GRPO:  0.54 → 0.02  (−96%)    │
  │  P3O:   0.52 → 0.52  (stable)  │
  └─────────────────────────────────┘
  ```

- Hold chart for ~5 seconds

- At **"held-out benchmarks"** → cut in benchmark table:

  | Method | AIME24 | AIME25 | AIME26 | AMC |
  |--------|--------|--------|--------|-----|
  | Baseline | 0.371 | 0.471 | 0.396 | 0.618 |
  | FP8 GRPO iter 30 | 0.002 | 0.000 | 0.002 | 0.029 |
  | **FP8 P3O iter 30** | **0.173** | **0.254** | **0.175** | **0.529** |

  - Highlight GRPO iter 30 row in red
  - Highlight P3O iter 30 row in green
  - Hold for ~7 seconds

- Return to avatar

---

## Segment 8 — What This Means

**Avatar says:** "The same story plays out across all three experiments…"

- At **"P three O handles all three regimes"** → insert three-column summary graphic:

  ```
  Clip Sensitivity   |  Temperature Shift  |  FP8 Precision
  GRPO:  brittle     |  GRPO:  collapses   |  GRPO:  −96%
  P3O:   stable      |  P3O:   stable      |  P3O:   stable
  ```

- At **"Off-policy data becomes a feature"** → add pull-quote overlay:
  `"Off-policy data: a feature, not a burden"`

---

## Segment 9 — Try It Yourself

**Avatar says:** "P three O is available in Fine R L today…"

- Show end card with:
  - FeynRL GitHub repo URL (from paper/blog)
  - Paper link: arxiv.org/pdf/2605.12380
  - "P3O in FeynRL" → link to algorithm README
  - Subscribe / follow CTA

---

## Assets Checklist

| File | Used In |
|------|---------|
| `offpolicy_sources.png` | Segment 1 |
| `clip_sensitivity_reward.png` | Segment 5 |
| `temp_robustness_reward.png` | Segment 6 |
| `fp8_reward_curves.png` | Segment 7 |
| `fp8_pass_at_k.png` | Optional — Segment 7 extended cut |
| `clip_pass_at_k.png` | Optional — Segment 5 extended cut |

---
