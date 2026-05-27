# Image Generation Prompt — Opening Schematic (Episode 02)

## Context

This image replaces the opening schematic of the blog post "Trust the Batch." It should work as a
cinematic, editorial-quality hero image that communicates the core tension of the paper: three
distinct experimental perturbations each destabilize a fixed-clip optimizer in a different way,
while a single adaptive measurement (the Effective Sample Size) handles all of them. The three
experiments are: (1) sweeping the clip hyperparameter ε across {0.2, 0.4, 0.6}, (2) shifting
rollout temperature to T=1.2, and (3) generating rollouts in FP8 while training in BF16.

---

## Prompt

A dark, cinematic scientific editorial illustration, wide-aspect (16:9), for a machine learning
research blog post. The overall mood is that of a high-quality science magazine cover — think
Quanta Magazine or a Nature Physics spread — with a deep navy-to-black background (#050a14),
accented by a warm teal (#0f9b8e) and a burnt coral-red (#c0392b). Flat design with no 3D
gradients on objects, but atmospheric depth in the background.

**Composition:** The illustration is divided into three distinct "specimen panels" arranged
side-by-side across the upper two-thirds of the image, each inside a thin-bordered rectangular
frame with a faintly different background tint. Below all three panels, a single unifying band
runs the full width of the image. The lower-left corner has a small wordmark "FeynRL · Episode 02"
in a light monospace serif.

---

### Panel 1 — "The Clip Sweep"  *(left third)*

Background tint: very dark burgundy (#120608).

The panel depicts a single rigid metal bracket or clamp — like a precision measurement caliper —
with its jaws set to a fixed opening. The jaws represent the clip interval [1−ε, 1+ε]. Inside the
bracket, three overlapping translucent silk-like ribbons flow horizontally in coral, rose, and
dusty amber, each representing a training run at ε=0.2, ε=0.4, ε=0.6. The ribbons behave
differently: the tightest-jaw ribbon (ε=0.2) is kinked and crushed, the loosest (ε=0.6) billows
past the jaw uncontrolled, and the middle one wobbles. At the right edge of the panel, the ribbon
ends are frayed — some high, some low — illustrating divergent outcomes. A tiny engraved label on
the caliper reads "ε: fixed before training." The panel title in the top-left corner of the frame,
in dim monospace: "Clip Sensitivity."

---

### Panel 2 — "The Temperature Shift"  *(center third)*

Background tint: very dark warm charcoal (#0d0b08).

The panel shows a stylized probability density curve — smooth, bell-shaped, rendered as a glowing
amber ribbon — representing the distribution of per-token importance ratios ρ_t when rollouts are
sampled at temperature T=1.2. The curve is shifted rightward from center, partially overhanging a
thin vertical dashed line labeled "clip boundary (fixed)" in faint red. On the left side of the
dashed line, the curve's tail is clipped away, shown as a shadow/ghost of the missing mass. A
second, dimmer curve in blue-grey shows the original T=1.0 distribution centered on the line, for
contrast. Above the misaligned curve, a small thermometer icon with a red reading slightly above
the midpoint. Floating in the upper right of the panel, small italic text: "log π rescales
uniformly · clip does not." Panel title: "Temperature Shift."

---

### Panel 3 — "FP8 Precision Mismatch"  *(right third)*

Background tint: very dark slate (#080c12).

The panel is split diagonally from top-left to bottom-right into two zones. The upper-left zone is
labeled "BF16 (train)" in small teal text and shows a clean, high-fidelity waveform — smooth
sinusoidal oscillations representing token log-probabilities, rendered in glowing teal. The
lower-right zone is labeled "FP8 (rollout)" in coral and shows the same waveform but visibly
quantized — stepped, blocky, with rounding artifacts, like a lo-fi version of the teal wave. Where
the two zones meet along the diagonal edge, a thin bright line shows the mismatch gap, labeled
"Δρ_t ≠ 0." Along the bottom of the panel, two small stylized training-reward curves: the coral
one peaks then plunges to near-zero (labeled "GRPO: collapse"), the teal one holds steady
(unlabeled, clean). Panel title: "FP8 Quantization."

---

### Unifying lower band — "ESS reads the room"

A full-width horizontal band (~25% of image height) at the bottom, background slightly lighter
than the panels (#0c1520), separated from the panels by a thin glowing teal horizontal rule.

In the center of this band: a large circular gauge instrument, rendered like a precision scientific
dial from a physics laboratory — think galvanometer or mass spectrometer readout. The face of the
gauge shows a single needle pointing to a value between 0 and 1. The scale has two labeled zones:
a teal arc labeled "ESS ≈ 1 · on-policy" and a coral arc labeled "ESS ↓ · off-policy." The
needle position shifts contextually: it is shown three times as a ghost trail — once pointing high
(teal zone, for the on-policy case), once mid-range, once low (coral zone, for the FP8 case) —
suggesting the gauge is dynamic and responsive.

From each of the three panels above, a thin beam of light descends into the gauge. Each beam
carries a faint texture matching its panel's visual language (ribbon filaments, bell-curve shadow,
stepped waveform fragment). The gauge absorbs all three beams into a single reading.

To the right of the gauge, a clean reward-vs-step line chart showing two curves over roughly 35
training steps: a coral curve that rises to ~0.65 then plunges to near zero (labeled in small
coral text "fixed clip"), and a teal curve that rises to ~0.65 and holds flat through the end
(labeled "ESS-adaptive"). The chart has no axis frames — just the curves, a faint horizontal
baseline, and step-count tick marks. The teal curve glows softly.

---

### Style notes

- Monospace typeface throughout all labels and annotations (suggest IBM Plex Mono or Source Code Pro)
- All panel borders: 1px bright teal (#0f9b8e) with a very faint glow bloom
- No photorealism; the style is technical-editorial illustration, like a well-designed arXiv figure
  rendered with the atmosphere of a dark-mode science magazine
- Avoid any cartoon or painterly style; keep it precise, cold, and scientific
- The overall impression should feel like: "this is a rigorous paper, and something important is
  breaking — and being fixed"
