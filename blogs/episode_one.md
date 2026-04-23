# FeynRL: Shifting the Focus Back to the RL Algorithm

Reinforcement learning has always had a reputation for brittleness, and not by accident: the learner collects data by acting under a policy, optimizes using rewards that are often imperfect or delayed, and repeatedly updates the very policy that determines future data. That makes RL fundamentally different from fixed-data supervised learning and highly sensitive to reward design, exploration, optimization, and the interaction between data collection and policy improvement.

These difficulties become more pronounced in large models. RL for LLMs and VLMs adds reward models or judges, expensive rollouts, distributed training, orchestration, synchronization, and other heavy systems concerns. Better systems make large-scale RL feasible, but they do not solve RL’s core problems such as reward misspecification, sparse or delayed feedback, credit assignment, exploration, or stale policy-dependent data. FeynRL is motivated by this gap: it makes the systems required for realistic large-model training available while keeping the algorithmic layer modular, legible, and separable.

![FeynRL separates data, system, and algorithm concerns](assets/episode_one/feynrl_separation.png)

## High-Level Overview

That separation of concerns is the central design principle. If someone wants to build a new RL algorithm, they should not need to work through a deeply entangled codebase spanning rollout workers, orchestration, data plumbing, and distributed execution just to change a loss or update rule. At the same time, this does not mean FeynRL is a toy framework: it is designed to support realistic large-model post-training with the systems needed to run at scale while remaining interpretable enough that researchers can understand what is happening and modify it without rewriting the whole stack.

Concretely, FeynRL supports supervised fine-tuning, preference learning, and reinforcement learning in a shared structure. It includes methods such as SFT, DPO, PPO, GRPO, CISPO, and P3O, together with rollout engines such as [vLLM](https://github.com/vllm-project/vllm), orchestration with [Ray](https://github.com/ray-project/ray), distributed optimization with [DeepSpeed](https://github.com/microsoft/DeepSpeed), sync and overlap execution modes, and modular reward, data, and evaluation layers. These features matter not because the project is trying to maximize scope, but because they provide the minimum substrate needed to investigate RL methods seriously at large-model scale while keeping the components separable.

Under the hood, FeynRL is organized along three axes that can be worked on independently: **algorithms** (the loss and update rules), **rollouts** (generation, rewards, and replay), and **orchestration** (distributed execution and weight synchronization between the training and inference engines). These layers communicate through narrow interfaces, so a new RL method is usually a new loss and update rule rather than a rewrite of the execution graph.

![FeynRL architecture across algorithms, rollouts, and orchestration](assets/episode_one/feynrl_architecture.png)

## Sync vs Overlap

Where algorithms and systems interact most visibly is the training-rollout schedule, and FeynRL supports two modes:

- **Sync mode:** each epoch generates all rollouts, trains on them, synchronizes the updated weights to the rollout engines, and repeats. This is fully on-policy, easy to reason about, and the right default when data freshness matters more than throughput.
- **Overlap mode:** generation and training run concurrently on separate GPU pools. A persistent producer continuously feeds prompts to the rollout engines through a bounded queue; each engine keeps generation pipelined so there is always one batch in flight; and a second bounded queue returns completed rollouts to a replay buffer that feeds training. At the end of each round, a single weight synchronization fires: generation briefly drains, the new policy weights are gathered on the training side while the drain completes, broadcast to the rollout engines, and generation resumes. A configurable bound on how far the replay data may lag behind the current policy keeps the staleness-throughput tradeoff explicit rather than emergent.

Sync and overlap compute the same thing; they differ only in how generation and training are interleaved. Sync serialises them for clarity and strict on-policy data; overlap runs them concurrently to reclaim GPU idle time when generation is the bottleneck.

![FeynRL sync vs. overlap mode](assets/episode_one/sync_vs_overlap.png)

## Overlap Mode

In overlap mode, generation and training run concurrently on separate GPU pools, with bounded queues, replay, and periodic weight synchronization making the throughput-staleness tradeoff explicit.

![FeynRL overlap mode architecture](assets/episode_one/overlap_arch.png)

## Results

The current release includes two initial experiments, and the results page will continue to expand as more runs are added.

See the [results page](../results.html) for the full learning curves and release metrics.

See also [`examples/RESULTS.md`](https://github.com/boson-ai/FeynRL/blob/main/examples/RESULTS.md) in the main repository.

### [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) on [GSM8K](https://huggingface.co/datasets/openai/gsm8k)

Training data comes from [GSM8K](https://huggingface.co/datasets/openai/gsm8k). Evaluation uses the shared mathematical reasoning benchmark suite reported across the release, spanning [GSM8K](https://huggingface.co/datasets/openai/gsm8k), [AIME](https://huggingface.co/datasets/HuggingFaceH4/aime_2024), [AMC](https://huggingface.co/datasets/AI-MO/aimo-validation-amc), [AMO](https://huggingface.co/datasets/meituan-longcat/AMO-Bench), [Brumo](https://huggingface.co/datasets/MathArena/brumo_2025), [HMMT](https://huggingface.co/datasets/MathArena/hmmt_feb_2025), and [Olympiad-style](https://huggingface.co/datasets/Hothan/OlympiadBench) sets.

| Run | Pass@1 | Pass@16 |
| --- | ---: | ---: |
| Base | 12.0% | 26.4% |
| FeynRL | 12.2% | 27.0% |

### [Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507) on [DeepScaler](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset)

Training data comes from the [DeepScaler preview dataset](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset). Evaluation again uses the same benchmark suite, with prompt formatting aligned to the model's released setup.

| Run | Pass@1 | Pass@16 |
| --- | ---: | ---: |
| Base | 12.2% | 19.7% |
| FeynRL | 27.0% | 40.2% |

The motivation for releasing FeynRL is therefore simple. Progress in RL for large models will require more than better systems for running current recipes faster. It will also require infrastructure that makes algorithmic questions easier to isolate, failure modes easier to understand, and new methods easier to build. FeynRL is an attempt to provide that layer: not a replacement for other frameworks, but a framework with a specific purpose.
