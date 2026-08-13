---
title: "Test-Time Training: Writing Context into Weights"
date: 2026-08-13
summary: "A note from a park bench, written while watching kids on the swings—with my agents."
description: "A first-principles note on test-time training, long context, and the uneasy boundary between runtime adaptation and learning."
tags: [Test-Time Training, Continual Learning, Long Context]
lang: en
math: true
reading_time: "9 min read"
english_url: /blog/2026/08/13/ttt-writing-context-into-weights/
chinese_url: /blog/2026/08/13/ttt-writing-context-into-weights/zh/
---
When Test-Time Training (TTT) first came up, the first question was practical: if a model changes its weights while generating tokens, does this actually run? If the update costs more than the problem it is meant to solve, the idea is not very useful.

Most discussions about language models keep a clean line between training and inference. Training changes the parameters; inference uses them. TTT draws a line through that boundary. A sequence can be kept in a growing KV cache, or some of its structure can be written into temporary weights while the model is reading it.

The short version is: if training can compress a dataset into weights, perhaps weights can also absorb the context in front of the model. That sounds simple. The interesting part is working out what “absorb” means, and what gets lost along the way.

## A different kind of memory

One useful way to look at TTT is to stop treating a sequence model as only a predictor. It is also a memory system, and different architectures make different choices about what to keep.

Self-attention leaves the past in an explicit table of keys and values. The current query can go back and retrieve a particular piece, which is why attention is so good at using source material. The table grows with the context, though, and reading from it becomes increasingly expensive.

An RNN or SSM makes the opposite choice. It repeatedly rewrites a fixed-size state. This is cheap and neat, but every new observation has to pass through the same update rule and fit into the capacity of that state.

TTT changes the update rule itself. During pretraining, the outer model learns how a small inner learner should update; at inference time, that learner changes as it reads the current sequence. The first TTT paper describes this as a hidden state with expressive, learnable dynamics [[1]](#ref-1).

So the rough picture is simple: attention keeps the past as a table that can be searched, an RNN keeps it as a fixed state that is repeatedly rewritten, and TTT keeps it in a small model that is trained for a moment.

That last option is the reason the idea stays interesting. Training is no longer only something that happened before deployment. It becomes one possible way of storing the current context.

## When the state can learn

The update in the original TTT layer is compact:

$$
W_t = W_{t-1} - \eta\nabla_W \ell(W_{t-1}; x_t).
$$

Here $W_t$ is not the full parameter set of the language model. It is the state of a small learner inside the TTT layer. After each token representation arrives, the learner takes a self-supervised step and is then queried to produce the layer output. From an RNN point of view, $W_t$ is the hidden state, gradient descent is the state transition, and the current learner is the readout.

This creates two time scales. The slow weights learn the representation, the initialization, and the update rule during ordinary pretraining. The fast weights learn patterns from the sequence currently being processed. In the original setup, those fast weights are normally reset when the sequence is over.

That distinction matters. It is not “retrain the whole 7B or 70B model every time a token arrives.” The online learner is deliberately small [[1]](#ref-1). In-Place TTT takes a different route and reuses a projection matrix inside an existing MLP block as the fast weights [[4]](#ref-4). In both cases, only part of the model is allowed to change online.

The appealing mental model is that the hidden state is no longer just a vector. It is a little model that can learn something, and can immediately be asked about what it learned.

## It looks like attention, until it does not

The original TTT work uses a reconstruction-style inner objective:

$$
\ell(W;x_t)=\left\|f(\theta_Kx_t;W)-\theta_Vx_t\right\|^2.
$$

The key-like view is the input to the inner learner, the value-like view is the target it is asked to reconstruct, and the query-like view is used to read the updated learner. Attention writes an explicit key–value record and later looks it up. TTT writes by taking an optimization step and later asks the small model to make a prediction.

This is also where the connection to linear attention appears. With a linear inner model, zero initialization, and the update used in the paper, the TTT layer becomes equivalent to linear attention [[1]](#ref-1). A later analysis shows that a broad family of key–value-binding TTT layers can be rewritten in the same language [[3]](#ref-3).

That result makes the proposal less mystical, not less useful. “Writing into weights” is not automatically a new capability. The difference comes from the learner, the objective, the optimizer, and—most importantly—what the outer training process teaches the inner loop to preserve.

## The part that gets messy in practice

The token-by-token version is easy to describe and awkward to run. Each token creates a training view, updates the fast weights, and then gets used to predict the next token. The updates depend on one another, so a naive implementation is almost entirely sequential.

Linear FLOPs do not guarantee low latency. Accelerators are happiest with large matrix multiplications; a long chain of small, dependent updates is a different kind of workload. The original TTT implementation groups tokens into mini-batches and uses a dual form. Gradients can be computed in parallel relative to the parameters at the start of a block, while their effects are still accumulated in causal order. The dual form substitutes those accumulated gradients into the output calculation instead of materializing every intermediate $W_t$ [[1]](#ref-1).

Block size is the practical knob. Small blocks stay closer to token-by-token adaptation but expose less parallelism. Large blocks use the hardware better and behave more like a batch update. The dual form rearranges the work; it does not make the learning step disappear.

The original paper used blocks of 16 and reported a wall-clock improvement on its TPU setup [[1]](#ref-1). TTT-E2E reports constant inference latency as context grows and a $2.7\times$ advantage over full attention at 128K in a 3B-model experiment [[2]](#ref-2). Those are useful measurements, but they belong to particular implementations and workloads.

The memory trade-off is just as important. In a pure TTT layer, fixed-size fast weights can replace the context-growing KV cache [[1]](#ref-1). In a hybrid design such as TTT-E2E, sliding-window attention keeps local context while weight updates compress more distant history [[2]](#ref-2). Fast weights, gradients, optimizer state, and temporary activations still take space; memory has not vanished, it has changed shape.

And fixed state has a limit. A model can keep reading without keeping everything it has read. New information can overwrite old information, unrelated facts can interfere in parameter space, and a bad update can stay around long enough to affect later predictions. The KV cache is closer to an archive. Fast weights are closer to a notebook with a fixed number of pages.

That leaves the most interesting question: what should be kept? Test-Time Context Distillation (TTCD) uses a long-window teacher to train short-window fast weights to retain information that will be useful for future prediction [[5]](#ref-5). The problem is no longer just how to store more context, but how to decide what deserves to survive compression.

## Is this continual learning yet?

The word “continual” can cover several rather different things here.

At the smallest scale, there is adaptation within one sequence. This is the clearest result of the original TTT layers: fast weights change while a context is being processed and are usually reset afterward [[1]](#ref-1). It looks more like working memory than lifelong learning.

The next scale is learning within a task. TTT-E2E uses next-token prediction on the current context as its inner objective and meta-learns an initialization that works well after those updates [[2]](#ref-2). TTT-Discover takes another route, using verifiable rewards and test-time reinforcement learning to improve a policy for one scientific or engineering problem [[6]](#ref-6). Both involve real weight updates, but neither is yet a general mechanism for learning across an open-ended stream of tasks.

The largest scale would be population-level continual learning: using experience from many deployed users and agents to improve shared capabilities without spreading noise, private preferences, or attacks to everyone else. Leaving an optimizer on is not enough. Experience has to be checked, consolidated, evaluated, versioned, and sometimes rolled back. Privacy, ownership, poisoning, and incentives are part of the learning system too.

TTT supplies runtime plasticity. Continual learning still has to decide which changes are worth keeping, which should stay local, and how a validated change can be shared without destabilizing the model.

## A few half-formed thoughts

TTT has not shown that it will replace the Transformer, but “replace” may be the wrong question. The more important shift is that it loosens a long-standing assumption: after training, the model’s parameters are fixed and inference merely runs the function they define. TTT gives the running model another possibility—a state that can learn, change, and carry something forward.

The existing results, from 1.3B to 4B parameters and up to 128K context [[1]](#ref-1), [[2]](#ref-2), [[4]](#ref-4), are better read as the first coordinates on this map than as a verdict. The larger question is whether an online state can grow from short-term context into task experience, and from task experience into capabilities that transfer across tasks and instances—while remaining verifiable, controllable, and reversible.

That is close to the part of the AGI problem that matters here. A useful system should not only predict the next token; it should acquire knowledge while interacting with the world, form skills, adjust its internal machinery, and bring the useful parts of experience into the next task. TTT is one possible way to update memory. Retrieval, recurrent state, and parameter updates may eventually work together as a larger learning system.

Whether the final mechanism is still called TTT is secondary. The harder and more consequential question is whether a model can keep forming online state without becoming unstable, and turn enough of that state into reusable ability. There is no settled answer yet, but this is a direction worth making much larger.

## References

1. <span id="ref-1"></span>Y. Sun et al., “[Learning to (Learn at Test Time): RNNs with Expressive Hidden States](https://proceedings.mlr.press/v267/sun25h.html),” in Proceedings of the 42nd International Conference on Machine Learning (ICML), PMLR 267, pp. 57503–57522, 2025. [Code](https://github.com/test-time-training/ttt-lm-pytorch).
2. <span id="ref-2"></span>S. Tandon et al., “[End-to-End Test-Time Training for Long Context](https://arxiv.org/abs/2512.23675),” arXiv preprint arXiv:2512.23675, 2025. [Code](https://github.com/test-time-training/e2e).
3. <span id="ref-3"></span>J. Liu et al., “[Test-Time Training with KV Binding Is Secretly Linear Attention](https://arxiv.org/abs/2602.21204),” in Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026.
4. <span id="ref-4"></span>Y. Feng et al., “[In-Place Test-Time Training](https://openreview.net/forum?id=dTWfCLSoyl),” in International Conference on Learning Representations (ICLR), 2026, oral presentation. [Code](https://github.com/ByteDance-Seed/In-Place-TTT).
5. <span id="ref-5"></span>Y. Wang et al., “[Learning What to Remember: Test-Time Training via Context Distillation](https://arxiv.org/abs/2608.01672),” arXiv preprint arXiv:2608.01672, 2026.
6. <span id="ref-6"></span>B. Yuksekgonul et al., “[Learning to Discover at Test Time](https://arxiv.org/abs/2601.16175),” in Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026. [Code](https://github.com/test-time-training/discover).
