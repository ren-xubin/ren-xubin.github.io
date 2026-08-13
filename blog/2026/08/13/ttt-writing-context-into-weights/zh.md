---
layout: post
title: "Test-Time Training：把上下文写进权重"
date: 2026-08-13
summary: "坐在公园长椅上，看着小孩荡秋千，和我的 Agents 一起写下的一篇笔记。"
description: "从长上下文和运行时更新出发，聊聊 TTT 到底改变了什么，又离真正的持续学习有多远。"
permalink: /blog/2026/08/13/ttt-writing-context-into-weights/zh/
tags: [Test-Time Training, Continual Learning, Long Context]
lang: zh
math: true
reading_time: "约 9 分钟"
english_url: /blog/2026/08/13/ttt-writing-context-into-weights/
chinese_url: /blog/2026/08/13/ttt-writing-context-into-weights/zh/
preview_home: /
---
第一次看到 Test-Time Training（TTT）时，先想到的是一个很现实的问题：模型一边生成 token，一边还要改权重，这真的跑得动吗？如果更新本身比它要解决的问题还贵，想法再漂亮也没有用。

平时谈语言模型，训练和推理之间有一条很清楚的线：训练改参数，推理用参数。TTT 偏偏从这条线中间穿了过去。模型读上下文时，可以把历史放在不断增长的 KV cache 里，也可以把其中一部分结构写进临时权重。

最短的说法是：既然训练可以把一堆数据压进权重，那权重能不能也吸收眼前这段上下文？话听起来不复杂，真正有意思的是，“吸收”到底是什么意思，以及压缩之后究竟还剩下什么。

## 另一种记忆

把序列模型看成预测器还不够，它们其实也在做 memory system。不同架构的差别，很多时候就是：过去到底要以什么形式留下来。

Self-attention 把历史保存在一张明确的 KV 表里。当前 query 可以回头取某一条记录，这也是 attention 很擅长使用原始资料的原因。代价也很直接：上下文越长，表越大，读取它也越贵。

RNN 或 SSM 走的是相反的路。它们不断改写一份固定大小的状态，算起来省，也很整齐；但每条新信息都要经过同一套更新规则，最后还得塞进这份状态有限的容量里。

TTT 改的是更新规则本身。预训练阶段，outer model 学会一个小 learner 应该怎么更新；推理时，这个 learner 随着当前序列一起变化。第一篇 TTT 论文把它描述成一种拥有可学习动态的 hidden state [[1]](#ref-1)。

可以先粗略地这样想：attention 把过去留成一张可以回查的表，RNN 把过去留成一份不断改写的固定状态，而 TTT 把过去留进一个临时训练出来的小模型。

第三种方式一直让人感兴趣，原因就在这里：训练不再只是部署前发生过的一件事，也可以成为保存当前上下文的一种手段。

## 当状态本身会学习

原始 TTT layer 的更新只有一行：

$$
W_t = W_{t-1} - \eta\nabla_W \ell(W_{t-1}; x_t).
$$

这里的 $W_t$ 不是语言模型的全部参数，而是 TTT layer 内一个小 learner 的状态。每读到一个 token 表示，它就在自监督任务上学习一步，然后用更新后的 learner 产生输出。从 RNN 的角度看，$W_t$ 是 hidden state，梯度下降是 state transition，当前 learner 则负责读出结果。

这样就有了两种时间尺度。Slow weights 在正常预训练时学习表示、初始化和更新方式；fast weights 则只记录正在处理的这段序列里的模式。原始设置中，这些 fast weights 通常会在序列结束后重置。

这点很重要。它不是“每来一个 token，就把整个 7B 或 70B 模型重新训练一遍”。在线学习的部分本来就被设计得很小 [[1]](#ref-1)。In-Place TTT 走了另一条路：复用现有 MLP block 里的投影矩阵，把它当作 fast weights [[4]](#ref-4)。无论哪一种，真正允许在线变化的都只是模型的一部分。

一个还算好用的直觉是：hidden state 不再只是一串数字，而是一个可以学点东西、又能马上被问一句的小模型。

## 看起来像 attention，但又不是

原始 TTT 工作使用了一个 reconstruction 风格的 inner objective：

$$
\ell(W;x_t)=\left\|f(\theta_Kx_t;W)-\theta_Vx_t\right\|^2.
$$

Key-like 的 view 是 inner learner 的输入，value-like 的 view 是要求它重建的目标，query-like 的 view 则用来读取更新后的 learner。Attention 显式写下一条 key–value 记录，之后再把它取出来；TTT 则通过一次优化更新写入，之后让这个小模型做预测。

这也解释了它为什么会和 linear attention 联系起来：当 inner model 是线性的、从零初始化，并使用论文里的更新规则时，TTT layer 等价于 linear attention [[1]](#ref-1)。后来的分析进一步指出，一大类做 key–value binding 的 TTT layer，都可以用 learned linear attention 的语言重新写一遍 [[3]](#ref-3)。

这让事情少了一点神秘感，但不代表它没意思。“把东西写进权重”本身并不会自动产生新能力。真正不同的地方，在于 learner、学习目标、优化器，以及 outer training 最后教会 inner loop 保留什么。

## 真正麻烦的是跑起来

逐 token 的版本很好描述，实际执行却有点麻烦。每个 token 先形成一个 training view，更新 fast weights，再拿更新后的权重去预测下一个 token。因为每一步都依赖上一步，最朴素的实现几乎完全串行。

FLOPs 是线性的，不代表延迟就低。加速器喜欢大矩阵乘法，成千上万次互相依赖的小更新则是另一种 workload。原始 TTT 实现用 mini-batch 和 dual form 来处理：一个 block 里的 token 可以相对于 block 开头的参数并行算梯度，但对后续位置的影响仍按因果顺序累积。Dual form 再把累计梯度直接代回输出计算，省掉显式构造每个中间 $W_t$ 的步骤 [[1]](#ref-1)。

Block size 是这里最实际的旋钮。小一点，更接近逐 token 适应，但并行度会下降；大一点，更适合硬件，也更像一次 batch update。Dual form 只是重新安排计算，并没有让学习这件事消失。

原始论文使用大小为 16 的 block，并在它的 TPU 实现上报告了 wall-clock 加速 [[1]](#ref-1)。TTT-E2E 则在 3B 模型实验中报告，推理延迟不随上下文长度增长，128K 时比 full attention 快 $2.7\times$ [[2]](#ref-2)。这些数字有参考价值，但都属于特定实现和 workload，先别急着往所有模型和 GPU 上套。

内存的取舍同样关键。如果用纯 TTT layer 替代 self-attention，随上下文增长的 KV cache 可以换成固定大小的 fast weights [[1]](#ref-1)。在 TTT-E2E 这样的混合设计里，sliding-window attention 继续负责局部上下文，权重更新则压缩更远的历史 [[2]](#ref-2)。Fast weights、梯度、优化器状态和临时 activation 仍然要占空间；memory 没有消失，只是换了形状。

而固定状态终究有容量上限。模型可以一直读，不代表它能记住读过的一切。新信息可能覆盖旧信息，不相关的事实可能在参数空间里互相干扰，错误更新也可能留到后面。KV cache 更像档案库，fast weights 更像一本页数固定、需要反复重写的笔记。

于是问题变成了：到底应该留下什么？Test-Time Context Distillation（TTCD）给了一个方向：用长窗口 teacher 训练短窗口 fast weights，让它保留对未来预测有用的信息 [[5]](#ref-5)。重点不再只是“怎么存更多上下文”，而是压缩时该把什么留下来。

## 这算 continual learning 吗？

这里的“continual”其实包含了几种尺度完全不同的事情。

最小的一层，是一段序列内部的适应。这是原始 TTT layer 最明确的结果：fast weights 在处理当前上下文时变化，结束后通常重置 [[1]](#ref-1)。它更像 working memory，还谈不上 lifelong learning。

再往外一层，是任务内部的学习。TTT-E2E 用当前上下文上的 next-token prediction 作为 inner objective，通过 meta-learning 找到一个适合继续更新的初始化 [[2]](#ref-2)。TTT-Discover 则是另一种用法：针对一道科学或工程问题，利用可验证 reward 在测试时做强化学习 [[6]](#ref-6)。这些都是真实的权重更新，但还不是面对开放任务流的通用学习机制。

最大的一层，才是群体式的 continual learning：利用许多用户和 agent 的部署经验改进共享能力，同时不把噪声、私人偏好或攻击一起传给所有实例。把 optimizer 一直开着远远不够。经验要经过检查、consolidation、评测、版本化，必要时还得回滚。隐私、所有权、投毒和激励，也都是 learning system 的一部分。

TTT 提供的是 runtime plasticity。持续学习还得决定：哪些变化值得保留，哪些应该留在本地，以及经过验证的变化怎样安全地分享出去。

## 一些混沌的想法

TTT 还没有证明自己会取代 Transformer，但“取代”可能不是最重要的问题。更值得注意的是，它松动了一个长期默认：训练结束后，模型参数就固定了，推理只是运行由这些参数定义的函数。TTT 给运行中的模型增加了另一种可能——状态可以学习、变化，并把一些东西带到下一步。

现有结果从 1.3B 到 4B 参数、从实验性的长上下文到 128K context [[1]](#ref-1)、[[2]](#ref-2)、[[4]](#ref-4)，更像是这张地图上的第一批坐标，而不是结论。真正的问题是：online state 能不能从短期上下文长成任务经验，再从任务经验变成可以跨任务、跨实例迁移的能力，同时还保持可验证、可控制、可回滚。

这和 AGI 讨论里真正重要的部分更接近。一个有用的系统不应该只会预测下一个 token，还应该能在与世界交互时获得知识、形成技能、调整内部机制，并把经验里有价值的部分带到下一次任务。TTT 只是更新记忆的一种可能方式；检索式记忆、状态式记忆和参数式更新，最终也许会共同组成一个更大的 learning system。

最后这个机制还叫不叫 TTT，其实是次要的。更难、也更重要的问题是：模型能不能持续形成 online state，却不把自己弄得不稳定，并且把其中一部分变成可复用的能力。现在还没有定论，但这条路值得被做得更大。

## 参考文献

1. <span id="ref-1"></span>Y. Sun et al., “[Learning to (Learn at Test Time): RNNs with Expressive Hidden States](https://proceedings.mlr.press/v267/sun25h.html),” in Proceedings of the 42nd International Conference on Machine Learning (ICML), PMLR 267, pp. 57503–57522, 2025. [Code](https://github.com/test-time-training/ttt-lm-pytorch).
2. <span id="ref-2"></span>S. Tandon et al., “[End-to-End Test-Time Training for Long Context](https://arxiv.org/abs/2512.23675),” arXiv preprint arXiv:2512.23675, 2025. [Code](https://github.com/test-time-training/e2e).
3. <span id="ref-3"></span>J. Liu et al., “[Test-Time Training with KV Binding Is Secretly Linear Attention](https://arxiv.org/abs/2602.21204),” in Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026.
4. <span id="ref-4"></span>Y. Feng et al., “[In-Place Test-Time Training](https://openreview.net/forum?id=dTWfCLSoyl),” in International Conference on Learning Representations (ICLR), 2026, oral presentation. [Code](https://github.com/ByteDance-Seed/In-Place-TTT).
5. <span id="ref-5"></span>Y. Wang et al., “[Learning What to Remember: Test-Time Training via Context Distillation](https://arxiv.org/abs/2608.01672),” arXiv preprint arXiv:2608.01672, 2026.
6. <span id="ref-6"></span>B. Yuksekgonul et al., “[Learning to Discover at Test Time](https://arxiv.org/abs/2601.16175),” in Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026. [Code](https://github.com/test-time-training/discover).
