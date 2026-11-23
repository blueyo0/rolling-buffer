---
title: "要复现一次推理，得对齐多少处？"
highlight: true
---

*DeepInfer 的批次不变性与确定性推理*

先把 temperature 设为 0。让请求 A 单独跑两次，再让它和三条更长的请求一起跑。前两次可以完全相同，第三次却可能在同一个前缀上给出不同的分数，甚至选出另一个 token。这里做了两个实验：先问**同一批重跑会不会变**，再问**换一批邻居会不会变**。

[Thinking Machines Lab 的文章](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/#batch-invariance-and-determinism)从这一区别解释了为什么贪心解码也会漂移。沿着这个问题读 DeepInfer，除了要追计算走了哪个 kernel，还要问调度器怎样保证两次都在计算 A 的同一个逻辑 token。下面只追踪公开[快照 `774ce244`](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/README.md)中 DeepSeek-V4.1 的 NVIDIA 路径；没有在 GPU 上复现，也没有做性能对比。

## 两种等式，两个实验

用 `F(A, B; C)` 表示执行 batch `B` 后，取出请求 A 在指定逻辑位置的结果。`C` 固定权重、硬件、软件版本、精度、并行拓扑和执行配置。在算子层，比的是同一逻辑输入对应的张量，目标可以是逐位一致；在服务层，得先约定比 logits、logprob 还是生成的 token。三者不能互换。

| 性质 | 固定什么 | 改变什么 | 验什么 |
| --- | --- | --- | --- |
| Run-to-run determinism（重复执行确定性） | A、完整的 B、C 和初始状态 | 重跑一次 | 两次 F(A, B; C) 相等 |
| Batch invariance（批次不变性） | A、C，及 A 在比较点的逻辑前缀和位置 | B 的大小、其他成员或 A 的物理行号 | F(A, B; C) = F(A, B′; C) |

如果比较抽样后的整条生成轨迹，还需固定采样参数和请求 seed，并且从同样的初始状态开始。对服务调用者，其他请求不在请求体里，却参与了每轮实际执行。一个固定 batch 重跑完全确定的系统，仍可能随负载对 A 给出不同的输出。反过来，如果同一批重跑已经不稳定，看到两批不同还不能把差异归因给换批。改变 prefill 分块、缓存重放策略或并行拓扑是额外的实验，不能混在同一组对照里。

## 同一组数，换一副括号

设 `a = 2²⁴, b = 1, c = −2²⁴`，每次加法都用 FP32、最接近且平局取偶。`a + b` 正好落在两个可表示数之间，舍回 `a`；`b + c` 却能精确表示。两棵规约树各自多跑几次都稳定，只是稳定在不同的值：

```python
from struct import pack, unpack

def add32(x, y):
    return unpack('f', pack('f', x + y))[0]

a, b, c = 2**24, 1, -2**24
assert add32(add32(a, b), c) == 0.0
assert add32(a, add32(b, c)) == 1.0
```

把这三项想成一行矩阵乘的部分积。如果小 batch 下把 K 维拆给不同 block，大 batch 下让一个 block 完成整段规约，A 的输入行没变，括号却可能变了。这个反例没有原子加法竞争，也没有随机数。它只解释不同路径为什么可能产生不同数值，不是 FlashMLA 的复现实验。

## FlashMLA：一行 attention 为什么要归约两次？

Decode 时每个请求的新 query 很少；只沿 query/head 并行，设备可能吃不满。Split-KV 再开一条并行轴，把同一个 query 要看的 K/V 分给多个块。每块产出本段归一化后的输出 `o_s`，以及 `lse_s = log(sum(exp(score_s)))`。忽略 attention sink 等细节，合并时先算 `L = logsumexp(lse_s)`，再求 `sum(exp(lse_s - L) * o_s)`。在实数里，怎样划段都等价；在有限精度里，分段边界和合并顺序都是计算的一部分。

DeepInfer 的[非融合 decode wrapper](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/layers/attention_backend.py#L100-L134)将稀疏索引、可选的 extra cache 和 BI 标志交给 FlashMLA。锁定版本的 [CUDA sparse decode 判定](https://github.com/deepseek-ai/FlashMLA/blob/2e5429fc5653bab6e081f09477126f731882a6a9/csrc/api/sparse_decode.cpp#L177-L203)可以化简为：

```text
split_kv = not batch_invariant and (topk + extra_topk > 640)
```

这里的两个 top-k 是索引张量的宽度，不是采样参数，也不是每个请求实际有效的索引数。开 split 时，算子根据 batch、query 数和索引长度生成或复用调度元数据，把各段输出和 LSE 写到 `o_accum`、`lse_accum`，最后再调用 combine。开 BI 会绕过这层**跨片归约**；索引总宽不超过 640 时，关掉 BI 也不会走 split。这里关的是跨片合并，不等于 kernel 内部完全不再分 tile。[FlashMLA 调度与合并](https://github.com/deepseek-ai/FlashMLA/blob/2e5429fc5653bab6e081f09477126f731882a6a9/csrc/api/sparse_decode.cpp#L328-L399)

另一条路是保留 KV 并行，但固定每个逻辑前缀的分段边界和合并顺序。[Thinking Machines 的设计](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/#batch-invariant-attention)固定的是段长，不是段数：当推理引擎以不同方式切开 query 或组织缓存时，固定段数本身并不足以对齐归约。这和直接取消 split 是两种取舍；实际延迟和吞吐要按并发、KV 长度分别测。

[DeepInfer 的融合选择](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/models/v4_modules/attn.py#L74-L89)则是另一个条件。在支持的 head 数等条件下，稀疏或非 packed-KV 仍可走融合，dense packed-KV decode 在 BI 下不走融合。融合 wrapper 提前返回，不经过上述带 640 阈值的接口；[它调用的 FlashMLA CUDA 实现](https://github.com/deepseek-ai/FlashMLA/blob/2e5429fc5653bab6e081f09477126f731882a6a9/csrc/api/fused_norm_rope_attn_rope_cast_fwd.cpp#L308-L343)把 split-KV 固定为 false。也就是说，非融合路径可以由 BI 关掉 split，融合路径本身就没有这条分片路径。

Attention 之外，[Worker 初始化](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/worker/runtime_config.py#L48-L62)会给 DeepGEMM、TileKernels 打开进程级确定性选项；BI 与动态 EPLB 同时开启时，[EP buffer](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/parallel_utils/userbuffer.py#L67-L74)还会关掉 multi-reduction。这里能确认的是约束加在哪些入口，不能只凭一个开关断言库里每条算子路径、每种通信拓扑都满足逐位不变。

## 采样：随机数属于哪个 token？

A 从第 0 行挪到第 3 行。如果用一条按 batch 行消费的共享随机流，A 可能抽到另一个随机数。DeepInfer 的 [TileLang 随机采样入口](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/layers/tilelang_ops/random_sampling.py#L11-L35)把概率行与 `seq_seed`、`sample_position`、`is_nextn_predict` 一起传入；物理行号用于查表，不是随机键。[参考实现](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/layers/tilelang_ops/random_sampling.py#L53-L79)把这些键映射成均匀数，再在累计概率中找落点，不应把它的哈希实现当成 GPU kernel 的实现。

在 [CUDA fused sampler](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/layers/tilelang_ops/cuda/fused_sample.py#L350-L478)还可以看到另一层：同一个逻辑位置 `p` 为普通抽样、拒绝后的残差抽样和验证接受判定分别使用 `3p`、`3p+1`、`3p+2` 的随机流索引。行换了、同一个请求的概率和键一起搬过去，抽样问题仍然相同；如果 logits 变了，累计概率边界也可能变，同一个随机数仍能落到另一个 token 上。关闭抽样并不能让前向重新一致。

## ZOS：下一轮先跑，怎么拿到正确的前缀？

从 `LLM.generate` 进入离线引擎，C++ Scheduler 形成这一轮的请求和 token 预算，Worker/ModelRunner 执行设备前向与采样。启用 Zero Overhead Scheduling（ZOS）时，上一轮 decode 的 CPU 后处理留到本轮设备前向启动之后。这样可以重叠工作，但 CPU 侧的长度可能还没包含上一轮真实接受的 token。

假设 A 上一轮在第 0 行，这一轮在第 3 行。上一轮准备了几个 draft，与实际接受了几个 token，不是同一个数。直接沿用旧长度，哪怕 attention 和采样都逐位稳定，也可能在错误的前缀上稳定地算出一个结果。代码把这件事拆成三步：

1. **按序列身份找回上一轮结果。** [`prepare_zos_inputs`](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/csrc/engine/zos.cpp#L130-L172)按 `seq_id` 建立跨轮 remap，而不是沿用 batch 行号。两轮有重叠时，开启 NextN 的路径用上轮设备端的 accepted-token counts 修正 `seqlens_k`；varlen 分支还按置信度重新分配 `seqlens_q`。
2. **设备先用修正后的 metadata 发起前向。** 此后 CPU 才后处理上一轮。普通 prefill 和关闭 ZOS 的步骤则在本轮完成后处理。
3. **采样前补齐 CPU metadata。** varlen 需要把修正后的 q 长度异步回传；Worker 等相应事件后更新采样 metadata，再对当前轮抽样。ZOS 隐藏的是部分 CPU 开销，没消除这些依赖。

[Worker 的执行顺序](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/worker/worker.py#L85-L171) · [长度修正与设备结果](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/worker/worker.py#L174-L223)

可变长 MTP 还引入了请求间的耦合。[调度器](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/csrc/engine/mtp_varlen.hpp#L135-L239)用候选接受概率和估计的 forward/sampler 耗时选择整批 query 预算，每条请求先拿一个名额，剩余名额按置信度分给 draft 候选。换邻居后，即使 A 的置信度不变，A 本轮可能也会多验证或少验证一个 token。这不是在说动态验证必然产生不同 token，而是说只控制 batch 内核和 seed，还没有控制验证日程。官方对 NVIDIA 的确定性采样建议是 `DEEPINFER_BATCH_INVARIANT=1`、`DEEPINFER_ENABLE_MTP_VARLEN=0`，并显式固定 `SamplingParams.seq_seed`。关的是动态验证长度，不是所有 MTP；[README](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/README.md#L18-L32)也没有要求关闭 ZOS。SWA 有界重放是另一项控制变量，换批测试不能证明完整缓存状态与近似重放等价。[生产部署说明](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/doc/inference_deployment_qa.md)

## 公开测试究竟对齐了什么？

[模型回归](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/tests/models/test_batch_invariant.py#L29-L99)先提交 64 条不同长度的合成 prompt，配置 EP=8、NextN=5、temperature=1、top-p=0.7，每条最多生成 50 个 token，记录 top-5。64 是提交数，不是每个 GPU step 的实际 batch size。它再逐条跑，把原 batch 的输出指定为 `golden_token_ids`。最后用 Python 的精确相等比较四项：输出 token ID、该 token 的 logprob、top-5 ID、top-5 logprob。[采样器在取选中 token 的分数前先应用 golden override](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/layers/sampler.py#L429-L472)。

这是一种控制变量的方法：强制两边走同一条轨迹，才能对比**同一个前缀上对指定 token 和 top-5 的打分**。它能发现 token 一样、分数却漂了的问题；它不比较整个词表的 logits，也不能证明去掉 golden 后两边仍会独立选中同一个 token。[SGLang 的 Mamba-1 报告](https://github.com/sgl-project/sglang/issues/50138)给出了一个相邻的实测例子：在报告者的 GB200 配置中，单独 decode 和四条一起跑的 greedy token 与 prefill 分数相同，但 decode logprob 不同。这不是 DeepInfer 的测量结果。

| 想回答的问题 | 最小对照 | 看什么 |
| --- | --- | --- |
| 算子换批是否变值？ | 固定目标位置的 Q/K/V、稀疏索引和逻辑长度，只换邻居或物理行；对同一批先做重复运行 | 目标位置的 attention output、LSE、实际 kernel 分支 |
| 抽样是否依赖所在行？ | 固定概率行、seed、逻辑位置和分支，把它们一起重排 | 选出的 ID 是否相同；若真实模型的概率行已经变了，先查前向 |
| 请求能否独立复现？ | 不注入 golden，固定请求与配置，先用相同 batch 重跑，再换 batch；缓存或 ZOS 另做对照 | 第一次分叉前的 logits/logprob、采样位置和 draft 接受数 |

比最终文本之前，先把两次执行对齐到**同一个前缀的同一个位置**。分数先变，就沿 kernel 与布局往回追；分数没变，再看随机键和调度 metadata。到了下一个 token，轨迹一旦分叉，两边往往已经在给不同的前缀打分了。

## 参考文献

- Horace He 与 Thinking Machines Lab，[Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)，Connectionism，2025-09-10。
