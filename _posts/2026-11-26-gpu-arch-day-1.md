---
layout: post
title: "一块矩阵，谁在算，数据放在哪？"
description: "从 Triton 矩阵乘的一块输出入手，看懂 NVIDIA GPU 的线程组织与存储层次。"
highlight: true
---

*Rolling Buffer · GPU 架构 7 天 · Day 1/7 · 11 月 27 日*

读 Triton 的矩阵乘，第一行 `pid = tl.program_id(0)` 看起来像是在拿某种“线程编号”。可紧接着，代码用这个数字圈出输出矩阵的一整块。一个编号怎么分到这么多元素？更具体一点：`pid=0` 拿到的活，是一个线程干、一个 warp 干，还是一整颗 SM 干？

还有另一个问题：这块矩阵的数据在显存、L2、L1、shared memory 和寄存器里各扮演什么角色？这篇只追一块输出，用它把执行和存储两张地图叠起来。Roofline 可以告诉我们该先怀疑算力还是带宽，但画图前要说清是哪一级的字节：反复从 shared memory 读到的字节，不能直接拿去除以显存带宽。

<div markdown="0">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1220" width="1000" style="display:block;width:100%;height:auto" role="img" aria-labelledby="rb-gpu-arch-title rb-gpu-arch-desc">
<title id="rb-gpu-arch-title">一块矩阵的两张地图</title>
<desc id="rb-gpu-arch-desc">上半部分把输出 tile 映射到由四个 warp 组成的一个线程块。下半部分展示显存、GPU 共享的 L2，以及各颗 SM 本地的 L1、shared memory、寄存器和执行单元。</desc>
<rect width="1000" height="1220" fill="#f1f5f9"/>
<text x="45" y="71" fill="#15263b" font-family="system-ui, Noto Sans CJK SC, sans-serif" font-size="43" font-weight="700">一块矩阵的两张地图</text>
<text x="47" y="114" fill="#435872" font-family="system-ui, Noto Sans CJK SC, sans-serif" font-size="24" font-weight="400">NVIDIA GPU  ·  Rolling Buffer  ·  第 1 课</text>
<rect x="38" y="140" width="924" height="407" rx="15" fill="#fff" stroke="#c6d3e1" stroke-width="2"/>
<text x="62" y="193" fill="#15263b" font-family="system-ui, Noto Sans CJK SC, sans-serif" font-size="32" font-weight="700">01  哪块数据，交给哪些线程？</text>
<text x="64" y="248" fill="#15263b" font-family="system-ui, Noto Sans CJK SC, sans-serif" font-size="24" font-weight="700">输出矩阵 C（5×6）</text>
<rect x="66" y="264" width="36" height="33" rx="0" fill="#b8d3ff" stroke="#c6d3e1" stroke-width="1"/>
<rect x="103" y="264" width="36" height="33" rx="0" fill="#b8d3ff" stroke="#c6d3e1" stroke-width="1"/>
<rect x="140" y="264" width="36" height="33" rx="0" fill="#b8d3ff" stroke="#c6d3e1" stroke-width="1"/>
<rect x="177" y="264" width="36" height="33" rx="0" fill="#f0f4fa" stroke="#c6d3e1" stroke-width="1"/>
<rect x="214" y="264" width="36" height="33" rx="0" fill="#f0f4fa" stroke="#c6d3e1" stroke-width="1"/>
<rect x="251" y="264" width="36" height="33" rx="0" fill="#f0f4fa" stroke="#c6d3e1" stroke-width="1"/>
<rect x="66" y="298" width="36" height="33" rx="0" fill="#b8d3ff" stroke="#c6d3e1" stroke-width="1"/>
<rect x="103" y="298" width="36" height="33" rx="0" fill="#b8d3ff" stroke="#c6d3e1" stroke-width="1"/>
<rect x="140" y="298" width="36" height="33" rx="0" fill="#b8d3ff" stroke="#c6d3e1" stroke-width="1"/>
<rect x="177" y="298" width="36" height="33" rx="0" fill="#f0f4fa" stroke="#c6d3e1" stroke-width="1"/>
<rect x="214" y="298" width="36" height="33" rx="0" fill="#f0f4fa" stroke="#c6d3e1" stroke-width="1"/>
<rect x="251" y="298" width="36" height="33" rx="0" fill="#f0f4fa" stroke="#c6d3e1" stroke-width="1"/>
<rect x="66" y="332" width="36" height="33" rx="0" fill="#f0f4fa" stroke="#c6d3e1" stroke-width="1"/>
<rect x="103" y="332" width="36" height="33" rx="0" fill="#f0f4fa" stroke="#c6d3e1" stroke-width="1"/>
<rect x="140" y="332" width="36" height="33" rx="0" fill="#f0f4fa" stroke="#c6d3e1" stroke-width="1"/>
<rect x="177" y="332" width="36" height="33" rx="0" fill="#f0f4fa" stroke="#c6d3e1" stroke-width="1"/>
<rect x="214" y="332" width="36" height="33" rx="0" fill="#f0f4fa" stroke="#c6d3e1" stroke-width="1"/>
<rect x="251" y="332" width="36" height="33" rx="0" fill="#f0f4fa" stroke="#c6d3e1" stroke-width="1"/>
<rect x="66" y="366" width="36" height="33" rx="0" fill="#f0f4fa" stroke="#c6d3e1" stroke-width="1"/>
<rect x="103" y="366" width="36" height="33" rx="0" fill="#f0f4fa" stroke="#c6d3e1" stroke-width="1"/>
<rect x="140" y="366" width="36" height="33" rx="0" fill="#f0f4fa" stroke="#c6d3e1" stroke-width="1"/>
<rect x="177" y="366" width="36" height="33" rx="0" fill="#f0f4fa" stroke="#c6d3e1" stroke-width="1"/>
<rect x="214" y="366" width="36" height="33" rx="0" fill="#f0f4fa" stroke="#c6d3e1" stroke-width="1"/>
<rect x="251" y="366" width="36" height="33" rx="0" fill="#f0f4fa" stroke="#c6d3e1" stroke-width="1"/>
<rect x="66" y="400" width="36" height="33" rx="0" fill="#f0f4fa" stroke="#c6d3e1" stroke-width="1"/>
<rect x="103" y="400" width="36" height="33" rx="0" fill="#f0f4fa" stroke="#c6d3e1" stroke-width="1"/>
<rect x="140" y="400" width="36" height="33" rx="0" fill="#f0f4fa" stroke="#c6d3e1" stroke-width="1"/>
<rect x="177" y="400" width="36" height="33" rx="0" fill="#f0f4fa" stroke="#c6d3e1" stroke-width="1"/>
<rect x="214" y="400" width="36" height="33" rx="0" fill="#f0f4fa" stroke="#c6d3e1" stroke-width="1"/>
<rect x="251" y="400" width="36" height="33" rx="0" fill="#f0f4fa" stroke="#c6d3e1" stroke-width="1"/>
<line x1="66" y1="332" x2="288" y2="332" stroke="#7790ac" stroke-width="2"/>
<line x1="66" y1="400" x2="288" y2="400" stroke="#7790ac" stroke-width="2"/>
<line x1="177" y1="264" x2="177" y2="434" stroke="#7790ac" stroke-width="2"/>
<text x="65" y="478" fill="#245bc7" font-family="system-ui, Noto Sans CJK SC, sans-serif" font-size="22" font-weight="700">左上角 tile：2×3</text>
<line x1="300" y1="320" x2="386" y2="320" stroke="#245bc7" stroke-width="4"/>
<polygon points="386.00,320.00 374.92,326.79 374.92,313.21" fill="#245bc7"/>
<rect x="392" y="249" width="521" height="200" rx="15" fill="#edf4ff" stroke="#b6ccef" stroke-width="2"/>
<text x="411" y="290" fill="#15263b" font-family="system-ui, Noto Sans CJK SC, sans-serif" font-size="25" font-weight="700">一个 program → 一个 CTA（线程块）</text>
<rect x="411" y="315" width="109" height="48" rx="15" fill="#fff" stroke="#bdcdec" stroke-width="2"/>
<text x="426" y="344" fill="#245bc7" font-family="system-ui, Noto Sans CJK SC, sans-serif" font-size="20" font-weight="700">warp 0</text>
<rect x="530" y="315" width="109" height="48" rx="15" fill="#fff" stroke="#bdcdec" stroke-width="2"/>
<text x="545" y="344" fill="#245bc7" font-family="system-ui, Noto Sans CJK SC, sans-serif" font-size="20" font-weight="700">warp 1</text>
<rect x="649" y="315" width="109" height="48" rx="15" fill="#fff" stroke="#bdcdec" stroke-width="2"/>
<text x="664" y="344" fill="#245bc7" font-family="system-ui, Noto Sans CJK SC, sans-serif" font-size="20" font-weight="700">warp 2</text>
<rect x="768" y="315" width="109" height="48" rx="15" fill="#fff" stroke="#bdcdec" stroke-width="2"/>
<text x="783" y="344" fill="#245bc7" font-family="system-ui, Noto Sans CJK SC, sans-serif" font-size="20" font-weight="700">warp 3</text>
<text x="409" y="408" fill="#15263b" font-family="system-ui, Noto Sans CJK SC, sans-serif" font-size="23" font-weight="400">示例：4 个 warp × 32 个线程</text>
<text x="392" y="495" fill="#435872" font-family="system-ui, Noto Sans CJK SC, sans-serif" font-size="22" font-weight="400">CTA 在某颗 SM 上执行；编号不指定哪颗 SM。</text>
<rect x="38" y="570" width="924" height="607" rx="15" fill="#fff" stroke="#c6d3e1" stroke-width="2"/>
<text x="62" y="624" fill="#15263b" font-family="system-ui, Noto Sans CJK SC, sans-serif" font-size="32" font-weight="700">02  同一张 GPU 上，数据放在哪？</text>
<rect x="272" y="649" width="456" height="56" rx="15" fill="#edf4ff" stroke="#b6ccef" stroke-width="2"/>
<text x="296" y="685" fill="#245bc7" font-family="system-ui, Noto Sans CJK SC, sans-serif" font-size="23" font-weight="700">板载显存（DRAM）   A / B / C 的分配</text>
<line x1="500" y1="706" x2="500" y2="732" stroke="#245bc7" stroke-width="4"/>
<polygon points="500.00,732.00 493.21,720.92 506.79,720.92" fill="#245bc7"/>
<polygon points="500.00,706.00 506.79,717.08 493.21,717.08" fill="#245bc7"/>
<rect x="272" y="733" width="456" height="58" rx="15" fill="#edf4ff" stroke="#b6ccef" stroke-width="2"/>
<text x="296" y="772" fill="#245bc7" font-family="system-ui, Noto Sans CJK SC, sans-serif" font-size="24" font-weight="700">L2：一张 GPU 上的 SM 共享</text>
<line x1="277" y1="792" x2="277" y2="833" stroke="#245bc7" stroke-width="4"/>
<polygon points="277.00,833.00 270.21,821.92 283.79,821.92" fill="#245bc7"/>
<polygon points="277.00,792.00 283.79,803.08 270.21,803.08" fill="#245bc7"/>
<line x1="725" y1="792" x2="725" y2="833" stroke="#245bc7" stroke-width="4"/>
<polygon points="725.00,833.00 718.21,821.92 731.79,821.92" fill="#245bc7"/>
<polygon points="725.00,792.00 731.79,803.08 718.21,803.08" fill="#245bc7"/>
<rect x="72" y="834" width="405" height="236" rx="15" fill="#f7fafc" stroke="#c6d3e1" stroke-width="2"/>
<text x="87" y="872" fill="#15263b" font-family="system-ui, Noto Sans CJK SC, sans-serif" font-size="25" font-weight="700">SM 0</text>
<rect x="87" y="886" width="173" height="64" rx="15" fill="#e7efff" stroke="#bdd1f3" stroke-width="2"/>
<text x="99" y="917" fill="#245bc7" font-family="system-ui, Noto Sans CJK SC, sans-serif" font-size="20" font-weight="700">L1 · 硬件缓存</text>
<rect x="271" y="886" width="191" height="64" rx="15" fill="#e3f4ef" stroke="#badad0" stroke-width="2"/>
<text x="282" y="916" fill="#006b61" font-family="system-ui, Noto Sans CJK SC, sans-serif" font-size="20" font-weight="700">shared memory</text>
<text x="283" y="938" fill="#006b61" font-family="system-ui, Noto Sans CJK SC, sans-serif" font-size="16" font-weight="400">CTA 显式使用</text>
<rect x="87" y="971" width="173" height="70" rx="15" fill="#fff2e5" stroke="#ead2bb" stroke-width="2"/>
<text x="100" y="1010" fill="#974718" font-family="system-ui, Noto Sans CJK SC, sans-serif" font-size="21" font-weight="700">线程寄存器</text>
<rect x="271" y="971" width="191" height="70" rx="15" fill="#fff2e5" stroke="#ead2bb" stroke-width="2"/>
<text x="282" y="1001" fill="#974718" font-family="system-ui, Noto Sans CJK SC, sans-serif" font-size="20" font-weight="700">执行单元</text>
<text x="282" y="1027" fill="#974718" font-family="system-ui, Noto Sans CJK SC, sans-serif" font-size="17" font-weight="400">含 Tensor Core</text>
<rect x="523" y="834" width="405" height="236" rx="15" fill="#f7fafc" stroke="#c6d3e1" stroke-width="2"/>
<text x="538" y="872" fill="#15263b" font-family="system-ui, Noto Sans CJK SC, sans-serif" font-size="25" font-weight="700">SM 1</text>
<rect x="538" y="886" width="173" height="64" rx="15" fill="#e7efff" stroke="#bdd1f3" stroke-width="2"/>
<text x="550" y="917" fill="#245bc7" font-family="system-ui, Noto Sans CJK SC, sans-serif" font-size="20" font-weight="700">L1 · 硬件缓存</text>
<rect x="722" y="886" width="191" height="64" rx="15" fill="#e3f4ef" stroke="#badad0" stroke-width="2"/>
<text x="733" y="916" fill="#006b61" font-family="system-ui, Noto Sans CJK SC, sans-serif" font-size="20" font-weight="700">shared memory</text>
<text x="734" y="938" fill="#006b61" font-family="system-ui, Noto Sans CJK SC, sans-serif" font-size="16" font-weight="400">CTA 显式使用</text>
<rect x="538" y="971" width="173" height="70" rx="15" fill="#fff2e5" stroke="#ead2bb" stroke-width="2"/>
<text x="551" y="1010" fill="#974718" font-family="system-ui, Noto Sans CJK SC, sans-serif" font-size="21" font-weight="700">线程寄存器</text>
<rect x="722" y="971" width="191" height="70" rx="15" fill="#fff2e5" stroke="#ead2bb" stroke-width="2"/>
<text x="733" y="1001" fill="#974718" font-family="system-ui, Noto Sans CJK SC, sans-serif" font-size="20" font-weight="700">执行单元</text>
<text x="733" y="1027" fill="#974718" font-family="system-ui, Noto Sans CJK SC, sans-serif" font-size="17" font-weight="400">含 Tensor Core</text>
<text x="73" y="1110" fill="#435872" font-family="system-ui, Noto Sans CJK SC, sans-serif" font-size="21" font-weight="400">蓝：硬件缓存或显存   绿：软件管理的缓冲   橙：线程工作值和执行</text>
<text x="73" y="1147" fill="#15263b" font-family="system-ui, Noto Sans CJK SC, sans-serif" font-size="22" font-weight="700">这是一张范围图；L1 和 shared memory 不是串联的两级缓存。</text>
</svg>
</div>

## 先看要算哪块，不急着数线程

设 `A` 是 5×5，`B` 是 5×6，输出 `C` 就是 5×6。把 `C` 划成名义上 2 行 × 3 列的小块，行方向向上取整得到 3 块，列方向得到 2 块。总共启动 6 个 Triton program。这里的 **tile** 就是矩阵中的一块区域，比如左上角 `C[0:2, 0:3]`。

这块输出需要 `A` 的前两行和 `B` 的前三列，但要走完全部 5 个 `K`。若每轮沿 `K` 走两个元素，同一个 program 会处理 `[0,2)`、`[2,4)`、`[4,5)`，把三轮结果加进同一块累加器，最后才写回。**多一次 K 循环，不会多启动一个 program。** 最后一轮只有一个有效 K；最后一排输出 tile 也只有一行有效。原代码分别用输入的 K 掩码和输出的行列掩码处理这两个边界。[Triton notebook：分块、累加和写回](https://github.com/blueyo0/Triton-Practice-Notebook/blob/fcb50a3d5bb150049aaa5c376dac120036a30db2/examples/3.matmul.py#L39-L92)

## 一个 program 里面有谁？

拿这份普通的 NVIDIA 单 CTA 配置来说，可以把关系读成：

- **program / CTA（线程块）**：一个 program 实例由一个 CTA 执行，负责上面那块输出。它是这个 kernel 的分工单位。
- **warp**：CTA 内部的 32 个线程组成一组，由硬件按 warp 发射指令。假如编译配置是 `num_warps=4`，这个 CTA 就有 4 个 warp、128 个线程协作；不是说每个线程固定负责一个输出元素。具体怎么把 tile 的元素分给线程，由编译后的布局决定。
- **SM（流式多处理器）**：GPU 上实际运行这些线程块的硬件单元。一个普通 CTA 在一颗 SM 上运行；资源允许时，一颗 SM 可以同时容纳多个 CTA。`pid` 没有指定 SM 编号，也没有承诺哪一个 CTA 先跑完。

所以读这段程序时，`BLOCK_SIZE_M/N` 告诉你每份活有多大，`num_warps` 告诉你多少线程一起干。它们在同一个配置里，却回答的是两个问题。代码里看到一个 128×128 的 tile，不要据此推断会启动 16,384 个线程。[Triton 编译配置](https://triton-lang.org/main/python-api/generated/triton.Config.html) · [CUDA 执行模型](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)

## 同一块矩阵的数据，住在哪里？

假设 A、B、C 都是普通显存上的张量。这里先画一张常见分块实现的**逻辑图**，不是说每条 GPU 指令都必须依次经过这些位置。

```text
一张 GPU
  板载显存（DRAM）      A、B、C 的全局内存分配
           ↕
  L2 缓存               同一张 GPU 上各 SM 共享，硬件管理
           ↕
  每颗 SM 内
    L1 缓存             本 SM 的硬件缓存
    shared memory      程序显式安排，普通用法由 CTA 内线程共享
    寄存器              每个线程自己的工作值
    执行单元            普通算术、Tensor Core 等
```

| 位置 | 谁管、谁用 | 放进这次矩阵乘里看 |
| --- | --- | --- |
| **全局内存 / 显存** | 全局内存是程序可用的地址空间；这里 A、B、C 的后备存储是 GPU 板载 DRAM。各 SM 都能访问。 | 输入和最终输出的存放处。访问全局地址不代表每次都要碰到显存，缓存命中时可以就近拿到数据。 |
| **L2** | 由硬件管理，一张卡的 SM 共享。 | 两个 CTA 即使落在不同 SM，只要第二次读 `B` 时缓存行还在，就有机会通过 L2 复用。同一张卡共享，不等于跨卡共享。 |
| **L1** | 每颗 SM 本地的硬件缓存。 | 重读全局地址时，是否使用 L1、能否命中，要看指令和当时的缓存状态。程序不能把一个张量当作可直接索引的 L1 数组。 |
| **Shared memory** | 程序选择放什么；普通用法是一份缓冲给同一个 CTA 内的线程协作使用。跨线程搬运与读取需要合适的同步。 | 可以先把 A/B 的一小块搬进来，再让线程重复使用，避免把复用全交给缓存运气。 |
| **寄存器** | 硬件资源归 SM 管，通常按线程分配，编译器安排具体值。 | 一般用来放线程正在用的操作数、索引和部分累加值；不是所有线程可以直接共享的数组。 |

同样叫“片上”，不代表作用一样。**L1 是硬件决定哪些全局内存数据值得暂留；shared memory 是程序显式为某段计算安排的缓冲。** 在 CUDA 所描述的一些 NVIDIA 设计上，二者可以分用同一片物理资源，但使用方式和可见范围仍不同。Tensor Core 则是**计算单元**，不是一种缓存；Blackwell 的某些矩阵乘路径另有 TMEM 存放累加结果，不能拿它概括所有 GPU 或这份 Triton 代码的实际 lowering。[CUDA：GPU 硬件和存储模型](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html) · [NVIDIA：矩阵乘如何复用 shared memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-in-matrix-multiplication-c-ab) · [Blackwell 特性对照](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/rubin_support.html)

现在对左上角 2×3 输出块只看第一轮 K：它用 **4 个不同的 A 元素、6 个不同的 B 元素**做 **12 次乘加**。复用在算式里已经存在：每个 A 元素参与 3 个输出列，每个 B 元素参与 2 个输出行。实际 kernel 可以用 shared memory 帮 CTA 内线程兑现这种复用；这是数据流选择，不是说看到 Triton 的 `tl.load` 就能断定编译器用了 shared memory。相比之下，下面的派工顺序解决的是**两个不同 CTA** 能不能有机会复用同一块 `B`，所以讨论的是 L2。

这也解释了之前的 `L1::no_allocate`：如果结果写完后主要交给别的执行单元消费，写入方可以请求别为这次写入新占本地 L1。写入的地址和值没变，shared memory 也没动。这个提示不等于绕过 L2，更不等于另一颗 SM 已经可以安全读取；后者还要看同步协议。[PTX：store 的缓存提示](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st) · [vLLM 写回和完成通知源码](https://github.com/vllm-project/vllm/blob/b3bad87eb137c6d2fedda152ff2d3e3df3a32952/vllm/model_executor/kernels/linear/cute_dsl/gemm_rs_ar.py#L709-L771)

## 动一个变量：给 program 换张派工表

很自然的写法是让编号先扫完第一行的两个 tile，再扫第二行。但 notebook 里的 grouped ordering 会在一小组行块中，先让前两个 program 拿同一列的输出：`pid=0 → (0,0)`，`pid=1 → (1,0)`。这两个 program 需要不同的 `A` 行块，却在每轮 K 上读同一块 `B`。这样排是为了创造复用 L2 的机会；编号相邻不保证硬件也把它们排在一起，更不能从这张表推出命中率。[Triton 官方矩阵乘教程](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#l2-cache-optimizations)

下面只换派工表。两边用同一份输入、同样的 tile 和 K 分块。在 CPU 上检查输出是否一致，再看前两个编号需要哪几列 `B`。

```python
M, N, K = 5, 6, 5
BM, BN, BK = 2, 3, 2
pairs = [(i, j, k) for i in range(BM)
         for j in range(BN) for k in range(BK)]
assert (len({(i, k) for i, j, k in pairs}),
        len({(k, j) for i, j, k in pairs}), len(pairs)) == (4, 6, 12)
mt, nt = (M + BM - 1) // BM, (N + BN - 1) // BN
A = [[i + 2 * k for k in range(K)] for i in range(M)]
B = [[k - j for j in range(N)] for k in range(K)]

row_first = [divmod(pid, nt) for pid in range(mt * nt)]
grouped = [
    (m, n)
    for start in range(0, mt, 2)
    for n in range(nt)
    for m in range(start, min(start + 2, mt))
]

def run(order):
    C = [[None] * N for _ in range(M)]
    for m, n in order:
        for i in range(m * BM, min((m + 1) * BM, M)):
            for j in range(n * BN, min((n + 1) * BN, N)):
                assert C[i][j] is None
                C[i][j] = sum(
                    sum(A[i][k] * B[k][j] for k in range(p, min(p + BK, K)))
                    for p in range(0, K, BK)
                )
    return C

ref = [[sum(a * b for a, b in zip(row, col))
        for col in zip(*B)] for row in A]
assert run(row_first) == run(grouped) == ref
for name, order in [("row-first", row_first), ("grouped", grouped)]:
    print(name, order[:2], [n * BN for _, n in order[:2]])
```

```text
row-first [(0, 0), (0, 1)] [0, 3]
grouped [(0, 0), (1, 0)] [0, 0]
```

最后一列是两个 program 读的 `B` 列块起点。同样的 30 个输出元素，一种安排让这两个编号分别读两块，另一种让它们有机会复用同一块。这个小实验只验证**分工和潜在复用**，不模拟 SM 调度，也不测 GPU 时间。

读到 CuTe DSL 时，还是这两层：先问整个输出 tile 分给谁，再看 tile 内的线程与矩阵乘加指令怎么协作。下一篇把视角缩到**一个 warp 的一次读入**：32 个线程各读一个数，为什么地址挨在一起和隔着步长会有不同成本？
