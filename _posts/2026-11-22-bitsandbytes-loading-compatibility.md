---
title: "四位权重，三路投影：bitsandbytes 的加载契约"
highlight: true
---

checkpoint 把 `q_proj`、`k_proj`、`v_proj` 分开存，推理引擎却可能把它们装进同一个 `qkv_proj`。即使两边都认识 NF4，加载器仍得知道：这份权重和它的量化状态，应该交给目标参数的哪一片？vLLM 给 Gemma 2 和 Qwen2.5 补 bitsandbytes（bnb）支持的两个 PR，改的正是这一层。

## 三份投影，一套打包参数

[Gemma 2 #8338](https://github.com/vllm-project/vllm/pull/8338) 和 [Qwen2.5 #9467](https://github.com/vllm-project/vllm/pull/9467) 给两个模型类加了同一组 `bitsandbytes_stacked_params_mapping`。后者回应 [Qwen2.5 支持请求 #8941](https://github.com/vllm-project/vllm/issues/8941)，并参考了前一个修复。

| checkpoint 里的投影 | 引擎里的参数 | 逻辑分片编号 |
|---|---|---|
| `q_proj` | `qkv_proj` | 0 |
| `k_proj` | `qkv_proj` | 1 |
| `v_proj` | `qkv_proj` | 2 |
| `gate_proj` | `gate_up_proj` | 0 |
| `up_proj` | `gate_up_proj` | 1 |

以 `k_proj` 为例：加载器先定位到 `qkv_proj`，再用编号 1 找到 K 的逻辑分片。Q、K、V 共用一个参数名，分片编号让加载器能把各自的权重和量化状态对上。这个编号不是字节偏移，也没有规定 tensor parallelism（TP）怎样切卡。后来 Qwen2 路径才在 [#9574](https://github.com/vllm-project/vllm/pull/9574) 另补加载时量化所需的模型侧 TP 映射；不能据此推断如今的 bnb 预量化路径支持多卡 TP。

## NF4 解释数值，映射决定落点

NF4 用面向近似正态分布权重的非均匀 16 值码本，结合分块缩放来表示权重；FP4 是另一种编码。QLoRA 的 double quantization 对第一层量化常数再量化，不是再量化一次原始权重；论文给定配置约省 0.373 bit/weight。NF4 存储可以配 BF16 计算，不能据此宣称原生 4-bit 乘加或吞吐收益。作为对照，LLM.int8() 对常规激活维度走 INT8，把离群维度留给 FP16 矩阵乘，再合并结果，是另一条路径。[LLM.int8 论文](https://arxiv.org/abs/2208.07339) · [QLoRA §3](https://arxiv.org/html/2305.14314#S3) · [Transformers bitsandbytes](https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes)

## 从 Unsloth 带到推理端的是什么？

做 QLoRA 时，可以用 Unsloth 的 `FastLanguageModel.from_pretrained(..., load_in_4bit=True)`，再加 LoRA 适配器并做 SFT。训练的是适配器，不是冻结的 4-bit 基座。要确认 checkpoint 的格式，查 `quantization_config`；`load_in_4bit=True` 本身说明不了磁盘上的权重怎么存。Unsloth 的 Dynamic 4-bit 会选择性保留某些模块的较高精度，不能假定与每层都量化的 checkpoint 有同样的兼容性。

部署时区分三条路：**保存适配器**并在目标框架验证对应 base revision 和 LoRA 支持；**合并导出 16-bit**，再按服务目标重做量化；或**直接部署 bnb 产物**，前提是目标加载器支持该模型、格式和并行配置。把 4-bit 张量转成 16-bit 并不能恢复已丢失的信息；若工具取回原始高精度 base 再合并，那是不同的流程。[Unsloth 微调](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide) · [Dynamic 4-bit](https://unsloth.ai/blog/dynamic-4bit) · [vLLM 导出](https://unsloth.ai/docs/basics/inference-and-deployment/vllm-guide) · [SGLang 导出](https://unsloth.ai/docs/basics/inference-and-deployment/sglang-guide)

## 部署时再查：框架走的是哪条加载路径？

以下对照基于 vLLM v0.34.0 文档、外置插件 v0.0.3 源码和 SGLang v0.5.23 源码。确认的是代码或文档列出的路径，未运行模型或比较性能。

| | vLLM v0.34.0 | SGLang v0.5.23 |
|---|---|---|
| bnb 预量化 checkpoint | 官方 bnb 插件列出 W4/W8；可通过模型配置识别。[v0.0.3 加载器](https://github.com/vllm-project/vllm-bnb-plugin/blob/d4914a9e3c5927b33b7ef941385a4fda2187e2b9/vllm_bnb_plugin/bitsandbytes_loader.py#L583-L625) 拒绝预量化 TP > 1。 | 版本源码有 W4/W8 恢复分支；此加载器明确限制预量化 checkpoint 的 TP > 1。 |
| bnb 加载时量化 | 官方文档要求安装外置插件；插件明确列出加载时 W4，不把 W8 推断为支持。 | CLI 有 bnb 选项；版本源码有加载时 NF4 分支，并依赖模型提供映射。总览的在线量化说明与源码不完全同步，这里只确认路径存在，不保证所有配置可运行。 |
| 平台 | 按该版本官方硬件矩阵核对 GPU 与后端，不把未列出的 GPU 当作明确不支持。 | 官方矩阵：NVIDIA 支持，AMD ROCm 实验性，Ascend 未列为支持。 |
| 怎么比较其他量化方案 | AWQ/GPTQ 与 bnb 都要核对格式和内核；Marlin 是内核/后端，不和格式混为一谈。FP8、NVFP4、模型权重与 KV-cache 量化另查。 | 同样按格式、模型、GPU、并行配置逐项核对；FP8、ModelOpt NVFP4 等是其他部署路线，和 bnb 加载不是一回事。 |

[vLLM bnb 文档](https://docs.vllm.ai/en/v0.34.0/features/quantization/bnb/) · [官方插件](https://github.com/vllm-project/vllm-bnb-plugin) · [vLLM 方法与硬件矩阵](https://docs.vllm.ai/en/v0.34.0/features/quantization/#supported-hardware) · [SGLang 量化文档](https://docs.sglang.io/docs/advanced_features/quantization) · [SGLang CLI](https://docs.sglang.io/docs/advanced_features/server_arguments) · [SGLang v0.5.23 加载器](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/model_loader/loader.py)

SGLang 有自己的演进历史：[早期 Gemma bnb 修复 #2557](https://github.com/sgl-project/sglang/pull/2557)，后续 [#12268](https://github.com/sgl-project/sglang/pull/12268) 清理依赖并移除旧路径，再由 [#15325](https://github.com/sgl-project/sglang/pull/15325) 恢复适配后的实现和测试。这几次改动解释了映射为什么存在。部署时能不能走通，以安装版本的加载器为准。

## 出错时，先对哪一层？

如果 checkpoint 在训练侧能读、到推理端却在组装投影时出错，优先核对 checkpoint revision、`quantization_config` 和目标模型的分片映射；先固定 TP 和加载方式。若能启动但输出异常，在相同 tokenizer、chat template 与输入下，对照已知正确的同 checkpoint 路径，比只看一句生成结果更容易定位问题。需要拆分量化误差时，再和高精度基线比较。确认加载和输出都对得上，再测显存和吞吐。
