---
layout: post
title: "模型一个字没变，为什么少了一次工具调用？"
description: "SGLang 流式 tool calling 的分块不变性：从 parallel_tool_calls 的接口语义，到 parser 怎样处理最后一块文本。"
highlight: true
---

模型吐出了两个完整的工具调用：先查巴黎天气，再查 UTC 时间。把两个调用分别交给 parser，和把它们一起塞进最后一块文本，客户端应该收到同样的两次调用。可在 [SGLang #50615](https://github.com/sgl-project/sglang/pull/50615) 追踪的路径里，后一种切法可能只留下第一个函数名，参数和第二次调用都没发出来。

模型没换，采样也没重跑。**只改变服务端递给 parser 的文本边界，应用就可能少走一步。** 这和 [DeepInfer 那篇](https://blueyo0.github.io/rolling-buffer/notes/2026/11/22/deepinfer-batch-invariance/) 里讨论的批次不变性有一点相像：先固定应该不变的东西，再动系统允许调整的边界。这里固定的是模型已经生成的字符串，不是权重、种子或 logits。

## 少一次调用，有时是策略，有时是 bug

2025 年我在 SGLang 提 [#2576](https://github.com/sgl-project/sglang/pull/2576)、参与 [#2700 的评审](https://github.com/sgl-project/sglang/pull/2700#issuecomment-2568724605) 时，讨论过一个看起来相反的问题：请求不允许并行工具调用时，模型吐出多个调用，服务应该怎么办？当时的评审指出：在讨论中的非流式实现里，Qwen 2.5 可以交出多个，某些 Llama、InternLM 路径却只取第一个。提案先把行为收敛到单调用，再补 `stream / non-stream × parallel_tool_calls=true / false` 四个组合的接口测试。

这个历史片段里，**少交一条是显式策略**。#2576 最终关闭、未合入；合入的是另一条实现 #2700。这里引用的是 #2576 和相关评审提出的接口问题，不是说 #2576 实现了今天的多调用能力，也不是用当时的默认行为描述当前版本。

现在把条件反过来：请求允许返回多个调用，模型文本里也确实有两个合法调用。如果 parser 只因为它们挤在最后一块文本里就丢掉一个，那就不是单调用策略。两段历史谈的是同一个接口的两条约束：**请求策略决定允许交付什么；流式切分不该决定实际交付什么。** “并行”在这里也不等于服务端真的并发执行了函数。parser 只负责描述调用，怎样执行由应用决定。

## 一个输入块，不一定只够状态机走一步

拿 Qwen 2.5 为例，[detector](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/function_call/qwen25_detector.py) 识别 `<tool_call>` 包裹的调用。一路往下看，有三个不同的边界：

1. **生成与解码。** 调度、批量 detokenization、较大的 `stream_interval` 或投机接受多个 token，都可能让一次回调带上更多文本。这里的 chunk 指 parser 收到的文本增量。
2. **流式解析。** detector 保留缓冲和调用进度，可能这一步先交函数名，下一步才交参数。一个增量可以只含半个结束标签，也可以装下两个完整调用。
3. **协议封装。** 服务将普通文本和结构化调用片段投到 API 响应。客户端按调用索引拼参数；有多个候选响应时，索引还得放在各自的 choice 下。SSE 事件和 TCP 读到多少字节，是更外面的边界。

问题出在第二层对第一层作了隐含假设：**处理一次文本增量，推进一次 detector，就够了。** 假如这一步只发了函数名，参数仍在内部等下一次解析。如果后面恰好还有文本，下一次回调可能顺带把它吐出来，掩盖了问题；如果这已经是最后一块，就没有下一次回调可借了。结束流并不会自动替状态机补走几步。

[#50615 的改动](https://github.com/sgl-project/sglang/pull/50615/files) 把补走的逻辑放在 `FunctionCallParser.parse_stream_chunk`：先送入新的文本，再用空串反复推进 detector。有普通文本、有调用片段，或缓冲区发生变化，就继续；看不到进展才停下来等新输入。这让已经到手的信息可以在本轮交付，也不会把“还差半个标签”误当成需要忙等的工作。这里的进展判断依赖 detector 的约定；写新的 detector 时，尤其要测试空输入不能重复吐出同一个片段。

结束时还有一条独立的责任：已经攒下的结果是否被收尾路径正确交付。[#32167](https://github.com/sgl-project/sglang/pull/32167) 指出了另一种漏法：底层先发出函数名，却没有把已解析的调用登记到收尾所读的状态里，chat 层 flush 无参数可补。[#41039](https://github.com/sgl-project/sglang/pull/41039) 则把 Qwen 2.5 的补解析放进 detector 的 `finish()`。前者问「最后怎么知道还有参数」，后者问「结束时怎么把积压工作做完」；#50615 是「每一块到手就先做完能做的」。这几条 PR 说明的是不同的修复位置，不是已经连续合入的三步升级。

## 不跑模型，也能把错钉在 parser 上

要拆掉因果里的噪声，先把生成阶段整个拿走。令 `T = A + B`，其中 A 和 B 是两个闭合的工具调用。构造不同的 chunk 列表，但让它们拼出来总是同一个 T。每次重放都从空状态开始，送入最后一块以后，再走该实现的结束处理。

分块不变性该比较的是**重组后的语义**：普通文本、调用个数、顺序、函数名和每个调用的参数对象。不同切法可以产生数量不同的 delta；不能要求事件逐个相等。反过来，只看 `finish_reason` 或「有没有收到工具名」也太松了。有名字没参数、第二个调用消失、参数拼到另一个索引，这三种错都得能测出来。可以拿非流式解析做第二个对照，但预期值还应手写一份，避免两个实现犯同一个错。

下面是用标准库就能跑的缩小版。它不导入 SGLang：toy detector 见到一帧完整的文本后，把交付刻意拆成两步（先名字、后参数）。唯一的开关是每次回调走一步，还是继续推进到无进展。它的结束动作只读已经交付的事件，所以不能用于评价 SGLang 真正的 finish 路径。

| 输入切法 | 一块只推进一步 | 推进到无进展 |
| --- | --- | --- |
| `[A, B]` | A 完整，B 留在缓冲区 | A、B 都完整 |
| `[A + B]` | 只有 A 的名字 | A、B 都完整 |

完整脚本如下；只需要 Python 3：

```python
"""CPU-only model of parser progress; not an SGLang implementation."""
import json

START, END = "<tool_call>", "</tool_call>"
EXPECTED = [("get_weather", {"city": "Paris"}), ("get_time", {"timezone": "UTC"})]


def frame(name, args):
    return START + json.dumps({"name": name, "arguments": args}) + END


class ToyDetector:
    def __init__(self):
        self.buf, self.index, self.pending = "", 0, None

    def step(self, delta=""):
        self.buf += delta
        if self.pending is not None:
            args = self.pending
            self.pending = None
            self.index += 1
            return self.index - 1, "arguments", json.dumps(args)
        end = self.buf.find(END)
        if end < 0 or not self.buf.startswith(START):
            return None
        item = json.loads(self.buf[len(START):end])
        self.buf = self.buf[end + len(END):]
        self.pending = item["arguments"]
        return self.index, "name", item["name"]


def replay(chunks, drain):
    parser, calls = ToyDetector(), {}

    def collect(event):
        if event is not None:
            index, field, value = event
            entry = calls.setdefault(index, {})
            assert field not in entry, (index, field)
            entry[field] = value

    for chunk in chunks:
        collect(parser.step(chunk))
        while drain:
            before = parser.buf
            event = parser.step()
            if event is None and parser.buf == before:
                break
            collect(event)

    # Keep names-only entries visible instead of hiding incomplete calls.
    return [
        (data.get("name"), json.loads(data["arguments"]) if "arguments" in data else None)
        for _, data in sorted(calls.items())
    ]


if __name__ == "__main__":
    a, b = (frame(*call) for call in EXPECTED)
    whole = a + b
    for label, chunks in (("one call per chunk", [a, b]), ("both in final chunk", [whole])):
        baseline, fixed = replay(chunks, False), replay(chunks, True)
        print(f"{label}:\n  one step: {baseline}\n  drain:    {fixed}")
        assert fixed == EXPECTED
    assert replay([a, b], False) == EXPECTED[:1]
    assert replay([whole], False) == [(EXPECTED[0][0], None)]
    for cut in range(1, len(whole)):
        assert replay([whole[:cut], whole[cut:]], True) == EXPECTED, cut
    assert replay(list(whole), True) == EXPECTED
    assert replay([whole[:-1]], True) == EXPECTED[:1]
    print(f"PASS: {len(whole)-1} two-piece cuts, character-wise replay and incomplete tail")
```

这份 toy 实测通过了 154 个单切点（两段）、逐字符投喂，以及尾部故意缺一个字符的检查。单切点不是所有可能的多段切法，逐字符也不是逐 token。尾部用例只说明该 toy 不会凭未闭合的帧补出第二个调用，不规定所有生产 parser 的错误处理。

真正的框架对照看 [#50615 的 CPU fixture](https://github.com/sgl-project/sglang/pull/50615/files)：同样的两个调用用 Qwen 2.5、Llama 3 和 GLM 4.5 的格式各喂一遍，覆盖每块一条和同块两条，包含 `parse_stream_end()`，按索引重组参数，也检查普通文本没有混入调用标记。这是 parser 单测，不等于把 Chat Completions、Responses、客户端 SDK 都跑了一遍。另有 [#50609](https://github.com/sgl-project/sglang/pull/50609) 提议把切分矩阵扩展到整个 registry；[SGLang 1.0 tracker](https://github.com/sgl-project/sglang/issues/50045) 将其列为发布要求。上述近期 PR 在本文核对时仍是提案；上游结果不冒充本地运行记录。

## 要调延迟，先固定这条契约

如果把回调间隔调大，只看 token 吞吐和首 token 延迟，很容易放过这类退化。对工具调用链，更直接的指标是：**从请求发出到应用拿到第一个可执行调用的时间**。名字出现不算，参数得完整、能过 schema 校验，应用还得按协议确认这一调用可以交付。做延迟分布时，把没交付有效调用的样本另报失败率，别只对成功请求算 p99。

这个实验不回答多推进几次 parser 会花多少 CPU，也不回答哪种回调间隔更快。它给性能实验划出一条线：可以晚一点交、可以多攒一些一起交，但不能因为交付方式换了，同一段模型输出就少做一件事。
