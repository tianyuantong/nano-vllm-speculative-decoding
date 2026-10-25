<div align="center">

# nano-vllm-speculative-decoding

为 [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 实现投机解码：独立 draft 模型、DFlash 块式 draft，以及带 Markov 修正的 DSpark。

**C=1 吞吐 2.08–2.20 倍 · C=8 为 1.73–1.80 倍 · C=32 为 1.43 倍 · C=64 为 1.15 倍**

[![cpu-tests](https://github.com/tianyuantong/nano-vllm-speculative-decoding/actions/workflows/cpu-tests.yml/badge.svg)](https://github.com/tianyuantong/nano-vllm-speculative-decoding/actions/workflows/cpu-tests.yml)
![Python 3.10–3.12](https://img.shields.io/badge/python-3.10--3.12-blue)
![CUDA Graph](https://img.shields.io/badge/CUDA_Graph-enabled-76b900)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

[English](README.md) | **简体中文** · [快速开始](#快速开始) · [结果](docs/RESULTS.md) · [实现设计](docs/design/batched-speculative-decoding.md)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/speedup-k128b-dark.svg">
  <img alt="K128b 确认集上相对不投机的吞吐倍数，3 个 seed 的中位数和范围。DSpark：并发 1 为 2.08–2.20 倍，8 为 1.73–1.80 倍，32 为 1.43 倍，64 为 1.15 倍。DFlash：1.94–2.06、1.68、1.31、1.01 倍。" src="docs/assets/speedup-k128b-light.svg" width="760">
</picture>

</div>

Qwen3-8B 接 DSpark draft，RTX PRO 6000。输出分布不变（已检验）；GSM8K 准确率与不投机相差 −1.7 … +0.8 个百分点。

怎么测的：同一引擎内对不投机的稳态吞吐，未看过的 128 条确认集，每档 3 个 seed，写中位数（最小–最大）。DSpark 每档都比 DFlash 快；C=32 到 64 时，这份领先里至少有 2–6 个百分点来自 DSpark 更省的采样路径。和 vLLM 的 DSpark 比（同型号的卡、同一份权重，开发版 `372477d`，开发集，挂钟吞吐）：C=1 到 32 慢 7.6–7.9%。[测量协议、闸门和待解的问题 →](docs/RESULTS.md)

| C | DFlash（k=7） | DSpark（k=7） | DSpark ÷ DFlash | 每轮 token 数（DFlash / DSpark） |
|---:|---|---|---|---|
| 1 | 1.94–2.06 倍 | 2.08–2.20 倍 | 1.07（1.04–1.08） | 2.59 / 2.82 |
| 8 | 1.68 倍（1.67–1.71） | 1.73–1.80 倍 | 1.01–1.08 | 2.53 / 2.81 |
| 32 | 1.31 倍（1.31–1.33） | 1.43 倍（1.42–1.45） | 1.10（1.09–1.10） | 2.70 / 2.95 |
| 64 | 1.01 倍（1.00–1.01） | 1.15 倍（1.13–1.15） | 1.14（1.11–1.15） | 2.65 / 2.93 |

并发越高收益越小：verify 每行的成本在高并发时变陡，而这个引擎的 prefill 步会让 decode 停下，C=32 时占 DSpark 稳态时间的 31%。C=64 时 DSpark 仍比不投机快，DFlash 则不然。0.6B draft 和 DFlash 在之前 K128 面板上的结果见 [DFlash 结果](docs/v2/RESULTS.md)。

## 新增了什么

在 0.6B draft 版本之上，加了两种块式 draft。

**DFlash**：12 个引擎文件新增 434 行、删 107 行。
- draft 块（最后一个已提交 token 加 6 个占位 token）走 decode 路径，一次 7 个 query、非因果的注意力。
- target 在 prefill 和 verify 时导出 5 层的 hidden state（hidden + residual）。
- verify 之后把 8 个位置的上下文 K/V 全部写入；被拒的位置在被读到之前一定会被下一块覆盖。
- draft 和 target 共用块表和 pad 块，不需要第二套 KV 分配。
- 两张 CUDA graph：verify（同时导出那 5 层）和 draft 块前向。

**DSpark**，建在 DFlash 的 draft 上：4 个引擎文件新增 217 行、删 12 行（引擎现共 2832 行）。
- Markov head 给第 j 行的 logits 加一项低秩偏置，由第 j−1 行抽出的 token 决定（第 0 行用最后一个已提交的 token），所以 7 个 draft 从左到右逐个抽。
- 每一行交给接受规则的，就是它实际被抽样时用的分布，也就是加过偏置之后的分布。
- 所有请求都用 top-k 截断时，每行只在前 max_top_k + 64 个 logits 里抽；截断规则不变，只有超过 64 个 token 和截断点同分时才会和整词表不同。
- 7 步采样录进一张 CUDA graph，随机数在 graph 外面先抽好，所以 graph 和 eager 用的是同一份噪声。

关键代码：`SpeculativeDecoder.run_round`（一轮五步）、`DFlashDrafter.propose`、`_markov_rows`（DSpark 从左到右的逐行采样）和 `after_verify`、`Attention.forward` 的 decode 分支。

## 正确性

接受规则是标准的投机采样：u < min(1, p_target(x) / p_draft(x)) 时保留 draft x，否则从 normalize(max(p_target − p_draft, 0)) 重新采样，所以输出分布就是 target 的分布；DSpark 的 p_draft 是每行加过 Markov 偏置之后的分布。在 GPU 上检验过：采样器的多步输出分布（29 例）、DSpark 的块 logits 和 Markov 修正后的 logits 对 DeepSpec（相对误差 0.59–1.08%，门槛 2%）、verify 对逐步 decode 的 logits（最大差 0.84，门槛 3.0）、GSM8K 准确率（与不投机相差 −1.7 … +0.8 个百分点）。

## 快速开始

使用 Python 3.10–3.12 和 CUDA GPU，显存需容纳两个模型及其 KV cache。

```bash
git clone https://github.com/tianyuantong/nano-vllm-speculative-decoding.git
cd nano-vllm-speculative-decoding
pip install torch triton flash-attn "transformers>=4.51" xxhash
python -m pip install -e . --no-deps
```

填入本地模型路径即可生成：

```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    "/models/Qwen3-8B",
    draft_model="/models/dspark_qwen3_8b_block7",    # deepseek-ai/dspark_qwen3_8b_block7
    draft_type="dspark",
    max_num_seqs=8,
    enable_prefix_cache=False,
    kv_cache_memory_bytes=10 << 30,
    seed=0,
)
params = SamplingParams(temperature=0.7, top_k=20, top_p=0.8, max_tokens=256)

try:
    outputs = llm.generate(["Explain KV caching."], params)
    print(outputs[0]["text"])
finally:
    llm.exit()
```

DSpark 和 DFlash 的 draft 每轮提出块大小（7）个 token，上下文 K/V 存在 target 的块里，所以不用设 `num_speculative_tokens`，也不用 draft 的 KV 预算；`draft_type` 必须和权重对得上。用 DFlash 时传 `draft_model="/models/dflash_qwen3_8b_block7"` 和 `draft_type="dflash"`；用 0.6B draft 时传 `draft_model="/models/Qwen3-0.6B"`、`num_speculative_tokens=3` 和 `draft_kv_cache_memory_bytes`；普通解码省略 `draft_model` 即可。
[模型配置与 API →](docs/DECODING.md)

## 复现

```bash
python -m pytest -q    # CPU 测试：220 个通过，约 6 秒
python bench_spec.py --mode closed-loop --protocol benchmarks/protocol-v2.1.json --panel K128b --line dspark_k7 \
    --concurrency 8 --seed 17011 --commit <sha> --output result.json
```

面板由 `tools/fetch_panels.py`（D48）和 `tools/build_panel_k128.py`（K128、K128b）生成；`tools/report_v2.py` 按 protocol 的判据出结论，`tools/gain_factors.py` 把一条线对另一条线的增益拆成几个因子，`tools/plot_speedup.py` 画上面那张图。

## 深入了解

| 文档 | 内容 |
|---|---|
| [结果](docs/RESULTS.md) | 两个面板上 DSpark 对 DFlash、领先从哪来、和 vLLM 的对比、闸门 |
| [DFlash 结果](docs/v2/RESULTS.md) | K128 上的 DFlash 移植：对 0.6B draft、一轮的各阶段、prefill 停顿 |
| [实现设计](docs/design/batched-speculative-decoding.md) | 双模型执行、KV 状态、批处理与 CUDA Graph |
| [使用指南](docs/DECODING.md) | 模型配置、采样参数与计时 API |
| [工具](tools/README.md) | 测量、闸门和分析脚本，每个一行说明 |
| [9 月结果](docs/v1/RESULTS.md) | 0.6B draft 版本：不同 batch size 的性能、阶段耗时、重复性 |
| [优化历程](docs/PERFORMANCE.md) | 第一版的性能分析，以及推动本次重写的主要瓶颈 |

CPU 测试通过 [GitHub Actions](.github/workflows/cpu-tests.yml) 运行。

## 致谢

基于 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)。这个仓库加的部分：用独立 draft 模型的批量投机解码（0.6B 那条线，9 月发布）；DFlash 的这几样：draft 块的 7-query 非因果 decode 模式，target 导出 5 层的 hidden + residual，verify 后给所有被验证的位置写上下文 K/V、靠下一块覆盖被拒的位置，draft 和 target 共用块表，两张 CUDA graph（verify 和 draft 块前向）；以及 DSpark 的这几样：块内从左到右、带 Markov 修正的采样，截断行只在候选上采样，7 步采样的 CUDA graph。

DFlash 的 draft 设计来自 [DFlash](https://github.com/z-lab/dflash)（[arXiv:2602.06036](https://arxiv.org/abs/2602.06036)），DSpark 来自 DeepSeek（[arXiv:2607.05147](https://arxiv.org/abs/2607.05147)）；两份权重，以及移植时逐项比对的参考实现，都来自 [DeepSpec](https://github.com/deepseek-ai/DeepSpec)。模型用 [Qwen3](https://huggingface.co/Qwen/Qwen3-8B)，kernel 用 [FlashAttention](https://github.com/Dao-AILab/flash-attention) 与 [Triton](https://github.com/triton-lang/triton)。采样算法参考 Leviathan 等人的 *Fast Inference from Transformers via Speculative Decoding*（2023）和 Chen 等人的 *Accelerating Large Language Model Decoding with Speculative Sampling*（2023）。
[上游版本与项目历史 →](docs/UPSTREAM.md)

## 许可

[MIT](LICENSE)。
