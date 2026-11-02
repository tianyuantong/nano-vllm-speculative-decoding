<div align="center">

# nano-vllm-speculative-decoding

在 [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 上实现块式投机解码：DFlash 与带 Markov 修正的 DSpark。

**C=1 吞吐 2.10× · C=8 为 1.78× · C=32 为 1.43× · C=64 为 1.15×**

[![cpu-tests](https://github.com/tianyuantong/nano-vllm-speculative-decoding/actions/workflows/cpu-tests.yml/badge.svg)](https://github.com/tianyuantong/nano-vllm-speculative-decoding/actions/workflows/cpu-tests.yml)
![Python 3.10–3.12](https://img.shields.io/badge/python-3.10--3.12-blue)
![CUDA Graph](https://img.shields.io/badge/CUDA_Graph-enabled-76b900)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

[English](README.md) | **简体中文** · [快速开始](#快速开始) · [结果](docs/RESULTS.md) · [使用指南](docs/DECODING.md)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/speedup-overview-dark.svg">
  <img alt="两组 128 条留出确认集上相对普通解码的吞吐倍数（3 个 seed 中位数）。左区（块式草稿对比 0.6B 独立小模型）：Qwen3-0.6B 在并发 1/8/32/64 下为 1.54×、1.21×、0.96×、0.82×，DFlash 为 2.19×、1.80×、1.40×、1.08×；右区（DSpark 对比 DFlash）：DFlash 为 2.00×、1.68×、1.31×、1.01×，DSpark 为 2.10×、1.78×、1.43×、1.15×。" src="docs/assets/speedup-overview-light.svg" width="760">
</picture>

</div>

基于 Qwen3-8B 与 RTX PRO 6000 实测。按标准拒绝采样保持 target 输出分布无损（GSM8K 准确率相差 −1.7 ~ +0.8 个百分点）。

| C | DFlash（k=7） | DSpark（k=7） | DSpark ÷ DFlash | 每轮 token 数（DFlash / DSpark） |
|---:|---|---|---|---|
| 1 | 2.00× | 2.10× | 1.07 | 2.59 / 2.82 |
| 8 | 1.68× | 1.78× | 1.06 | 2.53 / 2.81 |
| 32 | 1.31× | 1.43× | 1.10 | 2.70 / 2.95 |
| 64 | 1.01× | 1.15× | 1.14 | 2.65 / 2.93 |

口径：同一引擎与作业下相对普通解码的稳态吞吐比，在 128 条未接触过的留出确认集上取 3 个 seed 的中位数（C=64 时 DFlash 与普通解码接近打平，DSpark 仍保持正收益）。[波动范围、测量协议和闸门 →](docs/RESULTS.md)

**对比独立小模型基线**（Qwen3-0.6B，k=3，分支 [`baseline/qwen3-0.6b-draft`](https://github.com/tianyuantong/nano-vllm-speculative-decoding/tree/baseline/qwen3-0.6b-draft)）：在上一批 128 条留出确认集上，0.6B draft 从 C=32 起慢于普通解码（C = 1 / 8 / 32 / 64 为 **1.54× / 1.21× / 0.96× / 0.82×**，每轮 2.49–2.53 token）；DFlash 在同一确认集上为 **2.11–2.24× / 1.75–1.85× / 1.32–1.40× / 1.08×**（每轮 2.75–2.85 token），各并发均为 0.6B 基线的 **1.29–1.56 倍**。[0.6B 基线详细对照 →](docs/v2/RESULTS.md)

## 加速从哪来

- **块式草稿消除了串行自回归开销。** 独立 0.6B 模型每轮要串行跑 3 步 decode 并维护双份 KV；块式 draft 一次前向同时提出 7 个候选 token，将草稿耗时压低。
- **DSpark 靠 Markov 头进一步拉高接受率。** 每轮接受 token 数比 DFlash 高 7–12%。在 C=32–64 下，针对 top-k 候选窗口裁剪的采样路径还省去了全词表 softmax 开销（在领先幅度中约占 2–6 个百分点）。
- **高并发下收益收窄。** 并发升高后，target 验证步每行的计算成本上升；同时本引擎的 prefill 步会暂停 decode，在 C=32 时占了 DSpark 稳态窗口 31% 的时间。
- **对比 vLLM 的 DSpark**（同权重、同型号 GPU，开发版 `372477d`，48 条开发集，全流程挂钟吞吐）：C=1–32 下比 vLLM 慢 7.6–7.9%；关闭投机时，nano 比 vLLM 快 1–5%。

## 新增了什么

当前引擎共 2,570 行，从 9 月份的 0.6B 独立小模型版（[`results-batched-v1`](https://github.com/tianyuantong/nano-vllm-speculative-decoding/tree/results-batched-v1)）演进而来：
- **DFlash 块式 draft**：复用 decode 注意力路径，一次跑 7 个 query 的非因果注意力，读取 target 的 5 层辅助特征，并将所有被验证位置的上下文 K/V 直接写入与 target 共用的块表（12 个引擎文件，+434/−107）。
- **DSpark 块内修正**：通过 Markov head 由左向右依次采样 7 行，并将每行加偏置后的实际采样分布交给接受规则（4 个引擎文件，+224/−12）。
- **三段 CUDA Graph**：覆盖 target 验证前向、draft 块前向，以及 DSpark 的 7 步块内采样。

核心代码：`SpeculativeDecoder.run_round`（一轮五步流水）、`DFlashDrafter.propose`、`_markov_rows` 与 `after_verify`，以及 `Attention.forward` 的 decode 分支。

## 正确性

候选 token 按标准拒绝采样验证：$u < \min(1, p_{\text{target}}(x) / p_{\text{draft}}(x))$ 时接受 $x$，否则从 $\text{normalize}(\max(p_{\text{target}} - p_{\text{draft}}, 0))$ 残差分布重采样，输出分布严格等于 target 分布（DSpark 中 $p_{\text{draft}}$ 取每行叠加 Markov 偏置后的分布）。GPU 实测闸门覆盖：采样器多步输出分布（29 个用例）、DSpark 块输出与 Markov 修正后 logits 对比参考实现 DeepSpec（相对误差 0.59–1.08%，阈值 2%）、验证步 logits 对比逐步 decode（最大绝对差 0.84，阈值 3.0），以及 GSM8K 准确率（相对普通解码 −1.7 ~ +0.8 个百分点）。

## 踩过的坑

- 输入已经是 fp32 时，`x.float()` 返回原张量引用，导致上游 RMSNorm 在 fp32 下原地改写了输入。bf16 运行不受影响，该问题是在 fp32 下与 DeepSpec 对齐 draft 数值时暴露的。
- `torch.compile` 默认重编译上限为 8，超出后新形状会回退到未编译路径，破坏了 CUDA Graph 与 eager 模式的逐位一致性；将上限调至 64 后恢复。
- 对齐 vLLM 基线时，需将其 draft 采样从默认的 greedy 改为按概率分布采样，并让 DeepSpec 的 DFlash 权重走 vLLM 的 `dspark` 路径以匹配块布局（vLLM `372477d`）。

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

DSpark 与 DFlash 每轮固定提出 7 个候选 token（由权重的块大小决定），上下文 K/V 直接存在 target 的块中，因此无需设置 `num_speculative_tokens` 或单独的 draft KV 预算；`draft_type` 需与权重匹配。用 DFlash 时传 `draft_model="/models/dflash_qwen3_8b_block7"` 和 `draft_type="dflash"`；普通解码省略 `draft_model` 即可。0.6B draft 在分支 `baseline/qwen3-0.6b-draft` 上运行。
[模型配置与 API →](docs/DECODING.md)

## 复现

```bash
python -m pytest -q    # CPU 测试，约 6 秒
python bench_spec.py --mode closed-loop --protocol benchmarks/protocol-v2.1.json --panel K128b --line dspark_k7 \
    --concurrency 8 --seed 17011 --commit <sha> --output result.json
```

测试面板由 `tools/fetch_panels.py`（D48）和 `tools/build_panel_k128.py`（K128、K128b）构建；`tools/report_v2.py` 按测量协议输出判定结论，`tools/gain_factors.py` 拆解各因素对加速比的贡献，`tools/plot_speedup.py` 绘制首屏对比图。各表对应的原始结果 JSON、闸门输出与测试面板均已打包在 release 附件 [`results-dflash-dspark`](https://github.com/tianyuantong/nano-vllm-speculative-decoding/releases/tag/results-dflash-dspark) 中。

## 深入了解

| 文档 | 内容 |
|---|---|
| [结果](docs/RESULTS.md) | DSpark 对比 DFlash、领先来源、与 vLLM 对比及各项闸门 |
| [DFlash 结果](docs/v2/RESULTS.md) | DFlash 对比 0.6B 独立小模型、单轮阶段耗时与 prefill 停顿 |
| [使用指南](docs/DECODING.md) | 模型配置、采样参数与计时 API |
| [工具](tools/README.md) | 测量、闸门与分析脚本清单 |
| [9 月版设计](docs/design/batched-speculative-decoding.md) | 0.6B draft 版本：双模型执行、KV 状态、批处理与 CUDA Graph |
| [9 月结果](docs/v1/RESULTS.md) | 0.6B draft 版本：不同 batch size 的性能、阶段耗时与重复性 |
| [优化历程](docs/PERFORMANCE.md) | 第一版性能剖析，以及推动本次重写的主要瓶颈 |

CPU 测试通过 [GitHub Actions](.github/workflows/cpu-tests.yml) 运行。

## 致谢

基于 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 构建。DFlash 的块式草稿设计源自 [DFlash](https://github.com/z-lab/dflash)（[arXiv:2602.06036](https://arxiv.org/abs/2602.06036)），DSpark 源自 DeepSeek（[arXiv:2607.05147](https://arxiv.org/abs/2607.05147)）；两份 draft 权重及对齐验证所用的参考实现均来自 [DeepSpec](https://github.com/deepseek-ai/DeepSpec)。基座模型使用 [Qwen3](https://huggingface.co/Qwen/Qwen3-8B)，底层算子基于 [FlashAttention](https://github.com/Dao-AILab/flash-attention) 与 [Triton](https://github.com/triton-lang/triton)。采样算法参考 Leviathan 等人的 *Fast Inference from Transformers via Speculative Decoding*（2023）与 Chen 等人的 *Accelerating Large Language Model Decoding with Speculative Sampling*（2023）。
[上游版本与项目历史 →](docs/UPSTREAM.md)

## 许可

[MIT](LICENSE)。
