<div align="center">

# nano-vllm-speculative-decoding

基于 [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 的块式投机解码：DFlash 并行草稿与 DSpark 块内 Markov 修正。

**单请求 2.10× · 8 并发 1.78× · 32 并发 1.43× · 64 并发 1.15×**

[![cpu-tests](https://github.com/tianyuantong/nano-vllm-speculative-decoding/actions/workflows/cpu-tests.yml/badge.svg)](https://github.com/tianyuantong/nano-vllm-speculative-decoding/actions/workflows/cpu-tests.yml)
![Python 3.10–3.12](https://img.shields.io/badge/python-3.10--3.12-blue)
![CUDA Graph](https://img.shields.io/badge/CUDA_Graph-enabled-76b900)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

[English](README.md) | **简体中文** · [快速开始](#快速开始) · [结果](docs/RESULTS.md) · [使用指南](docs/DECODING.md)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/speedup-overview-dark.svg">
  <img alt="128 条留出测试集上相对普通解码的吞吐倍数（3 个 seed 中位数）。左图（DFlash 对比 0.6B 独立小模型）：0.6B 在 1/8/32/64 并发下为 1.54×、1.21×、0.96×、0.82×，DFlash 为 2.19×、1.80×、1.40×、1.08×；右图（DSpark 对比 DFlash）：DFlash 为 2.00×、1.68×、1.31×、1.01×，DSpark 为 2.10×、1.78×、1.43×、1.15×。" src="docs/assets/speedup-overview-light.svg" width="760">
</picture>

</div>

基于 Qwen3-8B 与单卡 RTX PRO 6000（600 W）实测，通过拒绝采样保证输出分布与原模型一致。

| 并发请求数 | 0.6B 小模型 (k=3) | DFlash (k=7) | DSpark (k=7) | DSpark 相对 DFlash | 每轮生成 token 数 (DFlash / DSpark) |
|---:|---|---|---|---|---|
| 1 | 1.54× | 2.00× | **2.10×** | 1.07× | 2.59 / 2.82 |
| 8 | 1.21× | 1.68× | **1.78×** | 1.06× | 2.53 / 2.81 |
| 32 | 0.96× | 1.31× | **1.43×** | 1.10× | 2.70 / 2.95 |
| 64 | 0.82× | 1.01× | **1.15×** | 1.14× | 2.65 / 2.93 |

表中为 128 条留出测试集上相对普通解码的稳态吞吐倍数（3 个 seed 中位数）。其中 0.6B 独立小模型（[`baseline/qwen3-0.6b-draft`](https://github.com/tianyuantong/nano-vllm-speculative-decoding/tree/baseline/qwen3-0.6b-draft)）测于第一批 128 题测试集（同集下 DFlash 为 2.19× / 1.80× / 1.40× / 1.08×，对应左图）；其余各列均在同一批测试集（对应右图）中背靠背测得。[完整数据与评测协议 →](docs/RESULTS.md) · [0.6B 基线对照详情 →](docs/v2/RESULTS.md)

## 性能拆解与横向对比

- **块式草稿消除了串行自回归开销。** 独立 0.6B 模型每轮需串行跑 3 步 decode 并维护独立 KV cache；块式 draft 一次前向同时提出 7 个候选 token，大幅压低草稿耗时。
- **DSpark 靠 Markov head 拉高接受率。** 每轮产出 token 数比 DFlash 高 7–12%。在 C=32–64 下，仅在 top-k 候选窗口内采样还省去了全词表 softmax 开销（阶段耗时拆解显示该项约占领先幅度的 2–6 个百分点）。
- **高并发下收益收窄。** 并发升高后，target 验证步每行的计算成本上升；同时本引擎的 prefill 步会暂停 decode，在 C=32 时占了 DSpark 稳态窗口 31% 的时间。
- **对比 vLLM**（48 条开发集，全流程挂钟吞吐，vLLM `372477d`）：跑 DSpark 时，C=1–32 下比 vLLM 慢 7.6–7.9%；关闭投机时，本引擎比 vLLM 快 1–5%。

## 核心实现

当前引擎共 2,570 行，从 9 月份的 0.6B 独立小模型版（[`results-batched-v1`](https://github.com/tianyuantong/nano-vllm-speculative-decoding/tree/results-batched-v1)）演进而来：
- **DFlash 块式 draft**：复用 decode 注意力路径，一次跑 7 个 query 的非因果注意力，读取 target 的 5 层辅助特征；5 层 draft KV cache 与 target 共用块表（`block_tables`），在验证后写入已接受位置的上下文 K/V。
- **DSpark 块内修正**：通过 Markov head 由左向右依次采样 7 个位置，并以叠加偏置后的分布参与拒绝采样验证。
- **三段 CUDA Graph**：覆盖 target 验证前向、draft 块前向，以及 DSpark 的 7 步块内采样。

入口代码：`SpeculativeDecoder.run_round`（单轮五步流水）、`DFlashDrafter.propose`、`_markov_rows` 与 `after_verify`，以及 `Attention.forward` 的 decode 分支。

## 正确性验证

候选 token 按标准拒绝采样验证：$u < \min(1, p_{\text{target}}(x) / p_{\text{draft}}(x))$ 时接受 $x$，否则从残差分布 $\text{normalize}(\max(p_{\text{target}} - p_{\text{draft}}, 0))$ 重采样，输出分布与原模型严格一致（DSpark 中 $p_{\text{draft}}$ 取每行叠加 Markov 偏置后的分布）。

GPU 实测覆盖四项检验：
- 采样器多步输出分布（29 个测试用例）
- DSpark 块输出与 Markov 修正后 logits 对比参考实现 DeepSpec（相对误差 0.59–1.08%）
- 验证步 logits 对比逐步 decode（最大绝对差 0.84）
- GSM8K 数学推理数据集准确率（84.7% 对比普通解码 85.2%，差值 −0.5 个百分点，95% 置信区间 −1.7 ~ +0.8 个百分点）

## 踩过的坑

- **fp32 下原地改写输入**：输入已经是 fp32 时，`x.float()` 直接返回原张量引用，导致上游 RMSNorm 原地修改了输入张量。bf16 路径不受影响，该问题在 fp32 下与 DeepSpec 对齐数值时暴露。
- **`torch.compile` 重编译回退**：默认重编译上限（8 次）耗尽后，新形状会回退到未编译路径，破坏 CUDA Graph 与 eager 模式的逐位一致性；将上限调至 64 后恢复。
- **vLLM 基线对齐**：vLLM（`372477d`）默认对 draft 做 greedy 采样，且 DeepSpec 的 DFlash 权重需走其 `dspark` 路径以匹配块布局；对照测试时已将其统一改为同分布随机采样。

## 快速开始

环境要求：Python 3.10–3.12，CUDA GPU（显存需容纳 target 与 draft 模型及其 KV cache）。

```bash
git clone https://github.com/tianyuantong/nano-vllm-speculative-decoding.git
cd nano-vllm-speculative-decoding
pip install torch triton flash-attn "transformers>=4.51" xxhash pytest
python -m pip install -e . --no-deps
```

填入本地模型路径即可运行：

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

DSpark 与 DFlash 每轮固定提出 7 个候选 token（由权重块大小决定），无需设置 `num_speculative_tokens`。`kv_cache_memory_bytes` 按 target 层数分配块预算，5 层 draft KV 共用同一张块表、在预算外独立分配。切换 DFlash 时传入 `draft_model="/models/dflash_qwen3_8b_block7"` 与 `draft_type="dflash"`；普通解码省略 `draft_model` 即可。0.6B 独立小模型版位于分支 `baseline/qwen3-0.6b-draft`。
[模型配置与 API →](docs/DECODING.md)

## 复现

`bench_spec.py` 开跑前会校验 GPU 型号是否与 `protocol["gpu"]["name"]` 一致。先构建测试面板，再在同一目录下跑完普通解码（`k0`）与投机解码的各并发及 seed：

```bash
python -m pytest -q    # CPU 测试，约 6 秒
python tools/fetch_panels.py && python tools/build_panel_k128.py
for c in 1 8 32 64; do
  for s in 17011 110017 130031; do
    for line in k0 dflash_k7 dspark_k7; do
      python bench_spec.py --mode closed-loop --protocol benchmarks/protocol-v2.1.json --panel K128b \
          --line "$line" --concurrency "$c" --seed "$s" --models-root /path/to/models \
          --commit "$(git rev-parse --short HEAD)" --output "runs/${line}_c${c}_s${s}.json"
    done
  done
done
python tools/report_v2.py --protocol benchmarks/protocol-v2.1.json --results runs
```

`tools/report_v2.py` 汇总吞吐与加速比，`tools/gain_factors.py` 拆解各因素贡献，`tools/plot_speedup.py` 生成加速比柱状图。各表对应的原始结果 JSON 与测试面板已打包在 release 附件 [`results-dflash-dspark`](https://github.com/tianyuantong/nano-vllm-speculative-decoding/releases/tag/results-dflash-dspark) 中。

## 深入了解

| 文档 | 内容 |
|---|---|
| [结果](docs/RESULTS.md) | DSpark 对比 DFlash、领先来源、与 vLLM 对比及正确性检验 |
| [DFlash 结果](docs/v2/RESULTS.md) | DFlash 对比 0.6B 独立小模型、单轮阶段耗时与 prefill 停顿 |
| [使用指南](docs/DECODING.md) | 模型配置、采样参数与计时 API |
| [工具](tools/README.md) | 测量、校验与分析脚本清单 |
| [9 月版设计](docs/design/batched-speculative-decoding.md) | 0.6B draft 版本：双模型执行、KV 状态、批处理与 CUDA Graph |
| [9 月结果](docs/v1/RESULTS.md) | 0.6B draft 版本：不同 batch size 的性能、阶段耗时与重复性 |
| [优化历程](docs/PERFORMANCE.md) | 第一版性能剖析，以及推动本次重写的主要瓶颈 |

CPU 测试通过 [GitHub Actions](.github/workflows/cpu-tests.yml) 运行。

## 致谢

基于 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 构建。DFlash 块式草稿设计源自 [DFlash](https://github.com/z-lab/dflash)（[arXiv:2602.06036](https://arxiv.org/abs/2602.06036)），DSpark 源自 DeepSeek（[arXiv:2607.05147](https://arxiv.org/abs/2607.05147)）；两份 draft 权重及对齐所用的参考实现均来自 [DeepSpec](https://github.com/deepseek-ai/DeepSpec)。基座模型使用 [Qwen3](https://huggingface.co/Qwen/Qwen3-8B)，底层算子基于 [FlashAttention](https://github.com/Dao-AILab/flash-attention) 与 [Triton](https://github.com/triton-lang/triton)。采样算法参考 Leviathan 等人的 *Fast Inference from Transformers via Speculative Decoding*（2023）与 Chen 等人的 *Accelerating Large Language Model Decoding with Speculative Sampling*（2023）。
[上游版本与项目历史 →](docs/UPSTREAM.md)

## 许可

[MIT](LICENSE)。
