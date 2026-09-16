# Frozen design documents

- [Earlier optimization plan](next-optimization.zh-CN.md): full original proposal; its expanded retest failed (PR #6). Its conditional model-head/single-row probability graphs were not implemented under that plan.
- [Executed 15% plan](15pct-execution.zh-CN.md): later authorized greedy and random experiments. Greedy stopped at G2 and random at R3; neither met the throughput target.

These originals are preserved verbatim, including historical status and workspace-relative references.
They record design history, not current execution instructions. The later random implementation captures
post-p/q verification; it is not the earlier single-row softmax proposal.
Current usage and limits: [decoding guide](../DECODING.md).
Full reports, referenced design materials and execution evidence: [experiment attachment](https://github.com/tianyuantong/serve-nano-vllm/releases/tag/greedy-verify-sampling-results).
