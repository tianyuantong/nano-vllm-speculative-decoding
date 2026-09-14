# nano-vLLM 完整性能修复包 R1

## 状态与授权边界

本包基于用户附件 `current-engine-source.md` 中的27份当前引擎文件，不是仅基于7977c30的仓库。
修复包允许一次性部署到现有GPU环境，但GPU门禁未通过之前，不认可性能结果。
本地仅有CPU：Python3.13、PyTorch2.10.0+cpu。已运行63项CPU测试、全包语法检查、shell语法检查与补丁应用检查。
CUDA/FlashAttention2.8.3/Qwen/RTX5090的真实结果必须由附带作业产生；本包不预填PASS。
不更改或覆盖原216次结果。此次不重跑216、不用新24确认集、不承诺超过B。

## 已落实的修复

1. `engine/verify_graph.py`：target多位置VERIFY的精确形状Graph。
   - 最多16个key：B=1/2/3/4 × 每请求q=2/3/4/5。
   - q表示target未缓存query长度，最大q5对应最多4枚候选加pending输入。
   - 不填虚拟请求或token，不递归扩展提案，不改变剩余预算。
   - 混合query长度、未缓存前缀、B>4或资源准入不满足时沿用现有eager；q1继续原decode Graph。
2. `model_runner.py` + backend/controller：draft prefill/catchup仍计算同一模型及KV，只省去无人消费的LM head；不省去必要前向。
3. controller：第一次写入检查完整prompt；后续只检查实际未缓存后缀。所有commit token检查保留。已缓存历史不能在generate中被外部并发修改。
4. backend：在已有commit边界将invalid标志与token/count/tail一次性回传，先检查整批invalid再返回任何可提交结果。不是删掉错误检查。
5. N：一个请求的dense one-hot q一次构造为[k,V]；ID先在host验证；FP64 mass=1及invalid=false由精确构造保证。仍为dense q，不是新稀疏算法。
6. sampler：noise shape/dtype/generator/调用次数保持，仅复用新建scratch做safe noise和scores；不原地修改p/q/logits。保留非法noise、finite、非法概率检查。
7. backend：先选q的一行而不复制(k+1,V) padded q；全接受仍选同一bonus p行。

## 明确保留

`from_probs`、`from_logits`、`accept`、`residual`的数学运算及FP32/FP64精度不变。
FP64 mass、有效归一化、严格u<threshold、正部残差及其复检都保留。
correction与bonus仍都抽样，之后选择；包括未使用分支的RNG消耗。
请求级role seed和完整词表exponential抽样形状不变，不做RNG向量化或Graph内采样。
不调k/温度/EOS/filter/penalty/prefix，不提高Dynamo上限，不启用eager norm，不改attention或FA内核。
默认`RandomLLM(..., performance_mode=True, verify_graphs=True)`启用本包；
`performance_mode=False`保留同源码参考分支，但正式前后比较使用包内完整原27文件，而非只靠此开关。

## Graph中的动态信息与生命周期

- 每key精确B/q，不pad模型行数。静态ID/position/slot/cu_q/cu_k/page-table地址。
- `max_seqlen_q=q`；`max_seqlen_k=config.max_model_len`固定，当前3328。
- page table行stride固定`ceil(max_model_len / block_size)`，当前13个int32。
- `cu_q`是真实query累计长度，`cu_k`是真实每请求完整KV长度的累计值，绝不是3328乘行数。
- slot按实际逻辑position和物理块映射计算；page-table尾部用合法0初始化，读取范围由实际cu_k限定。
- 源码依据是用户指定FA2.8.3实际cu_k长度契约。不同固定maxK/stride是否在目标二进制中保持同输出，必须通过GPU门禁，不靠接口描述默认通过。
- 每次更新用新pinned host临时张量异步copy；不重写还未完成H2D的复用host缓冲。
- 首次合格调用lazy capture：当前流3次warm、旁路流2次warm，再capture。同一真实未缓存suffix可重复写，禁止写有效prefix/更改池归属。
- capture前更新固定输入；旁路流与owner流显式等待/synchronize。后续仅owner流replay，不支持跨流或并发请求进入同cache。
- 每key独立Graph private pool，不假设不同shape固定重放顺序；不共享原单步Graph pool。
- 输出`clone()`再split，调用者不会拿到下次replay改写的view。
- model/head/KV的device、dtype、pointer、shape/stride和maxK绑定核对。关闭时先释放新增Graphs，再释放模型/KV/runtime。
- capture/执行异常不自动转eager重试，RandomLLM fail-closed。仅支持范围之外、freeze miss、已知准入失败走明确fallback。

## 显存控制（不是假的硬上限）

默认进程reserved增长准入预算2GiB，设备free下限1GiB，在新增Graph前后检查；key最多16。
这是资源准入及事后检查，不是CUDA allocator硬隔离或瞬时峰值保证。OOM/capture后超界均报错停止，不减KV、不换输入、不静默降低精度。
逐次新key、blocked reason、replay key次数、capture秒数全部可从`performance_metadata()`读取。
真实显存gate保留原allocated/reserved及device-free口径；不把after-free当最低剩余。

## 捕获与性能计时

完整同seed自然warm生成后调用`freeze_performance_caches()`。
冻结之后新key走旧eager，不在计时中临时capture；结果记录capture数必须保持不变。
该策略不承诺任意未来流量100%命中Graph；报告实际key/回退计数。
加载、初始化、warm（包含lazy capture）单列；正式计时为同步边界包住完整generate。
本轮Graph几何是新的数值执行路径：若同条件自然输出/stop/RNG不一致，先停止性能裁决，不用旧TV限制豁免，也不改原TV结论。

## 应用方式

解压外层发布包后设`PACKAGE=/.../nano-vllm-performance-repair-r1`。
在包含当前`nanovllm/`的仓库根目录执行：

```bash
python "$PACKAGE/apply_repair.py" --root "$PWD" --check
python "$PACKAGE/apply_repair.py" --root "$PWD" --apply
python -m pytest -q tests/perf_repair
python -m compileall -q nanovllm tools tests/perf_repair
```

应用器先要求27份原源码hash全部匹配；拒绝新增引擎文件/局部改过的版本，拒绝覆盖同名新增文件。
补丁是“附件当前版本 → R1”，不要再叠加附件中那份7977c30 diff。
外层`reference-engine/nanovllm`就是附件原27份源码，为同期原版分母保留；同一Python环境通过源码路径隔离，不复制环境或下载模型。

## GPU作业

使用现有Torch2.9.1+cu130、FA2.8.3、RTX5090环境。作业不安装依赖、不访问网络、不检查SSH、不自动sbatch。
先用原流程核对远端队列、GPU、模型17文件与源码身份；环境路径由Codex提供。
Slurm模板的partition/node/account必须沿用本站既有RTX5090参数，不根据模板臆造。

```bash
export TARGET_MODEL=/existing/Qwen3-8B
export DRAFT_MODEL=/existing/Qwen3-0.6B
export REFERENCE_ROOT="$PACKAGE/reference-engine"
export FROZEN_DEBUG_INPUTS=/existing/frozen_debug8_r1_schema.json
export REPAIR_RESULTS=/existing/results/perf-repair-r1-NEW
# 从已应用补丁的仓库根目录，带本站既有GPU/partition参数提交：
# sbatch ... tools/perf_repair_job.sbatch
```

`FROZEN_DEBUG_INPUTS`严格schema如下；只是封装既有旧8条token IDs与稳定request_id，不能重新编码/截断/改ID：

```json
{"groups":[
  {"group_id":"0","requests":[{"request_id":"既有ID","token_ids":[17,18]}]},
  {"group_id":"1","requests":[{"request_id":"另一个既有ID","token_ids":[19,20]}]}
]}
```

上例仅解释schema，不是可运行负载；实际每组必须恰好4请求，合计8个原稳定ID。
此处没有原实验输入附件，不能伪造它们。Codex只需把现有冻结数组/ID原样封装，控制器会记录文件与逐任务数据。
可先在CPU运行`tools/perf_repair_compare.py ... --output-dir /new/dry-run --dry-run`，仅展开16进程/32计时调用，不加载模型。

作业内顺序固定，任一失败即停止后续步骤：

1. `PERF_REPAIR_REQUIRE_CUDA=1 pytest`：同源码数学、状态、metadata测试；实际CUDA概率测试只有CUDA可用时才运行，缺CUDA直接失败，不能CPU通过冒充GPU。
2. `perf_repair_gpu_gate.py`：q5优先，实际Qwen覆盖16个B/q key；动态position/cu_k/物理块改变，真实最长K2774，混合3/1/2回退；Graph/原eager logits与KV比较；输出生命周期与A→B→A；同engine参考/S0/S1无hook输出-stop-RNG。不是性能测量。
3. `perf_repair_compare.py`：旧2组×4模式×原版/修复版各独立进程，每进程完整same-seed warm＋2次计时。共32次计时，不跑216或新24。cap512，自然EOS，seed17011，共同5328/3328与原KV预算。B同时重测。

比较器读取完整原版源码，不拿老216时间当分母。无hook/profiler；每模式/版本独立进程。
每次同warm重放；同模式原版/修复版以及同版本S0/S1严格核对输出-stop-RNG。
性能摘要先mean重复再sum两组，明确不是32调用实际总时间。结果原样保留，不剔异常值。
此包多项优化合并，效果只能称修复包净收益，不能拆成每项优化贡献或纯D2H/launch收益。

## 停止与产物

默认Slurm上限45分钟，其中自然比较内部上限1800秒，单worker300秒。预算是硬上限，不承诺耗时，不自动重试。
身份不匹配、GPU等式失败、KV/概率/RNG/回收错误、capture/OOM、warm后新增capture、worker失败/timeout都停止。
unsupported/mixed/frozen miss是明确fallback，不是将运行错误吞掉；报告这些计数。
若性能没改善或仍慢于B，如实完成报告；不能自动换模型、调k或重跑找好结果。
回传：CPU/CUDA pytest结果、gpu-gate.json、比较plan/completion、16进程原结果与log、实际Graph key/capture/回退、源码及模型身份、清理/Slurm状态。
原2/64 TV门禁未通过、严格B实现分布等价OPEN和质量未评估的记录全部保留。
