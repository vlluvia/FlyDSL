# 复现脚本

每个脚本对应优化线上的一版，可以单独运行，不需要参数。它们不只打印"优化前 vs
优化后"，而是按当时的实际顺序复现三段：**症状测量 → 定位实验 → 前后对照**。

结论是怎么得出来的，比结论本身更值得看。

## 环境

需要 gfx11\*（RDNA3）。本项目在容器里跑：

```bash
docker exec flydsl-rocm bash -lc 'cd /workspace/repo/FlyDSL && python3 docs/rdna3_moe_optimization/repro/<脚本>.py'
```

宿主机上直接跑也可以，只要 `python3 -c "import torch; torch.cuda.is_available()"`
为真。脚本自己会把仓库根目录和 `build-fly/python_packages` 加进 `sys.path`，
不用设 `PYTHONPATH`。

## 脚本

| 脚本 | 对应 | 用几张卡 | 大致耗时 |
|---|---|---|---|
| `roofline.py` | 方法本身 | 1 | ~10 s |
| `ladder.py` | o1–o6 累积阶梯（一次跑完全部） | 1 | ~30 s |
| `o1_routing_kernel.py` | o1 routing 核 | 1 | ~2 min |
| `o2_sync_free_routing.py` | o2 免同步 | 1 | ~15 s |
| `o3_tile_table.py` | o3 tile 表 | 1 | ~10 s |
| `o3_isa_dump.py` | o3 的证据（寄存器溢出） | 1 | ~10 s |
| `o4_ep_dispatch.py` | o4 EP dispatch | 1 或 2 | ~10 s / ~20 s |
| `o5_shape_override.py` | o5 shape override | 1 | ~15 s |
| `o6_grid_order.py` | o6 派发顺序 | 1 | ~10 s |
| `o7_padding_predict.py` | o7 上半：给消除 padding 定价 | 1 | ~1 min |
| `o7_padding_measure.py` | o7 下半：做出来，是负的 | 1 | ~15 s |
| `o8_ep_breakdown.py` | o8 上半：EP dispatch 逐段计时 | 2/4/8 | ~35 s |
| `o8_alltoall_size.py` | o8 的证据：all-to-all 的两个尺寸效应 | 2/4/8 | ~20 s |
| `o8_merge_meta_ab.py` | o8 下半：合并 collective，同进程 A/B | 2/4/8 | ~30 s |
| `o9_decode_overhead.py` | o9 上半：先给"拆掉主机同步"定价 | 2/4 | ~35 s（`--capture` ~2.5 min） |
| `o9_capacity_ab.py` | o9 下半：真实固定容量 + 容量因子扫描 | 2/4/8 | ~1 min（`--capture` ~4 min） |

**建议从 `roofline.py` 开始。** 它不涉及任何 MoE 特有的东西，讲的是"先量天花板、
再数你真正做了多少活、然后写一个能证伪的模型"这套方法——o6 就是这么找出来的。

**然后接着看 o7 那两个**，它们是同一套方法把自己的模型证伪掉的过程：`roofline.py` 说
padding 占了 24% 的 MAC，`o7_padding_predict.py` 据此算出混合 tile 值 +3.7%，
`o7_padding_measure.py` 把它做出来，实测 −9.5%，然后找出漏掉的那笔成本（权重流量是按
launch 付的，不按行数），最后量出整件事的天花板只有约 11%。**一个能预测的模型仍然会
在它没建模的维度上骗你**——这一课比 o1–o6 任何一版的结论都值钱。o7 没有产生任何生产
代码改动，这也是它的结论。

**o8 那三个是同一课的第二遍**，换到了多卡这一侧。`o8_ep_breakdown.py` 把 dispatch
拆成七步分别计时，加起来说"把 metadata 折进 payload 的 collective 能省 20%"；
`o8_merge_meta_ab.py` 把两条路径放进同一个进程交错跑，实测是 **+4%**。中间那个
`o8_alltoall_size.py` 是排查的产物，也是三个里最值得单独看的一个：它跟 MoE 无关，
量的是 `all_to_all_single` 本身的两个尺寸效应——**每目的地块不是 16 字节整数倍就掉到
1/10 带宽**（21 GB/s → 2 GB/s），以及**成本在每目的地 4 MiB 处有个台阶**
（16.7 → 20.7 GB/s）。第一个是地雷：给 bf16 行加两个 int32 得到 8200 字节，行数为偶
数时对齐、为奇数时不对齐，所以它只在一部分路由下爆炸。第二个解释了为什么微基准会给出
+20% 的假象——那个收益来自块尺寸跨过台阶，不来自合并。o8 同样没有产生生产代码改动。

**o9 那两个是这一课学会之后的样子**，也是唯一一轮顺序反过来的：先定价，再决定要不要写
机制。`o9_decode_overhead.py` 不写任何机制就量出了天花板——**基准测试的路由是不变的，
所以可以把 counts 算出来一次、再当常量传进 dispatch**，少掉的正好是要拆的那两步，数值
完全正确。它说 EP=4 decode 整层能省 13.5%，加图捕获 15.3%。然后
`o9_capacity_ab.py` 才是真东西（要付填充带宽），实测 10.9% / 11.6%——**第一次出现分步
估值偏保守而不是偏乐观**，因为那个 `.tolist()` 堵的是整条队列，不只是它自己那 22 µs。

`o9_capacity_ab.py` 还扫容量因子，这是它最该被单独看的地方：我原来的默认值是溢不出来的
那个上界（`tokens*topk`），实测 **+7.0%**——4× 载荷把省下的全还回去了。因子 2 才对。
同一个脚本换成 `--tokens 1024` 就能看到整条路在 prefill 上是 **+4.7%**，所以它是 decode
专用路径，不是新默认值。

**`ep_strong_scaling.py` 补的是一个口径漏洞。** 其他所有 EP 数据都是每卡 token 固定，
加卡等于加 batch；那是弱扩展，答的是"这台机器还能吃多少"。它把 batch 固定在整组上再切，
才能答"给定一个 batch，第八张卡还买到了什么"。两个口径的结论不一样，而且交换段的行为
正好相反——弱扩展里交换基本持平，强扩展里每卡发出的字节按 1/W 缩，prefill 的交换是往下
走的，decode 的反而在涨。

它顺带演示了怎么合法地跨进程比较。§1.4 说不同 session 的数不可信，但扫 world_size 只能
跨进程。办法是每轮额外量一个**漂移探针**：512 行、单专家，在任何 W 下工作量完全一样。
四轮 2458.5 / 2449.5 / 2451.6 / 2446.1 µs，散布 0.5%，于是主表可以读。探针要是动了，
主表就得作废——这比"我觉得机器挺稳的"强得多，任何跨运行对比都值得带一个。

**`v5_shape_coverage.py` 是 o5 那一课的推广，但结论反过来。** o5 在 `D4096/I14336` 上学到
"要更高的 tile_m"；当代模型的 `inter_dim` 只有 512–3072、专家数到 896，EP=8 每卡持有 112 个
专家，2048 行到达时每专家只有 18 行，`tile_m=128` 是 86% 填充。它按 EP 真实喂法配置
（expert 卡上 `topk=1`，`--rows` 是到达行数），四个新 shape 全部编译通过并对齐 torch。

它最该被单独看的是 `of floor` 那一列，而不是 TFLOP/s。低 TFLOP/s 里大部分是物理不是欠账：
Kimi-K3 流 7.40 GB 权重做 135 GFLOP，算术强度 18 FLOP/byte，实测已是下界的 82%。真正欠
2× 的是 Qwen3-30B（45%）和 Qwen3.5-35B（50%）。**先量天花板再谈优化，否则会去调一个调不动
的 shape。** 同一列还诚实地暴露了 o6 模型的一个失效区间，标成 `model falsified`。

**`v5_stage_gap.py` 和 `v5_tile_ab.py` 是"先拆缺口、再动手"的一对**，值得连着读。前者的
结论是我自己的假设被推翻：我以为 `epr` 落在 o1 的慢区间、缺口在 routing，实测 routing 只占
10%/18%，`gate_up` 一个人占 55–59%。后者的结论又反过来——`gate_up` 的最优就是通用表已有的
tile，没有空间；能挖的是 `down`，把 `BLOCK_K` 从 32 拉回 64，层级上值 +9.7% 到 +15.9%。

它们还各暴露一个陷阱。**`sweep_rdna3_moe_tiles.py` 量的不是生产用的内核变体**：它走
`exact` routing、`bounded_blocks=False`，而 `moe_forward` 走 o2 的 `bounded_blocks=True`，
grid 取上界、多余 workgroup 提前退出，尾部行为不同。同一个配置单 stage 说通用赢 5.3%、
整层说深 K 赢 1.2%，符号是相反的——所以 `v5_tile_ab.py` 才用 `moe_forward` 做 A/B。
另一个是 `--raw`：不加它只会测 EP 口径（`topk=1`），而 o3/o5 调表用的是 `topk=2`，正是会
被这个改动打回 −10.1% 的那个配置。**改通用表之前必须用 `--raw` 打一遍原配置。**

**`v5_m_major_ab.py` 是这三个里唯一一个结论为"不改代码"的**，也是它最有用的地方：o6 那条
"权重超过缓存就把 M 换到最内层"的判据，按权重总量读会让四个新 shape 全部进表、亏 1.1–18.5%，
按单个专家的 slab 读则只有 Mixtral 该进、赚 36.9%。两种读法在 o6 自己的 shape 上给出同一个
答案，所以它当年写成哪种都不影响结果，直到 shape 变了才分叉。**一条判据在只有一个数据点时
是欠定的**——这个脚本存在的意义就是把那个点变成六个。它还顺手证伪了我自己的预测（"每专家只
有一个 tile 时应该没影响"，实测 −22.5%，因为换顺序同时也把 A 复用让了出去）。

**`v5_routing_scale.py` 是"两个待办里挑一个"的那次。** o1 留了两条账：`E > 256` 建不起来，
以及 walk 的代价随 E 涨（E=128 时 89 µs，输给主机 builder），连修法都写好了。第 2 部分量的
是占比而不是微秒：0.2% 到 6.3%，而且 **Kimi-K3 的 routing 是 Qwen3.5-35B 的 2.6 倍，占比却
只有它的 1/8**——让 routing 变慢的那个 E 同时也让层变大。所以慢的那条不修，建不起来的那条修。
**"有个已知优化没做"和"值得做"是两件事，中间隔着一次测量。**

想先看全貌就跑 `ladder.py`：它在同一次运行里把 o1→o6 逐级打开，三个 (shape, token)
组合各出一张表，能直接看到"decode 的收益全在 o1/o2、prefill 的全在 o3"。单版的细节
再回到对应的脚本。实测数据整理在 [`../BENCHMARK.md`](../BENCHMARK.md)。

`o1_routing_kernel.py` 慢一些，因为它要为好几个 shape 编译核。

## 两个需要特殊命令的

`o3_isa_dump.py` 要在进程启动前设好环境变量，且**一个进程只能建一个配置**，
否则 dump 目录会混在一起：

```bash
cd <仓库根目录>
for cfg in 4,4,2,2,2 4,1,2,2,2; do
  rm -rf /tmp/isa_$cfg
  FLYDSL_RUNTIME_ENABLE_CACHE=0 FLYDSL_DUMP_IR=1 FLYDSL_DEBUG_DUMP_ASM=1 \
  FLYDSL_DUMP_DIR=/tmp/isa_$cfg \
    python3 docs/rdna3_moe_optimization/repro/o3_isa_dump.py gate_up $cfg
done
```

会看到坏 tile 是 `vgpr_count: 256 / vgpr_spill_count: 769`，修好后是 `201 / 0`。
`FLYDSL_RUNTIME_ENABLE_CACHE=0` 不能省：缓存开着的话，你可能拿到改动之前编译的
二进制而毫无察觉。

`o4_ep_dispatch.py` 单卡跑的是诊断部分（torch 各个算子的开销），双卡才能跑
host/device 两种 plan 的对照：

```bash
rm -f /dev/shm/nccl-*
torchrun --nproc_per_node=2 --master_port=29870 \
  docs/rdna3_moe_optimization/repro/o4_ep_dispatch.py
```

`rm -f /dev/shm/nccl-*` 是清理上一次残留的段。

o8 和 o9 的脚本同样要 `torchrun`，而且**必须在 `--shm-size=2g` 的容器里跑**：默认容器
`/dev/shm` 只有 64 MB，RCCL 每个 rank 要 20.25 MB，超过三个 rank 就起不来。EP=4/8
的所有数据（包括 `tests/kernels/test_rdna3_moe_ep.py` 里那两个此前一直被 skip 的用例）
都是在这样的容器里第一次跑通的：

```bash
docker run -d --name flydsl-ep \
  --network host --ipc private \
  --device /dev/kfd --device /dev/dri \
  --group-add 44 --group-add 993 \
  --security-opt seccomp=unconfined \
  --shm-size=2g \
  -e PYTHONPATH=/workspace/repo/FlyDSL/python:/workspace/repo/FlyDSL \
  -v /home/aydn/repo:/workspace/repo -w /workspace/repo/FlyDSL \
  <镜像> sleep infinity

rm -f /dev/shm/nccl-*
HIP_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29901 \
  docs/rdna3_moe_optimization/repro/o8_ep_breakdown.py --experts-per-rank 2
```

`HIP_VISIBLE_DEVICES` 决定哪几张物理卡参与。这台机器上 GPU0–3 在 NUMA 0、GPU4–7 在
NUMA 1，`rocm-smi --showtopo` 给出的权重是组内 40、跨组 72，但 EP=2 上实测**没有可测
差别**（跨组甚至略快，是噪声）——全是 PCIe，瓶颈在根桥而不在跳数，所以不用挑卡。

## 跑 o9 的图捕获部分

两个 o9 脚本的 `--capture` 都要用 `timeout` 包一层，原因不是脚本可能出错，而是
**录过图之后 `destroy_process_group()` 会挂住**，所以进程正常跑完也不会退出：

```bash
rm -f /dev/shm/nccl-*
HIP_VISIBLE_DEVICES=0,1,2,3 timeout 300 torchrun --nproc_per_node=4 --master_port=29961 \
  docs/rdna3_moe_optimization/repro/o9_capacity_ab.py --experts-per-rank 2 --capture
```

结果表在挂住之前就已经打印完了，所以 `timeout` 杀掉它不影响读数。另外
**捕获前要先在默认流上跑几轮预热**：RCCL 第一次用才分配 channel 和 buffer，直接录会
录到一个还没成形的通信器上，然后挂死（不是报错）。两个脚本都这么做了。

写多卡脚本时还有一个坑值得记：`o9_capacity_ab.py` 里"这个容量溢不溢出"最初是各 rank
各自算的，结果**一个 rank 溢出、另一个没有，两边就得到不同的 variant 列表，发出不同的
collective 序列，整个 run 挂死**——不报错，就是挂住。任何影响控制流的设备侧数值，在多卡
脚本里都必须先 `all_reduce` 成全组一致。

## 读数注意

- **只信同一次会话内的对照。** 跨会话的绝对值不可比：同一份配置在不同时间测出来
  可以差 5–10%（时钟、温度、其他负载）。所有脚本都把 A/B 放在同一个进程里跑，
  就是为了这个。
- **不要并行跑两个脚本。** 它们会争同一张卡，数出来的数没有意义。
- o5/o6 的脚本里 torch 参考实现是个朴素的 eager 循环，比
  `scripts/bench_rdna3_moe.py` 里的基线略慢，所以加速比会偏高。要报数用后者，
  这里的脚本用来看 A/B。

## 相关的正式脚本

这个文件夹是给人读的；仓库里另有三个用来干活的：

- `scripts/bench_rdna3_moe.py` — 单卡端到端，含 torch 对比
- `scripts/sweep_rdna3_moe_tiles.py` — tile 扫描，o3/o5/o6 的调参都靠它
- `scripts/bench_rdna3_moe_ep.py` — EP 分段计时
- `scripts/bench_qwen38_moe_decode.py` — Qwen3.8-Flash-Next TP=8（`E=513/topk=11/I=80`）的
  分段计时，o11/o12 用的就是它。它多打两样东西：**建了多少 tile vs 有用的有多少**，以及
  **为此流了多少权重字节 vs 有用的有多少**——空专家那笔账只在这两列上看得见，整层微秒
  数只告诉你"慢"，不告诉你慢在读权重。`--check` 对 eager torch 参考；`--vllm` 在同一个
  进程里加测 vLLM 的 `fused_experts`（o13 的对照口径），要在能 import vllm 的镜像里跑。
- `scripts/tune_vllm_moe.py` — 给 vLLM 的 triton `fused_moe` 调表，八卡按**搜索空间**分片
  而不是按 batch size 分片（o13；不需要 Ray）。`--max-block` 剪掉编译代价过高的大 tile。
