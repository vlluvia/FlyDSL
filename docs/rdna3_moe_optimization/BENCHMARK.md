# 实测数据：v1–v3 / o1–o14 / torch

gfx1100（W7900 级）· bf16 · wave32 · 2026-08-22 · 一次会话内测完。

基线是 eager torch：按专家分组后每专家一次 `matmul`。TFLOP/s 只算**有用**的 FLOP
（`2 × tokens × topk × N × K`），routing 补齐出来的 padding 行是内核真做了但不记功的活。

复现命令在每节标题下。

---

## 0. 先说可比性

这份数据里有两条不同的线，**不能放进同一个 TFLOP/s 列去比**。

**v1 / v2 / v3 不是版本递进，是一层里的三个阶段。** v1 是单个 `X@W`（N=inter），v2 是融合
gate+up（N=2×inter，含 silu 和 scatter），v3 是 down 投影（K=inter，含路由权重）。它们算的
不是同一件事，所以"v3 > v1"没有意义；三者的 TFLOP/s 只能横向读作"这个阶段把机器用得多好"。

唯一有意义的跨版本对比是 **v1 linear vs v2 gemm1**：融合有没有让效率变差。见 §3。

**o1–o6 才是真正的递进，但作用域不重叠：**

| 版本 | 动的是 | 在什么指标上体现 | 在什么指标上是 0 |
|---|---|---|---|
| o1 / o2 | routing | `layer+route`，且只在 token 少时显著 | 纯 GEMM |
| o3 / o5 / o6 | GEMM 的 tile 与派发顺序 | 所有 GEMM 指标 | routing |
| o4 | EP dispatch 的 plan | 双卡下的 `dispatch` µs | 单卡任何指标 |

所以下面按作用域分了四节，而不是硬凑成一条曲线。

**torch 基线本身会随专家数变差。** 它是逐专家的 python 循环，E=16 就跑 16 次。shape C
的加速比因此被抬高了一截——那部分是"基线差"不是"内核快"。跨 shape 比加速比要记着这件事；
比 TFLOP/s 则没有这个问题。

**跨会话的绝对值不可比。** 同一配置在不同时间能差 10%（时钟、温度、其他负载）。除 §12
之外，本页所有对照都在同一次运行内完成。§12 要扫 world_size，只能跨进程，所以它每轮
额外量一个工作量恒定的**漂移探针**来证明机器没动——需要跨运行对比时照这个做。

---

## 1. o1–o6 累积阶梯

```bash
python3 docs/rdna3_moe_optimization/repro/ladder.py
```

测的是 `layer+route`：一层 GEMM 加上每次调用重建 routing，也就是一个 decode step 真正付的钱。
每一级都是当前代码里真实存在的开关（`routing_impl` / `_TILES` / `_SHAPE_TILES` /
`_SHAPE_M_MAJOR`），所以这是真的累积测量，不是复述。

| 配置 | A / 32 / t16 | 累积 | A / 1024 / t128 | 累积 | B / 1024 / t128 | 累积 |
|---|---|---|---|---|---|---|
| torch | 1.067 ms | – | 1.338 ms | – | 15.32 ms | – |
| v3 起点 | 0.250 ms | 1.00× | 2.257 ms | 1.00× | 61.72 ms | 1.00× |
| +o1 routing 核 | 0.210 ms | 1.19× | 2.098 ms | 1.08× | 62.77 ms | 0.98× |
| +o2 免同步 | **0.094 ms** | **2.65×** | 2.114 ms | 1.07× | 66.81 ms | 0.92× |
| +o3 tile 表 | 0.096 ms | 2.60× | **0.402 ms** | **5.61×** | **12.64 ms** | **4.88×** |
| +o5 shape override | 0.101 ms | 2.47× | 0.403 ms | 5.61× | 11.39 ms | 5.42× |
| +o6 派发顺序 | 0.102 ms | 2.46× | 0.403 ms | 5.59× | **11.04 ms** | **5.59×** |

A = D2048/I768 E8 topk2，B = D4096/I14336 E8 topk2。

**怎么读：**

- **decode 的收益全在 o1/o2**（2.65×），之后走平——那时瓶颈是 routing 的启动与同步，不是算力。
- **prefill 的收益全在 o3**（一步 4.9–5.6×），因为 pre-o3 的 128 tile 会寄存器溢出 769 次。
  o1/o2 在这里几乎没动：routing 的百来微秒淹没在毫秒级的 GEMM 里。
- **A 在 +o5/+o6 上原地不动是正确的**：那两张表里只有 D4096/I14336 的键，对别的 shape
  按定义就是空操作。0.02× 的抖动是噪声。
- **B 的 +o1/+o2 是负的**（1.00 → 0.98 → 0.92）。整层 62 ms 的时候 routing 是噪声，这几个
  数只反映会话内漂移，不要当成 o1/o2 有害。

> pre-o3 的 tile 表是**重建**的：o3 把一张共享表换成了按 stage 分的表，原表已不在代码里。
> 这里用的是 o3 记录中的"原 tile"一列，它复现出了文档里那次 128 tile 塌陷，所以至少是对的
> 那张表，但别当成逐位还原的历史代码。

---

## 2. 各 shape 下的整层吞吐

```bash
python3 scripts/bench_rdna3_moe.py --tokens 32,256,1024 --model-dim 4096 --inter-dim 14336 --tiles 32,64,128
```

每格取最好 tile_m。`layer` = gemm1 + gemm2 + reduce，routing 预先建好。

| shape | 配置 | token | 最好 tile_m | FlyDSL | TFLOP/s | torch | TFLOP/s | 加速比 |
|---|---|---|---|---|---|---|---|---|
| A D2048/I768 | E8 topk2 | 32 | 16 | 70.9 µs | 8.52 | 1076.8 µs | 0.56 | 15.19× |
| | | 256 | 64 | 140.4 µs | 34.42 | 1116.2 µs | 4.33 | 7.95× |
| | | 1024 | 64 | 345.2 µs | 55.98 | 1340.4 µs | 14.42 | 3.88× |
| B D4096/I14336 | E8 topk2 | 32 | 64 | 4.44 ms | 5.07 | 7.18 ms | 3.14 | 1.62× |
| | | 256 | 128 | 5.21 ms | 34.65 | 8.06 ms | 22.38 | 1.55× |
| | | 1024 | 128 | **11.71 ms** | **61.62** | 15.16 ms | 47.59 | 1.29× |
| C D2048/I1408 | E16 topk4 | 32 | 64 | 476.0 µs | 4.65 | 2062.2 µs | 1.07 | 4.33× |
| | | 1024 | 128 | 1.33 ms | 53.20 | 3.18 ms | 22.26 | 2.39× |

C 从来没有为它调过任何参数，走的是 o3 的通用表——2.39× / 53.2 TF 说明通用表本身站得住，
不是只在调过的 shape 上才行。

**B 在 32 token 上只有 1.62×，而且这条是真的慢。** 2.82 GB 权重、只有 64 行要算，完全是带宽题：
把全部权重读一遍在 830 GB/s 下就要 3.4 ms，实测整层 4.44 ms，已经贴着 DRAM 地板。这个 shape
的 decode 想再快只能少读权重（量化，或者别把整层权重都过一遍），调 tile 没有意义。

---

## 3. v1 vs v2：融合有没有让效率变差

1024 token，最好 tile，单位 TFLOP/s。

| shape | tile_m | v1 linear（未融合） | v2 gemm1（融合 gate+up） | v3 gemm2 | 融合的效率差 |
|---|---|---|---|---|---|
| A D2048/I768 | 64 | 54.92 | 58.36 | 55.58 | **+6%** |
| B D4096/I14336 | 128 | 55.96 | 63.83 | 57.88 | **+14%** |
| C D2048/I1408 | 128 | 55.57 | 56.27 | 56.19 | **+1%** |

v2 的 N 是 v1 的两倍且多带 silu 与 scatter，却在三个 shape 上都比 v1 效率更高——一份 A 喂
两条 B 流，A 的 LDS 流量被摊薄了。这也是 o3 当时的验收目标：融合后的 gemm1 要反超未融合的
linear。

---

## 4. o1 / o2 的作用域

```bash
python3 docs/rdna3_moe_optimization/repro/o2_sync_free_routing.py
```

| shape / token / tile | 整层（routing 预建） | o2 免同步 | o1 读回 | pre-o1 主机 | pre-o1 额外开销 |
|---|---|---|---|---|---|
| A / 32 / 16 | 70.9 µs | 87.1 µs | 171.9 µs | 251.3 µs | **+254%** |
| A / 1024 / 64 | 345.2 µs | 358.9 µs | 461.1 µs | 584.5 µs | +69% |
| B / 1024 / 128 | 11709.5 µs | 11849.2 µs | 11778.1 µs | 11868.5 µs | +1.4% |

三种实现输出**逐位相同**（maxdiff 0.0e+00）。

routing 的绝对开销基本是常数（主机约 130 µs、核约 15 µs）。整层 11.7 ms 时这 130 µs 是 1%
的噪声；整层 70 µs 时同样的 130 µs 是 254%。**所以 o1/o2 是 decode 优化，报它们的收益必须
说清 token 数。**

---

## 5. v3-EP + o4：专家并行

```bash
rm -f /dev/shm/nccl-*
torchrun --nproc_per_node=2 --master_port=29881 scripts/bench_rdna3_moe_ep.py --tokens 1024 --tiles 64,128
```

EP=2，每 rank 的分段耗时。D2048/I768、每卡 8 专家（共 16）、topk=2。

| token/rank | tile | plan | dispatch | layer | combine | total | TFLOP/s |
|---|---|---|---|---|---|---|---|
| 1024 | 64 | **device (o4)** | **493.8** | 393.4 | 352.0 | **1180.1** | 25.48 |
| 1024 | 64 | host | 604.7 | 382.2 | 351.3 | 1339.3 | 22.45 |
| 1024 | 128 | **device (o4)** | **481.2** | 430.1 | 353.1 | **1207.4** | 24.90 |
| 1024 | 128 | host | 673.2 | 429.6 | 351.9 | 1445.8 | 20.79 |
| 32 | 16 | **device (o4)** | **379.2** | 105.9 | 126.8 | **581.4** | 1.62 |
| 32 | 16 | host | 539.7 | 107.2 | 105.5 | 827.9 | 1.13 |

这里的 TFLOP/s **不该和 §2 的单卡数字比**：分母里含 dispatch 和 combine，而那两段大部分时间
在 PCIe 上，不是算力。

**o4 买到了什么**：dispatch 在 decode 上 539.7 → 379.2 µs（−30%），prefill 上 604.7 → 493.8
（−18%）、tile 128 时 673.2 → 481.2（−29%）。两条路径产出的 permutation / counts / payload
逐位相同，所以这是纯开销的削减。

**剩下的是传输本身**：prefill 每次 dispatch 有 4.19 MB 出卡，实测 payload all-to-all 就要
305.4 µs——这些卡之间只有 PCIe P2P（12–19 GB/s，无 xGMI）。decode 时 payload 只有 0.13 MB
却仍要 49.2 µs：每个 collective 不管搬多少都有约 45–50 µs 的固定开销。

EP=4/8 测不了：`/dev/shm` 只有 64 MB，RCCL 每个 communicator 固定要 20.25 MB，需要
`docker run --shm-size=2g`。

---

## 6. 把这些数字放到天花板上读

```bash
python3 docs/rdna3_moe_optimization/repro/roofline.py
```

| 量 | 实测 |
|---|---|
| rocBLAS 稠密 bf16 `4096³` | 66.73 TFLOP/s |
| rocBLAS 稠密 bf16 `8192³` | 68.99 TFLOP/s |
| 流式读带宽 | 830.5 GB/s |
| rocBLAS 在单专家形状 `256×28672×4096` | 61.78 TFLOP/s |
| rocBLAS 在单专家形状 `256×4096×14336` | 56.31 TFLOP/s |

**§2 里 61.62 TF 看着是 dense 峰值的 89%，其实已经超过了。** `tile_m=128` 下 B 有 **24.4%**
的 MAC 花在 padding 行上（每专家平均 256 行，正好卡在 128 的边界）。TFLOP/s 那列不给这些行
记功，所以 61.6 是"有用吞吐"；按真实做过的 MAC 算是 **81.8 TF/s**，高于 rocBLAS 的 69。

这就是 o6 的由来：既然库不是天花板，62 TF 就不是终点。

顺带注意 rocBLAS 在单专家形状上只有 56–62 TF，低于它自己在 `4096³` 上的 66.7：M 只有 256 行
的瘦矩阵本来就是库放弃吞吐的地方，而这恰好是 MoE 必然产生的形状。

---

## 7. o7：padding 那 24.4% 值多少时间（不是 24.4%）

```bash
python3 docs/rdna3_moe_optimization/repro/o7_padding_predict.py
python3 docs/rdna3_moe_optimization/repro/o7_padding_measure.py
```

§6 那个 24.4% 很容易读成"还有 24.4% 的时间可捡"。o7 把它直接量了：喂一份恰好不需要
补齐的 routing（每专家正好 256 行 = 两个满 tile），和真实 routing 比。

| stage | routing | tiles | 行数 | 浪费 | ms | 有用 TF/s |
|---|---|---:|---:|---:|---:|---:|
| gate_up | 真实 | 20 | 2560 | 25.0% | 6.99 | 68.9 |
| gate_up | 均衡 | 16 | 2048 | 0% | 6.38 | 75.4 |
| down | 真实 | 20 | 2560 | 25.0% | 3.87 | 62.2 |
| down | 均衡 | 16 | 2048 | 0% | 3.32 | 72.5 |

**整层 1.16 ms / 10.80 ms ≈ 11%，而且这是"免费拿掉"的上限。** 一个 padding 行的边际成本
约 1.1 µs，平均行 2.7 µs。**工作量占 24% 不等于时间占 24%。**

实际方案（每专家一个 128 主 tile + 一个 64 尾 tile，浪费 25% → 12.5%）：

| | ms | vs uniform |
|---|---:|---:|
| uniform 128 | 10.80 | — |
| mixed 128+64 | 11.92 | **−9.5%** |

预测是 +3.7%。差在这张表——把 stage1 的两次 launch 单独计时：

| launch | block | tiles | 行数 | ms | MAC TF/s |
|---|---|---:|---:|---:|---:|
| uniform | 128x64 | 20 | 2560 | 7.01 | 85.8 |
| main | 128x64 | 12 | 1536 | 5.13 | 70.3 |
| tail | 64x128 | 12 | 768 | 3.56 | 50.7 |

同一个 128 高度，行数 2560 → 1536，速率 85.8 → 70.3。每次 launch 都要整流一遍 8 个专家的
权重（1.88 GB，约 2.26 ms），这笔钱按 launch 付，摊在这次带了多少行上。**拆成两次 launch
就是白买一遍权重，2.26 ms 比整个 1.16 ms 的预算还大**——所以在这个 shape 上任何两次 launch
的方案都不可能赢，尾 tile 取多大都一样。详见主文档 §5.1。

---

## 8. EP 扩展：单机多卡到底能走多远

`/dev/shm` 默认 64 MB、RCCL 每 rank 要 20.25 MB，所以在此之前 EP 只测过 2 卡。换成
`--shm-size=2g` 的容器后 EP=4/8 第一次跑通（`test_rdna3_moe_ep.py` 里那两个用例此前一直
被 skip，现在是真跑过的）。

以下全部在同一个容器、`D4096/I14336`、8 专家、topk=2、每 rank 1024 或 32 token。注意
tokens 是**每 rank** 的，所以 EP 翻倍时总 token 数也翻倍——prefill 那列量的是吞吐扩展。

```bash
HIP_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29897 \
  scripts/bench_rdna3_moe_ep.py --tokens 1024 --model-dim 4096 --inter-dim 14336 \
  --experts-per-rank 2 --topk 2 --tiles 128
```

| | EP=1 | EP=2 | EP=4 |
|---|---:|---:|---:|
| prefill layer (µs) | 11258 | 10417 | 9619 |
| prefill dispatch + combine | 310 + 167 | 630 + 539 | 1053 + 855 |
| prefill total | 11869 | 11554 | 11451 |
| **prefill 吞吐加速** | 1.00× | **2.05×** | **4.15×** |
| decode layer (µs) | 4373 | 2186 | 1162 |
| decode dispatch + combine | 315 + 95 | 362 + 117 | 371 + 132 |
| decode total | 4713 | 2581 | 1579 |
| **decode layer 加速** | 1.00× | **2.00×** | **3.76×** |
| 交换占比 prefill / decode | 4.0% / 8.7% | 10.1% / 18.5% | 16.7% / 31.8% |

两件事值得注意。

**prefill 是超线性的**（4.15× > 4×），因为每 rank 持的专家权重随 EP 变少，layer 从 11.3 ms
降到 9.6 ms。**decode 的 layer 近乎完美线性**，原因更直白：32 个 token 的 layer 要 4.37 ms，
而把 2.82 GB 权重读一遍就是这个量级——decode 是权重带宽绑定的，EP 把权重切开就等于把时间
切开。

**交换段几乎不随 EP 增长**（decode 409 → 478 → 502 µs），但 layer 在缩，所以占比从 8.7% 涨到
31.8%。按这个趋势外推 EP=8，decode 的 layer 约 600 µs 而交换约 520 µs，扩展会在那里明显拐弯。
prefill 还早得多。

EP=1 那一列有个数字值得单独看：**payload 是 0.00 MB，交换段仍然要 409 µs。** 那是纯开销，
和 o1/o2/o4 是同一类问题，也是 §5.4 真正要打的东西。

顺带排除一个变量：这台机器 GPU0–3 在 NUMA 0、GPU4–7 在 NUMA 1，`rocm-smi --showtopo` 的权重
是组内 40、跨组 72。EP=2 上实测 (0,1) prefill 最佳总时 1211.6 µs、(0,4) 1170.3 µs——**跨 NUMA
甚至略快，说明没有可测差别**。全是 PCIe，瓶颈在根桥不在跳数，EP=4 不必挤在一个 NUMA 节点里。

---

## 9. o8：合并那个 collective 值多少（不值）

```bash
torchrun --nproc_per_node=4 docs/rdna3_moe_optimization/repro/o8_ep_breakdown.py --experts-per-rank 2
torchrun --nproc_per_node=4 docs/rdna3_moe_optimization/repro/o8_alltoall_size.py
torchrun --nproc_per_node=4 docs/rdna3_moe_optimization/repro/o8_merge_meta_ab.py --experts-per-rank 2
```

dispatch 送两个 collective：activation，和每行 8 字节的 metadata。小的那个几乎和大的一样贵
（固定开销 45–60 µs），把它折进大的行里看起来是白捡。逐段计时也是这么说的（EP=4, prefill）：

| 步 | µs |
|---|---:|
| plan 核 (o4) | 20.0 |
| counts all-to-all | 56.8 |
| payload gather | 41.0 |
| counts 回读 `.tolist()` | 22.5 |
| payload all-to-all | 875.9 |
| meta all-to-all | 48.5 |
| 合并方案：填充 + 一次 all-to-all + 解包 | 49.4 + 656.1 + 28.9 |

加起来是 −20%。**实测是 +4%。** 同进程交错、7 轮取中位数：

| | dispatch (µs) | vs split | layer (µs) | vs split |
|---|---:|---:|---:|---:|
| split（现状，2 次 a2a） | 1021.9 [1007.9, 1032.5] | — | 11742.9 | — |
| merged | 1065.5 [1042.7, 1066.3] | **+4.3%** | 11475.8 | −2.3% |
| merged+256B | 936.3 [930.3, 937.7] | **−8.4%** | 11369.8 | −3.2% |
| merged+512B | 958.0 [949.4, 984.8] | −6.3% | 11617.2 | −1.1% |

关键在 `merged` 和 `merged+256B` 的差：**它们做的事完全一样，只差行宽。** 收益不来自少一次
collective，来自块尺寸。`o8_alltoall_size.py` 把 `all_to_all_single` 单独拎出来扫，两个效应：

| 每目的地块 | 16 字节整数倍 | GB/s |
|---:|---|---:|
| 3.943 MB | 否 | **1.02** |
| 4.194 MB | 是 | 16.70 |
| 4.228 MB | 否 | **2.06** |
| 4.260 MB | 是 | **20.65** |
| 4.719 MB | 是 | 20.76 |
| 8.389 MB | 是 | 21.50 |

**其一是地雷：块不是 16 字节整数倍就掉到 1/10 带宽。** 块是 `行数 × 行宽`，行数由路由决定，
所以只有行宽本身是 16 的倍数才安全。给 D4096 的 bf16 行加两个 int32 得到 8200 字节——偶数行
对齐、奇数行不对齐，于是它只在一部分路由下爆炸。第一版合并实现比它要替换的东西慢 10 倍，
全部是这个原因。

**其二是台阶：4.194 MB（正好 4 MiB）之下 16.7 GB/s，之上 20.7。** dispatch 的 payload 是
`tokens × topk / world_size` 行 × `model_dim × 2` 字节，在真实模型里全是 2 的幂，所以它**按
构造就落在整数尺寸上**——EP=4、每 rank 1024 token、D4096 的块恰好是 4 MiB，在慢的那一侧。
把行填过去值 8.4%。

`NCCL_PROTO`（Simple/LL128）和 `NCCL_BUFFSIZE`（1/2/8 MiB）四种设置曲线完全重合，所以这
不是能用环境变量调掉的东西。

**没有改生产代码。** 合并本身在 prefill 是 +4%、decode 的 −2% 区间重叠；唯一硬的收益
（−8.4%）来自跨过 4 MiB 台阶，而那个台阶在哪一侧跟着 `world_size` 走——同样的填充在 EP=2
（块已经是 8.4 MB）是 +6.6%。要吃这块收益得按块尺寸做条件判断，为 layer 层面约 2% 的量。
详见主文档 §5.3。

---

## 10. o11 / o12：专家数远大于行数时（Qwen3.8-Flash-Next, TP=8）

```bash
python3 scripts/bench_qwen38_moe_decode.py --tokens 1,16,760 --check
```

`H=2560 / I=80 / E=513 / topk=11`，2026-09-11 一次会话内测完。这个 shape 与 §1–§9 全部
不同：**一步里的行数少于专家数**（decode 时 11 行对 513 个专家）。

### 逐段，M=1

| | routing | gemm1 | gemm2 | reduce | 整层 |
|---|---:|---:|---:|---:|---:|
| 改前 | 109.4 | 585.0 | 269.2 | 12.9 | **1029.5** |
| +o11（空专家不占 tile） | 109.0 | 25.4 | 13.2 | 12.9 | 160.4 |
| +o12（原子直方图 routing） | 22.1 | 25.4 | 13.2 | 12.9 | **52.2** |
| 同上，图重放 | — | — | — | — | **41.3** |

tile 数 513 → 11，为此流的权重 601 MiB → 12.9 MiB。12.9 MiB 在这张卡上就是约 16 µs，
所以 38 µs 的两个 GEMM 已经贴着带宽下界，剩下的空间在 launch 数不在内核里。

### routing 随 token 数

| tokens | 配对遍历（o1 布局） | 原子直方图（o12） |
|---:|---:|---:|
| 1 | 16 | 6 |
| 16 | 42 | 6 |
| 760 | 1649 | 13 |

旧布局是每线程 `E × rows / 256` 次配对检查，斜率随 E 和 rows 同时涨；新的每行只读两遍。
E=513 的 routing 核 MLIR 编译时间也从约 700 s 降到约 15 s。

### 整层

| | M=1 | M=760 |
|---|---:|---:|
| FlyDSL 改前 | 1031 µs | 3063 µs |
| FlyDSL 改后 | **41–52 µs** | **1121 µs** |

`vllm bench serve`，760 输入 / 16 输出、TP=8、CUDA Graph、4 个 seed：

| | TPOT | TTFT |
|---|---:|---:|
| 原生 vLLM（默认 MoE 配置） | 19.38 ms | — |
| FlyDSL 改前 | 66.94 ms | 412 ms |
| FlyDSL 改后 | **19.86 ms** | 332 ms |

改后的 trace 里 MoE 只占 decode 步约 11%（每层每步 `grouped_gemm` 36.5 µs × 2 +
`routing` 7.0 + `moe_reduction` 3.3），attention 26%、TP all-reduce 23%——继续压 MoE 的
边际收益已经很小。原生那一列走的是默认配置，补上调优后的对照见 §11。

---

## 11. o13：把原生 vLLM 调优补上，同口径重比

```bash
# 调表（八卡分片，剪掉编译代价过高的大 tile）
python3 scripts/tune_vllm_moe.py --num-experts 513 --topk 11 --hidden-size 2560 \
  --shard-intermediate-size 160 --batch-size 1 2 4 8 16 512 --max-block 128 --save-dir /out

# 同一进程里比
VLLM_TUNED_CONFIG_FOLDER=/out python3 scripts/bench_qwen38_moe_decode.py --tokens 1,16,760 --vllm
```

2026-09-12，gfx1100（W7900D）。§10 的 291 / 1852 µs 来自另一套测量，这里全部重测。

### 内核，同一进程 / 同一输入 / 同一 `_bench_graph`

`E=513 / topk=11 / K=2560 / N=80`，图重放，三轮取中位数（µs）：

| | M=1 | M=16 | M=760 |
|---|---:|---:|---:|
| vLLM 默认配置 | 61.1 | 311.5 | 1920.9 |
| vLLM 调优后 | **52.7** | **295.0** | **1097.1** |
| FlyDSL（o12 之后） | 57.1 | 303.2 | 1133.6 |
| 调优值多少 | −13.8% | −5.3% | **−42.9%** |

同一批运行里 FlyDSL 那一列是漂移探针：57.1 / 56.6 / 57.1 / 57.2 / 54.2 / 57.2，散布 1%，
所以上表可读。

M=760 那 43% 的来源看配置就清楚了：默认启发式挑 `BLOCK 128x128x64 / 8 warps`，调优选
`16x64x128 / 4 warps`。**专家数远大于行数时高的 M tile 是错的**——这正是 o11 的那一课，
triton 这边同样成立。

### 端到端，四个 seed

| | TPOT | TTFT（去掉冷启的第一个 seed） |
|---|---:|---:|
| 原生 vLLM，默认 | 19.83 ms | 343.9 ms |
| 原生 vLLM，调优后 | 19.84 ms | **328.6 ms** |
| FlyDSL（两轮） | 19.88 / 20.00 ms | 331.0 / 333.5 ms |

TPOT 四个臂全落在 19.7–20.0，分不开；调优吃到的是 TTFT −4.4%，因为 prefill 正是内核上
差 43% 的那个尺寸。**内核上 3–8% 的差距进不了 TPOT**：MoE 只占 decode 步 11%，每层 5 µs
摊进去就没了。

### 一个把十五小时冲掉的坑

第一张表按 `E=513 / topk=11` 调——那是 FlyDSL adapter 把共享专家折进去之后的视角。服务起来
日志说：

```
Using default MoE config. Config file not found at .../E=512,N=80,device_name=AMD_Radeon_Pro_W7900D.json
```

`VLLM_ROCM_USE_RDNA3_MOE=0` 时共享专家**没有**被折进 fused MoE，而是单独走一个 dense MLP，
所以运行时查的是 `E=512 / topk=10`。查表的键来自 `w2.shape`，不是模型 config 里的
`num_experts`。跨实现对照之前，先让服务把它要找的文件名打出来。

---

## 12. o14：`inter_dim=80` 那一维

```bash
# 同一进程里比（要在能 import vllm 的镜像里跑）
VLLM_TUNED_CONFIG_FOLDER=/tune python3 scripts/bench_qwen38_moe_decode.py --tokens 1,16,760 --vllm
```

2026-09-13，gfx1100（W7900D），`E=513 / topk=11 / K=2560 / N=80`，图重放，三轮取最小。
`_bench_graph` 一张图捕 10 次调用，所以 `graph.replay()` 的主机开销摊掉了——§11 的 M=1
一列没有这个修正，两张表的 M=1 不要交叉比。

### 为什么只剩 BLOCK_K 一个旋钮

`BLOCK_N = 16 · reg_n · waves_n` 必须整除 `n_out`。`gate_up` 的 `n_out` 是 `inter_dim=80`，
`80 % 32 != 0`，所以 BLOCK_N 只能取 16；取 16 就把 `reg_n` 和 `waves_n` 一起钉成 1，
`tile_m=16` 再把 `waves_m` 钉成 1。结果是**每个 workgroup 一个 wave32**，`_SHAPE_TILES` /
`_BUCKET_TILES` / `_TILES` 三张表全不命中，落到 `_narrowings` 兜底。

### gemm1 单独计时，只扫 BLOCK_K

| BLOCK_K | block | M=1 | M=16 | M=760 |
|---|---|---:|---:|---:|
| 16 | 16x16x16 | 27.5 | 203.0 | 1561.7 |
| 32 | 16x16x32 | 23.1 | 189.7 | 817.4 |
| 64 | 16x16x64 | 19.0 | 149.1 | 626.4 |
| **128** | **16x16x128** | **15.3** | **132.5** | **574.3** |
| 256 | 16x16x256 | 17.2 | — | 593.7 |

`_narrowings` 的 `reg_k` 只枚举到 4，所以 128 这一档从来没被试过。并行度全程没变（始终
一个 wave），省的是 k-loop 每步那个 `s_barrier`——单 wave 的 workgroup 没有第二个 wave
可以拿来盖住它。256 之后掉头。

### gemm2 单独计时，扫 BLOCK_N

`k_dim=80` 同理把 BLOCK_K 钉在 16，BLOCK_N 自由：

| block | M=1 | M=760 |
|---|---:|---:|
| 16x32x16 | 8.0 | 409.3 |
| **16x64x16** | **7.4** | 413.7 |
| 16x128x16 | 8.3 | 408.7 |

M=760 上三档全在 409 µs 上下——已经贴在带宽墙上，tile 换不出东西。只有 M=1 值 0.9 µs。

### 整层，同一进程 / 同一输入

| | M=1 | M=16 | M=760 |
|---|---:|---:|---:|
| FlyDSL o13 | 42.0 | 292.7 | 1123.0 |
| vLLM 默认配置 | 44.4 | 302.4 | 1883.6 |
| vLLM 调优后 | 37.7 | 284.0 | 1085.5 |
| **FlyDSL o14** | **34.4** | **261.7** | **1073.1** |

分段（图重放，µs）：

| | routing | gemm1 | gemm2 | reduce | 整层 |
|---|---:|---:|---:|---:|---:|
| M=1，o13 | 8.2 | 20.5 | 9.3 | 4.8 | 42.0 |
| M=1，o14 | 8.2 | 16.0 | 8.5 | 4.7 | 34.4 |
| M=760，o13 | 21.8 | 631.1 | 408.6 | 17.6 | 1123.0 |
| M=760，o14 | 21.8 | 576.2 | 413.7 | 17.8 | 1073.1 |

### 两个旁证

**vLLM 的调优器独立选中了同一个形状。** M=1 和 M=16 上它给出 `BLOCK M16xN16xK128 /
1 warp`——和上面扫出来的 `16x16x128` 完全一致。两条独立的搜索收敛到一处。

**M=760 只领先 1.2%，因为两边都在带宽墙上。** FlyDSL 那一档读 843.8 MiB / 1073 µs
≈ 784 GB/s，是 W7900D 864 GB/s 的 91%。要再拉开只能减流量：513 个专家权重合计 601 MiB，
远超 96 MB Infinity Cache，720 个 tile 里同专家的重复读基本都 miss。
