"""Tune vLLM's Triton ``fused_moe`` for one shape, without Ray.

Why this exists rather than ``benchmarks/kernels/benchmark_moe.py --tune``:

* Ray is not in the serving image, and there is no route to PyPI from here.
* Ray or not, that script's fan-out is one batch size per GPU, and a batch size
  is not divisible. At ``E=513`` the ROCm pruner leaves 25760 configs for
  ``M=1024`` and about 0.9 s goes into each, so that one size is nine hours on
  the GPU that draws it while the others sit idle.

So the fan-out here is over the *search space*: every GPU takes a stride of the
configs for the same batch size, reports its best time, and the driver keeps the
minimum. Batch sizes then run one after another, each at 8x. Everything else --
the search space, the pruning, the timing, the file format -- is
``benchmark_moe``'s, called directly.

``import benchmark_moe`` pulls in Ray at module scope, so a stub goes into
``sys.modules`` first; nothing on the tuning path actually touches it.

The shape is given explicitly rather than read from a model config, because the
number that matters is the one the *runtime* looks up -- ``w2.shape`` -- and a
model that folds its shared expert into the fused kernel has one more expert
there than its config admits (and one more slot in topk).

    python3 scripts/tune_vllm_moe.py \
        --num-experts 513 --topk 11 --hidden-size 2560 \
        --shard-intermediate-size 160 \
        --batch-size 1 2 4 8 16 512 1024 --save-dir /out

``--shard-intermediate-size`` is ``2 * moe_intermediate_size / tp_size``, i.e.
the gate and up halves together; the config filename carries half of it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

VLLM_ROOT = Path(os.environ.get("VLLM_ROOT", "/workspace/vllm-qwen38"))
KERNELS = VLLM_ROOT / "benchmarks" / "kernels"

DEFAULT_BATCH_SIZES = [
    1, 2, 4, 8, 16, 24, 32, 48, 64, 96,
    128, 256, 512, 1024, 1536, 2048, 3072, 4096,
]


def _stub_ray() -> None:
    """Enough of Ray for ``benchmark_moe`` to import."""
    if "ray" in sys.modules:
        return
    from tqdm import tqdm  # noqa: PLC0415

    ray = types.ModuleType("ray")
    ray.__path__ = []  # a package, so ``ray.experimental`` can be imported
    ray.remote = lambda *a, **kw: (a[0] if a and callable(a[0]) else (lambda cls: cls))
    ray.get_gpu_ids = lambda: [0]
    ray.init = lambda *a, **kw: None
    ray.get = lambda x: x
    ray.available_resources = lambda: {"GPU": 1}

    experimental = types.ModuleType("ray.experimental")
    experimental.__path__ = []
    tqdm_ray = types.ModuleType("ray.experimental.tqdm_ray")
    tqdm_ray.tqdm = tqdm
    experimental.tqdm_ray = tqdm_ray
    ray.experimental = experimental

    sys.modules["ray"] = ray
    sys.modules["ray.experimental"] = experimental
    sys.modules["ray.experimental.tqdm_ray"] = tqdm_ray


def _load_benchmark_moe():
    _stub_ray()
    sys.path.insert(0, str(KERNELS))
    import benchmark_moe  # noqa: PLC0415

    return benchmark_moe


def run_worker(args: argparse.Namespace) -> None:
    """Time this worker's stride of the search space for one batch size."""
    import torch  # noqa: PLC0415
    import triton  # noqa: PLC0415

    bm = _load_benchmark_moe()
    torch.set_default_device("cuda")

    dtype = getattr(torch, args.activation_dtype)
    m = args.batch_size
    space = bm.prune_rocm_search_space(
        m, args.shard_intermediate_size, args.hidden_size,
        bm.get_configs_compute_bound(True, None), True, args.topk,
    )
    if args.max_block:
        # A 256x256 tile can sit in LLVM for over a minute, and at E=513 the
        # rows per expert are far too few for it to ever win: every batch size
        # up to 16 picked a 16-wide M tile. Capping the tile is how a large-M
        # sweep finishes at all -- the tail of the uncapped space cost longer
        # than the other 96% of it put together.
        space = [c for c in space if max(
            c["BLOCK_SIZE_M"], c["BLOCK_SIZE_N"], c["BLOCK_SIZE_K"]
        ) <= args.max_block]
    mine = space[args.shard :: args.num_shards]

    best_config, best_time = None, float("inf")
    began = time.time()
    for idx, config in enumerate(mine):
        try:
            kernel_time = bm.benchmark_config(
                config, m, args.num_experts, args.shard_intermediate_size,
                args.hidden_size, args.topk, dtype,
                False, False, False, num_iters=20,
                block_quant_shape=None, use_deep_gemm=False,
            )
        except triton.runtime.autotuner.OutOfResources:
            continue
        except Exception as exc:  # a config the backend refuses to build
            if args.verbose:
                print(f"  skip {config}: {exc}", flush=True)
            continue
        if kernel_time < best_time:
            best_time, best_config = kernel_time, config
        if idx and idx % 200 == 0:
            rate = (time.time() - began) / idx
            left = (len(mine) - idx) * rate
            print(f"  {idx}/{len(mine)} best={best_time:.1f}us eta={left / 60:.0f}min",
                  flush=True)
        if bm.TRITON_CACHE_CLEAR_INTERVAL > 0 and idx and idx % bm.TRITON_CACHE_CLEAR_INTERVAL == 0:
            bm.clear_triton_cache()
    bm.clear_triton_cache()

    Path(args.out).write_text(json.dumps({
        "batch_size": m, "time_us": best_time, "config": best_config,
        "tried": len(mine), "seconds": time.time() - began,
    }))
    print(f"M={m} shard {args.shard}: best {best_time:.1f} us over {len(mine)} configs",
          flush=True)


def _write_table(bm, args, configs: dict[int, dict], timings: dict[int, float]) -> None:
    import torch  # noqa: PLC0415

    bm.save_configs(
        {m: configs[m] for m in sorted(configs)},
        args.num_experts, args.shard_intermediate_size, args.hidden_size,
        args.topk, getattr(torch, args.activation_dtype),
        False, False, False, None, args.save_dir,
    )
    (Path(args.save_dir) / "tuned_times.json").write_text(
        json.dumps({str(m): timings[m] for m in sorted(timings)}, indent=2)
    )


def merge_partials(args: argparse.Namespace) -> None:
    """Build the table from whatever shards finished.

    A run stopped part way through still has every batch size it completed
    sitting in ``_partial``, and a table missing a size is not wrong -- the
    runtime picks the nearest key it has. This is how you cash in a run you
    decided not to wait out.
    """
    bm = _load_benchmark_moe()
    tmp = Path(args.save_dir) / "_partial"

    by_m: dict[int, list[dict]] = {}
    for path in tmp.glob("m*_shard*.json"):
        rec = json.loads(path.read_text())
        by_m.setdefault(rec["batch_size"], []).append(rec)

    configs, timings = {}, {}
    for m, recs in sorted(by_m.items()):
        if len(recs) != len(args.gpus.split(",")):
            print(f"M={m}: only {len(recs)} shards finished, skipping")
            continue
        best = min(recs, key=lambda r: r["time_us"])
        configs[m] = bm.sort_config(best["config"])
        timings[m] = best["time_us"]
        print(f"M={m}: {best['time_us']:.1f} us  {configs[m]}")
    if not configs:
        raise SystemExit(f"no complete batch size in {tmp}")
    _write_table(bm, args, configs, timings)


def run_driver(args: argparse.Namespace) -> None:
    bm = _load_benchmark_moe()

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    batch_sizes = sorted(args.batch_size or DEFAULT_BATCH_SIZES)
    tmp = Path(args.save_dir) / "_partial"
    tmp.mkdir(parents=True, exist_ok=True)

    configs: dict[int, dict] = {}
    timings: dict[int, float] = {}
    began = time.time()

    for m in batch_sizes:
        started = time.time()
        procs = []
        for shard, gpu in enumerate(gpus):
            out = tmp / f"m{m}_shard{shard}.json"
            cmd = [
                sys.executable, __file__, "--worker",
                "--out", str(out),
                "--num-experts", str(args.num_experts),
                "--topk", str(args.topk),
                "--hidden-size", str(args.hidden_size),
                "--shard-intermediate-size", str(args.shard_intermediate_size),
                "--activation-dtype", args.activation_dtype,
                "--batch-size", str(m),
                "--shard", str(shard),
                "--num-shards", str(len(gpus)),
                "--max-block", str(args.max_block),
            ]
            # Pin with ROCR alone. HIP_VISIBLE_DEVICES indexes into whatever
            # ROCR already filtered, so setting both means asking for device 3
            # of a one-device set; and vLLM rejects HIP and CUDA disagreeing.
            env = dict(os.environ)
            env.pop("HIP_VISIBLE_DEVICES", None)
            env.pop("CUDA_VISIBLE_DEVICES", None)
            env["ROCR_VISIBLE_DEVICES"] = gpu
            log = open(tmp / f"m{m}_shard{shard}.log", "w")
            procs.append((shard, out, log,
                          subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)))

        failed = []
        for shard, _out, log, proc in procs:
            if proc.wait() != 0:
                failed.append(shard)
            log.close()
        if failed:
            raise SystemExit(f"M={m}: shards {failed} failed; see {tmp}/m{m}_shard*.log")

        best = min((json.loads(o.read_text()) for _s, o, _l, _p in procs),
                   key=lambda r: r["time_us"])
        configs[m] = bm.sort_config(best["config"])
        timings[m] = best["time_us"]
        print(f"M={m}: {best['time_us']:.1f} us  {configs[m]}  "
              f"({time.time() - started:.0f}s)", flush=True)

    _write_table(bm, args, configs, timings)
    print(f"Tuning took {time.time() - began:.0f} seconds")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-experts", type=int, required=True)
    p.add_argument("--topk", type=int, required=True)
    p.add_argument("--hidden-size", type=int, required=True)
    p.add_argument("--shard-intermediate-size", type=int, required=True,
                   help="2 * moe_intermediate_size / tp_size")
    p.add_argument("--activation-dtype", default="bfloat16",
                   choices=["bfloat16", "float16"])
    p.add_argument("--batch-size", type=int, nargs="+")
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p.add_argument("--save-dir", default="./")
    p.add_argument("--max-block", type=int, default=0,
                   help="skip configs with any BLOCK dimension above this")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--merge-only", action="store_true",
                   help="write the table from the shards already in _partial")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--out", help=argparse.SUPPRESS)
    p.add_argument("--shard", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--num-shards", type=int, default=1, help=argparse.SUPPRESS)
    return p


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    if parsed.worker:
        (parsed.batch_size,) = parsed.batch_size
        run_worker(parsed)
    elif parsed.merge_only:
        merge_partials(parsed)
    else:
        run_driver(parsed)
