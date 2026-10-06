#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Build exactly one tile and report its register usage from the ISA.

This is the tool that settled o3. It builds a single kernel so the dump
directory is unambiguous, then greps the assembly for the counters that matter.

The dump only happens if the environment asks for it, so the script must be run
like this (one config per process):

    cd <repo root>
    for cfg in 4,4,2,2,2 4,1,2,2,2; do
      rm -rf /tmp/isa_$cfg
      FLYDSL_RUNTIME_ENABLE_CACHE=0 FLYDSL_DUMP_IR=1 FLYDSL_DEBUG_DUMP_ASM=1 \
      FLYDSL_DUMP_DIR=/tmp/isa_$cfg \
        python docs/rdna3_moe_optimization/repro/o3_isa_dump.py gate_up $cfg
    done

FLYDSL_RUNTIME_ENABLE_CACHE=0 matters: with the cache on you may be handed a
binary built before your change and never notice.

What to read:
    vgpr_count              256 is the gfx11 cap
    vgpr_spill_count        anything above 0 means the tile does not fit
    group_segment_fixed_size  LDS bytes per workgroup
"""

from __future__ import annotations

import os
import re
import sys

import torch

from common import REPO, require_gfx11
from kernels.moe.rdna3_moe.grouped_gemm import create_grouped_gemm_module
from kernels.moe.rdna3_moe.routing_kernel import build_routing_device

MODEL_DIM, INTER_DIM, EXPERTS, TOPK, TOKENS = 2048, 768, 8, 2, 256
_WANTED = ("vgpr_count", "vgpr_spill_count", "sgpr_spill_count", "group_segment_fixed_size")


def run_once(launch, stage, k_dim, n_out, tile_m):
    """Compilation is lazy, so the kernel has to actually run before an ISA exists."""
    dev = "cuda"
    rows = TOKENS * TOPK
    a_rows = rows if stage == "down" else TOKENS
    a = torch.zeros(a_rows, k_dim, dtype=torch.bfloat16, device=dev)
    w = torch.zeros(EXPERTS, (2 if stage == "gate_up" else 1) * n_out, k_dim, dtype=torch.bfloat16, device=dev)
    ids = torch.randint(0, EXPERTS, (TOKENS, TOPK), dtype=torch.int32, device=dev)
    wts = torch.rand(TOKENS, TOPK, dtype=torch.float32, device=dev)
    r = build_routing_device(ids, experts=EXPERTS, tile_m=tile_m, exact=True, reuse=False)
    c = torch.zeros(TOKENS, TOPK, n_out, dtype=torch.bfloat16, device=dev)
    empty = torch.empty(0, dtype=torch.float32, device=dev)
    launch(c, a, w, r.sorted_ids, r.expert_ids, wts if stage == "down" else empty,
           TOKENS, r.num_blocks, torch.cuda.current_stream())
    torch.cuda.synchronize()


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    stage, cfg_s = argv[1], argv[2]
    reg_m, reg_n, reg_k, waves_m, waves_n = (int(v) for v in cfg_s.split(","))
    require_gfx11()

    k_dim, n_out = (MODEL_DIM, INTER_DIM) if stage in ("linear", "gate_up") else (INTER_DIM, MODEL_DIM)
    launch, bm, bn, bk = create_grouped_gemm_module(
        k_dim=k_dim, n_out=n_out, experts=EXPERTS, stage=stage, topk=TOPK,
        doweight=(stage == "down"),
        reg_m=reg_m, reg_n=reg_n, reg_k=reg_k, waves_m=waves_m, waves_n=waves_n,
    )
    print(f"stage={stage} cfg={cfg_s}  block={bm}x{bn}x{bk}  threads={launch.threads} accs={launch.acc_vectors}")
    print(f"  accumulator VGPRs alone: {8 * launch.acc_vectors} of the 256 a gfx11 thread gets")
    run_once(launch, stage, k_dim, n_out, bm)

    dump_dir = os.environ.get("FLYDSL_DUMP_DIR")
    if not dump_dir or os.environ.get("FLYDSL_DEBUG_DUMP_ASM") != "1":
        print()
        print("  No ISA dumped. Re-run with the environment shown in this file's docstring.")
        return 0

    found = False
    for root, _dirs, files in os.walk(dump_dir):
        for name in sorted(files):
            if not name.endswith("_isa.s"):
                continue
            path = os.path.join(root, name)
            hits = []
            with open(path, "r", errors="replace") as fh:
                for line in fh:
                    for key in _WANTED:
                        if re.search(rf"\b{key}\b", line):
                            hits.append(line.strip())
            if hits:
                found = True
                print(f"\n  {os.path.relpath(path, REPO if path.startswith(REPO) else dump_dir)}")
                for h in dict.fromkeys(hits):
                    print(f"    {h}")
    if not found:
        print(f"\n  nothing matched under {dump_dir}; list it to see what was written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
