import os
import json
import time
import argparse
import torch
import triton
import random
import pandas as pd

from streamk_kernel import streamk_gemm

torch.manual_seed(123)
random.seed(123)

# ---------------------------------------------------------------------
# TorchInductor config (must be set BEFORE any torch.compile is triggered)
# ---------------------------------------------------------------------
torch._inductor.config.triton.unique_kernel_names = True   # stable Triton kernel names for profiling
torch._inductor.config.coordinate_descent_tuning = True    # broader param search for autotune
torch._inductor.config.freezing = True                     # inference-oriented opts (use under no_grad)
torch._inductor.config.max_autotune = True                 # enlarge autotune search space
torch._inductor.config.triton.unique_user_kernel_names = True
torch._dynamo.config.recompile_limit = 256

total_sm = 304
print(f"total SMs: {total_sm}")

# --------------------------------------------------------
# Argument parsing
# --------------------------------------------------------
parser = argparse.ArgumentParser(description="Run GEMM cases (Eager, torch.compile, Stream-K) and save to Excel")
parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
parser.add_argument("--output", type=str, default="gemm_results.xlsx", help="Output Excel filename")
parser.add_argument("--repeat", type=int, default=10, help="Number of repetitions for averaging (default: 10)")
parser.add_argument("--warmup", type=int, default=2, help="Number of warmup iterations (default: 2)")
args = parser.parse_args()

# --------------------------------------------------------
# Helper
# --------------------------------------------------------
def perf_tflops(m, n, k, ms):
    return 2 * m * n * k * 1e-12 / (ms * 1e-3)

class matmul(torch.autograd.Function):
    @staticmethod
    def _call(a, b, c, bias, P, locks,
              total_programs_streamk, BLK_M, BLK_N, BLK_K,
              gsize_m, two_tiles, num_stages, num_warps,
              waves_per_eu, mfmaInstrSize, kpack):
        M, K = a.shape
        _, N = b.shape
        even_k = K % BLK_K == 0
        grids = total_programs_streamk
        stride_bias = bias.stride(0) if bias is not None else 0
        num_xcds = 8

        streamk_gemm[(grids,)](
            a, b, c, bias, P, locks,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            stride_bias,
            BLOCK_SIZE_M=BLK_M, BLOCK_SIZE_N=BLK_N, BLOCK_SIZE_K=BLK_K,
            GROUP_SIZE_M=gsize_m, NUM_SMS=total_programs_streamk,
            STREAMK_TILES=0, NUM_XCDS=num_xcds,
            BIAS=False, EVEN_K=even_k,
            num_stages=num_stages, num_warps=num_warps,
            waves_per_eu=waves_per_eu,
            matrix_instr_nonkdim=mfmaInstrSize, kpack=kpack
        )
        return c

# --------------------------------------------------------
# Main Loop
# --------------------------------------------------------
with open(args.config, "r") as f:
    configs = json.load(f)

results = []

for idx, case in enumerate(configs, start=1):
    op_name = case.get("op_name", "")
    replay_ir = case.get("replay_ir", {})
    args_list = replay_ir.get("list_pos_args", [])

    a_shape = args_list[0]["value"]["shape"]
    b_shape = args_list[1]["value"]["shape"]
    m, k = a_shape
    k2, n = b_shape
    assert k == k2, f"Incompatible shapes: {a_shape}, {b_shape}"

    print(f"\n[Case {idx}] {op_name}: M={m}, N={n}, K={k}")

    # Allocate tensors
    A = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    B = torch.randn((k, n), device="cuda", dtype=torch.bfloat16)
    C = torch.zeros((m, n), device="cuda", dtype=torch.bfloat16)
    bias = torch.zeros((m,), device="cuda", dtype=torch.bfloat16)
    P = torch.empty(1, device="cuda")
    locks = torch.empty(1, device="cuda")

    # ================================
    # Eager (torch.matmul)
    # ================================
    for _ in range(args.warmup):
        _ = torch.matmul(A, B, out=C)
    torch_times = []
    for i in range(args.repeat):
        torch.cuda.synchronize()
        t0 = time.time()
        torch.matmul(A, B, out=C)
        torch.cuda.synchronize()
        t1 = time.time()
        torch_times.append((t1 - t0) * 1e3)
    torch_valid = torch_times[2:] if len(torch_times) > 2 else torch_times
    torch_avg = sum(torch_valid) / len(torch_valid)
    torch_tflops = perf_tflops(m, n, k, torch_avg)
    print(f"  torch.matmul avg: {torch_avg:.3f} ms | {torch_tflops:.3f} TFLOPs")

    # ================================
    # torch.compile(triton)
    # ================================
    compiled_matmul = torch.compile(
                        torch.matmul,
                        backend="inductor",
                        dynamic=False,
                    )
    for _ in range(args.warmup):
        _ = compiled_matmul(A, B)
    compiled_times = []
    for i in range(args.repeat):
        torch.cuda.synchronize()
        t0 = time.time()
        compiled_matmul(A, B)
        torch.cuda.synchronize()
        t1 = time.time()
        compiled_times.append((t1 - t0) * 1e3)
    compiled_valid = compiled_times[2:] if len(compiled_times) > 2 else compiled_times
    compiled_avg = sum(compiled_valid) / len(compiled_valid)
    compiled_tflops = perf_tflops(m, n, k, compiled_avg)
    print(f"  torch.compile avg: {compiled_avg:.3f} ms | {compiled_tflops:.3f} TFLOPs")

    # ================================
    # Stream-K
    # ================================
    for _ in range(args.warmup):
        _ = matmul._call(A, B, C, bias, P, locks,
                         total_programs_streamk=total_sm,
                         BLK_M=256, BLK_N=256, BLK_K=64,
                         gsize_m=8, two_tiles=True,
                         num_stages=2, num_warps=8,
                         waves_per_eu=0, mfmaInstrSize=16, kpack=2)
    sk_times = []
    for i in range(args.repeat):
        torch.cuda.synchronize()
        t0 = time.time()
        matmul._call(A, B, C, bias, P, locks,
                     total_programs_streamk=total_sm,
                     BLK_M=256, BLK_N=256, BLK_K=64,
                     gsize_m=8, two_tiles=True,
                     num_stages=2, num_warps=8,
                     waves_per_eu=0, mfmaInstrSize=16, kpack=2)
        torch.cuda.synchronize()
        t1 = time.time()
        sk_times.append((t1 - t0) * 1e3)
    sk_valid = sk_times[2:] if len(sk_times) > 2 else sk_times
    sk_avg = sum(sk_valid) / len(sk_valid)
    sk_tflops = perf_tflops(m, n, k, sk_avg)
    print(f"  Stream-K avg: {sk_avg:.3f} ms | {sk_tflops:.3f} TFLOPs")

    # ================================
    # Speedup
    # ================================
    speedup_compiled = torch_avg / compiled_avg if compiled_avg > 0 else float('nan')
    speedup_streamk = torch_avg / sk_avg if sk_avg > 0 else float('nan')
    print(f"  ⚡ Speedup (Eager/Compiled): {speedup_compiled:.2f}x")
    print(f"  ⚡ Speedup (Eager/Stream-K): {speedup_streamk:.2f}x")

    results.append({
        "case_id": idx,
        "op_name": op_name,
        "m": m, "n": n, "k": k,
        "torch_time_ms": torch_avg,
        "torch_tflops": torch_tflops,
        "compiled_time_ms": compiled_avg,
        "compiled_tflops": compiled_tflops,
        "streamk_time_ms": sk_avg,
        "streamk_tflops": sk_tflops,
        "speedup_compiled_vs_torch": speedup_compiled,
        "speedup_streamk_vs_torch": speedup_streamk,
        "repeat": args.repeat,
        "warmup": args.warmup
    })

# --------------------------------------------------------
# Save to Excel
# --------------------------------------------------------
df = pd.DataFrame(results)
df.to_excel(args.output, index=False)
print(f"\n✅ All {len(results)} cases done. Results saved to {args.output}")
