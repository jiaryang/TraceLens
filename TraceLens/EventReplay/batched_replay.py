# run_repro.py
import os
import json
import argparse
from typing import Any, List, Dict
from pathlib import Path
from datetime import datetime

import torch
import pandas as pd
from torch.profiler import (
    profile,
    record_function,
    ProfilerActivity,
    tensorboard_trace_handler,
)

# You should provide these in utils.py
from utils import TensorCfg, build_tensor, benchmark_func

# ---------------------------------------------------------------------
# TorchInductor config (must be set BEFORE any torch.compile is triggered)
# ---------------------------------------------------------------------
torch._inductor.config.triton.unique_kernel_names = True   # stable Triton kernel names for profiling
torch._inductor.config.coordinate_descent_tuning = True    # broader param search for autotune
torch._inductor.config.freezing = True                     # inference-oriented opts (use under no_grad)
torch._inductor.config.max_autotune = True                 # enlarge autotune search space


# -------------------------------
# Helpers
# -------------------------------
def deep_clone_inputs(pos_args, kwargs):
    """Clone tensor-like inputs so in-place ops don't affect subsequent runs."""
    def _clone(x):
        if isinstance(x, torch.Tensor):
            return x.clone()
        elif isinstance(x, tuple):
            return tuple(_clone(v) for v in x)
        elif isinstance(x, list):
            return [_clone(v) for v in x]
        elif isinstance(x, dict):
            return {k: _clone(v) for k, v in x.items()}
        else:
            return x
    return _clone(pos_args), _clone(kwargs)


def make_thunk(func, pos_args, kwargs):
    """Wrap an op call into a zero-arg callable for benchmarking/compiling."""
    def _call():
        return func(*pos_args, **kwargs)
    return _call


def profile_once(tag: str, fn, steps: int = 40, warmup: int = 5, logdir: str = "prof_logs"):
    """
    Record a short torch.profiler trace.
    - tag:    name to show in trace (e.g., 'eager/aten::mm' or 'compiled/aten::mm')
    - fn:     zero-argument callable to run
    - steps:  number of active steps to record
    - warmup: number of warmup steps (not recorded)
    - logdir: output directory for TensorBoard traces
    """
    os.makedirs(logdir, exist_ok=True)
    run_dir = os.path.join(logdir, tag.replace("/", "_"))
    acts = [ProfilerActivity.CPU, ProfilerActivity.CUDA]  # ROCm also uses CUDA label

    with profile(
        activities=acts,
        schedule=torch.profiler.schedule(wait=0, warmup=warmup, active=steps, repeat=1),
        on_trace_ready=tensorboard_trace_handler(run_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,        # can be disabled if too heavy
        with_modules=True,
    ) as prof:
        total = warmup + steps
        for _ in range(total):
            with record_function(tag):
                _ = fn()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            prof.step()

    trace_json = os.path.join(run_dir, f"{tag}.json")
    try:
        prof.export_chrome_trace(trace_json)  # open via chrome://tracing or Perfetto
    except Exception:
        trace_json = None

    print(f"[Profiler] TensorBoard run dir: {run_dir}")
    if trace_json:
        print(f"[Profiler] Chrome trace: {trace_json}")
    return run_dir, trace_json


def resolve_dynamo_callable(op_name: str):
    """
    Return a Dynamo-traceable callable (OpOverload) for 'namespace::op[.overload]'.
    Examples:
      'aten::mm'               -> torch.ops.aten.mm.default
      'aten::add.Tensor'       -> torch.ops.aten.add.Tensor
      'aten::copy_'            -> torch.ops.aten.copy_
      'aten::_to_copy'         -> torch.ops.aten._to_copy.default
    """
    if "::" not in op_name:
        raise ValueError(f"Unsupported op_name format: {op_name}")

    ns, rest = op_name.split("::", 1)
    if "." in rest:
        base, overload = rest.split(".", 1)
    else:
        base, overload = rest, "default"

    packet = getattr(getattr(torch.ops, ns), base)  # e.g. torch.ops.aten.mm / add / copy_
    target = getattr(packet, overload) if hasattr(packet, overload) else getattr(packet, "default")
    return target


def _get_args_kwargs_from_ir(event_replay_IR: Dict[str, Any], device: str = "cuda"):
    """
    Reconstruct positional and keyword arguments from the replay IR.
    Assumes Tensor arguments are serialized as TensorCfg dictionaries.
    """
    pos_args: List[Any] = []
    for arg in event_replay_IR["list_pos_args"]:
        val = arg["value"]
        if arg["arg_type"].startswith("Tensor") and val:
            cfg = TensorCfg(**val)
            pos_args.append(build_tensor(cfg, device=device))
        else:
            pos_args.append(val)

    kwargs: Dict[str, Any] = {}
    for arg in event_replay_IR["list_kwargs"]:
        val = arg["value"]
        key = arg["arg_name"]
        if arg["arg_type"].startswith("Tensor") and val:
            cfg = TensorCfg(**val)
            kwargs[key] = build_tensor(cfg, device=device)
        else:
            kwargs[key] = val
    return pos_args, kwargs


def _first_tensor_shape(obj):
    """Return a quick shape hint by scanning for the first Tensor."""
    if isinstance(obj, torch.Tensor):
        return tuple(obj.shape)
    if isinstance(obj, (list, tuple)):
        for v in obj:
            s = _first_tensor_shape(v)
            if s:
                return s
    if isinstance(obj, dict):
        for v in obj.values():
            s = _first_tensor_shape(v)
            if s:
                return s
    return None


def _extract_tensor_meta_from_ir_arg(ir_arg):
    """
    Given one entry from replay_ir['list_pos_args'] or ['list_kwargs'],
    return (shape:list[int] or None, strides:list[int] or None)
    """
    if not isinstance(ir_arg, dict):
        return None, None
    if ir_arg.get("arg_type", "").startswith("Tensor"):
        val = ir_arg.get("value", {})
        shape = val.get("shape")
        strides = val.get("strides")
        return shape, strides
    return None, None


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Replay PyTorch ops from a repro JSON and benchmark eager vs compiled."
    )
    parser.add_argument("repro_file", type=str, help="Path to the JSON repro file generated by extract_repro.py")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Execution device")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose result printing")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop immediately if any op fails")
    parser.add_argument("--op-limit", type=int, default=None, help="Limit the number of operations to replay")
    parser.add_argument(
        "--op-filter", type=str, default=None, help="Only replay ops whose name contains this substring (e.g. 'aten::add')"
    )
    parser.add_argument("--profile", action="store_true", help="Record short torch.profiler traces for eager/compiled")
    parser.add_argument(
        "--compile-mode", type=str, default="default",
        choices=["default", "max-autotune", "reduce-overhead"], help="torch.compile mode"
    )
    parser.add_argument(
        "--excel", type=str, default=None,
        help="Path to the Excel file for results (default: ./op_bench_YYYYmmdd_HHMMSS.xlsx)"
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Optional: also dump a CSV with the same rows"
    )

    args = parser.parse_args()

    # Device check
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"

    print(f"Running repro from '{args.repro_file}' on device '{args.device}'")

    # Load repro data
    with open(args.repro_file, "r") as f:
        repro_data_list = json.load(f)

    rows: List[Dict[str, Any]] = []
    replayed_count = 0
    errors = 0

    for i, repro_info in enumerate(repro_data_list):
        op_name = repro_info["op_name"]
        if args.op_filter and args.op_filter not in op_name:
            continue
        if args.op_limit is not None and replayed_count >= args.op_limit:
            break

        replay_ir = repro_info["replay_ir"]
        print(f"\n[{replayed_count + 1}/{len(repro_data_list)}] Replaying: {op_name}")

        # Eager path: JIT PyCapsule (works well for direct calls)
        try:
            func, _ = torch._C._jit_get_operation(op_name)
        except Exception as e:
            print(f"  Error: Could not find op '{op_name}'. Is the PyTorch version compatible? Error: {e}")
            if args.stop_on_error:
                raise
            errors += 1
            continue

        # Compiled path: prefer a Dynamo-traceable torch.ops callable
        func_for_compile = None
        try:
            func_for_compile = resolve_dynamo_callable(op_name)
        except Exception as e:
            print(f"  Warning: torch.ops lookup failed for '{op_name}'. Will try PyCapsule with allow_in_graph. Error: {e}")

        # Reconstruct args
        try:
            if args.verbose:
                print(f"  Reconstructing arguments for '{op_name}'...")
                print(f"  Positional Args:")
                for arg in replay_ir["list_pos_args"]:
                    print(f"    {arg['arg_name']} {arg['arg_type']}: {arg['value']}")
                print(f"  Keyword Args:")
                for arg in replay_ir["list_kwargs"]:
                    print(f"    {arg['arg_name']} {arg['arg_type']}: {arg['value']}")
            pos_args, kwargs = _get_args_kwargs_from_ir(replay_ir, device=args.device)
        except Exception as e:
            print(f"  Error: Failed to reconstruct args for '{op_name}'. Error: {e}")
            if args.stop_on_error:
                raise
            errors += 1
            continue

        # Eager thunk + one dry run (not timed)
        eager_pos_args, eager_kwargs = deep_clone_inputs(pos_args, kwargs)
        eager_thunk = make_thunk(func, eager_pos_args, eager_kwargs)

        if args.profile:
            profile_once(tag=f"eager/{op_name}", fn=eager_thunk, steps=30, warmup=5, logdir="prof_logs")

        result = None
        try:
            if args.device == "cuda":
                torch.cuda.synchronize()
            result = eager_thunk()
            if args.device == "cuda":
                torch.cuda.synchronize()
        except Exception as e:
            print(f"  Error: Failed to execute '{op_name}' (eager). Error: {e}")
            if args.stop_on_error:
                raise
            errors += 1
            continue

        # Compiled thunk + warmup
        compiled_thunk = None
        try:
            comp_pos_args, comp_kwargs = deep_clone_inputs(pos_args, kwargs)

            if func_for_compile is not None:
                # Preferred path: torch.ops.* overload (Dynamo-friendly)
                compiled_thunk = torch.compile(
                    make_thunk(func_for_compile, comp_pos_args, comp_kwargs),
                    backend="inductor",
                    mode=args.compile_mode,
                )
            else:
                # Fallback: allow_in_graph wrapper around the PyCapsule
                from torch.compiler import allow_in_graph

                @allow_in_graph
                def _capsule_call(f, *a, **k):
                    return f(*a, **k)

                compiled_thunk = torch.compile(
                    lambda: _capsule_call(func, *comp_pos_args, **comp_kwargs),
                    backend="inductor",
                    mode=args.compile_mode,
                )

            if args.device == "cuda":
                torch.cuda.synchronize()
            _ = compiled_thunk()  # trigger compilation once
            if args.device == "cuda":
                torch.cuda.synchronize()

            if args.profile:
                profile_once(tag=f"compiled/{op_name}", fn=compiled_thunk, steps=30, warmup=5, logdir="prof_logs")

        except Exception as e:
            print(f"  Warning: torch.compile failed for '{op_name}'. Fallback to eager-only. Error: {e}")
            compiled_thunk = None

        # Benchmark eager
        if args.device == "cuda":
            torch.cuda.synchronize()
        eager_time_us = benchmark_func(eager_thunk, args.device, warmup=50, avg_steps=100)

        # Benchmark compiled
        compiled_time_us = None
        if compiled_thunk is not None:
            if args.device == "cuda":
                torch.cuda.synchronize()
            compiled_time_us = benchmark_func(compiled_thunk, args.device, warmup=20, avg_steps=100)

        # Print results
        print(f"  Eager avg time:    {eager_time_us:.2f} microseconds")
        if compiled_time_us is not None:
            print(f"  Compiled avg time: {compiled_time_us:.2f} microseconds")
            if compiled_time_us > 0:
                print(f"  Speedup (Eager/Compiled): {eager_time_us/compiled_time_us:.2f}x")
        else:
            print("  Compiled avg time: N/A (compile failed)")

        # Workload estimate (prefer compiled if available)
        mean_time_us = compiled_time_us if compiled_thunk is not None else eager_time_us
        print(f"  Average time taken: {mean_time_us:.2f} microseconds")
        if "count" in repro_info:
            count_workload = repro_info["count"]
            total_time_us = mean_time_us * count_workload
            print(f"  Count in workload: {count_workload}")
            print(f"  Est time in workload: {total_time_us:.2f} microseconds")

        if args.device == "cuda":
            torch.cuda.synchronize()

        # ----- extract tensor meta for the first two positional tensor args (self, mat2) -----
        t1_shape = t1_stride = t2_shape = t2_stride = None
        pos_args_ir = replay_ir.get("list_pos_args", [])
        if len(pos_args_ir) >= 1:
            s, st = _extract_tensor_meta_from_ir_arg(pos_args_ir[0])
            t1_shape, t1_stride = s, st
        if len(pos_args_ir) >= 2:
            s, st = _extract_tensor_meta_from_ir_arg(pos_args_ir[1])
            t2_shape, t2_stride = s, st

        max1 = max(t1_shape) if t1_shape else None
        max2 = max(t2_shape) if t2_shape else None
        stride1 = None if t1_stride is None else ",".join(str(x) for x in t1_stride)
        stride2 = None if t2_stride is None else ",".join(str(x) for x in t2_stride)

        # Optional bmm-friendly dims if detectable: [B,M,K] x [B,K,N]
        B = M = K = N = None
        if t1_shape and t2_shape and len(t1_shape) == 3 and len(t2_shape) == 3:
            B, M, K = t1_shape[0], t1_shape[1], t1_shape[2]
            if t2_shape[1] == K:
                N = t2_shape[2]

        # Accumulate a row for Excel/CSV
        rows.append({
            "op_name": op_name,
            "device": args.device,
            "eager_us": float(eager_time_us) if eager_time_us is not None else None,
            "compiled_us": float(compiled_time_us) if compiled_time_us is not None else None,
            "speedup_eager_over_compiled": (
                float(eager_time_us) / float(compiled_time_us)
                if (compiled_time_us is not None and compiled_time_us > 0) else None
            ),
            "used_us_for_est": float(mean_time_us) if mean_time_us is not None else None,
            "count_in_workload": int(repro_info.get("count", 0)) if "count" in repro_info else None,
            "est_total_us": (
                float(mean_time_us) * int(repro_info.get("count", 0))
                if ("count" in repro_info and mean_time_us is not None) else None
            ),
            "compile_ok": bool(compiled_thunk is not None),
            "unique_kernel_names": bool(getattr(torch._inductor.config.triton, "unique_kernel_names", False)),
            "coordinate_descent_tuning": bool(getattr(torch._inductor.config, "coordinate_descent_tuning", False)),
            "freezing": bool(getattr(torch._inductor.config, "freezing", False)),
            "max_autotune": bool(getattr(torch._inductor.config, "max_autotune", False)),

            # shape/stride metadata
            "t1_shape": None if t1_shape is None else str(t1_shape),
            "t2_shape": None if t2_shape is None else str(t2_shape),
            "t1_stride": stride1,
            "t2_stride": stride2,
            "max1": max1,
            "max2": max2,

            # explode per-dimension (safe guards included)
            "t1_shape_0": (t1_shape[0] if (t1_shape and len(t1_shape) > 0) else None),
            "t1_shape_1": (t1_shape[1] if (t1_shape and len(t1_shape) > 1) else None),
            "t1_shape_2": (t1_shape[2] if (t1_shape and len(t1_shape) > 2) else None),
            "t1_stride_0": (t1_stride[0] if (t1_stride and len(t1_stride) > 0) else None),
            "t1_stride_1": (t1_stride[1] if (t1_stride and len(t1_stride) > 1) else None),
            "t1_stride_2": (t1_stride[2] if (t1_stride and len(t1_stride) > 2) else None),

            "t2_shape_0": (t2_shape[0] if (t2_shape and len(t2_shape) > 0) else None),
            "t2_shape_1": (t2_shape[1] if (t2_shape and len(t2_shape) > 1) else None),
            "t2_shape_2": (t2_shape[2] if (t2_shape and len(t2_shape) > 2) else None),
            "t2_stride_0": (t2_stride[0] if (t2_stride and len(t2_stride) > 0) else None),
            "t2_stride_1": (t2_stride[1] if (t2_stride and len(t2_stride) > 1) else None),
            "t2_stride_2": (t2_stride[2] if (t2_stride and len(t2_stride) > 2) else None),

            # inferred bmm dims (if applicable)
            "B": B, "M": M, "K": K, "N": N,

            # quick preview string
            "first_tensor_shape": str(_first_tensor_shape((pos_args, kwargs))),
        })

        # Verbose result preview
        if args.verbose:
            print(f"  Successfully executed {op_name}.")
            if isinstance(result, torch.Tensor):
                print(f"  Result: Tensor(shape={tuple(result.shape)}, dtype={result.dtype}, device={result.device})")
            elif isinstance(result, (list, tuple)) and any(isinstance(r, torch.Tensor) for r in result):
                for r in result:
                    if isinstance(r, torch.Tensor):
                        print(f"  Result: Tensor(shape={tuple(r.shape)}, dtype={r.dtype}, device={r.device})")
                    else:
                        print(f"  Result: {r}")
            else:
                print(f"  Result: {result}")

        replayed_count += 1

    # Summary
    print("\n--- Replay Summary ---")
    print(f"Total operations in file: {len(repro_data_list)}")
    if args.op_filter:
        print(f"Filter applied: '{args.op_filter}'")
    print(f"Attempted replays: {replayed_count}")
    print(f"Successful replays: {replayed_count - errors}")
    print(f"Errors encountered: {errors}")
    print("----------------------")

    # ---------- Write Excel/CSV ----------
    df = pd.DataFrame(rows)

    # Optional: reorder columns so key metadata appears earlier
    col_priority = [
        "op_name", "device",
        "B","M","K","N",
        "t1_shape","t1_stride","t2_shape","t2_stride",
        "t1_shape_0","t1_shape_1","t1_shape_2",
        "t1_stride_0","t1_stride_1","t1_stride_2",
        "t2_shape_0","t2_shape_1","t2_shape_2",
        "t2_stride_0","t2_stride_1","t2_stride_2",
        "max1","max2",
        "eager_us","compiled_us","speedup_eager_over_compiled",
        "used_us_for_est","count_in_workload","est_total_us",
        "compile_ok","unique_kernel_names","coordinate_descent_tuning","freezing","max_autotune",
        "first_tensor_shape",
    ]
    df = df[[c for c in col_priority if c in df.columns] + [c for c in df.columns if c not in col_priority]]

    if not df.empty and "est_total_us" in df.columns:
        df = df.sort_values(by=["est_total_us", "op_name"], ascending=[False, True], na_position="last")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_path = Path(args.excel or f"op_bench_{ts}.xlsx")
    csv_path = Path(args.csv) if args.csv else None

    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        sheet = "summary"
        df.to_excel(writer, index=False, sheet_name=sheet)
        ws = writer.sheets[sheet]
        wb = writer.book

        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, max(len(df), 1), max(len(df.columns) - 1, 0))

        # reasonable column widths
        for col_idx, col_name in enumerate(df.columns):
            try:
                maxlen = int(df[col_name].astype(str).map(len).max())
            except ValueError:
                maxlen = 12
            width = max(12, min(40, maxlen))
            ws.set_column(col_idx, col_idx, width)

        # format speedup with 2 decimals + color scale
        if "speedup_eager_over_compiled" in df.columns:
            fmt = wb.add_format({"num_format": "0.00"})
            c = df.columns.get_loc("speedup_eager_over_compiled")
            ws.set_column(c, c, 14, fmt)
            ws.conditional_format(1, c, len(df) + 1, c, {"type": "3_color_scale"})

    print(f"[Excel] Saved results to: {excel_path}")

    if csv_path:
        df.to_csv(csv_path, index=False)
        print(f"[CSV]   Saved results to: {csv_path}")

    raise SystemExit(1 if errors > 0 else 0)
