# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jason Prater (JP Astrophotography)
"""Engine command line.

  python -m gpustacker.engine run JOB.json     JSON-lines events on stdout. The job stops cleanly when
        [--cancel-file F] [--parent-pid P]     file F appears or process P exits (the UI went away).
  python -m gpustacker.engine probe            GPU info as one JSON line
  python -m gpustacker.engine stack SEQ [...]  stack from the command line with readable progress

Cancellation deliberately avoids reading stdin: on Windows a thread blocked in a synchronous
read of the stdin pipe stalls every DLL load in the process (CRT start-up calls GetFileType on
the standard handles), which deadlocks imports such as numpy's.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback


def _json_out():
    lock = threading.Lock()

    def emit(ev: dict) -> None:
        line = json.dumps(ev, ensure_ascii=True, separators=(",", ":"))
        with lock:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
    return emit


def _process_handle(pid: int):
    """Handle that becomes signalled when process `pid` exits (Windows), or the pid itself."""
    if os.name != "nt":
        return pid
    import ctypes
    synchronize = 0x00100000
    h = ctypes.windll.kernel32.OpenProcess(synchronize, False, int(pid))
    return h or None


def _alive(handle) -> bool:
    if handle is None:
        return False
    if os.name != "nt":
        try:
            os.kill(handle, 0)
            return True
        except OSError:
            return False
    import ctypes
    return ctypes.windll.kernel32.WaitForSingleObject(handle, 0) == 0x102     # WAIT_TIMEOUT


def _watch(cancel: threading.Event, cancel_file: str | None, parent_pid: int | None) -> None:
    handle = _process_handle(parent_pid) if parent_pid else None
    while not cancel.is_set():
        if cancel_file and os.path.exists(cancel_file):
            break
        if parent_pid and not _alive(handle):
            break
        time.sleep(0.25)
    cancel.set()


def _run(job_cfg: dict, emit, cancel) -> int:
    from .job import Job, JobConfig, JobError
    from .stack import Cancelled
    try:
        Job(JobConfig.from_dict(job_cfg), emit, cancel).run()
        return 0
    except Cancelled:
        emit({"ev": "cancelled"})
        return 3
    except (JobError, ValueError) as e:
        emit({"ev": "error", "msg": str(e), "user": True})
        return 2
    except Exception as e:                           # noqa: BLE001 - report everything to the UI
        msg = str(e)
        if "out of memory" in msg.lower() or type(e).__name__ == "OutOfMemoryError":
            msg = ("The GPU ran out of memory. Lower the GPU memory limit in Advanced settings or close "
                   "other GPU-heavy programs. (" + msg + ")")
        emit({"ev": "error", "msg": msg, "trace": traceback.format_exc(), "user": False})
        return 1


def _human(ev: dict) -> None:
    kind = ev.get("ev")
    if kind == "log":
        print(("WARNING: " if ev.get("level") == "warn" else "") + ev["msg"], flush=True)
    elif kind == "phase":
        print(f"== {ev['label']}", flush=True)
    elif kind == "progress":
        eta = ev.get("eta")
        eta_s = f", ETA {int(eta) // 60}:{int(eta) % 60:02d}" if eta else ""
        rate = ev.get("rate")
        rate_s = f", {rate / 1e6:.0f} MB/s" if rate else ""
        pct = 100.0 * ev["done"] / ev["total"] if ev.get("total") else 0
        print(f"\r  {ev['phase']}: {pct:5.1f}%{rate_s}{eta_s}     ", end="", flush=True)
    elif kind == "frame_bad":
        print(f"\n  skipping frame #{ev['number']}{' (' + ev['name'] + ')' if ev.get('name') else ''}: "
              f"{ev['reason']}", flush=True)
    elif kind == "done":
        if ev.get("analysis"):
            print(f"\nanalysis done: {ev['frames']} frames registered on {ev['reference']} "
                  f"in {ev['seconds_total']:.0f} s", flush=True)
        else:
            print(f"\nwrote {ev['output']} ({ev['width']} x {ev['height']}, {ev['frames']} frames, "
                  f"{ev['livetime'] / 3600:.2f} h) in {ev['seconds_total']:.0f} s", flush=True)
    elif kind == "registered":
        print(f"\n  registered {ev['registered']} frames (failed {ev['failed']}), median residual "
              f"{ev['rms_median']:.2f} px, rotation {ev['rotation'][0]:.2f}..{ev['rotation'][1]:.2f} deg", flush=True)
    elif kind == "error":
        print(f"\nERROR: {ev['msg']}", flush=True)
        if ev.get("trace"):
            print(ev["trace"], flush=True)
    elif kind == "cancelled":
        print("\ncancelled", flush=True)


def main(argv=None) -> int:
    if os.environ.get("GPUSTACKER_DUMP_AFTER"):           # debugging hangs: dump all thread stacks
        import faulthandler
        faulthandler.dump_traceback_later(float(os.environ["GPUSTACKER_DUMP_AFTER"]), repeat=True)
    ap = argparse.ArgumentParser(prog="python -m gpustacker.engine")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run", help="run a JSON job file, JSON-lines events on stdout")
    p_run.add_argument("job")
    p_run.add_argument("--cancel-file", help="stop cleanly when this file appears")
    p_run.add_argument("--parent-pid", type=int, help="stop cleanly when this process exits")
    sub.add_parser("probe", help="print GPU info")
    p_st = sub.add_parser("stack", help="stack a Siril sequence (.seq) or a folder of raw lights")
    p_st.add_argument("seq", help=".seq file, or a folder of raw lights (or its parent with a 'lights' folder)")
    p_st.add_argument("--scale", type=float, default=1.0, help="drizzle scale (raw lights)")
    p_st.add_argument("--pixfrac", type=float, default=1.0, help="drizzle pixfrac (raw lights)")
    p_st.add_argument("--analyze-only", action="store_true", help="raw lights: detect stars and register only")
    p_st.add_argument("--out", default="")
    p_st.add_argument("--rejection", default="winsorized", choices=("winsorized", "sigma", "none"))
    p_st.add_argument("--sigma", type=float, nargs=2, default=(3.0, 3.0))
    p_st.add_argument("--weight", default="noise", choices=("none", "noise", "wfwhm", "nbstars"))
    p_st.add_argument("--feather", type=float, default=0.0)
    p_st.add_argument("--no-norm", action="store_true", help="no addscale normalization")
    p_st.add_argument("--no-rgb-equal", action="store_true")
    p_st.add_argument("--no-output-norm", action="store_true")
    p_st.add_argument("--rows", type=int, nargs=2)
    p_st.add_argument("--crop", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    p_st.add_argument("--overwrite", action="store_true")
    p_st.add_argument("--gpu-mem-gib", type=float, default=0.0)
    args = ap.parse_args(argv)

    if args.cmd == "probe":
        from .job import probe_gpu
        try:
            print(json.dumps({"ev": "probe", **probe_gpu()}), flush=True)
            return 0
        except Exception as e:                       # noqa: BLE001
            print(json.dumps({"ev": "probe", "error": f"{type(e).__name__}: {e}"}), flush=True)
            return 1

    if args.cmd == "run":
        with open(args.job, encoding="utf-8") as f:
            cfg = json.load(f)
        cancel = threading.Event()
        if args.cancel_file or args.parent_pid:
            threading.Thread(target=_watch, args=(cancel, args.cancel_file, args.parent_pid), daemon=True).start()
        return _run(cfg, _json_out(), cancel)

    source = dict(lights_dir=args.seq, drizzle_scale=args.scale, pixfrac=args.pixfrac,
                  analyze_only=args.analyze_only) if os.path.isdir(args.seq) else dict(seq_path=args.seq)
    cfg = dict(**source, output_path=args.out, rejection=args.rejection, sigma_low=args.sigma[0],
               sigma_high=args.sigma[1], weighting=args.weight, feather=args.feather,
               normalization="none" if args.no_norm else "addscale", rgb_equal=not args.no_rgb_equal,
               output_norm=not args.no_output_norm, rows=list(args.rows) if args.rows else None,
               crop=list(args.crop) if args.crop else None, overwrite=args.overwrite, gpu_mem_gib=args.gpu_mem_gib)
    cancel = threading.Event()
    t = threading.Thread(target=lambda: setattr(t, "rc", _run(cfg, _human, cancel)), daemon=True)
    t.rc = 1
    t.start()
    try:
        while t.is_alive():
            time.sleep(0.2)
    except KeyboardInterrupt:
        cancel.set()
        t.join()
    return t.rc


if __name__ == "__main__":
    sys.exit(main())
