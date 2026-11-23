"""Shared helpers for the experiment scripts (stats, CSV, environment, charts)."""
import csv
import gc
import os
import platform
import statistics
import sys

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def environment():
    return {"python": sys.version.split()[0], "implementation": platform.python_implementation(),
            "os": f"{platform.system()} {platform.release()}", "machine": platform.machine(),
            "processor": platform.processor() or "n/a"}


def write_csv(name, rows, fieldnames=None):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    if not rows:
        return path
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return path


def stats(values):
    vals = list(values)
    return {"mean": statistics.fmean(vals), "median": statistics.median(vals),
            "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals), "max": max(vals)}


def quiet_gc():
    gc.collect()


def try_pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None
