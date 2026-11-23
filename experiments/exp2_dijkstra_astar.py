"""Experiment 2 - Dijkstra vs A*.

Objective   Compare Dijkstra and A* (same cost model, same instances): runtime,
            expanded vertices, relaxed edges, heap operations, final cost.
Input       make_grid_graph(n, density, seed) road networks (distance >= straight
            line, coordinates documented by the generator) plus the supplied
            sample circuit.  PAIRS fixed (src,dst) pairs per graph from a seeded
            RNG.  Presets: Balanced, Speed, Economy, Safety, Distance.
Procedure   1 warm-up + RUNS timed runs per (graph, preset, pair, algorithm).
            Counters are deterministic, time is averaged.  timeMs includes the
            heap / label dict initialisation and excludes graph construction,
            cost-model construction, rendering and file I/O.
Checks      A* cost == Dijkstra cost (float tolerance) for EVERY pair; reported as
            cost_match (separate from the timed runs).
Report      per pair: whether A* expanded fewer / equal / more vertices.  The
            heuristic is only as strong as ratio = min cost-per-distance, so for
            presets dominated by difficulty/obstacle terms the ratio is tiny and A*
            behaves almost like Dijkstra.
Run         python -m experiments.exp2_dijkstra_astar [--sizes ...] [--runs 5] [--pairs 8]
Output      experiments/results/exp2_raw.csv, exp2_summary.csv, exp2_*.png
"""
import argparse
import math
import os
import random

from experiments.common_exp import environment, quiet_gc, stats, try_pyplot, write_csv
from experiments.graphgen import ROOT, make_grid_graph
from graph import CostModel, loadGraph
from graph.routing import astar, dijkstra

PRESET_LIST = ("Balanced", "Speed", "Economy", "Safety", "Distance")


def instances(sizes, densities, seed):
    for density in densities:
        for n in sizes:
            yield f"grid-{density}", make_grid_graph(n, density, seed)
    yield "sample_circuit", loadGraph(os.path.join(ROOT, "data", "sample_circuit.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[200, 500, 1000, 2000, 4000])
    ap.add_argument("--densities", nargs="+", default=["sparse", "medium"])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    print("environment:", environment())
    raw, summary = [], []
    for label, g in instances(a.sizes, a.densities, a.seed):
        V, E = len(g.vertices), len(g.edges)
        rng = random.Random(a.seed * 7919 + V)
        ids = list(g.vertices)
        pairs = [(rng.choice(ids), rng.choice(ids)) for _ in range(a.pairs)]
        pairs = [(s, t) for s, t in pairs if s != t]
        for preset in PRESET_LIST:
            model = CostModel(g, preset)
            agg = {"d": [], "a": [], "fewer": 0, "equal": 0, "more": 0,
                   "d_exp": [], "a_exp": [], "mismatch": 0}
            for s, t in pairs:
                rd, ra = dijkstra(g, s, t, model), astar(g, s, t, model)   # correctness check
                match = (rd.found == ra.found) and (not rd.found or math.isclose(rd.cost, ra.cost, rel_tol=1e-9, abs_tol=1e-12))
                agg["mismatch"] += (not match)
                for algo, fn in (("dijkstra", dijkstra), ("astar", astar)):
                    fn(g, s, t, model)                                      # warm-up
                    times = []
                    for run in range(1, a.runs + 1):
                        quiet_gc()
                        r = fn(g, s, t, model)
                        times.append(r.counters.time_ms)
                    c = r.counters
                    raw.append({"graph": label, "V": V, "E": E, "preset": preset, "src": s, "dst": t,
                                "algorithm": algo, "mean_time_ms": round(sum(times) / len(times), 4),
                                "expanded": c.expanded, "relaxed": c.relaxed, "examined_edges": c.examined_edges,
                                "heap_ops": c.heap_ops, "visited": c.visited, "reopened": c.reopened,
                                "path_cost": r.cost if r.found else "inf", "cost_match": match})
                    agg["d" if algo == "dijkstra" else "a"].append(sum(times) / len(times))
                agg["d_exp"].append(rd.counters.expanded)
                agg["a_exp"].append(ra.counters.expanded)
                key = "fewer" if ra.counters.expanded < rd.counters.expanded else \
                      "equal" if ra.counters.expanded == rd.counters.expanded else "more"
                agg[key] += 1
            n = len(pairs)
            summary.append({"graph": label, "V": V, "E": E, "preset": preset, "pairs": n,
                            "heuristic_ratio": round(model.heuristic_ratio(), 6),
                            "dijkstra_mean_ms": round(stats(agg["d"])["mean"], 4),
                            "astar_mean_ms": round(stats(agg["a"])["mean"], 4),
                            "dijkstra_mean_expanded": round(sum(agg["d_exp"]) / n, 1),
                            "astar_mean_expanded": round(sum(agg["a_exp"]) / n, 1),
                            "astar_fewer": agg["fewer"], "astar_equal": agg["equal"],
                            "astar_more": agg["more"], "cost_mismatches": agg["mismatch"]})
            s_ = summary[-1]
            print(f"{label:14s} V={V:5d} {preset:9s} ratio={s_['heuristic_ratio']:.5f} "
                  f"exp D/A={s_['dijkstra_mean_expanded']:8.1f}/{s_['astar_mean_expanded']:8.1f} "
                  f"ms D/A={s_['dijkstra_mean_ms']:7.3f}/{s_['astar_mean_ms']:7.3f} "
                  f"A* fewer/equal/more={s_['astar_fewer']}/{s_['astar_equal']}/{s_['astar_more']} "
                  f"mismatch={s_['cost_mismatches']}")
    print("wrote", write_csv("exp2_raw.csv", raw))
    print("wrote", write_csv("exp2_summary.csv", summary))
    plt = try_pyplot()
    if plt:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for algo, key in (("Dijkstra", "dijkstra_mean_expanded"), ("A*", "astar_mean_expanded")):
            pts = [(r["V"], r[key]) for r in summary if r["graph"] == "grid-sparse" and r["preset"] == "Distance"]
            if pts:
                ax.plot(*zip(*pts), marker="o", label=f"{algo} (Distance preset)")
        ax.set_xlabel("V"); ax.set_ylabel("mean expanded vertices")
        ax.set_title("Exp 2: expanded vertices, sparse grids"); ax.legend(); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(os.path.join(os.path.dirname(__file__), "results", "exp2_expanded.png"), dpi=150)


if __name__ == "__main__":
    main()
