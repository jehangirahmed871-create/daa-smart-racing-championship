"""Experiment 1 - BFS and DFS.

Objective   Measure BFS/DFS runtime, visited vertices and examined edges on sparse
            and denser graphs of growing size; compare growth with O(V+E).
Input       make_grid_graph(n, density, seed): connected jittered-grid graphs,
            sparse (E~2V), medium (E~4V), dense (E~8V).  Same graph instance for
            BFS and DFS.  Source = vertex 0.
Procedure   1 warm-up run + RUNS timed runs per (algorithm, density, size).
            Timing = algorithm call only (includes its own queue/stack/visited
            initialisation; excludes graph construction, correctness checks,
            rendering and file I/O).  Correctness checks run SEPARATELY first.
Checks      BFS levels == independent unit-weight Dijkstra (heapq) levels for every
            vertex; DFS weak-component count == BFS component count; DFS cycle
            flag == (E > V - components) which is exact for undirected graphs.
Run         python -m experiments.exp1_bfs_dfs [--sizes 500 1000 ...] [--runs 5]
Output      experiments/results/exp1_raw.csv, exp1_summary.csv, exp1_*.png
"""
import argparse
import heapq
import time

from experiments.common_exp import environment, quiet_gc, stats, try_pyplot, write_csv
from experiments.graphgen import make_grid_graph
from graph.traversal import bfs, dfs, weak_components

DENSITIES = ("sparse", "medium", "dense")


def verify(g):
    n = len(g.vertices)
    ref = {0: 0}
    heap = [(0, 0)]
    while heap:                                   # unit-weight Dijkstra reference
        d, u = heapq.heappop(heap)
        if d > ref.get(u, 1 << 60):
            continue
        for _, v in g.neighbors(u):
            if d + 1 < ref.get(v, 1 << 60):
                ref[v] = d + 1
                heapq.heappush(heap, (d + 1, v))
    res = bfs(g, 0)
    assert res.dist == ref, "BFS levels differ from reference"
    comps = weak_components(g)
    seen, n_bfs = set(), 0
    for v in g.vertices:
        if v not in seen:
            seen |= bfs(g, v).reachable
            n_bfs += 1
    assert len(comps) == n_bfs, "component counts differ"
    forest = dfs(g)
    assert forest.has_cycle == (len(g.edges) > n - len(comps)), "cycle flag wrong"
    assert len(forest.order) == n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[500, 1000, 2000, 4000, 8000, 16000])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    raw, summary = [], []
    print("environment:", environment())
    for density in DENSITIES:
        for n in a.sizes:
            g = make_grid_graph(n, density, a.seed)
            V, E = len(g.vertices), len(g.edges)
            verify(g)
            for name, fn in (("BFS", lambda: bfs(g, 0)), ("DFS", lambda: dfs(g, 0))):
                fn()                                    # warm-up (discarded)
                times = []
                for run in range(1, a.runs + 1):
                    quiet_gc()
                    r = fn()
                    c = r.counters
                    raw.append({"algorithm": name, "density": density, "V": V, "E": E,
                                "run": run, "time_ms": round(c.time_ms, 4),
                                "visited": c.visited, "examined_edges": c.examined_edges,
                                "expanded": c.expanded})
                    times.append(c.time_ms)
                s = stats(times)
                summary.append({"algorithm": name, "density": density, "V": V, "E": E,
                                "V_plus_E": V + E, "mean_ms": round(s["mean"], 4),
                                "median_ms": round(s["median"], 4), "std_ms": round(s["std"], 4),
                                "visited": c.visited, "examined_edges": c.examined_edges,
                                "us_per_V_plus_E": round(s["mean"] * 1000 / (V + E), 4)})
                print(f"{name} {density:6s} V={V:6d} E={E:7d} mean={s['mean']:9.3f} ms "
                      f"visited={c.visited} examined={c.examined_edges}")
    print("wrote", write_csv("exp1_raw.csv", raw))
    print("wrote", write_csv("exp1_summary.csv", summary))
    plt = try_pyplot()
    if plt:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for alg, ls in (("BFS", "-"), ("DFS", "--")):
            for d in DENSITIES:
                pts = [(r["V_plus_E"], r["mean_ms"]) for r in summary
                       if r["algorithm"] == alg and r["density"] == d]
                ax.plot(*zip(*pts), ls, marker="o", label=f"{alg} {d}")
        ax.set_xlabel("V + E"); ax.set_ylabel("mean runtime (ms)")
        ax.set_title("Exp 1: BFS/DFS runtime vs V+E"); ax.legend(fontsize=7); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(f"{__file__.rsplit('/', 1)[0]}/results/exp1_runtime.png", dpi=150)


if __name__ == "__main__":
    main()
