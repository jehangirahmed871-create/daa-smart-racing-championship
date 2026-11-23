"""Act 2 - Route Optimization: Dijkstra and A* (Section 5.2).

Both searches minimise the scalar cost of graph/cost.py (non-negative), skip
Blocked edges, and obey one-way roads.  Each is a generator `*_steps` (Lab step
hooks) with a plain wrapper running the same code with emit=False.

Complexity
  Dijkstra (binary heap, lazy deletion)   O((V+E) log V)
  A*                                      worst case like Dijkstra; practical
                                          speed depends on the heuristic

Counters (see common/counters.py): expanded, relaxed (improving relaxations),
examinedEdges, heapOps, visited, reopened (A*), pathCost, timeMs.  timeMs
includes data-structure initialisation (heap, label dicts) and excludes
graph construction, rendering and file I/O.

A* heuristic: h(v) = ratio * euclid(v, target) with ratio = minimum
cost-per-straight-line-distance over all edges (CostModel.heuristic_ratio).
It is admissible AND consistent, so closed vertices are never reopened; the
code still supports reopening so an admissible-but-inconsistent custom
heuristic remains optimal.  If the ratio is 0, h = 0 (A* == Dijkstra).

Ties: heap entries carry an increasing sequence number, and adjacency order is
edge-insertion order, so equal-cost situations resolve deterministically.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from common.counters import Counters
from common.heap import MinHeap
from common.steps import run_to_completion
from graph.cost import CostModel, PRESETS
from graph.graph import Graph

INF = math.inf


@dataclass
class RouteResult:
    found: bool
    src: int
    dst: Optional[int]
    algorithm: str
    preset: str
    vertices: List[int]
    edges: List[int]
    cost: float                       # the EXACT scalar cost that was optimised
    distance: float
    time: float
    fuel: float
    counters: Counters
    legs: List["RouteResult"] = field(default_factory=list)
    failed_leg: Optional[Tuple[int, int]] = None   # (from, to) that is unreachable
    dist_map: Optional[Dict[int, float]] = None    # labels at termination

    def summary(self) -> dict:
        return {"found": self.found, "algorithm": self.algorithm,
                "preset": self.preset, "vertices": self.vertices,
                "edges": self.edges, "cost": self.cost,
                "distance": self.distance, "time": self.time, "fuel": self.fuel,
                "counters": self.counters.to_dict()}


def route_metrics(graph: Graph, model: CostModel, edge_ids: Sequence[int]) -> dict:
    """Totals along a route: the scalar cost is the model's cost, unchanged."""
    dist = time = fuel = cost = 0.0
    for eid in edge_ids:
        e = graph.edges[eid]
        dist += e.distance
        time += e.effective_time
        fuel += e.effective_fuel
        cost += model.edge_cost(e)
    return {"distance": dist, "time": time, "fuel": fuel, "cost": cost}


def _require(graph: Graph, vid: Optional[int], what: str) -> None:
    if vid is None or vid not in graph.vertices:
        raise ValueError(f"unknown {what} id {vid}")


def _build_result(graph, model, algorithm, src, dst, found, parent_v, parent_e,
                  cost, counters, dist_map) -> RouteResult:
    verts: List[int] = []
    edges: List[int] = []
    if found and dst is not None:
        v = dst
        while v is not None:
            verts.append(v)
            if parent_e.get(v) is not None:
                edges.append(parent_e[v])
            v = parent_v.get(v)
        verts.reverse()
        edges.reverse()
        m = route_metrics(graph, model, edges)
        counters.path_cost = cost
        return RouteResult(True, src, dst, algorithm, model.preset_name, verts,
                           edges, cost, m["distance"], m["time"], m["fuel"],
                           counters, dist_map=dist_map)
    counters.path_cost = None
    return RouteResult(found, src, dst, algorithm, model.preset_name, [], [],
                       INF if dst is not None else 0.0, 0.0, 0.0, 0.0, counters,
                       dist_map=dist_map)


# =========================================================================
# Dijkstra
# =========================================================================
def dijkstra_steps(graph: Graph, src: int, dst: Optional[int], model: CostModel,
                   emit: bool = True):
    """Dijkstra with a binary min-heap and lazy deletion of stale entries.

    dst=None -> single-source labels to every reachable vertex (dist_map).
    """
    _require(graph, src, "source vertex")
    if dst is not None:
        _require(graph, dst, "target vertex")
    c = Counters()
    t0 = perf_counter()
    heap = MinHeap()
    dist: Dict[int, float] = {src: 0.0}
    parent_v: Dict[int, Optional[int]] = {src: None}
    parent_e: Dict[int, Optional[int]] = {src: None}
    settled = set()
    seq = 0
    heap.push((0.0, seq, src))
    seq += 1
    c.visited = 1
    if emit:
        yield {"op": "push", "vertex": src, "cost": 0.0}
    found = False
    edge_cost = model.edge_cost
    while heap:
        d, _, u = heap.pop()
        if u in settled:
            continue                      # stale entry (lazy deletion)
        settled.add(u)
        c.expanded += 1
        if emit:
            yield {"op": "expand", "vertex": u, "cost": d}
        if u == dst:
            found = True
            break
        for e, v in graph.neighbors(u):
            c.examined_edges += 1
            if v in settled:
                continue
            nd = d + edge_cost(e)
            if emit:
                yield {"op": "examine", "edge": e.id, "from": u, "to": v, "cost": nd}
            if nd < dist.get(v, INF):
                if v not in dist:
                    c.visited += 1
                dist[v] = nd
                parent_v[v] = u
                parent_e[v] = e.id
                heap.push((nd, seq, v))
                seq += 1
                c.relaxed += 1
                if emit:
                    yield {"op": "relax", "edge": e.id, "vertex": v, "cost": nd}
    c.heap_ops = heap.ops
    c.time_ms = (perf_counter() - t0) * 1000.0
    if dst is None:
        found = True
        return _build_result(graph, model, "dijkstra", src, None, True, parent_v,
                             parent_e, 0.0, c, dist)
    return _build_result(graph, model, "dijkstra", src, dst, found, parent_v,
                         parent_e, dist.get(dst, INF), c, dist)


def dijkstra(graph: Graph, src: int, dst: Optional[int], model: CostModel) -> RouteResult:
    return run_to_completion(dijkstra_steps(graph, src, dst, model, False))


# =========================================================================
# A*
# =========================================================================
Heuristic = Union[str, Callable[[Graph, int, int], float]]


def _make_heuristic(graph: Graph, model: CostModel, dst: int, heuristic: Heuristic):
    if callable(heuristic):
        return lambda v: heuristic(graph, v, dst)
    if heuristic == "zero":
        return lambda v: 0.0
    if heuristic == "euclid":
        ratio = model.heuristic_ratio()
        if ratio == 0.0:
            return lambda v: 0.0
        tv = graph.vertices[dst]
        tx, ty = tv.x, tv.y
        verts = graph.vertices
        cache: Dict[int, float] = {}

        def h(v):
            val = cache.get(v)
            if val is None:
                p = verts[v]
                val = ratio * math.hypot(p.x - tx, p.y - ty)
                cache[v] = val
            return val
        return h
    raise ValueError(f"unknown heuristic {heuristic!r}")


def astar_steps(graph: Graph, src: int, dst: int, model: CostModel,
                emit: bool = True, heuristic: Heuristic = "euclid"):
    """A*: f(v) = g(v) + h(v).  Reopens a closed vertex if a cheaper g is found."""
    _require(graph, src, "source vertex")
    _require(graph, dst, "target vertex")
    c = Counters()
    t0 = perf_counter()
    h = _make_heuristic(graph, model, dst, heuristic)
    heap = MinHeap()
    g: Dict[int, float] = {src: 0.0}
    parent_v: Dict[int, Optional[int]] = {src: None}
    parent_e: Dict[int, Optional[int]] = {src: None}
    closed = set()
    seq = 0
    heap.push((h(src), seq, 0.0, src))
    seq += 1
    c.visited = 1
    if emit:
        yield {"op": "push", "vertex": src, "cost": 0.0}
    found = False
    edge_cost = model.edge_cost
    while heap:
        f, _, gu, u = heap.pop()
        if gu > g[u]:
            continue                      # stale entry
        if u in closed:
            c.reopened += 1               # only possible if h is inconsistent
        closed.add(u)
        c.expanded += 1
        if emit:
            yield {"op": "expand", "vertex": u, "cost": gu, "f": f}
        if u == dst:
            found = True
            break
        for e, v in graph.neighbors(u):
            c.examined_edges += 1
            ng = gu + edge_cost(e)
            if emit:
                yield {"op": "examine", "edge": e.id, "from": u, "to": v, "cost": ng}
            if ng < g.get(v, INF):
                if v not in g:
                    c.visited += 1
                g[v] = ng
                parent_v[v] = u
                parent_e[v] = e.id
                heap.push((ng + h(v), seq, ng, v))
                seq += 1
                c.relaxed += 1
                if emit:
                    yield {"op": "relax", "edge": e.id, "vertex": v, "cost": ng}
    c.heap_ops = heap.ops
    c.time_ms = (perf_counter() - t0) * 1000.0
    name = "astar" if heuristic == "euclid" else f"astar({heuristic if isinstance(heuristic, str) else 'custom'})"
    return _build_result(graph, model, name, src, dst, found, parent_v, parent_e,
                         g.get(dst, INF), c, g)


def astar(graph: Graph, src: int, dst: int, model: CostModel,
          heuristic: Heuristic = "euclid") -> RouteResult:
    return run_to_completion(astar_steps(graph, src, dst, model, False, heuristic))


# =========================================================================
# convenience: shortest path + checkpoint-ordered planning
# =========================================================================
def shortest_path(graph: Graph, src: int, dst: int, preset="Balanced",
                  algorithm: str = "dijkstra", model: Optional[CostModel] = None,
                  heuristic: Heuristic = "euclid") -> RouteResult:
    """shortestPath(src, dst, preset).  algorithm = 'dijkstra' | 'astar'."""
    model = model or CostModel(graph, preset)
    if algorithm == "dijkstra":
        return dijkstra(graph, src, dst, model)
    if algorithm == "astar":
        return astar(graph, src, dst, model, heuristic)
    raise ValueError(f"unknown algorithm {algorithm!r}")


def plan_route_steps(graph: Graph, current_vertex: int, next_checkpoint: int,
                     model: CostModel, finish: Optional[int] = None,
                     algorithm: str = "dijkstra", emit: bool = True,
                     heuristic: Heuristic = "euclid"):
    """Checkpoint-ORDERED routing for ONE lap.

    Legs: current vertex -> checkpoint #next_checkpoint -> #(next+1) -> ... ->
    last checkpoint -> Finish, each leg an optimal shortest path.  Order can
    never be skipped because every leg ends exactly at the next required
    checkpoint.  Call again after any closure / cost change (from the racer's
    CURRENT vertex) and again with next_checkpoint=1 for the next lap.

    If a leg is unreachable the result has found=False and failed_leg set, so
    the race engine can apply its documented recovery rule (never freeze).
    """
    _require(graph, current_vertex, "current vertex")
    fin = graph.finish_id if finish is None else finish
    if fin is None:
        raise ValueError("graph has no Finish vertex; pass finish=")
    targets = [v.id for v in graph.checkpoints() if v.checkpoint_order >= next_checkpoint]
    targets.append(fin)
    legs: List[RouteResult] = []
    total = Counters()
    cur = current_vertex
    failed = None
    for t in targets:
        if algorithm == "dijkstra":
            leg = yield from dijkstra_steps(graph, cur, t, model, emit)
        elif algorithm == "astar":
            leg = yield from astar_steps(graph, cur, t, model, emit, heuristic)
        else:
            raise ValueError(f"unknown algorithm {algorithm!r}")
        legs.append(leg)
        total.merge(leg.counters)
        if emit:
            yield {"op": "leg_done", "from": cur, "to": t, "found": leg.found}
        if not leg.found:
            failed = (cur, t)
            break
        cur = t
    verts: List[int] = [current_vertex]
    edges: List[int] = []
    cost = dist = time = fuel = 0.0
    for leg in legs:
        if leg.found:
            verts.extend(leg.vertices[1:])
            edges.extend(leg.edges)
            cost += leg.cost
            dist += leg.distance
            time += leg.time
            fuel += leg.fuel
    ok = failed is None
    total.path_cost = cost if ok else None
    return RouteResult(ok, current_vertex, fin, algorithm, model.preset_name,
                       verts, edges, cost if ok else INF, dist, time, fuel,
                       total, legs=legs, failed_leg=failed)


def plan_route(graph: Graph, current_vertex: int, next_checkpoint: int,
               model: CostModel, finish: Optional[int] = None,
               algorithm: str = "dijkstra",
               heuristic: Heuristic = "euclid") -> RouteResult:
    return run_to_completion(plan_route_steps(
        graph, current_vertex, next_checkpoint, model, finish, algorithm,
        False, heuristic))


def compare_presets(graph: Graph, src: int, dst: int,
                    algorithm: str = "dijkstra") -> dict:
    """Route chosen by each preset (Challenge 2 demo).

    Returns {"routes": {preset: RouteResult}, "distinct_routes": n}.  Each
    route is also re-costed under EVERY preset so you can show the trade-off.
    """
    routes = {}
    for name in PRESETS:
        routes[name] = shortest_path(graph, src, dst, name, algorithm)
    distinct = {tuple(r.edges) for r in routes.values() if r.found}
    cross = {}
    for name, r in routes.items():
        cross[name] = {other: CostModel(graph, other).path_cost(r.edges)
                       for other in PRESETS}
    return {"routes": routes, "distinct_routes": len(distinct),
            "cost_under_each_preset": cross}
