"""Act 1 - Circuit Explorer: BFS and DFS (Section 5.1).

All traversals respect edge direction and skip Blocked edges (unless
include_blocked=True).  Every traversal is ITERATIVE (explicit stack/queue) so
long roads cannot hit Python's recursion limit.

Each algorithm is a generator `*_steps(..., emit=True)` (step hooks for the
Algorithm Lab, final result = generator return value) plus a plain wrapper
that runs the SAME code with emit=False for gameplay and timing.

Complexity
  BFS                    O(V+E) time, O(V) auxiliary space
  DFS / components       O(V+E) time, O(V) auxiliary space
  dead-end analysis      O(V+E)
  alternative routes     exponential worst case -> hard limits (see below)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, Iterable, List, Optional, Sequence, Set

from common.counters import Counters
from common.steps import run_to_completion
from graph.graph import Graph


def _require_vertex(graph: Graph, vid: int, what: str = "vertex") -> None:
    if vid not in graph.vertices:
        raise ValueError(f"unknown {what} id {vid}")


# =========================================================================
# BFS
# =========================================================================
@dataclass
class BFSResult:
    start: int
    target: Optional[int]
    found: bool                      # target reached (True if no target given)
    order: List[int]                 # expansion order
    dist: Dict[int, int]             # minimum number of edges from start
    parent: Dict[int, Optional[int]]
    parent_edge: Dict[int, Optional[int]]
    path_vertices: List[int]
    path_edges: List[int]
    counters: Counters

    @property
    def reachable(self) -> Set[int]:
        return set(self.dist)


def bfs_steps(graph: Graph, start: int, target: Optional[int] = None,
              emit: bool = True, include_blocked: bool = False):
    """BFS: minimum NUMBER OF EDGES (not minimum distance/time/cost!)."""
    _require_vertex(graph, start, "start vertex")
    if target is not None:
        _require_vertex(graph, target, "target vertex")
    c = Counters()
    t0 = perf_counter()
    dist = {start: 0}
    parent: Dict[int, Optional[int]] = {start: None}
    pedge: Dict[int, Optional[int]] = {start: None}
    order: List[int] = []
    queue = deque([start])
    c.visited = 1
    if emit:
        yield {"op": "discover", "vertex": start, "parent": None, "edge": None}
    reached = target is not None and target == start
    while queue and not reached:
        u = queue.popleft()
        c.expanded += 1
        order.append(u)
        if emit:
            yield {"op": "expand", "vertex": u}
        for e, v in graph.neighbors(u, include_blocked):
            c.examined_edges += 1
            if emit:
                yield {"op": "examine", "edge": e.id, "from": u, "to": v}
            if v not in dist:
                dist[v] = dist[u] + 1
                parent[v] = u
                pedge[v] = e.id
                c.visited += 1
                queue.append(v)
                if emit:
                    yield {"op": "discover", "vertex": v, "parent": u, "edge": e.id}
                if v == target:
                    reached = True
                    break
    found = True if target is None else (target in dist)
    pv: List[int] = []
    pe: List[int] = []
    if target is not None and found:
        v = target
        while v is not None:
            pv.append(v)
            if pedge[v] is not None:
                pe.append(pedge[v])
            v = parent[v]
        pv.reverse()
        pe.reverse()
        c.path_cost = float(len(pe))
    c.time_ms = (perf_counter() - t0) * 1000.0
    return BFSResult(start, target, found, order, dist, parent, pedge, pv, pe, c)


def bfs(graph: Graph, start: int, target: Optional[int] = None,
        include_blocked: bool = False) -> BFSResult:
    return run_to_completion(bfs_steps(graph, start, target, False, include_blocked))


def reachability_report(graph: Graph, start: int,
                        destinations: Optional[Iterable[int]] = None) -> dict:
    """Which destinations can be reached from `start`?  (Never crashes/loops.)

    Default destinations = every mandatory checkpoint + the Finish vertex.
    """
    res = bfs(graph, start)
    if destinations is None:
        dests = [v.id for v in graph.checkpoints()]
        if graph.finish_id is not None:
            dests.append(graph.finish_id)
    else:
        dests = list(destinations)
    reachable = [d for d in dests if d in res.dist]
    unreachable = [d for d in dests if d not in res.dist]
    return {"start": start, "reachable": reachable, "unreachable": unreachable,
            "min_edges": {d: res.dist[d] for d in reachable},
            "counters": res.counters}


# =========================================================================
# DFS
# =========================================================================
@dataclass
class DFSResult:
    order: List[int]                 # discovery order
    discovery: Dict[int, int]
    finish: Dict[int, int]
    parent: Dict[int, Optional[int]]
    parent_edge: Dict[int, Optional[int]]
    back_edges: List[int]            # edge ids that close a cycle
    has_cycle: bool
    counters: Counters


def dfs_steps(graph: Graph, start: Optional[int] = None,
              emit: bool = True, include_blocked: bool = False):
    """Iterative DFS with white/gray/black colouring.

    start=None -> DFS forest over every vertex (used for whole-graph cycle
    detection); otherwise a single tree from `start`.

    Cycle rule: while scanning edge e out of u,
        v unseen            -> tree edge
        v gray (on stack)   -> BACK EDGE = cycle, unless e is the very edge we
                               arrived on (that is just the way back up an
                               undirected road; parallel roads still count)
        v black (finished)  -> ignored (forward/cross edge, no new cycle)
    """
    if start is not None:
        _require_vertex(graph, start, "start vertex")
        roots = [start]
    else:
        roots = list(graph.vertices)
    c = Counters()
    t0 = perf_counter()
    color: Dict[int, int] = {}           # missing=white, 1=gray, 2=black
    disc: Dict[int, int] = {}
    fin: Dict[int, int] = {}
    parent: Dict[int, Optional[int]] = {}
    pedge: Dict[int, Optional[int]] = {}
    order: List[int] = []
    back: List[int] = []
    clock = 0
    for root in roots:
        if root in color:
            continue
        color[root] = 1
        clock += 1
        disc[root] = clock
        parent[root] = None
        pedge[root] = None
        order.append(root)
        c.visited += 1
        if emit:
            yield {"op": "discover", "vertex": root, "parent": None, "edge": None}
        stack = [(root, graph.neighbors(root, include_blocked))]
        while stack:
            u, it = stack[-1]
            descended = False
            for e, v in it:
                c.examined_edges += 1
                if emit:
                    yield {"op": "examine", "edge": e.id, "from": u, "to": v}
                state = color.get(v, 0)
                if state == 0:
                    color[v] = 1
                    clock += 1
                    disc[v] = clock
                    parent[v] = u
                    pedge[v] = e.id
                    order.append(v)
                    c.visited += 1
                    if emit:
                        yield {"op": "discover", "vertex": v, "parent": u, "edge": e.id}
                    stack.append((v, graph.neighbors(v, include_blocked)))
                    descended = True
                    break
                if state == 1 and e.id != pedge[u]:
                    back.append(e.id)
                    if emit:
                        yield {"op": "cycle", "edge": e.id, "from": u, "to": v}
            if not descended:
                stack.pop()
                color[u] = 2
                clock += 1
                fin[u] = clock
                c.expanded += 1
                if emit:
                    yield {"op": "finish", "vertex": u}
    c.time_ms = (perf_counter() - t0) * 1000.0
    return DFSResult(order, disc, fin, parent, pedge, back, bool(back), c)


def dfs(graph: Graph, start: Optional[int] = None,
        include_blocked: bool = False) -> DFSResult:
    return run_to_completion(dfs_steps(graph, start, False, include_blocked))


def has_cycle(graph: Graph, include_blocked: bool = False) -> bool:
    return dfs(graph, None, include_blocked).has_cycle


def weak_components(graph: Graph, include_blocked: bool = False) -> List[List[int]]:
    """Connected components, ignoring edge direction (weak connectivity).

    Iterative DFS over an undirected view.  O(V+E).  Returned as sorted
    vertex lists, ordered by their smallest vertex id.
    """
    und: Dict[int, List[int]] = {v: [] for v in graph.vertices}
    for e in graph.edges.values():
        if e.is_blocked and not include_blocked:
            continue
        und[e.frm].append(e.to)
        und[e.to].append(e.frm)
    seen: Set[int] = set()
    comps: List[List[int]] = []
    for s in graph.vertices:
        if s in seen:
            continue
        seen.add(s)
        comp = []
        stack = [s]
        while stack:
            u = stack.pop()
            comp.append(u)
            for w in und[u]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        comps.append(sorted(comp))
    comps.sort(key=lambda c: c[0])
    return comps


# =========================================================================
# dead ends
# =========================================================================
def find_dead_ends(graph: Graph, finish_id: Optional[int] = None,
                   include_blocked: bool = False) -> Dict[int, str]:
    """Dead-end analysis.  Returns {vertex_id: reason}.

    Definition (Section 5.1): a NON-MANDATORY vertex from which the finish
    cannot be reached without reversing.  Implemented as the union of

      "cannot-reach-finish"  no directed path to the finish exists at all
                             (reverse BFS from the finish, honours one-way
                             roads and closures);
      "cul-de-sac"           vertex in a pendant branch: repeatedly peel
                             non-protected vertices of degree <= 1 (every
                             route out of a branch must turn round at its tip).

    Start, Finish and mandatory checkpoints are never reported.  A spur whose
    tip is a mandatory checkpoint is NOT a dead end: the racer is required to
    go there.  Both parts are O(V+E).
    """
    finish = graph.finish_id if finish_id is None else finish_id
    if finish is None:
        raise ValueError("no Finish vertex in the graph; pass finish_id")
    _require_vertex(graph, finish, "finish vertex")
    usable = [e for e in graph.edges.values() if include_blocked or not e.is_blocked]

    # (A) reverse reachability to the finish
    rev: Dict[int, List[int]] = {v: [] for v in graph.vertices}
    for e in usable:
        rev[e.to].append(e.frm)
        if not e.directed:
            rev[e.frm].append(e.to)
    can_reach = {finish}
    stack = [finish]
    while stack:
        u = stack.pop()
        for w in rev[u]:
            if w not in can_reach:
                can_reach.add(w)
                stack.append(w)

    # (B) leaf peeling on the undirected structure
    inc: Dict[int, List[int]] = {v: [] for v in graph.vertices}
    for e in usable:
        inc[e.frm].append(e.id)
        inc[e.to].append(e.id)
    deg = {v: len(ids) for v, ids in inc.items()}
    removed_edges: Set[int] = set()
    queued: Set[int] = set()
    queue = deque()
    for vid, v in graph.vertices.items():
        if not v.is_protected and deg[vid] <= 1:
            queue.append(vid)
            queued.add(vid)
    peeled: Set[int] = set()
    while queue:
        u = queue.popleft()
        peeled.add(u)
        for eid in inc[u]:
            if eid in removed_edges:
                continue
            removed_edges.add(eid)
            e = graph.edges[eid]
            w = e.other(u)
            deg[w] -= 1
            if (w not in queued and not graph.vertices[w].is_protected
                    and deg[w] <= 1):
                queued.add(w)
                queue.append(w)

    result: Dict[int, str] = {}
    for vid, v in graph.vertices.items():
        if v.is_protected:
            continue
        if vid in peeled:
            result[vid] = "cul-de-sac"
        elif vid not in can_reach:
            result[vid] = "cannot-reach-finish"
    return result


# =========================================================================
# alternative routes (bounded!)
# =========================================================================
@dataclass
class AlternativeRoute:
    vertices: List[int]
    edges: List[int]


@dataclass
class AlternativeRoutes:
    routes: List[AlternativeRoute]
    truncated: bool                  # a limit stopped the search early
    reason: Optional[str]            # "max_routes" | "max_expansions" | None
    expansions: int                  # edges examined (work done)


def alternative_routes(graph: Graph, src: int, dst: int,
                       main_edges: Sequence[int] = (), max_routes: int = 10,
                       max_edges: Optional[int] = None,
                       max_expansions: int = 200_000,
                       include_blocked: bool = False) -> AlternativeRoutes:
    """Enumerate simple src->dst paths that differ from the main route.

    Alternative route (spec): a simple path (no repeated vertex) whose edge
    sequence differs from the selected main route by at least one edge.
    One DFS is O(V+E), but ENUMERATING all simple paths is exponential in the
    worst case, so three documented limits apply:
      max_routes      stop after this many routes (default 10)
      max_edges       maximum path length in edges (default V-1, the longest
                      possible simple path)
      max_expansions  hard cap on edges examined (default 200 000)
    Pruning: only vertices that can still reach dst are entered (reverse BFS),
    which cuts dead branches without removing any valid route.
    Iterative DFS, deterministic order (adjacency insertion order).
    """
    _require_vertex(graph, src, "source vertex")
    _require_vertex(graph, dst, "destination vertex")
    if src == dst:
        return AlternativeRoutes([], False, None, 0)
    if max_edges is None:
        max_edges = len(graph.vertices) - 1
    main = tuple(main_edges)

    rev: Dict[int, List[int]] = {v: [] for v in graph.vertices}
    for e in graph.edges.values():
        if e.is_blocked and not include_blocked:
            continue
        rev[e.to].append(e.frm)
        if not e.directed:
            rev[e.frm].append(e.to)
    can_reach = {dst}
    st = [dst]
    while st:
        u = st.pop()
        for w in rev[u]:
            if w not in can_reach:
                can_reach.add(w)
                st.append(w)
    if src not in can_reach:
        return AlternativeRoutes([], False, None, 0)

    routes: List[AlternativeRoute] = []
    expansions = 0
    truncated = False
    reason: Optional[str] = None
    on_path = {src}
    path_v = [src]
    path_e: List[int] = []
    stack = [(src, graph.neighbors(src, include_blocked))]
    while stack and not truncated:
        u, it = stack[-1]
        descended = False
        for e, v in it:
            expansions += 1
            if expansions > max_expansions:
                truncated, reason = True, "max_expansions"
                break
            if v in on_path or v not in can_reach or len(path_e) + 1 > max_edges:
                continue
            if v == dst:
                if tuple(path_e) + (e.id,) != main:
                    routes.append(AlternativeRoute(path_v + [v], path_e + [e.id]))
                    if len(routes) >= max_routes:
                        truncated, reason = True, "max_routes"
                        break
                continue
            on_path.add(v)
            path_v.append(v)
            path_e.append(e.id)
            stack.append((v, graph.neighbors(v, include_blocked)))
            descended = True
            break
        if truncated:
            break
        if not descended:
            stack.pop()
            on_path.discard(u)
            if stack:                 # u was not the root: undo the step into u
                path_v.pop()
                path_e.pop()
    return AlternativeRoutes(routes, truncated, reason, expansions)
