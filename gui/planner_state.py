"""Route Planner logic (no pygame here, so it is unit-testable).

Everything shown on the screen comes from Member 1's real algorithms through
the Navigator - nothing is re-implemented in the GUI:
  optimal route        Dijkstra or A* under the chosen preset
  fewest roads         BFS (minimum number of edges - NOT minimum cost)
  alternatives         bounded DFS enumeration of simple paths, best 3 by cost
  lap plan             checkpoint-ordered planRoute
  reachable checkpoints  BFS from the selected source
Every candidate row shows distance, time, fuel and the exact scalar cost that
the router optimises (same CostModel).
"""
from __future__ import annotations

from typing import Dict, List, Optional

PRESET_ORDER = ["Balanced", "Speed", "Economy", "Safety"]


class PlannerState:
    def __init__(self, nav, max_alternatives: int = 3,
                 alt_limits: Optional[dict] = None):
        self.nav = nav
        self.graph = nav.graph
        self.preset = "Balanced"
        self.algorithm = "dijkstra"
        self.src: Optional[int] = None
        self.dst: Optional[int] = None
        self.candidates: List[dict] = []
        self.selected = 0
        self.lap_mode = False
        self.show_reachable = False
        self.reachable: set = set()
        self.unreachable_checkpoints: List[int] = []
        self.user_blocked: set = set()
        self.message = "Left-click a start vertex, then a destination."
        self.max_alternatives = max_alternatives
        self.alt_limits = alt_limits or {"max_routes": 200, "max_expansions": 50_000}
        self.alt_note = ""

    # ---- user actions ---------------------------------------------------
    def click_vertex(self, vid: int) -> None:
        if self.src is None or (self.src is not None and self.dst is not None):
            self.src, self.dst = vid, None
            self.message = f"Start = {vid}. Click a destination."
        else:
            self.dst = vid
            self.message = f"Route {self.src} -> {vid}"
        self.recompute()

    def set_preset(self, name: str) -> None:
        if name in PRESET_ORDER:
            self.preset = name
            self.recompute()

    def toggle_algorithm(self) -> None:
        self.algorithm = "astar" if self.algorithm == "dijkstra" else "dijkstra"
        self.recompute()

    def toggle_edge_block(self, edge_id: int) -> None:
        """Close/reopen a road; the route is recalculated immediately."""
        if edge_id in self.user_blocked:
            self.graph.unblock_edge(edge_id)
            self.user_blocked.discard(edge_id)
            self.message = f"Road {edge_id} reopened - route recalculated."
        else:
            self.graph.block_edge(edge_id)
            self.user_blocked.add(edge_id)
            self.message = f"Road {edge_id} closed - route recalculated."
        self.recompute()

    def clear_blocks(self) -> None:
        for eid in list(self.user_blocked):
            self.graph.unblock_edge(eid)
        self.user_blocked.clear()
        self.message = "All closures cleared."
        self.recompute()

    def toggle_lap(self) -> None:
        self.lap_mode = not self.lap_mode
        self.message = ("Lap plan: all checkpoints in order." if self.lap_mode
                        else "Point-to-point mode.")
        self.recompute()

    def toggle_reachable(self) -> None:
        self.show_reachable = not self.show_reachable
        self.recompute()

    def cycle_selected(self) -> None:
        if self.candidates:
            self.selected = (self.selected + 1) % len(self.candidates)

    # ---- computation ----------------------------------------------------
    @staticmethod
    def _row(label, kind, vertices, edges, m, counters=None):
        return {"label": label, "kind": kind, "vertices": list(vertices),
                "edges": list(edges), "distance": m["distance"], "time": m["time"],
                "fuel": m["fuel"], "cost": m["cost"], "counters": counters}

    def recompute(self) -> None:
        nav, p = self.nav, self.preset
        self.candidates, self.alt_note = [], ""
        self.reachable, self.unreachable_checkpoints = set(), []
        if self.show_reachable and self.src is not None:
            rep = nav.reachability(self.src)
            self.reachable = set(nav.bfs(self.src).dist)
            self.unreachable_checkpoints = rep["unreachable"]
        if self.src is None:
            return
        if self.lap_mode:
            r = nav.planRouteFrom(self.src, 1, p, self.algorithm)
            if r.found:
                self.candidates = [self._row("Full lap plan", "lap",
                                             r.vertices, r.edges, r.summary(),
                                             r.counters.to_dict())]
                self.message = f"Lap plan through {len(r.legs) - 1} checkpoints."
            else:
                a, b = r.failed_leg
                self.message = f"Lap blocked: no route {a} -> {b}. Recovery rule needed."
            self.selected = 0
            return
        if self.dst is None:
            return
        r = nav.shortestPath(self.src, self.dst, p, self.algorithm)
        if not r.found:
            self.message = f"No route {self.src} -> {self.dst}: destination unreachable."
            self.selected = 0
            return
        algo = "Dijkstra" if self.algorithm == "dijkstra" else "A*"
        rows = [self._row(f"Optimal ({p}, {algo})", "optimal", r.vertices,
                          r.edges, r.summary(), r.counters.to_dict())]
        seen = {tuple(r.edges)}
        b = nav.bfs(self.src, self.dst)
        if b.found and tuple(b.path_edges) not in seen:
            m = nav.compareRoutes([b.path_edges], p)[0]
            rows.append(self._row("Fewest roads (BFS)", "bfs", b.path_vertices,
                                  b.path_edges, m, b.counters.to_dict()))
            seen.add(tuple(b.path_edges))
        elif b.found:
            rows[0]["label"] += " = BFS"       # same route as the fewest-roads path
        alts = nav.alternativeRoutes(self.src, self.dst, r.edges, **self.alt_limits)
        scored = []
        for a in alts.routes:
            if tuple(a.edges) in seen:
                continue
            scored.append((nav.compareRoutes([a.edges], p)[0], a))
        scored.sort(key=lambda t: (t[0]["cost"], len(t[1].edges)))
        for k, (m, a) in enumerate(scored[: self.max_alternatives], 1):
            rows.append(self._row(f"Alternative {k}", "alt", a.vertices, a.edges, m))
        self.alt_note = (f"{len(alts.routes)} alternatives found"
                         + (f" (search limit: {alts.reason})" if alts.truncated else ""))
        self.candidates = rows
        self.selected = min(self.selected, len(rows) - 1)
