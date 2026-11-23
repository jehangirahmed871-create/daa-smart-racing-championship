"""Navigator: the single object Members 2 and 3 talk to (group contract).

    nav = Navigator(loadGraph("data/sample_circuit.json"))
    nav.shortestPath(src, dst, preset="Speed")        -> RouteResult
    nav.planRoute(racer, preset="Balanced")           -> RouteResult (one lap)
    nav.blockEdge(id) / unblockEdge(id) / setRestricted(id, factor) /
    nav.setEdgeEffects(id, ...) / setVertexActive(id, bool)   (graph updates)

`racer` may be any object (or dict) exposing current_vertex / currentVertex and
next_checkpoint / nextCheckpoint (the checkpointOrder it must visit next).
Cost models are cached per preset; the normalisation reference values are
computed once, so a model never changes just because an edge got blocked.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

from graph import traversal
from graph.cost import CostModel, compute_reference_values, get_weights
from graph.graph import Graph
from graph.routing import (RouteResult, compare_presets, plan_route,
                           plan_route_steps, route_metrics, shortest_path,
                           astar_steps, dijkstra_steps)


def _attr(obj, *names, default=None):
    for n in names:
        if isinstance(obj, dict):
            if n in obj:
                return obj[n]
        elif hasattr(obj, n):
            return getattr(obj, n)
    return default


class Navigator:
    def __init__(self, graph: Graph, default_preset: str = "Balanced"):
        self.graph = graph
        self.default_preset = default_preset
        self._refs = compute_reference_values(graph)
        self._refs_version = graph.structure_version
        self._models: Dict[str, CostModel] = {}

    # ---- cost models ---------------------------------------------------
    def model(self, preset=None) -> CostModel:
        preset = preset or self.default_preset
        if self._refs_version != self.graph.structure_version:
            self._refs = compute_reference_values(self.graph)
            self._refs_version = self.graph.structure_version
            self._models.clear()
        key = preset if isinstance(preset, str) else repr(get_weights(preset))
        if key not in self._models:
            self._models[key] = CostModel(self.graph, preset, self._refs)
        return self._models[key]

    # ---- graph updates (pass-through) ----------------------------------
    def blockEdge(self, edge_id): self.graph.block_edge(edge_id)
    def unblockEdge(self, edge_id): self.graph.unblock_edge(edge_id)
    def setRestricted(self, edge_id, time_factor=1.5): self.graph.set_restricted(edge_id, time_factor)
    def setEdgeEffects(self, edge_id, **kw): self.graph.set_edge_effects(edge_id, **kw)
    def setVertexActive(self, vertex_id, active): self.graph.set_vertex_active(vertex_id, active)

    # ---- routing -------------------------------------------------------
    def shortestPath(self, src: int, dst: int, preset=None,
                     algorithm: str = "dijkstra") -> RouteResult:
        return shortest_path(self.graph, src, dst, preset or self.default_preset,
                             algorithm, self.model(preset))

    def planRoute(self, racer, preset=None, algorithm: str = "dijkstra") -> RouteResult:
        cur = _attr(racer, "current_vertex", "currentVertex")
        if cur is None:
            raise ValueError("racer needs current_vertex / currentVertex")
        nxt = _attr(racer, "next_checkpoint", "nextCheckpoint", default=1)
        return plan_route(self.graph, cur, nxt, self.model(preset),
                          algorithm=algorithm)

    def planRouteFrom(self, current_vertex: int, next_checkpoint: int = 1,
                      preset=None, algorithm: str = "dijkstra") -> RouteResult:
        return plan_route(self.graph, current_vertex, next_checkpoint,
                          self.model(preset), algorithm=algorithm)

    def comparePresets(self, src: int, dst: int, algorithm: str = "dijkstra") -> dict:
        return compare_presets(self.graph, src, dst, algorithm)

    def compareRoutes(self, edge_lists: Sequence[Sequence[int]], preset=None):
        """Distance / time / fuel / scalar cost of candidate routes (Route Planner)."""
        m = self.model(preset)
        return [route_metrics(self.graph, m, el) for el in edge_lists]

    # ---- exploration (Act 1) ------------------------------------------
    def bfs(self, src, dst=None): return traversal.bfs(self.graph, src, dst)
    def dfs(self, src=None): return traversal.dfs(self.graph, src)
    def components(self): return traversal.weak_components(self.graph)
    def hasCycle(self): return traversal.has_cycle(self.graph)
    def deadEnds(self): return traversal.find_dead_ends(self.graph)
    def reachability(self, src, destinations=None):
        return traversal.reachability_report(self.graph, src, destinations)
    def alternativeRoutes(self, src, dst, main_edges=(), **limits):
        return traversal.alternative_routes(self.graph, src, dst, main_edges, **limits)

    # ---- step hooks for the Algorithm Lab ------------------------------
    # (the SAME generators the plain functions run, with emit=True)
    def bfsSteps(self, src, dst=None):
        return traversal.bfs_steps(self.graph, src, dst, True)
    def dfsSteps(self, src=None):
        return traversal.dfs_steps(self.graph, src, True)
    def dijkstraSteps(self, src, dst, preset=None):
        return dijkstra_steps(self.graph, src, dst, self.model(preset), True)
    def astarSteps(self, src, dst, preset=None):
        return astar_steps(self.graph, src, dst, self.model(preset), True)
    def planRouteSteps(self, current_vertex, next_checkpoint=1, preset=None,
                       algorithm="dijkstra"):
        return plan_route_steps(self.graph, current_vertex, next_checkpoint,
                                self.model(preset), algorithm=algorithm, emit=True)
