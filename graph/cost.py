"""Scalar edge cost (Section 3.3).

    cost(e) = w_d*D(e) + w_t*T(e) + w_f*F(e) + w_r*R(e) + w_o*O(e)

Every term is NORMALIZED to a comparable scale before weighting:

    D = distance / D_ref        D_ref = max edge distance in the circuit
    T = effectiveTime / T_ref   T_ref = max edge baseTime in the circuit
    F = effectiveFuel / F_ref   F_ref = max edge fuelCost * (1 + 0.1*(5-1))
                                        (the worst-case fuel multiplier, so a
                                         base-state edge has F <= 1)
    R = (effDifficulty - 1) / 4 difficulty penalty, 0 (easy) .. 1 (hard)
    O = obstaclePenalty / 1     obstacle penalty, already on a 0..1 scale

The reference values come from STATIC edge attributes only, so they do not
move when roads are blocked or restricted mid-race (the cost of an unchanged
edge never changes because of someone else's event).

All weights >= 0 and every term >= 0  =>  every edge cost >= 0, so Dijkstra
is applicable.  The cost shown to the player and the cost optimized by the
router are the SAME function (CostModel.edge_cost).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Union

from graph.graph import FUEL_DIFFICULTY_STEP, MAX_DIFFICULTY, Edge, Graph


@dataclass(frozen=True)
class Weights:
    d: float   # distance
    t: float   # travel time
    f: float   # fuel
    r: float   # difficulty penalty
    o: float   # obstacle penalty

    def __post_init__(self):
        for name in ("d", "t", "f", "r", "o"):
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                raise ValueError(f"weight {name} must be finite and >= 0, got {v!r}")


# The four player presets (Section 3.3).  Each sums to 1 so costs of different
# presets are on the same 0..~1-per-edge scale.
PRESETS: Dict[str, Weights] = {
    "Balanced": Weights(d=0.25, t=0.25, f=0.20, r=0.15, o=0.15),
    "Speed":    Weights(d=0.20, t=0.60, f=0.05, r=0.05, o=0.10),
    "Economy":  Weights(d=0.20, t=0.10, f=0.55, r=0.05, o=0.10),
    "Safety":   Weights(d=0.05, t=0.10, f=0.05, r=0.40, o=0.40),
}
# Not a player preset: used by deterministic Opponent 3 ("minimum total distance").
INTERNAL_PRESETS: Dict[str, Weights] = {
    "Distance": Weights(d=1.0, t=0.0, f=0.0, r=0.0, o=0.0),
}


def get_weights(preset: Union[str, Weights]) -> Weights:
    if isinstance(preset, Weights):
        return preset
    if preset in PRESETS:
        return PRESETS[preset]
    if preset in INTERNAL_PRESETS:
        return INTERNAL_PRESETS[preset]
    raise ValueError(f"unknown preset {preset!r}; choose from "
                     f"{list(PRESETS) + list(INTERNAL_PRESETS)}")


def compute_reference_values(graph: Graph) -> Dict[str, float]:
    edges = list(graph.edges.values())
    d_ref = max((e.distance for e in edges), default=0.0) or 1.0
    t_ref = max((e.base_time for e in edges), default=0.0) or 1.0
    worst_mult = 1.0 + FUEL_DIFFICULTY_STEP * (MAX_DIFFICULTY - 1)
    f_ref = (max((e.fuel_cost for e in edges), default=0.0) * worst_mult) or 1.0
    return {"D": d_ref, "T": t_ref, "F": f_ref,
            "R": float(MAX_DIFFICULTY - 1), "O": 1.0}


class CostModel:
    def __init__(self, graph: Graph, preset: Union[str, Weights] = "Balanced",
                 refs: Optional[Dict[str, float]] = None):
        self.graph = graph
        self.preset_name = preset if isinstance(preset, str) else "Custom"
        self.weights = get_weights(preset)
        self.refs = dict(refs) if refs else compute_reference_values(graph)
        self._ratio: Optional[float] = None
        self._ratio_version = -1

    # ---- the cost function -------------------------------------------
    def components(self, e: Edge) -> Dict[str, float]:
        r = self.refs
        return {
            "D": e.distance / r["D"],
            "T": e.effective_time / r["T"],
            "F": e.effective_fuel / r["F"],
            "R": (e.effective_difficulty - 1) / r["R"],
            "O": e.obstacle_penalty / r["O"],
        }

    def edge_cost(self, e: Edge) -> float:
        w, r = self.weights, self.refs
        return (w.d * (e.distance / r["D"])
                + w.t * (e.effective_time / r["T"])
                + w.f * (e.effective_fuel / r["F"])
                + w.r * ((e.effective_difficulty - 1) / r["R"])
                + w.o * (e.obstacle_penalty / r["O"]))

    def breakdown(self, e: Edge) -> Dict[str, float]:
        """Weighted contribution of each criterion (for the planner screen)."""
        c, w = self.components(e), self.weights
        parts = {"D": w.d * c["D"], "T": w.t * c["T"], "F": w.f * c["F"],
                 "R": w.r * c["R"], "O": w.o * c["O"]}
        parts["total"] = sum(parts.values())
        return parts

    def path_cost(self, edge_ids: Iterable[int]) -> float:
        edges = self.graph.edges
        return sum(self.edge_cost(edges[i]) for i in edge_ids)

    def describe(self) -> dict:
        """Everything the report must state: weights + normalization refs."""
        w = self.weights
        return {"preset": self.preset_name,
                "weights": {"w_d": w.d, "w_t": w.t, "w_f": w.f, "w_r": w.r, "w_o": w.o},
                "reference_values": dict(self.refs)}

    # ---- A* heuristic -------------------------------------------------
    def _base_cost(self, e: Edge) -> float:
        """Cost of an edge in its BEST possible state (no restriction/obstacle)."""
        w, r = self.weights, self.refs
        fuel = e.fuel_cost * (1.0 + FUEL_DIFFICULTY_STEP * (e.difficulty - 1))
        return (w.d * (e.distance / r["D"])
                + w.t * (e.base_time / r["T"])
                + w.f * (fuel / r["F"])
                + w.r * ((e.difficulty - 1) / r["R"]))

    def heuristic_ratio(self) -> float:
        """Minimum cost-per-straight-line-distance over all edges.

        h(v) = ratio * euclid(v, target).  Admissible: any path's cost is
        sum cost(e) >= ratio * sum euclid(e) >= ratio * euclid(v, target)
        (triangle inequality).  It is also CONSISTENT, because
        h(u) <= ratio*euclid(u,v) + h(v) <= cost(u,v) + h(v).
        Effective costs are never below base costs (modifiers only worsen
        edges), so the bound stays valid while the track changes.
        If any usable-geometry edge has zero cost the ratio is 0 and A* falls
        back to h = 0 (behaves like Dijkstra).  A (1 - 1e-9) factor protects
        against floating-point rounding.
        """
        if self._ratio is not None and self._ratio_version == self.graph.structure_version:
            return self._ratio
        verts = self.graph.vertices
        best = math.inf
        for e in self.graph.edges.values():
            a, b = verts[e.frm], verts[e.to]
            length = math.hypot(a.x - b.x, a.y - b.y)
            if length > 0.0:
                best = min(best, self._base_cost(e) / length)
        self._ratio = 0.0 if best == math.inf else best * (1.0 - 1e-9)
        self._ratio_version = self.graph.structure_version
        return self._ratio
