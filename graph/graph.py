"""Track graph G=(V,E) stored as adjacency lists (Section 3.1).

Memory O(V+E); neighbour iteration is O(deg(v)).  Adjacency lists are used
instead of a matrix because road networks are sparse (matrix = O(V^2)).

Dynamic state (changed WITHOUT rebuilding the graph):
  * edge status: Open / Blocked / Restricted
  * per-edge effective modifiers (written by Member 2's event engine):
        time_factor      >= 1   (speed restriction: time is multiplied)
        extra_difficulty >= 0   (oil spill: difficulty raised, capped at 5)
        obstacle_penalty >= 0   (obstacle term O of the route cost)
    All modifiers can only make an edge WORSE (never cheaper than its base
    state).  This is what keeps the A* heuristic admissible while the track
    changes during a race.
  * service vertex availability (FuelStation / NitroPickup `active` flag)

Design decisions (document in the report):
  * Blocked edges are skipped by every traversal/search by default.
  * Restricted edges stay traversable; their effective time/fuel/difficulty
    already include the restriction.
  * An inactive service vertex is still a normal road vertex (cars may drive
    through it); only the SERVICE is unavailable (see is_service_available).
  * An undirected edge (directed=False) is usable in both directions.
"""
from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

# ---- vertex types -------------------------------------------------------
START = "Start"
FINISH = "Finish"
CHECKPOINT = "Checkpoint"
INTERSECTION = "Intersection"
FUEL_STATION = "FuelStation"
NITRO_PICKUP = "NitroPickup"
DEAD_END = "DeadEnd"
VERTEX_TYPES = (START, FINISH, CHECKPOINT, INTERSECTION,
                FUEL_STATION, NITRO_PICKUP, DEAD_END)
SERVICE_TYPES = (FUEL_STATION, NITRO_PICKUP)

# ---- edge statuses ------------------------------------------------------
OPEN = "Open"
BLOCKED = "Blocked"
RESTRICTED = "Restricted"
EDGE_STATUSES = (OPEN, BLOCKED, RESTRICTED)

MAX_DIFFICULTY = 5
# Section 4.3: fuel multiplier = 1.0 + 0.1 * (difficulty - 1).  Shared with
# Member 2; if the team re-tunes it, change it here ONLY.
FUEL_DIFFICULTY_STEP = 0.1
# time factor assumed when a JSON edge is "Restricted" without a timeFactor
DEFAULT_RESTRICTED_TIME_FACTOR = 1.5


def _check_number(name: str, value, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}, got {value!r}")
    return float(value)


@dataclass
class Vertex:
    id: int
    x: float
    y: float
    type: str = INTERSECTION
    checkpoint_order: Optional[int] = None
    active: bool = True

    @property
    def is_mandatory(self) -> bool:
        return self.checkpoint_order is not None

    @property
    def is_protected(self) -> bool:
        """Start, Finish and mandatory checkpoints are never 'dead ends'."""
        return self.is_mandatory or self.type in (START, FINISH)


@dataclass
class Edge:
    id: int
    frm: int
    to: int
    distance: float
    base_time: float
    fuel_cost: float
    difficulty: int = 1
    status: str = OPEN
    is_shortcut: bool = False
    directed: bool = False
    # dynamic effective modifiers (see module docstring)
    time_factor: float = 1.0
    extra_difficulty: int = 0
    obstacle_penalty: float = 0.0

    # ---- effective values used by cost / fuel / time calculations -------
    @property
    def effective_difficulty(self) -> int:
        return min(MAX_DIFFICULTY, self.difficulty + self.extra_difficulty)

    @property
    def fuel_multiplier(self) -> float:
        return 1.0 + FUEL_DIFFICULTY_STEP * (self.effective_difficulty - 1)

    @property
    def effective_time(self) -> float:
        return self.base_time * self.time_factor

    @property
    def effective_fuel(self) -> float:
        return self.fuel_cost * self.fuel_multiplier

    def other(self, u: int) -> int:
        return self.to if self.frm == u else self.frm

    def can_leave(self, u: int) -> bool:
        """True if a car standing on vertex u may start traversing this edge."""
        return u == self.frm or (not self.directed and u == self.to)

    @property
    def is_blocked(self) -> bool:
        return self.status == BLOCKED


class Graph:
    def __init__(self) -> None:
        self.vertices: Dict[int, Vertex] = {}
        self.edges: Dict[int, Edge] = {}
        self._adj: Dict[int, List[int]] = {}
        self.structure_version = 0   # bumped by add_vertex / add_edge
        self.version = 0             # bumped by ANY change (UI / caches)
        self._start_id: Optional[int] = None
        self._finish_id: Optional[int] = None
        self._cp_cache: Optional[Tuple[int, List[Vertex]]] = None

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def add_vertex(self, v: Vertex) -> Vertex:
        if v.id in self.vertices:
            raise ValueError(f"duplicate vertex id {v.id}")
        if v.type not in VERTEX_TYPES:
            raise ValueError(f"vertex {v.id}: unknown type {v.type!r}")
        _check_number(f"vertex {v.id} x", v.x, minimum=-math.inf)
        _check_number(f"vertex {v.id} y", v.y, minimum=-math.inf)
        if v.type == CHECKPOINT and v.checkpoint_order is None:
            raise ValueError(f"checkpoint vertex {v.id} needs a checkpointOrder")
        if v.checkpoint_order is not None:
            if not isinstance(v.checkpoint_order, int) or v.checkpoint_order < 1:
                raise ValueError(f"vertex {v.id}: checkpointOrder must be an int >= 1")
            if any(o.checkpoint_order == v.checkpoint_order
                   for o in self.vertices.values()):
                raise ValueError(
                    f"duplicate checkpointOrder {v.checkpoint_order} (vertex {v.id})")
        self.vertices[v.id] = v
        self._adj[v.id] = []
        if v.type == START and self._start_id is None:
            self._start_id = v.id
        if v.type == FINISH and self._finish_id is None:
            self._finish_id = v.id
        self.structure_version += 1
        self.version += 1
        return v

    def add_edge(self, e: Edge) -> Edge:
        if e.id in self.edges:
            raise ValueError(f"duplicate edge id {e.id}")
        for end in (e.frm, e.to):
            if end not in self.vertices:
                raise ValueError(f"edge {e.id}: unknown endpoint vertex {end}")
        if e.frm == e.to:
            raise ValueError(f"edge {e.id}: self-loops are not allowed")
        e.distance = _check_number(f"edge {e.id} distance", e.distance)
        e.base_time = _check_number(f"edge {e.id} baseTime", e.base_time)
        e.fuel_cost = _check_number(f"edge {e.id} fuelCost", e.fuel_cost)
        d = e.difficulty
        if isinstance(d, bool) or not isinstance(d, (int, float)) or d != int(d) \
                or not 1 <= int(d) <= MAX_DIFFICULTY:
            raise ValueError(f"edge {e.id}: difficulty must be an integer 1..{MAX_DIFFICULTY}")
        e.difficulty = int(d)
        if e.status not in EDGE_STATUSES:
            raise ValueError(f"edge {e.id}: unknown status {e.status!r}")
        e.time_factor = _check_number(f"edge {e.id} timeFactor", e.time_factor, minimum=1.0)
        e.obstacle_penalty = _check_number(
            f"edge {e.id} obstaclePenalty", e.obstacle_penalty)
        if int(e.extra_difficulty) != e.extra_difficulty or e.extra_difficulty < 0:
            raise ValueError(f"edge {e.id}: extraDifficulty must be an int >= 0")
        e.extra_difficulty = int(e.extra_difficulty)
        self.edges[e.id] = e
        self._adj[e.frm].append(e.id)
        if not e.directed:
            self._adj[e.to].append(e.id)
        self._refresh_status(e)
        self.structure_version += 1
        self.version += 1
        return e

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------
    def has_vertex(self, vid: int) -> bool:
        return vid in self.vertices

    def vertex(self, vid: int) -> Vertex:
        try:
            return self.vertices[vid]
        except KeyError:
            raise KeyError(f"unknown vertex id {vid}") from None

    def edge(self, eid: int) -> Edge:
        try:
            return self.edges[eid]
        except KeyError:
            raise KeyError(f"unknown edge id {eid}") from None

    def neighbors(self, u: int, include_blocked: bool = False
                  ) -> Iterator[Tuple[Edge, int]]:
        """Yield (edge, neighbour) for every legal move out of u.

        Directed edges are only yielded from their tail; Blocked edges are
        skipped unless include_blocked=True.  Order = edge insertion order,
        so every algorithm is deterministic.
        """
        edges = self.edges
        for eid in self._adj[u]:
            e = edges[eid]
            if e.status == BLOCKED and not include_blocked:
                continue
            yield e, (e.to if e.frm == u else e.frm)

    @property
    def start_id(self) -> Optional[int]:
        return self._start_id

    @property
    def finish_id(self) -> Optional[int]:
        return self._finish_id

    def checkpoints(self) -> List[Vertex]:
        """Mandatory checkpoints sorted by required order."""
        if self._cp_cache is None or self._cp_cache[0] != self.structure_version:
            cps = sorted((v for v in self.vertices.values() if v.is_mandatory),
                         key=lambda v: v.checkpoint_order)
            self._cp_cache = (self.structure_version, cps)
        return self._cp_cache[1]

    def vertices_of_type(self, vtype: str) -> List[Vertex]:
        return [v for v in self.vertices.values() if v.type == vtype]

    def is_service_available(self, vid: int) -> bool:
        v = self.vertex(vid)
        return v.type in SERVICE_TYPES and v.active

    # ------------------------------------------------------------------
    # dynamic updates (no rebuild)
    # ------------------------------------------------------------------
    @staticmethod
    def _refresh_status(e: Edge) -> None:
        if e.status == BLOCKED:
            return
        has_effect = (e.time_factor > 1.0 or e.extra_difficulty > 0
                      or e.obstacle_penalty > 0.0)
        e.status = RESTRICTED if has_effect else OPEN

    def block_edge(self, edge_id: int) -> None:
        self.edge(edge_id).status = BLOCKED
        self.version += 1

    def unblock_edge(self, edge_id: int) -> None:
        e = self.edge(edge_id)
        if e.status == BLOCKED:
            e.status = OPEN
            self._refresh_status(e)   # back to Restricted if effects remain
        self.version += 1

    def set_edge_effects(self, edge_id: int, *, time_factor: float = 1.0,
                         extra_difficulty: int = 0,
                         obstacle_penalty: float = 0.0) -> None:
        """Write the FINAL effective modifiers of an edge in one call.

        Member 2 combines all active events (e.g. strictest speed cap, max
        difficulty penalty) and writes the result here, so expiring one effect
        never erases another.  Arguments are absolute values, not increments.
        """
        e = self.edge(edge_id)
        tf = _check_number("timeFactor", time_factor, minimum=1.0)
        op = _check_number("obstaclePenalty", obstacle_penalty)
        if int(extra_difficulty) != extra_difficulty or extra_difficulty < 0:
            raise ValueError("extraDifficulty must be an int >= 0")
        e.time_factor, e.obstacle_penalty = tf, op
        e.extra_difficulty = int(extra_difficulty)
        self._refresh_status(e)
        self.version += 1

    def set_restricted(self, edge_id: int,
                       time_factor: float = DEFAULT_RESTRICTED_TIME_FACTOR) -> None:
        e = self.edge(edge_id)
        self.set_edge_effects(edge_id, time_factor=time_factor,
                              extra_difficulty=e.extra_difficulty,
                              obstacle_penalty=e.obstacle_penalty)

    def clear_restriction(self, edge_id: int) -> None:
        e = self.edge(edge_id)
        self.set_edge_effects(edge_id, time_factor=1.0,
                              extra_difficulty=e.extra_difficulty,
                              obstacle_penalty=e.obstacle_penalty)

    def clear_effects(self, edge_id: int) -> None:
        self.set_edge_effects(edge_id)

    def set_vertex_active(self, vertex_id: int, active: bool) -> None:
        v = self.vertex(vertex_id)
        if v.type not in SERVICE_TYPES:
            raise ValueError(
                f"vertex {vertex_id} is a {v.type}, not a service vertex")
        v.active = bool(active)
        self.version += 1

    # camelCase aliases = the names agreed in the group contract
    blockEdge = block_edge
    unblockEdge = unblock_edge
    setRestricted = set_restricted
    clearRestriction = clear_restriction
    setEdgeEffects = set_edge_effects
    setVertexActive = set_vertex_active

    # ------------------------------------------------------------------
    # JSON  (keys exactly as in Section 3.1)
    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict) -> "Graph":
        if not isinstance(data, dict) or "vertices" not in data or "edges" not in data:
            raise ValueError("track JSON needs 'vertices' and 'edges' lists")
        g = cls()
        for i, d in enumerate(data["vertices"]):
            missing = [k for k in ("id", "x", "y") if k not in d]
            if missing:
                raise ValueError(f"vertex #{i} missing {missing}")
            g.add_vertex(Vertex(
                id=d["id"], x=d["x"], y=d["y"],
                type=d.get("type", INTERSECTION),
                checkpoint_order=d.get("checkpointOrder"),
                active=d.get("active", True)))
        required = ("id", "from", "to", "distance", "baseTime", "fuelCost", "difficulty")
        for i, d in enumerate(data["edges"]):
            missing = [k for k in required if k not in d]
            if missing:
                raise ValueError(f"edge #{i} missing {missing}")
            status = d.get("status", OPEN)
            tf = d.get("timeFactor")
            if tf is None:
                tf = DEFAULT_RESTRICTED_TIME_FACTOR if (
                    status == RESTRICTED and not d.get("extraDifficulty")
                    and not d.get("obstaclePenalty")) else 1.0
            g.add_edge(Edge(
                id=d["id"], frm=d["from"], to=d["to"],
                distance=d["distance"], base_time=d["baseTime"],
                fuel_cost=d["fuelCost"], difficulty=d["difficulty"],
                status=status,
                is_shortcut=bool(d.get("isShortcut", False)),
                directed=bool(d.get("directed", False)),
                time_factor=tf,
                extra_difficulty=d.get("extraDifficulty", 0),
                obstacle_penalty=d.get("obstaclePenalty", 0.0)))
        return g

    def to_dict(self) -> dict:
        verts = [{"id": v.id, "x": v.x, "y": v.y, "type": v.type,
                  "checkpointOrder": v.checkpoint_order, "active": v.active}
                 for v in self.vertices.values()]
        edges = []
        for e in self.edges.values():
            d = {"id": e.id, "from": e.frm, "to": e.to, "distance": e.distance,
                 "baseTime": e.base_time, "fuelCost": e.fuel_cost,
                 "difficulty": e.difficulty, "status": e.status,
                 "isShortcut": e.is_shortcut, "directed": e.directed}
            if e.time_factor != 1.0:
                d["timeFactor"] = e.time_factor
            if e.extra_difficulty:
                d["extraDifficulty"] = e.extra_difficulty
            if e.obstacle_penalty:
                d["obstaclePenalty"] = e.obstacle_penalty
            edges.append(d)
        return {"vertices": verts, "edges": edges}

    def copy(self) -> "Graph":
        return copy.deepcopy(self)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def load_graph(source) -> Graph:
    """loadGraph(json): accepts a dict, a JSON string, or a file path."""
    if isinstance(source, Graph):
        return source
    if isinstance(source, dict):
        return Graph.from_dict(source)
    if isinstance(source, (str, os.PathLike)):
        text = str(source)
        if text.lstrip().startswith("{"):
            return Graph.from_dict(json.loads(text))
        with open(text, "r", encoding="utf-8") as f:
            return Graph.from_dict(json.load(f))
    raise TypeError(f"cannot load a graph from {type(source).__name__}")
