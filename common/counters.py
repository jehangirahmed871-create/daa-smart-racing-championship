"""Operation counters shared by every algorithm (the agreed counter format).

Definitions (state these in the report):
  expanded        vertices taken off the queue/heap/stack and processed
  relaxed         edge relaxations that IMPROVED a label (Dijkstra / A*)
  examinedEdges   edges looked at, whether or not they improved anything
  heapOps         heap pushes + heap pops
  visited         vertices discovered (marked visited / given a label)
  reopened        A* only: an already-closed vertex expanded again
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class Counters:
    expanded: int = 0
    relaxed: int = 0
    heap_ops: int = 0
    visited: int = 0
    examined_edges: int = 0
    reopened: int = 0
    path_cost: Optional[float] = None
    time_ms: float = 0.0

    def to_dict(self):
        return {
            "expanded": self.expanded,
            "relaxed": self.relaxed,
            "heapOps": self.heap_ops,
            "visited": self.visited,
            "examinedEdges": self.examined_edges,
            "reopened": self.reopened,
            "pathCost": self.path_cost,
            "timeMs": self.time_ms,
        }

    def merge(self, other: "Counters") -> None:
        """Add another counter set into this one (used to total route legs)."""
        self.expanded += other.expanded
        self.relaxed += other.relaxed
        self.heap_ops += other.heap_ops
        self.visited += other.visited
        self.examined_edges += other.examined_edges
        self.reopened += other.reopened
        self.time_ms += other.time_ms
