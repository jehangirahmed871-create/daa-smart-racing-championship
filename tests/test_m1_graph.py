import os
import unittest

from common.heap import MinHeap
from graph import Edge, Graph, Vertex, loadGraph, BLOCKED, OPEN, RESTRICTED
from graph.graph import MAX_DIFFICULTY
from tests.m1_helpers import build

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class HeapTests(unittest.TestCase):
    def test_orders_and_counts(self):
        h = MinHeap()
        for x in [5, 3, 8, 1, 9, 2, 2]:
            h.push((x,))
        out = [h.pop()[0] for _ in range(7)]
        self.assertEqual(out, [1, 2, 2, 3, 5, 8, 9])
        self.assertEqual((h.pushes, h.pops, h.ops), (7, 7, 14))
        with self.assertRaises(IndexError):
            h.pop()

    def test_tuple_key_ties_are_deterministic(self):
        h = MinHeap()
        for item in [(1.0, 2, 1, "c"), (1.0, 1, 2, "b"), (1.0, 1, 0, "a"), (0.5, 9, 3, "z")]:
            h.push(item)
        self.assertEqual([h.pop()[3] for _ in range(4)], ["z", "a", "b", "c"])


class GraphModelTests(unittest.TestCase):
    def test_single_vertex(self):
        g = build([(0, 0, 0)], [])
        self.assertEqual(list(g.neighbors(0)), [])

    def test_undirected_and_directed_neighbors(self):
        g = build([(0, 0, 0), (1, 1, 0), (2, 2, 0)],
                  [(0, 0, 1, 1), (1, 1, 2, 1, 1, 1, 1, True)])
        self.assertEqual([v for _, v in g.neighbors(1)], [0, 2])   # undirected back + one-way out
        self.assertEqual([v for _, v in g.neighbors(2)], [])        # one-way cannot be reversed

    def test_blocked_edges_excluded_unless_requested(self):
        g = build([(0, 0, 0), (1, 1, 0)], [(0, 0, 1, 1)])
        g.block_edge(0)
        self.assertEqual(g.edge(0).status, BLOCKED)
        self.assertEqual(list(g.neighbors(0)), [])
        self.assertEqual(len(list(g.neighbors(0, include_blocked=True))), 1)
        g.unblock_edge(0)
        self.assertEqual(g.edge(0).status, OPEN)

    def test_restricted_stays_traversable_with_worse_values(self):
        g = build([(0, 0, 0), (1, 1, 0)], [(0, 0, 1, 10, 10, 10, 2)])
        e = g.edge(0)
        base = (e.effective_time, e.effective_fuel)
        g.set_restricted(0, 2.0)
        self.assertEqual(e.status, RESTRICTED)
        self.assertEqual(len(list(g.neighbors(0))), 1)
        self.assertEqual(e.effective_time, base[0] * 2)
        g.set_edge_effects(0, extra_difficulty=10)         # capped at 5
        self.assertEqual(e.effective_difficulty, MAX_DIFFICULTY)
        self.assertAlmostEqual(e.fuel_multiplier, 1.4)
        g.clear_effects(0)
        self.assertEqual(e.status, OPEN)

    def test_blocked_edge_keeps_effects_after_unblock(self):
        g = build([(0, 0, 0), (1, 1, 0)], [(0, 0, 1, 10)])
        g.set_restricted(0, 3.0)
        g.block_edge(0)
        g.unblock_edge(0)
        self.assertEqual(g.edge(0).status, RESTRICTED)
        self.assertEqual(g.edge(0).time_factor, 3.0)

    def test_service_vertex_toggle(self):
        g = Graph()
        g.add_vertex(Vertex(0, 0, 0, "FuelStation"))
        g.add_vertex(Vertex(1, 1, 0))
        self.assertTrue(g.is_service_available(0))
        g.set_vertex_active(0, False)
        self.assertFalse(g.is_service_available(0))
        with self.assertRaises(ValueError):
            g.set_vertex_active(1, False)      # not a service vertex
        with self.assertRaises(KeyError):
            g.set_vertex_active(99, True)      # missing target

    def test_validation_errors(self):
        g = build([(0, 0, 0), (1, 1, 0)], [])
        bad = [Edge(0, 0, 1, -1, 1, 1), Edge(0, 0, 5, 1, 1, 1), Edge(0, 0, 0, 1, 1, 1),
               Edge(0, 0, 1, 1, 1, 1, difficulty=6), Edge(0, 0, 1, float("nan"), 1, 1),
               Edge(0, 0, 1, 1, 1, 1, status="Weird")]
        for e in bad:
            with self.assertRaises(ValueError):
                g.add_edge(e)
        g.add_edge(Edge(0, 0, 1, 1, 1, 1))
        with self.assertRaises(ValueError):
            g.add_edge(Edge(0, 0, 1, 1, 1, 1))            # duplicate edge id
        with self.assertRaises(ValueError):
            g.set_edge_effects(0, time_factor=0.5)        # effects may only worsen

    def test_checkpoint_rules(self):
        g = Graph()
        g.add_vertex(Vertex(0, 0, 0, "Checkpoint", 1))
        with self.assertRaises(ValueError):
            g.add_vertex(Vertex(1, 0, 0, "Checkpoint", 1))      # duplicate order
        with self.assertRaises(ValueError):
            g.add_vertex(Vertex(2, 0, 0, "Checkpoint", None))   # missing order

    def test_json_round_trip_and_loader_inputs(self):
        path = os.path.join(ROOT, "data", "sample_circuit.json")
        g = loadGraph(path)
        g2 = Graph.from_dict(g.to_dict())
        self.assertEqual(g.to_dict(), g2.to_dict())
        self.assertEqual(loadGraph(g.to_dict()).to_dict(), g.to_dict())
        with self.assertRaises(ValueError):
            loadGraph({"vertices": [{"id": 0, "x": 0, "y": 0}], "edges": [{"id": 0}]})

    def test_sample_circuit_meets_spec_minimums(self):
        g = loadGraph(os.path.join(ROOT, "data", "sample_circuit.json"))
        self.assertGreaterEqual(len(g.checkpoints()), 10)
        self.assertGreaterEqual(sum(e.is_shortcut for e in g.edges.values()), 2)
        self.assertTrue(g.vertices_of_type("FuelStation"))
        self.assertTrue(g.vertices_of_type("NitroPickup"))
        self.assertTrue(g.vertices_of_type("DeadEnd"))


if __name__ == "__main__":
    unittest.main()
