import os
import unittest

from graph import Navigator, loadGraph
from graph.traversal import (alternative_routes, bfs, dfs, find_dead_ends,
                             has_cycle, reachability_report, weak_components,
                             bfs_steps, dfs_steps)
from common.steps import run_to_completion
from tests.m1_helpers import build, random_graph, unit_bfs_reference

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class BFSTests(unittest.TestCase):
    def test_min_edges_not_min_distance(self):
        # spec example: S-T length 10 (1 edge) vs S-A-B-T three edges of length 2
        g = build([(0, 0, 0), (1, 1, 0), (2, 2, 0), (3, 3, 0)],
                  [(0, 0, 3, 10), (1, 0, 1, 2), (2, 1, 2, 2), (3, 2, 3, 2)])
        r = bfs(g, 0, 3)
        self.assertEqual(r.path_vertices, [0, 3])
        self.assertEqual(r.counters.path_cost, 1.0)

    def test_single_vertex_and_source_equals_target(self):
        g = build([(0, 0, 0)], [])
        r = bfs(g, 0, 0)
        self.assertTrue(r.found)
        self.assertEqual((r.path_vertices, r.path_edges), ([0], []))
        self.assertEqual(bfs(g, 0).reachable, {0})

    def test_unreachable_target_no_crash(self):
        g = build([(0, 0, 0), (1, 1, 0), (2, 2, 0)], [(0, 0, 1, 1)])
        r = bfs(g, 0, 2)
        self.assertFalse(r.found)
        self.assertEqual(r.path_vertices, [])
        with self.assertRaises(ValueError):
            bfs(g, 0, 99)

    def test_directed_edges_respected(self):
        g = build([(0, 0, 0), (1, 1, 0)], [(0, 0, 1, 1, 1, 1, 1, True)])
        self.assertTrue(bfs(g, 0, 1).found)
        self.assertFalse(bfs(g, 1, 0).found)

    def test_blocked_edge_breaks_reachability(self):
        g = build([(0, 0, 0), (1, 1, 0), (2, 2, 0)], [(0, 0, 1, 1), (1, 1, 2, 1)])
        self.assertTrue(bfs(g, 0, 2).found)
        g.block_edge(1)
        self.assertFalse(bfs(g, 0, 2).found)

    def test_matches_reference_on_random_graphs(self):
        for seed in range(25):
            g = random_graph(40, 30, seed, directed_prob=0.3)
            ref = unit_bfs_reference(g, 0)
            res = bfs(g, 0)
            for v in g.vertices:
                if ref[v] == float("inf"):
                    self.assertNotIn(v, res.dist)
                else:
                    self.assertEqual(res.dist[v], ref[v])

    def test_reachability_report(self):
        g = loadGraph(os.path.join(ROOT, "data", "sample_circuit.json"))
        rep = reachability_report(g, g.start_id)
        self.assertEqual(rep["unreachable"], [])
        self.assertEqual(len(rep["reachable"]), 11)   # 10 checkpoints + finish


class DFSTests(unittest.TestCase):
    def test_cycle_detection(self):
        tri = build([(0, 0, 0), (1, 1, 0), (2, 0, 1)], [(0, 0, 1, 1), (1, 1, 2, 1), (2, 2, 0, 1)])
        self.assertTrue(has_cycle(tri))
        path = build([(0, 0, 0), (1, 1, 0), (2, 2, 0)], [(0, 0, 1, 1), (1, 1, 2, 1)])
        self.assertFalse(has_cycle(path))
        single = build([(0, 0, 0)], [])
        self.assertFalse(has_cycle(single))

    def test_parallel_roads_form_a_cycle_but_one_road_does_not(self):
        one = build([(0, 0, 0), (1, 1, 0)], [(0, 0, 1, 1)])
        two = build([(0, 0, 0), (1, 1, 0)], [(0, 0, 1, 1), (1, 0, 1, 2)])
        self.assertFalse(has_cycle(one))
        self.assertTrue(has_cycle(two))

    def test_directed_cycle_vs_dag(self):
        cyc = build([(0, 0, 0), (1, 1, 0), (2, 2, 0)],
                    [(0, 0, 1, 1, 1, 1, 1, True), (1, 1, 2, 1, 1, 1, 1, True),
                     (2, 2, 0, 1, 1, 1, 1, True)])
        dag = build([(0, 0, 0), (1, 1, 0), (2, 2, 0)],
                    [(0, 0, 1, 1, 1, 1, 1, True), (1, 1, 2, 1, 1, 1, 1, True),
                     (2, 0, 2, 1, 1, 1, 1, True)])
        self.assertTrue(has_cycle(cyc))
        self.assertFalse(has_cycle(dag))

    def test_blocking_an_edge_removes_the_cycle(self):
        tri = build([(0, 0, 0), (1, 1, 0), (2, 0, 1)], [(0, 0, 1, 1), (1, 1, 2, 1), (2, 2, 0, 1)])
        tri.block_edge(2)
        self.assertFalse(has_cycle(tri))

    def test_traversal_visits_each_reachable_vertex_once(self):
        g = random_graph(60, 40, 7, directed_prob=0.0)
        r = dfs(g, 0)
        self.assertEqual(len(r.order), len(set(r.order)))
        self.assertEqual(set(r.order), bfs(g, 0).reachable)
        self.assertTrue(all(r.discovery[v] < r.finish[v] for v in r.order))

    def test_deep_path_has_no_recursion_limit(self):
        n = 6000
        g = build([(i, i, 0) for i in range(n)], [(i, i, i + 1, 1) for i in range(n - 1)])
        self.assertEqual(len(dfs(g, 0).order), n)
        self.assertEqual(bfs(g, 0, n - 1).counters.path_cost, n - 1)

    def test_components(self):
        g = build([(i, i, 0) for i in range(6)], [(0, 0, 1, 1), (1, 1, 2, 1), (2, 3, 4, 1)])
        self.assertEqual(weak_components(g), [[0, 1, 2], [3, 4], [5]])
        g.block_edge(1)
        self.assertEqual(weak_components(g), [[0, 1], [2], [3, 4], [5]])

    def test_dfs_vs_bfs_component_count_random(self):
        for seed in range(15):
            g = random_graph(30, 5, seed, directed_prob=0.0)
            for e in list(g.edges.values())[::5]:
                g.block_edge(e.id)
            comps = weak_components(g)
            n_bfs = 0
            seen = set()
            for v in g.vertices:
                if v not in seen:
                    seen |= bfs(g, v).reachable
                    n_bfs += 1
            self.assertEqual(len(comps), n_bfs)


class DeadEndTests(unittest.TestCase):
    def test_spur_is_dead_end_but_checkpoint_spur_is_not(self):
        # ring 0-1-2-3-0 with finish=3, spur 2-4-5 (non-mandatory) and 1-6 (checkpoint tip)
        g = build([(0, 0, 0, "Start"), (1, 1, 0), (2, 2, 0), (3, 3, 0, "Finish"),
                   (4, 4, 4), (5, 5, 5, "DeadEnd"), (6, 6, 6, "Checkpoint", 1)],
                  [(0, 0, 1, 1), (1, 1, 2, 1), (2, 2, 3, 1), (3, 3, 0, 1),
                   (4, 2, 4, 1), (5, 4, 5, 1), (6, 1, 6, 1)])
        de = find_dead_ends(g)
        self.assertEqual(set(de), {4, 5})
        self.assertTrue(all(r == "cul-de-sac" for r in de.values()))

    def test_one_way_trap_cannot_reach_finish(self):
        # vertex 3 can be entered (one-way from 1) but never left
        g2 = build([(0, 0, 0, "Start"), (1, 1, 0), (2, 2, 0, "Finish"), (3, 1, 1)],
                   [(0, 0, 1, 1), (1, 1, 2, 1), (2, 1, 3, 1, 1, 1, 1, True)])
        self.assertEqual(find_dead_ends(g2), {3: "cul-de-sac"})
        g3 = build([(0, 0, 0, "Start"), (1, 1, 0), (2, 2, 0, "Finish"), (3, 1, 1), (4, 0, 1)],
                   [(0, 0, 1, 1), (1, 1, 2, 1), (2, 1, 3, 1, 1, 1, 1, True),
                    (3, 3, 4, 1, 1, 1, 1, True), (4, 4, 0, 1, 1, 1, 1, True)])
        # 3->4->0 loops back: finish reachable, so no dead end
        self.assertEqual(find_dead_ends(g3), {})

    def test_closure_creates_dead_end(self):
        g = build([(0, 0, 0, "Start"), (1, 1, 0), (2, 2, 0, "Finish"), (3, 1, 1)],
                  [(0, 0, 1, 1), (1, 1, 2, 1), (2, 1, 3, 1), (3, 0, 3, 1)])
        self.assertEqual(find_dead_ends(g), {})            # 0-1-3 cycle -> 3 not a dead end
        g.block_edge(3)
        self.assertEqual(set(find_dead_ends(g)), {3})

    def test_sample_circuit_spur(self):
        g = loadGraph(os.path.join(ROOT, "data", "sample_circuit.json"))
        self.assertEqual(set(find_dead_ends(g)), {16, 17})


class AlternativeRouteTests(unittest.TestCase):
    def setUp(self):
        # square 0-1-3 / 0-2-3 plus diagonal 1-2
        self.g = build([(0, 0, 0), (1, 1, 0), (2, 0, 1), (3, 1, 1)],
                       [(0, 0, 1, 1), (1, 1, 3, 1), (2, 0, 2, 1), (3, 2, 3, 1), (4, 1, 2, 1)])

    def test_enumerates_simple_paths_excluding_main(self):
        res = alternative_routes(self.g, 0, 3, main_edges=[0, 1])
        paths = {tuple(r.vertices) for r in res.routes}
        self.assertEqual(paths, {(0, 2, 3), (0, 1, 2, 3), (0, 2, 1, 3)})
        for r in res.routes:
            self.assertEqual(len(r.vertices), len(set(r.vertices)))   # simple
            self.assertNotEqual(r.edges, [0, 1])

    def test_route_count_limit(self):
        res = alternative_routes(self.g, 0, 3, main_edges=[0, 1], max_routes=2)
        self.assertEqual(len(res.routes), 2)
        self.assertTrue(res.truncated)
        self.assertEqual(res.reason, "max_routes")

    def test_expansion_limit_stops_exponential_blowup(self):
        n = 14   # complete graph: ~ n! simple paths
        g = build([(i, i, 0) for i in range(n)],
                  [(k, i, j, 1) for k, (i, j) in enumerate(
                      (i, j) for i in range(n) for j in range(i + 1, n))])
        res = alternative_routes(g, 0, n - 1, max_routes=10 ** 9, max_expansions=5000)
        self.assertTrue(res.truncated)
        self.assertEqual(res.reason, "max_expansions")

    def test_length_limit_and_unreachable(self):
        res = alternative_routes(self.g, 0, 3, main_edges=[0, 1], max_edges=2)
        self.assertEqual({tuple(r.vertices) for r in res.routes}, {(0, 2, 3)})
        g = build([(0, 0, 0), (1, 1, 0)], [])
        self.assertEqual(alternative_routes(g, 0, 1).routes, [])

    def test_blocked_edge_removes_alternative(self):
        self.g.block_edge(3)
        paths = {tuple(r.vertices) for r in alternative_routes(self.g, 0, 3, [0, 1]).routes}
        # road 2-3 closed: only 0-2-1-3 remains besides the main route 0-1-3
        self.assertEqual(paths, {(0, 2, 1, 3)})
        for r in alternative_routes(self.g, 0, 3, [0, 1]).routes:
            self.assertNotIn(3, r.edges)


class StepHookTests(unittest.TestCase):
    def test_steps_and_plain_call_agree(self):
        g = random_graph(30, 20, 3, directed_prob=0.0)
        gen = bfs_steps(g, 0, 17, emit=True)
        steps = []
        try:
            while True:
                steps.append(next(gen))
        except StopIteration as stop:
            via_steps = stop.value
        plain = bfs(g, 0, 17)
        self.assertEqual(via_steps.path_vertices, plain.path_vertices)
        self.assertTrue(steps and steps[0]["op"] == "discover")
        d = run_to_completion(dfs_steps(g, 0, emit=False))
        self.assertEqual(d.order, dfs(g, 0).order)


if __name__ == "__main__":
    unittest.main()
