import math
import os
import unittest

from graph import Navigator, loadGraph, CostModel, PRESETS, Weights
from graph.routing import (astar, astar_steps, dijkstra, dijkstra_steps,
                           plan_route, shortest_path)
from common.steps import run_to_completion
from tests.m1_helpers import bellman_ford, build, random_graph

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CIRCUIT = os.path.join(ROOT, "data", "sample_circuit.json")
DEMO = os.path.join(ROOT, "data", "demo_presets.json")


class CostTests(unittest.TestCase):
    def test_weights_must_be_non_negative(self):
        with self.assertRaises(ValueError):
            Weights(-0.1, 0, 0, 0, 0)
        with self.assertRaises(ValueError):
            Weights(float("nan"), 0, 0, 0, 0)

    def test_costs_non_negative_and_displayed_equals_optimised(self):
        g = random_graph(40, 40, 1)
        for preset in PRESETS:
            m = CostModel(g, preset)
            self.assertTrue(all(m.edge_cost(e) >= 0 for e in g.edges.values()))
            r = dijkstra(g, 0, 39, m)
            if r.found:
                self.assertAlmostEqual(r.cost, m.path_cost(r.edges), places=12)
                self.assertAlmostEqual(r.cost, sum(m.breakdown(g.edges[i])["total"] for i in r.edges), places=12)

    def test_normalisation_refs_are_static(self):
        g = random_graph(20, 20, 2)
        m = CostModel(g, "Balanced")
        refs = dict(m.refs)
        g.block_edge(0)
        g.set_restricted(1, 3.0)
        self.assertEqual(CostModel(g, "Balanced").refs, refs)

    def test_restriction_and_obstacle_raise_cost(self):
        g = build([(0, 0, 0), (1, 10, 0)], [(0, 0, 1, 10, 10, 10, 1)])
        m = CostModel(g, "Balanced")
        base = m.edge_cost(g.edge(0))
        g.set_restricted(0, 2.0)
        c1 = m.edge_cost(g.edge(0))
        g.set_edge_effects(0, time_factor=2.0, extra_difficulty=2, obstacle_penalty=0.5)
        c2 = m.edge_cost(g.edge(0))
        self.assertGreater(c1, base)
        self.assertGreater(c2, c1)


class DijkstraTests(unittest.TestCase):
    def test_optimal_vs_bellman_ford_random(self):
        for seed in range(40):
            g = random_graph(35, 45, seed, directed_prob=0.3, zero_prob=0.1)
            for preset in ("Balanced", "Speed", "Economy", "Safety", "Distance"):
                m = CostModel(g, preset)
                ref = bellman_ford(g, m, 0)
                res = dijkstra(g, 0, None, m)
                for v in g.vertices:
                    got = res.dist_map.get(v, math.inf)
                    self.assertAlmostEqual(got, ref[v], places=9) if ref[v] != math.inf \
                        else self.assertEqual(got, math.inf)

    def test_source_equals_target(self):
        g = build([(0, 0, 0)], [])
        r = shortest_path(g, 0, 0)
        self.assertTrue(r.found)
        self.assertEqual((r.vertices, r.edges, r.cost), ([0], [], 0.0))

    def test_unreachable_and_bad_ids(self):
        g = build([(0, 0, 0), (1, 5, 0), (2, 9, 0)], [(0, 0, 1, 5)])
        r = shortest_path(g, 0, 2)
        self.assertFalse(r.found)
        self.assertEqual(r.cost, math.inf)
        self.assertEqual(r.vertices, [])
        with self.assertRaises(ValueError):
            shortest_path(g, 0, 77)

    def test_blocked_edge_excluded_and_recalculation(self):
        g = build([(0, 0, 0), (1, 5, 5), (2, 10, 0)],
                  [(0, 0, 2, 10), (1, 0, 1, 8), (2, 1, 2, 8)])
        nav = Navigator(g)
        self.assertEqual(nav.shortestPath(0, 2, "Distance").edges, [0])
        nav.blockEdge(0)
        r = nav.shortestPath(0, 2, "Distance")           # immediately recalculated
        self.assertEqual(r.edges, [1, 2])
        self.assertNotIn(0, r.edges)
        nav.unblockEdge(0)
        self.assertEqual(nav.shortestPath(0, 2, "Distance").edges, [0])
        nav.blockEdge(0); nav.blockEdge(1)
        self.assertFalse(nav.shortestPath(0, 2, "Distance").found)

    def test_restriction_reroutes(self):
        g = build([(0, 0, 0), (1, 5, 5), (2, 10, 0)],
                  [(0, 0, 2, 10, 10, 1), (1, 0, 1, 8, 9, 1), (2, 1, 2, 8, 9, 1)])
        nav = Navigator(g)
        self.assertEqual(nav.shortestPath(0, 2, "Speed").edges, [0])
        nav.setRestricted(0, 5.0)                        # direct road becomes slow
        self.assertEqual(nav.shortestPath(0, 2, "Speed").edges, [1, 2])

    def test_equal_cost_paths_are_deterministic(self):
        g = build([(0, 0, 0), (1, 1, 1), (2, 1, -1), (3, 2, 0)],
                  [(0, 0, 1, 2), (1, 1, 3, 2), (2, 0, 2, 2), (3, 2, 3, 2)])
        a = shortest_path(g, 0, 3, "Distance")
        b = shortest_path(g, 0, 3, "Distance")
        self.assertEqual(a.edges, b.edges)
        self.assertAlmostEqual(a.cost, shortest_path(g, 0, 3, "Distance", "astar").cost)

    def test_zero_weight_edges(self):
        g = build([(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 5, 0)],
                  [(0, 0, 1, 0, 0, 0), (1, 1, 2, 0, 0, 0), (2, 2, 3, 5, 5, 5), (3, 0, 3, 9, 9, 9)])
        for algo in ("dijkstra", "astar"):
            r = shortest_path(g, 0, 3, "Distance", algo)
            self.assertEqual(r.edges, [0, 1, 2])
            self.assertAlmostEqual(r.cost, CostModel(g, "Distance").path_cost([0, 1, 2]))

    def test_long_route(self):
        n = 3000
        g = build([(i, i, 0) for i in range(n)], [(i, i, i + 1, 1) for i in range(n - 1)])
        for algo in ("dijkstra", "astar"):
            r = shortest_path(g, 0, n - 1, "Distance", algo)
            self.assertEqual(len(r.edges), n - 1)

    def test_directed_edges(self):
        g = build([(0, 0, 0), (1, 1, 0)], [(0, 0, 1, 1, 1, 1, 1, True)])
        self.assertTrue(shortest_path(g, 0, 1).found)
        self.assertFalse(shortest_path(g, 1, 0).found)


class AStarTests(unittest.TestCase):
    def test_same_cost_as_dijkstra_random(self):
        for seed in range(40):
            g = random_graph(45, 60, seed, directed_prob=0.25)
            for preset in PRESETS:
                m = CostModel(g, preset)
                for dst in (7, 22, 44):
                    d = dijkstra(g, 0, dst, m)
                    a = astar(g, 0, dst, m)
                    self.assertEqual(d.found, a.found)
                    if d.found:
                        self.assertAlmostEqual(d.cost, a.cost, places=9)

    def test_same_cost_after_dynamic_changes(self):
        g = random_graph(50, 70, 5, directed_prob=0.2)
        m = CostModel(g, "Balanced")
        for eid in (0, 3, 7, 11):
            g.block_edge(eid)
        for eid in (2, 5):
            g.set_edge_effects(eid, time_factor=3.0, extra_difficulty=2, obstacle_penalty=0.8)
        for dst in (10, 30, 49):
            d, a = dijkstra(g, 0, dst, m), astar(g, 0, dst, m)
            self.assertEqual(d.found, a.found)
            if d.found:
                self.assertAlmostEqual(d.cost, a.cost, places=9)

    def test_heuristic_is_admissible_and_consistent(self):
        for seed in range(15):
            g = random_graph(40, 50, seed, directed_prob=0.2)
            for preset in PRESETS:
                m = CostModel(g, preset)
                ratio = m.heuristic_ratio()
                tgt = 17
                tv = g.vertices[tgt]

                def h(v):
                    p = g.vertices[v]
                    return ratio * math.hypot(p.x - tv.x, p.y - tv.y)

                for u in g.vertices:        # consistency: h(u) <= cost(u,v) + h(v)
                    for e, v in g.neighbors(u):
                        self.assertLessEqual(h(u), m.edge_cost(e) + h(v) + 1e-9)
                # admissibility: h(v) <= true cost to target
                for v in g.vertices:
                    r = dijkstra(g, v, tgt, m)
                    if r.found:
                        self.assertLessEqual(h(v), r.cost + 1e-9)

    def test_no_reopening_with_consistent_heuristic(self):
        g = random_graph(80, 120, 9, directed_prob=0.0)
        m = CostModel(g, "Balanced")
        self.assertEqual(astar(g, 0, 60, m).counters.reopened, 0)

    def test_zero_ratio_falls_back_to_dijkstra_behaviour(self):
        g = build([(0, 0, 0), (1, 5, 0), (2, 10, 0)],
                  [(0, 0, 1, 0, 0, 0), (1, 1, 2, 5, 5, 5)])
        m = CostModel(g, "Distance")
        self.assertEqual(m.heuristic_ratio(), 0.0)
        self.assertAlmostEqual(astar(g, 0, 2, m).cost, dijkstra(g, 0, 2, m).cost)

    def test_inconsistent_but_admissible_heuristic_still_optimal(self):
        # S=0, A=1, V=2, G=3 (one-way roads).  S->V 2, S->A 1, A->V 0.5, V->G 1.
        # With the Distance preset costs are distance/2.  The hand-made h below is
        # admissible but NOT consistent (h(A)=0.75 > cost(A,V)+h(V)=0.25), so V is
        # closed with a worse g first and must be reopened when A finds a cheaper way.
        g = build([(0, 0, 0), (1, 1, 0), (2, 2, 0), (3, 3, 0)],
                  [(0, 0, 2, 2, 2, 2, 1, True), (1, 0, 1, 1, 1, 1, 1, True),
                   (2, 1, 2, 0.5, 0.5, 0.5, 1, True), (3, 2, 3, 1, 1, 1, 1, True)])
        m = CostModel(g, "Distance")
        hv = {0: 0.0, 1: 0.75, 2: 0.0, 3: 0.0}
        r = astar(g, 0, 3, m, heuristic=lambda gr, v, d: hv[v])
        d = dijkstra(g, 0, 3, m)
        self.assertAlmostEqual(r.cost, d.cost, places=12)       # still optimal
        self.assertEqual(r.edges, [1, 2, 3])
        self.assertGreaterEqual(r.counters.reopened, 1)         # reopening happened

    def test_unreachable_and_same_vertex(self):
        g = build([(0, 0, 0), (1, 5, 0), (2, 9, 0)], [(0, 0, 1, 5)])
        m = CostModel(g)
        self.assertFalse(astar(g, 0, 2, m).found)
        self.assertEqual(astar(g, 1, 1, m).cost, 0.0)

    def test_counters_present_and_sane(self):
        g = random_graph(60, 80, 4, directed_prob=0.0)
        m = CostModel(g)
        for r in (dijkstra(g, 0, 50, m), astar(g, 0, 50, m)):
            c = r.counters
            self.assertGreater(c.expanded, 0)
            self.assertGreaterEqual(c.examined_edges, c.relaxed)
            self.assertGreaterEqual(c.heap_ops, 2)
            self.assertEqual(c.path_cost, r.cost)
            self.assertGreaterEqual(c.time_ms, 0.0)
        self.assertEqual(set(dijkstra(g, 0, 50, m).counters.to_dict()),
                         {"expanded", "relaxed", "heapOps", "visited", "examinedEdges",
                          "reopened", "pathCost", "timeMs"})


class PlanRouteTests(unittest.TestCase):
    def setUp(self):
        self.g = loadGraph(CIRCUIT)
        self.nav = Navigator(self.g)

    def _order_ok(self, route, first_cp=1):
        cps = [v.id for v in self.g.checkpoints() if v.checkpoint_order >= first_cp]
        it = iter(route.vertices)
        return all(any(v == cp for v in it) for cp in cps)      # ordered subsequence

    def test_visits_all_checkpoints_in_order_then_finish(self):
        for algo in ("dijkstra", "astar"):
            r = self.nav.planRouteFrom(self.g.start_id, 1, "Balanced", algo)
            self.assertTrue(r.found)
            self.assertTrue(self._order_ok(r))
            self.assertEqual(r.vertices[-1], self.g.finish_id)
            self.assertEqual(len(r.legs), 11)

    def test_both_algorithms_same_total_cost(self):
        for preset in PRESETS:
            a = self.nav.planRouteFrom(0, 1, preset, "dijkstra")
            b = self.nav.planRouteFrom(0, 1, preset, "astar")
            self.assertAlmostEqual(a.cost, b.cost, places=9)

    def test_plan_from_mid_race_state(self):
        r = self.nav.planRoute({"currentVertex": 6, "nextCheckpoint": 7})
        self.assertTrue(r.found)
        self.assertEqual(r.vertices[0], 6)
        self.assertTrue(self._order_ok(r, 7))

        class Racer:                      # attribute style racer
            current_vertex = 3
            next_checkpoint = 4
        self.assertTrue(self.nav.planRoute(Racer()).found)

    def test_closure_forces_detour_but_keeps_order(self):
        before = self.nav.planRouteFrom(0, 1)
        used = set(before.edges)
        victim = sorted(used)[3]
        self.nav.blockEdge(victim)
        after = self.nav.planRouteFrom(0, 1)                     # recalculated, no restart
        self.assertTrue(after.found)
        self.assertNotIn(victim, after.edges)
        self.assertTrue(self._order_ok(after))

    def test_unreachable_checkpoint_reports_failed_leg(self):
        cp5 = [v.id for v in self.g.checkpoints() if v.checkpoint_order == 5][0]
        for e in list(self.g.edges.values()):
            if cp5 in (e.frm, e.to):               # close every road touching cp5
                self.g.block_edge(e.id)
        r = self.nav.planRouteFrom(0, 1)
        self.assertFalse(r.found)
        self.assertIsNotNone(r.failed_leg)
        self.assertEqual(r.failed_leg[1], cp5)
        self.assertEqual(r.cost, math.inf)

    def test_displayed_cost_is_optimised_cost(self):
        r = self.nav.planRouteFrom(0, 1, "Safety")
        self.assertAlmostEqual(r.cost, self.nav.model("Safety").path_cost(r.edges), places=9)
        self.assertAlmostEqual(r.cost, sum(l.cost for l in r.legs), places=9)


class PresetTests(unittest.TestCase):
    def test_different_presets_choose_different_routes(self):
        nav = Navigator(loadGraph(DEMO))
        res = nav.comparePresets(0, 5)
        self.assertEqual(res["distinct_routes"], 4)
        expect = {"Speed": [0, 1, 5], "Safety": [0, 2, 5], "Economy": [0, 3, 5], "Balanced": [0, 4, 5]}
        for name, verts in expect.items():
            self.assertEqual(res["routes"][name].vertices, verts)
        # each preset's own route is the cheapest under its own weights
        cross = res["cost_under_each_preset"]
        for name in PRESETS:
            self.assertEqual(min(cross, key=lambda r: cross[r][name]), name)

    def test_compare_routes_metrics(self):
        nav = Navigator(loadGraph(DEMO))
        mets = nav.compareRoutes([[0, 1], [2, 3]], "Speed")
        self.assertEqual(len(mets), 2)
        self.assertEqual(mets[0]["distance"], 80)
        self.assertEqual(mets[1]["distance"], 140)


class StepHookTests(unittest.TestCase):
    def test_lab_generators_return_same_result_as_gameplay_call(self):
        g = random_graph(40, 50, 8, directed_prob=0.0)
        m = CostModel(g)
        for steps_fn, plain in ((lambda: dijkstra_steps(g, 0, 33, m, True), dijkstra(g, 0, 33, m)),
                                (lambda: astar_steps(g, 0, 33, m, True), astar(g, 0, 33, m))):
            gen = steps_fn()
            ops = []
            try:
                while True:
                    ops.append(next(gen)["op"])
            except StopIteration as stop:
                res = stop.value
            self.assertEqual(res.edges, plain.edges)
            self.assertIn("expand", ops)
            self.assertIn("relax", ops)

    def test_plan_route_steps_yield_leg_events(self):
        nav = Navigator(loadGraph(CIRCUIT))
        events = []
        gen = nav.planRouteSteps(0, 1)
        try:
            while True:
                s = next(gen)
                if s["op"] == "leg_done":
                    events.append(s)
        except StopIteration as stop:
            res = stop.value
        self.assertEqual(len(events), 11)
        self.assertTrue(res.found)


if __name__ == "__main__":
    unittest.main()
