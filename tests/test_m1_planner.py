import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from graph import Navigator, loadGraph
from gui.planner_state import PRESET_ORDER, PlannerState

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def nav_for(name):
    return Navigator(loadGraph(os.path.join(ROOT, "data", name)))


class PlannerStateTests(unittest.TestCase):
    def test_two_clicks_make_a_route_third_click_restarts(self):
        st = PlannerState(nav_for("sample_circuit.json"))
        st.click_vertex(1)
        self.assertEqual((st.src, st.dst, st.candidates), (1, None, []))
        st.click_vertex(8)
        self.assertTrue(st.candidates)
        st.click_vertex(4)
        self.assertEqual((st.src, st.dst), (4, None))
        self.assertEqual(st.candidates, [])

    def test_candidates_use_the_real_algorithms_and_displayed_cost_is_optimised_cost(self):
        nav = nav_for("sample_circuit.json")
        st = PlannerState(nav)
        st.click_vertex(1); st.click_vertex(8)
        best = st.candidates[0]
        ref = nav.shortestPath(1, 8, "Balanced")
        self.assertEqual(best["edges"], ref.edges)
        self.assertAlmostEqual(best["cost"], ref.cost, places=9)
        # alternatives are valid simple routes, never cheaper than the optimum
        for c in st.candidates[1:]:
            self.assertGreaterEqual(c["cost"], best["cost"] - 1e-9)
            self.assertEqual(len(c["vertices"]), len(set(c["vertices"])))
            self.assertEqual((c["vertices"][0], c["vertices"][-1]), (1, 8))
        self.assertGreaterEqual(len(st.candidates), 2)      # >= 2 routes to compare

    def test_preset_and_algorithm_switch(self):
        st = PlannerState(nav_for("demo_presets.json"))
        st.click_vertex(0); st.click_vertex(5)
        got = {}
        for name in PRESET_ORDER:
            st.set_preset(name)
            got[name] = tuple(st.candidates[0]["vertices"])
        self.assertEqual(len(set(got.values())), 4)          # presets pick different routes
        cost_d = st.candidates[0]["cost"]
        st.toggle_algorithm()
        self.assertEqual(st.algorithm, "astar")
        self.assertAlmostEqual(st.candidates[0]["cost"], cost_d, places=9)

    def test_closing_a_road_recalculates_and_reopening_restores(self):
        st = PlannerState(nav_for("sample_circuit.json"))
        st.click_vertex(1); st.click_vertex(8)
        before = list(st.candidates[0]["edges"])
        victim = before[len(before) // 2]
        st.toggle_edge_block(victim)
        self.assertNotIn(victim, st.candidates[0]["edges"])
        for c in st.candidates:
            self.assertNotIn(victim, c["edges"])
        st.toggle_edge_block(victim)
        self.assertEqual(st.candidates[0]["edges"], before)

    def test_clear_blocks_only_reopens_user_closures(self):
        nav = nav_for("sample_circuit.json")
        nav.blockEdge(0)                                     # closed by the "race", not the planner
        st = PlannerState(nav)
        st.toggle_edge_block(3)
        st.clear_blocks()
        self.assertFalse(nav.graph.edge(3).is_blocked)
        self.assertTrue(nav.graph.edge(0).is_blocked)

    def test_unreachable_destination_message_no_crash(self):
        nav = nav_for("sample_circuit.json")
        st = PlannerState(nav)
        for e in list(nav.graph.edges.values()):
            if 5 in (e.frm, e.to):
                nav.blockEdge(e.id)
        st.click_vertex(1); st.click_vertex(5)
        self.assertEqual(st.candidates, [])
        self.assertIn("unreachable", st.message)

    def test_lap_mode_and_reachability_overlay(self):
        st = PlannerState(nav_for("sample_circuit.json"))
        st.click_vertex(0)
        st.toggle_lap()
        lap = st.candidates[0]
        self.assertEqual(lap["kind"], "lap")
        cps = [v.id for v in st.graph.checkpoints()]
        it = iter(lap["vertices"])
        self.assertTrue(all(any(v == cp for v in it) for cp in cps))   # ordered visit
        st.toggle_lap()
        st.toggle_reachable()
        self.assertEqual(st.unreachable_checkpoints, [])
        for e in list(st.graph.edges.values()):
            if 5 in (e.frm, e.to):
                st.graph.block_edge(e.id)
        st.recompute()
        self.assertEqual(st.unreachable_checkpoints, [5])

    def test_bfs_row_differs_when_fewest_roads_is_not_cheapest(self):
        # demo circuit: every route has 2 roads, so BFS ties; build a case where it differs
        from tests.m1_helpers import build
        g = build([(0, 0, 0), (1, 10, 0), (2, 20, 0), (3, 30, 0)],
                  [(0, 0, 3, 100, 100, 100, 5), (1, 0, 1, 10, 10, 1, 1),
                   (2, 1, 2, 10, 10, 1, 1), (3, 2, 3, 10, 10, 1, 1)])
        st = PlannerState(Navigator(g))
        st.click_vertex(0); st.click_vertex(3)
        kinds = [c["kind"] for c in st.candidates]
        self.assertEqual(kinds[:2], ["optimal", "bfs"])
        self.assertEqual(st.candidates[1]["edges"], [0])     # 1 road but far more expensive
        self.assertGreater(st.candidates[1]["cost"], st.candidates[0]["cost"])


class ScreenSmokeTests(unittest.TestCase):
    def test_draw_and_events_do_not_crash_headless(self):
        import pygame
        from gui.route_planner import RoutePlannerScreen
        pygame.init()
        size = (1180, 720)
        surf = pygame.display.set_mode(size)
        for fname in ("sample_circuit.json", "demo_presets.json"):
            scr = RoutePlannerScreen(nav_for(fname), size)
            scr.draw(surf)
            first, second = list(scr.nav.graph.vertices)[:2]
            for vid in (first, second):
                pos = scr.to_screen(scr.nav.graph.vertices[vid].x, scr.nav.graph.vertices[vid].y)
                scr.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=1))
            self.assertIsNotNone(scr.state.dst)
            for key in (pygame.K_2, pygame.K_a, pygame.K_TAB, pygame.K_r, pygame.K_l, pygame.K_x):
                scr.handle_event(pygame.event.Event(pygame.KEYDOWN, key=key))
                scr.draw(surf)
            self.assertEqual(scr.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)), "back")
        pygame.quit()

    def test_right_click_closes_the_road_under_the_cursor(self):
        import pygame
        from gui.route_planner import RoutePlannerScreen
        pygame.init()
        size = (1180, 720)
        pygame.display.set_mode(size)
        scr = RoutePlannerScreen(nav_for("demo_presets.json"), size)
        scr.draw(pygame.display.get_surface())
        g = scr.nav.graph
        e = g.edge(0)
        a, b = g.vertices[e.frm], g.vertices[e.to]
        mid = scr.to_screen((a.x + b.x) / 2, (a.y + b.y) / 2)
        scr.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=mid, button=3))
        self.assertTrue(g.edge(0).is_blocked)
        pygame.quit()


if __name__ == "__main__":
    unittest.main()
