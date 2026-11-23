"""Track Explorer / Route Planner screen (pygame).

Standalone:   python -m gui.route_planner [data/sample_circuit.json]
Integration (Member 3's menu):
    screen_obj = RoutePlannerScreen(nav)
    each frame:  for ev in events: if screen_obj.handle_event(ev) == "back": ...
                 screen_obj.draw(surface)

Controls
  left click vertex   pick start, then destination (third click restarts)
  right click road    close / reopen it -> route is recalculated immediately
  1 2 3 4             preset: Balanced / Speed / Economy / Safety
  A                   Dijkstra <-> A*          TAB   cycle candidate route
  R                   show reachable vertices / unreachable checkpoints (BFS)
  L                   full-lap plan through all checkpoints in order
  X                   reopen every road closed here           ESC  back
"""
from __future__ import annotations

import math
import sys
from typing import Optional, Tuple

import pygame

from graph.graph import (BLOCKED, CHECKPOINT, DEAD_END, FINISH, FUEL_STATION,
                         NITRO_PICKUP, RESTRICTED, START)
from gui.planner_state import PRESET_ORDER, PlannerState

PANEL_W = 380
BG = (16, 20, 28)
MAP_BG = (24, 29, 40)
TEXT = (225, 230, 240)
DIM = (140, 150, 168)
EDGE = (88, 99, 120)
EDGE_RESTRICTED = (240, 160, 40)
EDGE_BLOCKED = (225, 70, 70)
EDGE_SHORTCUT = (160, 130, 255)
ROUTE_COLORS = [(70, 225, 140), (80, 175, 255), (255, 210, 80), (255, 120, 200), (200, 200, 200)]


class RoutePlannerScreen:
    def __init__(self, nav, size: Tuple[int, int] = (1180, 720)):
        self.state = PlannerState(nav)
        self.nav = nav
        self.size = size
        self._fonts = {}
        self._xf = None

    # ------------------------------------------------------------------
    def font(self, size):
        if size not in self._fonts:
            self._fonts[size] = pygame.font.Font(None, size)
        return self._fonts[size]

    @property
    def map_rect(self):
        return pygame.Rect(0, 0, self.size[0] - PANEL_W, self.size[1])

    def _transform(self):
        verts = self.nav.graph.vertices.values()
        xs = [v.x for v in verts]
        ys = [v.y for v in verts]
        r = self.map_rect.inflate(-90, -90)
        w = max(max(xs) - min(xs), 1e-9)
        h = max(max(ys) - min(ys), 1e-9)
        s = min(r.width / w, r.height / h)
        ox = r.centerx - s * (min(xs) + w / 2)
        oy = r.centery - s * (min(ys) + h / 2)
        return s, ox, oy

    def to_screen(self, x, y):
        s, ox, oy = self._xf or self._transform()
        return int(ox + s * x), int(oy + s * y)

    # ---- picking ---------------------------------------------------------
    def pick_vertex(self, pos, radius=16) -> Optional[int]:
        best, bd = None, radius
        for v in self.nav.graph.vertices.values():
            d = math.dist(pos, self.to_screen(v.x, v.y))
            if d <= bd:
                best, bd = v.id, d
        return best

    def pick_edge(self, pos, radius=9) -> Optional[int]:
        g = self.nav.graph
        best, bd = None, radius
        px, py = pos
        for e in g.edges.values():
            ax, ay = self.to_screen(g.vertices[e.frm].x, g.vertices[e.frm].y)
            bx, by = self.to_screen(g.vertices[e.to].x, g.vertices[e.to].y)
            dx, dy = bx - ax, by - ay
            L2 = dx * dx + dy * dy
            t = 0 if L2 == 0 else max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / L2))
            d = math.hypot(px - (ax + t * dx), py - (ay + t * dy))
            if d <= bd:
                best, bd = e.id, d
        return best

    # ---- events ----------------------------------------------------------
    def handle_event(self, ev) -> Optional[str]:
        st = self.state
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_ESCAPE:
                return "back"
            keymap = {pygame.K_1: 0, pygame.K_2: 1, pygame.K_3: 2, pygame.K_4: 3}
            if ev.key in keymap:
                st.set_preset(PRESET_ORDER[keymap[ev.key]])
            elif ev.key == pygame.K_a:
                st.toggle_algorithm()
            elif ev.key == pygame.K_TAB:
                st.cycle_selected()
            elif ev.key == pygame.K_r:
                st.toggle_reachable()
            elif ev.key == pygame.K_l:
                st.toggle_lap()
            elif ev.key == pygame.K_x:
                st.clear_blocks()
        elif ev.type == pygame.MOUSEBUTTONDOWN and self.map_rect.collidepoint(ev.pos):
            if ev.button == 1:
                v = self.pick_vertex(ev.pos)
                if v is not None:
                    st.click_vertex(v)
            elif ev.button == 3:
                e = self.pick_edge(ev.pos)
                if e is not None:
                    st.toggle_edge_block(e)
        return None

    # ---- drawing ---------------------------------------------------------
    @staticmethod
    def _dashed(surf, color, a, b, width=2, dash=9, gap=6):
        L = math.dist(a, b)
        if L == 0:
            return
        ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
        d = 0.0
        while d < L:
            e = min(d + dash, L)
            pygame.draw.line(surf, color, (a[0] + ux * d, a[1] + uy * d),
                             (a[0] + ux * e, a[1] + uy * e), width)
            d += dash + gap

    @staticmethod
    def _arrow(surf, color, a, b, size=8):
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        ang = math.atan2(b[1] - a[1], b[0] - a[0])
        pts = [(mx + size * math.cos(ang), my + size * math.sin(ang)),
               (mx + size * math.cos(ang + 2.5), my + size * math.sin(ang + 2.5)),
               (mx + size * math.cos(ang - 2.5), my + size * math.sin(ang - 2.5))]
        pygame.draw.polygon(surf, color, pts)

    def _text(self, surf, txt, pos, size=22, color=TEXT):
        surf.blit(self.font(size).render(txt, True, color), pos)

    def draw(self, surf) -> None:
        self._xf = self._transform()
        g, st = self.nav.graph, self.state
        surf.fill(BG)
        pygame.draw.rect(surf, MAP_BG, self.map_rect)
        pos = {v.id: self.to_screen(v.x, v.y) for v in g.vertices.values()}
        route_edges = {}
        for i, c in enumerate(st.candidates):
            for eid in c["edges"]:
                route_edges.setdefault(eid, set()).add(i)

        # roads
        for e in g.edges.values():
            a, b = pos[e.frm], pos[e.to]
            if e.status == BLOCKED:
                col = EDGE_BLOCKED
            elif e.status == RESTRICTED:
                col = EDGE_RESTRICTED
            elif e.is_shortcut:
                col = EDGE_SHORTCUT
            else:
                col = EDGE
            if e.is_shortcut or e.status == BLOCKED:
                self._dashed(surf, col, a, b, 2)
            else:
                pygame.draw.line(surf, col, a, b, 2)
            if e.status == BLOCKED:
                mx, my = (a[0] + b[0]) // 2, (a[1] + b[1]) // 2
                pygame.draw.line(surf, col, (mx - 6, my - 6), (mx + 6, my + 6), 3)
                pygame.draw.line(surf, col, (mx - 6, my + 6), (mx + 6, my - 6), 3)
            if e.directed:
                self._arrow(surf, col, a, b)

        # candidate routes (selected one drawn last and thickest)
        order = [i for i in range(len(st.candidates)) if i != st.selected] + \
                ([st.selected] if st.candidates else [])
        for i in order:
            c = st.candidates[i]
            col = ROUTE_COLORS[i % len(ROUTE_COLORS)]
            w = 7 if i == st.selected else 3
            for u, v in zip(c["vertices"], c["vertices"][1:]):
                pygame.draw.line(surf, col, pos[u], pos[v], w)

        # vertices
        for v in g.vertices.values():
            p = pos[v.id]
            dim = st.show_reachable and v.id not in st.reachable
            self._vertex(surf, v, p, dim)
            if v.id in st.unreachable_checkpoints or (
                    st.show_reachable and v.is_mandatory and v.id not in st.reachable):
                pygame.draw.circle(surf, EDGE_BLOCKED, p, 17, 3)
        if st.src is not None:
            pygame.draw.circle(surf, (70, 225, 140), pos[st.src], 19, 3)
            self._text(surf, "START", (pos[st.src][0] - 18, pos[st.src][1] - 36), 20, (70, 225, 140))
        if st.dst is not None and st.dst in pos and not st.lap_mode:
            pygame.draw.circle(surf, (255, 90, 90), pos[st.dst], 19, 3)
            self._text(surf, "DEST", (pos[st.dst][0] - 16, pos[st.dst][1] - 36), 20, (255, 90, 90))
        self._panel(surf)

    def _vertex(self, surf, v, p, dim):
        k = 0.45 if dim else 1.0

        def c(col):
            return tuple(int(x * k) for x in col)
        if v.type == START:
            pygame.draw.circle(surf, c((60, 200, 100)), p, 11)
        elif v.type == FINISH:
            pygame.draw.circle(surf, c((240, 240, 240)), p, 11)
            pygame.draw.circle(surf, c((20, 20, 20)), p, 5)
        elif v.type == CHECKPOINT or v.is_mandatory:
            pygame.draw.circle(surf, c((80, 160, 255)), p, 11)
            t = self.font(18).render(str(v.checkpoint_order), True, (10, 15, 30))
            surf.blit(t, t.get_rect(center=p))
        elif v.type == FUEL_STATION:
            r = pygame.Rect(0, 0, 16, 16)
            r.center = p
            pygame.draw.rect(surf, c((250, 210, 60)) if v.active else (90, 80, 40), r,
                             0 if v.active else 2)
        elif v.type == NITRO_PICKUP:
            pts = [(p[0], p[1] - 10), (p[0] + 10, p[1]), (p[0], p[1] + 10), (p[0] - 10, p[1])]
            pygame.draw.polygon(surf, c((60, 220, 230)) if v.active else (40, 90, 95), pts,
                                0 if v.active else 2)
        elif v.type == DEAD_END:
            pygame.draw.circle(surf, c((150, 150, 150)), p, 7)
            pygame.draw.line(surf, c((60, 60, 60)), (p[0] - 4, p[1] - 4), (p[0] + 4, p[1] + 4), 2)
            pygame.draw.line(surf, c((60, 60, 60)), (p[0] - 4, p[1] + 4), (p[0] + 4, p[1] - 4), 2)
        else:
            pygame.draw.circle(surf, c((140, 150, 170)), p, 5)

    def _panel(self, surf):
        st = self.state
        x0 = self.size[0] - PANEL_W + 16
        pygame.draw.rect(surf, (20, 25, 35), (self.size[0] - PANEL_W, 0, PANEL_W, self.size[1]))
        y = 14
        self._text(surf, "ROUTE PLANNER", (x0, y), 30)
        y += 34
        self._text(surf, f"Preset: {st.preset}    Algorithm: "
                   f"{'Dijkstra' if st.algorithm == 'dijkstra' else 'A*'}", (x0, y), 22, DIM)
        y += 24
        w = self.nav.model(st.preset).weights
        self._text(surf, f"weights d{w.d:.2f} t{w.t:.2f} f{w.f:.2f} r{w.r:.2f} o{w.o:.2f}",
                   (x0, y), 19, DIM)
        y += 24
        src = "-" if st.src is None else st.src
        dst = "-" if st.dst is None else st.dst
        self._text(surf, f"From {src}  [LAP MODE: all checkpoints, then finish]" if st.lap_mode
                   else f"From {src}  To {dst}", (x0, y), 22)
        y += 30
        for i, c in enumerate(st.candidates):
            col = ROUTE_COLORS[i % len(ROUTE_COLORS)]
            box = pygame.Rect(x0 - 8, y - 4, PANEL_W - 16, 78)
            if i == st.selected:
                pygame.draw.rect(surf, (36, 44, 62), box, border_radius=6)
            pygame.draw.rect(surf, col, (x0 - 8, y - 4, 5, 78))
            self._text(surf, c["label"][:34], (x0 + 4, y), 21, col)
            self._text(surf, f"dist {c['distance']:.0f}   time {c['time']:.0f}   fuel {c['fuel']:.1f}",
                       (x0 + 4, y + 22), 20)
            self._text(surf, f"cost {c['cost']:.4f}   roads {len(c['edges'])}", (x0 + 4, y + 42), 20)
            if c["counters"]:
                k = c["counters"]
                self._text(surf, f"exp {k['expanded']} relax {k['relaxed']} heap {k['heapOps']} "
                           f"{k['timeMs']:.2f} ms", (x0 + 4, y + 60), 17, DIM)
            y += 86
        if st.alt_note:
            self._text(surf, st.alt_note, (x0, y), 18, DIM)
            y += 22
        if st.show_reachable:
            n = len(st.unreachable_checkpoints)
            self._text(surf, f"Unreachable checkpoints: {n}" + (
                f" {st.unreachable_checkpoints}" if n else ""), (x0, y), 19,
                EDGE_BLOCKED if n else (70, 225, 140))
            y += 22
        self._text(surf, st.message[:46], (x0, max(y, 480)), 20, (255, 210, 80))
        helps = ["click: start / dest      right-click road: close/open",
                 "1-4 preset   A algorithm   TAB cycle route",
                 "R reachable   L lap plan   X reopen all   ESC back"]
        for i, h in enumerate(helps):
            self._text(surf, h, (x0, self.size[1] - 70 + i * 20), 17, DIM)

    # ---- standalone loop ---------------------------------------------------
    def run(self, screen=None):
        pygame.init()
        screen = screen or pygame.display.set_mode(self.size)
        pygame.display.set_caption("Smart Racing - Route Planner")
        clock = pygame.time.Clock()
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or self.handle_event(ev) == "back":
                    return
            self.draw(screen)
            pygame.display.flip()
            clock.tick(30)


def main():
    import os
    from graph import Navigator, loadGraph
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "data", "sample_circuit.json")
    RoutePlannerScreen(Navigator(loadGraph(path))).run()


if __name__ == "__main__":
    main()
