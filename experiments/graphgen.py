"""Deterministic graph builders for tests, demos and experiments.

make_grid_graph      connected jittered-grid road network (sparse/medium/dense)
make_sample_circuit  hand-designed circuit: 10 checkpoints, 2 shortcuts, cycle,
                     dead end, fuel stations, nitro pickup, one one-way road
make_demo_presets    4 parallel routes S->F where each preset picks a different one
All use seeded RNGs so results are reproducible.
"""
import json
import math
import os
import random

from graph.graph import Edge, Graph, Vertex, START, FINISH, INTERSECTION

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_grid_graph(n, density="sparse", seed=1, cell=100.0, jitter=0.25):
    """Connected road-like graph with n vertices.

    sparse : grid edges only            (E ~ 2V)
    medium : + both diagonals           (E ~ 4V)
    dense  : + random local chords      (E ~ 8V)
    Edge distance = straight-line length * (1..1.4), so Euclidean-based A*
    heuristics are valid.  Vertex 0 = Start, vertex n-1 = Finish.
    """
    if density not in ("sparse", "medium", "dense"):
        raise ValueError("density must be sparse | medium | dense")
    rng = random.Random(seed)
    m = max(2, math.isqrt(n - 1) + 1)
    g = Graph()
    for i in range(n):
        r, c = divmod(i, m)
        x = c * cell + rng.uniform(-jitter, jitter) * cell
        y = r * cell + rng.uniform(-jitter, jitter) * cell
        vtype = START if i == 0 else (FINISH if i == n - 1 else INTERSECTION)
        g.add_vertex(Vertex(i, x, y, vtype))
    eid = [0]
    pairs = set()

    def add(u, v):
        key = (min(u, v), max(u, v))
        if key in pairs or u == v:
            return False
        pairs.add(key)
        a, b = g.vertices[u], g.vertices[v]
        length = math.hypot(a.x - b.x, a.y - b.y)
        dist = round(length * (1 + rng.uniform(0, 0.4)), 2)
        diff = rng.randint(1, 5)
        time = round(dist * (0.5 + 0.1 * diff) * rng.uniform(0.9, 1.1), 2)
        fuel = round(dist * 0.1 * rng.uniform(0.8, 1.2), 2)
        g.add_edge(Edge(eid[0], u, v, dist, time, fuel, diff))
        eid[0] += 1
        return True

    for i in range(n):
        r, c = divmod(i, m)
        if c + 1 < m and i + 1 < n:
            add(i, i + 1)
        if i + m < n:
            add(i, i + m)
    if density in ("medium", "dense"):
        for i in range(n):
            r, c = divmod(i, m)
            if c + 1 < m and i + m + 1 < n:
                add(i, i + m + 1)
            if c > 0 and i + m - 1 < n:
                add(i, i + m - 1)
    if density == "dense":
        target = 8 * n
        tries = 0
        while len(g.edges) < target and tries < target * 20:
            tries += 1
            i = rng.randrange(n)
            r, c = divmod(i, m)
            rr, cc = r + rng.randint(-3, 3), c + rng.randint(-3, 3)
            j = rr * m + cc
            if 0 <= rr and 0 <= cc < m and 0 <= j < n:
                add(i, j)
    return g


def make_demo_presets():
    """S(0) -> F(5) with four parallel 2-edge routes (via 1,2,3,4).

      via 1  short, fast, very hard + obstacle   -> Speed
      via 2  long, easy, high fuel               -> Safety
      via 3  medium, LOW fuel                    -> Economy
      via 4  medium everything                   -> Balanced
    """
    V = [(0, 0, 0, "Start"), (5, 70, 0, "Finish"),
         (1, 35, 3, "Intersection"), (2, 35, -35, "Intersection"),
         (3, 35, 18, "Intersection"), (4, 35, -12, "Intersection")]
    #        id from to  dist time fuel diff obstacle
    E = [
        (0, 0, 1, 40, 30, 20, 5, 0.0), (1, 1, 5, 40, 30, 20, 5, 0.0),
        (2, 0, 2, 70, 70, 25, 1, 0.0), (3, 2, 5, 70, 70, 25, 1, 0.0),
        (4, 0, 3, 55, 50, 8, 3, 0.0), (5, 3, 5, 55, 50, 8, 3, 0.0),
        (6, 0, 4, 50, 45, 15, 2, 0.0), (7, 4, 5, 50, 45, 15, 2, 0.0),
    ]
    return {
        "vertices": [{"id": i, "x": x, "y": y, "type": t, "checkpointOrder": None,
                      "active": True} for i, x, y, t in V],
        "edges": [{"id": i, "from": a, "to": b, "distance": d, "baseTime": t,
                   "fuelCost": f, "difficulty": df, "status": "Open",
                   "isShortcut": False, "directed": False,
                   **({"obstaclePenalty": o} if o else {})}
                  for i, a, b, d, t, f, df, o in E],
    }


def make_sample_circuit():
    """A small but complete circuit used by tests and as a format example.

    Ring: Start(0) -> checkpoints 1..10 -> Finish(11) -> back to Start.
    Shortcuts: 2-5 and 6-9.  Alternatives: 3-12-5 (12->5 is one-way) and
    4-15-5 via the nitro pickup.  Fuel stations 13 (7-13-8) and 14 (0-14-1).
    Dead end: spur 9-16-17.  The ring itself is the cycle.
    """
    rx, ry = 250.0, 150.0
    verts, edges = [], []

    def V(i, x, y, t="Intersection", order=None):
        verts.append({"id": i, "x": round(x, 1), "y": round(y, 1), "type": t,
                      "checkpointOrder": order, "active": True})

    def pos(slot, scale=1.0):
        a = -math.pi / 2 + slot * 2 * math.pi / 11
        return rx * scale * math.cos(a), ry * scale * math.sin(a)

    V(0, *pos(0), "Start")
    for k in range(1, 11):
        V(k, *pos(k), "Checkpoint", k)
    V(11, *pos(10.5), "Finish")
    V(12, 0, 30)                                       # inner intersection
    V(13, *pos(7.5, 1.18), "FuelStation")
    V(14, *pos(0.5, 1.18), "FuelStation")
    V(15, *pos(4.5, 1.18), "NitroPickup")
    V(16, *pos(9, 1.25))
    V(17, *pos(9, 1.45), "DeadEnd")
    by_id = {v["id"]: v for v in verts}

    def E(a, b, shortcut=False, directed=False):
        i = len(edges)
        va, vb = by_id[a], by_id[b]
        length = math.hypot(va["x"] - vb["x"], va["y"] - vb["y"])
        dist = round(length * (1.02 if shortcut else 1.05 + 0.05 * (i % 4)), 1)
        diff = 4 + (i % 2) if shortcut else (i % 5) + 1
        edges.append({"id": i, "from": a, "to": b, "distance": dist,
                      "baseTime": round(dist * (0.6 + 0.1 * diff), 1),
                      "fuelCost": round(dist * 0.12, 1), "difficulty": diff,
                      "status": "Open", "isShortcut": shortcut, "directed": directed})

    for k in range(0, 10):
        E(k, k + 1)                 # ids 0..9 : ring 0-1 ... 9-10
    E(10, 11)                       # id 10
    E(11, 0)                        # id 11
    E(2, 5, shortcut=True)          # id 12
    E(6, 9, shortcut=True)          # id 13
    E(3, 12)                        # id 14
    E(12, 5, directed=True)         # id 15 (one-way 12 -> 5)
    E(4, 15)                        # id 16
    E(15, 5)                        # id 17
    E(7, 13)                        # id 18
    E(13, 8)                        # id 19
    E(0, 14)                        # id 20
    E(14, 1)                        # id 21
    E(9, 16)                        # id 22
    E(16, 17)                       # id 23
    return {"vertices": verts, "edges": edges}


def write_data_files():
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    for name, data in (("demo_presets.json", make_demo_presets()),
                       ("sample_circuit.json", make_sample_circuit())):
        with open(os.path.join(ROOT, "data", name), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
        print("wrote data/" + name)


if __name__ == "__main__":
    write_data_files()
