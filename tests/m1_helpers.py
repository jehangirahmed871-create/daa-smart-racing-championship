"""Shared helpers for Member 1 tests (reference implementations)."""
import math
import random

from graph.graph import Edge, Graph, Vertex, START, FINISH


def build(vertices, edges):
    """vertices: [(id,x,y[,type[,order]])]; edges: [(id,u,v,dist[,time[,fuel[,diff[,directed]]]])]"""
    g = Graph()
    for t in vertices:
        i, x, y = t[:3]
        vt = t[3] if len(t) > 3 else "Intersection"
        order = t[4] if len(t) > 4 else None
        g.add_vertex(Vertex(i, x, y, vt, order))
    for t in edges:
        eid, u, v, d = t[:4]
        tm = t[4] if len(t) > 4 else d
        fu = t[5] if len(t) > 5 else d
        df = t[6] if len(t) > 6 else 1
        di = t[7] if len(t) > 7 else False
        g.add_edge(Edge(eid, u, v, d, tm, fu, df, directed=di))
    return g


def random_graph(n, extra, seed, directed_prob=0.2, zero_prob=0.0):
    """Connected-ish random graph, coordinates consistent with distances."""
    rng = random.Random(seed)
    g = Graph()
    for i in range(n):
        g.add_vertex(Vertex(i, rng.uniform(0, 100), rng.uniform(0, 100)))
    eid = 0

    def add(u, v):
        nonlocal eid
        a, b = g.vertices[u], g.vertices[v]
        length = math.hypot(a.x - b.x, a.y - b.y)
        d = 0.0 if rng.random() < zero_prob else length * rng.uniform(1.0, 1.6)
        tm = 0.0 if d == 0.0 else d * rng.uniform(0.5, 1.5)
        fu = 0.0 if d == 0.0 else d * rng.uniform(0.05, 0.2)
        g.add_edge(Edge(eid, u, v, d, tm, fu, rng.randint(1, 5),
                        directed=rng.random() < directed_prob))
        eid += 1

    for i in range(1, n):
        add(rng.randrange(i), i)
    for _ in range(extra):
        u, v = rng.randrange(n), rng.randrange(n)
        if u != v:
            add(u, v)
    return g


def bellman_ford(graph, model, src):
    """Reference shortest-path costs (independent of Dijkstra/heaps)."""
    dist = {v: math.inf for v in graph.vertices}
    dist[src] = 0.0
    arcs = []
    for u in graph.vertices:
        for e, v in graph.neighbors(u):
            arcs.append((u, v, model.edge_cost(e)))
    for _ in range(len(graph.vertices) - 1):
        changed = False
        for u, v, w in arcs:
            if dist[u] + w < dist[v] - 1e-15:
                dist[v] = dist[u] + w
                changed = True
        if not changed:
            break
    return dist


def unit_bfs_reference(graph, src):
    """Reference minimum edge counts via repeated relaxation."""
    dist = {v: math.inf for v in graph.vertices}
    dist[src] = 0
    arcs = [(u, v) for u in graph.vertices for _, v in graph.neighbors(u)]
    for _ in range(len(graph.vertices)):
        changed = False
        for u, v in arcs:
            if dist[u] + 1 < dist[v]:
                dist[v] = dist[u] + 1
                changed = True
        if not changed:
            break
    return dist
