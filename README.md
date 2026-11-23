# Smart Racing Championship - Member 1 (Navigator) module

Graph model, scalar cost model + presets, BFS, DFS, Dijkstra, A*, checkpoint-ordered
route planning, Algorithm-Lab step hooks, tests and Experiments 1-2.
Python 3.10+, standard library only (matplotlib optional for charts, pygame-ce for the GUI).
Windows: use `py` instead of `python`, run everything from INSIDE the smart_racing folder, and
install the GUI library with `py -m pip install pygame-ce` (not `pygame`).

## Run
    python -m unittest discover -s tests -t . -p "test_m1_*.py" -v     # 78 tests
    python -m experiments.exp1_bfs_dfs                                  # Experiment 1
    python -m experiments.exp2_dijkstra_astar                           # Experiment 2
    python -m experiments.graphgen                                      # rewrite data/*.json
    python -m gui.route_planner [data/sample_circuit.json]              # Route Planner screen (needs pygame-ce)

## Layout
    common/       counters.py (agreed counter format), heap.py (counted min-heap, shared),
                  steps.py (step-hook helper)
    graph/        graph.py (model + dynamic updates + JSON), cost.py (cost, presets, A* ratio),
                  traversal.py (BFS, DFS, components, dead ends, alternative routes),
                  routing.py (Dijkstra, A*, planRoute, comparePresets), navigator.py (facade)
    data/         sample_circuit.json (10 checkpoints, 2 shortcuts, cycle, dead end,
                  2 fuel stations, nitro, one-way road), demo_presets.json (4 presets -> 4 routes)
    report/       screenshots/ (tests passing + Route Planner runs, for the report's testing section)
    gui/          planner_state.py (screen logic, no pygame), route_planner.py (pygame screen)
    experiments/  graphgen.py, exp1_bfs_dfs.py, exp2_dijkstra_astar.py, results/
    tests/        test_m1_*.py (+ m1_helpers.py with independent reference algorithms)

## API for teammates
    from graph import loadGraph, Navigator
    g   = loadGraph("data/sample_circuit.json")     # path, dict, or JSON string
    nav = Navigator(g)                              # one per circuit

    # routing (RouteResult: found, vertices, edges, cost, distance, time, fuel, counters, legs, failed_leg)
    nav.shortestPath(src, dst, preset="Balanced", algorithm="dijkstra"|"astar")
    nav.planRoute(racer, preset=None, algorithm="dijkstra")   # racer: current_vertex/currentVertex,
                                                               #        next_checkpoint/nextCheckpoint (order number)
    nav.planRouteFrom(current_vertex, next_checkpoint=1, preset=None)
    nav.comparePresets(src, dst)     nav.compareRoutes([edge_ids, ...], preset)

    # live graph updates - no rebuild; every later search uses the new state
    nav.blockEdge(id)  nav.unblockEdge(id)  nav.setRestricted(id, time_factor)
    nav.setEdgeEffects(id, time_factor=1.0, extra_difficulty=0, obstacle_penalty=0.0)  # ABSOLUTE final values
    nav.setVertexActive(id, bool)                    # FuelStation / NitroPickup only

    # exploration
    nav.bfs(src, dst=None)  nav.dfs(src=None)  nav.components()  nav.hasCycle()
    nav.deadEnds()  nav.reachability(src)  nav.alternativeRoutes(src, dst, main_edges, max_routes=10, ...)

    # Algorithm Lab: the SAME code as gameplay, emitting one dict per step
    for step in nav.bfsSteps(src, dst): ...     # also dfsSteps, dijkstraSteps, astarSteps, planRouteSteps
    # the generator's return value (StopIteration.value) is the final result

Step dict ops - BFS/DFS: discover, expand/finish, examine, cycle.  Dijkstra/A*: push, expand,
examine, relax.  planRoute: all of those + leg_done.

Counters (dict): expanded, relaxed, heapOps, visited, examinedEdges, reopened, pathCost, timeMs.

## Route Planner screen (gui/route_planner.py)
Class `RoutePlannerScreen(nav)` with `handle_event(event) -> "back" | None` and `draw(surface)`, so
Member 3's menu can embed it in its own loop (`.run()` is the standalone loop).  Left-click start then
destination; 1-4 presets; A Dijkstra/A*; TAB cycles candidates; right-click a road closes/reopens it
and the route is recalculated at once; R reachable checkpoints (BFS); L full-lap plan; X reopen all.
Candidates shown with distance, time, fuel, the exact scalar cost and counters: optimal route
(Dijkstra/A*), fewest-roads route (BFS) and the best 3 alternatives (bounded DFS enumeration).

## Rules / definitions to state in the report
* Edge cost = w_d*D + w_t*T + w_f*F + w_r*R + w_o*O, each term normalised:
  D=dist/maxDist, T=effTime/maxBaseTime, F=effFuel/(maxFuelCost*1.4), R=(effDifficulty-1)/4,
  O=obstaclePenalty (0..1 scale).  Reference maxima are static (computed from base edge data).
  Presets (w_d,w_t,w_f,w_r,w_o): Balanced .25/.25/.20/.15/.15, Speed .20/.60/.05/.05/.10,
  Economy .20/.10/.55/.05/.10, Safety .05/.10/.05/.40/.40.  "Distance" (1,0,0,0,0) is internal
  (Opponent 3).  The displayed cost IS the optimised cost (CostModel.edge_cost).
* Effective values: time = baseTime*timeFactor; difficulty = min(5, difficulty+extra);
  fuel = fuelCost*(1+0.1*(effDifficulty-1)).  Modifiers can only worsen an edge.
* Blocked edges are skipped; Restricted edges stay traversable at their worse effective cost.
  An inactive service vertex can still be driven through; only its service is unavailable.
* Counters: relaxed = improving relaxations; examinedEdges = all edges looked at.
* A*: h = ratio*euclid, ratio = min(baseCost(e)/euclid(e)); admissible AND consistent (so no
  reopening); ratio 0 -> h=0.  Reopening is still supported for custom heuristics.
* Dead end = non-mandatory vertex that cannot reach the finish without reversing: a vertex with
  no directed path to the finish, or in a pendant branch (leaf peeling).  Spurs ending in a
  mandatory checkpoint are not dead ends.
* Alternative route = simple path differing from the main route by >= 1 edge; limits:
  max_routes (10), max_edges (V-1), max_expansions (200 000); pruned to vertices that reach dst.
* planRoute plans ONE lap: current vertex -> checkpoint k -> ... -> last checkpoint -> Finish.
  If a leg is unreachable: found=False, failed_leg=(from,to) -> Member 2 applies the recovery rule.
  Call it again from the racer's current vertex after each batch of same-time events.
