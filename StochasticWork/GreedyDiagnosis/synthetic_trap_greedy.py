"""
synthetic_trap.py
=================

A fully self-contained synthetic experiment designed to expose
greedy's structural weakness vs the adaptive policy.

The wind field is static (does not change between stages or forecasts)
but is deliberately designed as a "trap" for greedy:

  - Southern nodes (N0, N1) have very high wind at early stages
    but very low wind at later stages → greedy gets drawn south
    and gets stuck paying high costs + penalties to escape

  - Northern nodes (N3, N4) have low wind early but high wind
    at later stages → adaptive should recognise this and route north

  - Middle nodes (N2) are consistently moderate throughout

The 6 "forecasts" are small perturbations of the same base field
so the forecast signal is reliable (low error) — this isolates
whether the structural problem is policy logic, not forecast noise.

No GRIB files needed. Entirely self-contained.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ============================================================
# GRID PARAMETERS
# ============================================================

NUM_STAGES      = 8
NODES_PER_STAGE = 5
NUM_FORECASTS   = 6
LATERAL_PENALTY = 0.05

NODE_LATS  = [20.0, 30.0, 40.0, 50.0, 60.0]
STAGE_LONS = [-90.0, -85.0, -80.0, -75.0, -70.0, -65.0, -60.0, -55.0]

# Middle node index and START/END anchor
MID_IDX    = NODES_PER_STAGE // 2    # = 2
LON_STEP   = 5.0
START_COORD = (NODE_LATS[MID_IDX], STAGE_LONS[0]  - LON_STEP)
END_COORD   = (NODE_LATS[MID_IDX], STAGE_LONS[-1] + LON_STEP)

START_WIND = 10
END_WIND = 10

# ============================================================
# WIND FIELD 1 — STATIC TRAP
# ============================================================

BASE_WIND_STATIC = {
    #          S0    S1    S2    S3    S4    S5    S6    S7
    0: np.array([10.0, 16.0, 15.0, 10.0,  5.0,  3.0,  2.0,  2.0]),  # south: tempting then collapses
    1: np.array([ 8.0,  10.0,  7.0,  7.0,  7.0,  7.0,  7.0,  7.0]),  # moderate throughout
    2: np.array([20.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0]),  # high at S0 only — both go here first
    3: np.array([ 8.0,  7.0,  8.0,  8.0,  8.0,  8.0,  8.0,  8.0]),  # moderate throughout
    4: np.array([ 5.0, 16.0, 17.0, 17.0, 16.0, 17.0, 16.0, 17.0]),  # consistent high — adaptive goes here
}

# ============================================================
# WIND FIELD 2 — DYNAMIC TRAP
# ============================================================

BASE_WIND_DYNAMIC = {
    #          S0    S1    S2    S3    S4    S5    S6    S7
    0: np.array([20.0, 18.0, 15.0,  8.0,  4.0,  3.0,  3.0,  3.0]),
    1: np.array([14.0, 12.0, 10.0,  7.0,  6.0,  5.0,  4.0,  4.0]),
    2: np.array([ 9.0, 10.0, 11.0, 11.0, 11.0, 11.0, 10.0,  9.0]),
    3: np.array([ 5.0,  6.0,  8.0, 11.0, 14.0, 16.0, 17.0, 18.0]),
    4: np.array([ 3.0,  4.0,  5.0,  7.0, 12.0, 17.0, 19.0, 20.0]),
}

DYNAMIC_NOISE_STD  = 1.2
DYNAMIC_NOISE_GROW = 0.15

# ============================================================
# WIND FIELD 3 — MAXIMUM DIVERGENCE  (static, no time variation)
# ============================================================
#
# Designed so greedy picks N0 (south) and adaptive picks N4
# (north) from START at stage 0 — and they never cross back.
#
# Mathematical guarantee of divergence:
#
#   START -> N0: arc_cost(START_WIND=10, N0_S0=22) + pen(mid->N0)
#              = 1/16 + 0.05 = 0.1125   ← greedy picks this (cheapest)
#
#   START -> N4: arc_cost(10, 2) + pen(mid->N4)
#              = 1/6  + 0.05 = 0.2167   ← greedy never picks this
#
#   But cost-to-go from N4 for stages 1-7:
#     7 stages * 1/22 = 7 * 0.0455 = 0.3182
#   Cost-to-go from N0 for stages 1-7:
#     7 stages * 1/2  = 7 * 0.5000 = 3.5000
#
#   Total adaptive score:
#     N4: 0.2167 + 0.3182 = 0.5349
#     N0: 0.1125 + 3.5000 = 3.6125
#
#   Adaptive picks N4 by a massive margin.
#   Greedy picks N0 because it only sees the immediate cost.
#
# N1, N2, N3 are set to moderate wind throughout so they are
# never the best immediate or best lookahead choice — all
# policies ignore them, keeping paths maximally separated.
#
# STATIC: wind values are the same at every stage.
# No noise — deterministic field so every forecast is identical.
# This isolates pure policy logic from forecast uncertainty.

BASE_WIND_DIVERGE = {
    #          S0    S1    S2    S3    S4    S5    S6    S7
    #
    # N0 (20°N): very high at S0, then collapses completely
    0: np.array([22.0,  2.0,  2.0,  2.0,  2.0,  2.0,  2.0,  2.0]),
    #
    # N1 (30°N): moderate throughout — not the best choice ever
    1: np.array([ 8.0,  8.0,  8.0,  8.0,  8.0,  8.0,  8.0,  8.0]),
    #
    # N2 (40°N): moderate throughout — not the best choice ever
    2: np.array([ 9.0,  9.0,  9.0,  9.0,  9.0,  9.0,  9.0,  9.0]),
    #
    # N3 (50°N): moderate throughout — not the best choice ever
    3: np.array([ 8.0,  8.0,  8.0,  8.0,  8.0,  8.0,  8.0,  8.0]),
    #
    # N4 (60°N): very low at S0, then peaks completely
    4: np.array([ 2.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0]),
}

# ============================================================
# WIND FIELD 4 — ENSEMBLE MAJORITY TRAP
# ============================================================
#
# Designed so Voting fails like Greedy despite full backward induction.
#
# FORECAST ENSEMBLE (what 5/6 members see):
#   N4 (60°N): very high wind throughout → ensemble majority loves north
#   N0 (20°N): low wind throughout → ensemble majority avoids south
#
# REALISED WIND (what actually happens):
#   N0 (20°N): high wind throughout → south is actually cheap
#   N4 (60°N): low wind throughout → north is actually expensive
#
# RESULT:
#   Voting: 5/6 members vote N4 at every stage → commits north → pays high cost
#   Greedy: picks lowest immediate forecast cost → also goes north → same failure
#   Adaptive: averages cost-to-go across ALL forecasts including minority →
#             minority member's low N4 ctg pulls mean up → picks south → wins
#   Hindsight: optimal on realised → picks south

# What 5 of 6 forecasts see (majority — wrong)
MAJORITY_WIND = {
    #          S0    S1    S2    S3    S4    S5    S6    S7
    0: np.array([ 3.0,  3.0,  3.0,  3.0,  3.0,  3.0,  3.0,  3.0]),  # N0 south: low
    1: np.array([ 6.0,  6.0,  6.0,  6.0,  6.0,  6.0,  6.0,  6.0]),  # N1
    2: np.array([ 9.0,  9.0,  9.0,  9.0,  9.0,  9.0,  9.0,  9.0]),  # N2 mid
    3: np.array([12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0]),  # N3
    4: np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0]),  # N4 north: high
}

# What 1 of 6 forecasts sees (minority — correct)
MINORITY_WIND = {
    #          S0    S1    S2    S3    S4    S5    S6    S7
    0: np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0]),  # N0 south: high
    1: np.array([12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0]),  # N1
    2: np.array([ 9.0,  9.0,  9.0,  9.0,  9.0,  9.0,  9.0,  9.0]),  # N2 mid
    3: np.array([ 6.0,  6.0,  6.0,  6.0,  6.0,  6.0,  6.0,  6.0]),  # N3
    4: np.array([ 3.0,  3.0,  3.0,  3.0,  3.0,  3.0,  3.0,  3.0]),  # N4 north: low
}

# Realised wind — matches minority (south is actually good)
REALISED_WIND = MINORITY_WIND

MAJORITY_TRAP_NOISE_STD = 0.3   # small noise so signal is clear

# No noise for this field — we want a perfectly deterministic
# outcome so the policy logic is the only variable
DIVERGE_NOISE_STD  = 0.0
DIVERGE_NOISE_GROW = 0.0

# Active wind field — overridden per experiment
BASE_WIND = BASE_WIND_STATIC

# ============================================================
# FORECAST GENERATION
# ============================================================
#
# 6 forecasts = small noise perturbations of the base field.
# Noise is small (±0.5 m/s) so the signal is reliable.
# All forecasts agree on the trap structure.



NOISE_STD = 0.5

def make_forecasts(base_wind, noise_std=0.5, noise_grow=0.0, seed=42):
    """
    Returns list of NUM_FORECASTS wind speed arrays.
    forecasts[k][node][stage] = wind speed m/s

    noise_grow: additional noise added per stage so uncertainty
    grows over time (realistic — forecast error grows with lead time).
    """
    rng = np.random.RandomState(seed)
    forecasts = []
    for k in range(NUM_FORECASTS):
        f = {}
        for node in range(NODES_PER_STAGE):
            stage_noise = np.array([
                rng.normal(0, noise_std + noise_grow * s)
                for s in range(NUM_STAGES)
            ])
            f[node] = np.clip(base_wind[node] + stage_noise, 0.5, 30.0)
        forecasts.append(f)
    return forecasts

def make_forecasts_majority_trap(noise_std=0.3, seed=42):
    """
    5 of 6 members drawn from MAJORITY_WIND (north looks good).
    1 of 6 members drawn from MINORITY_WIND (south looks good).
    Realised wind = MINORITY_WIND (south is actually correct).
    Returns (forecast_dicts, realised_dict).
    """
    rng = np.random.RandomState(seed)
    forecast_dicts = []

    for k in range(NUM_FORECASTS):
        base = MAJORITY_WIND if k < 5 else MINORITY_WIND
        f = {}
        for node in range(NODES_PER_STAGE):
            noise = np.array([
                rng.normal(0, noise_std)
                for _ in range(NUM_STAGES)
            ])
            f[node] = np.clip(base[node] + noise, 0.5, 30.0)
        forecast_dicts.append(build_forecast_dict(f))

    # Realised = deterministic minority (no noise for clarity)
    realised_dict = build_forecast_dict(REALISED_WIND)

    return forecast_dicts, realised_dict

# ============================================================
# ARC COST
# ============================================================

def arc_cost_fn(ws_a, ws_b):
    """1 / mean wind speed — lower = better."""
    return 1.0 / max((ws_a + ws_b) / 2.0, 0.1)


def lateral_pen(from_idx, to_idx):
    """Penalty for lateral movement beyond one node."""
    return LATERAL_PENALTY * max(0, abs(from_idx - to_idx) - 1)


def total_arc_cost(ws_a, ws_b, from_idx, to_idx):
    return arc_cost_fn(ws_a, ws_b) + lateral_pen(from_idx, to_idx)


def start_to_node_cost(forecast_winds, node_idx, stage=0):
    """Cost from START anchor to node at stage 0."""
    ws_node = forecast_winds[node_idx][stage]
    pen     = LATERAL_PENALTY * max(0, abs(node_idx - MID_IDX) - 1)
    return arc_cost_fn(START_WIND, ws_node) + pen


def node_to_end_cost(forecast_winds, node_idx, stage=NUM_STAGES-1):
    """Cost from final-stage node to END anchor."""
    ws_node = forecast_winds[node_idx][stage]
    pen     = LATERAL_PENALTY * max(0, abs(node_idx - MID_IDX) - 1)
    return arc_cost_fn(ws_node, END_WIND) + pen

# ============================================================
# BUILD COST DICT  (same format as policy_grib)
# ============================================================

def build_forecast_dict(forecast_winds, base_wind=None):
    """
    Converts wind array into a nested cost dict.
    base_wind unused here but kept for API consistency.
    """
    costs = {}

    # START -> stage-0 nodes
    costs["START"] = {}
    for i in range(NODES_PER_STAGE):
        node = f"S0_N{i}"
        costs["START"][node] = start_to_node_cost(forecast_winds, i, 0)

    # Stage s -> stage s+1
    for s in range(NUM_STAGES - 1):
        for i in range(NODES_PER_STAGE):
            from_node = f"S{s}_N{i}"
            ws_f      = forecast_winds[i][s]
            costs.setdefault(from_node, {})
            for j in range(NODES_PER_STAGE):
                to_node = f"S{s+1}_N{j}"
                ws_t    = forecast_winds[j][s+1]
                costs[from_node][to_node] = total_arc_cost(
                    ws_f, ws_t, i, j
                )

    # Last stage -> END
    for i in range(NODES_PER_STAGE):
        from_node = f"S{NUM_STAGES-1}_N{i}"
        costs.setdefault(from_node, {})
        costs[from_node]["END"] = node_to_end_cost(
            forecast_winds, i, NUM_STAGES-1
        )

    return costs

# ============================================================
# COST-TO-GO  (recursive DP)
# ============================================================

_ctg_cache = {}

def future_cost_to_go(node, stage, cost_dict):
    key = (node, stage)
    if key in _ctg_cache:
        return _ctg_cache[key]

    if stage >= NUM_STAGES:
        _ctg_cache[key] = 0.0
        return 0.0

    if node == "END":
        _ctg_cache[key] = 0.0
        return 0.0

    # Next layer
    if stage == NUM_STAGES:
        next_nodes = ["END"]
    else:
        next_nodes = [f"S{stage}_N{j}" for j in range(NODES_PER_STAGE)]
        if stage == NUM_STAGES:
            next_nodes = ["END"]

    best = float("inf")
    for nxt in next_nodes:
        c = cost_dict.get(node, {}).get(nxt, float("inf"))
        f = future_cost_to_go(nxt, stage + 1, cost_dict)
        best = min(best, c + f)

    _ctg_cache[key] = best
    return best

# ============================================================
# LAYERS HELPER
# ============================================================

def get_layers():
    lys = [["START"]]
    for s in range(NUM_STAGES):
        lys.append([f"S{s}_N{i}" for i in range(NODES_PER_STAGE)])
    lys.append(["END"])
    return lys

LAYERS = get_layers()

def next_layer_nodes(stage):
    """Returns the list of nodes in the layer after 'stage'."""
    # stage here is the decision stage index (0-based)
    # layer index = stage + 1
    return LAYERS[stage + 1]

# ============================================================
# GREEDY POLICY
# ============================================================

def run_greedy(forecast_dicts, realised_k):
    """
    Greedy: picks node with lowest IMMEDIATE arc cost,
    averaged across all forecasts. No lookahead.
    """
    global _ctg_cache
    realised_dict = forecast_dicts[realised_k]

    path        = ["START"]
    stage_costs = []

    for stage in range(NUM_STAGES):
        current = path[-1]
        candidates = next_layer_nodes(stage)

        best_node = min(
            candidates,
            key=lambda n: np.mean([
                f.get(current, {}).get(n, float("inf"))
                for f in forecast_dicts
            ])
        )

        # Realised cost
        cost = realised_dict.get(current, {}).get(best_node, float("inf"))
        stage_costs.append(cost)
        path.append(best_node)

    path.append("END")

    return path, stage_costs, sum(stage_costs)

# ============================================================
# ADAPTIVE POLICY
# ============================================================

def run_adaptive(forecast_dicts, realised_k):
    """
    Adaptive: at each stage picks node minimising
    mean(immediate + cost-to-go) across all forecasts.
    """
    realised_dict = forecast_dicts[realised_k]

    path        = ["START"]
    stage_costs = []

    for stage in range(NUM_STAGES):
        current    = path[-1]
        candidates = next_layer_nodes(stage)

        best_node      = None
        best_mean_cost = float("inf")

        for candidate in candidates:
            imm_costs = []
            ctg_vals  = []

            for f in forecast_dicts:
                imm = f.get(current, {}).get(candidate, float("inf"))
                imm_costs.append(imm)

                # Reset cache per forecast
                _ctg_cache_local = {}
                ctg = _ctg_dp(candidate, stage + 1, f, _ctg_cache_local)
                ctg_vals.append(ctg)

            mean_total = np.mean(imm_costs) + np.mean(ctg_vals)

            if mean_total < best_mean_cost:
                best_mean_cost = mean_total
                best_node      = candidate

        cost = realised_dict.get(current, {}).get(best_node, float("inf"))
        stage_costs.append(cost)
        path.append(best_node)

    path.append("END")

    return path, stage_costs, sum(stage_costs)

def run_voting(forecast_dicts, realised_k):
    """
    Voting: runs full backward induction under each forecast
    independently, then takes the plurality action at each node.
    """
    realised_dict = forecast_dicts[realised_k]

    path        = ["START"]
    stage_costs = []

    for stage in range(NUM_STAGES):
        current    = path[-1]
        candidates = next_layer_nodes(stage)

        # For each forecast, find its individually optimal next node
        votes = []
        for f in forecast_dicts:
            best_node = None
            best_cost = float("inf")
            for candidate in candidates:
                imm = f.get(current, {}).get(candidate, float("inf"))
                cache = {}
                ctg = _ctg_dp(candidate, stage + 1, f, cache)
                total = imm + ctg
                if total < best_cost:
                    best_cost = total
                    best_node = candidate
            votes.append(best_node)

        # Plurality: pick the candidate with the most votes
        from collections import Counter
        vote_counts = Counter(votes)
        best_node = vote_counts.most_common(1)[0][0]

        cost = realised_dict.get(current, {}).get(best_node, float("inf"))
        stage_costs.append(cost)
        path.append(best_node)

    return path, stage_costs, sum(stage_costs)


def _ctg_dp(node, stage, cost_dict, cache):
    """DP cost-to-go with local cache per forecast."""
    if node == "END" or stage > NUM_STAGES:
        return 0.0
    key = (node, stage)
    if key in cache:
        return cache[key]

    if stage == NUM_STAGES:
        next_nodes = ["END"]
    else:
        next_nodes = [f"S{stage}_N{j}" for j in range(NODES_PER_STAGE)]

    best = float("inf")
    for nxt in next_nodes:
        c = cost_dict.get(node, {}).get(nxt, float("inf"))
        f = _ctg_dp(nxt, stage + 1, cost_dict, cache)
        best = min(best, c + f)

    cache[key] = best
    return best

# ============================================================
# HINDSIGHT OPTIMAL
# ============================================================

def run_hindsight(realised_dict):
    """
    Backward induction on the realised cost dict.
    Guaranteed lower bound.
    """
    V         = {"END": 0.0}
    best_next = {}

    for t in range(len(LAYERS) - 2, -1, -1):
        for node in LAYERS[t]:
            best_cost = float("inf")
            best_node = None
            for nxt in LAYERS[t + 1]:
                c = realised_dict.get(node, {}).get(nxt, float("inf"))
                total = c + V.get(nxt, float("inf"))
                if total < best_cost:
                    best_cost = total
                    best_node = nxt
            V[node]         = best_cost
            best_next[node] = best_node

    path = ["START"]
    node = "START"
    while node != "END":
        node = best_next[node]
        path.append(node)

    stage_costs = []
    for t in range(NUM_STAGES):
        fn = path[t]
        tn = path[t+1]
        stage_costs.append(realised_dict.get(fn, {}).get(tn, float("inf")))

    return path, stage_costs, sum(stage_costs)

# ============================================================
# RUN EXPERIMENT
# ============================================================

def run_experiment(base_wind, noise_std=0.5, noise_grow=0.0,
                   n_seeds=30, label=""):
    """
    Runs greedy, adaptive, and hindsight over n_seeds.
    Returns results list and forecast dicts.
    """
    forecasts      = make_forecasts(base_wind, noise_std, noise_grow)
    forecast_dicts = [build_forecast_dict(forecasts[k], base_wind)
                      for k in range(NUM_FORECASTS)]

    greedy_costs    = []
    adaptive_costs  = []
    hindsight_costs = []
    all_results     = []

    for seed in range(n_seeds):
        rng        = np.random.RandomState(seed + 100)
        realised_k = rng.randint(0, NUM_FORECASTS)

        g_path, g_sc, g_cost = run_greedy(forecast_dicts, realised_k)
        a_path, a_sc, a_cost = run_adaptive(forecast_dicts, realised_k)
        h_path, h_sc, h_cost = run_hindsight(forecast_dicts[realised_k])

        greedy_costs.append(g_cost)
        adaptive_costs.append(a_cost)
        hindsight_costs.append(h_cost)

        all_results.append({
            "seed"       : seed,
            "realised_k" : realised_k,
            "greedy"     : {"path": g_path, "costs": g_sc, "total": g_cost},
            "adaptive"   : {"path": a_path, "costs": a_sc, "total": a_cost},
            "hindsight"  : {"path": h_path, "costs": h_sc, "total": h_cost},
        })

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {label}")
    print(f"Noise: ±{noise_std} m/s  grow: {noise_grow}/stage")
    print(f"{'='*60}")

    g_arr = np.array(greedy_costs)
    a_arr = np.array(adaptive_costs)
    h_arr = np.array(hindsight_costs)

    print(f"\n{'Policy':<12} {'Mean':>8} {'Std':>8} {'Gap%':>8}")
    print("-" * 40)
    for lbl2, arr in [("Hindsight", h_arr),
                      ("Adaptive",  a_arr),
                      ("Greedy",    g_arr)]:
        gap = np.mean((arr - h_arr) / h_arr * 100)
        print(f"{lbl2:<12} {np.mean(arr):>8.4f} "
              f"{np.std(arr):>8.4f} {gap:>7.1f}%")

    adap_wins   = np.sum(a_arr < g_arr)
    greedy_wins = np.sum(g_arr < a_arr)
    ties        = np.sum(np.abs(a_arr - g_arr) < 1e-6)
    print(f"\nAdaptive wins: {adap_wins}  "
          f"Greedy wins: {greedy_wins}  Ties: {ties}")

    # Show stage-0 divergence rate
    s0_diverge = sum(
        1 for r in all_results
        if r["greedy"]["path"][1] != r["adaptive"]["path"][1]
    )
    print(f"Stage-0 divergence (different first move): "
          f"{s0_diverge}/{n_seeds}")

    return all_results, forecasts, forecast_dicts


def run_all_fields(n_seeds=30):
    """Runs all three wind fields and prints a comparison summary."""

    fields = [
        ("Static Trap",   BASE_WIND_STATIC,  0.5, 0.0),
        ("Dynamic Trap",  BASE_WIND_DYNAMIC, DYNAMIC_NOISE_STD,
                                             DYNAMIC_NOISE_GROW),
        ("Max Divergence",BASE_WIND_DIVERGE, DIVERGE_NOISE_STD,
                                             DIVERGE_NOISE_GROW),
    ]

    all_field_results = {}

    for label, bw, ns, ng in fields:
        res, forecasts, fdicts = run_experiment(
            bw, ns, ng, n_seeds, label
        )
        all_field_results[label] = (res, forecasts, fdicts, bw)

    # Summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY ACROSS ALL FIELDS")
    print(f"{'='*60}")
    print(f"{'Field':<20} {'Adap wins':>10} {'Grdy wins':>10} "
          f"{'S0 diverge':>12}")
    print("-" * 55)

    for label, (res, _, _, _) in all_field_results.items():
        a_arr = np.array([r["adaptive"]["total"] for r in res])
        g_arr = np.array([r["greedy"]["total"]   for r in res])
        s0d   = sum(1 for r in res
                    if r["greedy"]["path"][1] != r["adaptive"]["path"][1])
        print(f"{label:<20} {np.sum(a_arr<g_arr):>10} "
              f"{np.sum(g_arr<a_arr):>10} {s0d:>12}")

    return all_field_results

def run_majority_trap(seed_idx=0):
    """
    Runs the ensemble majority trap for a single seed.
    All policies use the same forecast_dicts.
    Realised costs evaluated on REALISED_WIND (minority was right).
    """
    forecast_dicts, realised_dict = make_forecasts_majority_trap(seed=seed_idx)

    # Find which index in forecast_dicts is the realised one.
    # We pass a special realised_k=5 (the minority member).
    realised_k = 5  # index of the minority forecast

    g_path, g_sc, g_cost = run_greedy(forecast_dicts, realised_k)
    v_path, v_sc, v_cost = run_voting(forecast_dicts, realised_k)
    a_path, a_sc, a_cost = run_adaptive(forecast_dicts, realised_k)
    h_path, h_sc, h_cost = run_hindsight(realised_dict)

    print(f"\nEnsemble Majority Trap — Seed {seed_idx}")
    print(f"  Greedy   path: {g_path[1]}  total: {g_cost:.4f}")
    print(f"  Voting   path: {v_path[1]}  total: {v_cost:.4f}")
    print(f"  Adaptive path: {a_path[1]}  total: {a_cost:.4f}")
    print(f"  Hindsight:     {h_path[1]}  total: {h_cost:.4f}")

    result = {
        "seed"      : seed_idx,
        "realised_k": realised_k,
        "greedy"    : {"path": g_path, "costs": g_sc, "total": g_cost},
        "voting"    : {"path": v_path, "costs": v_sc, "total": v_cost},
        "adaptive"  : {"path": a_path, "costs": a_sc, "total": a_cost},
        "hindsight" : {"path": h_path, "costs": h_sc, "total": h_cost},
    }

    return result, forecast_dicts, realised_dict

# ============================================================
# VISUALISATION
# ============================================================

def wind_color(ws, vmin=2.0, vmax=20.0):
    t = np.clip((ws - vmin) / (vmax - vmin), 0, 1)
    r = 1.0 - t
    g = t
    return (r, g, 0.2)


def visualise(results, forecasts, forecast_dicts, base_wind,
              seed_idx=0, animate=True):
    """
    Interactive stage-by-stage visualisation for one seed.
    Shows:
      - Geographic node layout coloured by BASE wind speed
      - Greedy and adaptive paths with arrows
      - Panel showing immediate cost, penalty, cost-to-go
        for each candidate next node at the current stage
      - Cumulative cost tracking
    """
    r          = results[seed_idx]
    realised_k = r["realised_k"]
    g_res      = r["greedy"]
    a_res      = r["adaptive"]
    h_res      = r["hindsight"]
    fd         = forecast_dicts[realised_k]

    print(f"\nVisualising seed {seed_idx}  "
          f"(realised forecast {realised_k})")
    print(f"Greedy total:   {g_res['total']:.4f}")
    print(f"Adaptive total: {a_res['total']:.4f}")
    print(f"Hindsight total:{h_res['total']:.4f}")

    # Node x positions = longitudes, y positions = latitudes
    def node_xy(stage, node_idx):
        return STAGE_LONS[stage], NODE_LATS[node_idx]

    start_xy = (START_COORD[1], START_COORD[0])
    end_xy   = (END_COORD[1],   END_COORD[0])

    for stage in range(NUM_STAGES):

        fig = plt.figure(figsize=(20, 11))
        fig.suptitle(
            f"Synthetic Trap Field  |  Seed {seed_idx}  "
            f"|  Stage {stage} → {stage+1}  "
            f"|  Realised forecast: {realised_k}  "
            f"|  λ={LATERAL_PENALTY}",
            fontsize=11, fontweight="bold"
        )

        gs = GridSpec(2, 3, figure=fig,
                      hspace=0.45, wspace=0.35)

        ax_map  = fig.add_subplot(gs[:, 0:2])   # geographic map
        ax_tbl  = fig.add_subplot(gs[0, 2])      # cost table
        ax_cum  = fig.add_subplot(gs[1, 2])      # cumulative cost

        # ---- MAP ----
        ax_map.set_facecolor("#e8f4f8")
        ax_map.set_xlabel("Longitude (°W)", fontsize=9)
        ax_map.set_ylabel("Latitude (°N)",  fontsize=9)
        ax_map.set_title(
            f"Stage {stage}: choosing next move\n"
            f"Green = high wind (cheap), Red = low wind (expensive)",
            fontsize=9
        )

        # All faint arcs
        for s in range(NUM_STAGES - 1):
            for i in range(NODES_PER_STAGE):
                x1, y1 = node_xy(s, i)
                for j in range(NODES_PER_STAGE):
                    x2, y2 = node_xy(s+1, j)
                    ax_map.plot([x1,x2],[y1,y2],
                                color="lightgray", alpha=0.25,
                                linewidth=0.4, zorder=1)

        # START and END markers
        ax_map.scatter(*start_xy, s=300, color="gold",
                       edgecolors="black", zorder=6, marker="*")
        ax_map.text(start_xy[0], start_xy[1]+1.2,
                    "START", ha="center", fontsize=8, fontweight="bold")

        ax_map.scatter(*end_xy, s=300, color="gold",
                       edgecolors="black", zorder=6, marker="*")
        ax_map.text(end_xy[0], end_xy[1]+1.2,
                    "END", ha="center", fontsize=8, fontweight="bold")

        # Draw paths so far
        path_styles = {
            "greedy"  : ("#e34948", 2.5, "Greedy"),
            "adaptive": ("#2a78d6", 2.5, "Adaptive"),
            "hindsight":("#2c3e50", 1.5, "Hindsight"),
        }

        for pol, (col, lw, lbl) in path_styles.items():
            res  = r[pol]
            path = res["path"]
            for t in range(min(stage + 1, len(path) - 1)):
                fn = path[t]
                tn = path[t+1]

                if fn == "START":
                    x1, y1 = start_xy
                elif fn == "END":
                    x1, y1 = end_xy
                else:
                    si = int(fn.split("_N")[0][1:])
                    ni = int(fn.split("_N")[1])
                    x1, y1 = node_xy(si, ni)

                if tn == "END":
                    x2, y2 = end_xy
                elif tn == "START":
                    x2, y2 = start_xy
                else:
                    si = int(tn.split("_N")[0][1:])
                    ni = int(tn.split("_N")[1])
                    x2, y2 = node_xy(si, ni)

                offset = {"greedy": -0.3,
                          "adaptive": 0.3,
                          "hindsight": 0.0}[pol]

                ax_map.annotate(
                    "",
                    xy=(x2, y2+offset), xytext=(x1, y1+offset),
                    arrowprops=dict(
                        arrowstyle="->", color=col,
                        lw=lw, alpha=0.85
                    ),
                    zorder=4
                )

        # All stage nodes coloured by base_wind for THIS field
        for s in range(NUM_STAGES):
            for i in range(NODES_PER_STAGE):
                ws  = base_wind[i][s]
                col = wind_color(ws)
                x, y = node_xy(s, i)

                # Highlight current stage nodes larger
                size = 350 if s == stage else 200
                alpha = 1.0 if s in (stage, stage+1) else 0.5

                ax_map.scatter(x, y, s=size, color=col,
                               edgecolors="black", linewidths=0.8,
                               zorder=5, alpha=alpha)
                ax_map.text(x, y, f"N{i}\n{ws:.0f}",
                            ha="center", va="center",
                            fontsize=6, zorder=6, color="black")

        # Mark current positions
        for pol, (col, lw, lbl) in path_styles.items():
            path = r[pol]["path"]
            if len(path) > stage:
                cn = path[stage]
                if cn == "START":
                    cx, cy = start_xy
                else:
                    si = int(cn.split("_N")[0][1:])
                    ni = int(cn.split("_N")[1])
                    cx, cy = node_xy(si, ni)
                ax_map.scatter(cx, cy, s=500, color=col,
                               edgecolors="white", linewidths=2,
                               zorder=7, alpha=0.9,
                               marker="o")

        # Legend
        handles = [
            mpatches.Patch(color="#e34948", label="Greedy"),
            mpatches.Patch(color="#2a78d6", label="Adaptive"),
            mpatches.Patch(color="#2c3e50", label="Hindsight"),
            mpatches.Patch(color="green",   label="High wind"),
            mpatches.Patch(color="red",     label="Low wind"),
        ]
        ax_map.legend(handles=handles, fontsize=8,
                      loc="lower left")
        ax_map.grid(True, alpha=0.3, linewidth=0.4)

        # ---- COST TABLE ----
        ax_tbl.clear()
        ax_tbl.axis("off")

        # Compute options from each policy's current node
        g_cur = r["greedy"]["path"][stage]
        a_cur = r["adaptive"]["path"][stage]

        def option_rows(current_node, is_adaptive):
            rows    = []
            options = []

            for j in range(NODES_PER_STAGE):
                if stage < NUM_STAGES:
                    next_node = f"S{stage}_N{j}"
                else:
                    next_node = "END"

                if stage == 0 and current_node == "START":
                    next_node = f"S0_N{j}"

                imm_vals = []
                ctg_vals = []
                for fdk in forecast_dicts:
                    imm = fdk.get(current_node, {}).get(next_node, 999)
                    imm_vals.append(imm)
                    c = {}
                    ctg = _ctg_dp(next_node, stage+1, fdk, c)
                    ctg_vals.append(ctg)

                cur_node_idx = (int(current_node.split("_N")[1])
                                if "_N" in current_node else MID_IDX)
                nxt_idx = j
                pen = lateral_pen(cur_node_idx, nxt_idx)

                avg_imm = np.mean(imm_vals)
                avg_ctg = np.mean(ctg_vals)

                options.append((next_node, avg_imm, pen,
                                avg_imm - pen, avg_ctg,
                                avg_imm + avg_ctg))

            return options

        # Adaptive table
        a_options = option_rows(a_cur, True)
        a_pick    = min(a_options, key=lambda x: x[5])[0]
        g_options = option_rows(g_cur, False)
        g_pick    = min(g_options, key=lambda x: x[1])[0]

        title = (
            f"Stage {stage} Decision\n"
            f"Greedy from {g_cur} → picks {g_pick}\n"
            f"Adaptive from {a_cur} → picks {a_pick}"
        )
        ax_tbl.set_title(title, fontsize=8, pad=4)

        col_labels = ["Node", "Wind\nCost", "Penalty",
                      "Avg\nImm", "Avg\nCtG", "Mean\nTotal"]
        rows_data  = [
            [opt[0],
             f"{opt[3]:.4f}",
             f"{opt[2]:.4f}",
             f"{opt[1]:.4f}",
             f"{opt[4]:.4f}",
             f"{opt[5]:.4f}"]
            for opt in sorted(a_options, key=lambda x: x[5])
        ]

        tbl = ax_tbl.table(
            cellText=rows_data,
            colLabels=col_labels,
            loc="center", cellLoc="center"
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.5)
        tbl.scale(1, 1.5)

        # Colour rows
        for row_i, opt in enumerate(
            sorted(a_options, key=lambda x: x[5])
        ):
            node = opt[0]
            bg   = "white"
            if node == a_pick == g_pick:
                bg = "#d5f5e3"
            elif node == a_pick:
                bg = "#d6eaf8"
            elif node == g_pick:
                bg = "#fdebd0"
            for ci in range(len(col_labels)):
                tbl[row_i+1, ci].set_facecolor(bg)

        ax_tbl.text(
            0.02, 0.01,
            "Blue=Adaptive  Orange=Greedy  Green=Both",
            transform=ax_tbl.transAxes,
            fontsize=7, va="bottom", color="dimgray"
        )

        # ---- CUMULATIVE COST ----
        ax_cum.clear()
        x_cum = list(range(stage + 2))

        for pol, (col, lw, lbl) in path_styles.items():
            sc  = r[pol]["costs"][:stage+1]
            cum = [0.0] + list(np.cumsum(sc))
            ax_cum.plot(x_cum[:len(cum)], cum,
                        color=col, linewidth=lw,
                        marker="o", markersize=5,
                        label=f"{lbl} ({r[pol]['total']:.4f})")

        ax_cum.set_xlabel("Stage", fontsize=8)
        ax_cum.set_ylabel("Cumulative cost", fontsize=8)
        ax_cum.set_title("Cumulative Cost", fontsize=9)
        ax_cum.legend(fontsize=7)
        ax_cum.grid(True, alpha=0.3)

        # Wind profile inset
        ax_inset = ax_cum.inset_axes([0.55, 0.55, 0.43, 0.43])
        for i in range(NODES_PER_STAGE):
            ws_arr = [base_wind[i][s] for s in range(NUM_STAGES)]
            ax_inset.plot(range(NUM_STAGES), ws_arr,
                          marker=".", linewidth=1.2,
                          label=f"N{i} {NODE_LATS[i]}°N")
        ax_inset.axvline(stage, color="black",
                         linewidth=1.0, linestyle="--", alpha=0.6)
        ax_inset.set_title("Wind profile", fontsize=6)
        ax_inset.tick_params(labelsize=5)
        ax_inset.legend(fontsize=5, loc="upper right")

        plt.show()

        if animate and stage < NUM_STAGES - 1:
            input(f"  Press Enter for stage {stage+1}...")

    # ---- Final summary ----
    fig2, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig2.suptitle(
        f"Final Summary — Seed {seed_idx}  λ={LATERAL_PENALTY}  "
        f"Realised forecast {realised_k}",
        fontsize=11, fontweight="bold"
    )

    # Cumulative
    x = list(range(NUM_STAGES + 1))
    for pol, (col, lw, lbl) in path_styles.items():
        sc  = r[pol]["costs"]
        cum = [0.0] + list(np.cumsum(sc))
        axes[0].plot(x, cum, color=col, linewidth=lw,
                     marker="o",
                     label=f"{lbl} ({r[pol]['total']:.4f})")
    axes[0].set_title("Cumulative Cost")
    axes[0].set_xlabel("Stage")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Wind field heatmap
    wind_matrix = np.array([base_wind[i]
                            for i in range(NODES_PER_STAGE)])
    im = axes[1].imshow(wind_matrix, cmap="RdYlGn",
                        aspect="auto", vmin=2, vmax=20)
    axes[1].set_xlabel("Stage")
    axes[1].set_ylabel("Node (0=south 20°N, 4=north 60°N)")
    axes[1].set_title("BASE Wind Speed (m/s)\nGreen=fast, Red=slow")
    axes[1].set_xticks(range(NUM_STAGES))
    axes[1].set_yticks(range(NODES_PER_STAGE))
    axes[1].set_yticklabels([f"N{i} {NODE_LATS[i]}°N"
                             for i in range(NODES_PER_STAGE)])
    plt.colorbar(im, ax=axes[1], label="Wind speed (m/s)")

    # Path comparison on wind heatmap
    for pol, (col, lw, lbl) in path_styles.items():
        path = r[pol]["path"]
        xs, ys = [], []
        for t, node in enumerate(path):
            if node in ("START", "END"):
                continue
            s = int(node.split("_N")[0][1:])
            n = int(node.split("_N")[1])
            xs.append(s)
            ys.append(n)
        axes[1].plot(xs, ys, color=col, linewidth=lw+0.5,
                     marker="o", markersize=6,
                     label=lbl, zorder=5)
    axes[1].legend(fontsize=7)

    # Cost breakdown: wind vs penalty
    pols   = ["greedy", "adaptive", "hindsight"]
    labels = ["Greedy", "Adaptive", "Hindsight"]
    colors = ["#e34948", "#2a78d6", "#2c3e50"]
    wind_c = []
    pen_c  = []

    for pol in pols:
        path = r[pol]["path"]
        sc   = r[pol]["costs"]
        wt, pt = 0.0, 0.0
        for t in range(len(sc)):
            fn = path[t]
            tn = path[t+1]
            if fn == "START":
                fi, ti = MID_IDX, int(tn.split("_N")[1])
            elif tn == "END":
                fi, ti = int(fn.split("_N")[1]), MID_IDX
            elif fn == "END" or tn == "START":
                fi, ti = MID_IDX, MID_IDX
            else:
                fi = int(fn.split("_N")[1])
                ti = int(tn.split("_N")[1])
            pen = lateral_pen(fi, ti)
            pt += pen
            wt += sc[t] - pen
        wind_c.append(wt)
        pen_c.append(pt)

    x3 = np.arange(len(pols))
    axes[2].bar(x3, wind_c, color=colors, alpha=0.85,
                label="Wind cost")
    axes[2].bar(x3, pen_c, bottom=wind_c, color=colors,
                alpha=0.4, hatch="//", label="Lateral penalty")
    axes[2].set_xticks(x3)
    axes[2].set_xticklabels(labels)
    axes[2].set_title("Cost Decomposition\n(hatched = penalty)")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.show()

def visualise_report(results, forecasts, forecast_dicts, base_wind,
                     seed_idx=0, save_path="trap2_report.png"):
    """
    Single-panel report figure: geographic map only, high resolution.
    Shows Greedy, Adaptive, Voting, and Hindsight paths.
    No RHS table panel.
    """
    r          = results[seed_idx]
    realised_k = r["realised_k"]

    def node_xy(stage, node_idx):
        return STAGE_LONS[stage], NODE_LATS[node_idx]

    start_xy = (START_COORD[1], START_COORD[0])
    end_xy   = (END_COORD[1],   END_COORD[0])

    # Use final stage (full paths visible)
    stage = NUM_STAGES - 1

    fig, (ax_map, ax_cum) = plt.subplots(
        1, 2, figsize=(14, 6),
        gridspec_kw={"width_ratios": [2, 1]}
    )

    fig.suptitle(
        f"Corridor Trap  |  Seed {seed_idx}  "
        f"|  λ={LATERAL_PENALTY}",
        fontsize=12, fontweight="bold"
    )

    path_styles = {
        "greedy"   : ("#e34948", 2.5, "Greedy"),
        "adaptive" : ("#2a78d6", 2.5, "Adaptive"),
        "hindsight": ("#2c3e50", 1.5, "Hindsight"),
    }

    # ---- MAP ----
    ax_map.set_facecolor("#e8f4f8")
    ax_map.set_xlabel("Longitude (°W)", fontsize=10)
    ax_map.set_ylabel("Latitude (°N)",  fontsize=10)
    ax_map.set_title(
        "Green = high wind (cheap), Red = low wind (expensive)",
        fontsize=9
    )

    # Faint background arcs
    for s in range(NUM_STAGES - 1):
        for i in range(NODES_PER_STAGE):
            x1, y1 = node_xy(s, i)
            for j in range(NODES_PER_STAGE):
                x2, y2 = node_xy(s+1, j)
                ax_map.plot([x1,x2],[y1,y2],
                            color="lightgray", alpha=0.2,
                            linewidth=0.4, zorder=1)

    # START / END
    ax_map.scatter(*start_xy, s=300, color="gold",
                   edgecolors="black", zorder=6, marker="*")
    ax_map.text(start_xy[0], start_xy[1]+1.5, "START",
                ha="center", fontsize=9, fontweight="bold")
    ax_map.scatter(*end_xy, s=300, color="gold",
                   edgecolors="black", zorder=6, marker="*")
    ax_map.text(end_xy[0], end_xy[1]+1.5, "END",
                ha="center", fontsize=9, fontweight="bold")

    # Full paths
    offsets = {"greedy": -0.5,
               "adaptive": 0.5, "hindsight": 0.0}
    for pol, (col, lw, lbl) in path_styles.items():
        if pol not in r:
            continue
        path = r[pol]["path"]
        off  = offsets[pol]
        for t in range(len(path) - 1):
            fn, tn = path[t], path[t+1]
            x1 = start_xy[0] if fn == "START" else node_xy(int(fn.split("_N")[0][1:]), int(fn.split("_N")[1]))[0]
            y1 = start_xy[1] if fn == "START" else node_xy(int(fn.split("_N")[0][1:]), int(fn.split("_N")[1]))[1]
            if tn == "END":
                x2, y2 = end_xy
            else:
                x2 = node_xy(int(tn.split("_N")[0][1:]), int(tn.split("_N")[1]))[0]
                y2 = node_xy(int(tn.split("_N")[0][1:]), int(tn.split("_N")[1]))[1]
            ax_map.annotate("", xy=(x2, y2+off), xytext=(x1, y1+off),
                            arrowprops=dict(arrowstyle="->", color=col,
                                            lw=lw, alpha=0.85), zorder=4)

    # Nodes
    for s in range(NUM_STAGES):
        for i in range(NODES_PER_STAGE):
            ws  = base_wind[i][s]
            col = wind_color(ws)
            x, y = node_xy(s, i)
            ax_map.scatter(x, y, s=220, color=col,
                           edgecolors="black", linewidths=0.8,
                           zorder=5)
            ax_map.text(x, y, f"N{i}\n{ws:.0f}",
                        ha="center", va="center",
                        fontsize=6, zorder=6)

    handles = [
        mpatches.Patch(color="#e34948", label=f"Greedy ({r['greedy']['total']:.4f})"),
        mpatches.Patch(color="#2a78d6", label=f"Adaptive ({r['adaptive']['total']:.4f})"),
        mpatches.Patch(color="#2c3e50", label=f"Hindsight ({r['hindsight']['total']:.4f})"),
        mpatches.Patch(color="green",   label="High wind"),
        mpatches.Patch(color="red",     label="Low wind"),
    ]
    ax_map.legend(handles=handles, fontsize=8, loc="lower left")
    ax_map.grid(True, alpha=0.3, linewidth=0.4)

    # ---- CUMULATIVE COST ----
    x = list(range(NUM_STAGES + 1))
    for pol, (col, lw, lbl) in path_styles.items():
        if pol not in r:
            continue
        cum = [0.0] + list(np.cumsum(r[pol]["costs"]))
        ax_cum.plot(x, cum, color=col, linewidth=lw,
                    marker="o", markersize=4,
                    label=f"{lbl} ({r[pol]['total']:.4f})")

    ax_cum.set_xlabel("Stage", fontsize=10)
    ax_cum.set_ylabel("Cumulative cost", fontsize=10)
    ax_cum.set_title("Cumulative Cost", fontsize=10)
    ax_cum.legend(fontsize=8)
    ax_cum.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved to {save_path}")
    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SYNTHETIC WIND FIELD EXPERIMENTS")
    print(f"Stages: {NUM_STAGES}  Nodes: {NODES_PER_STAGE}")
    print(f"Lateral penalty: {LATERAL_PENALTY}")
    print(f"START: lat={START_COORD[0]}  lon={START_COORD[1]}")
    print(f"END:   lat={END_COORD[0]}    lon={END_COORD[1]}")
    print("=" * 60)

    # Print all three wind field designs
    for fname, bw in [("Static Trap",    BASE_WIND_STATIC),
                      ("Dynamic Trap",   BASE_WIND_DYNAMIC),
                      ("Max Divergence", BASE_WIND_DIVERGE)]:
        print(f"\n{fname}:")
        print(f"  {'Node':<8}  "
              + "  ".join(f"S{s}" for s in range(NUM_STAGES)))
        for i in range(NODES_PER_STAGE):
            vals = "  ".join(f"{bw[i][s]:>4.0f}"
                             for s in range(NUM_STAGES))
            print(f"  N{i} ({NODE_LATS[i]:>2.0f}°N)  {vals}")

    # Run all three fields
    all_field_results = run_all_fields(n_seeds=30)

    # Visualise — pick which field and seed to inspect
    FIELD_TO_VIS = "Static Trap"   # change to "Static Trap" or "Dynamic Trap"
    SEED_TO_VIS  = 1

    res, forecasts, fdicts, bw = all_field_results[FIELD_TO_VIS]

    print(f"\nVisualising: {FIELD_TO_VIS}  seed={SEED_TO_VIS}")
    visualise_report(res, forecasts, fdicts, bw,
              seed_idx=SEED_TO_VIS, save_path="trap1_report.png")
    