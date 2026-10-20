import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import copy
import os
import sys
import pickle

from policy_grib import (
    layers,
    generate_forecasts,
    choose_next_node,
    NUM_FORECASTS,
    NUM_STAGES,
    STAGE_DATES,
    MODELS,
    NODES_PER_STAGE,
)


# ============================================================
# PARAMETERS
# ============================================================

SEEDS     = 100      # Keep low initially — GRIB loading is slow
BASE_SEED = 100

# ============================================================
# LATERAL MOVEMENT PENALTY
# ============================================================

# 0.0 = disabled
# Larger values increasingly discourage large lateral movements.
LATERAL_PENALTY_SCALE = 0.5

# ============================================================
# KEY DIFFERENCE FROM SYNTHETIC EXPERIMENT
# ============================================================
#
# In the synthetic version, forecasts were EVOLVED at each stage
# using generate_forecasts(parent_forecasts=...) — the same
# correlated noise model updated the forecast distribution.
#
# In the GRIB version, each stage has its OWN set of 6 real
# GRIB files (one per day, one per model). generate_forecasts(stage)
# loads the real files for that day. There is no synthetic
# evolution — the forecast update IS the new day's GRIB data.
#
# forecast_snapshots[stage] = the 6 forecasts loaded for that day
# realised_sequence[stage]  = which model (0-5) was "realised"
# arc cost realised          = forecast_snapshots[stage][k][from][to]
#
# All policies share the same forecast_snapshots and
# realised_sequence so comparisons are fair.
# ============================================================

# ============================================================
# GRID CONFIGURATION
# ============================================================

NODE_LATS  = [0.0, 5.0, 10.0, 15.0, 20.0]
STAGE_LONS = [-95, -80, -65, -50, -35, -20, -5, 0]

def configure_grid(node_lats, stage_lons):
    """
    Rebuild all node coordinates for a new routing corridor.
    """

    global NODE_LATS
    global STAGE_LONS
    global NODE_COORDS

    NODE_LATS = list(node_lats)
    STAGE_LONS = list(stage_lons)

    NODE_COORDS = {}

    for s in range(NUM_STAGES):
        for i in range(NODES_PER_STAGE):
            NODE_COORDS[f"S{s}_N{i}"] = (
                NODE_LATS[i],
                STAGE_LONS[s],
            )

def set_cost_function(name):
    global ACTIVE_COST_FN
    ACTIVE_COST_FN = name

# ============================================================
# LATERAL MOVEMENT PENALTY
# ============================================================

def lateral_movement_penalty(from_node, to_node):
    """
    Penalise large changes in node index between consecutive stages.

    Penalty = LATERAL_PENALTY_SCALE * max(0, |i - j| - 1)^2

    Moving 0 positions: no penalty
    Moving 1 position : no penalty
    Moving 2 positions: scale * 1^2
    Moving 3 positions: scale * 2^2
    Moving 4 positions: scale * 3^2

    START and END are artificial nodes and receive no penalty.
    """

    if LATERAL_PENALTY_SCALE == 0.0:
        return 0.0

    # START -> first stage and final stage -> END
    if from_node in ("START", "END") or to_node in ("START", "END"):
        return 0.0

    # Extract node indices from names such as:
    # S0_N0, S1_N3, S5_N4
    try:
        i = int(from_node.split("_N")[1])
        j = int(to_node.split("_N")[1])
    except (ValueError, IndexError):
        return 0.0

    index_difference = abs(i - j)

    return (
        LATERAL_PENALTY_SCALE
        * max(0, index_difference - 1) ** 2
    )


# ============================================================
# PRE-LOAD ALL STAGE FORECASTS ONCE
# (avoids re-reading GRIB files for every seed)
# ============================================================

print("Pre-loading all GRIB forecasts...")
ALL_STAGE_FORECASTS = []

for s in range(NUM_STAGES):

    stage_forecasts = generate_forecasts(stage=s)

    # Apply lateral movement penalty to every forecast
    for forecast in stage_forecasts:

        for from_node in forecast:
            for to_node in forecast[from_node]:

                forecast[from_node][to_node] += lateral_movement_penalty(
                    from_node,
                    to_node
                )

    ALL_STAGE_FORECASTS.append(stage_forecasts)

print("All stages loaded.\n")


# =============================================================
# Helpers for Simulation Driver
# =============================================================

import pickle

RESULTS_FILE = "forecast_policy_routes.pkl"

def save_results(results):

    with open(RESULTS_FILE, "wb") as f:
        pickle.dump(results, f)

    print(f"Saved routes to {RESULTS_FILE}")

# ============================================================
# GRID CONFIGURATION
# ============================================================

NODE_LATS  = [-45, -40, -35, -30, -25]
STAGE_LONS = [100,110,120,130,140,150]

def configure_grid(node_lats, stage_lons):
    """
    Rebuild all node coordinates for a new routing corridor.
    """

    global NODE_LATS
    global STAGE_LONS
    global NODE_COORDS

    NODE_LATS = list(node_lats)
    STAGE_LONS = list(stage_lons)

    NODE_COORDS = {}

    for s in range(NUM_STAGES):
        for i in range(NODES_PER_STAGE):
            NODE_COORDS[f"S{s}_N{i}"] = (
                NODE_LATS[i],
                STAGE_LONS[s],
            )

def set_cost_function(name):
    global ACTIVE_COST_FN
    ACTIVE_COST_FN = name

# ============================================================
# LATERAL MOVEMENT PENALTY
# ============================================================

def lateral_movement_penalty(from_node, to_node):
    """
    Penalise large changes in node index between consecutive stages.

    Penalty = LATERAL_PENALTY_SCALE * max(0, |i - j| - 1)^2

    Moving 0 positions: no penalty
    Moving 1 position : no penalty
    Moving 2 positions: scale * 1^2
    Moving 3 positions: scale * 2^2
    Moving 4 positions: scale * 3^2

    START and END are artificial nodes and receive no penalty.
    """

    if LATERAL_PENALTY_SCALE == 0.0:
        return 0.0

    # START -> first stage and final stage -> END
    if from_node in ("START", "END") or to_node in ("START", "END"):
        return 0.0

    # Extract node indices from names such as:
    # S0_N0, S1_N3, S5_N4
    try:
        i = int(from_node.split("_N")[1])
        j = int(to_node.split("_N")[1])
    except (ValueError, IndexError):
        return 0.0

    index_difference = abs(i - j)

    return (
        LATERAL_PENALTY_SCALE
        * max(0, index_difference - 1) ** 2
    )

# ============================================================
# SHARED COST-TO-GO HELPER
# ============================================================

def _cost_to_go(current_node, current_stage, forecast):
    if current_stage == len(layers) - 1:
        return 0.0
    return min(
        forecast[current_node][next_node] +
        _cost_to_go(next_node, current_stage + 1, forecast)
        for next_node in layers[current_stage + 1]
    )

# ============================================================
# OPTIMAL PATH UNDER A SINGLE FORECAST
# ============================================================

def compute_optimal_path(forecast):

    V = {
        "END": 0.0
    }

    best_next = {}

    # Work backwards through the layers
    for layer_idx in range(len(layers) - 2, -1, -1):

        for node in layers[layer_idx]:

            best_cost = float("inf")
            best_node = None

            for next_node in layers[layer_idx + 1]:

                # ------------------------------------------------
                # Terminal END node
                # ------------------------------------------------
                if next_node == "END":

                    arc_cost = 0.0
                    future_cost = 0.0

                else:

                    arc_cost = forecast[node][next_node]
                    future_cost = V[next_node]

                total_cost = arc_cost + future_cost

                if total_cost < best_cost:

                    best_cost = total_cost
                    best_node = next_node

            V[node] = best_cost
            best_next[node] = best_node

    # ------------------------------------------------------------
    # Reconstruct route
    # ------------------------------------------------------------

    route = ["START"]

    node = "START"

    while node != "END":

        node = best_next[node]
        route.append(node)

    return route, V["START"]

# ============================================================
# HINDSIGHT OPTIMAL
# ============================================================

def compute_hindsight_optimal(forecast_snapshots, realised_sequence):
    """
    Computes the optimal path with perfect hindsight — i.e. knowing
    exactly which model was realised at every stage.

    Layer/snapshot index mapping:
        layer 0 = START       -> stage 0 transition -> snapshot[0]
        layer 1 = S0 nodes    -> stage 1 transition -> snapshot[1]
        ...
        layer 5 = S4 nodes    -> stage 5 transition -> snapshot[5]
        layer 6 = S5 nodes    -> END (no further transition)
        layer 7 = END

    The DP works backward from END. At each layer_idx we need the
    realised cost matrix for transitions FROM that layer, which is
    forecast_snapshots[stage] where stage = layer_idx (since layer 0
    = START = stage 0, layer 1 = S0 = stage 1 decision, etc.)
    But layer 0 (START) transitions use snapshot[0], and the mapping
    is simply: snap_idx = layer_idx, clamped to NUM_STAGES - 1.

    Wait — let's be precise:
        Decision at stage t means: currently AT layers[t], moving TO layers[t+1]
        Cost comes from forecast_snapshots[t][realised_k]
        t ranges 0..NUM_STAGES-1

    So in the backward DP:
        layer_idx = t (the stage of the decision)
        snap_idx  = t = layer_idx
        BUT layer_idx goes up to len(layers)-2 = 6, while snap_idx max = 5
        Layer 6 = S5 nodes transitioning to END — this is stage 5 (t=5)
        So snap_idx = min(layer_idx, NUM_STAGES - 1) is actually correct
        IF layer_idx maps directly to stage index t.

    The layers list is:
        layers[0] = ["START"]
        layers[1] = ["S0_N0",..]   <- stage 0 nodes
        layers[2] = ["S1_N0",..]   <- stage 1 nodes
        ...
        layers[6] = ["S5_N0",..]   <- stage 5 nodes
        layers[7] = ["END"]

    Decision at stage t: AT layers[t], choose from layers[t+1]
    So layer_idx in the DP = t (stage of decision), snap_idx = t.
    layer_idx goes 0..6, but snapshots only go 0..5.
    layer_idx=6 means S5 nodes -> END, which is stage 5, snap_idx=5. ✓
    So: snap_idx = min(layer_idx, NUM_STAGES - 1) IS correct.

    The real issue was something else — let's rebuild cleanly using
    stage index directly to avoid any confusion.
    """

    V         = {}
    best_next = {}
    V["END"]  = 0.0

    # Work backwards through ALL layer transitions.
    # layers[0]=START, layers[1]=S0,..,layers[6]=S5, layers[7]=END
    # Transition FROM layers[t] TO layers[t+1] uses snapshot[min(t, NUM_STAGES-1)]
    #
    # t=6: S5 -> END    uses snapshot[5]  (last snapshot)
    # t=5: S4 -> S5     uses snapshot[5]
    # t=4: S3 -> S4     uses snapshot[4]
    # ...
    # t=0: START -> S0  uses snapshot[0]
    #
    # Note: t=5 and t=6 both use snapshot[5] because there are 6 stages
    # but 7 layer transitions (including START->S0 and S5->END).

    for t in range(len(layers) - 2, -1, -1):

        snap_idx    = min(t, NUM_STAGES - 1)
        realised_k  = realised_sequence[snap_idx]
        cost_matrix = forecast_snapshots[snap_idx][realised_k]

        for node in layers[t]:
            best_cost = float("inf")
            best_node = None
            for next_node in layers[t + 1]:
                cost = cost_matrix[node][next_node] + V[next_node]
                if cost < best_cost:
                    best_cost = cost
                    best_node = next_node
            V[node]         = best_cost
            best_next[node] = best_node

    path = ["START"]
    node = "START"
    while node != "END":
        node = best_next[node]
        path.append(node)

    stage_costs = []
    for t in range(NUM_STAGES):
        k         = realised_sequence[t]
        from_node = path[t]
        to_node   = path[t + 1]
        stage_costs.append(forecast_snapshots[t][k][from_node][to_node])

    total_cost = sum(stage_costs)
    return path, total_cost, stage_costs

# ============================================================
# REALISE A PATH UNDER SHARED FORECAST SNAPSHOTS
# ============================================================

def realise_path(path, forecast_snapshots, realised_sequence):

    total_cost = 0.0
    stage_costs = []

    weather_step = 0

    for i in range(len(path) - 1):

        from_node = path[i]
        to_node = path[i + 1]

        # --------------------------------------------------
        # START is an artificial source node
        # --------------------------------------------------

        if from_node == "START":
            continue

        # --------------------------------------------------
        # END is an artificial sink node
        # --------------------------------------------------

        if to_node == "END":
            continue

        # --------------------------------------------------
        # This is a genuine weather arc
        # --------------------------------------------------

        realised_k = realised_sequence[weather_step]

        cost = (
            forecast_snapshots[weather_step]
            [realised_k]
            [from_node]
            [to_node]
        )

        total_cost += cost
        stage_costs.append(cost)

        weather_step += 1

    return total_cost, stage_costs

# ============================================================
# GENERATE REALISED SEQUENCE FOR A SEED
# (which model index was "realised" at each stage)
# ============================================================

def make_realised_sequence(seed):
    rng = random.Random(seed)
    return [rng.randint(0, NUM_FORECASTS - 1) for _ in range(NUM_STAGES)]

# ============================================================
# ADAPTIVE POLICY
# ============================================================

def run_adaptive(seed, forecast_snapshots, realised_sequence):
    """
    Adaptive policy using GRIB forecasts.
    At each stage loads the pre-computed forecasts for that day,
    runs choose_next_node, then pays the realised cost.
    """

    stdout_backup = sys.stdout
    sys.stdout    = open(os.devnull, "w")

    try:
        current_node = "START"
        stage        = 0
        path         = [current_node]
        total_cost   = 0.0
        stage_costs  = []

        while current_node != "END" and stage < NUM_STAGES:

            forecasts = forecast_snapshots[stage]

            next_node, _, _ = choose_next_node(
                current_node, stage, forecasts
            )

            realised_k = realised_sequence[stage]
            cost       = forecast_snapshots[stage][realised_k][current_node][next_node]
            total_cost += cost
            stage_costs.append(cost)

            current_node = next_node
            path.append(current_node)
            stage += 1

        return total_cost, path, stage_costs

    finally:
        sys.stdout.close()
        sys.stdout = stdout_backup

# ============================================================
# DISCOUNTED ADAPTIVE POLICY
# ============================================================

def run_discounted_adaptive(
    forecast_snapshots,
    realised_sequence
):
    """
    Discounted Adaptive policy.

    Same basic structure as Adaptive, but uses
    choose_next_node_discounted().

    The realised route cost is NOT discounted.
    Discounting is used only when evaluating candidate
    routes for making the decision.

    This is important: the policy is being evaluated under
    the same realised weather as all other policies.
    """

    from policy_grib import choose_next_node_discounted

    current_node = "START"
    stage        = 0
    total_cost   = 0.0
    path         = [current_node]
    stage_costs  = []

    while current_node != "END" and stage < NUM_STAGES:

        forecasts = forecast_snapshots[stage]

        next_node, _, _ = choose_next_node_discounted(
            current_node,
            stage,
            forecasts
        )

        # ----------------------------------------------------
        # Pay the ACTUAL realised cost.
        #
        # The discounting is only used for the decision,
        # not for measuring the policy's eventual performance.
        # ----------------------------------------------------
        realised_k = realised_sequence[stage]

        cost = (
            forecast_snapshots[stage]
            [realised_k]
            [current_node]
            [next_node]
        )

        total_cost += cost
        stage_costs.append(cost)

        current_node = next_node
        path.append(current_node)
        stage += 1

    return total_cost, path, stage_costs

# ============================================================
# ORACLE RUNS — one per initial forecast at stage 0
# ============================================================

def run_oracles(forecast_snapshots, realised_sequence):
    """
    6 oracle paths, each committed at t=0 under one initial forecast.
    Realised under the shared forecast snapshots.
    """
    initial_forecasts = forecast_snapshots[0]
    oracle_costs      = []
    oracle_paths      = []

    for k in range(NUM_FORECASTS):
        path, _ = compute_optimal_path(initial_forecasts[k])
        cost, _ = realise_path(path, forecast_snapshots, realised_sequence)
        oracle_costs.append(cost)
        oracle_paths.append(path)

    return oracle_costs, oracle_paths

# ============================================================
# POLICY 1 — VOTING
# ============================================================

def run_voting(forecast_snapshots, realised_sequence):

    current_node = "START"
    stage        = 0
    total_cost   = 0.0
    path         = [current_node]
    stage_costs  = []

    while current_node != "END" and stage < NUM_STAGES:

        forecasts  = forecast_snapshots[stage]
        next_layer = layers[stage + 1]

        vote_count = {node: 0 for node in next_layer}
        for forecast in forecasts:
            path_f, _ = compute_optimal_path(forecast)
            vote_count[path_f[stage + 1]] += 1

        max_votes  = max(vote_count.values())
        candidates = [n for n, v in vote_count.items() if v == max_votes]
        next_node  = (
            candidates[0] if len(candidates) == 1
            else min(candidates,
                     key=lambda n: np.mean([f[current_node][n] for f in forecasts]))
        )

        realised_k = realised_sequence[stage]
        cost       = forecast_snapshots[stage][realised_k][current_node][next_node]
        total_cost += cost
        stage_costs.append(cost)

        current_node = next_node
        path.append(current_node)
        stage += 1

    return total_cost, path, stage_costs

# ============================================================
# POLICY 2 — GREEDY
# ============================================================

def run_greedy(forecast_snapshots, realised_sequence):

    current_node = "START"
    stage        = 0
    total_cost   = 0.0
    path         = [current_node]
    stage_costs  = []

    while current_node != "END" and stage < NUM_STAGES:

        forecasts  = forecast_snapshots[stage]
        next_layer = layers[stage + 1]

        next_node = min(
            next_layer,
            key=lambda n: np.mean([f[current_node][n] for f in forecasts])
        )

        realised_k = realised_sequence[stage]
        cost       = forecast_snapshots[stage][realised_k][current_node][next_node]
        total_cost += cost
        stage_costs.append(cost)

        current_node = next_node
        path.append(current_node)
        stage += 1

    return total_cost, path, stage_costs

# ============================================================
# POLICY 3 — AVERAGED ARC
# ============================================================

def run_averaged_arc(forecast_snapshots, realised_sequence):

    current_node = "START"
    stage        = 0
    total_cost   = 0.0
    path         = [current_node]
    stage_costs  = []

    while current_node != "END" and stage < NUM_STAGES:

        forecasts    = forecast_snapshots[stage]
        avg_forecast = {
            from_node: {
                to_node: np.mean([f[from_node][to_node] for f in forecasts])
                for to_node in forecasts[0][from_node]
            }
            for from_node in forecasts[0]
        }

        V         = {}
        best_next = {}
        V["END"]  = 0.0
        for stage_idx in range(len(layers) - 2, -1, -1):
            for node in layers[stage_idx]:
                best_cost = float("inf")
                best_node = None
                for next_node in layers[stage_idx + 1]:
                    cost = avg_forecast[node][next_node] + V[next_node]
                    if cost < best_cost:
                        best_cost = cost
                        best_node = next_node
                V[node]         = best_cost
                best_next[node] = best_node

        next_node = best_next[current_node]

        realised_k = realised_sequence[stage]
        cost       = forecast_snapshots[stage][realised_k][current_node][next_node]
        total_cost += cost
        stage_costs.append(cost)

        current_node = next_node
        path.append(current_node)
        stage += 1

    return total_cost, path, stage_costs

# ============================================================
# POLICY 4 — MINIMAX
# ============================================================

def run_minimax(forecast_snapshots, realised_sequence):

    current_node = "START"
    stage        = 0
    total_cost   = 0.0
    path         = [current_node]
    stage_costs  = []

    while current_node != "END" and stage < NUM_STAGES:

        forecasts  = forecast_snapshots[stage]
        next_layer = layers[stage + 1]

        best_node     = None
        best_max_cost = float("inf")

        for candidate in next_layer:
            worst_case = max(
                forecast[current_node][candidate] +
                _cost_to_go(candidate, stage + 1, forecast)
                for forecast in forecasts
            )
            if worst_case < best_max_cost:
                best_max_cost = worst_case
                best_node     = candidate

        realised_k = realised_sequence[stage]
        cost       = forecast_snapshots[stage][realised_k][current_node][best_node]
        total_cost += cost
        stage_costs.append(cost)

        current_node = best_node
        path.append(current_node)
        stage += 1

    return total_cost, path, stage_costs

# ============================================================
# POLICY 5 — OPTIMISTIC
# ============================================================

def run_optimistic(forecast_snapshots, realised_sequence):

    current_node = "START"
    stage        = 0
    total_cost   = 0.0
    path         = [current_node]
    stage_costs  = []

    while current_node != "END" and stage < NUM_STAGES:

        forecasts  = forecast_snapshots[stage]
        next_layer = layers[stage + 1]

        best_node     = None
        best_min_cost = float("inf")

        for candidate in next_layer:
            best_case = min(
                forecast[current_node][candidate] +
                _cost_to_go(candidate, stage + 1, forecast)
                for forecast in forecasts
            )
            if best_case < best_min_cost:
                best_min_cost = best_case
                best_node     = candidate

        realised_k = realised_sequence[stage]
        cost       = forecast_snapshots[stage][realised_k][current_node][best_node]
        total_cost += cost
        stage_costs.append(cost)

        current_node = best_node
        path.append(current_node)
        stage += 1

    return total_cost, path, stage_costs

# ============================================================
# POLICY 6 — WEIGHTED FORECAST
# ============================================================

def run_weighted_forecast(forecast_snapshots, realised_sequence):

    current_node = "START"
    stage        = 0
    total_cost   = 0.0
    path         = [current_node]
    stage_costs  = []
    weights      = np.ones(NUM_FORECASTS) / NUM_FORECASTS
    lr           = 0.3

    while current_node != "END" and stage < NUM_STAGES:

        forecasts  = forecast_snapshots[stage]
        next_layer = layers[stage + 1]

        best_node       = None
        best_total_cost = float("inf")

        for candidate in next_layer:
            weighted_total = sum(
                weights[k] * (
                    forecast[current_node][candidate] +
                    _cost_to_go(candidate, stage + 1, forecast)
                )
                for k, forecast in enumerate(forecasts)
            )
            if weighted_total < best_total_cost:
                best_total_cost = weighted_total
                best_node       = candidate

        realised_k = realised_sequence[stage]
        cost       = forecast_snapshots[stage][realised_k][current_node][best_node]
        total_cost += cost
        stage_costs.append(cost)

        weights[realised_k] += lr
        weights              = weights / weights.sum()

        current_node = best_node
        path.append(current_node)
        stage += 1

    return total_cost, path, stage_costs

# ============================================================
# POLICY 7 — RANDOM
# ============================================================

def run_random_policy(forecast_snapshots, realised_sequence, seed):

    rng          = np.random.RandomState(seed + 99999)
    current_node = "START"
    stage        = 0
    total_cost   = 0.0
    path         = [current_node]
    stage_costs  = []

    while current_node != "END" and stage < NUM_STAGES:

        next_node  = rng.choice(layers[stage + 1])
        realised_k = realised_sequence[stage]
        cost       = forecast_snapshots[stage][realised_k][current_node][next_node]
        total_cost += cost
        stage_costs.append(cost)

        current_node = next_node
        path.append(current_node)
        stage += 1

    return total_cost, path, stage_costs

# ============================================================
# RUN ALL POLICIES FOR ONE SEED
# ============================================================

def run_all_policies(seed):
    """
    For a given seed:
    - Generates a realised sequence (which model was realised each stage)
    - Uses the pre-loaded GRIB forecast snapshots (same for all seeds)
    - Runs all policies under the same realised sequence
    - Returns all costs
    """

    realised_sequence = make_realised_sequence(seed)

    # Use pre-loaded GRIB data — same forecasts for all seeds
    forecast_snapshots = ALL_STAGE_FORECASTS

    # Adaptive
    a_cost,  a_path,  a_sc  = run_adaptive(seed, forecast_snapshots, realised_sequence)

    # Oracles
    oracle_costs, oracle_paths = run_oracles(forecast_snapshots, realised_sequence)

    # Hindsight
    h_path, h_cost, h_sc = compute_hindsight_optimal(forecast_snapshots, realised_sequence)

    # Other policies
    v_cost,  v_path,  v_sc  = run_voting(forecast_snapshots, realised_sequence)
    g_cost,  g_path,  g_sc  = run_greedy(forecast_snapshots, realised_sequence)
    aa_cost, aa_path, aa_sc = run_averaged_arc(forecast_snapshots, realised_sequence)
    mm_cost, mm_path, mm_sc = run_minimax(forecast_snapshots, realised_sequence)
    op_cost, op_path, op_sc = run_optimistic(forecast_snapshots, realised_sequence)
    wf_cost, wf_path, wf_sc = run_weighted_forecast(forecast_snapshots, realised_sequence)
    rnd_cost, rnd_path, rnd_sc = run_random_policy(forecast_snapshots, realised_sequence, seed)
    da_cost, da_path, da_sc = run_discounted_adaptive(forecast_snapshots,realised_sequence)


    return {
        "seed"              : seed,
        "realised_sequence" : realised_sequence,
        "forecast_snapshots": forecast_snapshots,
        "adaptive"          : (a_cost,   a_path,   a_sc),
        "oracle_costs"      : oracle_costs,
        "oracle_paths"      : oracle_paths,
        "hindsight"         : (h_cost,   h_path,   h_sc),
        "voting"            : (v_cost,   v_path,   v_sc),
        "greedy"            : (g_cost,   g_path,   g_sc),
        "averaged_arc"      : (aa_cost,  aa_path,  aa_sc),
        "minimax"           : (mm_cost,  mm_path,  mm_sc),
        "optimistic"        : (op_cost,  op_path,  op_sc),
        "weighted"          : (wf_cost,  wf_path,  wf_sc),
        "random"            : (rnd_cost, rnd_path, rnd_sc),
        "discounted_adaptive": (da_cost, da_path, da_sc),
    }

# ============================================================
# RUN EXPERIMENTS
# ============================================================

def run_experiments():

    results = []

    for i in range(SEEDS):

        seed = BASE_SEED + i
        r    = run_all_policies(seed)
        results.append(r)

        best_oracle = min(r["oracle_costs"])

        print(
            f"Seed {seed}: "
            f"adap={r['adaptive'][0]:.4f}  "
            f"vote={r['voting'][0]:.4f}  "
            f"grdy={r['greedy'][0]:.4f}  "
            f"aarc={r['averaged_arc'][0]:.4f}  "
            f"mmax={r['minimax'][0]:.4f}  "
            f"opti={r['optimistic'][0]:.4f}  "
            f"wght={r['weighted'][0]:.4f}  "
            f"rand={r['random'][0]:.4f}  "
            f"hint={r['hindsight'][0]:.4f}  "
            f"best_ora={best_oracle:.4f}"
            f"disc={r['discounted_adaptive'][0]:.4f}  "
        )

    return results

# ============================================================
# STATS
# ============================================================

def compute_stats(results):

    adaptive      = np.array([r["adaptive"][0]     for r in results])
    hindsight     = np.array([r["hindsight"][0]    for r in results])
    voting        = np.array([r["voting"][0]        for r in results])
    greedy        = np.array([r["greedy"][0]        for r in results])
    averaged_arc  = np.array([r["averaged_arc"][0]  for r in results])
    minimax       = np.array([r["minimax"][0]        for r in results])
    optimistic    = np.array([r["optimistic"][0]     for r in results])
    weighted      = np.array([r["weighted"][0]       for r in results])
    random_pol    = np.array([r["random"][0]         for r in results])
    discounted_adaptive = np.array([r["discounted_adaptive"][0] for r in results])

    oracle_costs_all = np.array([r["oracle_costs"] for r in results])
    best_oracle      = oracle_costs_all.min(axis=1)
    mean_oracle      = oracle_costs_all.mean(axis=1)

    hindsight_gap = adaptive - hindsight

    def policy_stats(costs, name):
        diff  = costs - adaptive
        gap   = costs - hindsight
        return {
            f"{name}_mean"             : float(np.mean(costs)),
            f"{name}_std"              : float(np.std(costs)),
            f"adaptive_wins_vs_{name}" : float(np.mean(diff > 0)),
            f"ties_vs_{name}"          : float(np.mean(diff == 0)),
            f"policy_wins_vs_{name}"   : float(np.mean(diff < 0)),
            f"{name}_hindsight_gap"    : float(np.mean(gap)),
            f"{name}_hindsight_gap_pct": float(np.mean(gap / hindsight * 100)),
        }

    stats = {
        "adaptive_mean"              : float(np.mean(adaptive)),
        "adaptive_std"               : float(np.std(adaptive)),
        "mean_regret_vs_best_oracle" : float(np.mean(best_oracle - adaptive)),
        "win_rate_vs_best_oracle"    : float(np.mean(best_oracle - adaptive > 0)),
        "mean_regret_vs_mean_oracle" : float(np.mean(mean_oracle - adaptive)),
        "win_rate_vs_mean_oracle"    : float(np.mean(mean_oracle - adaptive > 0)),
        "mean_hindsight_gap"         : float(np.mean(hindsight_gap)),
        "std_hindsight_gap"          : float(np.std(hindsight_gap)),
        "hindsight_gap_pct"          : float(np.mean(hindsight_gap / hindsight * 100)),
        "per_forecast_win_rate"      : [
            float(np.mean(oracle_costs_all[:, k] - adaptive > 0))
            for k in range(oracle_costs_all.shape[1])
        ],
    }

    for name, costs in [
        ("voting",       voting),
        ("greedy",       greedy),
        ("averaged_arc", averaged_arc),
        ("minimax",      minimax),
        ("optimistic",   optimistic),
        ("weighted",     weighted),
        ("random",       random_pol),
        ("discounted_adaptive", discounted_adaptive)
    ]:
        stats.update(policy_stats(costs, name))

    return stats, {
        "adaptive"    : adaptive,
        "hindsight"   : hindsight,
        "voting"      : voting,
        "greedy"      : greedy,
        "averaged_arc": averaged_arc,
        "minimax"     : minimax,
        "optimistic"  : optimistic,
        "weighted"    : weighted,
        "random"      : random_pol,
        "best_oracle" : best_oracle,
        "mean_oracle" : mean_oracle,
        "discounted_adaptive" : discounted_adaptive
    }

# ============================================================
# PLOTTING
# ============================================================

def plot_results(cost_arrays, stats):

    adaptive     = cost_arrays["adaptive"]
    hindsight    = cost_arrays["hindsight"]
    best_oracle  = cost_arrays["best_oracle"]
    mean_oracle  = cost_arrays["mean_oracle"]

    policies = {
        "Hindsight"  : (hindsight,                    "purple"),
        "Adaptive"   : (adaptive,                     "steelblue"),
        "Disc Adaptive": (cost_arrays["discounted_adaptive"],"magenta"),
        "Avg Arc"    : (cost_arrays["averaged_arc"],  "darkorange"),
        "Weighted"   : (cost_arrays["weighted"],      "teal"),
        "Voting"     : (cost_arrays["voting"],        "green"),
        "Minimax"    : (cost_arrays["minimax"],       "navy"),
        "Optimistic" : (cost_arrays["optimistic"],    "gold"),
        "Greedy"     : (cost_arrays["greedy"],        "red"),
        "Random"     : (cost_arrays["random"],        "lightcoral"),
    }

    fig, axs = plt.subplots(2, 3, figsize=(22, 13))
    fig.suptitle(
        f"Policy Comparison — Real GRIB Data\n"
        f"Dates: {STAGE_DATES[0]} to {STAGE_DATES[-1]}  |  "
        f"Models: {', '.join(MODELS)}",
        fontsize=12, fontweight="bold"
    )

    # Cost distributions
    for label, (arr, color) in policies.items():
        axs[0, 0].hist(arr, bins=10, alpha=0.45, label=label, color=color)
    axs[0, 0].set_title("Cost Distributions")
    axs[0, 0].legend(fontsize=6)

    # Mean cost bar chart
    names  = list(policies.keys())
    means  = [np.mean(v[0] - hindsight) for v in policies.values()]
    stds   = [np.std(v[0] - hindsight)  for v in policies.values()]
    colors = [v[1]          for v in policies.values()]
    axs[0, 1].bar(names, means, yerr=stds, capsize=4,
                  color=colors, alpha=0.85)
    axs[0, 1].set_title("Mean Delta Cost from Hindsight Optimal ± Std (lower = better)")
    axs[0, 1].tick_params(axis="x", labelsize=7, rotation=20)

    # Win/tie/loss vs adaptive
    comp_labels = [l for l in policies if l not in ("Adaptive", "Hindsight")]
    comp_labels.append("Best Oracle")
    comp_arrays = {l: policies[l][0] for l in policies
                   if l not in ("Adaptive", "Hindsight")}
    comp_arrays["Best Oracle"] = best_oracle

    win_rates  = [np.mean(arr - adaptive > 0) * 100 for arr in comp_arrays.values()]
    tie_rates  = [np.mean(arr - adaptive == 0) * 100 for arr in comp_arrays.values()]
    loss_rates = [np.mean(arr - adaptive < 0) * 100 for arr in comp_arrays.values()]

    x = np.arange(len(comp_labels))
    w = 0.25
    axs[0, 2].bar(x - w, win_rates,  w, label="Adaptive wins", color="steelblue", alpha=0.85)
    axs[0, 2].bar(x,     tie_rates,  w, label="Tied",          color="gray",      alpha=0.85)
    axs[0, 2].bar(x + w, loss_rates, w, label="Policy wins",   color="salmon",    alpha=0.85)
    axs[0, 2].axhline(50, linestyle="--", color="black", alpha=0.4)
    axs[0, 2].set_title("Adaptive Win / Tie / Loss (%)")
    axs[0, 2].set_xticks(x)
    axs[0, 2].set_xticklabels(comp_labels, fontsize=7, rotation=20)
    axs[0, 2].legend(fontsize=7)
    axs[0, 2].set_ylim(0, 100)

    # Sorted cost curves
    idx = np.argsort(adaptive)
    for label, (arr, color) in policies.items():
        ls = "--" if label == "Hindsight" else "-"
        axs[1, 0].plot(arr[idx], label=label, color=color,
                       linewidth=1.5, linestyle=ls)
    axs[1, 0].set_title("Sorted Cost Comparison")
    axs[1, 0].legend(fontsize=6)

    # Hindsight gap boxplots
    gap_labels = [l for l in policies if l != "Hindsight"]
    gap_data   = [policies[l][0] - hindsight for l in gap_labels]
    gap_colors = [policies[l][1] for l in gap_labels]
    bp = axs[1, 1].boxplot(gap_data, labels=gap_labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], gap_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axs[1, 1].axhline(0, linestyle="--", color="black", alpha=0.5)
    axs[1, 1].set_title("Hindsight Gap by Policy")
    axs[1, 1].tick_params(axis="x", labelsize=7, rotation=20)

    # Adaptive rank
    rt_labels = ["Adaptive", "Discounted_Adaptive", "Voting", "Greedy", "Avg Arc",
                 "Minimax", "Optimistic", "Weighted", "Random"]
    rt_arrays = np.stack([policies[l][0] for l in rt_labels], axis=1)
    ranks     = np.argsort(np.argsort(rt_arrays, axis=1), axis=1)
    adap_rank = ranks[:, 0] + 1
    n_pol     = len(rt_labels)
    rank_counts = [np.sum(adap_rank == r) for r in range(1, n_pol + 1)]
    rank_labels = [f"{r}" + ("st" if r == 1 else "nd" if r == 2
                  else "rd" if r == 3 else "th") for r in range(1, n_pol + 1)]
    bar_colors_rank = ["gold", "silver", "#cd7f32"] + ["lightcoral"] * (n_pol - 3)
    axs[1, 2].bar(rank_labels, rank_counts,
                  color=bar_colors_rank, edgecolor="black", alpha=0.9)
    axs[1, 2].set_title(f"Adaptive Rank Among {n_pol} Policies")
    axs[1, 2].set_xlabel("Rank (1 = lowest cost)")

    plt.tight_layout()
    plt.show()

# ============================================================
# VERBOSE SINGLE-SEED VISUALISATION
# ============================================================

def run_verbose(seed):

    r = run_all_policies(seed)

    realised_sequence  = r["realised_sequence"]
    forecast_snapshots = r["forecast_snapshots"]

    adaptive_cost, adaptive_path, adaptive_sc  = r["adaptive"]
    h_cost,        h_path,        h_sc         = r["hindsight"]
    v_cost,        v_path,        v_sc         = r["voting"]
    g_cost,        g_path,        g_sc         = r["greedy"]
    aa_cost,       aa_path,       aa_sc        = r["averaged_arc"]
    mm_cost,       mm_path,       mm_sc        = r["minimax"]
    op_cost,       op_path,       op_sc        = r["optimistic"]
    wf_cost,       wf_path,       wf_sc        = r["weighted"]
    rnd_cost,      rnd_path,      rnd_sc       = r["random"]
    oracle_costs                                = r["oracle_costs"]
    da_cost, da_path, da_sc = r["discounted_adaptive"]

    all_totals = {
        "Adaptive"  : adaptive_cost, "Disc Adaptive" : da_cost,
        "Voting"    : v_cost,
        "Greedy"    : g_cost,        "Avg Arc"   : aa_cost,
        "Minimax"   : mm_cost,       "Optimistic": op_cost,
        "Weighted"  : wf_cost,       "Random"    : rnd_cost,
        "Hindsight" : h_cost,
    }
    winner = min(all_totals, key=all_totals.get)

    # Print table
    col_w = 16
    print(f"\n{'='*80}")
    print(f"SEED {seed}  |  Dates: {STAGE_DATES[0]} → {STAGE_DATES[-1]}  |  Winner: {winner}")
    print(f"{'='*80}")
    headers = ["Adap", "DiscAdap", "Vote", "Grdy", "AvgArc", "Mmax", "Opti", "Wght", "Rand", "Hint"]
    print(f"{'Stage':<6} {'Date':<12} {'Model':<8}", end="")
    for h in headers:
        print(f"{h:>{col_w}}", end="")
    print()
    print("-" * 80)

    path_map = dict(zip(headers, [adaptive_path, da_path, v_path, g_path, aa_path,
                                   mm_path, op_path, wf_path, rnd_path, h_path]))
    cost_map = dict(zip(headers, [adaptive_sc, da_sc, v_sc, g_sc, aa_sc,
                                   mm_sc, op_sc, wf_sc, rnd_sc, h_sc]))

    for step in range(len(realised_sequence)):
        model = MODELS[realised_sequence[step]]
        date  = STAGE_DATES[step]
        print(f"{step+1:<6} {date:<12} {model:<8}", end="")
        for key in headers:
            node = path_map[key][step + 1]
            cost = cost_map[key][step]
            cell = f"{node}({cost:.3f})"
            print(f"{cell:>{col_w}}", end="")
        print()

    print("-" * 80)
    print(f"{'TOTAL':<6} {'':<12} {'':<8}", end="")
    for key, total in zip(headers, [adaptive_cost, da_cost, v_cost, g_cost, aa_cost,
                                     mm_cost, op_cost, wf_cost, rnd_cost, h_cost]):
        print(f"{total:>{col_w}.4f}", end="")
    print()
    print(f"\nOracle costs: {[f'{c:.4f}' for c in oracle_costs]}")
    print(f"Best oracle : {min(oracle_costs):.4f}")
    print("=" * 80)

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # Single seed verbose first to confirm everything works
    print("\n--- Single seed verbose run (seed=100) ---")
    run_verbose(seed=100)

    # Full experiment
    print(f"\n--- Running {SEEDS} seeds ---")
    results = run_experiments()

    save_results(results)

    with open("forecast_policy_routes.pkl", "wb") as f:
        pickle.dump(results, f)

    print("Saved forecast routes.")

    stats, cost_arrays = compute_stats(results)

    print("\n================ ADAPTIVE POLICY ================")
    print(f"Mean cost                   : {stats['adaptive_mean']:.4f}")
    print(f"Std  cost                   : {stats['adaptive_std']:.4f}")

    print("\n================ ORACLE BENCHMARKS ================")
    print(f"Mean regret vs best oracle  : {stats['mean_regret_vs_best_oracle']:.4f}")
    print(f"Win rate vs best oracle     : {stats['win_rate_vs_best_oracle']*100:.1f}%")
    print(f"Mean regret vs mean oracle  : {stats['mean_regret_vs_mean_oracle']:.4f}")
    print(f"Win rate vs mean oracle     : {stats['win_rate_vs_mean_oracle']*100:.1f}%")

    print("\n================ HINDSIGHT GAP ================")
    print(f"Mean gap                    : {stats['mean_hindsight_gap']:.4f}")
    print(f"Std  gap                    : {stats['std_hindsight_gap']:.4f}")
    print(f"Gap as % of hindsight cost  : {stats['hindsight_gap_pct']:.1f}%")

    print("\n================ POLICY COMPARISON ================")
    for name in ["voting", "greedy", "averaged_arc",
                 "minimax", "optimistic", "weighted", "random"]:
        print(f"\n  {name.upper()}")
        print(f"    Mean cost          : {stats[f'{name}_mean']:.4f}")
        print(f"    Adaptive wins      : {stats[f'adaptive_wins_vs_{name}']*100:.1f}%")
        print(f"    Tied               : {stats[f'ties_vs_{name}']*100:.1f}%")
        print(f"    Policy wins        : {stats[f'policy_wins_vs_{name}']*100:.1f}%")
        print(f"    Hindsight gap      : {stats[f'{name}_hindsight_gap']:.4f}")
        print(f"    Hindsight gap (%)  : {stats[f'{name}_hindsight_gap_pct']:.1f}%")

    plot_results(cost_arrays, stats)