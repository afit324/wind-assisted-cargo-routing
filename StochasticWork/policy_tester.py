import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import copy
import os
import sys

# Import the layered DAG structure, forecast generator, adaptive policy decision
# function, and the number of forecast scenarios from the synthetic policy module.
from policy import (
    layers,           # list of node lists: [["START"], [stage-1 nodes], ..., ["END"]]
    generate_forecasts,   # generates a set of NUM_FORECASTS stochastic arc-cost dicts
    choose_next_node,     # adaptive MPC decision: picks next node given current forecasts
    NUM_FORECASTS         # number of parallel forecast scenarios (e.g. 6)
)

# ============================================================
# PARAMETERS
# ============================================================

SEEDS     = 100   # number of independent Monte Carlo trials
BASE_SEED = 100   # starting seed; trial i uses seed BASE_SEED + i

# ============================================================
# OPTIMAL PATH UNDER A SINGLE FORECAST
# ============================================================

def compute_optimal_path(forecast):
    """
    Runs backward dynamic programming over the full layered DAG to find
    the minimum-cost path from START to END under a single deterministic
    forecast (one arc-cost dict).

    Returns:
        path  : list of node names from START to END
        cost  : total cost of that path
    """
    V         = {}    # value function: V[node] = min cost from node to END
    best_next = {}    # best_next[node] = the next node on the optimal path
    V["END"]  = 0.0   # base case: cost at terminal node is zero

    # Sweep backwards from the second-to-last layer to the first (START layer)
    for stage_idx in range(len(layers) - 2, -1, -1):
        current_layer = layers[stage_idx]
        next_layer    = layers[stage_idx + 1]
        for node in current_layer:
            best_cost = float("inf")
            best_node = None
            for next_node in next_layer:
                # Bellman update: immediate arc cost + optimal cost-to-go
                cost = forecast[node][next_node] + V[next_node]
                if cost < best_cost:
                    best_cost = cost
                    best_node = next_node
            V[node]         = best_cost
            best_next[node] = best_node

    # Reconstruct the optimal path by following best_next from START
    path = ["START"]
    node = "START"
    while node != "END":
        node = best_next[node]
        path.append(node)

    return path, V["START"]


# ============================================================
# SHARED COST-TO-GO HELPER
# ============================================================

def _cost_to_go(current_node, current_stage, forecast):
    """
    Recursively computes the minimum cost from current_node at current_stage
    to END under a single deterministic forecast.

    Used by minimax, optimistic, and weighted policies so they all share
    the same lookahead mechanism rather than each reimplementing it.

    Args:
        current_node  : node we are currently at
        current_stage : index into `layers` for current_node
        forecast      : a single arc-cost dict

    Returns:
        float: minimum cost-to-go from current_node to END
    """
    # Base case: already at the final layer, no cost remaining
    if current_stage == len(layers) - 1:
        return 0.0

    # Recursive case: min over all next nodes of (arc cost + cost-to-go)
    return min(
        forecast[current_node][next_node] +
        _cost_to_go(next_node, current_stage + 1, forecast)
        for next_node in layers[current_stage + 1]
    )


# ============================================================
# HINDSIGHT OPTIMAL ORACLE — TRUE LOWER BOUND
# ============================================================

def compute_hindsight_optimal(forecast_snapshots, realised_sequence):
    """
    Computes the optimal path using full knowledge of which forecast
    scenario was realised at every stage — an oracle that is impossible
    to implement in practice but provides a theoretical lower bound
    (the hindsight optimum, Ĉ*).

    Args:
        forecast_snapshots : list of length NUM_STAGES; each element is a
                             list of NUM_FORECASTS arc-cost dicts (one per
                             scenario) representing the forecasts available
                             at that stage
        realised_sequence  : list of length NUM_STAGES; realised_sequence[t]
                             is the index (0..NUM_FORECASTS-1) of the forecast
                             scenario that was actually realised at stage t

    Returns:
        path       : optimal node sequence from START to END
        total_cost : minimum achievable cost with perfect hindsight
    """
    # Build the sequence of realised arc-cost matrices, one per stage
    realised_costs = [
        forecast_snapshots[t][realised_sequence[t]]
        for t in range(len(realised_sequence))
    ]

    V         = {}
    best_next = {}
    V["END"]  = 0.0

    # Backward DP using the realised cost at each stage
    for stage_idx in range(len(layers) - 2, -1, -1):
        current_layer = layers[stage_idx]
        next_layer    = layers[stage_idx + 1]
        cost_matrix   = realised_costs[stage_idx]
        for node in current_layer:
            best_cost = float("inf")
            best_node = None
            for next_node in next_layer:
                cost = cost_matrix[node][next_node] + V[next_node]
                if cost < best_cost:
                    best_cost = cost
                    best_node = next_node
            V[node]         = best_cost
            best_next[node] = best_node

    # Reconstruct path
    path = ["START"]
    node = "START"
    while node != "END":
        node = best_next[node]
        path.append(node)

    return path, V["START"]


def compute_hindsight_optimal_with_costs(forecast_snapshots, realised_sequence):
    """
    Same as compute_hindsight_optimal but also returns the per-stage arc
    costs along the hindsight-optimal path (used for visualisation).

    Returns:
        path        : optimal node sequence
        total_cost  : total cost along the path
        stage_costs : list of arc costs at each stage
    """
    path, total_cost = compute_hindsight_optimal(
        forecast_snapshots, realised_sequence
    )
    stage_costs = []
    for t in range(len(realised_sequence)):
        k         = realised_sequence[t]          # which scenario was realised
        from_node = path[t]
        to_node   = path[t + 1]
        cost      = forecast_snapshots[t][k][from_node][to_node]
        stage_costs.append(cost)

    return path, total_cost, stage_costs


# ============================================================
# REALISE AN ORACLE PATH
# ============================================================

def realise_oracle_path(path, forecast_snapshots, realised_sequence):
    """
    Evaluates a fixed pre-committed path (e.g. the optimal path under one
    forecast scenario chosen at the start) by replacing each arc's planned
    cost with the cost that was actually realised.

    This simulates a non-adaptive policy that commits to a route upfront
    and pays the realised weather cost along it.

    Args:
        path               : list of node names (committed route)
        forecast_snapshots : all forecast arc-cost dicts per stage
        realised_sequence  : which scenario was realised at each stage

    Returns:
        total_cost : realised cost of the committed path
    """
    total_cost = 0.0
    for step in range(len(path) - 1):
        from_node  = path[step]
        to_node    = path[step + 1]
        realised_k = realised_sequence[step]
        total_cost += forecast_snapshots[step][realised_k][from_node][to_node]
    return total_cost


def realise_oracle_path_with_costs(path, forecast_snapshots, realised_sequence):
    """
    Same as realise_oracle_path but also returns per-stage arc costs
    (used for visualisation and breakdown analysis).
    """
    total_cost  = 0.0
    stage_costs = []
    for step in range(len(path) - 1):
        from_node  = path[step]
        to_node    = path[step + 1]
        realised_k = realised_sequence[step]
        cost       = forecast_snapshots[step][realised_k][from_node][to_node]
        total_cost += cost
        stage_costs.append(cost)
    return total_cost, stage_costs


# ============================================================
# ADAPTIVE POLICY RUN
# ============================================================

def run_adaptive(seed):
    """
    Simulates the rolling-horizon adaptive MPC policy for one Monte Carlo
    trial (one seed).

    At each stage the policy:
      1. Calls choose_next_node() to select the next node using the current
         set of forecasts (MPC decision).
      2. Draws a realised scenario index uniformly at random from
         [0, NUM_FORECASTS-1].
      3. Pays the arc cost under that realised scenario.
      4. Updates forecasts by conditioning on the realised scenario
         (Bayesian forecast update).

    Suppresses stdout to avoid verbose output from choose_next_node.

    Args:
        seed : integer random seed for reproducibility

    Returns:
        total_cost         : total realised cost of the adaptive route
        realised_sequence  : list of which scenario was realised each stage
        initial_forecasts  : the forecast set at stage 0 (before any update)
        forecast_snapshots : list of forecast sets seen at each stage
    """
    # Suppress any print statements from choose_next_node
    stdout_backup = sys.stdout
    sys.stdout    = open(os.devnull, "w")

    try:
        random.seed(seed)
        np.random.seed(seed)

        initial_forecasts  = generate_forecasts()           # stage-0 forecast set
        forecasts          = copy.deepcopy(initial_forecasts)
        forecast_snapshots = [copy.deepcopy(forecasts)]     # record stage-0 forecasts

        current_node      = "START"
        stage             = 0
        path              = [current_node]
        total_cost        = 0.0
        realised_sequence = []

        while current_node != "END":

            # MPC step: pick next node based on current forecast set
            next_node, chosen_forecast, _ = choose_next_node(
                current_node, stage, forecasts
            )

            # Draw the realised scenario uniformly (equiprobable scenarios)
            realised_k = random.randint(0, NUM_FORECASTS - 1)
            realised_sequence.append(realised_k)
            # Pay the arc cost under the realised scenario
            total_cost += forecasts[realised_k][current_node][next_node]

            current_node = next_node
            path.append(current_node)
            stage += 1

            # Stop if we have just entered the final layer (END)
            if stage == len(layers) - 1:
                break

            # Forecast update: condition new forecasts on the realised scenario
            # (parent_forecasts seeds the correlated noise model at the next stage)
            parent    = forecasts[realised_k]
            forecasts = generate_forecasts(
                parent_forecasts=[parent] * NUM_FORECASTS
            )
            forecast_snapshots.append(copy.deepcopy(forecasts))

        return total_cost, realised_sequence, initial_forecasts, forecast_snapshots

    finally:
        sys.stdout.close()
        sys.stdout = stdout_backup


def run_adaptive_full(seed):
    """
    Same as run_adaptive but additionally returns the chosen path and
    per-stage costs (needed for the verbose visualisation).

    Returns:
        total_cost, realised_sequence, initial_forecasts,
        forecast_snapshots, path, stage_costs
    """
    stdout_backup = sys.stdout
    sys.stdout    = open(os.devnull, "w")

    try:
        random.seed(seed)
        np.random.seed(seed)

        initial_forecasts  = generate_forecasts()
        forecasts          = copy.deepcopy(initial_forecasts)
        forecast_snapshots = [copy.deepcopy(forecasts)]

        current_node      = "START"
        stage             = 0
        path              = [current_node]
        total_cost        = 0.0
        realised_sequence = []
        stage_costs       = []

        while current_node != "END":

            next_node, chosen_forecast, _ = choose_next_node(
                current_node, stage, forecasts
            )

            realised_k = random.randint(0, NUM_FORECASTS - 1)
            realised_sequence.append(realised_k)

            cost        = forecasts[realised_k][current_node][next_node]
            total_cost += cost
            stage_costs.append(cost)   # store per-stage cost for visualisation

            current_node = next_node
            path.append(current_node)
            stage += 1

            if stage == len(layers) - 1:
                break

            parent    = forecasts[realised_k]
            forecasts = generate_forecasts(
                parent_forecasts=[parent] * NUM_FORECASTS
            )
            forecast_snapshots.append(copy.deepcopy(forecasts))

        return (
            total_cost, realised_sequence, initial_forecasts,
            forecast_snapshots, path, stage_costs
        )

    finally:
        sys.stdout.close()
        sys.stdout = stdout_backup


# ============================================================
# POLICY 1 — VOTING
# ============================================================

def run_voting(forecast_snapshots, realised_sequence):
    """
    Voting policy: at each stage, each forecast scenario computes its
    own shortest path and votes for its preferred next node. The node
    with the plurality of votes is chosen. Ties are broken by lowest
    average immediate arc cost across all forecasts.

    This is a consensus-based policy — it picks the node most scenarios
    agree on, rather than optimising a single aggregate cost.

    Args:
        forecast_snapshots : forecast sets seen at each stage (from adaptive run)
        realised_sequence  : which scenario was realised at each stage

    Returns:
        total_cost : realised cost of the voting policy's route
    """
    current_node = "START"
    stage        = 0
    total_cost   = 0.0

    while current_node != "END":

        forecasts  = forecast_snapshots[stage]
        next_layer = layers[stage + 1]

        # Count votes: each forecast votes for the first step of its optimal path
        vote_count = {node: 0 for node in next_layer}
        for forecast in forecasts:
            path_f, _ = compute_optimal_path(forecast)
            vote_count[path_f[stage + 1]] += 1

        # Move to plurality winner; break ties by lowest mean immediate cost
        max_votes  = max(vote_count.values())
        candidates = [n for n, v in vote_count.items() if v == max_votes]
        next_node  = (
            candidates[0] if len(candidates) == 1
            else min(candidates,
                     key=lambda n: np.mean([f[current_node][n] for f in forecasts]))
        )

        # Pay the realised arc cost
        realised_k  = realised_sequence[stage]
        total_cost += forecast_snapshots[stage][realised_k][current_node][next_node]

        current_node = next_node
        stage       += 1
        if stage == len(layers) - 1:
            break

    return total_cost


def run_voting_with_costs(forecast_snapshots, realised_sequence):
    """Same as run_voting but also returns the path taken and per-stage costs."""

    current_node = "START"
    stage        = 0
    total_cost   = 0.0
    path         = [current_node]
    stage_costs  = []

    while current_node != "END":

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
        if stage == len(layers) - 1:
            break

    return path, total_cost, stage_costs


# ============================================================
# POLICY 2 — SIMPLE GREEDY
# ============================================================

def run_greedy(forecast_snapshots, realised_sequence):
    """
    Greedy policy: at each stage move to the node with the lowest
    average immediate arc cost across all forecast scenarios.

    No lookahead — it ignores future stages entirely. This is the
    simplest non-trivial heuristic and serves as a diagnostic baseline
    to expose myopic decision-making failures (e.g. honeypot traps).

    Args:
        forecast_snapshots : forecast sets seen at each stage
        realised_sequence  : which scenario was realised at each stage

    Returns:
        total_cost : realised cost of the greedy route
    """
    current_node = "START"
    stage        = 0
    total_cost   = 0.0

    while current_node != "END":

        forecasts  = forecast_snapshots[stage]
        next_layer = layers[stage + 1]

        # Choose the node with the lowest mean immediate cost — no lookahead
        next_node = min(
            next_layer,
            key=lambda n: np.mean([f[current_node][n] for f in forecasts])
        )

        realised_k  = realised_sequence[stage]
        total_cost += forecast_snapshots[stage][realised_k][current_node][next_node]

        current_node = next_node
        stage       += 1
        if stage == len(layers) - 1:
            break

    return total_cost


def run_greedy_with_costs(forecast_snapshots, realised_sequence):
    """Same as run_greedy but also returns the path taken and per-stage costs."""

    current_node = "START"
    stage        = 0
    total_cost   = 0.0
    path         = [current_node]
    stage_costs  = []

    while current_node != "END":

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
        if stage == len(layers) - 1:
            break

    return path, total_cost, stage_costs


# ============================================================
# POLICY 3 — AVERAGED-ARC SHORTEST PATH
# ============================================================

def run_averaged_arc(forecast_snapshots, realised_sequence):
    """
    Averaged-arc policy: at each stage, average every arc cost across all
    forecast scenarios to produce a single deterministic graph, then run
    shortest-path (DP) on it and take only the first step of that path.
    Repeat with the updated forecast set at the next stage.

    This is a receding-horizon deterministic approximation — it collapses
    forecast uncertainty into a point estimate and re-plans each stage.
    It has lookahead (full shortest-path over all remaining stages) but
    no explicit treatment of uncertainty.

    Args:
        forecast_snapshots : forecast sets seen at each stage
        realised_sequence  : which scenario was realised at each stage

    Returns:
        total_cost : realised cost of the averaged-arc route
    """
    current_node = "START"
    stage        = 0
    total_cost   = 0.0

    while current_node != "END":

        forecasts = forecast_snapshots[stage]

        # Collapse the NUM_FORECASTS arc-cost dicts into one mean arc-cost dict
        avg_forecast = {
            from_node: {
                to_node: np.mean([f[from_node][to_node] for f in forecasts])
                for to_node in forecasts[0][from_node]
            }
            for from_node in forecasts[0]
        }

        # Run backward DP on the averaged graph over all remaining stages
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

        # Take only the first step of the optimal path on the averaged graph
        next_node = best_next[current_node]

        realised_k  = realised_sequence[stage]
        total_cost += forecast_snapshots[stage][realised_k][current_node][next_node]

        current_node = next_node
        stage       += 1
        if stage == len(layers) - 1:
            break

    return total_cost


def run_averaged_arc_with_costs(forecast_snapshots, realised_sequence):
    """Same as run_averaged_arc but also returns the path taken and per-stage costs."""

    current_node = "START"
    stage        = 0
    total_cost   = 0.0
    path         = [current_node]
    stage_costs  = []

    while current_node != "END":

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
        if stage == len(layers) - 1:
            break

    return path, total_cost, stage_costs


# ============================================================
# POLICY 4 — WORST-CASE (MINIMAX)
# ============================================================

def run_minimax(forecast_snapshots, realised_sequence):
    """
    Minimax policy: at each stage pick the next node that minimises the
    maximum total cost (immediate arc + cost-to-go) across all forecast
    scenarios.

    This is the most risk-averse policy — it hedges against the worst
    possible realisation rather than optimising the expected cost.
    Equivalent to a robust optimisation approach.

    Args:
        forecast_snapshots : forecast sets seen at each stage
        realised_sequence  : which scenario was realised at each stage

    Returns:
        total_cost : realised cost of the minimax route
    """
    current_node = "START"
    stage        = 0
    total_cost   = 0.0

    while current_node != "END":

        forecasts  = forecast_snapshots[stage]
        next_layer = layers[stage + 1]

        best_node     = None
        best_max_cost = float("inf")

        for candidate in next_layer:
            # Evaluate the worst-case total cost over all scenarios for this candidate
            worst_case = max(
                forecast[current_node][candidate] +
                _cost_to_go(candidate, stage + 1, forecast)
                for forecast in forecasts
            )
            # Choose the candidate that minimises the worst case
            if worst_case < best_max_cost:
                best_max_cost = worst_case
                best_node     = candidate

        realised_k  = realised_sequence[stage]
        total_cost += forecast_snapshots[stage][realised_k][current_node][best_node]

        current_node = best_node
        stage       += 1
        if stage == len(layers) - 1:
            break

    return total_cost


def run_minimax_with_costs(forecast_snapshots, realised_sequence):
    """Same as run_minimax but also returns the path taken and per-stage costs."""

    current_node = "START"
    stage        = 0
    total_cost   = 0.0
    path         = [current_node]
    stage_costs  = []

    while current_node != "END":

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
        if stage == len(layers) - 1:
            break

    return path, total_cost, stage_costs


# ============================================================
# POLICY 5 — BEST-CASE (OPTIMISTIC)
# ============================================================

def run_optimistic(forecast_snapshots, realised_sequence):
    """
    Optimistic policy: at each stage pick the next node that minimises the
    minimum total cost (immediate arc + cost-to-go) across all forecast
    scenarios.

    This is the most risk-seeking policy — it always bets on the most
    favourable scenario playing out. It tends to over-commit to cheap
    corridors that only look attractive under the best-case forecast.

    Args:
        forecast_snapshots : forecast sets seen at each stage
        realised_sequence  : which scenario was realised at each stage

    Returns:
        total_cost : realised cost of the optimistic route
    """
    current_node = "START"
    stage        = 0
    total_cost   = 0.0

    while current_node != "END":

        forecasts  = forecast_snapshots[stage]
        next_layer = layers[stage + 1]

        best_node     = None
        best_min_cost = float("inf")

        for candidate in next_layer:
            # Evaluate the best-case total cost over all scenarios for this candidate
            best_case = min(
                forecast[current_node][candidate] +
                _cost_to_go(candidate, stage + 1, forecast)
                for forecast in forecasts
            )
            # Choose the candidate that minimises even the best case
            if best_case < best_min_cost:
                best_min_cost = best_case
                best_node     = candidate

        realised_k  = realised_sequence[stage]
        total_cost += forecast_snapshots[stage][realised_k][current_node][best_node]

        current_node = best_node
        stage       += 1
        if stage == len(layers) - 1:
            break

    return total_cost


def run_optimistic_with_costs(forecast_snapshots, realised_sequence):
    """Same as run_optimistic but also returns the path taken and per-stage costs."""

    current_node = "START"
    stage        = 0
    total_cost   = 0.0
    path         = [current_node]
    stage_costs  = []

    while current_node != "END":

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
        if stage == len(layers) - 1:
            break

    return path, total_cost, stage_costs


# ============================================================
# POLICY 6 — WEIGHTED FORECAST
# ============================================================

def run_weighted_forecast(forecast_snapshots, realised_sequence):
    """
    Weighted forecast policy: tracks which forecast scenario has been most
    accurate historically and weights scenarios by past accuracy when
    choosing the next node.

    Starts with uniform weights across all NUM_FORECASTS scenarios. After
    each stage, the weight of the scenario that was realised is increased
    by lr and the weights are renormalised. This implements a simple online
    learning / multiplicative-weight update rule.

    The decision at each stage picks the node minimising the
    weighted-average total cost (immediate + cost-to-go) across scenarios.

    Args:
        forecast_snapshots : forecast sets seen at each stage
        realised_sequence  : which scenario was realised at each stage

    Returns:
        total_cost : realised cost of the weighted-forecast route
    """
    current_node = "START"
    stage        = 0
    total_cost   = 0.0
    weights      = np.ones(NUM_FORECASTS) / NUM_FORECASTS   # start: uniform weights
    lr           = 0.3   # learning rate: how strongly to boost the realised scenario

    while current_node != "END":

        forecasts  = forecast_snapshots[stage]
        next_layer = layers[stage + 1]

        best_node       = None
        best_total_cost = float("inf")

        for candidate in next_layer:
            # Weighted sum of (arc cost + cost-to-go) across all scenarios
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

        realised_k  = realised_sequence[stage]
        total_cost += forecast_snapshots[stage][realised_k][current_node][best_node]

        # Online weight update: boost the realised scenario and renormalise
        weights[realised_k] += lr
        weights              = weights / weights.sum()

        current_node = best_node
        stage       += 1
        if stage == len(layers) - 1:
            break

    return total_cost


def run_weighted_forecast_with_costs(forecast_snapshots, realised_sequence):
    """Same as run_weighted_forecast but also returns the path and per-stage costs."""

    current_node = "START"
    stage        = 0
    total_cost   = 0.0
    path         = [current_node]
    stage_costs  = []
    weights      = np.ones(NUM_FORECASTS) / NUM_FORECASTS
    lr           = 0.3

    while current_node != "END":

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
        if stage == len(layers) - 1:
            break

    return path, total_cost, stage_costs


# ============================================================
# POLICY 7 — RANDOM
# ============================================================

def run_random_policy(forecast_snapshots, realised_sequence, seed):
    """
    Random policy: at each stage pick the next node uniformly at random,
    ignoring all forecast information.

    This is the floor baseline — any policy that adds less value than
    random selection is actively harmful. Uses a fixed seed offset
    (seed + 99999) to keep randomness independent of the adaptive
    policy's random state while still being reproducible.

    Args:
        forecast_snapshots : forecast sets (not used for decisions, only for cost lookup)
        realised_sequence  : which scenario was realised at each stage
        seed               : base random seed for this trial

    Returns:
        total_cost : realised cost of the random route
    """
    rng          = np.random.RandomState(seed + 99999)   # isolated RNG stream
    current_node = "START"
    stage        = 0
    total_cost   = 0.0

    while current_node != "END":

        # Ignore forecasts — pick uniformly from available next nodes
        next_node  = rng.choice(layers[stage + 1])
        realised_k = realised_sequence[stage]
        total_cost += forecast_snapshots[stage][realised_k][current_node][next_node]

        current_node = next_node
        stage       += 1
        if stage == len(layers) - 1:
            break

    return total_cost


def run_random_policy_with_costs(forecast_snapshots, realised_sequence, seed):
    """Same as run_random_policy but also returns the path and per-stage costs."""

    rng          = np.random.RandomState(seed + 99999)
    current_node = "START"
    stage        = 0
    total_cost   = 0.0
    path         = [current_node]
    stage_costs  = []

    while current_node != "END":

        next_node  = rng.choice(layers[stage + 1])
        realised_k = realised_sequence[stage]
        cost       = forecast_snapshots[stage][realised_k][current_node][next_node]
        total_cost += cost
        stage_costs.append(cost)

        current_node = next_node
        path.append(current_node)
        stage += 1
        if stage == len(layers) - 1:
            break

    return path, total_cost, stage_costs


# ============================================================
# ORACLE RUNS
# ============================================================

def run_oracles(seed):
    """
    Runs a single Monte Carlo trial (seed) for all policies and oracle benchmarks.

    First runs the adaptive policy to generate the shared forecast_snapshots
    and realised_sequence, then evaluates all other policies on the same
    stochastic realisation so comparisons are fair (same weather, same forecasts).

    Also evaluates the per-forecast oracle: for each of the NUM_FORECASTS
    scenarios, computes the optimal path assuming that scenario is always
    correct, then measures the cost of following that path under the actual
    realised weather.

    Returns a tuple of all policy costs and supporting data.
    """
    # Step 1: run adaptive to generate the shared stochastic realisation
    (
        adaptive_cost,
        realised_sequence,
        initial_forecasts,
        forecast_snapshots
    ) = run_adaptive(seed)

    # Step 2: per-forecast oracle — for each scenario, commit to its optimal
    # path and pay the realised cost (non-adaptive benchmark)
    oracle_costs = []
    oracle_paths = []
    for k, forecast in enumerate(initial_forecasts):
        path, _       = compute_optimal_path(forecast)
        realised_cost = realise_oracle_path(
            path, forecast_snapshots, realised_sequence
        )
        oracle_costs.append(realised_cost)
        oracle_paths.append(path)

    # Step 3: hindsight optimal (true lower bound, Ĉ*)
    hindsight_path, hindsight_cost = compute_hindsight_optimal(
        forecast_snapshots, realised_sequence
    )

    # Step 4: evaluate all other policies on the same realisation
    voting_cost       = run_voting(forecast_snapshots, realised_sequence)
    greedy_cost       = run_greedy(forecast_snapshots, realised_sequence)
    averaged_arc_cost = run_averaged_arc(forecast_snapshots, realised_sequence)
    minimax_cost      = run_minimax(forecast_snapshots, realised_sequence)
    optimistic_cost   = run_optimistic(forecast_snapshots, realised_sequence)
    weighted_cost     = run_weighted_forecast(forecast_snapshots, realised_sequence)
    random_cost       = run_random_policy(forecast_snapshots, realised_sequence, seed)

    return (
        adaptive_cost,
        oracle_costs,
        oracle_paths,
        hindsight_cost,
        hindsight_path,
        realised_sequence,
        forecast_snapshots,
        voting_cost,
        greedy_cost,
        averaged_arc_cost,
        minimax_cost,
        optimistic_cost,
        weighted_cost,
        random_cost
    )


# ============================================================
# RUN EXPERIMENTS
# ============================================================

def run_experiments():
    """
    Runs SEEDS independent Monte Carlo trials and collects results for
    all policies and benchmarks.

    Each trial uses a different random seed (BASE_SEED + i) so the
    stochastic weather and forecast realisations differ across trials.

    Returns arrays of costs (length SEEDS) for each policy, plus a dict
    all_results keyed by seed containing every policy's cost for that trial.
    """
    # Collect costs across all seeds for each policy
    adaptive_costs     = []
    oracle_costs_all   = []
    hindsight_costs    = []
    voting_costs       = []
    greedy_costs       = []
    averaged_arc_costs = []
    minimax_costs      = []
    optimistic_costs   = []
    weighted_costs     = []
    random_costs       = []

    all_results = {}   # full per-seed result dict for downstream analysis

    for i in range(SEEDS):

        seed = BASE_SEED + i

        (
            a_cost,
            oracle_costs,
            oracle_paths,
            h_cost,
            h_path,
            realised_sequence,
            forecast_snapshots,
            v_cost,
            g_cost,
            aa_cost,
            mm_cost,
            op_cost,
            wf_cost,
            rnd_cost
        ) = run_oracles(seed)

        adaptive_costs.append(a_cost)
        oracle_costs_all.append(oracle_costs)
        hindsight_costs.append(h_cost)
        voting_costs.append(v_cost)
        greedy_costs.append(g_cost)
        averaged_arc_costs.append(aa_cost)
        minimax_costs.append(mm_cost)
        optimistic_costs.append(op_cost)
        weighted_costs.append(wf_cost)
        random_costs.append(rnd_cost)

        best_oracle = min(oracle_costs)   # best single-forecast oracle for this seed

        all_results[seed] = {
            "adaptive"     : a_cost,
            "oracle_costs" : oracle_costs,
            "best_oracle"  : best_oracle,
            "hindsight"    : h_cost,
            "voting"       : v_cost,
            "greedy"       : g_cost,
            "averaged_arc" : aa_cost,
            "minimax"      : mm_cost,
            "optimistic"   : op_cost,
            "weighted"     : wf_cost,
            "random"       : rnd_cost,
        }

        # Progress line printed after each seed
        print(
            f"Seed {seed}: "
            f"adap={a_cost:.1f}  "
            f"vote={v_cost:.1f}  "
            f"grdy={g_cost:.1f}  "
            f"aarc={aa_cost:.1f}  "
            f"mmax={mm_cost:.1f}  "
            f"opti={op_cost:.1f}  "
            f"wght={wf_cost:.1f}  "
            f"rand={rnd_cost:.1f}  "
            f"hint={h_cost:.1f}"
        )

    return (
        np.array(adaptive_costs),
        np.array(oracle_costs_all),
        np.array(hindsight_costs),
        np.array(voting_costs),
        np.array(greedy_costs),
        np.array(averaged_arc_costs),
        np.array(minimax_costs),
        np.array(optimistic_costs),
        np.array(weighted_costs),
        np.array(random_costs),
        all_results
    )


# ============================================================
# STATS
# ============================================================

def compute_extended_stats(adaptive, oracle_costs_all, hindsight_costs,
                           voting, greedy, averaged_arc,
                           minimax, optimistic, weighted, random_pol):
    """
    Computes a comprehensive set of performance statistics across all seeds.

    Key metrics computed:
      - Mean and std of each policy's cost
      - Hindsight gap Δ^π = Ĉ^π − Ĉ* (how far each policy is from the
        hindsight optimal lower bound), both absolute and as % of Ĉ*
      - Win rates: fraction of seeds where the adaptive policy outperforms
        each comparison policy (and vice versa)
      - Regret vs oracle benchmarks (best single-forecast and mean oracle)
      - Per-forecast oracle win rates

    Args:
        adaptive         : array of adaptive policy costs (length SEEDS)
        oracle_costs_all : array of shape (SEEDS, NUM_FORECASTS) — per-forecast oracle costs
        hindsight_costs  : array of hindsight optimal costs (length SEEDS)
        voting, greedy, averaged_arc, minimax, optimistic, weighted, random_pol :
                           arrays of each policy's costs (length SEEDS)

    Returns:
        stats         : dict of scalar statistics
        hindsight_gap : array of Δ^adaptive = adaptive − hindsight_costs
    """
    adaptive         = np.array(adaptive)
    oracle_costs_all = np.array(oracle_costs_all)
    hindsight_costs  = np.array(hindsight_costs)

    best_oracle   = oracle_costs_all.min(axis=1)    # best single-forecast oracle per seed
    mean_oracle   = oracle_costs_all.mean(axis=1)   # mean single-forecast oracle per seed
    hindsight_gap = adaptive - hindsight_costs       # Δ^adaptive per seed

    def policy_stats(costs, name):
        """Computes comparative statistics between a policy and adaptive."""
        costs = np.array(costs)
        gap   = costs - hindsight_costs   # this policy's hindsight gap
        diff  = costs - adaptive          # positive → adaptive is cheaper

        adaptive_wins = float(np.mean(diff > 0))    # fraction of seeds adaptive wins
        ties          = float(np.mean(diff == 0))   # fraction of seeds tied
        policy_wins   = float(np.mean(diff < 0))    # fraction of seeds policy wins

        return {
            f"{name}_mean"                  : float(np.mean(costs)),
            f"{name}_std"                   : float(np.std(costs)),
            f"adaptive_wins_vs_{name}"      : adaptive_wins,
            f"ties_vs_{name}"               : ties,
            f"policy_wins_vs_{name}"        : policy_wins,
            f"{name}_gap_vs_hindsight_mean" : float(np.mean(gap)),
            f"{name}_gap_vs_hindsight_pct"  : float(np.mean(
                gap / hindsight_costs * 100
            )),
        }

    stats = {
        "adaptive_mean"                  : float(np.mean(adaptive)),
        "adaptive_std"                   : float(np.std(adaptive)),
        # Regret relative to oracle benchmarks (negative = adaptive beats oracle)
        "mean_regret_vs_best_oracle"     : float(np.mean(best_oracle - adaptive)),
        "win_rate_vs_best_oracle"        : float(np.mean(best_oracle - adaptive > 0)),
        "mean_regret_vs_mean_oracle"     : float(np.mean(mean_oracle - adaptive)),
        "win_rate_vs_mean_oracle"        : float(np.mean(mean_oracle - adaptive > 0)),
        # Hindsight gap statistics for the adaptive policy
        "mean_hindsight_gap"             : float(np.mean(hindsight_gap)),
        "std_hindsight_gap"              : float(np.std(hindsight_gap)),
        "median_hindsight_gap"           : float(np.median(hindsight_gap)),
        "max_hindsight_gap"              : float(np.max(hindsight_gap)),
        "hindsight_gap_pct_of_hindsight" : float(np.mean(
            hindsight_gap / hindsight_costs * 100
        )),
        # Win rate against each individual scenario's oracle
        "per_forecast_win_rate"          : [
            float(np.mean(oracle_costs_all[:, k] - adaptive > 0))
            for k in range(oracle_costs_all.shape[1])
        ],
    }

    # Add comparative stats for every other policy
    for name, costs in [
        ("voting",       voting),
        ("greedy",       greedy),
        ("averaged_arc", averaged_arc),
        ("minimax",      minimax),
        ("optimistic",   optimistic),
        ("weighted",     weighted),
        ("random",       random_pol),
    ]:
        stats.update(policy_stats(costs, name))

    return stats, hindsight_gap


# ============================================================
# PLOTTING
# ============================================================

def plot_enhanced_results(adaptive, oracle_costs_all, hindsight_costs,
                          voting, greedy, averaged_arc,
                          minimax, optimistic, random_pol):
    """
    Produces a bar chart of mean hindsight gap (Δ^π = Ĉ^π − Ĉ*) for each
    policy, with ±1 s.d. error bars and value labels.

    Uses a muted, print-safe colour palette that is also distinguishable
    under common forms of colour-vision deficiency.

    Saves the figure to 'Policy_Comp_ties.png' at 300 dpi.
    """
    import matplotlib as mpl
    import matplotlib.ticker as ticker

    # Reset to defaults then apply a clean, publication-ready style
    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rcParams.update({
        'font.family'       : 'DejaVu Sans',
        'font.size'         : 12,
        'axes.spines.top'   : False,
        'axes.spines.right' : False,
        'axes.spines.left'  : True,
        'axes.spines.bottom': True,
        'axes.grid'         : True,
        'axes.grid.axis'    : 'y',
        'grid.alpha'        : 0.22,
        'grid.linewidth'    : 0.7,
        'grid.color'        : '#b0b0b0',
        'axes.titlesize'    : 15,
        'axes.titleweight'  : 'bold',
        'axes.labelsize'    : 12,
        'axes.labelcolor'   : '#222222',
        'xtick.labelsize'   : 11.5,
        'ytick.labelsize'   : 10.5,
        'xtick.color'       : '#333333',
        'ytick.color'       : '#555555',
        'figure.facecolor'  : 'white',
        'axes.facecolor'    : '#fafafa',
        'figure.dpi'        : 300,
    })

    adaptive        = np.array(adaptive)
    hindsight_costs = np.array(hindsight_costs)

    # Policy name → (costs array, bar colour)
    # Muted, print-safe palette — distinct under CVD
    policies = {
        "Adaptive"    : (adaptive,     "#3a7fc1"),   # steel blue
        "Averaged-Arc": (averaged_arc, "#e6873a"),   # warm orange
        "Voting"      : (voting,       "#4aab6d"),   # mid green
        "Minimax"     : (minimax,      "#9b59b6"),   # violet
        "Optimistic"  : (optimistic,   "#c9a800"),   # golden
        "Greedy"      : (greedy,       "#c0392b"),   # crimson red
    }

    names     = list(policies.keys())
    # Compute mean and std of hindsight gap for each policy
    gap_means = [np.mean(v[0] - hindsight_costs) for v in policies.values()]
    gap_stds  = [np.std(v[0]  - hindsight_costs) for v in policies.values()]
    col_list  = [v[1] for v in policies.values()]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.patch.set_facecolor('white')

    x     = np.arange(len(names))
    bar_w = 0.52

    # Draw bars
    bars = ax.bar(x, gap_means, width=bar_w,
                  color=col_list, alpha=0.90,
                  edgecolor='white', linewidth=0.8,
                  zorder=3)

    # Overlay ±1 s.d. error bars in dark grey
    ax.errorbar(x, gap_means, yerr=gap_stds, fmt='none',
                elinewidth=1.5, ecolor='#3a3a3a',
                capsize=5, capthick=1.5, zorder=4)

    # Value labels positioned just above the top of the error bar cap
    for i, (mean, std, col) in enumerate(zip(gap_means, gap_stds, col_list)):
        ax.text(i, mean + std + max(gap_means) * 0.025,
                f"{mean:.2f}",
                ha='center', va='bottom',
                fontsize=10, fontweight='bold',
                color=col)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11.5)
    ax.set_xlim(-0.55, len(names) - 0.45)
    ax.set_ylim(bottom=0)   # gap is always non-negative (hindsight is a lower bound)

    ax.set_title("Mean Hindsight Gap by Policy",
                 pad=16, fontsize=15, fontweight='bold', color='#1a1a1a')
    ax.set_ylabel("Mean Hindsight Gap",
                  labelpad=10, fontsize=12, color='#333333')
    ax.set_xlabel("Routing Policy", labelpad=10, fontsize=12, color='#333333')

    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=False, nbins=7))
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)

    # Footer note
    fig.text(0.99, 0.015,
             "Error bars: ±1 s.d. across 100 seeds",
             ha='right', va='bottom',
             fontsize=8.5, color='#888888', style='italic')

    fig.tight_layout(rect=[0, 0.02, 1, 1])
    plt.savefig('Policy_Comp_ties.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.show()

# ============================================================
# VERBOSE VISUALISATION HELPERS
# ============================================================

def build_positions():
    """
    Assigns 2-D (x, y) coordinates to every node in the layered DAG
    for use in matplotlib network plots.

    Nodes are laid out left-to-right by stage (x = stage_idx * x_spacing)
    and centred vertically within each stage.

    Returns:
        positions : dict mapping node name → (x, y)
    """
    positions = {}
    x_spacing = 2.5   # horizontal distance between stages
    y_spacing = 1.5   # vertical distance between nodes within a stage

    for stage_idx, layer in enumerate(layers):
        x       = stage_idx * x_spacing
        # Centre the layer vertically around y = 0
        y_start = -(len(layer) - 1) * y_spacing / 2
        for i, node in enumerate(layer):
            positions[node] = (x, y_start + i * y_spacing)

    return positions


def draw_graph(ax, positions, path_taken, stage_costs, title, path_color):
    """
    Draws the full DAG on ax, highlighting the path taken by one policy.

    All edges are drawn in light grey. The taken path is overlaid in
    path_color with arc cost annotations at the midpoint of each edge.
    Nodes are coloured by role: green (START), coral (END), yellow
    (on the taken path), light blue (other).

    Args:
        ax          : matplotlib Axes to draw on
        positions   : dict from build_positions()
        path_taken  : list of node names on the policy's chosen route
        stage_costs : list of realised arc costs along path_taken
        title       : panel title (policy name + total cost)
        path_color  : colour for the highlighted path and cost labels
    """
    # Draw all DAG edges in light grey
    for stage in range(len(layers) - 1):
        for i in layers[stage]:
            for j in layers[stage + 1]:
                x1, y1 = positions[i]
                x2, y2 = positions[j]
                ax.plot([x1, x2], [y1, y2],
                        color="lightgray", alpha=0.3, zorder=1)

    # Overlay the chosen path in the policy's colour, with cost labels
    for k in range(len(path_taken) - 1):
        i = path_taken[k]
        j = path_taken[k + 1]
        x1, y1 = positions[i]
        x2, y2 = positions[j]
        ax.plot([x1, x2], [y1, y2],
                color=path_color, linewidth=3, zorder=2)
        if k < len(stage_costs):
            # Annotate the arc cost at the edge midpoint
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2
            ax.text(mx, my + 0.15, f"{stage_costs[k]:.2f}",
                    ha="center", va="bottom", fontsize=7,
                    color=path_color, fontweight="bold")

    # Draw node markers with role-based colours
    for node, (x, y) in positions.items():
        if node == "START":
            color = "lightgreen"
        elif node == "END":
            color = "lightcoral"
        elif node in path_taken:
            color = "yellow"       # on the chosen path
        else:
            color = "lightblue"    # not visited
        ax.scatter(x, y, s=600, color=color, edgecolors="black", zorder=3)
        ax.text(x, y, node, ha="center", va="center", fontsize=6, zorder=4)

    ax.set_title(title, fontsize=32, fontweight="bold")
    ax.axis("off")


def draw_cost_bars(ax, cost_dict):
    """
    Draws a grouped bar chart of per-stage realised costs for all policies.

    Each stage has one group of bars (one per policy), allowing direct
    comparison of where each policy incurs cost within the route.

    Args:
        ax        : matplotlib Axes to draw on
        cost_dict : dict mapping policy label → list of per-stage costs
    """
    color_map = {
        "Adaptive"  : "steelblue", "Voting"    : "green",
        "Greedy"    : "red",       "Avg Arc"   : "darkorange",
        "Minimax"   : "navy",      "Optimistic": "gold",
        "Weighted"  : "teal",      "Random"    : "lightcoral",
        "Hindsight" : "purple",
    }

    n_stages = max(len(v) for v in cost_dict.values())
    x        = np.arange(n_stages)
    n        = len(cost_dict)
    width    = 0.8 / n   # total bar group width = 0.8, split evenly

    for idx, (label, costs) in enumerate(cost_dict.items()):
        offset = (idx - n / 2 + 0.5) * width
        ax.bar(x + offset, costs, width, label=label,
               color=color_map.get(label, "gray"), alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"S{s+1}" for s in range(n_stages)], fontsize=8)
    ax.set_ylabel("Realised Cost", fontsize=8)
    ax.set_title("Per-Stage Costs", fontsize=9)
    ax.legend(fontsize=6)


def draw_cumulative(ax, cost_dict):
    """
    Draws cumulative realised cost curves for all policies on ax.

    Each curve shows the running total cost from stage 0 to stage t,
    making it easy to see where policies diverge. The hindsight optimal
    is shown as a dashed line.

    Args:
        ax        : matplotlib Axes to draw on
        cost_dict : dict mapping policy label → list of per-stage costs
    """
    color_map = {
        "Adaptive"  : "steelblue", "Voting"    : "green",
        "Greedy"    : "red",       "Avg Arc"   : "darkorange",
        "Minimax"   : "navy",      "Optimistic": "gold",
        "Weighted"  : "teal",      "Random"    : "lightcoral",
        "Hindsight" : "purple",
    }

    for label, costs in cost_dict.items():
        stages = list(range(len(costs) + 1))
        cum    = [0] + list(np.cumsum(costs))   # starts at 0 cost before stage 1
        ls     = "--" if label == "Hindsight" else "-"
        ax.plot(stages, cum, marker="o", markersize=4, linewidth=2,
                linestyle=ls, color=color_map.get(label, "black"),
                label=f"{label} ({sum(costs):.2f})")

    ax.set_xlabel("Stage", fontsize=8)
    ax.set_ylabel("Cumulative Cost", fontsize=8)
    ax.set_title("Cumulative Cost", fontsize=9)
    ax.legend(fontsize=6)


# ============================================================
# VERBOSE VISUALISATION
# ============================================================

def run_verbose(seed, animate=False):
    """
    Runs all policies under the same seed and produces a comprehensive
    visualisation for a single trial, showing:
      - A printed stage-by-stage table of every policy's node choice and
        arc cost, plus final totals and the winner
      - A 9-panel row of DAG graphs (one per policy) with highlighted routes
      - A grouped per-stage cost bar chart
      - A cumulative cost curve for all policies

    Useful for qualitative inspection of individual trials and debugging
    policy behaviour.

    Args:
        seed    : random seed for the trial
        animate : not currently used; reserved for future animation support
    """
    # Run all policies on the same seed
    (
        adaptive_total,
        realised_sequence,
        initial_forecasts,
        forecast_snapshots,
        adaptive_path,
        adaptive_stage_costs
    ) = run_adaptive_full(seed)

    voting_path, voting_total, voting_sc       = run_voting_with_costs(forecast_snapshots, realised_sequence)
    greedy_path, greedy_total, greedy_sc       = run_greedy_with_costs(forecast_snapshots, realised_sequence)
    aa_path,     aa_total,     aa_sc           = run_averaged_arc_with_costs(forecast_snapshots, realised_sequence)
    mm_path,     mm_total,     mm_sc           = run_minimax_with_costs(forecast_snapshots, realised_sequence)
    op_path,     op_total,     op_sc           = run_optimistic_with_costs(forecast_snapshots, realised_sequence)
    wf_path,     wf_total,     wf_sc           = run_weighted_forecast_with_costs(forecast_snapshots, realised_sequence)
    rnd_path,    rnd_total,    rnd_sc          = run_random_policy_with_costs(forecast_snapshots, realised_sequence, seed)
    hint_path,   hint_total,   hint_sc         = compute_hindsight_optimal_with_costs(forecast_snapshots, realised_sequence)

    num_stages = len(adaptive_stage_costs)

    # Identify the cheapest policy for this seed
    all_totals = {
        "Adaptive"  : adaptive_total, "Voting"    : voting_total,
        "Greedy"    : greedy_total,   "Avg Arc"   : aa_total,
        "Minimax"   : mm_total,       "Optimistic": op_total,
        "Weighted"  : wf_total,       "Random"    : rnd_total,
        "Hindsight" : hint_total,
    }
    winner = min(all_totals, key=all_totals.get)

    # ── Print stage-by-stage table ────────────────────────────
    col_w = 14
    sep   = "-" * (6 + 7 + 9 * col_w)

    print("\n" + "=" * len(sep))
    print(f"SEED {seed}   |   Winner: {winner}")
    print("=" * len(sep))
    print(f"{'Stage':<6} {'RealF':<7}", end="")
    for h in ["Adap", "Vote", "Grdy", "AvgArc", "Mmax", "Opti", "Wght", "Rand", "Hint"]:
        print(f"{h:>{col_w}}", end="")
    print()
    print(sep)

    # Maps for looking up path and costs by short policy key
    path_map = {
        "Adap"  : adaptive_path, "Vote"  : voting_path,
        "Grdy"  : greedy_path,   "AvgArc": aa_path,
        "Mmax"  : mm_path,       "Opti"  : op_path,
        "Wght"  : wf_path,       "Rand"  : rnd_path,
        "Hint"  : hint_path,
    }
    cost_map = {
        "Adap"  : adaptive_stage_costs, "Vote"  : voting_sc,
        "Grdy"  : greedy_sc,            "AvgArc": aa_sc,
        "Mmax"  : mm_sc,                "Opti"  : op_sc,
        "Wght"  : wf_sc,                "Rand"  : rnd_sc,
        "Hint"  : hint_sc,
    }

    # Print each stage row: stage number, realised scenario index, then
    # each policy's chosen node and arc cost
    for step in range(num_stages):
        print(f"{step+1:<6} {realised_sequence[step]:<7}", end="")
        for key in ["Adap", "Vote", "Grdy", "AvgArc",
                    "Mmax", "Opti", "Wght", "Rand", "Hint"]:
            node = path_map[key][step + 1]
            cost = cost_map[key][step]
            print(f"{node+' ('+f'{cost:.2f})':>{col_w}}", end="")
        print()

    print(sep)
    print(f"{'TOTAL':<6} {'':<7}", end="")
    for key, total in zip(
        ["Adap", "Vote", "Grdy", "AvgArc", "Mmax", "Opti", "Wght", "Rand", "Hint"],
        [adaptive_total, voting_total, greedy_total, aa_total,
         mm_total, op_total, wf_total, rnd_total, hint_total]
    ):
        print(f"{total:>{col_w}.2f}", end="")
    print()
    print("=" * len(sep))

    # ── Build matplotlib figure ───────────────────────────────
    positions  = build_positions()

    # One (path, costs, title, colour) tuple per policy for the graph panels
    panel_data = [
        (adaptive_path, adaptive_stage_costs, f"Adaptive\n{adaptive_total:.2f}",    "steelblue"),
        (voting_path,   voting_sc,            f"Voting\n{voting_total:.2f}",         "green"),
        (greedy_path,   greedy_sc,            f"Greedy\n{greedy_total:.2f}",         "red"),
        (aa_path,       aa_sc,                f"Avg Arc\n{aa_total:.2f}",            "darkorange"),
        (mm_path,       mm_sc,                f"Minimax\n{mm_total:.2f}",            "navy"),
        (op_path,       op_sc,                f"Optimistic\n{op_total:.2f}",         "gold"),
        (wf_path,       wf_sc,                f"Weighted\n{wf_total:.2f}",           "teal"),
        (rnd_path,      rnd_sc,               f"Random\n{rnd_total:.2f}",            "lightcoral"),
        (hint_path,     hint_sc,              f"Hindsight\n{hint_total:.2f}",        "purple"),
    ]

    fig = plt.figure(figsize=(30, 14))
    fig.suptitle(
        f"FINAL SUMMARY  |  Seed {seed}  |  Winner: {winner}",
        fontsize=14, fontweight="bold"
    )

    # Top row: one DAG graph panel per policy (9 columns)
    gs = GridSpec(3, 9, figure=fig, hspace=0.5, wspace=0.3)
    for col, (path, costs, title, color) in enumerate(panel_data):
        ax = fig.add_subplot(gs[0, col])
        draw_graph(ax, positions, path, costs, title, color)

    # Bottom two rows: per-stage bar chart (left) and cumulative cost (right)
    cost_dict_plot = {
        "Adaptive"  : adaptive_stage_costs, "Voting"    : voting_sc,
        "Greedy"    : greedy_sc,            "Avg Arc"   : aa_sc,
        "Minimax"   : mm_sc,                "Optimistic": op_sc,
        "Weighted"  : wf_sc,                "Random"    : rnd_sc,
        "Hindsight" : hint_sc,
    }

    ax_bars = fig.add_subplot(gs[1:, 0:5])
    ax_cum  = fig.add_subplot(gs[1:, 5:9])

    draw_cost_bars(ax_bars, cost_dict_plot)
    draw_cumulative(ax_cum, cost_dict_plot)

    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # Run all SEEDS Monte Carlo trials and collect results
    (
        adaptive,
        oracle_costs_all,
        hindsight_costs,
        voting,
        greedy,
        averaged_arc,
        minimax,
        optimistic,
        weighted,
        random_pol,
        all_results
    ) = run_experiments()

    # Compute summary statistics across all seeds
    ext_stats, hindsight_gap = compute_extended_stats(
        adaptive, oracle_costs_all, hindsight_costs,
        voting, greedy, averaged_arc,
        minimax, optimistic, weighted, random_pol
    )

    # ── Print results summary ─────────────────────────────────
    print("\n================ ADAPTIVE POLICY ================")
    print(f"Mean cost                   : {ext_stats['adaptive_mean']:.3f}")
    print(f"Std  cost                   : {ext_stats['adaptive_std']:.3f}")

    print("\n================ ORACLE BENCHMARKS ================")
    print(f"Mean regret vs best oracle  : {ext_stats['mean_regret_vs_best_oracle']:.3f}")
    print(f"Win rate vs best oracle     : {ext_stats['win_rate_vs_best_oracle']*100:.1f}%")
    print(f"Mean regret vs mean oracle  : {ext_stats['mean_regret_vs_mean_oracle']:.3f}")
    print(f"Win rate vs mean oracle     : {ext_stats['win_rate_vs_mean_oracle']*100:.1f}%")

    print("\n================ HINDSIGHT GAP (ADAPTIVE) ================")
    print(f"Mean gap                    : {ext_stats['mean_hindsight_gap']:.3f}")
    print(f"Std  gap                    : {ext_stats['std_hindsight_gap']:.3f}")
    print(f"Median gap                  : {ext_stats['median_hindsight_gap']:.3f}")
    print(f"Max  gap                    : {ext_stats['max_hindsight_gap']:.3f}")
    print(f"Gap as % of hindsight cost  : {ext_stats['hindsight_gap_pct_of_hindsight']:.1f}%")

    print("\n================ POLICY COMPARISON ================")
    for name in ["voting", "greedy", "averaged_arc",
                 "minimax", "optimistic", "weighted", "random"]:
        print(f"\n  {name.upper()}")
        print(f"    Mean cost               : {ext_stats[f'{name}_mean']:.3f}")
        print(f"    Std  cost               : {ext_stats[f'{name}_std']:.3f}")
        print(f"    Adaptive wins           : {ext_stats[f'adaptive_wins_vs_{name}']*100:.1f}%")
        print(f"    Tied                    : {ext_stats[f'ties_vs_{name}']*100:.1f}%")
        print(f"    Policy wins             : {ext_stats[f'policy_wins_vs_{name}']*100:.1f}%")
        print(f"    Gap vs hindsight (abs)  : {ext_stats[f'{name}_gap_vs_hindsight_mean']:.3f}")
        print(f"    Gap vs hindsight (%)    : {ext_stats[f'{name}_gap_vs_hindsight_pct']:.1f}%")

    print("\nWin rate vs each oracle forecast:")
    for k, wr in enumerate(ext_stats["per_forecast_win_rate"]):
        print(f"  Forecast {k}: {wr*100:.1f}%")

    # Generate the mean hindsight gap bar chart
    plot_enhanced_results(
        adaptive, oracle_costs_all, hindsight_costs,
        voting, greedy, averaged_arc,
        minimax, optimistic, random_pol
    )

    # Show a verbose single-seed breakdown for seed 100
    run_verbose(seed=100, animate=False)