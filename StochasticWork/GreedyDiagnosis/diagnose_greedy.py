"""
diagnose_greedy.py
==================
Traces exactly what each policy does at each stage for one seed,
showing the arc costs, penalties, and cost-to-go values so we can
understand why greedy outperforms the lookahead policies.

Run from your working directory alongside policy_grib.py and
experiments_gribs.py.
"""

import numpy as np
import policy_grib
import experiments_gribs
from policy_grib import (
    layers, NUM_STAGES, NODES_PER_STAGE, STAGE_DATES,
    MODELS, LATERAL_PENALTY, future_cost_to_go, lateral_penalty,
    COST_FUNCTIONS, ACTIVE_COST_FN
)

# ============================================================
# USE THE PRE-LOADED FORECASTS
# ============================================================

ALL_FORECASTS = experiments_gribs.ALL_STAGE_FORECASTS

# ============================================================
# HELPER — DECOMPOSE ARC COST INTO WIND + PENALTY
# ============================================================

def decompose_cost(forecast, from_node, to_node):
    """
    Splits forecast[from][to] into:
        wind_cost = arc cost without penalty
        penalty   = lateral_penalty(from, to)
        total     = wind_cost + penalty
    """
    total     = forecast[from_node][to_node]
    pen       = lateral_penalty(from_node, to_node)
    wind_cost = total - pen
    return wind_cost, pen, total


# ============================================================
# FULL STAGE TRACE FOR ONE FORECAST SET
# ============================================================

def trace_stage(stage, current_node, forecasts):
    """
    For every candidate next node, prints:
      - wind cost (per model and mean)
      - lateral penalty
      - total immediate cost
      - cost-to-go under each model
      - mean total cost
    Also marks what greedy would pick vs what adaptive would pick.
    """

    next_layer = layers[stage + 1]
    print(f"\n{'─'*80}")
    print(f"Stage {stage} ({STAGE_DATES[stage]})  |  "
          f"Current: {current_node}  |  "
          f"Penalty={LATERAL_PENALTY}  Measure={ACTIVE_COST_FN}")
    print(f"{'─'*80}")

    # Compute all options
    options = []
    for candidate in next_layer:

        # Decompose per model
        wind_costs = []
        penalties  = []
        totals_imm = []
        for f in forecasts:
            wc, pen, tot = decompose_cost(f, current_node, candidate)
            wind_costs.append(wc)
            penalties.append(pen)
            totals_imm.append(tot)

        avg_wind = np.mean(wind_costs)
        avg_pen  = penalties[0]          # penalty is same across models
        avg_imm  = np.mean(totals_imm)   # = avg_wind + avg_pen

        # Cost-to-go under each model
        ctg_per_model = [
            future_cost_to_go(candidate, stage + 1, f)
            for f in forecasts
        ]
        avg_ctg = np.mean(ctg_per_model)

        mean_total = avg_imm + avg_ctg    # what adaptive minimises

        options.append({
            "node"        : candidate,
            "avg_wind"    : avg_wind,
            "penalty"     : avg_pen,
            "avg_imm"     : avg_imm,
            "wind_per_model" : wind_costs,
            "ctg_per_model"  : ctg_per_model,
            "avg_ctg"     : avg_ctg,
            "mean_total"  : mean_total,
        })

    # What would each policy pick?
    greedy_pick   = min(options, key=lambda x: x["avg_imm"])["node"]
    adaptive_pick = min(options, key=lambda x: x["mean_total"])["node"]
    minimax_pick  = min(
        options,
        key=lambda x: max(
            x["wind_per_model"][k] + x["ctg_per_model"][k]
            for k in range(len(forecasts))
        )
    )["node"]

    # Print table header
    print(f"  {'Node':<10} {'AvgWind':>9} {'Penalty':>9} "
          f"{'AvgImm':>9} {'AvgCtg':>9} {'MeanTotal':>11} "
          f"{'Greedy?':>8} {'Adaptive?':>10}")
    print(f"  {'-'*80}")

    for opt in sorted(options, key=lambda x: x["mean_total"]):
        g_mark = "<<" if opt["node"] == greedy_pick   else ""
        a_mark = "<<" if opt["node"] == adaptive_pick else ""
        same   = " [SAME]" if greedy_pick == adaptive_pick else ""

        print(
            f"  {opt['node']:<10} "
            f"{opt['avg_wind']:>9.4f} "
            f"{opt['penalty']:>9.4f} "
            f"{opt['avg_imm']:>9.4f} "
            f"{opt['avg_ctg']:>9.4f} "
            f"{opt['mean_total']:>11.4f} "
            f"{g_mark:>8} "
            f"{a_mark:>10}"
        )

    print(f"\n  Greedy picks:   {greedy_pick}")
    print(f"  Adaptive picks: {adaptive_pick}"
          + (" [SAME AS GREEDY]" if greedy_pick == adaptive_pick else ""))
    print(f"  Minimax picks:  {minimax_pick}")

    # Per-model detail for the greedy vs adaptive disagreement
    if greedy_pick != adaptive_pick:
        print(f"\n  DISAGREEMENT — why adaptive chose {adaptive_pick} "
              f"over greedy's {greedy_pick}:")
        for opt in options:
            if opt["node"] in (greedy_pick, adaptive_pick):
                print(f"\n    {opt['node']}:")
                for k, model in enumerate(MODELS):
                    print(
                        f"      {model:<20} "
                        f"imm={opt['wind_per_model'][k]:.4f}+"
                        f"{opt['penalty']:.4f}  "
                        f"ctg={opt['ctg_per_model'][k]:.4f}  "
                        f"total={opt['wind_per_model'][k]+opt['penalty']+opt['ctg_per_model'][k]:.4f}"
                    )

    return greedy_pick, adaptive_pick


# ============================================================
# SIMULATE BOTH POLICIES STEP BY STEP
# ============================================================

def simulate_both(seed):
    """
    Runs greedy and adaptive side by side, tracking their routes
    and showing exactly where and why they diverge.
    """

    rng = np.random.RandomState(seed)
    realised_sequence = [
        rng.randint(0, len(MODELS)) for _ in range(NUM_STAGES)
    ]

    print(f"\n{'='*80}")
    print(f"SEED {seed}  |  Realised sequence: {realised_sequence}")
    print(f"  ({[MODELS[k] for k in realised_sequence]})")
    print(f"{'='*80}")

    greedy_path   = ["START"]
    adaptive_path = ["START"]
    greedy_cost   = 0.0
    adaptive_cost = 0.0
    greedy_costs  = []
    adaptive_costs = []

    for stage in range(NUM_STAGES):

        forecasts = ALL_FORECASTS[stage]
        realised_k = realised_sequence[stage]

        # Trace the stage
        g_pick, a_pick = trace_stage(
            stage, greedy_path[-1], forecasts
        )

        # Realise costs
        g_cost = forecasts[realised_k][greedy_path[-1]][g_pick]
        a_cost = forecasts[realised_k][adaptive_path[-1]][a_pick]

        greedy_cost   += g_cost
        adaptive_cost += a_cost
        greedy_costs.append(g_cost)
        adaptive_costs.append(a_cost)

        greedy_path.append(g_pick)
        adaptive_path.append(a_pick)

    # Final summary
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY — seed {seed}")
    print(f"{'='*80}")
    print(f"\n  {'Stage':<8} {'Date':<13} {'Model':<20} "
          f"{'Greedy':>10} {'Adaptive':>10} {'G Cost':>9} {'A Cost':>9}")
    print(f"  {'-'*82}")

    for t in range(NUM_STAGES):
        model = MODELS[realised_sequence[t]]
        same  = " =" if greedy_path[t+1] == adaptive_path[t+1] else ""
        print(
            f"  {t:<8} {STAGE_DATES[t]:<13} {model:<20} "
            f"{greedy_path[t+1]:>10} {adaptive_path[t+1]:>10} "
            f"{greedy_costs[t]:>9.4f} {adaptive_costs[t]:>9.4f}{same}"
        )

    print(f"  {'-'*82}")
    print(f"  {'TOTAL':<43} {greedy_cost:>9.4f} {adaptive_cost:>9.4f}")

    winner = "GREEDY" if greedy_cost < adaptive_cost else "ADAPTIVE"
    margin = abs(greedy_cost - adaptive_cost)
    print(f"\n  Winner: {winner}  (margin: {margin:.4f})")

    # Decompose into wind vs penalty
    print(f"\n  Cost decomposition:")
    for label, path, costs in [
        ("Greedy",   greedy_path,   greedy_costs),
        ("Adaptive", adaptive_path, adaptive_costs),
    ]:
        total_wind = 0.0
        total_pen  = 0.0
        for t in range(NUM_STAGES):
            fn  = path[t]
            tn  = path[t+1]
            pen = lateral_penalty(fn, tn)
            total_wind += costs[t] - pen
            total_pen  += pen
        print(
            f"    {label:<10}: wind={total_wind:.4f}  "
            f"penalty={total_pen:.4f}  "
            f"total={total_wind+total_pen:.4f}"
        )

    return greedy_cost, adaptive_cost


# ============================================================
# RUN ACROSS MULTIPLE SEEDS TO FIND PATTERN
# ============================================================

def run_diagnostic(n_seeds=10):

    print(f"\nRunning diagnostic across {n_seeds} seeds...")
    print(f"Lateral penalty: {LATERAL_PENALTY}")
    print(f"Cost measure: {ACTIVE_COST_FN}")
    print(f"Stages: {NUM_STAGES}  Nodes: {NODES_PER_STAGE}")

    greedy_wins   = 0
    adaptive_wins = 0
    ties          = 0

    for seed in range(100, 100 + n_seeds):
        gc, ac = simulate_both(seed)
        if gc < ac - 1e-6:
            greedy_wins += 1
        elif ac < gc - 1e-6:
            adaptive_wins += 1
        else:
            ties += 1

    print(f"\n{'='*50}")
    print(f"SUMMARY OVER {n_seeds} SEEDS")
    print(f"{'='*50}")
    print(f"Greedy wins:   {greedy_wins}")
    print(f"Adaptive wins: {adaptive_wins}")
    print(f"Ties:          {ties}")
    print(f"\nGreedy winning suggests one of:")
    print(f"  1. Cost-to-go is misleading the lookahead policies")
    print(f"     (forecasts systematically wrong about future wind)")
    print(f"  2. Lateral penalty is discouraging adaptive from")
    print(f"     making moves that would be better in realisation")
    print(f"  3. Penalty={LATERAL_PENALTY} is large enough that")
    print(f"     avoiding lateral movement (greedy stays near")
    print(f"     current node) is genuinely optimal under actual")
    print(f"     weather, even if not under forecasts")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # Single seed detailed trace
    simulate_both(seed=100)

    # Multi-seed summary
    run_diagnostic(n_seeds=10)