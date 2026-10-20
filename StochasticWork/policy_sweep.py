import os
import sys
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from policy import (
    layers,
    generate_forecasts,
    choose_next_node,
    NUM_FORECASTS
)

# ============================================================
# PARAMETERS
# ============================================================

SEEDS     = 50
BASE_SEED = 100

PARAM_PAIRS = [
    # (noise, persistence)
    (0.02, 0.1),
    (0.02, 0.5),
    (0.02, 0.9),
    (0.10, 0.1),
    (0.10, 0.3),
    (0.10, 0.5),
    (0.10, 0.7),
    (0.10, 0.9),
    (0.20, 0.1),
    (0.20, 0.3),
    (0.20, 0.5),
    (0.20, 0.7),
    (0.20, 0.9),
    (0.35, 0.1),
    (0.35, 0.3),
    (0.35, 0.5),
    (0.35, 0.7),
    (0.35, 0.9),
    (0.50, 0.2),
    (0.50, 0.7),
]

# Policy names in display order
POLICY_NAMES = [
    "adaptive",
    "averaged_arc",
    "weighted",
    "minimax",
    "optimistic",
    "voting",
    "greedy",
    "random",
]

# Fine grid used only for the win-rate heatmap —
# matches the density of the original two-panel heatmap.
# Noise sweep: fixed persistence=0.7, vary noise
# Persistence sweep: fixed noise=0.08, vary persistence
HEATMAP_NOISE_VALS       = [0.01, 0.03, 0.05, 0.08, 0.12, 0.15, 0.20]
HEATMAP_PERSISTENCE_VALS = [0.0,  0.20, 0.40, 0.60, 0.80, 0.90, 0.95]
HEATMAP_FIXED_PERS       = 0.70   # held constant while noise varies
HEATMAP_FIXED_NOISE      = 0.08   # held constant while persistence varies

HEATMAP_PAIRS = (
    [(n, HEATMAP_FIXED_PERS) for n in HEATMAP_NOISE_VALS] +
    [(HEATMAP_FIXED_NOISE, p) for p in HEATMAP_PERSISTENCE_VALS]
)

POLICY_COLORS = {
    "adaptive"    : "steelblue",
    "averaged_arc": "darkorange",
    "weighted"    : "teal",
    "minimax"     : "navy",
    "optimistic"  : "gold",
    "voting"      : "green",
    "greedy"      : "red",
    "random"      : "lightcoral",
}

# ============================================================
# SHARED COST-TO-GO HELPER
# ============================================================

def _cost_to_go(current_node, current_stage, forecast):
    if current_stage == len(layers) - 1:
        return 0.0
    return min(
        forecast[current_node][n] +
        _cost_to_go(n, current_stage + 1, forecast)
        for n in layers[current_stage + 1]
    )


# ============================================================
# OPTIMAL PATH UNDER A SINGLE FORECAST (DP)
# ============================================================

def compute_optimal_path(forecast):
    V         = {}
    best_next = {}
    V["END"]  = 0.0

    for stage_idx in range(len(layers) - 2, -1, -1):
        current_layer = layers[stage_idx]
        next_layer    = layers[stage_idx + 1]
        for node in current_layer:
            best_cost = float("inf")
            best_node = None
            for next_node in next_layer:
                cost = forecast[node][next_node] + V[next_node]
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
    return path


# ============================================================
# HINDSIGHT OPTIMAL
# ============================================================

def compute_hindsight_optimal(forecast_snapshots, realised_sequence):
    realised_costs = [
        forecast_snapshots[t][realised_sequence[t]]
        for t in range(len(realised_sequence))
    ]

    V         = {}
    best_next = {}
    V["END"]  = 0.0

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

    path = ["START"]
    node = "START"
    while node != "END":
        node = best_next[node]
        path.append(node)
    return path, V["START"]


# ============================================================
# ALL POLICIES
# ============================================================

def policy_adaptive(current_node, stage, forecasts):
    next_node, _, _ = choose_next_node(current_node, stage, forecasts)
    return next_node


def policy_averaged_arc(current_node, stage, forecasts):
    avg_forecast = {
        fn: {
            tn: np.mean([f[fn][tn] for f in forecasts])
            for tn in forecasts[0][fn]
        }
        for fn in forecasts[0]
    }
    V         = {}
    best_next = {}
    V["END"]  = 0.0
    for si in range(len(layers) - 2, -1, -1):
        for node in layers[si]:
            best_cost = float("inf")
            best_node = None
            for nnode in layers[si + 1]:
                cost = avg_forecast[node][nnode] + V[nnode]
                if cost < best_cost:
                    best_cost = cost
                    best_node = nnode
            V[node]         = best_cost
            best_next[node] = best_node
    return best_next[current_node]


def policy_weighted(current_node, stage, forecasts, weights):
    next_layer = layers[stage + 1]
    best_node  = None
    best_cost  = float("inf")
    for candidate in next_layer:
        wt = sum(
            weights[k] * (
                f[current_node][candidate] +
                _cost_to_go(candidate, stage + 1, f)
            )
            for k, f in enumerate(forecasts)
        )
        if wt < best_cost:
            best_cost = wt
            best_node = candidate
    return best_node


def policy_minimax(current_node, stage, forecasts):
    next_layer    = layers[stage + 1]
    best_node     = None
    best_max_cost = float("inf")
    for candidate in next_layer:
        worst = max(
            f[current_node][candidate] + _cost_to_go(candidate, stage + 1, f)
            for f in forecasts
        )
        if worst < best_max_cost:
            best_max_cost = worst
            best_node     = candidate
    return best_node


def policy_optimistic(current_node, stage, forecasts):
    next_layer    = layers[stage + 1]
    best_node     = None
    best_min_cost = float("inf")
    for candidate in next_layer:
        best = min(
            f[current_node][candidate] + _cost_to_go(candidate, stage + 1, f)
            for f in forecasts
        )
        if best < best_min_cost:
            best_min_cost = best
            best_node     = candidate
    return best_node


def policy_voting(current_node, stage, forecasts):
    next_layer = layers[stage + 1]
    vote_count = {n: 0 for n in next_layer}
    for forecast in forecasts:
        path_f = compute_optimal_path(forecast)
        vote_count[path_f[stage + 1]] += 1
    max_votes  = max(vote_count.values())
    candidates = [n for n, v in vote_count.items() if v == max_votes]
    return (
        candidates[0] if len(candidates) == 1
        else min(candidates,
                 key=lambda n: np.mean([f[current_node][n] for f in forecasts]))
    )


def policy_greedy(current_node, stage, forecasts):
    next_layer = layers[stage + 1]
    return min(
        next_layer,
        key=lambda n: np.mean([f[current_node][n] for f in forecasts])
    )


def policy_random(current_node, stage, rng):
    return rng.choice(layers[stage + 1])


# ============================================================
# SINGLE COMBINED RUN — ALL POLICIES
# ============================================================

def run_single(seed, noise_std, persistence):
    """
    Runs all policies under the same seed, noise, and persistence.
    Returns a dict of costs keyed by policy name, plus hindsight cost.
    """

    stdout_backup = sys.stdout
    sys.stdout    = open(os.devnull, "w")

    try:
        random.seed(seed)
        np.random.seed(seed)

        initial_forecasts = generate_forecasts(
            noise_std=noise_std,
            correlation=persistence
        )
        forecasts          = copy.deepcopy(initial_forecasts)
        forecast_snapshots = [copy.deepcopy(forecasts)]

        # ── Adaptive run (generates the shared realised sequence) ──
        current_node      = "START"
        stage             = 0
        adaptive_cost     = 0.0
        realised_sequence = []
        weights           = np.ones(NUM_FORECASTS) / NUM_FORECASTS
        lr                = 0.3

        while current_node != "END":

            next_node, _, _ = choose_next_node(current_node, stage, forecasts)

            realised_k = random.randint(0, NUM_FORECASTS - 1)
            realised_sequence.append(realised_k)
            adaptive_cost += forecasts[realised_k][current_node][next_node]

            current_node = next_node
            stage       += 1

            if stage == len(layers) - 1:
                break

            parent    = forecasts[realised_k]
            forecasts = generate_forecasts(
                parent_forecasts=[parent] * NUM_FORECASTS,
                noise_std=noise_std,
                correlation=persistence
            )
            forecast_snapshots.append(copy.deepcopy(forecasts))

        # ── Replay all other policies through same snapshots ──────
        def replay(pick_fn):
            node  = "START"
            cost  = 0.0
            s     = 0
            wts   = np.ones(NUM_FORECASTS) / NUM_FORECASTS

            while node != "END":
                fs        = forecast_snapshots[s]
                next_node = pick_fn(node, s, fs, wts)
                rk        = realised_sequence[s]
                cost      += forecast_snapshots[s][rk][node][next_node]

                wts[rk] += lr
                wts      = wts / wts.sum()

                node = next_node
                s   += 1
                if s == len(layers) - 1:
                    break
            return cost

        rng = np.random.RandomState(seed + 99999)

        costs = {
            "adaptive"    : adaptive_cost,
            "averaged_arc": replay(lambda n, s, fs, w: policy_averaged_arc(n, s, fs)),
            "weighted"    : replay(lambda n, s, fs, w: policy_weighted(n, s, fs, w)),
            "minimax"     : replay(lambda n, s, fs, w: policy_minimax(n, s, fs)),
            "optimistic"  : replay(lambda n, s, fs, w: policy_optimistic(n, s, fs)),
            "voting"      : replay(lambda n, s, fs, w: policy_voting(n, s, fs)),
            "greedy"      : replay(lambda n, s, fs, w: policy_greedy(n, s, fs)),
            "random"      : replay(lambda n, s, fs, w: policy_random(n, s, rng)),
        }

        # ── Hindsight optimal ─────────────────────────────────────
        _, hindsight_cost = compute_hindsight_optimal(
            forecast_snapshots, realised_sequence
        )

        # ── Oracle benchmarks ─────────────────────────────────────
        # Best oracle: optimal path under each individual forecast member
        best_oracle_costs = []
        for k in range(NUM_FORECASTS):
            path_k = compute_optimal_path(forecast_snapshots[0][k])
            c = 0.0
            node = "START"
            for s_idx, step_node in enumerate(path_k[1:]):
                rk = realised_sequence[s_idx]
                c += forecast_snapshots[s_idx][rk][node][step_node]
                node = step_node
            best_oracle_costs.append(c)

        # Mean oracle: average-forecast path
        avg_fc = {
            fn: {
                tn: np.mean([forecast_snapshots[0][k][fn][tn]
                             for k in range(NUM_FORECASTS)])
                for tn in forecast_snapshots[0][0][fn]
            }
            for fn in forecast_snapshots[0][0]
        }
        mean_oracle_path = compute_optimal_path(avg_fc)
        mean_oracle_cost = 0.0
        node = "START"
        for s_idx, step_node in enumerate(mean_oracle_path[1:]):
            rk = realised_sequence[s_idx]
            mean_oracle_cost += forecast_snapshots[s_idx][rk][node][step_node]
            node = step_node

        costs["best_oracle"] = min(best_oracle_costs)
        costs["mean_oracle"] = mean_oracle_cost

        return costs, hindsight_cost

    finally:
        sys.stdout.close()
        sys.stdout = stdout_backup


# ============================================================
# EVALUATE A SINGLE (NOISE, PERSISTENCE) PAIR
# ============================================================

def evaluate_setting(noise_std, persistence):

    all_costs     = {p: [] for p in POLICY_NAMES + ["best_oracle", "mean_oracle"]}
    hindsight_arr = []

    for i in range(SEEDS):
        seed            = BASE_SEED + i
        costs, h_cost   = run_single(seed, noise_std, persistence)
        hindsight_arr.append(h_cost)
        for p in POLICY_NAMES + ["best_oracle", "mean_oracle"]:
            all_costs[p].append(costs[p])

    hindsight = np.array(hindsight_arr)
    results   = {}

    for p in POLICY_NAMES:
        arr  = np.array(all_costs[p])
        gap  = arr - hindsight
        diff = arr - np.array(all_costs["adaptive"])

        results[p] = {
            "mean"              : float(np.mean(arr)),
            "std"               : float(np.std(arr)),
            "gap_vs_hint_mean"  : float(np.mean(gap)),
            "gap_vs_hint_pct"   : float(np.mean(gap / hindsight * 100)),
            "adaptive_wins"     : float(np.mean(diff > 0)),
            "tied"              : float(np.mean(diff == 0)),
            "policy_wins"       : float(np.mean(diff < 0)),
        }

    adap_arr = np.array(all_costs["adaptive"])

    for oracle_key in ["best_oracle", "mean_oracle"]:
        oracle_arr = np.array(all_costs[oracle_key])
        regret     = oracle_arr - adap_arr          # oracle − adaptive (positive = adaptive wins)
        results[oracle_key] = {
            "mean"          : float(np.mean(oracle_arr)),
            "mean_regret"   : float(np.mean(regret)),     # oracle cost − adaptive cost
            "win_rate_pct"  : float(np.mean(adap_arr < oracle_arr) * 100),  # adaptive beats oracle
        }

    results["hindsight_mean"] = float(np.mean(hindsight))
    return results


# ============================================================
# RUN SWEEP
# ============================================================

def run_sweep():

    total   = len(PARAM_PAIRS)
    results = {}

    for idx, (noise, pers) in enumerate(PARAM_PAIRS):
        print(
            f"[{idx+1:>2}/{total}]  noise={noise:.2f}  persistence={pers:.2f}  ...",
            end="", flush=True
        )
        stats = evaluate_setting(noise, pers)
        results[(noise, pers)] = stats

        adap_gap = stats["adaptive"]["gap_vs_hint_pct"]
        grdy_gap = stats["greedy"]["gap_vs_hint_pct"]
        aarc_gap = stats["averaged_arc"]["gap_vs_hint_pct"]
        print(
            f"  adap={adap_gap:.1f}%  "
            f"avg_arc={aarc_gap:.1f}%  "
            f"greedy={grdy_gap:.1f}%"
        )

    return results


# ============================================================
# PLOTTING
# ============================================================

def plot_gap_heatmaps(results):
    """
    One heatmap per policy showing hindsight gap (%) across all
    20 (noise, persistence) pairs. Lower is better.
    """

    n_policies = len(POLICY_NAMES)
    n_cols     = 4
    n_rows     = (n_policies + n_cols - 1) // n_cols

    fig, axs = plt.subplots(n_rows, n_cols,
                             figsize=(20, 5 * n_rows))
    fig.suptitle(
        "Hindsight Gap (%) by Policy across Parameter Pairs\n"
        "(lower = closer to perfect foresight)",
        fontsize=14, fontweight="bold"
    )

    pair_labels = [f"n={n:.2f}\np={c:.1f}" for n, c in PARAM_PAIRS]
    gap_matrix  = {
        p: [results[(n, c)][p]["gap_vs_hint_pct"] for n, c in PARAM_PAIRS]
        for p in POLICY_NAMES
    }

    vmax = max(
        max(v) for p, v in gap_matrix.items() if p != "random"
    )

    for ax, policy in zip(axs.flat, POLICY_NAMES):
        vals  = np.array(gap_matrix[policy]).reshape(1, -1)
        vm    = 75 if policy == "random" else vmax

        im = ax.imshow(vals, cmap="RdYlGn_r", aspect="auto",
                       vmin=0, vmax=vm)
        ax.set_xticks(range(len(PARAM_PAIRS)))
        ax.set_xticklabels(pair_labels, fontsize=5, rotation=45, ha="right")
        ax.set_yticks([])
        ax.set_title(policy.upper(), fontsize=10,
                     color=POLICY_COLORS.get(policy, "black"),
                     fontweight="bold")
        plt.colorbar(im, ax=ax, label="Gap %")

        for j, v in enumerate(gap_matrix[policy]):
            ax.text(j, 0, f"{abs(v):.1f}%",
                    ha="center", va="center", fontsize=6, color="black")

    for ax in axs.flat[n_policies:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("HeatMap_sweep.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: HeatMap_sweep.png")


def plot_oracle_sweep(results):
    """
    Replicates the 'More_Accurate' plots:
      Left panel:  Persistence Sweep (noise fixed)
      Right panel: Noise Sweep (persistence fixed)
    Each panel shows two sub-plots — Mean Regret and Win Rate —
    comparing adaptive against Best Oracle and Mean Oracle.
    """

    # ── Persistence sweep (fix noise=0.10, vary persistence) ──────────────
    fixed_noise   = 0.10
    pers_pairs    = sorted([(n, p) for n, p in PARAM_PAIRS if n == fixed_noise],
                           key=lambda x: x[1])
    pers_vals     = [p for _, p in pers_pairs]

    best_regret_p = [results[(fixed_noise, p)]["best_oracle"]["mean_regret"] for p in pers_vals]
    mean_regret_p = [results[(fixed_noise, p)]["mean_oracle"]["mean_regret"] for p in pers_vals]
    best_winr_p   = [results[(fixed_noise, p)]["best_oracle"]["win_rate_pct"] for p in pers_vals]
    mean_winr_p   = [results[(fixed_noise, p)]["mean_oracle"]["win_rate_pct"] for p in pers_vals]

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Persistence Sweep (noise={fixed_noise})",
                 fontsize=13, fontweight="bold")

    axs[0].plot(pers_vals, best_regret_p, marker="o", color="steelblue",
                linewidth=2, label="vs Best Oracle")
    axs[0].plot(pers_vals, mean_regret_p, marker="s", color="darkorange",
                linewidth=2, label="vs Mean Oracle")
    axs[0].axhline(0, linestyle="--", color="black", linewidth=1)
    axs[0].set_xlabel("Persistence", fontsize=11)
    axs[0].set_ylabel("Mean Regret (oracle − adaptive)", fontsize=10)
    axs[0].set_title("Mean Regret vs Persistence")
    axs[0].legend(fontsize=9)
    axs[0].grid(alpha=0.3)

    axs[1].plot(pers_vals, best_winr_p, marker="o", color="steelblue",
                linewidth=2, label="vs Best Oracle")
    axs[1].plot(pers_vals, mean_winr_p, marker="s", color="darkorange",
                linewidth=2, label="vs Mean Oracle")
    axs[1].axhline(50, linestyle="--", color="black", linewidth=1)
    axs[1].set_xlabel("Persistence", fontsize=11)
    axs[1].set_ylabel("Win Rate (%)", fontsize=10)
    axs[1].set_title("Win Rate vs Persistence")
    axs[1].set_ylim(0, 100)
    axs[1].legend(fontsize=9)
    axs[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("Persistence_Sweep.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: Persistence_Sweep.png")

    # ── Noise sweep (fix persistence=0.5, vary noise) ─────────────────────
    fixed_pers   = 0.5
    noise_pairs  = sorted([(n, p) for n, p in PARAM_PAIRS if p == fixed_pers],
                          key=lambda x: x[0])
    noise_vals   = [n for n, _ in noise_pairs]

    best_regret_n = [results[(n, fixed_pers)]["best_oracle"]["mean_regret"] for n in noise_vals]
    mean_regret_n = [results[(n, fixed_pers)]["mean_oracle"]["mean_regret"] for n in noise_vals]
    best_winr_n   = [results[(n, fixed_pers)]["best_oracle"]["win_rate_pct"] for n in noise_vals]
    mean_winr_n   = [results[(n, fixed_pers)]["mean_oracle"]["win_rate_pct"] for n in noise_vals]

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Noise Sweep (persistence={fixed_pers})",
                 fontsize=13, fontweight="bold")

    axs[0].plot(noise_vals, best_regret_n, marker="o", color="steelblue",
                linewidth=2, label="vs Best Oracle")
    axs[0].plot(noise_vals, mean_regret_n, marker="s", color="darkorange",
                linewidth=2, label="vs Mean Oracle")
    axs[0].axhline(0, linestyle="--", color="black", linewidth=1)
    axs[0].set_xlabel("Noise Std", fontsize=11)
    axs[0].set_ylabel("Mean Regret (oracle − adaptive)", fontsize=10)
    axs[0].set_title("Mean Regret vs Noise")
    axs[0].legend(fontsize=9)
    axs[0].grid(alpha=0.3)

    axs[1].plot(noise_vals, best_winr_n, marker="o", color="steelblue",
                linewidth=2, label="vs Best Oracle")
    axs[1].plot(noise_vals, mean_winr_n, marker="s", color="darkorange",
                linewidth=2, label="vs Mean Oracle")
    axs[1].axhline(50, linestyle="--", color="black", linewidth=1)
    axs[1].set_xlabel("Noise Std", fontsize=11)
    axs[1].set_ylabel("Win Rate (%)", fontsize=10)
    axs[1].set_title("Win Rate vs Noise")
    axs[1].set_ylim(0, 100)
    axs[1].legend(fontsize=9)
    axs[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("Noise_Sweep.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: Noise_Sweep.png")


def plot_policy_ranking(results):
    """
    Adaptive rank per parameter pair, and mean gap per policy.
    """

    pair_labels = [f"n={n:.2f},p={c:.1f}" for n, c in PARAM_PAIRS]
    n_pairs     = len(PARAM_PAIRS)
    n_pol       = len(POLICY_NAMES)

    rank_matrix = []
    adap_ranks  = []

    for noise, pers in PARAM_PAIRS:
        means   = {p: results[(noise, pers)][p]["mean"] for p in POLICY_NAMES}
        ranked  = sorted(POLICY_NAMES, key=lambda p: means[p])
        rank_d  = {p: ranked.index(p) + 1 for p in POLICY_NAMES}
        rank_matrix.append(rank_d)
        adap_ranks.append(rank_d["adaptive"])

    fig, axs = plt.subplots(1, 2, figsize=(20, 7))
    fig.suptitle("Policy Rankings Across Parameter Pairs",
                 fontsize=13, fontweight="bold")

    colors_rank = ["gold", "silver", "#cd7f32"] + ["lightcoral"] * (n_pol - 3)
    bar_colors  = [colors_rank[r - 1] for r in adap_ranks]

    axs[0].bar(range(n_pairs), adap_ranks, color=bar_colors,
               edgecolor="black", alpha=0.9)
    axs[0].axhline(1, linestyle="--", color="gold",    alpha=0.6, label="1st")
    axs[0].axhline(2, linestyle="--", color="silver",  alpha=0.6, label="2nd")
    axs[0].axhline(3, linestyle="--", color="#cd7f32", alpha=0.6, label="3rd")
    axs[0].set_xticks(range(n_pairs))
    axs[0].set_xticklabels(pair_labels, fontsize=6, rotation=45, ha="right")
    axs[0].set_ylabel("Rank (1 = lowest cost)")
    axs[0].set_title("Adaptive Policy Rank per Parameter Pair")
    axs[0].set_ylim(0, n_pol + 0.5)
    axs[0].legend(fontsize=8)

    mean_gaps = {
        p: np.mean([results[(n, c)][p]["gap_vs_hint_pct"]
                    for n, c in PARAM_PAIRS])
        for p in POLICY_NAMES
    }
    sorted_pols = sorted(POLICY_NAMES, key=lambda p: mean_gaps[p])
    bar_vals    = [mean_gaps[p] for p in sorted_pols]
    bar_cols    = [POLICY_COLORS.get(p, "gray") for p in sorted_pols]

    axs[1].bar(sorted_pols, bar_vals, color=bar_cols,
               edgecolor="black", alpha=0.85)
    axs[1].set_ylabel("Mean Hindsight Gap (%) across all pairs")
    axs[1].set_title("Average Hindsight Gap per Policy\n(averaged across all 20 parameter pairs)")
    axs[1].tick_params(axis="x", labelsize=9, rotation=15)

    plt.tight_layout()
    plt.savefig("Policy_Rankings.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: Policy_Rankings.png")


def plot_gap_by_noise_and_corr(results):
    """
    Line plots of hindsight gap vs noise (at fixed persistence) and
    vs persistence (at fixed noise) for each policy.
    """

    fig, axs = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("Hindsight Gap (%) — Effect of Noise and Persistence",
                 fontsize=13, fontweight="bold")

    fixed_pers  = 0.5
    noise_pairs = [(n, c) for n, c in PARAM_PAIRS if c == fixed_pers]
    noise_vals  = [n for n, _ in noise_pairs]

    for p in POLICY_NAMES:
        gaps = [results[(n, fixed_pers)][p]["gap_vs_hint_pct"]
                for n in noise_vals]
        axs[0].plot(noise_vals, gaps, marker="o",
                    color=POLICY_COLORS.get(p, "black"),
                    linewidth=2, label=p)

    axs[0].set_xlabel("Noise Std", fontsize=10)
    axs[0].set_ylabel("Hindsight Gap (%)", fontsize=10)
    axs[0].set_title(f"Gap vs Noise  (persistence fixed at {fixed_pers})")
    axs[0].legend(fontsize=7)
    axs[0].grid(alpha=0.3)

    fixed_noise = 0.20
    pers_pairs  = [(n, c) for n, c in PARAM_PAIRS if n == fixed_noise]
    pers_vals   = [c for _, c in pers_pairs]

    for p in POLICY_NAMES:
        gaps = [results[(fixed_noise, c)][p]["gap_vs_hint_pct"]
                for c in pers_vals]
        axs[1].plot(pers_vals, gaps, marker="o",
                    color=POLICY_COLORS.get(p, "black"),
                    linewidth=2, label=p)

    axs[1].set_xlabel("Persistence", fontsize=10)
    axs[1].set_ylabel("Hindsight Gap (%)", fontsize=10)
    axs[1].set_title(f"Gap vs Persistence  (noise fixed at {fixed_noise})")
    axs[1].legend(fontsize=7)
    axs[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("Policy_vs_Persistence.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: Policy_vs_Persistence.png")


def plot_win_tie_loss(results):
    """
    Adaptive win/tie/loss rate against each other policy,
    averaged across all 20 parameter pairs.
    """

    other_policies = [p for p in POLICY_NAMES if p != "adaptive"]

    avg_wins  = []
    avg_ties  = []
    avg_loss  = []

    for p in other_policies:
        w = np.mean([results[(n, c)][p]["adaptive_wins"] for n, c in PARAM_PAIRS])
        t = np.mean([results[(n, c)][p]["tied"]          for n, c in PARAM_PAIRS])
        l = np.mean([results[(n, c)][p]["policy_wins"]   for n, c in PARAM_PAIRS])

        avg_wins.append(w * 100)
        avg_ties.append(t * 100)
        avg_loss.append(l * 100)

    x = np.arange(len(other_policies))
    w = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - w, avg_wins, w, label="Adaptive wins", color="steelblue", alpha=0.85)
    ax.bar(x,     avg_ties, w, label="Tied",          color="gray",      alpha=0.85)
    ax.bar(x + w, avg_loss, w, label="Policy wins",   color="salmon",    alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(other_policies, fontsize=9)
    ax.axhline(50, linestyle="--", color="black", alpha=0.4)
    ax.set_ylabel("% of seeds (averaged across all 20 parameter pairs)")
    ax.set_title("Adaptive Win / Tie / Loss vs Each Policy\n"
                 "(averaged across all 20 parameter pairs)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig("Win_Tie_Loss.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: Win_Tie_Loss.png")


# ============================================================
# FINE-GRID SWEEP FOR WIN-RATE HEATMAP
# ============================================================

def run_heatmap_sweep():
    """
    Runs evaluate_setting over HEATMAP_PAIRS (the fine grid).
    Skips any pair already evaluated in the main sweep results
    by returning a separate dict keyed the same way.
    """
    total   = len(HEATMAP_PAIRS)
    results = {}

    for idx, (noise, pers) in enumerate(HEATMAP_PAIRS):
        print(
            f"[heatmap {idx+1:>2}/{total}]  noise={noise:.3f}  "
            f"persistence={pers:.2f}  ...",
            end="", flush=True
        )
        stats = evaluate_setting(noise, pers)
        results[(noise, pers)] = stats
        gap = stats["adaptive"]["gap_vs_hint_pct"]
        print(f"  adaptive mean hindsight gap: {gap:.1f}%")

    return results


def plot_oracle_winrate_heatmap(heatmap_results):
    """
    Two-panel heatmap showing mean hindsight gap (%) for the adaptive policy:
      Left panel:  Noise Sweep  (persistence fixed at HEATMAP_FIXED_PERS)
      Right panel: Persistence Sweep (noise fixed at HEATMAP_FIXED_NOISE)
    Colour = adaptive mean hindsight gap (%) — lower is better.
    """
    import matplotlib.colors as mcolors

    noise_gaps = np.array(
        [heatmap_results[(n, HEATMAP_FIXED_PERS)]["adaptive"]["gap_vs_hint_pct"]
         for n in HEATMAP_NOISE_VALS]
    ).reshape(-1, 1)

    pers_gaps = np.array(
        [heatmap_results[(HEATMAP_FIXED_NOISE, p)]["adaptive"]["gap_vs_hint_pct"]
         for p in HEATMAP_PERSISTENCE_VALS]
    ).reshape(-1, 1)

    all_vals = np.concatenate([noise_gaps.flatten(), pers_gaps.flatten()])
    vmax     = max(30.0, float(np.ceil(all_vals.max())))

    cmap = plt.cm.RdYlGn_r          # red = high gap (bad), green = low gap (good)
    norm = mcolors.Normalize(vmin=0, vmax=vmax)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Mean Hindsight Gap (%) — Adaptive Policy\n(lower = closer to perfect foresight)",
                 fontsize=13, fontweight="bold")

    # ── Left: Noise Sweep ────────────────────────────────────
    im0 = axs[0].imshow(noise_gaps, cmap=cmap, norm=norm, aspect="auto")
    axs[0].set_yticks(range(len(HEATMAP_NOISE_VALS)))
    axs[0].set_yticklabels([f"{n:.2f}" for n in HEATMAP_NOISE_VALS], fontsize=10)
    axs[0].set_xticks([])
    axs[0].set_ylabel("Noise Std", fontsize=11)
    axs[0].set_title(f"Noise Sweep\n(persistence fixed at {HEATMAP_FIXED_PERS})",
                     fontsize=10)
    for i, v in enumerate(noise_gaps.flatten()):
        axs[0].text(0, i, f"{abs(v):.1f}%", ha="center", va="center",
                    fontsize=11, fontweight="bold", color="black")
    cbar0 = plt.colorbar(im0, ax=axs[0])
    cbar0.set_label("Mean Hindsight Gap (%)", fontsize=10)

    # ── Right: Persistence Sweep ─────────────────────────────
    im1 = axs[1].imshow(pers_gaps, cmap=cmap, norm=norm, aspect="auto")
    axs[1].set_yticks(range(len(HEATMAP_PERSISTENCE_VALS)))
    axs[1].set_yticklabels([f"{p:.2f}" for p in HEATMAP_PERSISTENCE_VALS], fontsize=10)
    axs[1].set_xticks([])
    axs[1].set_ylabel("Persistence", fontsize=11)
    axs[1].set_title(f"Persistence Sweep\n(noise fixed at {HEATMAP_FIXED_NOISE})",
                     fontsize=10)
    for i, v in enumerate(pers_gaps.flatten()):
        axs[1].text(0, i, f"{v:.1f}%", ha="center", va="center",
                    fontsize=11, fontweight="bold", color="black")
    cbar1 = plt.colorbar(im1, ax=axs[1])
    cbar1.set_label("Mean Hindsight Gap (%)", fontsize=10)

    plt.tight_layout()
    plt.savefig("HindsightGap_Heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: HindsightGap_Heatmap.png")

# ============================================================
# SUMMARY TABLE
# ============================================================

def print_summary_table(results):

    print("\n" + "=" * 120)
    print(
        f"{'Noise':<7} {'Pers':<6} "
        + "".join(f"{p[:8]:>10}" for p in POLICY_NAMES)
        + f"  {'Hint':>8}"
    )
    print(f"{'':13}" + "  (hindsight gap % — lower is better)")
    print("-" * 120)

    for noise, pers in PARAM_PAIRS:
        s = results[(noise, pers)]
        row = f"{noise:<7.2f} {pers:<6.1f} "
        for p in POLICY_NAMES:
            row += f"{s[p]['gap_vs_hint_pct']:>10.1f}"
        row += f"  {s['hindsight_mean']:>8.2f}"
        print(row)

    print("=" * 120)

    print("\nAVERAGE ACROSS ALL 20 PAIRS:")
    row = f"{'':13}"
    for p in POLICY_NAMES:
        avg = np.mean([results[(n, c)][p]["gap_vs_hint_pct"]
                       for n, c in PARAM_PAIRS])
        row += f"{avg:>10.1f}"
    print(row)
    print("=" * 120)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        f"\nRunning policy sweep: "
        f"{len(PARAM_PAIRS)} parameter pairs x "
        f"{SEEDS} seeds = "
        f"{len(PARAM_PAIRS) * SEEDS} simulations\n"
    )

    results = run_sweep()

    print_summary_table(results)
    plot_gap_heatmaps(results)
    plot_oracle_sweep(results)
    plot_policy_ranking(results)
    plot_gap_by_noise_and_corr(results)
    plot_win_tie_loss(results)

    print("\nRunning fine-grid sweep for win-rate heatmap...")
    heatmap_results = run_heatmap_sweep()
    plot_oracle_winrate_heatmap(heatmap_results)