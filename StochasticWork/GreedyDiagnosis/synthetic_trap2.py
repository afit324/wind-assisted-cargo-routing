"""
traps.py
========
Two synthetic scenario traps to test adaptive vs greedy policies.

Trap 1 — Honeypot
    A cheap arc early on leads to expensive arcs later.
    Adaptive sees through the trap via cost-to-go lookahead.
    Greedy falls for it every time.

Trap 2 — Switching Wind Fields
    Two wind fields (A and B) switch randomly each stage.
    Policy commits to next node BEFORE the field is drawn.
    Plans under current field; pays under realised field.
    Adaptive stays near centre nodes (robust to either field).
    Greedy commits hard to one extreme and pays heavily on switches.
"""

import os
import sys
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# ── silence policy prints ────────────────────────────────────
stdout_backup = sys.stdout
sys.stdout    = open(os.devnull, "w")

from policy_grib import (
    layers,
    NUM_FORECASTS,
    NUM_STAGES,
    NODES_PER_STAGE,
    NODE_COORDS,
    NODE_LATS,
    STAGE_LONS,
    choose_next_node,
    future_cost_to_go,
)

sys.stdout.close()
sys.stdout = stdout_backup

# ============================================================
# TRAP 2 PARAMETERS — tweak to make greedy suffer more
# ============================================================

SWITCH_PROB       = 0.55
FIELD_SEED        = 7
FIELD_A_LOW_COST  = 0.05    # very cheap for low-index nodes in field A
FIELD_A_HIGH_COST = 2.50    # very expensive for high-index nodes in field A
FIELD_B_LOW_COST  = 2.50    # very expensive for low-index nodes in field B
FIELD_B_HIGH_COST = 0.05    # very cheap for high-index nodes in field B
LATERAL_PENALTY = 0.05
MID_IDX           = NODES_PER_STAGE // 2  

ADAPTIVE_COLOR = "#1565C0"
GREEDY_COLOR   = "#BF360C"

# ============================================================
# SHARED HELPERS
# ============================================================

def _cost_to_go_single(node, stage, forecast):
    """Minimum cost from node to END under a single forecast dict."""
    if stage == len(layers) - 1:
        return 0.0
    return min(
        forecast[node][n] + _cost_to_go_single(n, stage + 1, forecast)
        for n in layers[stage + 1]
    )


def _greedy_next(current_node, stage, forecast):
    """One-step greedy: cheapest immediate arc."""
    return min(
        layers[stage + 1],
        key=lambda n: forecast[current_node][n]
    )


def _adaptive_next(current_node, stage, forecast):
    """Full lookahead under single forecast."""
    return min(
        layers[stage + 1],
        key=lambda n: (
            forecast[current_node][n] +
            _cost_to_go_single(n, stage + 1, forecast)
        )
    )


# ============================================================
# TRAP 1 — HONEYPOT
# ============================================================

def build_honeypot_forecasts(
    base_cost     = 1.0,
    honey_entry   = 0.1,
    honey_exit    = 5.0,
    honeypot_node = "S0_N0",
):
    forecast = {}
    for i, layer in enumerate(layers[:-1]):
        for from_node in layer:
            forecast[from_node] = {}
            for to_node in layers[i + 1]:
                if from_node == "START" and to_node == honeypot_node:
                    cost = honey_entry
                elif from_node == honeypot_node:
                    cost = honey_exit
                else:
                    cost = base_cost
                forecast[from_node][to_node] = cost
    return [copy.deepcopy(forecast) for _ in range(NUM_FORECASTS)]


def run_trap_1(seed, policy="adaptive",
               base_cost=1.0, honey_entry=0.1, honey_exit=5.0,
               honeypot_node="S0_N0"):
    random.seed(seed)
    np.random.seed(seed)

    forecasts    = build_honeypot_forecasts(
        base_cost, honey_entry, honey_exit, honeypot_node
    )
    current_node = "START"
    stage        = 0
    total_cost   = 0.0
    path         = [current_node]
    stage_costs  = []

    while current_node != "END":
        if policy == "adaptive":
            next_node = _adaptive_next(current_node, stage, forecasts[0])
        else:
            next_node = _greedy_next(current_node, stage, forecasts[0])

        cost        = forecasts[0][current_node][next_node]
        total_cost += cost
        stage_costs.append(cost)
        current_node = next_node
        path.append(current_node)
        stage += 1
        if stage == len(layers) - 1:
            break

    return total_cost, path, stage_costs


# ============================================================
# TRAP 2 — SWITCHING WIND FIELDS
# ============================================================

FIELD_A_WINDS = {
    0: np.array([20.0, 18.0, 16.0, 14.0,  3.0,  2.0,  2.0,  2.0]),  # N0: trap
    1: np.array([ 8.0,  8.0,  8.0,  8.0,  8.0,  8.0,  8.0,  8.0]),  # N1: moderate
    2: np.array([10.0, 10.0, 10.0, 10.0, 4.0, 6.0, 7.0, 8.0]),  # N2: steady
    3: np.array([ 5.0,  6.0,  8.0, 10.0, 14.0, 16.0, 18.0, 20.0]),  # N3: building
    4: np.array([ 2.0,  2.0,  3.0,  5.0, 14.0, 16.0, 18.0, 20.0]),  # N4: reward
}
 
FIELD_B_WINDS = {
    0: np.array([ 18.0,  16.0,  16.0,  4.0,  4.0,  3.0,  3.0,  8.0]),  # N0: moderate
    1: np.array([20.0, 18.0, 16.0, 14.0,  3.0,  2.0,  2.0,  2.0]),  # N1: trap
    2: np.array([10.0, 10.0, 10.0, 10.0, 10.0, 6.0, 6.0, 8.0]),  # N2: steady
    3: np.array([ 5.0,  6.0,  8.0, 10.0, 14.0, 16.0, 18.0, 20.0]),  # N3: building
    4: np.array([ 2.0,  2.0,  3.0,  5.0, 14.0, 16.0, 18.0, 20.0]),  # N4: reward
}

def build_switching_fields(field_seed=FIELD_SEED):
    """
    Two fields that both trap greedy but in different ways.

    Field A: N0 is the trap (cheap early, collapses late)
             N4 is the reward (expensive early, best late)
    Field B: N1 is the trap (cheap early, collapses late)
             N0/N4 are rewards

    N2 is steady (10 m/s) in BOTH fields — adaptive's
    averaged cost-to-go always identifies N2 as the safe
    corridor regardless of which field is active.

    When fields switch greedy is stuck in whichever trap
    node it committed to. Adaptive stays near N2 throughout.
    """

    def get_ws(node, stage, field_winds):
        if node in ("START", "END"):
            return 10.0
        idx = int(node.split("_N")[1])
        s   = min(stage, NUM_STAGES - 1)
        return float(field_winds[idx][s])

    def get_idx(node):
        if node in ("START", "END"):
            return MID_IDX
        return int(node.split("_N")[1])

    def build_field(field_winds):
        field = {}
        for i, layer in enumerate(layers[:-1]):
            for from_node in layer:
                field[from_node] = {}
                from_idx = get_idx(from_node)
                ws_from  = get_ws(from_node, i, field_winds)

                for to_node in layers[i + 1]:
                    to_idx = get_idx(to_node)
                    ws_to  = get_ws(to_node, i + 1, field_winds)

                    wind_cost = 1.0 / max((ws_from + ws_to) / 2.0, 0.1)
                    pen       = LATERAL_PENALTY * max(
                        0, abs(from_idx - to_idx) - 1
                    )
                    field[from_node][to_node] = wind_cost + pen

        return field

    field_A = build_field(FIELD_A_WINDS)
    field_B = build_field(FIELD_B_WINDS)

    return field_A, field_B


def run_trap_2(seed, policy="adaptive",
               switch_probability=SWITCH_PROB,
               field_seed=FIELD_SEED):
    """
    Policy commits to next node under CURRENT field.
    AFTER commitment, realised field is drawn randomly.
    Realised field becomes current field next stage.
    """
    random.seed(seed)
    np.random.seed(seed)

    field_A, field_B = build_switching_fields(field_seed)
    fields           = [field_A, field_B]

    current_node      = "START"
    stage             = 0
    total_cost        = 0.0
    path              = [current_node]
    stage_costs       = []
    field_sequence    = []
    current_field_idx = 0   # start on field A

    while current_node != "END":

        # Step 1: policy plans under CURRENT field (no switch knowledge)
        planning_field = fields[current_field_idx]

        if policy == "adaptive":
            next_node = _adaptive_next(current_node, stage, planning_field)
        else:
            next_node = _greedy_next(current_node, stage, planning_field)

        # Step 2: AFTER commitment, draw realised field
        if random.random() < switch_probability:
            realised_field_idx = 1 - current_field_idx
        else:
            realised_field_idx = current_field_idx

        realised_field = fields[realised_field_idx]
        field_sequence.append(realised_field_idx)

        # Step 3: pay cost under realised field
        cost        = realised_field[current_node][next_node]
        total_cost += cost
        stage_costs.append(cost)

        # Step 4: realised field becomes new current field
        current_field_idx = realised_field_idx
        current_node      = next_node
        path.append(current_node)
        stage += 1

        if stage == len(layers) - 1:
            break

    return total_cost, path, stage_costs, field_sequence


def run_trap_2_detailed(seed, switch_probability=SWITCH_PROB,
                        field_seed=FIELD_SEED):
    """
    Runs both policies and returns full step-by-step decision data
    for the visualisation.
    """
    random.seed(seed)
    np.random.seed(seed)

    field_A, field_B = build_switching_fields(field_seed)
    fields           = [field_A, field_B]
    field_names      = ["A (N0 cheap)", "B (N4 cheap)"]

    # Pre-draw field sequence so both policies face identical world
    field_sequence    = []
    current_field_idx = 0
    for _ in range(NUM_STAGES):
        if random.random() < switch_probability:
            current_field_idx = 1 - current_field_idx
        field_sequence.append(current_field_idx)

    def simulate_policy(policy_name):
        current_node      = "START"
        stage             = 0
        total_cost        = 0.0
        path              = ["START"]
        stage_costs       = []
        decisions         = []
        cur_field_idx     = 0

        while current_node != "END":
            planning_field = fields[cur_field_idx]
            next_layer     = layers[stage + 1]

            # Collect candidate info
            cand_info = []
            for candidate in next_layer:
                imm  = planning_field[current_node][candidate]
                ctg  = (_cost_to_go_single(candidate, stage + 1,
                                           planning_field)
                        if policy_name == "adaptive" else 0.0)
                score = imm + ctg
                cidx  = int(candidate.split("_N")[1]) \
                        if candidate != "END" else -1
                cand_info.append({
                    "node"    : candidate,
                    "node_idx": cidx,
                    "imm"     : imm,
                    "ctg"     : ctg,
                    "score"   : score,
                })

            best      = min(cand_info, key=lambda x: x["score"])
            next_node = best["node"]

            # ADD THIS CHECK HERE — before accessing field_sequence
            if next_node == "END":
                path.append("END")
                break

            # Realised field drawn AFTER commitment
            realised_field_idx = field_sequence[stage]
            realised_field     = fields[realised_field_idx]
            realised_cost      = realised_field[current_node][next_node]

            total_cost += realised_cost
            stage_costs.append(realised_cost)

            decisions.append({
                "stage"              : stage,
                "from_node"          : current_node,
                "chosen_node"        : next_node,
                "chosen_idx"         : best["node_idx"],
                "planning_field_idx" : cur_field_idx,
                "planning_field_name": field_names[cur_field_idx],
                "realised_field_idx" : realised_field_idx,
                "realised_field_name": field_names[realised_field_idx],
                "field_switched"     : realised_field_idx != cur_field_idx,
                "imm_planned"        : best["imm"],
                "ctg_planned"        : best["ctg"],
                "score_used"         : best["score"],
                "realised_cost"      : realised_cost,
                "surprise"           : realised_cost - best["imm"],
                "cumulative"         : float(total_cost),
                "all_candidates"     : cand_info,
            })

            cur_field_idx = realised_field_idx
            current_node  = next_node
            path.append(current_node)
            stage += 1
            if stage == len(layers) - 1:
                break

        return {
            "path"       : path,
            "stage_costs": stage_costs,
            "total_cost" : total_cost,
            "decisions"  : decisions,
        }

    adap   = simulate_policy("adaptive")
    greedy = simulate_policy("greedy")
    return adap, greedy, field_sequence, fields, field_names


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

def run_trap_experiment(n_seeds=20, base_seed=0,
                        switch_probability=SWITCH_PROB,
                        field_seed=FIELD_SEED):
    results = {
        "trap1_adaptive": [], "trap1_greedy": [],
        "trap2_adaptive": [], "trap2_greedy": [],
        "trap2_adaptive_paths": [], "trap2_greedy_paths": [],
    }

    for i in range(n_seeds):
        seed = base_seed + i

        c_a, _, _ = run_trap_1(seed, policy="adaptive")
        c_g, _, _ = run_trap_1(seed, policy="greedy")
        results["trap1_adaptive"].append(c_a)
        results["trap1_greedy"].append(c_g)

        c_a2, p_a, _, _ = run_trap_2(
            seed, policy="adaptive",
            switch_probability=switch_probability,
            field_seed=field_seed
        )
        c_g2, p_g, _, _ = run_trap_2(
            seed, policy="greedy",
            switch_probability=switch_probability,
            field_seed=field_seed
        )
        results["trap2_adaptive"].append(c_a2)
        results["trap2_greedy"].append(c_g2)
        results["trap2_adaptive_paths"].append(p_a)
        results["trap2_greedy_paths"].append(p_g)

    for k in results:
        if isinstance(results[k][0], (int, float)):
            results[k] = np.array(results[k])

    return results


# ============================================================
# STATIC PLOTS
# ============================================================

def plot_trap_results(results, switch_probability=SWITCH_PROB):

    fig, axs = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        "Trap Experiments: Adaptive vs Greedy",
        fontsize=14, fontweight="bold"
    )

    adap_col = "#1565C0"
    grdy_col = "#BF360C"

    # Trap 1 distributions
    axs[0, 0].hist(results["trap1_adaptive"], bins=20, alpha=0.7,
                   color=adap_col, label="Adaptive")
    axs[0, 0].hist(results["trap1_greedy"],   bins=20, alpha=0.7,
                   color=grdy_col, label="Greedy")
    axs[0, 0].axvline(np.mean(results["trap1_adaptive"]),
                      color=adap_col, linestyle="--", linewidth=2)
    axs[0, 0].axvline(np.mean(results["trap1_greedy"]),
                      color=grdy_col, linestyle="--", linewidth=2)
    axs[0, 0].set_title("Trap 1 — Honeypot: Cost Distribution")
    axs[0, 0].set_xlabel("Total cost")
    axs[0, 0].legend()

    # Trap 1 scatter
    axs[0, 1].scatter(results["trap1_greedy"],
                      results["trap1_adaptive"],
                      alpha=0.4, color="steelblue", s=15)
    lims = [
        min(results["trap1_adaptive"].min(),
            results["trap1_greedy"].min()) * 0.95,
        max(results["trap1_adaptive"].max(),
            results["trap1_greedy"].max()) * 1.05,
    ]
    axs[0, 1].plot(lims, lims, "k--", alpha=0.4)
    axs[0, 1].set_xlabel("Greedy cost")
    axs[0, 1].set_ylabel("Adaptive cost")
    axs[0, 1].set_title("Trap 1 — Scatter (below diagonal = adaptive wins)")

    # Trap 1 bar
    wr1 = np.mean(results["trap1_greedy"] > results["trap1_adaptive"]) * 100
    axs[0, 2].bar(["Adaptive", "Greedy"],
                  [np.mean(results["trap1_adaptive"]),
                   np.mean(results["trap1_greedy"])],
                  color=[adap_col, grdy_col], alpha=0.85,
                  yerr=[np.std(results["trap1_adaptive"]),
                        np.std(results["trap1_greedy"])],
                  capsize=6)
    axs[0, 2].set_title(
        f"Trap 1 — Mean Cost ± Std\nAdaptive wins: {wr1:.1f}%"
    )
    axs[0, 2].set_ylabel("Mean total cost")

    # Trap 2 distributions
    axs[1, 0].hist(results["trap2_adaptive"], bins=20, alpha=0.7,
                   color=adap_col, label="Adaptive")
    axs[1, 0].hist(results["trap2_greedy"],   bins=20, alpha=0.7,
                   color=grdy_col, label="Greedy")
    axs[1, 0].axvline(np.mean(results["trap2_adaptive"]),
                      color=adap_col, linestyle="--", linewidth=2)
    axs[1, 0].axvline(np.mean(results["trap2_greedy"]),
                      color=grdy_col, linestyle="--", linewidth=2)
    axs[1, 0].set_title(
        f"Trap 2 — Switching Fields (p={switch_probability}): Cost Distribution"
    )
    axs[1, 0].set_xlabel("Total cost")
    axs[1, 0].legend()

    # Trap 2 scatter
    axs[1, 1].scatter(results["trap2_greedy"],
                      results["trap2_adaptive"],
                      alpha=0.4, color="steelblue", s=15)
    lims2 = [
        min(results["trap2_adaptive"].min(),
            results["trap2_greedy"].min()) * 0.95,
        max(results["trap2_adaptive"].max(),
            results["trap2_greedy"].max()) * 1.05,
    ]
    axs[1, 1].plot(lims2, lims2, "k--", alpha=0.4)
    axs[1, 1].set_xlabel("Greedy cost")
    axs[1, 1].set_ylabel("Adaptive cost")
    axs[1, 1].set_title("Trap 2 — Scatter (below diagonal = adaptive wins)")

    # Trap 2 node visit distribution
    adap_counts = np.zeros(NODES_PER_STAGE)
    grdy_counts = np.zeros(NODES_PER_STAGE)
    for path in results["trap2_adaptive_paths"]:
        for node in path:
            if node not in ("START", "END"):
                adap_counts[int(node.split("_N")[1])] += 1
    for path in results["trap2_greedy_paths"]:
        for node in path:
            if node not in ("START", "END"):
                grdy_counts[int(node.split("_N")[1])] += 1

    x = np.arange(NODES_PER_STAGE)
    w = 0.35
    axs[1, 2].bar(x - w/2, adap_counts, w,
                  color=adap_col, alpha=0.85, label="Adaptive")
    axs[1, 2].bar(x + w/2, grdy_counts, w,
                  color=grdy_col, alpha=0.85, label="Greedy")
    axs[1, 2].set_xticks(x)
    axs[1, 2].set_xticklabels([f"N{i}" for i in range(NODES_PER_STAGE)])
    axs[1, 2].set_xlabel("Node index")
    axs[1, 2].set_ylabel("Total visits")
    axs[1, 2].set_title(
        "Trap 2 — Node Visit Distribution\n"
        "(adaptive centres on N2; greedy swings to extremes)"
    )
    axs[1, 2].legend()

    wr2   = np.mean(results["trap2_greedy"] > results["trap2_adaptive"]) * 100
    adv2  = np.mean(results["trap2_greedy"] - results["trap2_adaptive"])

    print("\n" + "=" * 60)
    print("TRAP EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"\nTrap 1 — Honeypot:")
    print(f"  Adaptive mean : {np.mean(results['trap1_adaptive']):.4f}")
    print(f"  Greedy   mean : {np.mean(results['trap1_greedy']):.4f}")
    print(f"  Adaptive wins : {wr1:.1f}%")
    print(f"\nTrap 2 — Switching Fields (p={switch_probability}):")
    print(f"  Adaptive mean : {np.mean(results['trap2_adaptive']):.4f}")
    print(f"  Greedy   mean : {np.mean(results['trap2_greedy']):.4f}")
    print(f"  Adaptive wins : {wr2:.1f}%")
    print(f"  Mean advantage: {adv2:.4f}")
    print("=" * 60)

    plt.tight_layout()
    plt.show()


# ============================================================
# INTERACTIVE STEP-BY-STEP VISUALISATION
# ============================================================

def build_interactive_viz(seed=42, switch_probability=SWITCH_PROB,
                          field_seed=FIELD_SEED):
    """
    Builds an interactive Plotly visualisation of Trap 2 showing
    exactly what each policy sees and decides at every stage.
    Saves to trap2_viz.html.
    """

    adap, greedy, field_sequence, fields, field_names = \
        run_trap_2_detailed(seed, switch_probability, field_seed)

    n_steps = NUM_STAGES + 1

    def field_arc_cost(field_idx, from_node, to_node):
        return fields[field_idx].get(from_node, {}).get(to_node, 1.0)

    def node_mean_cost(field_idx, node):
        """Average cost of arcs leading TO this node under field_idx."""
        s = int(node.split("_N")[0].replace("S", ""))
        parents = ["START"] if s == 0 else [
            f"S{s-1}_N{i}" for i in range(NODES_PER_STAGE)
        ]
        costs = [
            fields[field_idx][p][node]
            for p in parents
            if p in fields[field_idx] and node in fields[field_idx][p]
        ]
        return float(np.mean(costs)) if costs else 1.0

    frames_data = []

    for step in range(n_steps):

        if step == 0:
            cur_field_idx = 0
        else:
            cur_field_idx = adap["decisions"][step - 1]["realised_field_idx"]

        adap_visited   = set(adap["path"][:step + 1])
        greedy_visited = set(greedy["path"][:step + 1])

        # Current positions
        adap_cur   = adap["path"][min(step, len(adap["path"]) - 1)]
        greedy_cur = greedy["path"][min(step, len(greedy["path"]) - 1)]

        # ── nodes ─────────────────────────────────────────────
        node_lats  = []
        node_lons  = []
        node_costs = []
        node_sizes = []
        node_syms  = []
        node_text  = []

        for s in range(NUM_STAGES):
            for i in range(NODES_PER_STAGE):
                node     = f"S{s}_N{i}"
                lat, lon = NODE_COORDS[node]

                cost_curr = node_mean_cost(cur_field_idx, node)
                cost_A    = node_mean_cost(0, node)
                cost_B    = node_mean_cost(1, node)
                ctg_A     = _cost_to_go_single(node, s + 1, fields[0])
                ctg_B     = _cost_to_go_single(node, s + 1, fields[1])

                is_adap_cur   = (node == adap_cur   and adap_cur   != "START")
                is_greedy_cur = (node == greedy_cur and greedy_cur != "START")
                in_adap       = node in adap_visited
                in_greedy     = node in greedy_visited

                node_lats.append(lat)
                node_lons.append(lon)
                node_costs.append(cost_curr)
                node_sizes.append(
                    24 if (is_adap_cur or is_greedy_cur)
                    else 16 if (in_adap or in_greedy)
                    else 9
                )
                node_syms.append(
                    "star"    if (is_adap_cur and is_greedy_cur)
                    else "diamond" if is_adap_cur
                    else "square"  if is_greedy_cur
                    else "circle"
                )
                node_text.append(
                    f"<b>{node}</b> (N{i} at stage {s})<br>"
                    f"Lat: {lat:.1f}°  Lon: {lon:.1f}°<br>"
                    f"────────────────────<br>"
                    f"Cost under Field A: {cost_A:.3f}<br>"
                    f"Cost under Field B: {cost_B:.3f}<br>"
                    f"Cost under current ({['A','B'][cur_field_idx]}): "
                    f"{cost_curr:.3f}<br>"
                    f"────────────────────<br>"
                    f"CTG under Field A: {ctg_A:.3f}<br>"
                    f"CTG under Field B: {ctg_B:.3f}<br>"
                    f"────────────────────<br>"
                    + ("★ ADAPTIVE here<br>" if is_adap_cur else "")
                    + ("■ GREEDY here<br>"   if is_greedy_cur else "")
                    + ("◆ On adaptive path<br>"
                       if in_adap and not is_adap_cur else "")
                    + ("□ On greedy path<br>"
                       if in_greedy and not is_greedy_cur else "")
                )

        # ── arcs ──────────────────────────────────────────────
        arc_lats = []
        arc_lons = []
        for s in range(NUM_STAGES - 1):
            for i in range(NODES_PER_STAGE):
                fn = f"S{s}_N{i}"
                lat_f, lon_f = NODE_COORDS[fn]
                for j in range(NODES_PER_STAGE):
                    tn = f"S{s+1}_N{j}"
                    lat_t, lon_t = NODE_COORDS[tn]
                    arc_lats.extend([lat_f, lat_t, None])
                    arc_lons.extend([lon_f, lon_t, None])

        # ── policy paths ──────────────────────────────────────
        def path_coords(path, n):
            lats, lons = [], []
            for nd in path[:n + 1]:
                if nd in ("START", "END"):
                    continue
                lat, lon = NODE_COORDS.get(nd, (None, None))
                if lat is not None:
                    lats.append(lat)
                    lons.append(lon)
            return lats, lons

        adap_lats,   adap_lons   = path_coords(adap["path"],   step)
        greedy_lats, greedy_lons = path_coords(greedy["path"], step)

        # ── decision text ─────────────────────────────────────
        if step > 0 and step <= NUM_STAGES:
            d_a = adap["decisions"][step - 1]
            d_g = greedy["decisions"][step - 1]

            sw_str = "⚡ FIELD SWITCHED" if d_a["field_switched"] else "Field unchanged"

            # Adaptive text
            adap_txt = (
                f"<b>ADAPTIVE — Stage {step-1}</b><br>"
                f"Planning under Field "
                f"{['A','B'][d_a['planning_field_idx']]}<br>"
                f"────────────────────────<br>"
                f"Chose: {d_a['chosen_node']} (N{d_a['chosen_idx']})<br><br>"
                f"<b>Decision breakdown:</b><br>"
                f"  Immediate cost:    {d_a['imm_planned']:.4f}<br>"
                f"  Cost-to-go:        {d_a['ctg_planned']:.4f}<br>"
                f"  Total score:       {d_a['score_used']:.4f}<br><br>"
                f"<b>After commitment:</b><br>"
                f"  {sw_str}<br>"
                f"  Realised field:    "
                f"{'A' if d_a['realised_field_idx']==0 else 'B'}<br>"
                f"  Realised cost:     {d_a['realised_cost']:.4f}<br>"
                f"  Surprise:          {d_a['surprise']:+.4f}<br><br>"
                f"<b>Cumulative: {d_a['cumulative']:.4f}</b>"
            )

            # Greedy candidates
            cands_txt = ""
            for c in sorted(d_g["all_candidates"], key=lambda x: x["score"]):
                mk = "→ " if c["node"] == d_g["chosen_node"] else "   "
                cands_txt += (
                    f"{mk}N{c['node_idx']}: "
                    f"imm={c['imm']:.3f} (no CTG)<br>"
                )

            greedy_txt = (
                f"<b>GREEDY — Stage {step-1}</b><br>"
                f"Planning under Field "
                f"{['A','B'][d_g['planning_field_idx']]}<br>"
                f"────────────────────────<br>"
                f"Chose: {d_g['chosen_node']} (N{d_g['chosen_idx']})<br><br>"
                f"<b>All candidates (immediate only):</b><br>"
                + cands_txt +
                f"<br><b>After commitment:</b><br>"
                f"  {sw_str}<br>"
                f"  Realised field:    "
                f"{'A' if d_g['realised_field_idx']==0 else 'B'}<br>"
                f"  Realised cost:     {d_g['realised_cost']:.4f}<br>"
                f"  Surprise:          {d_g['surprise']:+.4f}<br><br>"
                f"<b>Cumulative: {d_g['cumulative']:.4f}</b>"
            )

            field_txt = (
                f"<b>Field Status</b><br>"
                f"────────────────────────<br>"
                f"Stage {step-1} draw:<br>"
                f"  Policy planned:  "
                f"{'A' if d_a['planning_field_idx']==0 else 'B'}<br>"
                f"  Realised:        "
                f"{'A' if d_a['realised_field_idx']==0 else 'B'}<br>"
                f"  {'⚡ SWITCHED' if d_a['field_switched'] else 'Same'}<br><br>"
                f"Full sequence:<br>"
                + "".join(
                    f"  S{t}: "
                    f"{'A' if f==0 else 'B'}"
                    f"{' ⚡' if t>0 and f != field_sequence[t-1] else ''}<br>"
                    for t, f in enumerate(field_sequence)
                )
                + f"<br>Adaptive:  {adap['total_cost']:.4f}<br>"
                f"Greedy:    {greedy['total_cost']:.4f}<br>"
                f"Advantage: "
                f"{greedy['total_cost']-adap['total_cost']:+.4f}"
            )

        else:
            adap_txt = (
                f"<b>ADAPTIVE</b><br>At START<br>"
                f"Planning under Field A<br>"
                f"Full lookahead: picks robust<br>"
                f"centre nodes (N2) that are<br>"
                f"acceptable under either field"
            )
            greedy_txt = (
                f"<b>GREEDY</b><br>At START<br>"
                f"Planning under Field A<br>"
                f"No lookahead: picks N0 (cheapest<br>"
                f"under A) — catastrophic when<br>"
                f"field switches to B"
            )
            field_txt = (
                f"<b>Field Status</b><br>"
                f"Starting on Field A<br>"
                f"Switch prob: {switch_probability:.0%}<br><br>"
                f"Field A: N0 very cheap<br>"
                f"Field B: N4 very cheap<br><br>"
                f"Greedy commits to N0 under A<br>"
                f"→ pays N0 cost under B = max"
            )

        frames_data.append({
            "step"         : step,
            "node_lats"    : node_lats,
            "node_lons"    : node_lons,
            "node_costs"   : node_costs,
            "node_sizes"   : node_sizes,
            "node_syms"    : node_syms,
            "node_text"    : node_text,
            "arc_lats"     : arc_lats,
            "arc_lons"     : arc_lons,
            "adap_lats"    : adap_lats,
            "adap_lons"    : adap_lons,
            "greedy_lats"  : greedy_lats,
            "greedy_lons"  : greedy_lons,
            "adap_txt"     : adap_txt,
            "greedy_txt"   : greedy_txt,
            "field_txt"    : field_txt,
            "field_idx"    : cur_field_idx,
        })

    # ── initial figure ────────────────────────────────────────
    f0 = frames_data[0]

    fig = make_subplots(
        rows=2, cols=3,
        specs=[
            [{"colspan": 3, "type": "scatter"}, None, None],
            [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
        ],
        subplot_titles=[
            "Node Layout — Cost under Current Field (hover for details)",
            "Adaptive Decision",
            "Greedy Decision",
            "Field Status",
        ],
        row_heights=[0.58, 0.42],
        vertical_spacing=0.14,
    )

    # Background arcs
    fig.add_trace(go.Scatter(
        x=f0["arc_lons"], y=f0["arc_lats"], mode="lines",
        line=dict(color="rgba(170,170,170,0.10)", width=0.6),
        hoverinfo="skip", showlegend=False, name="_arcs",
    ), row=1, col=1)

    # Nodes
    fig.add_trace(go.Scatter(
        x=f0["node_lons"], y=f0["node_lats"], mode="markers",
        marker=dict(
            size=f0["node_sizes"], color=f0["node_costs"],
            colorscale="RdYlGn_r", showscale=True,
            colorbar=dict(title="Arc cost<br>(cur field)",
                          x=1.01, thickness=13, len=0.55),
            symbol=f0["node_syms"],
            line=dict(color="white", width=1.2),
        ),
        text=f0["node_text"],
        hovertemplate="%{text}<extra></extra>",
        name="Nodes",
    ), row=1, col=1)

    # Policy paths
    fig.add_trace(go.Scatter(
        x=f0["adap_lons"], y=f0["adap_lats"], mode="lines+markers",
        line=dict(color=ADAPTIVE_COLOR, width=3.5),
        marker=dict(size=12, color=ADAPTIVE_COLOR, symbol="diamond",
                    line=dict(color="white", width=2)),
        name="Adaptive",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=f0["greedy_lons"], y=f0["greedy_lats"], mode="lines+markers",
        line=dict(color=GREEDY_COLOR, width=3.5, dash="dash"),
        marker=dict(size=12, color=GREEDY_COLOR, symbol="square",
                    line=dict(color="white", width=2)),
        name="Greedy",
    ), row=1, col=1)

    # Text panels
    for col, txt, nm in [
        (1, f0["adap_txt"],   "_atxt"),
        (2, f0["greedy_txt"], "_gtxt"),
        (3, f0["field_txt"],  "_ftxt"),
    ]:
        fig.add_trace(go.Scatter(
            x=[0], y=[0.95], mode="text", text=[txt],
            textposition="top left",
            textfont=dict(size=9, color="#1a1a2e"),
            showlegend=False, hoverinfo="skip", name=nm,
        ), row=2, col=col)

    # ── animation frames ──────────────────────────────────────
    plotly_frames = []
    for fd in frames_data:
        bg = ("rgba(25,118,210,0.06)" if fd["field_idx"] == 0
              else "rgba(230,74,25,0.06)")
        plotly_frames.append(go.Frame(
            data=[
                go.Scatter(x=fd["arc_lons"], y=fd["arc_lats"],
                           mode="lines",
                           line=dict(color="rgba(170,170,170,0.10)",
                                     width=0.6),
                           hoverinfo="skip", showlegend=False),
                go.Scatter(
                    x=fd["node_lons"], y=fd["node_lats"],
                    mode="markers",
                    marker=dict(
                        size=fd["node_sizes"], color=fd["node_costs"],
                        colorscale="RdYlGn_r", showscale=True,
                        colorbar=dict(title="Arc cost<br>(cur field)",
                                      x=1.01, thickness=13, len=0.55),
                        symbol=fd["node_syms"],
                        line=dict(color="white", width=1.2),
                    ),
                    text=fd["node_text"],
                    hovertemplate="%{text}<extra></extra>",
                ),
                go.Scatter(
                    x=fd["adap_lons"], y=fd["adap_lats"],
                    mode="lines+markers",
                    line=dict(color=ADAPTIVE_COLOR, width=3.5),
                    marker=dict(size=12, color=ADAPTIVE_COLOR,
                                symbol="diamond",
                                line=dict(color="white", width=2)),
                ),
                go.Scatter(
                    x=fd["greedy_lons"], y=fd["greedy_lats"],
                    mode="lines+markers",
                    line=dict(color=GREEDY_COLOR, width=3.5, dash="dash"),
                    marker=dict(size=12, color=GREEDY_COLOR,
                                symbol="square",
                                line=dict(color="white", width=2)),
                ),
                go.Scatter(x=[0], y=[0.95], mode="text",
                           text=[fd["adap_txt"]], textposition="top right",
                           textfont=dict(size=10, color="#1a1a2e"),
                           showlegend=False, hoverinfo="skip"),
                go.Scatter(x=[0], y=[0.95], mode="text",
                           text=[fd["greedy_txt"]], textposition="top right",
                           textfont=dict(size=10, color="#1a1a2e"),
                           showlegend=False, hoverinfo="skip"),
                go.Scatter(x=[0], y=[0.95], mode="text",
                           text=[fd["field_txt"]], textposition="top right",
                           textfont=dict(size=10, color="#1a1a2e"),
                           showlegend=False, hoverinfo="skip"),
            ],
            traces=[0, 1, 2, 3, 4, 5, 6],
            name=str(fd["step"]),
            layout=go.Layout(
                plot_bgcolor=bg,
                title=dict(text=(
                    f"Trap 2 — Switching Fields  |  "
                    f"Step {fd['step']}/{NUM_STAGES}  |  "
                    f"Current field: "
                    f"{'A (N0 cheapest)' if fd['field_idx']==0 else 'B (N4 cheapest)'}  |  "
                    f"Seed {seed}  |  "
                    f"Adaptive: {adap['total_cost']:.4f}  "
                    f"Greedy: {greedy['total_cost']:.4f}"
                ))
            )
        ))

    fig.frames = plotly_frames

    sliders = [dict(
        active=0,
        currentvalue=dict(prefix="Step: ", visible=True,
                          xanchor="center"),
        pad=dict(t=50, b=10),
        steps=[
            dict(
                method="animate",
                args=[[str(fd["step"])], dict(
                    frame=dict(duration=0, redraw=True),
                    mode="immediate",
                    transition=dict(duration=0),
                )],
                label=str(fd["step"]),
            )
            for fd in frames_data
        ],
    )]

    fig.update_layout(
        title=dict(
            text=(
                f"Trap 2 — Switching Wind Fields  |  "
                f"Seed {seed}  |  "
                f"◆ Blue = Adaptive  |  ■ Red = Greedy<br>"
                f"<sup>Blue tint = Field A active  |  "
                f"Orange tint = Field B active  |  "
                f"Node colour = arc cost under current field  |  "
                f"Use slider or ← → keys to step</sup>"
            ),
            x=0.5, xanchor="center",
            font=dict(size=12, color="#1a1a2e"),
        ),
        paper_bgcolor="#f8f9fa",
        plot_bgcolor="rgba(25,118,210,0.06)",
        height=1050,
        showlegend=True,
        legend=dict(x=0.01, y=0.62,
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#ccc", borderwidth=1,
                    font=dict(size=11)),
        sliders=sliders,
        updatemenus=[dict(
            type="buttons", showactive=False,
            x=0.5, xanchor="center", y=-0.07, yanchor="top",
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(
                         frame=dict(duration=2000, redraw=True),
                         fromcurrent=True,
                         transition=dict(duration=400),
                     )]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], dict(
                         frame=dict(duration=0, redraw=False),
                         mode="immediate",
                         transition=dict(duration=0),
                     )]),
            ],
        )],
        font=dict(family="Inter, Arial, sans-serif"),
        annotations=[dict(
            text=(
                f"Field A: N0/N1 very cheap — N3/N4 very expensive  |  "
                f"Field B: N3/N4 very cheap — N0/N1 very expensive  |  "
                f"Switch prob: {switch_probability:.0%}"
            ),
            x=0.5, y=0.395, xref="paper", yref="paper",
            xanchor="center", showarrow=False,
            font=dict(size=9.5, color="#555"),
        )],
    )

    fig.update_xaxes(title_text="Longitude (°)",
                     gridcolor="rgba(200,200,200,0.3)",
                     zeroline=False, row=1, col=1)
    fig.update_yaxes(title_text="Latitude (°)",
                     gridcolor="rgba(200,200,200,0.3)",
                     zeroline=False, row=1, col=1)
    for col in [1, 2, 3]:
        fig.update_xaxes(visible=False, range=[-0.05, 3.2],
                         row=2, col=col)
        fig.update_yaxes(visible=False, range=[-1.2, 1.1],
                         row=2, col=col)

    return fig, adap, greedy


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    N_SEEDS     = 10
    VIZ_SEED    = 8

    # ── 1. Run aggregate experiment and show static plots ─────
    print(f"Running trap experiments ({N_SEEDS} seeds)...")
    results = run_trap_experiment(
        n_seeds            = N_SEEDS,
        base_seed          = 0,
        switch_probability = SWITCH_PROB,
        field_seed         = FIELD_SEED,
    )
    plot_trap_results(results, switch_probability=SWITCH_PROB)

    # ── 2. Build and save interactive step-by-step viz ────────
    print(f"\nBuilding interactive visualisation (seed={VIZ_SEED})...")
    fig, adap, greedy = build_interactive_viz(
        seed               = VIZ_SEED,
        switch_probability = SWITCH_PROB,
        field_seed         = FIELD_SEED,
    )

    print(f"\nSeed {VIZ_SEED} results:")
    print(f"  Adaptive: {adap['total_cost']:.4f}")
    print(f"  Greedy:   {greedy['total_cost']:.4f}")
    print(f"  Advantage: {greedy['total_cost'] - adap['total_cost']:+.4f}")

    print("\nStep-by-step:")
    for d_a, d_g in zip(adap["decisions"], greedy["decisions"]):
        sw = "⚡SWITCH" if d_a["field_switched"] else "      "
        print(
            f"  S{d_a['stage']} {sw} "
            f"Plan:{['A','B'][d_a['planning_field_idx']]} "
            f"Real:{['A','B'][d_a['realised_field_idx']]}  |  "
            f"Adap→N{d_a['chosen_idx']} "
            f"(imm={d_a['imm_planned']:.3f} "
            f"ctg={d_a['ctg_planned']:.3f} "
            f"paid={d_a['realised_cost']:.3f})  |  "
            f"Greedy→N{d_g['chosen_idx']} "
            f"(imm={d_g['imm_planned']:.3f} "
            f"paid={d_g['realised_cost']:.3f})"
        )

    output = "trap2_viz.html"
    custom_js = f"""
<script>
var _step = 0;
var _total = {NUM_STAGES};
document.addEventListener('keydown', function(e) {{
    if (e.key === 'ArrowRight' && _step < _total) {{
        _step++;
        Plotly.animate('trap-viz', [String(_step)], {{
            transition: {{duration: 0}},
            frame: {{duration: 0, redraw: true}},
            mode: 'immediate'
        }});
    }} else if (e.key === 'ArrowLeft' && _step > 0) {{
        _step--;
        Plotly.animate('trap-viz', [String(_step)], {{
            transition: {{duration: 0}},
            frame: {{duration: 0, redraw: true}},
            mode: 'immediate'
        }});
    }}
}});
</script>"""

    pio.write_html(fig, output, include_plotlyjs=True,
                   full_html=True, div_id="trap-viz")
    with open(output, "a") as f:
        f.write(custom_js)

    print(f"\nInteractive viz saved: {output}")
    print("Controls: slider | ← → arrow keys | Play/Pause button")