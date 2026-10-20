import numpy as np
import random
import matplotlib.pyplot as plt

# ============================================================
# PARAMETERS
# ============================================================

NUM_FORECASTS = 6
NUM_STAGES = 6               # Number of forecast updates
NODES_PER_STAGE = 5            # Candidate nodes each stage

BASE_COST_MIN = 5
BASE_COST_MAX = 15

NOISE_STD = 0.08
FORECAST_CORRELATION = 0.7

SEED = 43

random.seed(SEED)
np.random.seed(SEED)

# ============================================================
# CREATE LAYERED DECISION STRUCTURE
# ============================================================

"""
Structure:

START
 ↓
Stage 1: 5 nodes
 ↓
Stage 2: 5 nodes
 ↓
...
 ↓
END

No geometry.
No Euclidean distance.
Only stochastic transition costs.
"""

layers = []

# Start node
layers.append(["START"])

# Intermediate stages
for s in range(NUM_STAGES):
    stage_nodes = [f"S{s}_N{i}" for i in range(NODES_PER_STAGE)]
    layers.append(stage_nodes)

# End node
layers.append(["END"])

# ============================================================
# BASE TRANSITION COSTS
# ============================================================

"""
base_costs[from_node][to_node]
"""

base_costs = {}

for stage in range(len(layers) - 1):

    current_layer = layers[stage]
    next_layer = layers[stage + 1]

    for i in current_layer:

        if i not in base_costs:
            base_costs[i] = {}

        for j in next_layer:

            base_cost = random.uniform(
                BASE_COST_MIN,
                BASE_COST_MAX
            )

            base_costs[i][j] = base_cost

# ============================================================
# FORECAST GENERATION
# ============================================================

"""
Each forecast is:

forecast_costs[forecast][from_node][to_node]

These are evolved over time from the chosen scenario.
"""

def generate_forecasts(parent_forecasts=None, noise_std = NOISE_STD, correlation = FORECAST_CORRELATION):

    forecasts = []

    for k in range(NUM_FORECASTS):

        scenario = {}

        for i in base_costs:

            scenario[i] = {}

            for j in base_costs[i]:

                base = base_costs[i][j]

                # ------------------------------------------------
                # Initial forecast generation
                # ------------------------------------------------

                if parent_forecasts is None:

                    noise = np.random.normal(0, noise_std)

                    cost = base * (1 + noise)

                # ------------------------------------------------
                # Correlated evolution
                # ------------------------------------------------

                else:

                    parent_cost = parent_forecasts[k][i][j]

                    noise = np.random.normal(0, noise_std)

                    cost = (
                        correlation * parent_cost
                        +
                        (1 - correlation)
                        * base * (1 + noise)
                    )

                scenario[i][j] = max(0.1, cost)

        forecasts.append(scenario)

    return forecasts

# ============================================================
# COST-TO-GO FUNCTION
# ============================================================

"""
Recursive future value approximation.

This replaces Dijkstra.

Computes:

minimum remaining cost from current node
to END under a specific forecast.
"""

def future_cost_to_go(
    current_node,
    current_stage,
    forecast
):

    # --------------------------------------------------------
    # Reached END
    # --------------------------------------------------------

    if current_stage == len(layers) - 1:
        return 0

    next_layer = layers[current_stage + 1]

    costs = []

    for next_node in next_layer:

        immediate = forecast[current_node][next_node]

        future = future_cost_to_go(
            next_node,
            current_stage + 1,
            forecast
        )

        total = immediate + future

        costs.append(total)

    return min(costs)

# ============================================================
# POLICY STEP
# ============================================================

def choose_next_node(
    current_node,
    current_stage,
    forecasts
):

    next_layer = layers[current_stage + 1]

    best_node = None
    best_total_cost = float("inf")
    best_forecast_index = None

    print("\n" + "-" * 50)
    print(f"Current node: {current_node}")
    print(f"Stage: {current_stage}")

    # --------------------------------------------------------
    # Evaluate each candidate node
    # --------------------------------------------------------

    for candidate in next_layer:

        # ----------------------------------------------------
        # Immediate cost across forecasts
        # ----------------------------------------------------

        immediate_costs = []

        for forecast in forecasts:

            c = forecast[current_node][candidate]

            immediate_costs.append(c)

        avg_immediate = np.mean(immediate_costs)

        # ----------------------------------------------------
        # Future cost under each forecast
        # ----------------------------------------------------

        future_costs = []

        for forecast in forecasts:

            future = future_cost_to_go(
                candidate,
                current_stage + 1,
                forecast
            )

            future_costs.append(future)

        # ----------------------------------------------------
        # Scenario totals
        # ----------------------------------------------------

        scenario_totals = []

        for future in future_costs:

            total = avg_immediate + future

            scenario_totals.append(total)

        # ----------------------------------------------------
        # Best scenario for this candidate
        # ----------------------------------------------------

        # Use mean for selection criterion (the fix your supervisor suggested)
        best_candidate_total = np.mean(scenario_totals)

        # Keep argmin as an integer index for the forecast
        best_scenario = int(np.argmin(scenario_totals))

        print(
            f"Candidate {candidate}: "
            f"Immediate={avg_immediate:.2f}, "
            f"BestTotal={best_candidate_total:.2f}"
        )

        # ----------------------------------------------------
        # Global best move
        # ----------------------------------------------------

        if best_candidate_total < best_total_cost:

            best_total_cost = best_candidate_total

            best_node = candidate

            best_forecast_index = best_scenario

    return (
        best_node,
        best_forecast_index,
        best_total_cost
    )

# ============================================================
# VISUALIZATION
# ============================================================

def visualize_layers(path_taken=None):

    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    fig.patch.set_facecolor('white')

    x_spacing = 2
    y_spacing = 1.5
    positions = {}

    for stage_idx, layer in enumerate(layers):
        x = stage_idx * x_spacing
        y_start = -(len(layer) - 1) * y_spacing / 2
        for i, node in enumerate(layer):
            positions[node] = (x, y_start + i * y_spacing)

    # Background edges
    for stage in range(len(layers) - 1):
        for i in layers[stage]:
            for j in layers[stage + 1]:
                x1, y1 = positions[i]
                x2, y2 = positions[j]
                ax.plot([x1, x2], [y1, y2], color='#cccccc', alpha=0.4,
                        linewidth=1.0, zorder=1)

    # Chosen path
    if path_taken is not None:
        for k in range(len(path_taken) - 1):
            i, j = path_taken[k], path_taken[k + 1]
            x1, y1 = positions[i]
            x2, y2 = positions[j]
            ax.plot([x1, x2], [y1, y2], linewidth=3.5, color='#2e7d32', zorder=2)

    # Node colours
    node_colors = {
        'default': '#aecde8',
        'START'  : '#81c784',
        'END'    : '#e57373',
        'path'   : '#fff176',
    }

    for node, (x, y) in positions.items():
        if node == 'START':
            color = node_colors['START']
        elif node == 'END':
            color = node_colors['END']
        elif path_taken and node in path_taken:
            color = node_colors['path']
        else:
            color = node_colors['default']

        ax.scatter(x, y, s=900, color=color, edgecolors='#333333',
                   linewidths=0.8, zorder=3)
        ax.text(x, y, node, ha='center', va='center',
                fontsize=8.5, fontweight='bold', color='#1a1a1a', zorder=4)

    ax.set_title("Layered Forecast Decision Process",
                 fontsize=17, fontweight='bold', pad=14, color='#1a1a1a')
    ax.axis('off')
    fig.tight_layout()
    plt.savefig('network_viz.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

# ============================================================
# MAIN SIMULATION LOOP
# ============================================================

current_node = "START"

current_stage = 0

path_taken = [current_node]

total_realized_cost = 0

# Store forecast information
chosen_forecasts = []
realized_forecasts = []

# Initial forecasts
forecasts = generate_forecasts()

print("\n" + "=" * 60)
print("STARTING SIMULATION")
print("=" * 60)

while current_node != "END":

    # --------------------------------------------------------
    # Policy decision
    # --------------------------------------------------------

    (
        next_node,
        chosen_forecast,
        estimated_total
    ) = choose_next_node(
        current_node,
        current_stage,
        forecasts
    )

    print(f"\nChosen node: {next_node}")
    print(f"Chosen forecast: {chosen_forecast}")

    # --------------------------------------------------------
    # Realized cost
    # --------------------------------------------------------

    realized_cost = forecasts[
        chosen_forecast
    ][current_node][next_node]

    total_realized_cost += realized_cost

    print(f"Realized transition cost: {realized_cost:.2f}")

    # --------------------------------------------------------
    # Move forward
    # --------------------------------------------------------

    current_node = next_node

    path_taken.append(current_node)

    current_stage += 1

    # --------------------------------------------------------
    # Final transition to END
    # --------------------------------------------------------

    if current_stage == len(layers) - 1:
        break

    # --------------------------------------------------------
    # REALIZED WEATHER SCENARIO
    # --------------------------------------------------------

    realized_forecast = random.randint(
        0,
        NUM_FORECASTS - 1
    )

    realized_forecasts.append(realized_forecast)

    print(
        f"Realized weather forecast: "
        f"{realized_forecast}"
    )

    # --------------------------------------------------------
    # Evolve forecasts from realized world
    # --------------------------------------------------------

    parent_forecasts = []

    for k in range(NUM_FORECASTS):

        parent_forecasts.append(
            forecasts[realized_forecast]
        )

    forecasts = generate_forecasts(
        parent_forecasts=parent_forecasts
    )


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print("\nPath taken:")

for p in path_taken:
    print(p)

print(f"\nTotal realized cost: {total_realized_cost:.2f}")

print("\nRealized forecasts:")
print(realized_forecasts)

# ============================================================
# VISUALIZE
# ============================================================

visualize_layers(path_taken)