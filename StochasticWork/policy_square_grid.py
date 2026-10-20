import networkx as nx
import numpy as np
import random
from scipy.fft import fft2, ifft2, fftshift
import matplotlib.pyplot as plt

# ============================================================
# PARAMETERS
# ============================================================

NUM_FORECASTS = 6
NOISE_STD = 0.05          # intensity of field effect
GRID_SIZE = 5
SEED = 42
GRF_LENGTH_SCALE = 2.0    # controls smoothness

random.seed(SEED)
np.random.seed(SEED)

# ============================================================
# CREATE BASE NETWORK
# ============================================================

G_base = nx.grid_2d_graph(GRID_SIZE, GRID_SIZE)

mapping = {node: i for i, node in enumerate(G_base.nodes())}
G_base = nx.relabel_nodes(G_base, mapping)

for u, v in G_base.edges():
    G_base[u][v]["base_cost"] = random.uniform(5, 15)

start = 0
target = GRID_SIZE * GRID_SIZE - 1

# ============================================================
# GRF FIELD GENERATION
# ============================================================

def generate_grf(grid_size, length_scale=2.0):
    """
    Gaussian Random Field via spectral filtering.
    """

    white_noise = np.random.normal(0, 1, (grid_size, grid_size))

    freq = fft2(white_noise)
    freq = fftshift(freq)

    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(x, y)

    dist2 = X**2 + Y**2

    filter_ = np.exp(-dist2 / (2 * (length_scale / grid_size)**2))

    filtered = freq * filter_

    field = np.real(ifft2(fftshift(filtered)))

    field = (field - np.mean(field)) / (np.std(field) + 1e-8)

    return field

# ============================================================
# NODE POSITION HELPERS
# ============================================================

def node_to_xy(node, grid_size):
    return divmod(node, grid_size)

# ============================================================
# FORECAST GENERATION (GRF-BASED)
# ============================================================

def generate_forecast_graphs(base_graph, parent_field=None):

    forecasts = []

    for _ in range(NUM_FORECASTS):

        G = base_graph.copy()

        # --------------------------------------------------------
        # STEP 1: generate spatial GRF field
        # --------------------------------------------------------

        if parent_field is None:
            field = generate_grf(GRID_SIZE, GRF_LENGTH_SCALE)
        else:
            # correlated evolution: perturb previous field
            noise = generate_grf(GRID_SIZE, GRF_LENGTH_SCALE)
            field = 0.7 * parent_field + 0.3 * noise

            field = (field - np.mean(field)) / (np.std(field) + 1e-8)

        # --------------------------------------------------------
        # STEP 2: assign edge costs from field
        # --------------------------------------------------------

        for u, v in G.edges():

            ux, uy = node_to_xy(u, GRID_SIZE)
            vx, vy = node_to_xy(v, GRID_SIZE)

            f_u = field[ux, uy]
            f_v = field[vx, vy]

            field_effect = (f_u + f_v) / 2

            base = base_graph[u][v]["base_cost"]

            cost = base * (1 + NOISE_STD * field_effect)

            G[u][v]["cost"] = max(0.1, cost)

        forecasts.append(G)

    return forecasts

# ============================================================
# POLICY STEP
# ============================================================

def choose_next_node(current, target, forecast_graphs):

    neighbors = list(G_base.neighbors(current))

    best_neighbor = None
    best_total_cost = float("inf")
    best_forecast_index = None

    print(f"\nCurrent node: {current}")
    print(f"Neighbors: {neighbors}")

    for neighbor in neighbors:

        # ----------------------------------------------------
        # immediate cost (averaged across forecasts)
        # ----------------------------------------------------

        immediate_costs = [
            G[current][neighbor]["cost"]
            for G in forecast_graphs
        ]

        avg_immediate = np.mean(immediate_costs)

        # ----------------------------------------------------
        # future shortest path per forecast
        # ----------------------------------------------------

        future_costs = []

        for G in forecast_graphs:

            try:
                d = nx.shortest_path_length(
                    G,
                    source=neighbor,
                    target=target,
                    weight="cost"
                )
            except nx.NetworkXNoPath:
                d = float("inf")

            future_costs.append(d)

        # ----------------------------------------------------
        # scenario evaluation (your policy)
        # ----------------------------------------------------

        scenario_totals = [
            avg_immediate + d for d in future_costs
        ]

        best_total_for_neighbor = min(scenario_totals)
        best_scenario = np.argmin(scenario_totals)

        print(
            f"Neighbor {neighbor}: "
            f"Immediate={avg_immediate:.2f}, "
            f"BestTotal={best_total_for_neighbor:.2f}"
        )

        # ----------------------------------------------------
        # track best move
        # ----------------------------------------------------

        if best_total_for_neighbor < best_total_cost:

            best_total_cost = best_total_for_neighbor
            best_neighbor = neighbor
            best_forecast_index = best_scenario

    return best_neighbor, best_forecast_index

def draw_graph(G):
    pos = {}

    # convert node index -> grid coordinates
    for node in G.nodes():
        pos[node] = (node % GRID_SIZE, node // GRID_SIZE)

    plt.figure(figsize=(6,6))

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=600
    )

    # edge labels = costs
    edge_labels = nx.get_edge_attributes(G, "cost")

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels={k: round(v,2) for k,v in edge_labels.items()},
        font_size=8
    )

    plt.title("Graph with Edge Costs")
    plt.show()

def draw_cost_grid(G):
    grid = np.zeros((GRID_SIZE, GRID_SIZE))

    for u, v, data in G.edges(data=True):

        x1, y1 = divmod(u, GRID_SIZE)
        x2, y2 = divmod(v, GRID_SIZE)

        grid[x1, y1] += data["cost"]
        grid[x2, y2] += data["cost"]

    plt.figure(figsize=(5,5))
    plt.imshow(grid, cmap="viridis", origin="lower")
    plt.colorbar(label="Accumulated edge cost")
    plt.title("Spatial Cost Map")
    plt.show()

def draw_field(field):
    plt.figure(figsize=(5,5))
    plt.imshow(field, cmap="coolwarm", origin="lower")
    plt.colorbar(label="Field intensity")
    plt.title("Gaussian Random Field (Weather)")
    plt.show()

# ============================================================
# MAIN LOOP
# ============================================================

current = start

forecast_graphs = generate_forecast_graphs(G_base)

# we also track the evolving GRF field
current_field = None

path_taken = [current]
step = 0

while current != target:

    print("\n" + "=" * 60)
    print(f"STEP {step}")

    next_node, best_forecast = choose_next_node(
        current,
        target,
        forecast_graphs
    )

    print(f"\nChosen next node: {next_node}")
    print(f"Chosen forecast branch: {best_forecast}")

    current = next_node
    path_taken.append(current)

    # --------------------------------------------------------
    # evolve from selected scenario field
    # --------------------------------------------------------

    parent_graph = forecast_graphs[best_forecast]

    # reconstruct approximate field from chosen forecast
    # (we rebuild a field snapshot from costs)
    field_snapshot = generate_grf(GRID_SIZE, GRF_LENGTH_SCALE)
    #draw_field(field_snapshot)

    forecast_graphs = generate_forecast_graphs(
        G_base,
        parent_field=field_snapshot
    )

    step += 1

    if step > 100:
        print("Stopping: too many steps.")
        break

# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print("Path taken:")
print(path_taken)

draw_graph(forecast_graphs[0])
draw_graph(forecast_graphs[1])
draw_graph(forecast_graphs[2])
draw_cost_grid(forecast_graphs[0])

print(f"Destination reached: {current == target}")