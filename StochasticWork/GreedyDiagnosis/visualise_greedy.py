"""
interactive_policy_viz.py
=========================
Interactive step-by-step visualisation comparing the Adaptive and Greedy
policies on a shared realised world.

Shows:
  - All nodes on a lat/lon map coloured by wind speed
  - Both policy paths drawn in different colours
  - Step-by-step animation with play/pause/next/prev controls
  - Decision panel showing immediate cost, cost-to-go, and why each
    policy chose the node it did at each stage
  - Wind speed tooltip on hover for every node

Run:
    python interactive_policy_viz.py
"""

import os
import sys
import random
import copy
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# ── silence policy_grib prints ───────────────────────────────
stdout_backup = sys.stdout
sys.stdout    = open(os.devnull, "w")

import policy_grib
from policy_grib import (
    layers,
    generate_forecasts,
    choose_next_node,
    future_cost_to_go,
    NODE_COORDS,
    NODE_LATS,
    STAGE_LONS,
    MODELS,
    NUM_FORECASTS,
    NUM_STAGES,
    STAGE_DATES,
    COST_FUNCTIONS,
    ACTIVE_COST_FN,
)

sys.stdout.close()
sys.stdout = stdout_backup

# ============================================================
# CONFIGURATION
# ============================================================

SEED       = 100
VIZ_TITLE  = "Adaptive vs Greedy — Step-by-Step Policy Comparison"

ADAPTIVE_COLOR = "#2196F3"   # blue
GREEDY_COLOR   = "#FF5722"   # deep orange
NODE_COLOR     = "#90CAF9"   # light blue
VISITED_COLOR  = "#FFF176"   # yellow
WIND_CMAP      = "RdYlGn"    # red=low wind, green=high wind

# ============================================================
# LOAD FORECASTS
# ============================================================

print("Loading forecasts for all stages...")
ALL_STAGE_FORECASTS = []
for s in range(NUM_STAGES):
    stdout_backup = sys.stdout
    sys.stdout    = open(os.devnull, "w")
    try:
        fc = generate_forecasts(stage=s)
    finally:
        sys.stdout.close()
        sys.stdout = stdout_backup
    ALL_STAGE_FORECASTS.append(fc)
print("Forecasts loaded.\n")

# ============================================================
# SIMULATE BOTH POLICIES
# ============================================================

def run_simulation(seed):
    """
    Runs adaptive and greedy policies under the same realised world.
    Returns detailed step-by-step records for both policies.
    """

    random.seed(seed)
    np.random.seed(seed)

    # Generate realised sequence (shared between policies)
    realised_sequence = [
        random.randint(0, NUM_FORECASTS - 1)
        for _ in range(NUM_STAGES)
    ]

    def simulate_policy(policy_name):

        random.seed(seed)
        np.random.seed(seed)

        current_node = "START"
        path         = ["START"]
        stage_costs  = []
        decisions    = []   # detailed decision info per stage
        total_cost   = 0.0
        weights      = np.ones(NUM_FORECASTS) / NUM_FORECASTS
        lr           = 0.3

        for stage in range(NUM_STAGES):

            forecasts  = ALL_STAGE_FORECASTS[stage]
            next_layer = layers[stage + 1]

            # ── compute costs for all candidates ─────────────
            candidate_info = []

            for candidate in next_layer:

                imm_costs = [f[current_node][candidate]
                             for f in forecasts]
                avg_imm   = float(np.mean(imm_costs))

                if policy_name == "adaptive":
                    future_costs = [
                        future_cost_to_go(candidate, stage + 1, f)
                        for f in forecasts
                    ]
                    avg_future = float(np.mean(future_costs))
                    total      = avg_imm + avg_future

                elif policy_name == "greedy":
                    avg_future = 0.0
                    total      = avg_imm   # no lookahead

                candidate_info.append({
                    "node"       : candidate,
                    "imm_cost"   : avg_imm,
                    "future_cost": avg_future,
                    "total_cost" : total,
                    "lat"        : NODE_COORDS[candidate][0],
                    "lon"        : NODE_COORDS[candidate][1],
                })

            # ── pick best candidate ───────────────────────────
            best = min(candidate_info, key=lambda x: x["total_cost"])

            # ── pay realised cost ─────────────────────────────
            realised_k    = realised_sequence[stage]
            realised_cost = forecasts[realised_k][current_node][best["node"]]

            stage_costs.append(realised_cost)
            total_cost   += realised_cost

            decisions.append({
                "stage"         : stage,
                "date"          : STAGE_DATES[min(stage, NUM_STAGES - 1)],
                "from_node"     : current_node,
                "chosen_node"   : best["node"],
                "realised_k"    : realised_k,
                "realised_model": MODELS[realised_k]
                                  if realised_k < len(MODELS) else f"F{realised_k}",
                "realised_cost" : realised_cost,
                "imm_cost"      : best["imm_cost"],
                "future_cost"   : best["future_cost"],
                "total_score"   : best["total_cost"],
                "all_candidates": candidate_info,
                "cumulative"    : float(np.sum(stage_costs)),
            })

            # Update weights for weighted forecast (not used here
            # but kept for consistency)
            weights[realised_k] += lr
            weights              = weights / weights.sum()

            current_node = best["node"]
            path.append(current_node)

        return {
            "path"        : path,
            "stage_costs" : stage_costs,
            "total_cost"  : total_cost,
            "decisions"   : decisions,
        }

    adaptive_result = simulate_policy("adaptive")
    greedy_result   = simulate_policy("greedy")

    return adaptive_result, greedy_result, realised_sequence


print(f"Simulating policies (seed={SEED})...")
adaptive_result, greedy_result, realised_sequence = run_simulation(SEED)
print(f"Adaptive total: {adaptive_result['total_cost']:.4f}")
print(f"Greedy total:   {greedy_result['total_cost']:.4f}\n")

# ============================================================
# EXTRACT WIND SPEEDS FOR VISUALISATION
# ============================================================

def get_wind_speeds(stage):
    """
    Returns dict: node_name -> mean wind speed across all forecasts
    for a given stage's forecasts.
    """
    forecasts = ALL_STAGE_FORECASTS[stage]
    wind_dict = {}

    for s in range(NUM_STAGES):
        for i in range(len(NODE_LATS)):
            node = f"S{s}_N{i}"
            # Derive wind speed from arc cost: ws = 1/cost (approximate)
            # Use START->node cost which uses node wind twice
            costs = [forecasts[k]["START"][node]
                     if node in forecasts[k].get("START", {})
                     else forecasts[k].get(f"S{max(0,s-1)}_N0", {}).get(node, 0.1)
                     for k in range(NUM_FORECASTS)]
            mean_cost     = float(np.mean(costs))
            wind_dict[node] = 1.0 / mean_cost if mean_cost > 0 else 0.0

    return wind_dict


# Pre-compute wind speeds for stage 0 (initial view)
wind_speeds_s0 = get_wind_speeds(0)

# ============================================================
# BUILD PLOTLY FIGURE
# ============================================================

def build_figure():

    n_steps = NUM_STAGES + 1   # START + each stage decision

    # ── pre-build all frame data ──────────────────────────────
    frames = []

    for step in range(n_steps):

        # Which stage are we showing decisions for?
        # step 0 = at START, about to make stage 0 decision
        # step s = just made decision s-1, at stage s-1 result node
        current_stage = min(step, NUM_STAGES - 1)
        forecasts     = ALL_STAGE_FORECASTS[current_stage]
        wind_speeds   = get_wind_speeds(current_stage)

        # ── node traces ───────────────────────────────────────
        node_lats  = []
        node_lons  = []
        node_text  = []
        node_winds = []
        node_sizes = []
        node_syms  = []

        adap_path_so_far  = adaptive_result["path"][:step + 1]
        greedy_path_so_far = greedy_result["path"][:step + 1]

        for s in range(NUM_STAGES):
            for i in range(len(NODE_LATS)):
                node       = f"S{s}_N{i}"
                lat, lon   = NODE_COORDS[node]
                ws         = wind_speeds.get(node, 0.0)
                arc_cost   = 1.0 / max(ws, 0.01)

                in_adap    = node in adap_path_so_far
                in_greedy  = node in greedy_path_so_far

                node_lats.append(lat)
                node_lons.append(lon)
                node_winds.append(ws)
                node_sizes.append(18 if (in_adap or in_greedy) else 12)
                node_syms.append(
                    "star"    if (in_adap and in_greedy)
                    else "diamond" if in_adap
                    else "square"  if in_greedy
                    else "circle"
                )

                # Hover text
                imm_costs = []
                if s > 0:
                    prev_layer = layers[s]
                    for prev in layers[s]:
                        if prev in forecasts[0]:
                            imm_c = float(np.mean(
                                [forecasts[k][prev][node]
                                 for k in range(NUM_FORECASTS)
                                 if node in forecasts[k].get(prev, {})]
                            )) if node in forecasts[0].get(prev, {}) else None
                            if imm_c:
                                imm_costs.append(imm_c)

                fut_cost = float(np.mean([
                    future_cost_to_go(node, s + 1, f)
                    for f in forecasts
                ])) if s < NUM_STAGES - 1 else 0.0

                node_text.append(
                    f"<b>{node}</b><br>"
                    f"Lat: {lat:.1f}°  Lon: {lon:.1f}°<br>"
                    f"Wind speed: {ws:.2f} m/s<br>"
                    f"Arc cost (1/wind): {arc_cost:.4f}<br>"
                    f"Future cost-to-go: {fut_cost:.4f}<br>"
                    f"Stage: {s}  ({STAGE_DATES[min(s, NUM_STAGES-1)]})<br>"
                    f"{'★ On ADAPTIVE path' if in_adap else ''}"
                    f"{'  ◆ On GREEDY path' if in_greedy else ''}"
                )

        # ── arc traces ────────────────────────────────────────
        arc_lats = []
        arc_lons = []

        for s in range(NUM_STAGES - 1):
            for i in range(len(NODE_LATS)):
                from_node    = f"S{s}_N{i}"
                lat_f, lon_f = NODE_COORDS[from_node]
                for j in range(len(NODE_LATS)):
                    to_node    = f"S{s+1}_N{j}"
                    lat_t, lon_t = NODE_COORDS[to_node]
                    arc_lats.extend([lat_f, lat_t, None])
                    arc_lons.extend([lon_f, lon_t, None])

        # ── adaptive path trace ───────────────────────────────
        adap_lats = [
            NODE_COORDS[n][0] if n in NODE_COORDS else None
            for n in adap_path_so_far if n != "START"
        ]
        adap_lons = [
            NODE_COORDS[n][1] if n in NODE_COORDS else None
            for n in adap_path_so_far if n != "START"
        ]

        # ── greedy path trace ─────────────────────────────────
        grdy_lats = [
            NODE_COORDS[n][0] if n in NODE_COORDS else None
            for n in greedy_path_so_far if n != "START"
        ]
        grdy_lons = [
            NODE_COORDS[n][1] if n in NODE_COORDS else None
            for n in greedy_path_so_far if n != "START"
        ]

        # ── decision annotation ───────────────────────────────
        if step > 0 and step <= NUM_STAGES:
            adap_dec  = adaptive_result["decisions"][step - 1]
            grdy_dec  = greedy_result["decisions"][step - 1]

            adap_info = (
                f"<b>ADAPTIVE — Stage {step-1} ({adap_dec['date']})</b><br>"
                f"Moved: {adap_dec['from_node']} → {adap_dec['chosen_node']}<br>"
                f"Avg immediate cost:  {adap_dec['imm_cost']:.4f}<br>"
                f"Avg future cost-to-go: {adap_dec['future_cost']:.4f}<br>"
                f"Total score used:    {adap_dec['total_score']:.4f}<br>"
                f"<i>Realised model: {adap_dec['realised_model']} "
                f"(index {adap_dec['realised_k']})</i><br>"
                f"Realised cost paid:  {adap_dec['realised_cost']:.4f}<br>"
                f"Cumulative cost:     {adap_dec['cumulative']:.4f}"
            )

            grdy_info = (
                f"<b>GREEDY — Stage {step-1} ({grdy_dec['date']})</b><br>"
                f"Moved: {grdy_dec['from_node']} → {grdy_dec['chosen_node']}<br>"
                f"Avg immediate cost:  {grdy_dec['imm_cost']:.4f}<br>"
                f"No lookahead (future cost ignored)<br>"
                f"Score used (imm only): {grdy_dec['total_score']:.4f}<br>"
                f"<i>Realised model: {grdy_dec['realised_model']} "
                f"(index {grdy_dec['realised_k']})</i><br>"
                f"Realised cost paid:  {grdy_dec['realised_cost']:.4f}<br>"
                f"Cumulative cost:     {grdy_dec['cumulative']:.4f}"
            )

            # Candidate comparison for this stage
            cand_text = "<b>All candidates this stage:</b><br>"
            all_cands = adap_dec["all_candidates"]
            for c in sorted(all_cands, key=lambda x: x["total_cost"]):
                marker = "✓ " if c["node"] == adap_dec["chosen_node"] else "   "
                cand_text += (
                    f"{marker}{c['node']} ({c['lat']:.0f}°N): "
                    f"imm={c['imm_cost']:.4f} "
                    f"fut={c['future_cost']:.4f} "
                    f"tot={c['total_cost']:.4f}<br>"
                )

        else:
            adap_info  = "<b>ADAPTIVE</b><br>At START — awaiting first decision"
            grdy_info  = "<b>GREEDY</b><br>At START — awaiting first decision"
            cand_text  = "Make the first step to see candidate costs."

        frames.append({
            "step"      : step,
            "node_lats" : node_lats,
            "node_lons" : node_lons,
            "node_text" : node_text,
            "node_winds": node_winds,
            "node_sizes": node_sizes,
            "node_syms" : node_syms,
            "arc_lats"  : arc_lats,
            "arc_lons"  : arc_lons,
            "adap_lats" : adap_lats,
            "adap_lons" : adap_lons,
            "grdy_lats" : grdy_lats,
            "grdy_lons" : grdy_lons,
            "adap_info" : adap_info,
            "grdy_info" : grdy_info,
            "cand_text" : cand_text,
            "stage"     : current_stage,
        })

    # ── build initial figure ──────────────────────────────────
    f0 = frames[0]

    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"colspan": 2, "type": "scatter"}, None],
            [{"type": "scatter"}, {"type": "scatter"}],
        ],
        subplot_titles=[
            "Node Layout — Wind Speed & Policy Paths",
            "Adaptive Policy Decision",
            "Greedy Policy Decision",
        ],
        row_heights=[0.65, 0.35],
        vertical_spacing=0.08,
    )

    # Background arcs
    fig.add_trace(go.Scatter(
        x=f0["arc_lons"], y=f0["arc_lats"],
        mode="lines",
        line=dict(color="rgba(150,150,150,0.15)", width=0.8),
        hoverinfo="skip",
        showlegend=False,
        name="arcs",
    ), row=1, col=1)

    # Nodes coloured by wind speed
    fig.add_trace(go.Scatter(
        x=f0["node_lons"], y=f0["node_lats"],
        mode="markers",
        marker=dict(
            size=f0["node_sizes"],
            color=f0["node_winds"],
            colorscale=WIND_CMAP,
            showscale=True,
            colorbar=dict(
                title="Wind<br>speed<br>(m/s)",
                x=1.02,
                thickness=15,
            ),
            symbol=f0["node_syms"],
            line=dict(color="white", width=1),
        ),
        text=f0["node_text"],
        hovertemplate="%{text}<extra></extra>",
        name="Nodes",
    ), row=1, col=1)

    # Adaptive path
    fig.add_trace(go.Scatter(
        x=f0["adap_lons"], y=f0["adap_lats"],
        mode="lines+markers",
        line=dict(color=ADAPTIVE_COLOR, width=3),
        marker=dict(size=10, color=ADAPTIVE_COLOR,
                    symbol="diamond", line=dict(color="white", width=1.5)),
        name="Adaptive",
    ), row=1, col=1)

    # Greedy path
    fig.add_trace(go.Scatter(
        x=f0["grdy_lons"], y=f0["grdy_lats"],
        mode="lines+markers",
        line=dict(color=GREEDY_COLOR, width=3, dash="dash"),
        marker=dict(size=10, color=GREEDY_COLOR,
                    symbol="square", line=dict(color="white", width=1.5)),
        name="Greedy",
    ), row=1, col=1)

    # ── decision panels (bottom row) ──────────────────────────
    # Adaptive decision text
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode="text",
        text=[f0["adap_info"]],
        textposition="top right",
        textfont=dict(size=11, color="#1a1a2e"),
        showlegend=False,
        hoverinfo="skip",
        name="adap_text",
    ), row=2, col=1)

    # Greedy decision text
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode="text",
        text=[f0["grdy_info"]],
        textposition="top right",
        textfont=dict(size=11, color="#1a1a2e"),
        showlegend=False,
        hoverinfo="skip",
        name="grdy_text",
    ), row=2, col=2)

    # ── plotly animation frames ───────────────────────────────
    plotly_frames = []
    for fd in frames:
        plotly_frames.append(go.Frame(
            data=[
                # arc trace (index 0)
                go.Scatter(
                    x=fd["arc_lons"], y=fd["arc_lats"],
                    mode="lines",
                    line=dict(color="rgba(150,150,150,0.15)", width=0.8),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                # node trace (index 1)
                go.Scatter(
                    x=fd["node_lons"], y=fd["node_lats"],
                    mode="markers",
                    marker=dict(
                        size=fd["node_sizes"],
                        color=fd["node_winds"],
                        colorscale=WIND_CMAP,
                        showscale=True,
                        colorbar=dict(
                            title="Wind<br>speed<br>(m/s)",
                            x=1.02, thickness=15,
                        ),
                        symbol=fd["node_syms"],
                        line=dict(color="white", width=1),
                    ),
                    text=fd["node_text"],
                    hovertemplate="%{text}<extra></extra>",
                ),
                # adaptive path (index 2)
                go.Scatter(
                    x=fd["adap_lons"], y=fd["adap_lats"],
                    mode="lines+markers",
                    line=dict(color=ADAPTIVE_COLOR, width=3),
                    marker=dict(
                        size=10, color=ADAPTIVE_COLOR, symbol="diamond",
                        line=dict(color="white", width=1.5)
                    ),
                ),
                # greedy path (index 3)
                go.Scatter(
                    x=fd["grdy_lons"], y=fd["grdy_lats"],
                    mode="lines+markers",
                    line=dict(color=GREEDY_COLOR, width=3, dash="dash"),
                    marker=dict(
                        size=10, color=GREEDY_COLOR, symbol="square",
                        line=dict(color="white", width=1.5)
                    ),
                ),
                # adaptive text (index 4)
                go.Scatter(
                    x=[0], y=[0], mode="text",
                    text=[fd["adap_info"]],
                    textposition="top right",
                    textfont=dict(size=11, color="#1a1a2e"),
                ),
                # greedy text (index 5)
                go.Scatter(
                    x=[0], y=[0], mode="text",
                    text=[fd["grdy_info"]],
                    textposition="top right",
                    textfont=dict(size=11, color="#1a1a2e"),
                ),
            ],
            traces=[0, 1, 2, 3, 4, 5],
            name=str(fd["step"]),
            layout=go.Layout(
                title=dict(
                    text=(
                        f"{VIZ_TITLE}<br>"
                        f"<sup>Step {fd['step']} / {NUM_STAGES}  |  "
                        f"Stage {fd['stage']} ({STAGE_DATES[fd['stage']]})  |  "
                        f"Seed {SEED}  |  "
                        f"Adaptive total: {adaptive_result['total_cost']:.4f}  |  "
                        f"Greedy total: {greedy_result['total_cost']:.4f}"
                        f"</sup>"
                    )
                )
            )
        ))

    fig.frames = plotly_frames

    # ── slider ────────────────────────────────────────────────
    sliders = [dict(
        active=0,
        currentvalue=dict(
            prefix="Step: ",
            visible=True,
            xanchor="center",
        ),
        pad=dict(t=50, b=10),
        steps=[
            dict(
                method="animate",
                args=[
                    [str(fd["step"])],
                    dict(
                        frame=dict(duration=0, redraw=True),
                        mode="immediate",
                        transition=dict(duration=0),
                    )
                ],
                label=str(fd["step"]),
            )
            for fd in frames
        ],
    )]

    # ── layout ────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=(
                f"{VIZ_TITLE}<br>"
                f"<sup>Step 0 / {NUM_STAGES}  |  Seed {SEED}  |  "
                f"Adaptive total: {adaptive_result['total_cost']:.4f}  |  "
                f"Greedy total: {greedy_result['total_cost']:.4f}</sup>"
            ),
            x=0.5, xanchor="center",
            font=dict(size=16, color="#1a1a2e"),
        ),
        paper_bgcolor="#f8f9fa",
        plot_bgcolor="#eef2f7",
        height=900,
        showlegend=True,
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#cccccc",
            borderwidth=1,
            font=dict(size=11),
        ),
        sliders=sliders,
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.5, xanchor="center",
                y=-0.08, yanchor="top",
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=1500, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=300),
                            )
                        ],
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                                transition=dict(duration=0),
                            )
                        ],
                    ),
                ],
            )
        ],
        font=dict(family="Inter, Arial, sans-serif"),
    )

    # ── axis styling ──────────────────────────────────────────
    fig.update_xaxes(
        title_text="Longitude (°)",
        gridcolor="rgba(200,200,200,0.4)",
        zeroline=False,
        row=1, col=1,
    )
    fig.update_yaxes(
        title_text="Latitude (°)",
        gridcolor="rgba(200,200,200,0.4)",
        zeroline=False,
        row=1, col=1,
    )
    fig.update_xaxes(visible=False, row=2, col=1)
    fig.update_yaxes(visible=False, row=2, col=1)
    fig.update_xaxes(visible=False, row=2, col=2)
    fig.update_yaxes(visible=False, row=2, col=2)

    # Set axis ranges for decision panels
    fig.update_xaxes(range=[-0.1, 1.5], row=2, col=1)
    fig.update_yaxes(range=[-0.5, 1.0], row=2, col=1)
    fig.update_xaxes(range=[-0.1, 1.5], row=2, col=2)
    fig.update_yaxes(range=[-0.5, 1.0], row=2, col=2)

    # Colour the decision panel backgrounds
    fig.update_layout(
        annotations=[
            dict(
                text="◆ Blue = Adaptive path  |  ■ Orange = Greedy path  |  "
                     "★ = Both policies  |  Node colour = wind speed",
                x=0.5, y=1.01,
                xref="paper", yref="paper",
                xanchor="center", yanchor="bottom",
                showarrow=False,
                font=dict(size=11, color="#555"),
            )
        ]
    )

    return fig


# ============================================================
# BUILD AND SAVE
# ============================================================

print("Building interactive figure...")
fig = build_figure()

output_path = "policy_comparison_viz.html"
pio.write_html(fig, output_path, include_plotlyjs=True, full_html=True)
print(f"\nSaved to: {output_path}")
print("Open in your browser to interact.")