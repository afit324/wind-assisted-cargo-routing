"""
interactive_network.py
======================

Interactive Plotly visualisation of the adaptive routing policy.

This file is completely independent of the optimisation.

Inputs:
    results = run_all_policies(seed)

Outputs:
    Interactive Plotly dashboard.

Author:
Thomas Fairweather
"""

import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

from policy_grib import (
    NODE_COORDS,
    NODE_LATS,
    STAGE_LONS,
    layers,
    MODELS,
    NUM_FORECASTS,
    NUM_STAGES,
    STAGE_DATES,
    generate_forecasts,
    future_cost_to_go,
    ACTIVE_COST_FN,
    DISPLAY_LABELS,
)

# ----------------------------------------------------------
# GRAPH LAYOUT
# ----------------------------------------------------------

def build_graph_layout():

    pos = {}

    pos["START"] = (
        STAGE_LONS[0] - 6,
        np.mean(NODE_LATS)
    )

    for node, (lat, lon) in NODE_COORDS.items():
        pos[node] = (lon, lat)

    pos["END"] = (
        STAGE_LONS[-1] + 6,
        np.mean(NODE_LATS)
    )

    return pos


GRAPH_POS = build_graph_layout()

# ----------------------------------------------------------
# NODE ORDER
# ----------------------------------------------------------

GRAPH_NODES = (
    ["START"]
    + [n for layer in layers[1:-1] for n in layer]
    + ["END"]
)

# ----------------------------------------------------------
# EDGE LIST
# ----------------------------------------------------------

def build_edge_list():

    edges = []

    for layer in range(len(layers)-1):

        for u in layers[layer]:

            for v in layers[layer+1]:

                edges.append((u,v))

    return edges


GRAPH_EDGES = build_edge_list()

# ----------------------------------------------------------
# COST TO GO
# ----------------------------------------------------------

def node_diagnostics(
    current_node,
    candidate_node,
    stage,
    forecasts,
):
    """
    Compute everything required for the hover panel.
    """

    immediate = []

    future = []

    total = []

    for forecast in forecasts:

        imm = forecast[current_node][candidate_node]

        fut = future_cost_to_go(
            candidate_node,
            stage + 1,
            forecast,
        )

        immediate.append(imm)
        future.append(fut)
        total.append(imm + fut)

    return {

        "immediate" : immediate,

        "future" : future,

        "total" : total,

        "mean_immediate" : np.mean(immediate),

        "mean_future" : np.mean(future),

        "mean_total" : np.mean(total),

        "best_forecast" : int(np.argmin(total)),

    }

# ----------------------------------------------------------
# HOVER TEXT
# ----------------------------------------------------------

def build_hover_text(
    node,
    stage,
    wind,
    diagnostics=None,
):

    if node == "START":
        return "<b>START</b>"
    if node == "END":
        return "<b>END</b>"

    lat, lon = NODE_COORDS[node]
    label    = DISPLAY_LABELS.get(ACTIVE_COST_FN, "Wind (m/s)")

    text = []
    text.append(f"<b>{node}</b>")
    text.append(f"Latitude: {lat:.1f}")
    text.append(f"Longitude: {lon:.1f}")
    text.append("<br>")
    text.append(f"{label}: {wind:.2f}")   # <-- was hardcoded "Wind: {wind:.2f} m/s"

    if diagnostics is not None:

        text.append("<br>")

        text.append(
            f"Mean Immediate: {diagnostics['mean_immediate']:.4f}"
        )

        text.append(
            f"Mean Future: {diagnostics['mean_future']:.4f}"
        )

        text.append(
            f"<b>Mean Total: {diagnostics['mean_total']:.4f}</b>"
        )

        text.append("<br>")

        text.append("<b>Forecast totals</b>")

        for i in range(NUM_FORECASTS):

            text.append(

                f"{MODELS[i]} : "

                f"{diagnostics['total'][i]:.4f}"

            )

    return "<br>".join(text)

# ----------------------------------------------------------
# LOAD ALL WINDS
# ----------------------------------------------------------

def load_stage_information():

    forecasts = []

    winds = []

    for stage in range(NUM_STAGES):

        f, w = generate_forecasts(
            stage,
            return_winds=True,
        )

        forecasts.append(f)

        winds.append(w)

    return forecasts, winds


# ----------------------------------------------------------
# GLOBAL WIND RANGE
# ----------------------------------------------------------

def compute_global_wind_limits(
    stage_winds,
):

    values = []

    for stage in stage_winds:

        for model in stage:

            values.extend(model.values())

    return (

        min(values),

        max(values),

    )

# ----------------------------------------------------------
# STAGE CACHE
# ----------------------------------------------------------

def build_stage_cache():

    forecasts, winds = load_stage_information()

    cache = []

    for stage in range(NUM_STAGES):

        cache.append(

            {

                "forecast" : forecasts[stage],

                "wind" : winds[stage],

                "date" : STAGE_DATES[stage],

            }

        )

    return cache

# ----------------------------------------------------------
# WIND -> COLOUR
# ----------------------------------------------------------

def wind_colour(
    wind,
    wind_min,
    wind_max,
):

    if wind_max == wind_min:
        t = 0.5
    else:
        t = (wind - wind_min) / (wind_max - wind_min)

    return sample_colorscale(
        "RdYlGn",
        [t]
    )[0]

# ----------------------------------------------------------
# BACKGROUND EDGES
# ----------------------------------------------------------

def build_edge_trace():

    x = []
    y = []

    for u, v in GRAPH_EDGES:

        x0, y0 = GRAPH_POS[u]
        x1, y1 = GRAPH_POS[v]

        x += [x0, x1, None]
        y += [y0, y1, None]

    return go.Scatter(

        x=x,
        y=y,

        mode="lines",

        line=dict(
            color="lightgrey",
            width=1,
        ),

        hoverinfo="skip",

        showlegend=False,

    )

# ----------------------------------------------------------
# POLICY EDGE
# ----------------------------------------------------------

def build_policy_edge_trace(

    from_node,
    to_node,

):

    x0, y0 = GRAPH_POS[from_node]
    x1, y1 = GRAPH_POS[to_node]

    return go.Scatter(

        x=[x0, x1],

        y=[y0, y1],

        mode="lines",

        line=dict(

            color="royalblue",

            width=5,

        ),

        hoverinfo="skip",

        showlegend=False,

    )

# ----------------------------------------------------------
# NODE TRACE
# ----------------------------------------------------------

def build_node_trace(
    stage,
    current_node,
    forecasts,
    winds,
    wind_min,
    wind_max,
    wind_display = "Average",
    realised_model = None,
):

    xs = []
    ys = []

    colours = []

    sizes = []

    hover = []

    border = []

    current_layer = min(stage, NUM_STAGES - 1)

    for node in GRAPH_NODES:

        x, y = GRAPH_POS[node]

        xs.append(x)
        ys.append(y)

        if node == "START":

            colours.append("dodgerblue")
            sizes.append(26)
            border.append(3)

            hover.append("START")

            continue

        if node == "END":

            colours.append("black")
            sizes.append(20)
            border.append(2)

            hover.append("END")

            continue

        if node == current_node:

            sizes.append(24)
            border.append(4)

        else:

            sizes.append(16)
            border.append(1)

        #
        # Wind colouring
        #

        model_mean = np.mean(

            [

                winds[k][node]

                for k in range(NUM_FORECASTS)

            ]

        )

        colours.append(

            wind_colour(

                model_mean,

                wind_min,

                wind_max,

            )

        )

        #
        # Hover text
        #

        if stage < NUM_STAGES:

            if node in layers[stage + 1]:

                diagnostics = node_diagnostics(

                    current_node,

                    node,

                    stage,

                    forecasts,

                )

            else:

                diagnostics = None

        else:

            diagnostics = None

        hover.append(

            build_hover_text(

                node,

                stage,

                model_mean,

                diagnostics,

            )

        )

    return go.Scatter(

        x=xs,

        y=ys,

        mode="markers+text",

        text=GRAPH_NODES,

        textposition="top center",

        hovertext=hover,

        hoverinfo="text",

        marker=dict(

            size=sizes,

            color=colours,

            line=dict(

                color="black",

                width=border,

            ),

        ),

        showlegend=False,

    )

# ----------------------------------------------------------
# BUILD FIGURE
# ----------------------------------------------------------

def create_stage_frame(

    results,

    stage=0,

):

    cache = build_stage_cache()

    wind_min, wind_max = compute_global_wind_limits(

        [

            c["wind"]

            for c in cache

        ]

    )

    adaptive_path = results["adaptive"][1]

    current_node = adaptive_path[stage]

    next_node = adaptive_path[stage + 1]

    fig = go.Figure()

    #
    # Background graph
    #

    fig.add_trace(

        build_edge_trace()

    )

    #
    # Chosen edge
    #

    fig.add_trace(

        build_policy_edge_trace(

            current_node,

            next_node,

        )

    )

    #
    # Nodes
    #

    fig.add_trace(

        build_node_trace(

            stage,

            current_node,

            cache[stage]["forecast"],

            cache[stage]["wind"],

            wind_min,

            wind_max,

        )

    )

    fig.update_layout(

        title=f"Adaptive Policy<br>{cache[stage]['date']}",

        width=1200,

        height=700,

        template="plotly_white",

        hovermode="closest",

        xaxis=dict(

            visible=False,

        ),

        yaxis=dict(

            visible=False,

            scaleanchor="x",

            scaleratio=1,

        ),

        margin=dict(

            l=20,

            r=20,

            t=60,

            b=20,

        ),

    )

    return fig

# ==========================================================
# BUILD ONE PLOTLY FRAME
# ==========================================================

def build_frame(
    results,
    cache,
    stage,
    wind_min,
    wind_max,
):

    adaptive_path = results["adaptive"][1]

    current_node = adaptive_path[stage]
    next_node = adaptive_path[stage + 1]

    edge_trace = build_policy_edge_trace(
        current_node,
        next_node,
    )

    node_trace = build_node_trace(
        stage,
        current_node,
        cache[stage]["forecast"],
        cache[stage]["wind"],
        wind_min,
        wind_max,
    )

    return go.Frame(

        name=str(stage),

        data=[
            build_edge_trace(),
            edge_trace,
            node_trace,
        ],

        layout=go.Layout(

            title=f"Adaptive Policy<br>{cache[stage]['date']}<br>Stage {stage}"

        )

    )

# ==========================================================
# BUILD ALL FRAMES
# ==========================================================

def build_frames(results):

    cache = build_stage_cache()

    wind_min, wind_max = compute_global_wind_limits(
        [c["wind"] for c in cache]
    )

    frames = []

    for stage in range(NUM_STAGES):

        frames.append(

            build_frame(

                results,

                cache,

                stage,

                wind_min,

                wind_max,

            )

        )

    return frames

# ==========================================================
# SLIDER
# ==========================================================

def build_slider():

    steps = []

    for stage in range(NUM_STAGES):

        step = dict(

            method="animate",

            args=[
                [str(stage)],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": True},
                    "transition": {"duration": 0},
                },
            ],

            label=f"{stage}",

        )

        steps.append(step)

    return [

        dict(

            active=0,

            currentvalue=dict(

                prefix="Stage "

            ),

            pad=dict(t=50),

            steps=steps,

        )

    ]

# ==========================================================
# PLAY BUTTONS
# ==========================================================

def build_animation_buttons():

    return [

        dict(

            type="buttons",

            showactive=False,

            x=0.0,

            y=1.12,

            buttons=[

                dict(

                    label="▶ Play",

                    method="animate",

                    args=[

                        None,

                        {

                            "frame": {

                                "duration": 1000,

                                "redraw": True,

                            },

                            "fromcurrent": True,

                        },

                    ],

                ),

                dict(

                    label="❚❚ Pause",

                    method="animate",

                    args=[

                        [None],

                        {

                            "frame": {

                                "duration": 0,

                                "redraw": False,

                            },

                            "mode": "immediate",

                        },

                    ],

                ),

            ],

        )

    ]

# ==========================================================
# DASHBOARD
# ==========================================================

def create_dashboard(results):

    frames = build_frames(results)

    fig = go.Figure(

        data=frames[0].data,

        frames=frames,

    )

    fig.update_layout(
        title=dict(
            text=f"<b>Adaptive Routing Dashboard</b><br>"
                f"<sup>Cost measure: {ACTIVE_COST_FN} — "
                f"{DISPLAY_LABELS.get(ACTIVE_COST_FN, '')}</sup>",
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
            font=dict(size=22),
        ),

        width=1200,

        height=700,

        template="plotly_white",

        hovermode="closest",

        sliders=build_slider(),

        updatemenus=build_animation_buttons(),

        xaxis=dict(

            visible=False,

        ),

        yaxis=dict(

            visible=False,

            scaleanchor="x",

            scaleratio=1,

        ),

        margin=dict(
            l=20,
            r=20,
            t=100,
            b=20,
        )

    )

    return fig



if __name__ == "__main__":

    from experiments_gribs import run_all_policies

    results = run_all_policies(100)

    fig = create_dashboard(results)

    fig.show()

