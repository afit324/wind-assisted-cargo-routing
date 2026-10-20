"""
trap_viz.py
===========
Interactive step-by-step visualisation of Trap 2 (switching wind fields).
Shows exactly what each policy sees, what it decides, and what it pays.

Run:
    python trap_viz.py
"""

import os
import sys
import random
import copy
import numpy as np
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
)

sys.stdout.close()
sys.stdout = stdout_backup

# ============================================================
# PARAMETERS — tweak these to make greedy suffer more
# ============================================================

SEED             = 42
FIELD_SEED       = 7
SWITCH_PROB      = 0.55    # slightly above 50% so fields do switch often

# Field contrast parameters — higher contrast = greedy suffers more
# Field A strongly favours LOW node indices (high-lat, roaring forties)
# Field B strongly favours HIGH node indices (low-lat, trade winds)
# The bigger the contrast, the more greedy gets punished for committing
FIELD_A_LOW_COST  = 0.05   # very cheap for low-index nodes in field A
FIELD_A_HIGH_COST = 2.50   # very expensive for high-index nodes in field A
FIELD_B_LOW_COST  = 2.50   # very expensive for low-index nodes in field B
FIELD_B_HIGH_COST = 0.05   # very cheap for high-index nodes in field B

ADAPTIVE_COLOR = "#1565C0"   # deep blue
GREEDY_COLOR   = "#BF360C"   # deep red/orange

# ============================================================
# BUILD STRONGLY CONTRASTING WIND FIELDS
# ============================================================

def build_fields():
    """
    Field A: low node index = very cheap, high index = very expensive.
             Optimal path goes through N0/N1 (high latitude).
    Field B: high node index = very cheap, low index = very expensive.
             Optimal path goes through N3/N4 (low latitude).

    The contrast is extreme by design — greedy, which commits hard
    to one direction, will face near-maximum costs every time the
    field switches. Adaptive, which balances cost-to-go across the
    full path, will tend toward middle nodes (N2) that are acceptable
    under either field.
    """

    rng = np.random.RandomState(FIELD_SEED)

    field_A = {}
    field_B = {}

    for i, layer in enumerate(layers[:-1]):
        for from_node in layer:
            field_A[from_node] = {}
            field_B[from_node] = {}

            for to_node in layers[i + 1]:

                if to_node == "END":
                    # Both fields neutral at END
                    field_A[from_node][to_node] = 0.3
                    field_B[from_node][to_node] = 0.3
                    continue

                to_idx = int(to_node.split("_N")[1])
                n      = NODES_PER_STAGE - 1   # max index

                # Smooth interpolation between extremes
                # t=0 (low index) -> t=1 (high index)
                t = to_idx / max(n, 1)

                # Add a little noise so not perfectly deterministic
                noise = rng.uniform(-0.02, 0.02)

                cost_A = FIELD_A_LOW_COST  + t * (FIELD_A_HIGH_COST - FIELD_A_LOW_COST)  + noise
                cost_B = FIELD_B_HIGH_COST + t * (FIELD_B_LOW_COST  - FIELD_B_HIGH_COST) + noise

                field_A[from_node][to_node] = max(cost_A, 0.03)
                field_B[from_node][to_node] = max(cost_B, 0.03)

    return field_A, field_B


FIELD_A, FIELD_B = build_fields()
FIELDS           = [FIELD_A, FIELD_B]
FIELD_NAMES      = ["A (favours N0/N1 — high lat)", "B (favours N3/N4 — low lat)"]
FIELD_COLORS     = ["#1976D2", "#E64A19"]   # blue for A, orange for B


# ============================================================
# COST-TO-GO UNDER A SINGLE FIELD
# ============================================================

def _ctg(node, stage, field):
    if stage == len(layers) - 1:
        return 0.0
    return min(
        field[node][n] + _ctg(n, stage + 1, field)
        for n in layers[stage + 1]
    )


# ============================================================
# SIMULATE BOTH POLICIES — COLLECT STEP-BY-STEP DATA
# ============================================================

def simulate(seed):

    random.seed(seed)
    np.random.seed(seed)

    # Pre-draw the field sequence so both policies face the same world
    field_sequence = []
    for _ in range(NUM_STAGES):
        # Field starts at A; each stage it may switch
        if not field_sequence:
            field_sequence.append(0)   # start on field A
        else:
            prev = field_sequence[-1]
            if random.random() < SWITCH_PROB:
                field_sequence.append(1 - prev)
            else:
                field_sequence.append(prev)

    def run_policy(policy_name):

        current_node   = "START"
        stage          = 0
        total_cost     = 0.0
        path           = ["START"]
        stage_costs    = []
        decisions      = []

        # Policy starts knowing it is on field A
        # It will update to the realised field after each step
        current_field_idx = 0

        while current_node != "END":

            planning_field = FIELDS[current_field_idx]

            # ── collect candidate scores ──────────────────────
            next_layer  = layers[stage + 1]
            cand_info   = []

            for candidate in next_layer:

                imm  = planning_field[current_node][candidate]
                ctg  = _ctg(candidate, stage + 1, planning_field)

                if policy_name == "adaptive":
                    score = imm + ctg
                else:   # greedy — no lookahead
                    score = imm
                    ctg   = 0.0   # doesn't compute this

                cand_info.append({
                    "node"       : candidate,
                    "node_idx"   : int(candidate.split("_N")[1])
                                   if candidate != "END" else -1,
                    "imm_cost"   : imm,
                    "ctg"        : ctg,
                    "score"      : score,
                    "lat"        : NODE_COORDS.get(candidate, (None, None))[0],
                    "lon"        : NODE_COORDS.get(candidate, (None, None))[1],
                })

            best_cand = min(cand_info, key=lambda x: x["score"])
            next_node = best_cand["node"]

            # ── realised field drawn AFTER commitment ─────────
            realised_field_idx = field_sequence[stage]
            realised_field     = FIELDS[realised_field_idx]

            realised_cost = realised_field[current_node][next_node]
            total_cost   += realised_cost
            stage_costs.append(realised_cost)

            decisions.append({
                "stage"              : stage,
                "from_node"          : current_node,
                "chosen_node"        : next_node,
                "chosen_idx"         : best_cand["node_idx"],
                "planning_field_idx" : current_field_idx,
                "planning_field_name": FIELD_NAMES[current_field_idx],
                "realised_field_idx" : realised_field_idx,
                "realised_field_name": FIELD_NAMES[realised_field_idx],
                "field_switched"     : realised_field_idx != current_field_idx,
                "imm_cost_planned"   : best_cand["imm_cost"],
                "ctg_planned"        : best_cand["ctg"],
                "score_used"         : best_cand["score"],
                "realised_cost"      : realised_cost,
                "surprise"           : realised_cost - best_cand["imm_cost"],
                "cumulative"         : float(total_cost),
                "all_candidates"     : cand_info,
            })

            # Realised field becomes planning field next stage
            current_field_idx = realised_field_idx
            current_node      = next_node
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

    adap   = run_policy("adaptive")
    greedy = run_policy("greedy")

    return adap, greedy, field_sequence


# ============================================================
# BUILD VISUALISATION
# ============================================================

def build_figure(seed=SEED):

    adap, greedy, field_sequence = simulate(seed)

    n_steps = NUM_STAGES + 1

    # ── pre-build node positions ──────────────────────────────
    all_nodes     = []
    all_positions = []

    for s in range(NUM_STAGES):
        for i in range(NODES_PER_STAGE):
            node = f"S{s}_N{i}"
            all_nodes.append(node)
            all_positions.append((NODE_COORDS[node][0], NODE_COORDS[node][1]))

    # ── build cost lookup for both fields ─────────────────────
    def field_cost_at_node(field_idx, node):
        """Proxy wind speed: average cost of arcs INTO this node."""
        f    = FIELDS[field_idx]
        costs = []
        s    = int(node.split("_S")[0].replace("S", "")) if "S" in node else 0
        s    = int(node.split("_N")[0].replace("S", ""))
        if s == 0:
            parent_layer = ["START"]
        else:
            parent_layer = [f"S{s-1}_N{i}" for i in range(NODES_PER_STAGE)]
        for parent in parent_layer:
            if parent in f and node in f[parent]:
                costs.append(f[parent][node])
        return float(np.mean(costs)) if costs else 1.0

    # ── frames ───────────────────────────────────────────────
    frames_data = []

    for step in range(n_steps):

        # Which field is the policy currently planning under?
        if step == 0:
            current_field_idx = 0
        elif step <= len(adap["decisions"]):
            current_field_idx = adap["decisions"][step - 1]["realised_field_idx"]
        else:
            current_field_idx = adap["decisions"][-1]["realised_field_idx"]

        current_field = FIELDS[current_field_idx]

        adap_path_so_far   = adap["decisions"][:step] if step > 0 else []
        greedy_path_so_far = greedy["decisions"][:step] if step > 0 else []

        adap_visited   = set(adap["path"][:step + 1])
        greedy_visited = set(greedy["path"][:step + 1])

        # ── node colours = cost under current field ───────────
        node_lats  = []
        node_lons  = []
        node_costs = []
        node_sizes = []
        node_text  = []
        node_syms  = []

        for node in all_nodes:
            lat, lon = NODE_COORDS[node]
            s        = int(node.split("_N")[0].replace("S", ""))
            i        = int(node.split("_N")[1])

            # Cost under current planning field
            cost_A = field_cost_at_node(0, node)
            cost_B = field_cost_at_node(1, node)
            cost_curr = field_cost_at_node(current_field_idx, node)

            in_adap   = node in adap_visited
            in_greedy = node in greedy_visited
            is_current_adap   = (step > 0 and step <= NUM_STAGES and
                                  node == adap["path"][step])
            is_current_greedy = (step > 0 and step <= NUM_STAGES and
                                  node == greedy["path"][step])

            node_lats.append(lat)
            node_lons.append(lon)
            node_costs.append(cost_curr)
            node_sizes.append(
                22 if (is_current_adap or is_current_greedy)
                else 16 if (in_adap or in_greedy)
                else 10
            )
            node_syms.append(
                "star"    if (is_current_adap and is_current_greedy)
                else "diamond" if is_current_adap
                else "square"  if is_current_greedy
                else "circle"
            )

            # Cost-to-go under both fields
            ctg_a = _ctg(node, s + 1, FIELD_A)
            ctg_b = _ctg(node, s + 1, FIELD_B)

            node_text.append(
                f"<b>{node}</b><br>"
                f"Lat: {lat:.1f}°  Lon: {lon:.1f}°<br>"
                f"Stage: {s}<br>"
                f"─────────────────<br>"
                f"Cost under Field A: {cost_A:.3f}<br>"
                f"Cost under Field B: {cost_B:.3f}<br>"
                f"Current field ({['A','B'][current_field_idx]}): {cost_curr:.3f}<br>"
                f"─────────────────<br>"
                f"CTG under Field A: {ctg_a:.3f}<br>"
                f"CTG under Field B: {ctg_b:.3f}<br>"
                f"─────────────────<br>"
                + ("★ ADAPTIVE currently here<br>" if is_current_adap else "")
                + ("■ GREEDY currently here<br>" if is_current_greedy else "")
                + ("◆ On adaptive path<br>" if in_adap and not is_current_adap else "")
                + ("□ On greedy path<br>" if in_greedy and not is_current_greedy else "")
            )

        # ── arc traces ────────────────────────────────────────
        arc_lats_A = []
        arc_lons_A = []
        arc_lats_B = []
        arc_lons_B = []

        for s in range(NUM_STAGES - 1):
            for i in range(NODES_PER_STAGE):
                fn      = f"S{s}_N{i}"
                lat_f, lon_f = NODE_COORDS[fn]
                for j in range(NODES_PER_STAGE):
                    tn      = f"S{s+1}_N{j}"
                    lat_t, lon_t = NODE_COORDS[tn]
                    arc_lats_A.extend([lat_f, lat_t, None])
                    arc_lons_A.extend([lon_f, lon_t, None])
                    arc_lats_B.extend([lat_f, lat_t, None])
                    arc_lons_B.extend([lon_f, lon_t, None])

        # ── policy path traces ────────────────────────────────
        def path_coords(decisions, n):
            lats = []
            lons = []
            nodes_so_far = (["START"] +
                            [d["chosen_node"] for d in decisions[:n]])
            for nd in nodes_so_far:
                if nd in ("START", "END"):
                    continue
                lat, lon = NODE_COORDS.get(nd, (None, None))
                if lat is not None:
                    lats.append(lat)
                    lons.append(lon)
            return lats, lons

        adap_lats, adap_lons     = path_coords(adap["decisions"],   step)
        greedy_lats, greedy_lons = path_coords(greedy["decisions"], step)

        # ── decision info text ────────────────────────────────
        if step > 0 and step <= NUM_STAGES:
            d_a = adap["decisions"][step - 1]
            d_g = greedy["decisions"][step - 1]

            switched_str = (
                "⚡ FIELD SWITCHED this step!"
                if d_a["field_switched"]
                else "Field unchanged"
            )

            adap_text = (
                f"<b>ADAPTIVE — Stage {step - 1}</b><br>"
                f"Planning under: Field {['A','B'][d_a['planning_field_idx']]}<br>"
                f"─────────────────────<br>"
                f"Chose: {d_a['chosen_node']} "
                f"(N{d_a['chosen_idx']})<br>"
                f"<br>"
                f"<b>What it considered:</b><br>"
                f"  Immediate cost (planned): {d_a['imm_cost_planned']:.4f}<br>"
                f"  Cost-to-go (planned):     {d_a['ctg_planned']:.4f}<br>"
                f"  Total score:              {d_a['score_used']:.4f}<br>"
                f"<br>"
                f"<b>What actually happened:</b><br>"
                f"  {switched_str}<br>"
                f"  Realised field: {['A','B'][d_a['realised_field_idx']]}<br>"
                f"  Realised cost:  {d_a['realised_cost']:.4f}<br>"
                f"  Surprise:       {d_a['surprise']:+.4f}<br>"
                f"<br>"
                f"<b>Running total: {d_a['cumulative']:.4f}</b>"
            )

            # Show all candidates for greedy to make trap visible
            cand_rows = ""
            for c in sorted(d_g["all_candidates"],
                             key=lambda x: x["score"]):
                marker = "→ " if c["node"] == d_g["chosen_node"] else "   "
                cand_rows += (
                    f"{marker}N{c['node_idx']}: "
                    f"imm={c['imm_cost']:.3f} "
                    f"(no CTG)<br>"
                )

            greedy_text = (
                f"<b>GREEDY — Stage {step - 1}</b><br>"
                f"Planning under: Field {['A','B'][d_g['planning_field_idx']]}<br>"
                f"─────────────────────<br>"
                f"Chose: {d_g['chosen_node']} "
                f"(N{d_g['chosen_idx']})<br>"
                f"<br>"
                f"<b>All candidates (immediate cost only):</b><br>"
                + cand_rows +
                f"<br>"
                f"<b>What actually happened:</b><br>"
                f"  {switched_str}<br>"
                f"  Realised field: {['A','B'][d_g['realised_field_idx']]}<br>"
                f"  Realised cost:  {d_g['realised_cost']:.4f}<br>"
                f"  Surprise:       {d_g['surprise']:+.4f}<br>"
                f"<br>"
                f"<b>Running total: {d_g['cumulative']:.4f}</b>"
            )

            field_text = (
                f"<b>Field Status</b><br>"
                f"{'─'*20}<br>"
                f"Stage {step - 1} draw:<br>"
                f"  Policy planned: "
                f"{'A' if d_a['planning_field_idx']==0 else 'B'}<br>"
                f"  Realised:       "
                f"{'A' if d_a['realised_field_idx']==0 else 'B'}<br>"
                f"  {'⚡ SWITCHED' if d_a['field_switched'] else 'Same field'}<br>"
                f"<br>"
                f"Full field sequence:<br>"
                + "".join(
                    f"  S{t}: Field {'A' if f == 0 else 'B'}<br>"
                    for t, f in enumerate(field_sequence)
                )
                + f"<br>"
                f"Adaptive total: {adap['total_cost']:.4f}<br>"
                f"Greedy  total:  {greedy['total_cost']:.4f}<br>"
                f"Advantage: {greedy['total_cost'] - adap['total_cost']:+.4f}"
            )

        else:
            adap_text   = ("<b>ADAPTIVE</b><br>At START<br>"
                           "Planning under Field A<br>"
                           "Full lookahead across all stages")
            greedy_text = ("<b>GREEDY</b><br>At START<br>"
                           "Planning under Field A<br>"
                           "No lookahead — immediate cost only")
            field_text  = (
                f"<b>Field Status</b><br>"
                f"Starting on Field A<br>"
                f"Switch probability: {SWITCH_PROB:.0%}<br><br>"
                f"Field A: N0/N1 cheapest<br>"
                f"Field B: N3/N4 cheapest<br><br>"
                f"Greedy commits to one extreme<br>"
                f"Adaptive stays near centre (N2)"
            )

        frames_data.append({
            "step"         : step,
            "node_lats"    : node_lats,
            "node_lons"    : node_lons,
            "node_costs"   : node_costs,
            "node_sizes"   : node_sizes,
            "node_syms"    : node_syms,
            "node_text"    : node_text,
            "arc_lats_A"   : arc_lats_A,
            "arc_lons_A"   : arc_lons_A,
            "arc_lats_B"   : arc_lats_B,
            "arc_lons_B"   : arc_lons_B,
            "adap_lats"    : adap_lats,
            "adap_lons"    : adap_lons,
            "greedy_lats"  : greedy_lats,
            "greedy_lons"  : greedy_lons,
            "adap_text"    : adap_text,
            "greedy_text"  : greedy_text,
            "field_text"   : field_text,
            "field_idx"    : current_field_idx,
        })

    # ── build figure ──────────────────────────────────────────
    f0  = frames_data[0]

    fig = make_subplots(
        rows=2, cols=3,
        specs=[
            [{"colspan": 3, "type": "scatter"}, None, None],
            [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
        ],
        subplot_titles=[
            "Node Layout — Cost under Current Field",
            "Adaptive Decision Detail",
            "Greedy Decision Detail",
            "Field Status",
        ],
        row_heights=[0.60, 0.40],
        vertical_spacing=0.10,
    )

    # Background arcs
    fig.add_trace(go.Scatter(
        x=f0["arc_lons_A"], y=f0["arc_lats_A"],
        mode="lines",
        line=dict(color="rgba(180,180,180,0.12)", width=0.6),
        hoverinfo="skip", showlegend=False, name="_arcs",
    ), row=1, col=1)

    # Nodes
    fig.add_trace(go.Scatter(
        x=f0["node_lons"], y=f0["node_lats"],
        mode="markers",
        marker=dict(
            size=f0["node_sizes"],
            color=f0["node_costs"],
            colorscale="RdYlGn_r",
            showscale=True,
            colorbar=dict(title="Arc cost<br>(current field)",
                          x=1.01, thickness=14, len=0.55),
            symbol=f0["node_syms"],
            line=dict(color="white", width=1.2),
        ),
        text=f0["node_text"],
        hovertemplate="%{text}<extra></extra>",
        name="Nodes",
    ), row=1, col=1)

    # Adaptive path
    fig.add_trace(go.Scatter(
        x=f0["adap_lons"], y=f0["adap_lats"],
        mode="lines+markers",
        line=dict(color=ADAPTIVE_COLOR, width=3.5),
        marker=dict(size=12, color=ADAPTIVE_COLOR, symbol="diamond",
                    line=dict(color="white", width=2)),
        name="Adaptive",
    ), row=1, col=1)

    # Greedy path
    fig.add_trace(go.Scatter(
        x=f0["greedy_lons"], y=f0["greedy_lats"],
        mode="lines+markers",
        line=dict(color=GREEDY_COLOR, width=3.5, dash="dash"),
        marker=dict(size=12, color=GREEDY_COLOR, symbol="square",
                    line=dict(color="white", width=2)),
        name="Greedy",
    ), row=1, col=1)

    # Decision text panels
    for col, text, name in [
        (1, f0["adap_text"],   "_adap_txt"),
        (2, f0["greedy_text"], "_grdy_txt"),
        (3, f0["field_text"],  "_field_txt"),
    ]:
        fig.add_trace(go.Scatter(
            x=[0], y=[0.95], mode="text",
            text=[text],
            textposition="top right",
            textfont=dict(size=10.5, color="#1a1a2e"),
            showlegend=False, hoverinfo="skip",
            name=name,
        ), row=2, col=col)

    # ── plotly animation frames ───────────────────────────────
    plotly_frames = []

    for fd in frames_data:

        field_bg = (
            "rgba(25,118,210,0.06)"   # blue tint for field A
            if fd["field_idx"] == 0
            else "rgba(230,74,25,0.06)"   # orange tint for field B
        )

        plotly_frames.append(go.Frame(
            data=[
                go.Scatter(x=fd["arc_lons_A"], y=fd["arc_lats_A"],
                           mode="lines",
                           line=dict(color="rgba(180,180,180,0.12)", width=0.6),
                           hoverinfo="skip", showlegend=False),
                go.Scatter(
                    x=fd["node_lons"], y=fd["node_lats"],
                    mode="markers",
                    marker=dict(
                        size=fd["node_sizes"],
                        color=fd["node_costs"],
                        colorscale="RdYlGn_r",
                        showscale=True,
                        colorbar=dict(title="Arc cost<br>(current field)",
                                      x=1.01, thickness=14, len=0.55),
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
                           text=[fd["adap_text"]],
                           textposition="top right",
                           textfont=dict(size=10.5, color="#1a1a2e"),
                           showlegend=False, hoverinfo="skip"),
                go.Scatter(x=[0], y=[0.95], mode="text",
                           text=[fd["greedy_text"]],
                           textposition="top right",
                           textfont=dict(size=10.5, color="#1a1a2e"),
                           showlegend=False, hoverinfo="skip"),
                go.Scatter(x=[0], y=[0.95], mode="text",
                           text=[fd["field_text"]],
                           textposition="top right",
                           textfont=dict(size=10.5, color="#1a1a2e"),
                           showlegend=False, hoverinfo="skip"),
            ],
            traces=[0, 1, 2, 3, 4, 5, 6],
            name=str(fd["step"]),
            layout=go.Layout(
                plot_bgcolor=field_bg,
                title=dict(text=(
                    f"Trap 2 — Switching Wind Fields  |  "
                    f"Step {fd['step']} / {NUM_STAGES}  |  "
                    f"Current field: "
                    f"{'A (favours N0)' if fd['field_idx']==0 else 'B (favours N4)'}  |  "
                    f"Seed {seed}  |  "
                    f"Adaptive: {adap['total_cost']:.4f}  "
                    f"Greedy: {greedy['total_cost']:.4f}"
                ))
            )
        ))

    fig.frames = plotly_frames

    # ── slider ────────────────────────────────────────────────
    sliders = [dict(
        active=0,
        currentvalue=dict(prefix="Step: ", visible=True, xanchor="center"),
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

    # ── layout ────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=(
                f"Trap 2 — Switching Wind Fields  |  "
                f"Step 0 / {NUM_STAGES}  |  Seed {seed}  |  "
                f"Field A = blue tint  |  Field B = orange tint<br>"
                f"<sup>◆ Blue = Adaptive  |  ■ Red = Greedy  |  "
                f"Node colour = arc cost under current field  |  "
                f"Hover nodes for full cost details</sup>"
            ),
            x=0.5, xanchor="center",
            font=dict(size=13, color="#1a1a2e"),
        ),
        paper_bgcolor="#f8f9fa",
        plot_bgcolor="rgba(25,118,210,0.06)",
        height=950,
        showlegend=True,
        legend=dict(
            x=0.01, y=0.62,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#ccc", borderwidth=1,
            font=dict(size=11),
        ),
        sliders=sliders,
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            x=0.5, xanchor="center",
            y=-0.07, yanchor="top",
            buttons=[
                dict(
                    label="▶ Play",
                    method="animate",
                    args=[None, dict(
                        frame=dict(duration=2000, redraw=True),
                        fromcurrent=True,
                        transition=dict(duration=400),
                    )],
                ),
                dict(
                    label="⏸ Pause",
                    method="animate",
                    args=[[None], dict(
                        frame=dict(duration=0, redraw=False),
                        mode="immediate",
                        transition=dict(duration=0),
                    )],
                ),
            ],
        )],
        font=dict(family="Inter, Arial, sans-serif"),
        annotations=[
            dict(
                text=(
                    "Field A (blue): low node index = cheap  |  "
                    "Field B (orange): high node index = cheap  |  "
                    f"Switch probability per stage: {SWITCH_PROB:.0%}"
                ),
                x=0.5, y=0.395, xref="paper", yref="paper",
                xanchor="center", showarrow=False,
                font=dict(size=10, color="#555"),
            )
        ],
    )

    for row, col in [(1, 1)]:
        fig.update_xaxes(title_text="Longitude (°)",
                         gridcolor="rgba(200,200,200,0.3)",
                         zeroline=False, row=row, col=col)
        fig.update_yaxes(title_text="Latitude (°)",
                         gridcolor="rgba(200,200,200,0.3)",
                         zeroline=False, row=row, col=col)

    for col in [1, 2, 3]:
        fig.update_xaxes(visible=False, range=[-0.05, 2.5],
                         row=2, col=col)
        fig.update_yaxes(visible=False, range=[-0.2, 1.1],
                         row=2, col=col)

    return fig, adap, greedy


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(f"Building visualisation (seed={SEED})...")
    fig, adap, greedy = build_figure(seed=SEED)

    print(f"\nResults for seed {SEED}:")
    print(f"  Adaptive total: {adap['total_cost']:.4f}")
    print(f"  Greedy   total: {greedy['total_cost']:.4f}")
    print(f"  Advantage:      {greedy['total_cost'] - adap['total_cost']:+.4f}")

    print("\nStep-by-step:")
    for d_a, d_g in zip(adap["decisions"], greedy["decisions"]):
        sw = "⚡SWITCH" if d_a["field_switched"] else "      "
        print(
            f"  S{d_a['stage']}  {sw}  "
            f"Plan:{['A','B'][d_a['planning_field_idx']]} "
            f"Real:{['A','B'][d_a['realised_field_idx']]}  |  "
            f"Adap→N{d_a['chosen_idx']} "
            f"(planned={d_a['imm_cost_planned']:.3f}+{d_a['ctg_planned']:.3f} "
            f"realised={d_a['realised_cost']:.3f})  |  "
            f"Greedy→N{d_g['chosen_idx']} "
            f"(planned={d_g['imm_cost_planned']:.3f} "
            f"realised={d_g['realised_cost']:.3f})"
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

    print(f"\nSaved: {output}")
    print("Controls: slider | ← → arrow keys | Play/Pause")