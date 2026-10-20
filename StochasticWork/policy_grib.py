import os
import numpy as np
import cfgrib
import warnings


warnings.filterwarnings("ignore")

# ============================================================
# PARAMETERS
# ============================================================

NUM_FORECASTS   = 6
NUM_STAGES      = 8
NODES_PER_STAGE = 5

PREDICTWIND_DIR = "PredictWind"
TARGET_AREA     = ""   # Not used — see _is_atlantic_area() below

# ============================================================
# ATLANTIC FILE DETECTION
# ============================================================
# Rather than matching a fixed area string (which changes daily),
# we parse the bounding box from the filename and check whether
# it covers the Atlantic node grid.
# Atlantic files have northern bound >= 50N and cover negative
# longitudes (west of Greenwich).
# Pacific files have low northern bounds (< 30N) or only positive
# longitudes.

def _parse_area_string(fname):
    """
    Parses the bounding box from a PredictWind filename.
    Format: ..._{N}N_{S}S_{E}E_{W}W.grb  (S and W may be negative)
    Returns (north, south, east, west) as floats or None if unparseable.
    """
    try:
        base   = fname.replace(".grb", "")
        parts  = base.split("_")
        # Find 4 consecutive tokens matching the pattern NNn, SSs, EEe, WWw
        for i in range(len(parts) - 3):
            try:
                n_str, s_str, e_str, w_str = parts[i:i+4]
                north = float(n_str.rstrip("N"))
                south = float(s_str.rstrip("S"))
                east  = float(e_str.rstrip("E"))
                west  = float(w_str.rstrip("W"))
                # Validate: north > south
                if north > south:
                    return north, south, east, west
            except (ValueError, AttributeError):
                continue
    except Exception:
        pass
    return None


def _is_atlantic_file(fname):
    """
    Returns True if the file covers the Atlantic corridor.
    Heuristic: northern bound >= 50N AND covers some negative longitudes.
    This distinguishes Atlantic downloads from Pacific ones.
    """
    bounds = _parse_area_string(fname)
    if bounds is None:
        return False
    north, south, east, west = bounds
    # Atlantic: high northern bound, negative longitudes present
    return north >= 50.0 and west < 0.0

# ============================================================
# STAGE -> DATE MAPPING  (8 consecutive Atlantic days)
# ============================================================

STAGE_DATES = [
    "2026-08-02",
    "2026-08-03",
    "2026-08-04",
    "2026-08-05",
    "2026-08-06",
    "2026-08-07",
    "2026-08-08",
    "2026-08-09",
]

# ============================================================
# MODELS  (index = forecast index 0-5)
# ============================================================

MODELS = [
    "PWG",
    "GFS",
    "PWE",
    "ECMWF_WRCTPaG",
    "ICON",
]

# Always derived from MODELS so they stay in sync
NUM_FORECASTS = len(MODELS)

# ============================================================
# NODE GRID  (configurable from benchmark_runner)
# ============================================================

NODE_LATS  = [20.0, 30.0, 40.0, 50.0, 60.0]
STAGE_LONS = [-90.0, -85.0, -80.0, -75.0, -70.0, -65.0, -60.0, -55.0]

# ============================================================
# START AND END ANCHOR POINTS
# Fixed geographic points one stage-step either side of the grid.
# Both sit at the middle node's latitude so they align with the
# centre of the node spread.
# Longitude spacing = gap between consecutive stage longitudes (5°).
# START_COORD and END_COORD are rebuilt by configure_grid() when
# the grid changes in benchmark_runner.
# ============================================================

def _lon_spacing():
    if len(STAGE_LONS) < 2:
        return 5.0
    return abs(STAGE_LONS[1] - STAGE_LONS[0])

def _middle_lat():
    return NODE_LATS[len(NODE_LATS) // 2]

def get_start_coord():
    """(lat, lon) of the START anchor point."""
    return (_middle_lat(), STAGE_LONS[0] - _lon_spacing())

def get_end_coord():
    """(lat, lon) of the END anchor point."""
    return (_middle_lat(), STAGE_LONS[-1] + _lon_spacing())

NODE_COORDS = {}
for _s in range(NUM_STAGES):
    for _i in range(NODES_PER_STAGE):
        NODE_COORDS[f"S{_s}_N{_i}"] = (NODE_LATS[_i], STAGE_LONS[_s])

# ============================================================
# LAYERS
# ============================================================

def _rebuild_layers():
    lys = [["START"]]
    for s in range(NUM_STAGES):
        lys.append([f"S{s}_N{i}" for i in range(NODES_PER_STAGE)])
    lys.append(["END"])
    return lys

layers = _rebuild_layers()

# ============================================================
# LATERAL MOVEMENT PENALTY
# ============================================================
#
# Applied to BOTH forecast planning and ERA5 realisation.
# penalty = LATERAL_PENALTY * max(0, |node_i - node_j| - 1)
#
# |diff| = 0 or 1 -> no penalty  (same node or one step)
# |diff| = 2      -> 1 * LATERAL_PENALTY
# |diff| = 3      -> 2 * LATERAL_PENALTY
# |diff| = 4      -> 3 * LATERAL_PENALTY  (e.g. N0 -> N4)
#
# Set to 0.0 to disable.

LATERAL_PENALTY = 0.05   # default; overridden from benchmark_runner

def lateral_penalty(from_node, to_node):
    """
    Returns the lateral movement penalty for a stage transition.
    START and END are virtual nodes — no penalty applied.
    """
    if from_node in ("START", "END") or to_node in ("START", "END"):
        return 0.0
    try:
        i = int(from_node.split("_N")[1])
        j = int(to_node.split("_N")[1])
        return LATERAL_PENALTY * max(0, abs(i - j) - 1)
    except (IndexError, ValueError):
        return 0.0

# ============================================================
# ARC COST FUNCTIONS
# ============================================================

def _cost_wind_speed(u_a, v_a, u_b, v_b):
    ws_a = np.sqrt(u_a**2 + v_a**2)
    ws_b = np.sqrt(u_b**2 + v_b**2)
    return 1.0 / max((ws_a + ws_b) / 2.0, 0.1)

def _cost_u_component(u_a, v_a, u_b, v_b):
    return 1.0 / max((u_a + u_b) / 2.0, 0.1)

def _cost_v_component(u_a, v_a, u_b, v_b):
    return 1.0 / max((v_a + v_b) / 2.0, 0.1)

def _cost_u_abs(u_a, v_a, u_b, v_b):
    return 1.0 / max((abs(u_a) + abs(u_b)) / 2.0, 0.1)

def _cost_v_abs(u_a, v_a, u_b, v_b):
    return 1.0 / max((abs(v_a) + abs(v_b)) / 2.0, 0.1)

def _cost_u_favourable(u_a, v_a, u_b, v_b):
    return 1.0 / max((u_a + u_b) / 2.0 + 0.5, 0.1)

def _cost_resultant_heading(u_a, v_a, u_b, v_b):
    # Projects onto due-east heading (Atlantic route goes east)
    return 1.0 / max((u_a + u_b) / 2.0, 0.1)

def _cost_wind_speed_squared(u_a, v_a, u_b, v_b):
    ws2_a = u_a**2 + v_a**2
    ws2_b = u_b**2 + v_b**2
    return 1.0 / max((ws2_a + ws2_b) / 2.0, 0.01)

COST_FUNCTIONS = {
    "wind_speed"        : _cost_wind_speed,
    "u_component"       : _cost_u_component,
    "v_component"       : _cost_v_component,
    "u_abs"             : _cost_u_abs,
    "v_abs"             : _cost_v_abs,
    "u_favourable"      : _cost_u_favourable,
    "resultant_heading" : _cost_resultant_heading,
    "wind_speed_squared": _cost_wind_speed_squared,
}

ACTIVE_COST_FN = "wind_speed"

# ============================================================
# FILE SCANNING
# ============================================================

def _get_model_tag(fname):
    # Check ECMWF_WRCTPaG before ECMWF to avoid partial match
    for model in MODELS:
        if f"_{model}_" in fname:
            return model
    return None


def _find_files_for_date(date):
    """
    Scans the date folder and returns dict: model -> [sorted file paths]
    TARGET_AREA files preferred; falls back to any area if not found.
    """
    folder = os.path.join(PREDICTWIND_DIR, date)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    model_files_target = {m: [] for m in MODELS}
    model_files_any    = {m: [] for m in MODELS}

    all_grbs = sorted(f for f in os.listdir(folder)
                      if f.endswith(".grb"))

    for fname in all_grbs:
        model = _get_model_tag(fname)
        if model is None:
            continue
        fpath = os.path.join(folder, fname)
        model_files_any[model].append(fpath)
        if TARGET_AREA in fname:
            model_files_target[model].append(fpath)

    # Diagnostic: show area strings found on first call
    area_strings = set()
    for fname in all_grbs:
        parts = fname.replace(".grb", "").split("_")
        # Area string is typically the last 4 underscore-joined tokens
        if len(parts) >= 4:
            area_strings.add("_".join(parts[-4:]))

    result  = {}
    missing = []
    fallback_used = []

    for model in MODELS:
        if model_files_target[model]:
            result[model] = model_files_target[model]
        elif model_files_any[model]:
            result[model] = model_files_any[model]
            fallback_used.append(model)
        else:
            missing.append(model)
            result[model] = []

    if fallback_used:
        print(f"    [INFO] {date}: TARGET_AREA='{TARGET_AREA}' not found "
              f"for {fallback_used}.")
        print(f"    [INFO] Area strings in folder: {sorted(area_strings)}")
        print(f"    [INFO] Using best available file for these models.")

    if missing:
        print(f"    [WARN] {date}: no .grb file at all for {missing}")

    return result, missing


# ============================================================
# WIND FIELD LOADING
# ============================================================

def _load_wind_field(filepath):
    """
    Opens a GRIB file, returns (u10_da, v10_da) or None.
    Suppresses cfgrib stderr noise for corrupt files.
    """
    try:
        if os.path.getsize(filepath) < 100:   # truly empty files only
            return None
    except OSError:
        return None

    u10_da = v10_da = None
    try:
        import io, contextlib
        with contextlib.redirect_stderr(io.StringIO()):
            datasets = cfgrib.open_datasets(
                filepath,
                errors="ignore"
            )
        for ds in datasets:
            if u10_da is None and "u10" in ds:
                u10_da = ds["u10"].isel(step=0)
            if v10_da is None and "v10" in ds:
                v10_da = ds["v10"].isel(step=0)
            if u10_da is not None and v10_da is not None:
                break
    except Exception:
        return None

    if u10_da is None or v10_da is None:
        return None

    return u10_da, v10_da


def _uv_at(u10_da, v10_da, lat, lon):
    """Returns (u, v) at nearest grid point."""
    u = float(u10_da.sel(latitude=lat, longitude=lon,
                          method="nearest").values)
    v = float(v10_da.sel(latitude=lat, longitude=lon,
                          method="nearest").values)
    return u, v


# ============================================================
# GENERATE FORECASTS
# ============================================================

def generate_forecasts(stage, return_winds=False):
    """
    Loads GRIB files for STAGE_DATES[stage].
    Returns list of NUM_FORECASTS cost dicts including lateral penalty.

    cost[from_node][to_node] = wind_cost(u,v) + lateral_penalty(i,j)

    If return_winds=True, also returns winds dict:
        winds[k][node_name] = wind_speed m/s
    """

    date = STAGE_DATES[stage]
    cost_fn = COST_FUNCTIONS[ACTIVE_COST_FN]

    model_files, missing = _find_files_for_date(date)

    # Load wind fields — try each file in order, stop on first success
    wind_fields  = {}
    failed_models = []

    for model in MODELS:
        files = model_files.get(model, [])
        field = None
        for fpath in files:
            field = _load_wind_field(fpath)
            if field is not None:
                break
        wind_fields[model] = field
        if field is None:
            failed_models.append(model)

    if failed_models:
        print(f"    [WARN] {date}: all files corrupt/empty for "
              f"{failed_models} — using neutral wind (1.0 m/s)")

    # Extract (u,v) at every node for each model
    uv_components = []
    wind_speed_cache = []   # for return_winds

    for k, model in enumerate(MODELS):
        field = wind_fields[model]
        uv    = {}
        ws    = {}

        for s in range(NUM_STAGES):
            for i in range(NODES_PER_STAGE):
                node     = f"S{s}_N{i}"
                lat, lon = NODE_COORDS[node]

                if field is not None:
                    try:
                        u10_da, v10_da = field
                        u, v = _uv_at(u10_da, v10_da, lat, lon)
                    except Exception:
                        u, v = 1.0, 1.0
                else:
                    u, v = 1.0, 1.0

                uv[node] = (u, v)
                ws[node] = float(np.sqrt(u**2 + v**2))

        uv_components.append(uv)
        wind_speed_cache.append(ws)

    # Build cost dicts
    forecasts = []

    for k in range(NUM_FORECASTS):

        scenario = {}

        # START -> stage-0 nodes
        # Arc cost uses actual wind at the START anchor point
        scenario["START"] = {}
        start_lat, start_lon = get_start_coord()

        if field is not None:
            try:
                u10_da, v10_da = field
                u_start, v_start = _uv_at(u10_da, v10_da,
                                           start_lat, start_lon)
            except Exception:
                u_start, v_start = 1.0, 1.0
        else:
            u_start, v_start = 1.0, 1.0

        for i in range(NODES_PER_STAGE):
            to_node  = f"S0_N{i}"
            u_t, v_t = uv_components[k][to_node]
            # Lateral penalty: START is at middle-node latitude,
            # so penalty scales with distance from middle node
            mid_idx = NODES_PER_STAGE // 2
            pen = LATERAL_PENALTY * max(0, abs(i - mid_idx) - 1)
            scenario["START"][to_node] = (
                cost_fn(u_start, v_start, u_t, v_t) + pen
            )

        # Stage s -> stage s+1  (lateral penalty applies here)
        for s in range(NUM_STAGES - 1):
            for i in range(NODES_PER_STAGE):
                from_node = f"S{s}_N{i}"
                u_f, v_f  = uv_components[k][from_node]
                scenario.setdefault(from_node, {})
                for j in range(NODES_PER_STAGE):
                    to_node  = f"S{s+1}_N{j}"
                    u_t, v_t = uv_components[k][to_node]
                    wind_cost = cost_fn(u_f, v_f, u_t, v_t)
                    pen       = lateral_penalty(from_node, to_node)
                    scenario[from_node][to_node] = wind_cost + pen

        # Last stage -> END
        # Arc cost uses actual wind at the END anchor point
        end_lat, end_lon = get_end_coord()
        last_s = NUM_STAGES - 1

        if field is not None:
            try:
                u10_da, v10_da = field
                u_end, v_end = _uv_at(u10_da, v10_da, end_lat, end_lon)
            except Exception:
                u_end, v_end = 1.0, 1.0
        else:
            u_end, v_end = 1.0, 1.0

        for i in range(NODES_PER_STAGE):
            from_node = f"S{last_s}_N{i}"
            u_f, v_f  = uv_components[k][from_node]
            scenario.setdefault(from_node, {})
            # Lateral penalty: END is at middle-node latitude
            mid_idx = NODES_PER_STAGE // 2
            pen = LATERAL_PENALTY * max(0, abs(i - mid_idx) - 1)
            scenario[from_node]["END"] = (
                cost_fn(u_f, v_f, u_end, v_end) + pen
            )

        forecasts.append(scenario)

    # Print loaded models
    loaded = [m for m in MODELS if wind_fields[m] is not None]
    print(f"  Stage {stage} ({date}) ... "
          f"loaded ({', '.join(loaded)})  "
          f"[measure: {ACTIVE_COST_FN}  "
          f"penalty: {LATERAL_PENALTY}]")

    if return_winds:
        return forecasts, wind_speed_cache

    return forecasts


# ============================================================
# COST-TO-GO
# ============================================================

def future_cost_to_go(current_node, current_stage, forecast):
    if current_stage == len(layers) - 1:
        return 0.0
    return min(
        forecast[current_node][n] +
        future_cost_to_go(n, current_stage + 1, forecast)
        for n in layers[current_stage + 1]
    )


# ============================================================
# POLICY STEP
# ============================================================

def choose_next_node(current_node, current_stage, forecasts):
    """
    Chooses next node by minimising mean total cost across forecasts.
    Arc costs already include lateral penalty from generate_forecasts.
    """
    next_layer          = layers[current_stage + 1]
    best_node           = None
    best_mean_cost      = float("inf")
    best_forecast_index = None

    for candidate in next_layer:
        imm     = [f[current_node][candidate] for f in forecasts]
        avg_imm = np.mean(imm)
        futures = [future_cost_to_go(candidate, current_stage + 1, f)
                   for f in forecasts]
        totals  = [avg_imm + fc for fc in futures]
        mean_t  = np.mean(totals)

        if mean_t < best_mean_cost:
            best_mean_cost      = mean_t
            best_node           = candidate
            best_forecast_index = int(np.argmin(totals))

    return best_node, best_forecast_index, best_mean_cost

# ============================================================
# DISCOUNTED ADAPTIVE POLICY
# ============================================================

# Discount applied to FUTURE arcs after the immediate candidate arc.
#
# Immediate arc:
#     weight = 1.0
#
# Future arcs:
#     0.9, 0.7, 0.5, 0.3, 0.1, 0.0, ...
#
# The policy therefore gives progressively less importance to
# arcs further away from the current node.

DISCOUNT_START = 0.9
DISCOUNT_STEP  = 0.2
DISCOUNT_FLOOR = 0.0


def discounted_weight(distance_from_candidate):
    """
    Returns the weight for a future arc.

    distance_from_candidate = 1:
        first future arc -> 0.9

    distance_from_candidate = 2:
        second future arc -> 0.7

    distance_from_candidate = 3:
        third future arc -> 0.5

    etc.

    The weight is never allowed to become negative.
    """
    return max(
        DISCOUNT_FLOOR,
        DISCOUNT_START - DISCOUNT_STEP * (distance_from_candidate - 1)
    )


def discounted_future_cost_to_go(
    current_node,
    current_stage,
    forecast,
    distance_from_candidate=1
):
    """
    Calculates the discounted optimal cost-to-go under ONE forecast.

    The first arc after the candidate node receives weight 0.9,
    the second receives 0.7, the third 0.5, etc.

    The route is still chosen optimally under this forecast;
    only the value assigned to later arcs is discounted.
    """

    # If we are already at END, there is no future cost.
    if current_node == "END":
        return 0.0

    # No more transitions remain.
    if current_stage >= len(layers) - 1:
        return 0.0

    weight = discounted_weight(distance_from_candidate)

    return min(
        weight * forecast[current_node][next_node]
        + discounted_future_cost_to_go(
            next_node,
            current_stage + 1,
            forecast,
            distance_from_candidate + 1
        )
        for next_node in layers[current_stage + 1]
    )


def choose_next_node_discounted(
    current_node,
    current_stage,
    forecasts
):
    """
    Discounted Adaptive policy.

    For each candidate next node:

        1. Average the immediate cost across all forecasts
           with weight 1.0.

        2. For each forecast separately, find the optimal
           future route using discounted future arc costs.

        3. Apply:
               0.9 to the first future arc
               0.7 to the second
               0.5 to the third
               etc.

        4. Average the discounted future cost across forecasts.

        5. Choose the candidate with the smallest:

               average immediate cost
               +
               average discounted future cost
    """

    next_layer = layers[current_stage + 1]

    best_node       = None
    best_mean_cost  = float("inf")
    best_forecast_index = None

    for candidate in next_layer:

        # ----------------------------------------------------
        # 1. Immediate cost
        # ----------------------------------------------------
        # All forecasts have equal weight = 1 here.
        immediate_costs = [
            forecast[current_node][candidate]
            for forecast in forecasts
        ]

        avg_immediate_cost = np.mean(immediate_costs)

        # ----------------------------------------------------
        # 2. Discounted optimal future route under each
        #    individual forecast
        # ----------------------------------------------------
        discounted_futures = [
            discounted_future_cost_to_go(
                candidate,
                current_stage + 1,
                forecast,
                distance_from_candidate=1
            )
            for forecast in forecasts
        ]

        # ----------------------------------------------------
        # 3. Average the discounted future costs
        # ----------------------------------------------------
        avg_discounted_future = np.mean(discounted_futures)

        # ----------------------------------------------------
        # 4. Total value of choosing this candidate
        # ----------------------------------------------------
        total_value = (
            avg_immediate_cost
            + avg_discounted_future
        )

        if total_value < best_mean_cost:
            best_mean_cost = total_value
            best_node = candidate

            # Keep this consistent with choose_next_node().
            # This is only diagnostic information.
            best_forecast_index = int(
                np.argmin(
                    np.array(immediate_costs)
                    + np.array(discounted_futures)
                )
            )

    return best_node, best_forecast_index, best_mean_cost

# ============================================================
# SANITY CHECK
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print(f"policy_grib.py  |  measure={ACTIVE_COST_FN}  "
          f"penalty={LATERAL_PENALTY}")
    print(f"Stages={NUM_STAGES}  Nodes={NODES_PER_STAGE}")
    print(f"Lats={NODE_LATS}")
    print(f"Lons={STAGE_LONS}")
    print(f"Target area: {TARGET_AREA}")
    print(f"Layers: {len(layers)}")
    for i, l in enumerate(layers):
        print(f"  layers[{i}] = {l}")
    print("\nLoading stage 0...")
    f0 = generate_forecasts(stage=0)
    print("\nSTART arc costs, forecast 0:")
    for node, cost in f0[0]["START"].items():
        print(f"  START -> {node}: {cost:.4f}")
    print("\nS0_N2 -> S1_N* costs (showing penalty effect):")
    for node, cost in f0[0]["S0_N2"].items():
        pen = lateral_penalty("S0_N2", node)
        print(f"  S0_N2 -> {node}: total={cost:.4f}  penalty={pen:.4f}")