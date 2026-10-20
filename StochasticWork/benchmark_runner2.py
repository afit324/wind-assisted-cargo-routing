"""
benchmark_runner.py  —  Atlantic corridor benchmark
====================================================

Sweeps across:
  - Lateral penalty values
  - Cost measures
  - Latitude / longitude grid corridors

For each combination:
  1. Patches policy_grib grid, cost function, penalty
  2. Reloads GRIB forecasts
  3. Runs forecast-based policy experiment (reduced seeds)
  4. Builds ERA5 cost matrices (wind cost + same lateral penalty)
  5. Computes ERA5 hindsight optimal (guaranteed lower bound)
  6. Realises all forecast-policy routes under ERA5
  7. Saves results to BenchmarkResults/Benchmark_NNN/

Key design decisions
--------------------
- Lateral penalty applied in BOTH forecast planning and ERA5 realisation
  so that a policy paying the penalty to jump laterally sees the same
  penalty applied when realised under actual weather
- Seeds reduced to 30 (stratified across forecast combinations)
- ERA5 hindsight recomputed per benchmark under actual weather
- configure_grid() patches policy_grib directly and rebuilds layers
"""

import os
import json
import pickle
import importlib
import numpy as np
import pandas as pd
import cfgrib
import warnings
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
import time

warnings.filterwarnings("ignore")

import policy_grib
import experiments_gribs

# ============================================================
# BENCHMARK SETTINGS
# ============================================================

OUTPUT_DIR = "BenchmarkResults"

PREDICTWIND_DIR = "PredictWind"

ERA5_FILE = Path(
    "/Users/thomasfairweather/Downloads/ActualWeatherSimAtlanticJul_Aug"
)

# --- Lateral penalty values to sweep ---
LATERAL_PENALTIES = [0.05, 0.1, 0.25, 0.4, 0.6]

# --- Cost measures to sweep ---
MEASURES = [
    "wind_speed",
    "u_abs",
]

LATITUDE_GRIDS = [
    [30, 35, 40, 45, 50],
    [30, 33, 37, 40, 43, 47, 50],
    [28, 33, 38, 43, 48],
    [28, 32, 36, 40, 44, 48],
    [27, 31, 35, 39, 43, 47],
    [27, 30, 34, 37, 41, 45, 48],
]

LONGITUDE_GRIDS = [
    [-80, -75, -70, -65, -60, -55, -50, -45],   # 5° spacing
    [-79, -74, -69, -64, -59, -54, -49, -44],   # shifted +1
    [-78, -73, -68, -63, -58, -53, -48, -43],   # shifted +2
    [-77, -72, -67, -62, -57, -52, -47, -42],   # shifted +3
    [-76, -71, -66, -61, -56, -51, -46, -41],   # shifted +4
    [-80, -73, -66, -59, -52, -45, -38, -31],   # ~7° spacing (wider spread)
]

# Reduced seeds — covers forecast space without full 100
SEEDS     = 3
BASE_SEED = 100

POLICIES = [
    "adaptive",
    "hindsight",
    "voting",
    "greedy",
    "averaged_arc",
    "minimax",
    "optimistic",
    "weighted",
    "random",
]

# ============================================================
# GRID / COST FUNCTION / PENALTY PATCHING
# ============================================================

def configure_grid(lats, lons):
    """Patches policy_grib module variables for a new grid."""

    policy_grib.NODE_LATS  = list(lats)
    policy_grib.STAGE_LONS = list(lons)
    policy_grib.NUM_STAGES = len(lons)

    # Rebuild NODE_COORDS
    coords = {}
    for s in range(len(lons)):
        for i in range(len(lats)):
            coords[f"S{s}_N{i}"] = (lats[i], lons[s])
    policy_grib.NODE_COORDS    = coords
    policy_grib.NODES_PER_STAGE = len(lats)

    # Rebuild layers
    lys = [["START"]]
    for s in range(len(lons)):
        lys.append([f"S{s}_N{i}" for i in range(len(lats))])
    lys.append(["END"])
    policy_grib.layers = lys

    # Sync to experiments_gribs
    experiments_gribs.NUM_STAGES    = len(lons)
    experiments_gribs.NODES_PER_STAGE = len(lats)

    print(f"  Grid: {len(lons)} stages, {len(lats)} nodes/stage  "
          f"lats={list(lats)}  lons={list(lons)}")


def set_cost_function(measure):
    if measure not in policy_grib.COST_FUNCTIONS:
        raise ValueError(f"Unknown: {measure}")
    policy_grib.ACTIVE_COST_FN = measure
    print(f"  Cost function: {measure}")


def set_lateral_penalty(penalty):
    policy_grib.LATERAL_PENALTY = float(penalty)
    print(f"  Lateral penalty: {penalty}")


def reload_forecasts():
    """
    Rebuilds ALL_STAGE_FORECASTS in experiments_gribs.
    Must be called after any grid / cost / penalty change.
    """
    print("  Reloading GRIB forecasts...", end="", flush=True)
    new_forecasts = []
    for s in range(policy_grib.NUM_STAGES):
        new_forecasts.append(policy_grib.generate_forecasts(stage=s))
    experiments_gribs.ALL_STAGE_FORECASTS = new_forecasts
    print(f" {policy_grib.NUM_STAGES} stages loaded.")
    return new_forecasts


# ============================================================
# ERA5 LOADING  (loaded once, shared across all benchmarks)
# ============================================================

_era5_cache = None


def load_era5():
    global _era5_cache
    if _era5_cache is not None:
        return _era5_cache

    print(f"\nLoading ERA5: {ERA5_FILE}")
    datasets = cfgrib.open_datasets(str(ERA5_FILE))

    u10_da = v10_da = None
    for d in datasets:
        if "u10" in d and u10_da is None:
            u10_da = d["u10"]
        if "v10" in d and v10_da is None:
            v10_da = d["v10"]

    if u10_da is None or v10_da is None:
        raise RuntimeError("ERA5 file missing u10/v10")

    # Find time dimension
    time_dim = None
    for td in ["time", "valid_time", "step"]:
        if td in u10_da.dims:
            time_dim = td
            break

    if time_dim is not None:
        raw_times = u10_da[time_dim].values
    else:
        raw_times = u10_da.coords.get(
            "valid_time",
            u10_da.coords.get("time")
        ).values
        if raw_times.ndim == 0:
            raw_times = np.array([raw_times])

    times = pd.to_datetime(raw_times)
    lats  = u10_da.latitude.values
    lons  = u10_da.longitude.values

    # Ensure ascending latitude for interpolation
    lat_flip = lats[0] > lats[-1]
    if lat_flip:
        lats = lats[::-1]

    u_vals = u10_da.values
    v_vals = v10_da.values

    if u_vals.ndim == 2:
        u_vals = u_vals[np.newaxis]
        v_vals = v_vals[np.newaxis]

    if lat_flip:
        u_vals = u_vals[:, ::-1, :]
        v_vals = v_vals[:, ::-1, :]

    # Handle longitude wrapping: convert ERA5 0-360 to -180-180 if needed
    if lons.max() > 180:
        lons = np.where(lons > 180, lons - 360, lons)
        sort_idx = np.argsort(lons)
        lons   = lons[sort_idx]
        u_vals = u_vals[:, :, sort_idx]
        v_vals = v_vals[:, :, sort_idx]

    print(f"  ERA5: {len(times)} timesteps, "
          f"lat {lats.min():.1f}–{lats.max():.1f}, "
          f"lon {lons.min():.1f}–{lons.max():.1f}")

    # Build one interpolator per timestep
    u_interps = []
    v_interps = []
    for t in range(len(times)):
        u_interps.append(RegularGridInterpolator(
            (lats, lons), u_vals[t],
            method="linear", bounds_error=False, fill_value=None))
        v_interps.append(RegularGridInterpolator(
            (lats, lons), v_vals[t],
            method="linear", bounds_error=False, fill_value=None))

    _era5_cache = {
        "times"    : times,
        "u"        : u_interps,
        "v"        : v_interps,
    }
    print("  ERA5 loaded and cached.")
    return _era5_cache


# ============================================================
# ERA5 UV AT POINT
# ============================================================

def _get_uv_era5(weather, timestamp, lat, lon):
    """Gets (u,v) from ERA5 at timestamp and location."""
    diffs = np.abs(weather["times"] - timestamp)
    t     = int(np.argmin(diffs))
    pt    = np.array([[lat, lon]])
    u     = float(weather["u"][t](pt)[0])
    v     = float(weather["v"][t](pt)[0])
    return u, v


# ============================================================
# ERA5 ARC COST  (24-hour daily mean + lateral penalty)
# ============================================================

def _era5_arc_cost(weather, stage, from_node, to_node):
    """
    ERA5 arc cost = daily-mean wind cost + lateral penalty.

    The lateral penalty is IDENTICAL to what was used in the
    forecast planning — so a policy that paid a penalty to jump
    from N0 to N4 in the forecast sees the same penalty in ERA5.
    The wind cost changes (actual vs forecast wind) but the
    structural penalty does not.
    """

    cost_fn = policy_grib.COST_FUNCTIONS[policy_grib.ACTIVE_COST_FN]
    day     = pd.Timestamp(policy_grib.STAGE_DATES[
                               min(stage, policy_grib.NUM_STAGES - 1)])

    # Get node coordinates, handling virtual START/END
    if from_node == "START":
        lat_a, lon_a = policy_grib.NODE_COORDS[to_node]
    elif from_node == "END":
        return 0.0
    else:
        lat_a, lon_a = policy_grib.NODE_COORDS[from_node]

    if to_node == "END":
        lat_b, lon_b = policy_grib.NODE_COORDS[from_node]
    elif to_node == "START":
        return 0.0
    else:
        lat_b, lon_b = policy_grib.NODE_COORDS[to_node]

    # Sample 24 hourly ERA5 values and average
    hourly_wind_costs = []
    for h in range(24):
        ts      = day + pd.Timedelta(hours=h)
        u_a, v_a = _get_uv_era5(weather, ts, lat_a, lon_a)
        u_b, v_b = _get_uv_era5(weather, ts, lat_b, lon_b)
        hourly_wind_costs.append(cost_fn(u_a, v_a, u_b, v_b))

    wind_cost = float(np.mean(hourly_wind_costs))

    # Add same lateral penalty as forecast planning
    pen = policy_grib.lateral_penalty(from_node, to_node)

    return wind_cost + pen


# ============================================================
# ERA5 COST MATRICES
# ============================================================

def _build_era5_cost_matrices(weather):
    """
    Builds ERA5 cost dict for all layer transitions.
    Returns list indexed by layer transition index (0..NUM_STAGES).
    """
    num_stages = policy_grib.NUM_STAGES
    lys        = policy_grib.layers
    matrices   = []

    for t in range(len(lys) - 1):

        matrix     = {}
        stage_idx  = min(t, num_stages - 1)

        for from_node in lys[t]:
            matrix[from_node] = {}
            for to_node in lys[t + 1]:
                matrix[from_node][to_node] = _era5_arc_cost(
                    weather, stage_idx, from_node, to_node
                )

        matrices.append(matrix)

    return matrices


# ============================================================
# ERA5 HINDSIGHT OPTIMAL
# ============================================================

def _compute_era5_hindsight(era5_matrices):
    """
    Backward-induction DP on ERA5 cost matrices.
    Guaranteed <= all ERA5 policy costs.
    Includes lateral penalty (same as planning).
    """
    lys        = policy_grib.layers
    num_stages = policy_grib.NUM_STAGES

    V         = {}
    best_next = {}
    V["END"]  = 0.0

    for t in range(len(lys) - 2, -1, -1):
        mat = era5_matrices[t]
        for node in lys[t]:
            best_cost = float("inf")
            best_node = None
            for next_node in lys[t + 1]:
                c = mat.get(node, {}).get(next_node, float("inf"))
                total = c + V[next_node]
                if total < best_cost:
                    best_cost = total
                    best_node = next_node
            V[node]         = best_cost
            best_next[node] = best_node

    path = ["START"]
    node = "START"
    while node != "END":
        node = best_next[node]
        path.append(node)

    stage_costs = []
    for t in range(num_stages):
        fn = path[t]
        tn = path[t + 1]
        stage_costs.append(era5_matrices[t].get(fn, {}).get(tn, float("inf")))

    return path, sum(stage_costs), stage_costs


# ============================================================
# REALISE A ROUTE UNDER ERA5
# ============================================================

def _realise_route_era5(route, era5_matrices):
    num_stages  = policy_grib.NUM_STAGES
    total_cost  = 0.0
    stage_costs = []
    for t in range(num_stages):
        fn  = route[t]
        tn  = route[t + 1]
        c   = era5_matrices[t].get(fn, {}).get(tn, float("inf"))
        total_cost += c
        stage_costs.append(c)
    return total_cost, stage_costs


# ============================================================
# RUN FORECAST EXPERIMENT  (reduced seeds)
# ============================================================

def _run_forecast_experiment():
    """
    Runs the forecast policy experiment with SEEDS seeds.
    Returns (results list, stats dict, cost arrays dict).
    """
    experiments_gribs.SEEDS     = SEEDS
    experiments_gribs.BASE_SEED = BASE_SEED

    results = experiments_gribs.run_experiments()
    stats, cost_arrays = experiments_gribs.compute_stats(results)
    return results, stats, cost_arrays


# ============================================================
# RUN ONE FULL BENCHMARK
# ============================================================

def _run_one_benchmark(lats, lons, measure, penalty, folder, weather):

    # --- configure ---
    configure_grid(lats, lons)
    set_cost_function(measure)
    set_lateral_penalty(penalty)
    reload_forecasts()

    # --- forecast experiment ---
    print(f"\n  Running forecast experiment ({SEEDS} seeds)...")
    forecast_results, forecast_stats, forecast_arrays = \
        _run_forecast_experiment()

    with open(os.path.join(folder, "forecast_routes.pkl"), "wb") as f:
        pickle.dump(forecast_results, f)

    # --- ERA5 cost matrices ---
    print("  Building ERA5 cost matrices (24h mean per arc)...")
    era5_matrices = _build_era5_cost_matrices(weather)

    # --- ERA5 hindsight ---
    print("  Computing ERA5 hindsight optimal...")
    h_path, h_cost, h_sc = _compute_era5_hindsight(era5_matrices)
    print(f"  ERA5 hindsight: {h_cost:.4f}  path: {h_path}")

    # --- ERA5 policy realisation ---
    print("  Realising forecast routes under ERA5...")

    era5_costs   = {"hindsight": h_cost}
    era5_details = []

    # Use median-cost seed for deterministic policies,
    # average across all seeds for random
    adap_costs = [r["adaptive"][0] for r in forecast_results]
    median_idx = int(np.argsort(adap_costs)[len(adap_costs) // 2])
    r_ref      = forecast_results[median_idx]

    for policy in POLICIES:

        if policy == "hindsight":
            continue

        if policy == "random":
            # Average ERA5 cost across all seeds
            seed_era5_costs = []
            for r in forecast_results:
                _, route, _ = r[policy]
                c, _ = _realise_route_era5(route, era5_matrices)
                seed_era5_costs.append(c)
            era5_costs[policy] = float(np.mean(seed_era5_costs))

        else:
            _, route, _ = r_ref[policy]
            c, sc = _realise_route_era5(route, era5_matrices)
            era5_costs[policy] = c

            for t, (sc_t, fn, tn) in enumerate(
                zip(sc, route[:-1], route[1:])
            ):
                pen = policy_grib.lateral_penalty(fn, tn)
                era5_details.append({
                    "policy"          : policy,
                    "stage"           : t,
                    "date"            : policy_grib.STAGE_DATES[
                                            min(t, policy_grib.NUM_STAGES-1)],
                    "from_node"       : fn,
                    "to_node"         : tn,
                    "era5_total_cost" : sc_t,
                    "lateral_penalty" : pen,
                    "era5_wind_cost"  : sc_t - pen,
                })

    # --- Verify hindsight is lower bound ---
    violations = {
        p: c for p, c in era5_costs.items()
        if c < h_cost - 1e-6 and p != "hindsight"
    }
    if violations:
        print(f"  [WARNING] ERA5 hindsight violated: {violations}")
        print("  This indicates ERA5 coverage gaps for these coordinates.")
    else:
        print("  ERA5 hindsight verified <= all policy costs ✓")

    # --- Print summary table ---
    print(f"\n  {'Policy':<22} {'Forecast C':>11} "
          f"{'ERA5 C':>10} {'ERA5 gap%':>10}")
    print(f"  {'-'*55}")
    print(f"  {'ERA5 Hindsight':<22} {'---':>11} "
          f"{h_cost:>10.4f}  (lower bound)")

    for pol in POLICIES:
        if pol == "hindsight":
            continue
        fc_arr = forecast_arrays.get(pol,
                 forecast_arrays.get("adaptive"))
        fc = float(np.mean(fc_arr)) if fc_arr is not None else float("nan")
        ec  = era5_costs[pol]
        gap = (ec - h_cost) / h_cost * 100 if h_cost > 0 else float("nan")
        print(f"  {pol:<22} {fc:>11.4f} {ec:>10.4f} {gap:>10.1f}%")

    # --- Save ---
    summary = {
        "measure"              : measure,
        "lateral_penalty"      : penalty,
        "latitudes"            : list(lats),
        "longitudes"           : list(lons),
        "seeds"                : SEEDS,
        "forecast_stats"       : {
            k: float(v) if isinstance(v, (np.floating, float)) else v
            for k, v in forecast_stats.items()
        },
        "era5_policy_costs"    : era5_costs,
        "era5_hindsight_cost"  : h_cost,
        "era5_hindsight_path"  : h_path,
    }

    with open(os.path.join(folder, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4, default=str)

    pd.DataFrame(era5_details).to_csv(
        os.path.join(folder, "era5_arc_details.csv"), index=False
    )

    return summary


# ============================================================
# MAIN BENCHMARK LOOP
# ============================================================

def main():

    weather   = load_era5()
    benchmark = 1
    summaries = []

    total = (len(LATERAL_PENALTIES) *
             len(MEASURES) *
             len(LATITUDE_GRIDS) *
             len(LONGITUDE_GRIDS))

    for penalty in LATERAL_PENALTIES:
        for measure in MEASURES:
            for lats in LATITUDE_GRIDS:
                for lons in LONGITUDE_GRIDS:

                    print("\n" + "=" * 70)
                    print(f"Benchmark {benchmark}/{total}  |  "
                          f"penalty={penalty}  measure={measure}")
                    print(f"  lats={lats}  lons={lons}")
                    print("=" * 70)

                    folder = os.path.join(
                        OUTPUT_DIR,
                        f"Benchmark_{benchmark:03d}"
                    )
                    os.makedirs(folder, exist_ok=True)

                    with open(
                        os.path.join(folder, "settings.json"), "w"
                    ) as f:
                        json.dump({
                            "benchmark"      : benchmark,
                            "measure"        : measure,
                            "lateral_penalty": penalty,
                            "latitudes"      : list(lats),
                            "longitudes"     : list(lons),
                        }, f, indent=4)

                    try:
                        s = _run_one_benchmark(
                            lats, lons, measure, penalty, folder, weather
                        )
                        s["benchmark"] = benchmark
                        summaries.append(s)
                        print(f"\n  Benchmark {benchmark} complete.")

                    except Exception as e:
                        print(f"\n  ERROR in benchmark {benchmark}: {e}")
                        import traceback
                        traceback.print_exc()

                    benchmark += 1

    # --- Master summary CSV ---
    rows = []
    for s in summaries:
        row = {
            "benchmark"      : s.get("benchmark"),
            "measure"        : s["measure"],
            "lateral_penalty": s["lateral_penalty"],
            "latitudes"      : str(s["latitudes"]),
            "longitudes"     : str(s["longitudes"]),
            "era5_hindsight" : s["era5_hindsight_cost"],
        }
        for pol, cost in s.get("era5_policy_costs", {}).items():
            row[f"era5_{pol}"] = cost
        # Forecast adaptive mean
        row["forecast_adaptive_mean"] = s.get(
            "forecast_stats", {}
        ).get("adaptive_mean", float("nan"))
        rows.append(row)

    master_path = os.path.join(OUTPUT_DIR, "master_summary.csv")
    pd.DataFrame(rows).to_csv(master_path, index=False)
    print(f"\nMaster summary: {master_path}")
    print("All benchmarks complete.")


if __name__ == "__main__":
    main()