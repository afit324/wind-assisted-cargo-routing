"""
simulate_actual_weather.py

Evaluates forecast-generated policy routes under realised ERA5 hourly weather.

IMPORTANT:
- Routes are generated using PredictWind forecasts.
- This script does NOT rebuild routes.
- It only replaces the forecast costs with realised ERA5 weather costs.

ERA5 file:
    /Users/thomasfairweather/Downloads/GribFileForActualWeatherSim1

Variables:
    u10
    v10

Output:
    actual_weather_results.csv
    actual_weather_arc_details.csv

"""

import pickle
import numpy as np
import pandas as pd
import xarray as xr
import os

from scipy.interpolate import RegularGridInterpolator
from pathlib import Path

from policy_grib import (
    COST_FUNCTIONS,
    ACTIVE_COST_FN,
    NODE_COORDS,
    NUM_STAGES,
    STAGE_DATES,
)


# ============================================================
# PARAMETERS
# ============================================================

RESULTS_FILE = "/Users/thomasfairweather/Documents/University/P4P/LeviCode/Part-4-Project---Wind-Assisted-Cargo-Vessels-main/forecast_policy_routes.pkl"

ERA5_FILE = Path(
    "/Users/thomasfairweather/Downloads/GribFileForActualWeatherSim1"
)

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
# LOAD ERA5 WEATHER
# ============================================================

def load_era5():

    print("\nLoading ERA5 actual weather...")

    ds = xr.open_dataset(
        ERA5_FILE,
        engine="cfgrib",
        backend_kwargs={"indexpath":""}
    )

    print(ds)

    times = pd.to_datetime(ds.time.values)

    latitudes = ds.latitude.values
    longitudes = ds.longitude.values

    u = ds.u10.values
    v = ds.v10.values


    # ERA5 latitude is descending
    # scipy requires increasing coordinates

    if latitudes[0] > latitudes[-1]:

        latitudes = latitudes[::-1]

        u = u[:, ::-1, :]
        v = v[:, ::-1, :]


    print(
        f"\nERA5 loaded:"
        f"\n  Time steps : {len(times)}"
        f"\n  Lat range  : {latitudes.min()} to {latitudes.max()}"
        f"\n  Lon range  : {longitudes.min()} to {longitudes.max()}"
    )


    # Create interpolators for every hour

    u_interp = []
    v_interp = []


    for t in range(len(times)):

        u_interp.append(
            RegularGridInterpolator(
                (latitudes, longitudes),
                u[t],
                method="linear",
                bounds_error=False,
                fill_value=None,
            )
        )

        v_interp.append(
            RegularGridInterpolator(
                (latitudes, longitudes),
                v[t],
                method="linear",
                bounds_error=False,
                fill_value=None,
            )
        )


    return {
        "times": times,
        "u_interp": u_interp,
        "v_interp": v_interp,
    }



# ============================================================
# ERA5 WIND AT LOCATION
# ============================================================

def get_uv(weather, time_index, lat, lon):

    point = np.array([[lat, lon]])

    u = float(
        weather["u_interp"][time_index](point)[0]
    )

    v = float(
        weather["v_interp"][time_index](point)[0]
    )

    return u, v

#======================================
# Helper for Simulation Driver
#=======================================

def configure_grid(node_lats, stage_lons):

    global NODE_COORDS

    NODE_COORDS = {}

    for s in range(NUM_STAGES):
        for i in range(len(node_lats)):
            NODE_COORDS[f"S{s}_N{i}"] = (
                node_lats[i],
                stage_lons[s],
            )

# ============================================================
# ROUTE COST UNDER ERA5
# ============================================================

def simulate_route(
        route,
        weather,
        arc_details,
        seed,
        policy
):

    cost_function = COST_FUNCTIONS[ACTIVE_COST_FN]

    total_cost = 0.0


    for stage in range(NUM_STAGES):

        from_node = route[stage]
        to_node   = route[stage + 1]

        if from_node == "START" or to_node == "END":
            continue


        # Stage -> day
        day = pd.Timestamp(
            STAGE_DATES[stage]
        )


        # 24 hourly samples

        hourly_costs = []
        hourly_u = []
        hourly_v = []


        for h in range(24):

            timestamp = day + pd.Timedelta(hours=h)


            # locate exact ERA5 hour

            idx = np.where(
                weather["times"] == timestamp
            )[0]


            if len(idx) == 0:
                raise ValueError(
                    f"Missing ERA5 time {timestamp}"
                )


            time_idx = idx[0]


            # Coordinates

            if from_node not in NODE_COORDS:
                print("Missing node:", from_node)
                print("Available nodes:", list(NODE_COORDS.keys())[:20])
                raise KeyError(from_node)

            lat_a, lon_a = NODE_COORDS[from_node]
            lat_b, lon_b = NODE_COORDS[to_node]


            u_a, v_a = get_uv(
                weather,
                time_idx,
                lat_a,
                lon_a
            )


            u_b, v_b = get_uv(
                weather,
                time_idx,
                lat_b,
                lon_b
            )


            c = cost_function(
                u_a,
                v_a,
                u_b,
                v_b
            )


            hourly_costs.append(c)

            hourly_u.append(
                (u_a + u_b) / 2
            )

            hourly_v.append(
                (v_a + v_b) / 2
            )


        # Average daily realised cost

        stage_cost = np.mean(hourly_costs)

        total_cost += stage_cost


        arc_details.append(
            {
                "seed": seed,
                "policy": policy,
                "stage": stage,
                "date": str(day.date()),
                "from": from_node,
                "to": to_node,
                "mean_u": np.mean(hourly_u),
                "mean_v": np.mean(hourly_v),
                "stage_cost": stage_cost,
            }
        )


    return total_cost



# ============================================================
# MAIN SIMULATION
# ============================================================

def run_simulation():

    print("\nLoading saved policy routes...")

    print("Loading from:", RESULTS_FILE)
    print("Exists:", os.path.exists(RESULTS_FILE))

    with open(RESULTS_FILE, "rb") as f:
        results = pickle.load(f)

    print(f"Loaded {len(results)} saved results")

    # Use one set of policy routes only
    # (actual weather is deterministic, so seeds are unnecessary)
    r = results[0]

    weather = load_era5()

    results_output = []
    arc_output = []

    print("\nRunning ERA5 simulations...")

    for policy in POLICIES:

        print(f"Simulating {policy}...")

        # Extract saved forecast route
        forecast_cost, route, stage_costs = r[policy]

        actual_cost = simulate_route(
            route,
            weather,
            arc_output,
            seed=None,
            policy=policy
        )

        results_output.append(
            {
                "policy": policy,
                "forecast_cost": forecast_cost,
                "actual_cost": actual_cost,
                "route": route,
            }
        )


    # ========================================================
    # SAVE
    # ========================================================

    results_df = pd.DataFrame(results_output)

    arc_df = pd.DataFrame(arc_output)


    results_df.to_csv(
        "actual_weather_results.csv",
        index=False
    )


    arc_df.to_csv(
        "actual_weather_arc_details.csv",
        index=False
    )


    print("\nFinished.")

    print(
        "\nSaved:"
        "\n  actual_weather_results.csv"
        "\n  actual_weather_arc_details.csv"
    )


    print("\nSummary:")
    print(
        results_df.groupby("policy")["actual_cost"]
        .agg(["mean","std"])
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print("="*70)
    print("ACTUAL WEATHER ROUTE SIMULATION")
    print("="*70)

    print(
        f"Active cost function: {ACTIVE_COST_FN}"
    )

    run_simulation()