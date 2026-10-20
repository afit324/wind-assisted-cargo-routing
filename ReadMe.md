# Folders

### Stochastic Routing (`StochasticWork/`)
This folder contains all Python code used to produce the experiments, results, and figures for the stochastic work on this project. The codebase implements a rolling-horizon Model Predictive Control (MPC) framework for routing wind-assisted cargo vessels across the North Atlantic, comparing eight routing policies under stochastic NWP forecast uncertainty.

The experiments progress from fully synthetic settings (no external data) through to real PredictWind GRIB forecasts evaluated against ERA5 reanalysis weather.

```text
StochasticWork/
│
├── policy.py                   # Synthetic MPC framework (no GRIB) — core policy logic
├── policy_grib.py              # GRIB-based MPC framework — loads real PredictWind forecasts
│
├── policy_tester.py            # Experiment: synthetic policy comparison (policy.py)
├── policy_sweep.py             # Experiment: parameter sweep over noise & forecast persistence
├── experiments_gribs.py        # Experiment: real NWP forecast policy comparison (policy_grib.py)
│
├── SimulationDriver.py         # Driver: loops over lat/lon grids and cost measures (NOT IN USE)
├── benchmark_runner2.py        # Driver: full Atlantic corridor benchmark sweep 
├── ActualWeatherSimulation.py  # Experiment: realise forecast routes under ERA5 actual weather
│
├── forecast_verification.py    # Forecast verification — MAE/RMSE/bias vs ERA5 per model
├── interactive_network.py      # Plotly visualisation of adaptive routing decisions
├── fixing_plot.py              # Utility: post-hoc plot corrections
│
├── GreedyDiagnosis/
│   ├── diagnose_greedy.py      # Trace per-stage decisions for one seed (GRIB-based)
│   ├── synthetic_trap_greedy.py # Synthetic trap: static wind field exposing greedy weakness
│   ├── synthetic_trap2.py      # Synthetic traps: honeypot + switching wind fields
│   ├── trap_visualisations.py  # Visualisations for trap experiments
│   └── visualise_greedy.py     # Additional greedy diagnosis plots
│
├── Benchmark/                  # Output directory — one sub-folder per benchmark run
│   └── benchmark_log.txt       # Log of all benchmark runs
│
├── VerificationResults/        # Output directory — forecast verification plots & tables
├── Plots/                      # Output directory — general plots
│
├── GribFileForActualWeatherSim1  # ERA5 GRIB file (u10/v10 wind components)
├── forecast_policy_routes.pkl    # Saved forecast-policy routes (used by ActualWeatherSimulation.py)
├── actual_weather_results.csv    # Output: policy costs under realised ERA5 weather
└── actual_weather_arc_details.csv # Output: arc-level cost breakdown under ERA5
```

For dijstra/deterministic routing - 

inputs/ - Source data used by the routing and experiment code.
    gfs/ - GFS forecast runs, one folder per cycle named gfs_top5_<tag>_<yyyymmdd>_00z.
    ClimatologyOceanData/
        route/ - Monthly-batched 6-hourly historical wind files (YYYY-MM.grib), used by the climatology DP and the archive wind source.
        hourly_reference/ - Hourly 2025 archive (YYYY-MM-hourly.grib + manifest.jld2), used by the verification tests.
    Polars/ - RISE polar CSV files, one per power level (RisePol<P>kW.csv).
    OceanData_split_fixed/ - Per-timestamp regional GRIB files (YYYYMMDD_HHMM.grib), used by the 2025 baseline for verification only.
OceanData/ - GSHHG land shapefile (GSHHS_i_L1.shp) for ocean-mask construction. ERA5 files downloaded on demand by the default wind loader land here too.
outputs/ - Everything the code computes and persists.
    runs/ - Route JLD2s (hindsight, committed, rolling, plus climatology variants), per-month climatology blends, and the per-month climatology_by_month.jld2 archive.
    tables/ - legs_<mode>_<method>_<tag>.csv, run_status.csv, selected/failed windows.
    convergence/ - Grid-step, radius, energy-bin, and heading-ablation study outputs.
    verification/ - Baseline-vs-improved, zero-wind oracle, and independent baseline results.
    analysis_report/ - Part A tables + plots (penalties, summary, paired, energy vs budget, power vs wind, progress plots).
        energy_wind/ - Sampled route CSVs for the interactive explorer.
    climatology_match/ - Distributional scores of archived days vs the monthly climatology, and validated window scores.
cadet_generic/ - RISE shared library for polar access.
map/ - HTML visualisations of routes. Open with the Live Server extension in VS Code (CORS).
sanity_<timestamp>/ - Scratch folder created by manual sanity runs; not read by any code, safe to delete.

# Files

create_polar.jl - Creates polars from the RISE shared library. Naming convention: "RisePol<P>kW.csv" where <P> is the power in kW.

download_all_wind_data.jl - Downloads wind data from the ERA5 database for a specified period. Files are named "$(time)-$(new_day)$(word_month)$(new_year).grib" (e.g. "1200-1-Nov2024.grib") and go to OceanData/.

download_gfs_rolling_series.sh - Downloads consecutive GFS forecast cycles for a given start date, one folder per cycle under inputs/gfs/gfs_top5_<tag>_<yyyymmdd>_00z. Called by route_experiments.jl when RUN.download_missing = true.

great_circle.jl - Great-circle path helpers.

plotting_polars.py - Polar visualisation scripts.

polars.jl - Wrappers that pull polar data from the RISE shared library.

routing.jl - The main solver: time- and energy-aware Dijkstra with radius bands, bilinear UV interpolation, stacked polar LUTs, and per-edge arc sampling. Also owns the wind-source registry (synthetic / GFS / hourly reference / climatology archive / per-timestamp ERA5), the epoch registry, and the loader dispatcher. Entry point: findDijkstraRoutefinal.

route_simulator.jl - Route steppers and interactive explorers. Builds RouteStepper2 from the solver output, provides wind reconstruction at an arbitrary elapsed time (WindCache / get_wind), and launches the convergence explorer (launch_simulator) and the forecast-vs-actual explorer (launch_forecast_actual_explorer).

climatology_routing.jl - Wind-rose climatology per grid cell, expected edge costs from a rose, fixed-power 2D backward DP, energy-aware 3D backward DP, climatology blend cache (build / save / load / set / clear), and the isochrone/route explorers for climatology runs.

climatology_experiments.jl - Forecast, replay, rolling-replan, and committed-run drivers (run_forecast_route, replay_route_real_wind, run_rolling_replan, run_single_forecast_committed, run_hindsight_route, plus the save/load layer for all of them). Also hosts the per-month climatology archive (build_climatologies_by_month, save/load_climatologies_by_month, month_from_filename) and the climatology-match scoring + validation (pooled_climatology_rose, load_month_days, jensen_shannon_distance, wasserstein1_speed_marginal, validate_climatology_match_windows).

route_experiments.jl - Experiments, analysis, and run drivers. Defines the top-level RUN config, the pipeline (run_pipeline) that runs the five methods (hindsight, committed, rolling, committed_clim, rolling_clim) for each selected window, the convergence study (run_convergence), the verification tests (run_verification_all / run_verification_variable / run_verification_full_estimate), the saved-run analysis (run_analysis_part_a / run_part_b), the sampling step for the interactive explorer (run_sampling), and the climatology-match driver (run_climatology_match). Ends with a bare main() call — running this file starts the pipeline.

read_write_data.jl - Read/write for polars, GRIBs, geojsons, and route CSVs. Also hosts getWindData and the .env credential loading.

times_and_dates.jl - Date/time arithmetic helpers (addSeconds, roundTime, monthWord, timeString).

wind.jl - Wind-field loading, bilinear interpolation, and per-cell (speed, direction) lookup. Also hosts twa_and_boat_speed and the polar lookup used by both solvers.

currents.jl - Extracts ocean currents from netCDF files.

discritise_domain.jl - Discretises a geographic bounding box into grid_points.

geo_helpers.jl - Small geographic helpers (getDistance, getBoatAngle).

plots.jl - Plotting utilities.

route_checker.jl - Independent route checker (simulated or against real wind) used to cross-validate solver output.

route_ship.jl - Grid/mask builder (build_ocean_mask_v2, _build_grid_and_mask), endpoint checks, and route-stat helpers used by every driver.

greedy_route.jl - Optional greedy routing comparison. Included lazily when RUN.greedy = true.

convert_to_GPX.py - Converts a route CSV to GPX for OpenCPN.


# Wind-Assisted Cargo Vessel Routing — Handover & Setup

Time- and energy-aware weather routing for cargo vessels. This project finds the fastest route between two points while respecting a hard energy budget, minimum boat speeds, land masks, and discrete power levels. 

The repository contains **source code only**. You must perform the following setup steps to populate the required data before running the pipeline.

---

## 1. Julia Environment
Instantiate the environment to install all required packages:
```bash
julia --project -e 'using Pkg; Pkg.instantiate()'
```
*Note for headless environments (e.g., servers without a display):* You must set `export GKSwstype=100` before running, otherwise the plotting libraries will crash.

---

## 2. API Credentials & Downloading ERA5 Wind Data
The code uses the Copernicus Climate Data Store (CDS) to download ERA5 data. 
1. Create a `.env` file in the root directory:
```text
CDSAPI_KEY = your_personal_key_here
CDSAPI_URL = your_api_url_here
```


## 3. Downloading Wind Data (ERA5)

The code supports two different ways of downloading ERA5 data, depending on your needs. Both methods automatically calculate the exact bounding box (`route_area`) needed to cover the route corridor plus padding.

**A. Bulk Download (For Climatology & Backtesting)**
To run the historical climatology blend or backtests, you need decades of data. Use the `batched_downloader.jl` script.
*   Open `wind_download.jl`.
*   Scroll to the bottom where `bulk_download_climatology_batched(...)` is called.
*   Set your `start_location`, `end_location`, `padding`, and `step` (must match your routing grid).
*   Set `start_year` and `end_year`.
*   Set 'out_dir' to "inputs/ClimatologyOceanData/route/"
*   Define `times_per_day` (e.g., `["00:00", "06:00", "12:00", "18:00"]` for 6-hourly).
*   Run it: `julia --project wind_download.jl`
*   **Output:** Saves `YYYY-MM.grib` files directly into `inputs/ClimatologyOceanData/route/`. It batches one request per month to speed up CDS processing.

**B. Hourly Reference (For Resolution Testing)**
If you need to test the difference between 6-hourly and hourly data, run the `download_hourly_reference(...)` function inside `wind_download.jl`. 
*   Set 'out_dir' to "inputs/ClimatologyOceanData/hourly_reference/"
*   Set 'year' to year wanted
*   **Output:** Saves `YYYY-MM-hourly.grib` files into `inputs/ClimatologyOceanData/hourly_reference/`.

**C. Automatic Per-Timestamp Downloads**
If a specific `.grib` file is missing from your archive when the default solver requests it, the code will automatically call `getWindData()` to fetch just that timestamp and save it into `OceanData/`.

---

## 4. Land Shapefile (GSHHG)
The grid builder requires the intermediate-resolution GSHHG shapefile.
1. Download from NOAA: https://www.ngdc.noaa.gov/mgg/shorelines/gshhs.html
2. Unzip and place `GSHHS_i_L1.shp` (and its sibling files `.shx`, `.dbf`, `.prj`) into the `OceanData/` folder.

---

## 5. GFS Forecast Data
You need local GFS forecast cycles for the Committed and Rolling methods.
1. Open a terminal (Git Bash) in the project root.
2. Run the download script for your target start date (e.g., `20221003`):
```bash
./download_gfs_rolling_series.sh 20221003 20 00 336 inputs/gfs/gfs_top5_20221003 12
```
*   This downloads 20 consecutive daily cycles of 00Z GFS out to 336 hours.
*   Data is saved to `inputs/gfs/gfs_top5_<tag>_<yyyymmdd>_00z/`.
*   *If a window's GFS folder is missing, the pipeline will skip that date.*

---

## 6. How to Configure the Pipeline
All configurations are handled in the `route_experiments.jl` file. Open it in your editor and look at the top and bottom of the file.

**A. Changing Start/End Locations**
Around line 45, change the coordinates (order is `lon, lat`):
```julia
const START_LOCATION = (-94.0, 28.5)  # e.g., Houston
const END_LOCATION   = (-5.5, 49.5)   # e.g., English Channel
```

**B. Changing Target Dates (Forecasts)**
Look for `RUN_TAGS`:
```julia
const RUN_TAGS = ["20221003", "20221215", "20231106", "20240616", "20240822"]
```
*You must download the GFS folders for any tags you put here.*

**C. Changing Data Folders**
If you move your data, update the paths in `route_experiments.jl`:
```julia
const GFS_DIR              = _fwd(joinpath(INPUT_DIR, "gfs"))
const HISTORICAL_DIR       = joinpath(CLIMATOLOGY_DATA_DIR, "route")
const HOURLY_REFERENCE_DIR = joinpath(CLIMATOLOGY_DATA_DIR, "hourly_reference")
```

**D. Changing from 6-Hourly to Hourly Wind Data**
The solver supports different wind sources. By default, the pipeline uses the 6-hourly climatology archive.
*   **To use 6-hourly (Default):** `with_archive()` calls `set_climatology_archive_source!()`, which reads from `HISTORICAL_DIR` and snaps times to 6-hour intervals (21600 seconds).
*   **To use Hourly:** If you have hourly data in `HOURLY_REFERENCE_DIR`, you must modify `reset_sources!()` or `with_archive()` in `route_experiments.jl` to call:
    ```julia
    set_hourly_reference_source!(dir = HOURLY_REFERENCE_DIR)
    ```
    This tells the solver to snap times to 1-hour intervals (3600 seconds) and read from the hourly folder.

**E. How to Run the Pipeline**
Go to the bottom of `route_experiments.jl` to the `const RUN` block.
```julia
const RUN = (
    pipeline            = true,
    only_tag            = "20221003",  # Set to nothing to run all RUN_TAGS
    run_c400            = true,        # Constant 400kW
    run_variable        = false,       # Variable power
    ...
)
```

---

### Resume System
The pipeline saves its progress. If it crashes or you stop it, **just restart it**. It checks for the existence of `outputs/tables/legs_<mode>_<method>_<tag>.csv` and skips completed work. To force a re-run, simply delete the corresponding `legs_*.csv` file.

# Conventions

- function names are CamelCase
- variable names are snake_case
- file names are snake_case
- order is lon, lat for geographical points

Unit Conventions:
  - wind speed is in m/s
  - wind direction is in degrees, it is the direction wind is coming FROM relative to north
  - current speed is in m/s
  - current direction is in degrees, it is the direction the current is going TO relative to north
