Overview
This folder contains all Python code used to produce the experiments, results, and figures in for the stochastic work on this project. The codebase implements a rolling-horizon Model Predictive Control (MPC) framework for routing wind-assisted cargo vessels across the North Atlantic, comparing eight routing policies under stochastic NWP forecast uncertainty.

The experiments progress from fully synthetic settings (no external data) through to real PredictWind GRIB forecasts evaluated against ERA5 reanalysis weather.

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

Dependencies

All experiments use standard scientific Python packages. Install with:
bash
pip install numpy pandas matplotlib scipy cfgrib xarray plotly eccodes
Package	Purpose
numpy	Numerical computation, cost matrices
pandas	Results tables, CSV output
matplotlib	Static plots and figures
scipy	Interpolation (RegularGridInterpolator)
cfgrib / eccodes	Reading GRIB weather forecast files
xarray	ERA5 NetCDF/GRIB data handling
plotly	Interactive network visualisation

Data Requirements
PredictWind GRIB Forecasts
Real-NWP experiments require a PredictWind/ directory containing GRIB files named in PredictWind's standard format:
PredictWind/
  ECMWF_<date>_<area>.grb
  GFS_<date>_<area>.grb
  ICON_<date>_<area>.grb
  ...

These are potentially provided in this folder or have been provided elsewhere due to magnitude of the file.

ERA5 Reanalysis

ActualWeatherSimulation.py and benchmark_runner2.py require an ERA5 GRIB file with u10 and v10 wind components. Update the ERA5_FILE path constant in each script to point to your local copy - there is one GRIB file attached which was used previously but would recommend re-downloading based on the time and area you are using from Copernicus Weather using ECMWF.

Running the Experiments
1 — Synthetic Policy Comparison

Tests all eight policies on a fully synthetic layered DAG (no GRIB data needed).

bash
python policy_tester.py

Key parameters in policy_tester.py:

Parameter	Default	Description
SEEDS	100	Number of Monte Carlo seeds
BASE_SEED	100	Random seed base

Outputs: bar charts and box plots of hindsight gap (Δ^π) per policy, saved to Plots/.

2 - Synthetic Parameter Sweep
Sweeps forecast noise and persistence to map policy performance across the uncertainty landscape.

3 — Real NWP Forecast Experiment
Runs all eight policies using real PredictWind GRIB forecasts over the North Atlantic.

bash
python experiments_gribs.py

Key parameters in experiments_gribs.py:

Parameter	Default	Description
SEEDS	100	Number of Monte Carlo realisations
LATERAL_PENALTY_SCALE	0.5	Penalty for lateral node jumps

Requires: PredictWind/ directory with Atlantic GRIB files.
Outputs: policy cost distributions, hindsight gap plots, route visualisations.

4 — Full Atlantic Benchmark Sweep

Sweeps lateral penalties, cost measures, and lat/lon grid corridors. Saves one result folder per combination.

bash
python benchmark_runner2.py

Key parameters in benchmark_runner2.py:

Parameter	Default	Description
LATERAL_PENALTIES	[0.05, 0.1, 0.25, 0.4, 0.6]	Penalty values to sweep
OUTPUT_DIR	"BenchmarkResults"	Results directory
ERA5_FILE	(set this)	Path to your ERA5 GRIB file
PREDICTWIND_DIR	"PredictWind"	Path to GRIB forecasts

Outputs: BenchmarkResults/Benchmark_NNN/ — one JSON results file per benchmark run. Analyse with Benchmark/analyse_benchmark.py.

5 - Verification Results
All code relating to the testing of the PredictWind Forecasts are available in this folder.

6 — Greedy Diagnosis Experiments

Self-contained synthetic traps demonstrating greedy's structural weakness against the adaptive policy. All scripts in GreedyDiagnosis/ are independent and require no GRIB data.

bash
cd GreedyDiagnosis
python synthetic_trap_greedy.py   # Static-field trap
python synthetic_trap2.py         # Honeypot + switching-wind traps
python diagnose_greedy.py         # Per-stage decision trac

7 — Interactive Network Visualisation
bash
python interactive_network.py

Opens a Plotly dashboard showing the adaptive policy's routing decisions across forecast stages. Requires policy_grib.py and a valid PredictWind/ directory.

Future Work
 - Recommendation is to use the PredictWind forecasting and policies defined in this code to be used in the Julia Code to simulate across the Atlantic.
 - If further testing is necessary before moving to a larger grid - adaptation of dates, grid sizes or benchmark can also be made in this code here.
 - Once implemented in Julia it is key to test other key convergence metrics and then work out the fuel integration into the model i.e. variable fuel work (some work has been done in the main folder for this on other policies)