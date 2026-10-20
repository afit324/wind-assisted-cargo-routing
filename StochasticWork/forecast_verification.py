"""
forecast_verification.py
========================
Forecast verification analysis comparing PredictWind model forecasts
against ERA5 reanalysis for Atlantic routing corridor.

Outputs (saved to VerificationResults/):
  - metrics_table.csv          : MAE/RMSE/bias per model/variable/lead time
  - skill_decay.png            : MAE/RMSE/bias vs lead time per model
  - spatial_error_24h.png      : Gridded error maps at 24h lead time
  - spatial_error_all.png      : Error maps for all lead times
  - regime_error.png           : Error in volatile vs steady conditions
  - model_ranking.csv          : Ranked summary per variable/lead time
  - decision_check.csv         : Did forecast pick same corridor as ERA5?
  - coverage_report.csv        : Which model/date/lead combos are missing
  - verification_summary.txt   : Written summary of findings

Usage:
    python forecast_verification.py

CHANGE LOG (vs original):
  - LEAD_TIMES extended from [0,6,12,24,48,72] out to 168h (7 days),
    at 24h steps beyond 72h. (Capped at 168h rather than 240h because
    not every model's GRIB files currently extend past 168h -- see
    ENSEMBLE_NAME note below. Bump back up to 240h once all five
    models have long-range files and the coverage report confirms it.)
  - Added a coverage/sanity check after build_records() that reports,
    per model x init_date, which lead times are missing data. This
    is meant to catch silent gaps (e.g. a model whose GRIB file was
    never found for most dates) before they quietly bias the
    downstream metrics -- this is what happened with ICON in the
    original run, which ended up scored on only 1 of 8 init dates.
  - plot_skill_decay() x-axis ticks thinned out for the longer
    lead-time range so labels don't overlap.
  - Added an ENSEMBLE_MEAN "model": at every (init_date, lead_h, lat,
    lon) where at least MIN_ENSEMBLE_MEMBERS of the five source
    models have data, their u10/v10 forecasts are averaged and wind
    speed/direction re-derived from that averaged vector (not by
    averaging speed and direction directly -- averaging direction
    naively is wrong near 0/360 wraparound, and averaging speed
    directly ignores cancellation between differing directions).
    This ensemble mean is carried through every downstream metric,
    plot, and the written summary exactly like any other model, so
    you can see whether simple multi-model averaging beats the best
    individual source.
"""

import os
import sys
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as mgridspec
import matplotlib.ticker as ticker
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import circmean, circstd
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

try:
    import cfgrib
    import xarray as xr
except ImportError:
    print("Install: pip install cfgrib eccodes xarray")
    sys.exit(1)

# ============================================================
# CONFIGURATION
# ============================================================

PREDICTWIND_DIR = "PredictWind"

ERA5_FILE = Path(
    "/Users/thomasfairweather/Downloads/ActualWeatherSimAtlanticJul_Aug"
)

OUTPUT_DIR = Path("VerificationResults")
OUTPUT_DIR.mkdir(exist_ok=True)

STAGE_DATES = [
    "2026-08-01", "2026-08-02", "2026-08-03", "2026-08-04",
    "2026-08-05", "2026-08-06", "2026-08-07", "2026-08-08",
]

ALL_KNOWN_MODELS = ["PWG", "GFS", "PWE", "ECMWF_WRCTPaG", "ICON"]

EXCLUDED_MODELS = ["ICON"]

MODELS = [m for m in ALL_KNOWN_MODELS if m not in EXCLUDED_MODELS]

LATITUDE_GRIDS = [
    [20, 30, 40, 50, 60],
    [20, 30, 40, 50, 60],
    [20, 30, 40, 50, 60],
    [20, 30, 40, 50, 60],
    [20, 30, 40, 50, 60],
]

LONGITUDE_GRIDS = [
    [-95, -90, -85, -80, -75, -70, -65, -60],
    [-90, -85, -80, -75, -70, -65, -60, -55],
    [-85, -80, -75, -70, -65, -60, -55, -50],
    [-80, -75, -70, -65, -60, -55, -50, -45],
    [-75, -70, -65, -60, -55, -50, -45, -40],
]

_eval_pts = set()
for lats, lons in zip(LATITUDE_GRIDS, LONGITUDE_GRIDS):
    for lat in lats:
        for lon in lons:
            _eval_pts.add((float(lat), float(lon)))
EVAL_POINTS = sorted(_eval_pts)

LEAD_TIMES = [
    0, 6, 12, 24, 48, 72,
    96, 120, 144, 168,
]

REGIONS = {
    "Western (lon < −70°)" : lambda lat, lon: lon < -70,
    "Central (−70° to −55°)": lambda lat, lon: -70 <= lon < -55,
    "Eastern (lon ≥ −55°)" : lambda lat, lon: lon >= -55,
}

VOLATILE_THRESHOLD_MS = 3.0

MODEL_COLORS = {
    "PWG"          : "#3a7fc1",   # steel blue
    "GFS"          : "#4aab6d",   # mid green
    "PWE"          : "#9b59b6",   # violet
    "ECMWF_WRCTPaG": "#e6873a",   # warm orange
    "ICON"         : "#c0392b",   # crimson
    "ENSEMBLE_MEAN": "#1a1a1a",   # near-black (dashed)
}

# Short display labels for plot axes/legends
MODEL_LABELS = {
    "ECMWF_WRCTPaG": "ECMWF",
    "ENSEMBLE_MEAN": "Ens. Mean",
}

ENSEMBLE_NAME = "ENSEMBLE_MEAN"
MIN_ENSEMBLE_MEMBERS = 3
ANALYSIS_MODELS = MODELS + [ENSEMBLE_NAME]

# ============================================================
# SHARED PLOT STYLE
# ============================================================

def model_label(model):
    """Return a short display name for plot axes and legends."""
    return MODEL_LABELS.get(model, model)


def apply_plot_style():
    """Apply consistent publication-quality rcParams."""
    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rcParams.update({
        'font.family'        : 'DejaVu Sans',
        'font.size'          : 11,
        'axes.titlesize'     : 12,
        'axes.titleweight'   : 'bold',
        'axes.labelsize'     : 11,
        'axes.labelcolor'    : '#222222',
        'axes.spines.top'    : False,
        'axes.spines.right'  : False,
        'axes.spines.left'   : True,
        'axes.spines.bottom' : True,
        'axes.grid'          : True,
        'axes.grid.axis'     : 'y',
        'grid.alpha'         : 0.22,
        'grid.linewidth'     : 0.7,
        'grid.color'         : '#b0b0b0',
        'xtick.labelsize'    : 10,
        'ytick.labelsize'    : 10,
        'xtick.color'        : '#333333',
        'ytick.color'        : '#555555',
        'figure.facecolor'   : 'white',
        'axes.facecolor'     : '#fafafa',
        'legend.framealpha'  : 0.92,
        'legend.edgecolor'   : '#cccccc',
        'legend.fontsize'    : 9,
        'figure.dpi'         : 300,
    })


def style_ax(ax, xlabel=None, ylabel=None, title=None, grid_axis='y'):
    """Apply consistent spine/tick styling to a single axes."""
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(axis='both', which='both', length=3, color='#aaaaaa')
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=6)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=6)
    if title:
        ax.set_title(title, pad=8)
    ax.grid(axis=grid_axis, alpha=0.22, linewidth=0.7, color='#b0b0b0')

# ============================================================
# FILE FINDING  (mirrors policy_grib logic)
# ============================================================

TARGET_AREA = "69N_9S_12E_-114W"


def _parse_area(fname):
    m = re.search(r'(\d+)N_(-?\d+)S_(-?\d+)E_(-?\d+)W', fname)
    if not m:
        return None
    lat_max =  float(m.group(1))
    lat_min_raw = float(m.group(2))
    lat_min = lat_min_raw if lat_min_raw < 0 else -lat_min_raw
    e_val   =  float(m.group(3))
    w_raw   =  float(m.group(4))
    w_signed = -w_raw if w_raw >= 0 else w_raw
    lon_lo  = min(e_val, w_signed)
    lon_hi  = max(e_val, w_signed)
    return lat_min, lat_max, lon_lo, lon_hi


def _covers_eval_points(fname, buffer=3.0):
    bounds = _parse_area(fname)
    if bounds is None:
        return True
    lat_min, lat_max, lon_lo, lon_hi = bounds
    for lat, lon in EVAL_POINTS:
        if not (lat_min - buffer <= lat <= lat_max + buffer):
            return False
        if not (lon_lo - buffer <= lon <= lon_hi + buffer):
            return False
    return True


def find_forecast_file(date_str, model):
    folder = Path(PREDICTWIND_DIR) / date_str
    if not folder.exists():
        print(f"    {model}: folder {folder} does not exist — skipping")
        return None
    candidates = [f for f in sorted(os.listdir(folder)) if f.endswith(".grb")]
    model_candidates = [f for f in candidates if f"_{model}_" in f]
    if not model_candidates:
        print(f"    {model}: no .grb file matching '_{model}_' in {folder} — skipping")
        return None
    for fname in model_candidates:
        if not _covers_eval_points(fname):
            print(f"    {model}: {fname} does not cover eval-point bounds — skipping")
            continue
        return str(folder / fname)
    return None


# ============================================================
# GRIB LOADING
# ============================================================

def load_forecast_timeseries(filepath):
    print(f"    Loading {Path(filepath).name}...", end="", flush=True)

    result = {}

    try:
        datasets = cfgrib.open_datasets(filepath)
    except Exception as e:
        print(f" FAILED ({e})")
        return result

    u10_ds = v10_ds = None
    for ds in datasets:
        if "u10" in ds and u10_ds is None:
            u10_ds = ds
        if "v10" in ds and v10_ds is None:
            v10_ds = ds
        if u10_ds is not None and v10_ds is not None:
            break

    if u10_ds is None or v10_ds is None:
        print(" no u10/v10")
        return result

    try:
        lats = u10_ds["u10"].latitude.values
        lons = u10_ds["u10"].longitude.values

        if lats[0] > lats[-1]:
            lats = lats[::-1]
            lat_flip = True
        else:
            lat_flip = False

        uses_360 = lons.max() > 180

        u10_var = u10_ds["u10"]
        v10_var = v10_ds["v10"]

        if "step" not in u10_var.dims:
            steps_h = [0]
            u_vals  = u10_var.values[np.newaxis]
            v_vals  = v10_var.values[np.newaxis]
        else:
            steps    = u10_var.step.values
            steps_h  = [
                int(s / np.timedelta64(1, "h"))
                for s in steps
            ]
            u_vals   = u10_var.values
            v_vals   = v10_var.values

        if lat_flip:
            u_vals = u_vals[:, ::-1, :]
            v_vals = v_vals[:, ::-1, :]

        max_available_step = max(steps_h) if steps_h else None

        for idx, lead_h in enumerate(steps_h):
            if lead_h not in LEAD_TIMES:
                continue
            u_interp = RegularGridInterpolator(
                (lats, lons), u_vals[idx],
                method="linear", bounds_error=False, fill_value=np.nan
            )
            v_interp = RegularGridInterpolator(
                (lats, lons), v_vals[idx],
                method="linear", bounds_error=False, fill_value=np.nan
            )
            result[lead_h] = (u_interp, v_interp, uses_360)

        missing_long_range = [
            lt for lt in LEAD_TIMES
            if lt > 72 and lt not in result
        ]
        if missing_long_range and max_available_step is not None:
            print(
                f" steps={sorted(result.keys())} "
                f"(file's longest step is {max_available_step}h — "
                f"missing {missing_long_range} from LEAD_TIMES) OK"
            )
        else:
            print(f" steps={sorted(result.keys())} OK")

    except Exception as e:
        print(f" ERROR ({e})")

    return result


def query_interp(u_interp, v_interp, uses_360, lat, lon):
    lon_q = lon + 360 if (uses_360 and lon < 0) else lon
    pt    = np.array([[lat, lon_q]])
    u     = float(u_interp(pt)[0])
    v     = float(v_interp(pt)[0])
    return u, v


# ============================================================
# ERA5 LOADING
# ============================================================

def load_era5():
    print(f"\nLoading ERA5 from {ERA5_FILE}...")

    datasets = cfgrib.open_datasets(str(ERA5_FILE))
    u10_ds = v10_ds = None
    for ds in datasets:
        if "u10" in ds and u10_ds is None:
            u10_ds = ds
        if "v10" in ds and v10_ds is None:
            v10_ds = ds
        if u10_ds and v10_ds:
            break

    if u10_ds is None:
        raise RuntimeError("ERA5: no u10/v10 found")

    lats     = u10_ds["u10"].latitude.values
    lons     = u10_ds["u10"].longitude.values
    lat_flip = lats[0] > lats[-1]
    if lat_flip:
        lats = lats[::-1]
    uses_360 = lons.max() > 180

    u10_var = u10_ds["u10"]
    v10_var = v10_ds["v10"]

    time_coord = None
    for tc in ["valid_time", "time"]:
        if tc in u10_var.coords:
            time_coord = tc
            break

    if time_coord and u10_var[time_coord].ndim > 0:
        times = pd.to_datetime(u10_var[time_coord].values)
        u_all = u10_var.values
        v_all = v10_var.values
        if u_all.ndim == 2:
            u_all = u_all[np.newaxis]
            v_all = v_all[np.newaxis]
    else:
        t_val = u10_var.coords.get("valid_time",
                u10_var.coords.get("time"))
        times = pd.to_datetime([t_val.values])
        u_all = u10_var.values[np.newaxis]
        v_all = v10_var.values[np.newaxis]

    if lat_flip:
        u_all = u_all[:, ::-1, :]
        v_all = v_all[:, ::-1, :]

    era5 = {}
    for i, ts in enumerate(times):
        u_interp = RegularGridInterpolator(
            (lats, lons), u_all[i],
            method="linear", bounds_error=False, fill_value=np.nan
        )
        v_interp = RegularGridInterpolator(
            (lats, lons), v_all[i],
            method="linear", bounds_error=False, fill_value=np.nan
        )
        era5[ts] = (u_interp, v_interp, uses_360)

    print(f"  ERA5: {len(era5)} timesteps, "
          f"{pd.Timestamp(min(era5)):%Y-%m-%d} to "
          f"{pd.Timestamp(max(era5)):%Y-%m-%d}")

    last_init  = pd.Timestamp(max(STAGE_DATES))
    needed_end = last_init + timedelta(hours=max(LEAD_TIMES))
    era5_end   = pd.Timestamp(max(era5))
    if era5_end < needed_end:
        print(
            f"  WARNING: ERA5 file ends {era5_end:%Y-%m-%d %H:%M}, but "
            f"the last init date ({last_init:%Y-%m-%d}) plus the "
            f"longest lead time ({max(LEAD_TIMES)}h) needs data out "
            f"to {needed_end:%Y-%m-%d %H:%M}."
        )

    if len(era5) > 1:
        sorted_times = sorted(era5.keys())
        deltas_h = sorted(set(
            round((t2 - t1).total_seconds() / 3600, 2)
            for t1, t2 in zip(sorted_times[:-1], sorted_times[1:])
        ))
        print(f"  ERA5 timestep spacing found (hours): {deltas_h}")

    return era5


def query_era5_at(era5, timestamp, lat, lon, max_gap_hours=6.0):
    times = list(era5.keys())
    diffs = [abs((t - timestamp).total_seconds()) for t in times]
    best_idx = int(np.argmin(diffs))
    best     = times[best_idx]

    gap_hours = diffs[best_idx] / 3600.0
    if gap_hours > max_gap_hours:
        return np.nan, np.nan

    u_i, v_i, uses_360 = era5[best]
    return query_interp(u_i, v_i, uses_360, lat, lon)


# ============================================================
# BUILD OBSERVATIONS DATAFRAME
# ============================================================

def build_records(era5):
    records = []

    for date_str in STAGE_DATES:
        init_time = pd.Timestamp(date_str)
        print(f"\n  Date: {date_str}")

        for model in MODELS:
            fpath = find_forecast_file(date_str, model)
            if fpath is None:
                continue

            fcst_data = load_forecast_timeseries(fpath)
            if not fcst_data:
                continue

            for lead_h in LEAD_TIMES:
                if lead_h not in fcst_data:
                    continue

                u_i, v_i, uses_360 = fcst_data[lead_h]
                valid_time          = init_time + timedelta(hours=lead_h)

                for lat, lon in EVAL_POINTS:
                    u_fcst, v_fcst = query_interp(u_i, v_i, uses_360, lat, lon)
                    u_era5, v_era5 = query_era5_at(era5, valid_time, lat, lon)

                    if np.isnan(u_fcst) or np.isnan(u_era5):
                        continue

                    ws_fcst = float(np.sqrt(u_fcst**2 + v_fcst**2))
                    ws_era5 = float(np.sqrt(u_era5**2 + v_era5**2))
                    wd_fcst = float(np.degrees(np.arctan2(u_fcst, v_fcst)) % 360)
                    wd_era5 = float(np.degrees(np.arctan2(u_era5, v_era5)) % 360)

                    region = "Unknown"
                    for rname, rfn in REGIONS.items():
                        if rfn(lat, lon):
                            region = rname
                            break

                    records.append({
                        "init_date"  : date_str,
                        "init_time"  : init_time,
                        "model"      : model,
                        "lead_h"     : lead_h,
                        "valid_time" : valid_time,
                        "lat"        : lat,
                        "lon"        : lon,
                        "region"     : region,
                        "u_fcst"     : u_fcst,
                        "v_fcst"     : v_fcst,
                        "u_era5"     : u_era5,
                        "v_era5"     : v_era5,
                        "ws_fcst"    : ws_fcst,
                        "ws_era5"    : ws_era5,
                        "wd_fcst"    : wd_fcst,
                        "wd_era5"    : wd_era5,
                        "u_err"      : u_fcst - u_era5,
                        "v_err"      : v_fcst - v_era5,
                        "ws_err"     : ws_fcst - ws_era5,
                    })

    return pd.DataFrame(records)


# ============================================================
# COVERAGE / SANITY CHECK
# ============================================================

def build_coverage_report(df):
    rows = []
    expected_leads = sorted(LEAD_TIMES)

    for model in MODELS:
        m_df = df[df["model"] == model]

        for lead_h in expected_leads:
            n_records = len(m_df[m_df["lead_h"] == lead_h])
            n_dates_at_lead = m_df[m_df["lead_h"] == lead_h]["init_date"].nunique()
            rows.append({
                "model": model,
                "lead_h": lead_h,
                "n_records": n_records,
                "n_init_dates_present": n_dates_at_lead,
                "n_init_dates_expected": len(STAGE_DATES),
                "full_coverage": n_dates_at_lead == len(STAGE_DATES),
            })

    cov_df = pd.DataFrame(rows)

    print("\n" + "=" * 60)
    print("COVERAGE CHECK")
    print("=" * 60)
    for model in MODELS:
        m_cov = cov_df[cov_df["model"] == model]
        if m_cov.empty:
            print(f"  {model:<15}: NO DATA AT ALL — check file naming/paths")
            continue
        min_dates = m_cov["n_init_dates_present"].min()
        if min_dates < len(STAGE_DATES):
            bad_leads = m_cov[
                m_cov["n_init_dates_present"] < len(STAGE_DATES)
            ].sort_values("lead_h")
            bad_lead_str = ", ".join(
                f"{int(r.lead_h)}h({int(r.n_init_dates_present)}/{len(STAGE_DATES)})"
                for r in bad_leads.itertuples()
            )
            print(
                f"  {model:<15}: WARNING — incomplete date coverage at "
                f"{len(bad_leads)} of {len(m_cov)} lead times: {bad_lead_str}."
            )
        else:
            print(f"  {model:<15}: OK — full {len(STAGE_DATES)}-date coverage at all leads")
    print("=" * 60)

    return cov_df


# ============================================================
# ENSEMBLE MEAN
# ============================================================

def build_ensemble_records(df, min_models=MIN_ENSEMBLE_MEMBERS):
    if df.empty:
        return pd.DataFrame()

    rows = []
    group_cols = ["init_date", "lead_h", "lat", "lon"]
    for key, g in df.groupby(group_cols):
        n_members = g["model"].nunique()
        if n_members < min_models:
            continue

        u_mean = float(g["u_fcst"].mean())
        v_mean = float(g["v_fcst"].mean())
        u_era5 = float(g["u_era5"].iloc[0])
        v_era5 = float(g["v_era5"].iloc[0])

        ws_fcst = float(np.sqrt(u_mean**2 + v_mean**2))
        ws_era5 = float(np.sqrt(u_era5**2 + v_era5**2))
        wd_fcst = float(np.degrees(np.arctan2(u_mean, v_mean)) % 360)
        wd_era5 = float(np.degrees(np.arctan2(u_era5, v_era5)) % 360)

        rows.append({
            "init_date"  : g["init_date"].iloc[0],
            "init_time"  : g["init_time"].iloc[0],
            "model"      : ENSEMBLE_NAME,
            "lead_h"     : g["lead_h"].iloc[0],
            "valid_time" : g["valid_time"].iloc[0],
            "lat"        : g["lat"].iloc[0],
            "lon"        : g["lon"].iloc[0],
            "region"     : g["region"].iloc[0],
            "u_fcst"     : u_mean,
            "v_fcst"     : v_mean,
            "u_era5"     : u_era5,
            "v_era5"     : v_era5,
            "ws_fcst"    : ws_fcst,
            "ws_era5"    : ws_era5,
            "wd_fcst"    : wd_fcst,
            "wd_era5"    : wd_era5,
            "u_err"      : u_mean - u_era5,
            "v_err"      : v_mean - v_era5,
            "ws_err"     : ws_fcst - ws_era5,
            "n_members"  : n_members,
        })

    ens_df = pd.DataFrame(rows)
    if not ens_df.empty:
        print(
            f"\nBuilt ENSEMBLE_MEAN: {len(ens_df)} records "
            f"(requires >= {min_models} of {len(MODELS)} models present); "
            f"member count per record ranges "
            f"{ens_df['n_members'].min()}-{ens_df['n_members'].max()}"
        )
    else:
        print(
            f"\nWARNING: no (init_date, lead_h, lat, lon) combination had "
            f">= {min_models} models present -- ENSEMBLE_MEAN is empty."
        )
    return ens_df


# ============================================================
# CIRCULAR DIRECTION ERROR
# ============================================================

def circular_diff(a, b):
    d = (np.array(a) - np.array(b)) % 360
    d[d > 180] -= 360
    return d


def add_dir_error(df):
    df["wd_err"] = circular_diff(
        df["wd_fcst"].values, df["wd_era5"].values
    )
    return df


# ============================================================
# REGIME CONDITIONING
# ============================================================

def classify_regime(df, era5):
    regime = []
    for _, row in df.iterrows():
        vt       = row["valid_time"]
        lat, lon = row["lat"], row["lon"]

        ws_now  = np.sqrt(row["u_era5"]**2 + row["v_era5"]**2)
        u_p6, v_p6 = query_era5_at(
            era5, vt + timedelta(hours=6), lat, lon
        )
        if np.isnan(u_p6):
            regime.append("unknown")
            continue
        ws_p6   = float(np.sqrt(u_p6**2 + v_p6**2))
        dws     = abs(ws_p6 - ws_now)

        regime.append("volatile" if dws > VOLATILE_THRESHOLD_MS
                      else "steady")

    df["regime"] = regime
    return df


# ============================================================
# METRICS
# ============================================================

def compute_metrics(df):
    rows = []

    for model in df["model"].unique():
        for lead_h in sorted(df["lead_h"].unique()):
            sub = df[(df["model"] == model) & (df["lead_h"] == lead_h)]
            if len(sub) == 0:
                continue

            for var, err_col in [
                ("u10",         "u_err"),
                ("v10",         "v_err"),
                ("wind_speed",  "ws_err"),
                ("wind_dir",    "wd_err"),
            ]:
                if err_col not in sub.columns:
                    continue
                errs = sub[err_col].dropna().values
                if len(errs) == 0:
                    continue

                rows.append({
                    "model"  : model,
                    "lead_h" : lead_h,
                    "variable": var,
                    "n"      : len(errs),
                    "bias"   : float(np.mean(errs)),
                    "mae"    : float(np.mean(np.abs(errs))),
                    "rmse"   : float(np.sqrt(np.mean(errs**2))),
                })

    return pd.DataFrame(rows)


def compute_metrics_by_region(df):
    rows = []
    for model in df["model"].unique():
        for lead_h in sorted(df["lead_h"].unique()):
            for region in df["region"].unique():
                sub = df[
                    (df["model"] == model) &
                    (df["lead_h"] == lead_h) &
                    (df["region"] == region)
                ]
                if len(sub) == 0:
                    continue
                errs = sub["ws_err"].dropna().values
                if len(errs) == 0:
                    continue
                rows.append({
                    "model"   : model,
                    "lead_h"  : lead_h,
                    "region"  : region,
                    "n"       : len(errs),
                    "bias"    : float(np.mean(errs)),
                    "mae"     : float(np.mean(np.abs(errs))),
                    "rmse"    : float(np.sqrt(np.mean(errs**2))),
                })
    return pd.DataFrame(rows)


def compute_metrics_by_regime(df):
    rows = []
    for model in df["model"].unique():
        for lead_h in sorted(df["lead_h"].unique()):
            for regime in ["volatile", "steady"]:
                sub = df[
                    (df["model"] == model) &
                    (df["lead_h"] == lead_h) &
                    (df["regime"] == regime)
                ]
                if len(sub) == 0:
                    continue
                errs = sub["ws_err"].dropna().values
                if len(errs) == 0:
                    continue
                rows.append({
                    "model"  : model,
                    "lead_h" : lead_h,
                    "regime" : regime,
                    "n"      : len(errs),
                    "bias"   : float(np.mean(errs)),
                    "mae"    : float(np.mean(np.abs(errs))),
                    "rmse"   : float(np.sqrt(np.mean(errs**2))),
                })
    return pd.DataFrame(rows)


# ============================================================
# MODEL RANKING
# ============================================================

def rank_models(metrics_df):
    rows = []
    for var in metrics_df["variable"].unique():
        for lead_h in sorted(metrics_df["lead_h"].unique()):
            sub = metrics_df[
                (metrics_df["variable"] == var) &
                (metrics_df["lead_h"]   == lead_h)
            ].sort_values("rmse")
            for rank, (_, row) in enumerate(sub.iterrows(), 1):
                rows.append({
                    "variable": var,
                    "lead_h"  : lead_h,
                    "rank"    : rank,
                    "model"   : row["model"],
                    "rmse"    : row["rmse"],
                    "mae"     : row["mae"],
                    "bias"    : row["bias"],
                })
    return pd.DataFrame(rows)


# ============================================================
# DECISION-RELEVANT CHECK
# ============================================================

def decision_corridor_check(df):
    rows = []
    check_leads = [lt for lt in LEAD_TIMES if lt >= 24]

    for init_date in df["init_date"].unique():
        for lead_h in check_leads:
            sub = df[
                (df["init_date"] == init_date) &
                (df["lead_h"]    == lead_h)
            ]
            if sub.empty:
                continue

            for lat in sorted(sub["lat"].unique()):
                lat_sub = sub[sub["lat"] == lat]
                if lat_sub.empty:
                    continue

                era5_best_lon = lat_sub.loc[
                    lat_sub["ws_era5"].idxmax(), "lon"
                ]

                for model in lat_sub["model"].unique():
                    m_sub = lat_sub[lat_sub["model"] == model]
                    if m_sub.empty:
                        continue
                    fcst_best_lon = m_sub.loc[
                        m_sub["ws_fcst"].idxmax(), "lon"
                    ]
                    match = (fcst_best_lon == era5_best_lon)

                    rows.append({
                        "init_date"    : init_date,
                        "lead_h"       : lead_h,
                        "lat"          : lat,
                        "model"        : model,
                        "era5_best_lon": era5_best_lon,
                        "fcst_best_lon": fcst_best_lon,
                        "corridor_match": match,
                    })

    return pd.DataFrame(rows)


# ============================================================
# PLOTTING
# ============================================================

def plot_skill_decay(metrics_df):
    """
    Main skill decay plot: MAE / RMSE / bias vs lead time,
    one line per model, one panel per variable.
    Publication-quality, 300 dpi.
    """
    apply_plot_style()

    variables    = metrics_df["variable"].unique()
    n_vars       = len(variables)
    metric_rows  = ["mae", "rmse", "bias"]
    metric_labels = ["MAE", "RMSE", "Bias"]
    metric_units  = {
        "u10"        : "m s⁻¹",
        "v10"        : "m s⁻¹",
        "wind_speed" : "m s⁻¹",
        "wind_dir"   : "°",
    }

    fig, axs = plt.subplots(
        3, n_vars,
        figsize=(4.8 * n_vars, 11),
        sharex=True,
        gridspec_kw={"hspace": 0.38, "wspace": 0.30}
    )
    if n_vars == 1:
        axs = axs.reshape(-1, 1)

    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Forecast Skill Decay by Lead Time\n"
        "Atlantic Routing Corridor — August 2026",
        fontsize=13, fontweight="bold", color="#1a1a1a", y=0.99
    )

    tick_vals = (LEAD_TIMES[::2] if len(LEAD_TIMES) > 8 else LEAD_TIMES)
    if LEAD_TIMES[-1] not in tick_vals:
        tick_vals = list(tick_vals) + [LEAD_TIMES[-1]]

    for col, var in enumerate(variables):
        var_df = metrics_df[metrics_df["variable"] == var]
        unit   = metric_units.get(var, "")

        for row, (metric, label) in enumerate(zip(metric_rows, metric_labels)):
            ax = axs[row, col]

            for model in ANALYSIS_MODELS:
                m_df = var_df[var_df["model"] == model].sort_values("lead_h")
                if m_df.empty:
                    continue
                is_ens = (model == ENSEMBLE_NAME)
                ax.plot(
                    m_df["lead_h"], m_df[metric],
                    marker="o",
                    markersize=4.5,
                    linewidth=2.2 if is_ens else 1.8,
                    linestyle="--" if is_ens else "-",
                    color=MODEL_COLORS.get(model, "#888888"),
                    label=model_label(model),
                    alpha=0.95,
                    zorder=3 if is_ens else 2,
                )

            if metric == "bias":
                ax.axhline(0, color="#888888", linestyle=":",
                           linewidth=1.0, alpha=0.7)

            col_title = var.replace("_", " ").title()
            style_ax(
                ax,
                ylabel=f"{label} ({unit})" if col == 0 else "",
                title=f"{col_title} — {label}" if row == 0 else label,
                grid_axis="y"
            )
            ax.set_xticks(tick_vals)
            ax.tick_params(axis="x", rotation=40, labelsize=9)
            if row == 2:
                ax.set_xlabel("Lead time (h)", labelpad=6)

            if row == 0 and col == 0:
                leg = ax.legend(
                    fontsize=8.5, loc="upper left",
                    framealpha=0.92, edgecolor="#cccccc"
                )

    fig.text(
        0.99, 0.005,
        "ERA5 reanalysis reference  |  ±1 init-date ensemble",
        ha="right", va="bottom",
        fontsize=7.5, color="#999999", style="italic"
    )

    plt.savefig(
        OUTPUT_DIR / "skill_decay.png",
        dpi=300, bbox_inches="tight", facecolor="white"
    )
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'skill_decay.png'}")


def plot_spatial_error(df, lead_h=24, save_suffix="24h"):
    """
    Gridded error map: mean absolute wind speed error at each
    eval point for a given lead time. Publication-quality, 300 dpi.
    """
    apply_plot_style()

    sub = df[df["lead_h"] == lead_h]
    if sub.empty:
        print(f"  No data for lead_h={lead_h}")
        return

    models_avail = [m for m in ANALYSIS_MODELS if m in sub["model"].unique()]
    n_models     = len(models_avail)
    if n_models == 0:
        return

    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axs = plt.subplots(
        n_rows, n_cols,
        figsize=(5.2 * n_cols, 4.2 * n_rows),
        squeeze=False,
        gridspec_kw={"hspace": 0.38, "wspace": 0.28}
    )
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"Spatial Wind Speed Error  |  fcst − ERA5  |  Lead {lead_h} h\n"
        "Atlantic Routing Corridor — August 2026",
        fontsize=13, fontweight="bold", color="#1a1a1a", y=1.01
    )

    all_lats = sorted(set(lat for lat, lon in EVAL_POINTS))
    all_lons = sorted(set(lon for lat, lon in EVAL_POINTS))
    vmax     = sub["ws_err"].abs().quantile(0.95)

    for idx, model in enumerate(models_avail):
        r   = idx // n_cols
        c   = idx  % n_cols
        ax  = axs[r, c]
        m_sub = sub[sub["model"] == model]

        err_grid = np.full((len(all_lats), len(all_lons)), np.nan)
        lat_idx  = {lat: i for i, lat in enumerate(all_lats)}
        lon_idx  = {lon: i for i, lon in enumerate(all_lons)}

        for _, row_data in m_sub.iterrows():
            li = lat_idx.get(row_data["lat"])
            lo = lon_idx.get(row_data["lon"])
            if li is not None and lo is not None:
                existing = err_grid[li, lo]
                new_val  = abs(row_data["ws_err"])
                err_grid[li, lo] = (
                    new_val if np.isnan(existing)
                    else (existing + new_val) / 2
                )

        im = ax.pcolormesh(
            all_lons, all_lats, err_grid,
            cmap="YlOrRd", vmin=0, vmax=vmax,
            shading="auto"
        )
        ax.scatter(
            [lon for _, lon in EVAL_POINTS],
            [lat for lat, _ in EVAL_POINTS],
            s=8, color="#333333", alpha=0.5, zorder=3
        )

        cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
        cb.set_label("|Error| (m s⁻¹)", fontsize=9)
        cb.ax.tick_params(labelsize=8)

        panel_title = model_label(model)
        ax.set_title(
            panel_title, fontsize=11, fontweight="bold",
            color=MODEL_COLORS.get(model, "#1a1a1a"), pad=6
        )
        ax.set_xlabel("Longitude (°)", fontsize=9, labelpad=4)
        ax.set_ylabel("Latitude (°)", fontsize=9, labelpad=4)
        ax.tick_params(labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cccccc')
        ax.spines['bottom'].set_color('#cccccc')
        ax.grid(alpha=0.15, linewidth=0.5, color='#aaaaaa')

    for idx in range(n_models, n_rows * n_cols):
        axs[idx // n_cols, idx % n_cols].set_visible(False)

    out = OUTPUT_DIR / f"spatial_error_{save_suffix}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out}")


def plot_spatial_error_all_leads(df):
    """Spatial error maps for all lead times."""
    for lead_h in LEAD_TIMES:
        plot_spatial_error(df, lead_h=lead_h, save_suffix=f"{lead_h}h")


def plot_regime_error(regime_df):
    """
    Bar chart: MAE in volatile vs steady conditions per model,
    for wind speed at key lead times. Publication-quality, 300 dpi.
    """
    apply_plot_style()

    key_leads = [lt for lt in [24, 72, 168, 240] if lt in LEAD_TIMES]
    n_panels  = len(key_leads)

    fig, axs = plt.subplots(
        1, n_panels,
        figsize=(5.5 * n_panels, 5),
        sharey=True,
        gridspec_kw={"wspace": 0.18}
    )
    if n_panels == 1:
        axs = [axs]

    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Wind Speed MAE: Volatile vs Steady Conditions\n"
        "Atlantic Routing Corridor — August 2026",
        fontsize=13, fontweight="bold", color="#1a1a1a", y=1.06
    )

    regime_palette = {
        "volatile": "#c0392b",   # crimson
        "steady"  : "#3a7fc1",   # steel blue
    }

    # Pre-compute global y maximum across ALL panels so sharey doesn't clip
    global_max = 0.0
    for lead_h in key_leads:
        sub = regime_df[regime_df["lead_h"] == lead_h]
        if not sub.empty:
            global_max = max(global_max, float(sub["mae"].max()))
    y_top = global_max * 1.18 if global_max > 0 else 1.0

    for ax, lead_h in zip(axs, key_leads):
        sub = regime_df[regime_df["lead_h"] == lead_h]
        if sub.empty:
            ax.set_visible(False)
            continue

        models_avail = [m for m in ANALYSIS_MODELS if m in sub["model"].unique()]
        x = np.arange(len(models_avail))
        w = 0.36

        ax.set_ylim(0, y_top)

        for offset, regime, label in [
            (-w / 2, "volatile", "Volatile"),
            ( w / 2, "steady",   "Steady"),
        ]:
            vals = []
            for model in models_avail:
                r = sub[(sub["model"] == model) & (sub["regime"] == regime)]
                vals.append(float(r["mae"].values[0]) if not r.empty else 0.0)

            bars = ax.bar(
                x + offset, vals, w,
                color=regime_palette[regime],
                alpha=0.88, edgecolor="white", linewidth=0.6,
                label=label, zorder=3
            )

            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        val + global_max * 0.02,
                        f"{val:.2f}",
                        ha="center", va="bottom",
                        fontsize=7.5, color=regime_palette[regime],
                        fontweight="bold"
                    )

        x_labels = [model_label(m) for m in models_avail]
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=15, fontsize=9, ha="right")
        style_ax(
            ax,
            ylabel="MAE (m s⁻¹)" if ax is axs[0] else "",
            grid_axis="y"
        )
        ax.set_title(f"Lead {lead_h} h", pad=18, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, framealpha=0.92, edgecolor="#cccccc")
        ax.set_xlim(-0.6, len(models_avail) - 0.4)

    out = OUTPUT_DIR / "regime_error.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out}")


def plot_region_error(region_df):
    """
    MAE by region per model at key lead times — wind speed.
    Publication-quality, 300 dpi.
    """
    apply_plot_style()

    region_order = list(REGIONS.keys())  # preserves West → Central → East insertion order
    regions = [r for r in region_order if r in region_df["region"].unique()]
    n_panels = len(regions)

    fig, axs = plt.subplots(
        1, n_panels,
        figsize=(5.5 * n_panels, 5),
        sharey=True,
        gridspec_kw={"wspace": 0.18}
    )
    if n_panels == 1:
        axs = [axs]

    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Wind Speed MAE by Sub-Region\n"
        "Atlantic Routing Corridor — August 2026",
        fontsize=13, fontweight="bold", color="#1a1a1a", y=1.06
    )

    tick_vals = (LEAD_TIMES[::2] if len(LEAD_TIMES) > 8 else LEAD_TIMES)
    if LEAD_TIMES[-1] not in tick_vals:
        tick_vals = list(tick_vals) + [LEAD_TIMES[-1]]

    # Pre-compute global y range across ALL region panels
    all_mae_vals = region_df["mae"].dropna()
    if not all_mae_vals.empty:
        g_lo = all_mae_vals.min()
        g_hi = all_mae_vals.max()
        g_span = g_hi - g_lo if g_hi > g_lo else g_hi * 0.1
        y_lo_global = max(0, g_lo - g_span * 0.12)
        y_hi_global = g_hi + g_span * 0.18
    else:
        y_lo_global, y_hi_global = 0, 3.0

    for ax, region in zip(axs, regions):
        sub = region_df[region_df["region"] == region]

        for model in ANALYSIS_MODELS:
            m_sub = sub[sub["model"] == model].sort_values("lead_h")
            if m_sub.empty:
                continue
            is_ens = (model == ENSEMBLE_NAME)
            ax.plot(
                m_sub["lead_h"], m_sub["mae"],
                marker="o", markersize=4,
                linewidth=2.2 if is_ens else 1.8,
                linestyle="--" if is_ens else "-",
                color=MODEL_COLORS.get(model, "#888888"),
                label=model_label(model),
                alpha=0.95,
                zorder=3 if is_ens else 2,
            )

        style_ax(
            ax,
            xlabel="Lead time (h)",
            ylabel="MAE (m s⁻¹)" if ax is axs[0] else "",
            grid_axis="y"
        )
        ax.set_title(region.strip(), pad=18, fontsize=12, fontweight="bold")
        ax.set_xticks(tick_vals)
        ax.tick_params(axis="x", rotation=40, labelsize=9)
        ax.set_ylim(y_lo_global, y_hi_global)
        ax.legend(fontsize=8.5, framealpha=0.92, edgecolor="#cccccc")

    out = OUTPUT_DIR / "region_error.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out}")


# ============================================================
# WRITTEN SUMMARY
# ============================================================

def write_summary(metrics_df, ranking_df, regime_df,
                  corridor_df, region_df, coverage_df):

    lines = []
    lines.append("=" * 70)
    lines.append("FORECAST VERIFICATION SUMMARY")
    lines.append("Atlantic Routing Corridor, August 2026")
    lines.append("=" * 70)

    incomplete_models = []
    for model in MODELS:
        m_cov = coverage_df[coverage_df["model"] == model]
        if m_cov.empty or m_cov["n_init_dates_present"].min() < len(STAGE_DATES):
            incomplete_models.append(model)
    if incomplete_models:
        lines.append(
            "\nNOTE: the following models do not have full "
            f"{len(STAGE_DATES)}-init-date coverage at all lead times: "
            f"{', '.join(incomplete_models)}. Their metrics below are "
            "computed on fewer/different cases than the fully-covered "
            "models and are NOT directly comparable to them. See "
            "coverage_report.csv for the exact gaps before trusting "
            "any ranking that includes these models."
        )

    lines.append(
        f"\nNOTE: {ENSEMBLE_NAME} is a derived forecast, not a raw "
        f"source -- at each (init_date, lead_h, lat, lon) it averages "
        f"the u10/v10 of whichever individual models have a record "
        f"there (only when >= {MIN_ENSEMBLE_MEMBERS} of {len(MODELS)} "
        f"are present), then re-derives wind speed/direction from "
        f"that averaged vector."
    )

    ws_metrics = metrics_df[metrics_df["variable"] == "wind_speed"]
    if not ws_metrics.empty:
        overall = ws_metrics.groupby("model")["rmse"].mean().sort_values()
        best    = overall.index[0]
        lines.append(f"\n1. MOST ACCURATE SOURCE (wind speed RMSE, all leads):")
        for model, rmse in overall.items():
            marker = " ← BEST" if model == best else ""
            if model in incomplete_models:
                marker += "  [INCOMPLETE COVERAGE]"
            lines.append(f"   {model:<20}: RMSE = {rmse:.3f} m/s{marker}")

    lines.append("\n2. SKILL DROP-OFF BY LEAD TIME (wind speed RMSE):")
    for model in ANALYSIS_MODELS:
        m = ws_metrics[ws_metrics["model"] == model].sort_values("lead_h")
        if m.empty:
            continue
        m_cov = coverage_df[coverage_df["model"] == model].set_index("lead_h")
        line_parts = [f"   {model:<20}:"]
        for _, row in m.iterrows():
            lead_h = int(row["lead_h"])
            star = ""
            if lead_h in m_cov.index:
                n_present = m_cov.loc[lead_h, "n_init_dates_present"]
                if n_present < len(STAGE_DATES):
                    star = "*"
            line_parts.append(f"  {lead_h}h={row['rmse']:.3f}{star}")
        lines.append("".join(line_parts))

    lines.append("\n3. SYSTEMATIC BIAS (mean wind speed error):")
    for model in ANALYSIS_MODELS:
        m = ws_metrics[ws_metrics["model"] == model].sort_values("lead_h")
        if m.empty:
            continue
        mean_bias = m["bias"].mean()
        direction = "over-forecast" if mean_bias > 0 else "under-forecast"
        lines.append(
            f"   {model:<20}: mean bias = {mean_bias:+.3f} m/s "
            f"({direction})"
        )

    lines.append("\n4. VOLATILE vs STEADY CONDITIONS (24h lead, wind speed MAE):")
    r24 = regime_df[regime_df["lead_h"] == 24]
    for model in ANALYSIS_MODELS:
        vol = r24[(r24["model"] == model) & (r24["regime"] == "volatile")]
        ste = r24[(r24["model"] == model) & (r24["regime"] == "steady")]
        if vol.empty or ste.empty:
            continue
        lines.append(
            f"   {model:<20}: volatile={vol['mae'].values[0]:.3f}  "
            f"steady={ste['mae'].values[0]:.3f}  "
            f"(ratio={vol['mae'].values[0]/max(ste['mae'].values[0],0.001):.2f}x)"
        )

    lines.append("\n5. WHERE ALL SOURCES PERFORM POORLY:")
    high_err = ws_metrics[ws_metrics["rmse"] > ws_metrics["rmse"].quantile(0.75)]
    if not high_err.empty:
        worst_leads = high_err.groupby("lead_h")["rmse"].mean().sort_values(
            ascending=False
        ).head(3)
        lines.append(
            "   High error (top 25%) concentrated at lead times: "
            + ", ".join(f"{int(h)}h" for h in worst_leads.index)
        )
        if not region_df.empty:
            worst_region = region_df.groupby("region")["rmse"].mean(
            ).sort_values(ascending=False).index[0]
            lines.append(f"   Most challenging region: {worst_region.strip()}")

    if not corridor_df.empty:
        lines.append("\n6. DECISION-RELEVANT CORRIDOR CHECK:")
        for lead_h in [lt for lt in [24, 48, 72, 120, 168, 240] if lt in LEAD_TIMES]:
            sub = corridor_df[corridor_df["lead_h"] == lead_h]
            if sub.empty:
                continue
            lines.append(f"   Lead {lead_h}h corridor agreement with ERA5:")
            for model in ANALYSIS_MODELS:
                m = sub[sub["model"] == model]
                if m.empty:
                    continue
                pct = m["corridor_match"].mean() * 100
                lines.append(f"     {model:<20}: {pct:.1f}% match")

    lines.append("\n7. MODEL RANKING SUMMARY (by wind speed RMSE):")
    if not ranking_df.empty:
        ws_rank = ranking_df[ranking_df["variable"] == "wind_speed"]
        for lead_h in [lt for lt in [24, 48, 72, 120, 168, 240] if lt in LEAD_TIMES]:
            sub = ws_rank[ws_rank["lead_h"] == lead_h].sort_values("rank")
            if sub.empty:
                continue
            ranked_str = ", ".join(
                f"{row['model']}({row['rmse']:.3f})"
                for _, row in sub.iterrows()
            )
            lines.append(f"   Lead {lead_h}h: {ranked_str}")

    lines.append("\n" + "=" * 70)

    text = "\n".join(lines)
    out  = OUTPUT_DIR / "verification_summary.txt"
    with open(out, "w") as f:
        f.write(text)
    print(text)
    print(f"\n  Summary saved: {out}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("FORECAST VERIFICATION ANALYSIS")
    print(f"Output: {OUTPUT_DIR.resolve()}")
    print(f"Lead times (h): {LEAD_TIMES}")
    print("=" * 60)

    era5 = load_era5()

    print("\nBuilding forecast vs ERA5 records...")
    df = build_records(era5)

    if df.empty:
        print("ERROR: No records built. Check file paths and dates.")
        return

    df = add_dir_error(df)
    print(f"\nTotal records: {len(df)}")
    print(f"Models found:  {df['model'].unique().tolist()}")
    print(f"Lead times:    {sorted(df['lead_h'].unique().tolist())}")

    coverage_df = build_coverage_report(df)
    coverage_df.to_csv(OUTPUT_DIR / "coverage_report.csv", index=False)

    ensemble_df = build_ensemble_records(df)
    if not ensemble_df.empty:
        df = pd.concat([df, ensemble_df], ignore_index=True, sort=False)
        df = add_dir_error(df)

    print("\nClassifying volatility regimes...")
    df = classify_regime(df, era5)
    print(f"  Volatile: {(df['regime']=='volatile').sum()}  "
          f"Steady: {(df['regime']=='steady').sum()}  "
          f"Unknown: {(df['regime']=='unknown').sum()}")

    print("\nComputing metrics...")
    metrics_df  = compute_metrics(df)
    region_df   = compute_metrics_by_region(df)
    regime_df_m = compute_metrics_by_regime(df)
    ranking_df  = rank_models(metrics_df)
    corridor_df = decision_corridor_check(df)

    metrics_df.to_csv(OUTPUT_DIR / "metrics_table.csv",  index=False)
    ranking_df.to_csv(OUTPUT_DIR / "model_ranking.csv",  index=False)
    corridor_df.to_csv(OUTPUT_DIR / "decision_check.csv", index=False)
    region_df.to_csv(OUTPUT_DIR / "region_metrics.csv",  index=False)
    regime_df_m.to_csv(OUTPUT_DIR / "regime_metrics.csv", index=False)
    df.to_csv(OUTPUT_DIR / "all_records.csv", index=False)
    print(f"  CSVs saved to {OUTPUT_DIR}/")

    print("\nGenerating plots...")
    plot_skill_decay(metrics_df)
    plot_spatial_error(df, lead_h=24, save_suffix="24h")
    plot_spatial_error_all_leads(df)
    plot_regime_error(regime_df_m)
    plot_region_error(region_df)

    print("\nWriting summary...")
    write_summary(metrics_df, ranking_df, regime_df_m,
                  corridor_df, region_df, coverage_df)

    print("\n" + "=" * 60)
    print("Done. All outputs in VerificationResults/")
    print("=" * 60)


if __name__ == "__main__":
    main()