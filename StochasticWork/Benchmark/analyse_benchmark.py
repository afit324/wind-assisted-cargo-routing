"""
analyse_benchmark_results.py

Analyses the benchmark suite stored as:

BenchmarkResults/
    Benchmark_001/
        era5_arc_details.csv
        forecast_routes.pkl
        settings.json
        summary.json
        actual_weather_arc_details.csv
        actual_weather_results.csv

    Benchmark_002/
        ...

The script automatically discovers all Benchmark_* folders.

Outputs are written to:

BenchmarkResults/Analysis/
"""

from pathlib import Path
import json
import pickle
import ast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = Path("BenchmarkResults")
OUTPUT_DIR = BASE_DIR / "Analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DISCOVER BENCHMARKS
# ============================================================

def find_benchmarks():

    benchmarks = sorted(
        p for p in BASE_DIR.glob("Benchmark_*")
        if p.is_dir()
    )

    print(f"Found {len(benchmarks)} benchmark folders.")

    for b in benchmarks:
        print(f"  {b.name}")

    return benchmarks


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path):

    if not path.exists():
        return {}

    with open(path, "r") as f:
        return json.load(f)


# ============================================================
# LOAD PICKLE
# ============================================================

def load_pickle(path):

    if not path.exists():
        return None

    with open(path, "rb") as f:
        return pickle.load(f)


# ============================================================
# NORMALISE ROUTE
# ============================================================

def normalise_route(route):

    if route is None:
        return None

    if isinstance(route, str):

        try:
            parsed = ast.literal_eval(route)

            if isinstance(parsed, (list, tuple)):
                route = parsed

        except Exception:
            pass

    if isinstance(route, (list, tuple)):

        return " -> ".join(str(x) for x in route)

    return str(route)


# ============================================================
# LOAD ONE BENCHMARK
# ============================================================

def load_benchmark(folder):

    print(f"\nLoading {folder.name}")

    settings = load_json(folder / "settings.json")
    summary = load_json(folder / "summary.json")

    forecast_routes = load_pickle(
        folder / "forecast_routes.pkl"
    )

    # --------------------------------------------------------
    # CSVs
    # --------------------------------------------------------

    forecast_results_path = folder / "era5_arc_details.csv"
    actual_results_path = folder / "actual_weather_results.csv"
    actual_arc_path = folder / "actual_weather_arc_details.csv"

    forecast_arc = (
        pd.read_csv(forecast_results_path)
        if forecast_results_path.exists()
        else pd.DataFrame()
    )

    actual_results = (
        pd.read_csv(actual_results_path)
        if actual_results_path.exists()
        else pd.DataFrame()
    )

    actual_arc = (
        pd.read_csv(actual_arc_path)
        if actual_arc_path.exists()
        else pd.DataFrame()
    )

    # --------------------------------------------------------
    # Add benchmark information
    # --------------------------------------------------------

    benchmark_id = folder.name

    for df in [
        forecast_arc,
        actual_results,
        actual_arc
    ]:

        if not df.empty:
            df["benchmark"] = benchmark_id

    # --------------------------------------------------------
    # Add settings to data
    # --------------------------------------------------------

    cost_function = settings.get(
        "cost_function",
        settings.get("active_cost_fn", "unknown")
    )

    if not forecast_arc.empty:
        forecast_arc["cost_function"] = cost_function

    if not actual_results.empty:
        actual_results["cost_function"] = cost_function

    if not actual_arc.empty:
        actual_arc["cost_function"] = cost_function

    return {
        "benchmark": benchmark_id,
        "settings": settings,
        "summary": summary,
        "forecast_routes": forecast_routes,
        "forecast_arc": forecast_arc,
        "actual_results": actual_results,
        "actual_arc": actual_arc,
    }


# ============================================================
# LOAD ALL BENCHMARKS
# ============================================================

def load_all_benchmarks():

    folders = find_benchmarks()

    benchmarks = []

    for folder in folders:

        try:
            benchmarks.append(
                load_benchmark(folder)
            )

        except Exception as e:

            print(
                f"WARNING: Failed to load "
                f"{folder.name}: {e}"
            )

    return benchmarks


# ============================================================
# COMBINE ACTUAL RESULTS
# ============================================================

def combine_actual_results(benchmarks):

    frames = [
        b["actual_results"]
        for b in benchmarks
        if not b["actual_results"].empty
    ]

    if not frames:
        return pd.DataFrame()

    df = pd.concat(
        frames,
        ignore_index=True
    )

    return df


# ============================================================
# EXTRACT ROUTES FROM PICKLES
# ============================================================

def extract_routes(benchmarks):

    rows = []

    for b in benchmarks:

        benchmark = b["benchmark"]
        routes = b["forecast_routes"]

        if routes is None:
            continue

        # ----------------------------------------------------
        # Case 1: dictionary
        # ----------------------------------------------------

        if isinstance(routes, dict):

            for key, value in routes.items():

                if isinstance(value, dict):

                    policy = value.get(
                        "policy",
                        key
                    )

                    route = value.get(
                        "route",
                        value.get("path")
                    )

                else:

                    policy = str(key)
                    route = value

                rows.append({
                    "benchmark": benchmark,
                    "policy": policy,
                    "route": normalise_route(route)
                })

        # ----------------------------------------------------
        # Case 2: list
        # ----------------------------------------------------

        elif isinstance(routes, list):

            for item in routes:

                if not isinstance(item, dict):
                    continue

                policy = item.get(
                    "policy",
                    "unknown"
                )

                route = item.get(
                    "route",
                    item.get("path")
                )

                rows.append({
                    "benchmark": benchmark,
                    "policy": policy,
                    "route": normalise_route(route)
                })

    return pd.DataFrame(rows)


# ============================================================
# ACTUAL WEATHER SUMMARY
# ============================================================

def actual_summary(df):

    if df.empty:
        return pd.DataFrame()

    required = {
        "policy",
        "actual_cost"
    }

    if not required.issubset(df.columns):
        print(
            "actual_weather_results.csv does not "
            "contain expected columns."
        )
        print(df.columns.tolist())
        return pd.DataFrame()

    summary = (
        df
        .groupby("policy")["actual_cost"]
        .agg([
            "mean",
            "std",
            "min",
            "max",
            "count"
        ])
        .sort_values("mean")
    )

    summary["range"] = (
        summary["max"] -
        summary["min"]
    )

    return summary


# ============================================================
# FORECAST COST FROM ARC DETAILS
# ============================================================

def calculate_forecast_costs(df):

    if df.empty:
        return pd.DataFrame()

    required = {
        "benchmark",
        "policy",
        "stage_cost"
    }

    if not required.issubset(df.columns):

        print(
            "Cannot calculate forecast costs. "
            f"Available columns: {df.columns.tolist()}"
        )

        return pd.DataFrame()

    grouped = (
        df
        .groupby(
            ["benchmark", "policy"],
            as_index=False
        )["stage_cost"]
        .sum()
    )

    grouped = grouped.rename(
        columns={
            "stage_cost":
            "forecast_cost"
        }
    )

    return grouped


# ============================================================
# HINDSIGHT ANALYSIS
# ============================================================

def hindsight_analysis(df, cost_col):

    if df.empty:
        return pd.DataFrame()

    if "policy" not in df.columns:
        return pd.DataFrame()

    if "hindsight" not in set(df["policy"]):

        print(
            f"No hindsight policy found "
            f"for {cost_col}."
        )

        return df

    group_cols = ["benchmark"]

    if "cost_function" in df.columns:
        group_cols.append("cost_function")

    # --------------------------------------------------------
    # Hindsight cost
    # --------------------------------------------------------

    hindsight = (
        df[df["policy"] == "hindsight"]
        .groupby(group_cols)[cost_col]
        .mean()
        .rename("hindsight_cost")
    )

    result = df.merge(
        hindsight,
        left_on=group_cols,
        right_index=True,
        how="left"
    )

    result["hindsight_gap"] = (
        result[cost_col] -
        result["hindsight_cost"]
    )

    result["hindsight_gap_pct"] = (
        100 *
        result["hindsight_gap"] /
        result["hindsight_cost"].replace(
            0,
            np.nan
        )
    )

    return result


# ============================================================
# CHECK HINDSIGHT
# ============================================================

def check_hindsight(df, cost_col):

    if df.empty:
        return

    if "policy" not in df.columns:
        return

    print("\n=== HINDSIGHT CHECK ===")

    problems = []

    for benchmark, group in df.groupby(
        "benchmark"
    ):

        if "hindsight" not in set(group["policy"]):
            continue

        h = group.loc[
            group["policy"] == "hindsight",
            cost_col
        ].iloc[0]

        for _, row in group.iterrows():

            policy = row["policy"]
            cost = row[cost_col]

            if cost < h - 1e-10:

                problems.append({
                    "benchmark": benchmark,
                    "policy": policy,
                    "policy_cost": cost,
                    "hindsight_cost": h,
                    "difference": cost - h
                })

    if problems:

        print(
            "WARNING: Hindsight is NOT optimal "
            "in the following benchmarks:"
        )

        problems_df = pd.DataFrame(problems)

        print(problems_df)

        problems_df.to_csv(
            OUTPUT_DIR /
            "hindsight_problems.csv",
            index=False
        )

    else:

        print(
            "✓ Hindsight is optimal in every "
            "benchmark."
        )


# ============================================================
# ROUTE DIVERSITY
# ============================================================

def route_diversity(routes):

    if routes.empty:
        return pd.DataFrame()

    rows = []

    for policy, group in routes.groupby(
        "policy"
    ):

        counts = (
            group["route"]
            .value_counts()
        )

        rows.append({
            "policy": policy,
            "benchmarks": len(group),
            "unique_routes":
                group["route"].nunique(),
            "unique_route_pct":
                100 *
                group["route"].nunique() /
                len(group),
            "most_common_route":
                counts.index[0]
                if len(counts)
                else None,
            "most_common_frequency":
                counts.iloc[0]
                if len(counts)
                else 0,
            "most_common_frequency_pct":
                100 *
                counts.iloc[0] /
                len(group)
                if len(counts)
                else 0
        })

    return (
        pd.DataFrame(rows)
        .sort_values(
            "unique_routes",
            ascending=False
        )
    )


# ============================================================
# PAIRWISE WIN RATES
# ============================================================

def pairwise_win_rates(
    df,
    cost_col,
    group_col="benchmark"
):

    if df.empty:
        return pd.DataFrame()

    pivot = df.pivot_table(
        index=group_col,
        columns="policy",
        values=cost_col,
        aggfunc="mean"
    )

    policies = list(pivot.columns)

    rows = []

    for p in policies:

        for q in policies:

            if p == q:
                continue

            common = pivot[
                [p, q]
            ].dropna()

            if common.empty:
                continue

            rows.append({
                "policy": p,
                "opponent": q,
                "comparisons": len(common),
                "p_wins_pct":
                    100 *
                    np.mean(
                        common[p] <
                        common[q]
                    ),
                "ties_pct":
                    100 *
                    np.mean(
                        common[p] ==
                        common[q]
                    ),
                "opponent_wins_pct":
                    100 *
                    np.mean(
                        common[p] >
                        common[q]
                    )
            })

    return pd.DataFrame(rows)


# ============================================================
# RANDOM BEATS ADAPTIVE
# ============================================================

def random_vs_adaptive(
    df,
    cost_col
):

    if df.empty:
        return pd.DataFrame()

    pivot = df.pivot_table(
        index="benchmark",
        columns="policy",
        values=cost_col,
        aggfunc="mean"
    )

    if not {
        "adaptive",
        "random"
    }.issubset(pivot.columns):

        return pd.DataFrame()

    result = pivot[
        ["adaptive", "random"]
    ].copy()

    result["random_beats_adaptive"] = (
        result["random"] <
        result["adaptive"]
    )

    result["difference_random_minus_adaptive"] = (
        result["random"] -
        result["adaptive"]
    )

    return result


# ============================================================
# FORECAST VS ACTUAL
# ============================================================

def forecast_vs_actual(
    forecast,
    actual
):

    if forecast.empty or actual.empty:
        return pd.DataFrame()

    keys = [
        "benchmark",
        "policy"
    ]

    if not all(
        k in forecast.columns
        for k in keys
    ):
        return pd.DataFrame()

    if not all(
        k in actual.columns
        for k in keys
    ):
        return pd.DataFrame()

    merged = forecast.merge(
        actual[
            keys +
            ["actual_cost"]
        ],
        on=keys,
        how="inner"
    )

    if merged.empty:
        return merged

    merged["forecast_error"] = (
        merged["actual_cost"] -
        merged["forecast_cost"]
    )

    merged["forecast_error_pct"] = (
        100 *
        merged["forecast_error"] /
        merged["forecast_cost"].replace(
            0,
            np.nan
        )
    )

    return merged


# ============================================================
# SETTINGS SUMMARY
# ============================================================

def settings_summary(benchmarks):

    rows = []

    for b in benchmarks:

        s = b["settings"]

        row = {
            "benchmark": b["benchmark"]
        }

        for key, value in s.items():

            if isinstance(
                value,
                (str, int, float, bool)
            ):

                row[key] = value

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# PLOTS
# ============================================================

def make_plots(
    actual,
    forecast
):

    # --------------------------------------------------------
    # Actual mean cost
    # --------------------------------------------------------

    if (
        not actual.empty and
        "actual_cost" in actual.columns
    ):

        means = (
            actual
            .groupby("policy")["actual_cost"]
            .mean()
            .sort_values()
        )

        plt.figure(
            figsize=(10, 6)
        )

        means.plot(
            kind="bar"
        )

        plt.ylabel(
            "Mean actual weather cost"
        )

        plt.title(
            "Mean Actual Weather Cost by Policy"
        )

        plt.xticks(
            rotation=30,
            ha="right"
        )

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR /
            "mean_actual_cost.png",
            dpi=200
        )

        plt.close()

    # --------------------------------------------------------
    # Forecast mean cost
    # --------------------------------------------------------

    if (
        not forecast.empty and
        "forecast_cost" in forecast.columns
    ):

        means = (
            forecast
            .groupby("policy")["forecast_cost"]
            .mean()
            .sort_values()
        )

        plt.figure(
            figsize=(10, 6)
        )

        means.plot(
            kind="bar"
        )

        plt.ylabel(
            "Mean forecast cost"
        )

        plt.title(
            "Mean Forecast Cost by Policy"
        )

        plt.xticks(
            rotation=30,
            ha="right"
        )

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR /
            "mean_forecast_cost.png",
            dpi=200
        )

        plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "BENCHMARK SUITE ANALYSIS"
    )

    print(
        "=" * 70
    )

    benchmarks = load_all_benchmarks()

    if not benchmarks:

        print(
            "\nNo benchmark folders found."
        )

        print(
            f"Expected folders inside: "
            f"{BASE_DIR.resolve()}"
        )

        return

    # ========================================================
    # COMBINE DATA
    # ========================================================

    actual = combine_actual_results(
        benchmarks
    )

    forecast_arc_frames = [
        b["forecast_arc"]
        for b in benchmarks
        if not b["forecast_arc"].empty
    ]

    if forecast_arc_frames:

        forecast_arc = pd.concat(
            forecast_arc_frames,
            ignore_index=True
        )

    else:

        forecast_arc = pd.DataFrame()

    # ========================================================
    # ROUTES
    # ========================================================

    routes = extract_routes(
        benchmarks
    )

    # ========================================================
    # FORECAST COSTS
    # ========================================================

    forecast = calculate_forecast_costs(
        forecast_arc
    )

    # ========================================================
    # SAVE COMBINED DATA
    # ========================================================

    if not actual.empty:

        actual.to_csv(
            OUTPUT_DIR /
            "all_actual_weather_results.csv",
            index=False
        )

    if not forecast_arc.empty:

        forecast_arc.to_csv(
            OUTPUT_DIR /
            "all_era5_arc_details.csv",
            index=False
        )

    if not routes.empty:

        routes.to_csv(
            OUTPUT_DIR /
            "all_forecast_routes.csv",
            index=False
        )

    # ========================================================
    # SETTINGS
    # ========================================================

    settings_df = settings_summary(
        benchmarks
    )

    settings_df.to_csv(
        OUTPUT_DIR /
        "benchmark_settings.csv",
        index=False
    )

    print(
        f"\nLoaded {len(benchmarks)} benchmarks."
    )

    # ========================================================
    # ACTUAL WEATHER
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "ACTUAL WEATHER RESULTS"
    )

    print(
        "=" * 70
    )

    actual_stats = actual_summary(
        actual
    )

    print(
        actual_stats
    )

    actual_stats.to_csv(
        OUTPUT_DIR /
        "actual_policy_summary.csv"
    )

    # ========================================================
    # FORECAST
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FORECAST RESULTS"
    )

    print(
        "=" * 70
    )

    if not forecast.empty:

        forecast_stats = (
            forecast
            .groupby("policy")["forecast_cost"]
            .agg([
                "mean",
                "std",
                "min",
                "max",
                "count"
            ])
            .sort_values("mean")
        )

        print(
            forecast_stats
        )

        forecast_stats.to_csv(
            OUTPUT_DIR /
            "forecast_policy_summary.csv"
        )

    # ========================================================
    # HINDSIGHT
    # ========================================================

    actual_hindsight = hindsight_analysis(
        actual,
        "actual_cost"
    )

    if not actual_hindsight.empty:

        actual_hindsight.to_csv(
            OUTPUT_DIR /
            "actual_hindsight_gaps.csv",
            index=False
        )

    if not forecast.empty:

        forecast_hindsight = hindsight_analysis(
            forecast,
            "forecast_cost"
        )

        forecast_hindsight.to_csv(
            OUTPUT_DIR /
            "forecast_hindsight_gaps.csv",
            index=False
        )

    # ========================================================
    # HINDSIGHT CHECK
    # ========================================================

    check_hindsight(
        actual,
        "actual_cost"
    )

    if not forecast.empty:

        check_hindsight(
            forecast,
            "forecast_cost"
        )

    # ========================================================
    # ROUTE DIVERSITY
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "ROUTE DIVERSITY"
    )

    print(
        "=" * 70
    )

    diversity = route_diversity(
        routes
    )

    print(
        diversity
    )

    diversity.to_csv(
        OUTPUT_DIR /
        "route_diversity.csv",
        index=False
    )

    # ========================================================
    # PAIRWISE FORECAST
    # ========================================================

    if not forecast.empty:

        print(
            "\n"
            + "=" * 70
        )

        print(
            "FORECAST PAIRWISE WIN RATES"
        )

        print(
            "=" * 70
        )

        wins = pairwise_win_rates(
            forecast,
            "forecast_cost"
        )

        print(
            wins
        )

        wins.to_csv(
            OUTPUT_DIR /
            "forecast_pairwise_win_rates.csv",
            index=False
        )

    # ========================================================
    # PAIRWISE ACTUAL
    # ========================================================

    if not actual.empty:

        print(
            "\n"
            + "=" * 70
        )

        print(
            "ACTUAL WEATHER PAIRWISE WIN RATES"
        )

        print(
            "=" * 70
        )

        wins = pairwise_win_rates(
            actual,
            "actual_cost"
        )

        print(
            wins
        )

        wins.to_csv(
            OUTPUT_DIR /
            "actual_pairwise_win_rates.csv",
            index=False
        )

    # ========================================================
    # RANDOM VS ADAPTIVE
    # ========================================================

    random_check = random_vs_adaptive(
        actual,
        "actual_cost"
    )

    if not random_check.empty:

        random_check.to_csv(
            OUTPUT_DIR /
            "random_vs_adaptive.csv"
        )

        print(
            "\n"
            + "=" * 70
        )

        print(
            "RANDOM VS ADAPTIVE"
        )

        print(
            "=" * 70
        )

        print(
            random_check
        )

        print(
            "\nPercentage of benchmarks "
            "where random beats adaptive:",
            100 *
            random_check[
                "random_beats_adaptive"
            ].mean(),
            "%"
        )

    # ========================================================
    # FORECAST VS ACTUAL
    # ========================================================

    comparison = forecast_vs_actual(
        forecast,
        actual
    )

    if not comparison.empty:

        comparison.to_csv(
            OUTPUT_DIR /
            "forecast_vs_actual.csv",
            index=False
        )

        print(
            "\n"
            + "=" * 70
        )

        print(
            "FORECAST VS ACTUAL"
        )

        print(
            "=" * 70
        )

        print(
            comparison
            .groupby("policy")
            [
                [
                    "forecast_cost",
                    "actual_cost",
                    "forecast_error_pct"
                ]
            ]
            .mean()
            .sort_values(
                "actual_cost"
            )
        )

    # ========================================================
    # PLOTS
    # ========================================================

    make_plots(
        actual,
        forecast
    )

    # ========================================================
    # FINISHED
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "ANALYSIS COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nAll outputs saved to:"
        f"\n{OUTPUT_DIR.resolve()}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()