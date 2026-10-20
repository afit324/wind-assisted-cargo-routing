# =============================================================================
# route_experiments.jl -- experiments, analysis and run drivers
#
#   0. Includes, folder layout, shared constants
#   1. Shared helpers (wind loader, wind sources, run-status tables, leg tables, geometry)
#   2. Convergence study        (grid step / radius / energy bin / heading ablation)
#   3. Verification vs the 2025 baseline solver
#   4. Analysis of saved runs   (Part A tables+plots, Part B maps+progress, greedy)
#   5. Sampled routes + interactive variable-power explorer
#   6. Climatology-match scoring
#   7. Main pipeline            (select windows -> download GFS -> five methods per window)
#   8. RUN SETTINGS             (bottom of the file: choose what runs)
#
# Only definitions run on include. Everything that executes is triggered from
# section 8. Run the heavy drivers (pipeline / convergence / verification) in
# their OWN Julia process, separate from the analysis.
# =============================================================================

# =============================================================================
# 0. Includes, folder layout, shared constants
# =============================================================================
include("route_ship.jl")
include("routing.jl")
include("route_simulator.jl")
include("climatology_routing.jl")      # must precede climatology_experiments
include("climatology_experiments.jl")  # uses Climatology, RouteStepper2 in signatures

using CSV, DataFrames, Dates, Statistics, Random, DataStructures, GRIB, JLD2, ProgressMeter
using CairoMakie
const CM = CairoMakie     # qualify Makie calls: the routing code also loads GLMakie
const GM = GLMakie
CM.activate!()

const PROJECT_DIR = @__DIR__
_fwd(p) = replace(p, '\\' => '/')   # forward slashes: safe on Windows and for bash

# ---------------------------------------------------------------- INPUTS (./inputs)
const INPUT_DIR            = joinpath(PROJECT_DIR, "inputs")
const GFS_DIR              = _fwd(joinpath(INPUT_DIR, "gfs"))                   # gfs_top5_<tag>_<yyyymmdd>_00z/ folders
const CLIMATOLOGY_DATA_DIR = joinpath(INPUT_DIR, "ClimatologyOceanData")
const HISTORICAL_DIR       = joinpath(CLIMATOLOGY_DATA_DIR, "route")            # YYYY-MM.grib, 6-hourly
const HOURLY_REFERENCE_DIR = joinpath(CLIMATOLOGY_DATA_DIR, "hourly_reference") # YYYY-MM-hourly.grib + manifest.jld2
const SPLIT_WIND_DIR       = joinpath(INPUT_DIR, "OceanData_split_fixed")       # YYYYMMDD_HHMM.grib (verification only)
const POLAR_DIR            = joinpath(INPUT_DIR, "Polars")                      # RisePol<P>kW.csv

# ---------------------------------------------------------------- OUTPUTS (./outputs)
const OUT_DIR       = joinpath(PROJECT_DIR, "outputs")
const RUNS_DIR      = joinpath(OUT_DIR, "runs")             # saved routes (.jld2), climatology caches
const TABLES_DIR    = joinpath(OUT_DIR, "tables")           # run_status.csv, legs_*.csv, selected/failed windows
const CONV_DIR      = joinpath(OUT_DIR, "convergence")
const VERIF_DIR     = joinpath(OUT_DIR, "verification")
const ADIR          = joinpath(OUT_DIR, "analysis_report")
const SDIR          = joinpath(ADIR, "energy_wind")         # sampled route csvs for the explorer
const CLIMMATCH_DIR = joinpath(OUT_DIR, "climatology_match")

for d in (GFS_DIR, RUNS_DIR, TABLES_DIR, CONV_DIR, VERIF_DIR, ADIR, SDIR, CLIMMATCH_DIR)
    mkpath(d)
end

out(f)         = joinpath(RUNS_DIR, f)
tab(f)         = joinpath(TABLES_DIR, f)
conv_out(f)    = joinpath(CONV_DIR, f)
ver_out(f)     = joinpath(VERIF_DIR, f)
ap(f)          = joinpath(ADIR, f)
match_out(f)   = joinpath(CLIMMATCH_DIR, f)
gfs_path(name) = _fwd(joinpath(GFS_DIR, name))

# routing.jl's load_polar_lut reads a hard-coded "./Polars". Patch it to use a
# POLAR_DIR_REF (see notes) so the polars are found under inputs/Polars.
if isdefined(@__MODULE__, :POLAR_DIR_REF)
    POLAR_DIR_REF[] = POLAR_DIR
elseif !isdir("./Polars")
    @warn "routing.jl reads polars from ./Polars (relative to the working directory), " *
          "but $POLAR_DIR is where this file keeps them. Patch load_polar_lut or run from a folder containing Polars/."
end

CLIMATOLOGY_ARCHIVE_DIR_REF[] = HISTORICAL_DIR

# ---------------------------------------------------------------- shared settings
const START_LOCATION = (-94.0, 28.5)
const END_LOCATION   = (-5.5, 49.5)
const MIN_SPEED      = 1.0
const RMAX           = 4
const LEVELS_ALL     = [0.0, 400.0, 1200.0, 2000.0]
const CONST_KW       = 400.0
const C400_MODE      = "c400"
const C400_MAX_E     = 600_000.0   # non-binding at 400 kW
const C400_BIN       = 6_000.0
const NDAYS          = 20          # GFS cycles per window
const CYCLE_HOUR     = 0
const OUTBASE_PREFIX = "gfs_top5"  # GFS folders: GFS_DIR/gfs_top5_<tag>_<yyyymmdd>_00z
const FULL_FORECAST_DAYS = 7.0
const RUN_TAGS       = ["20221003", "20221215", "20231106", "20240616", "20240822"]

const METHODS  = ["hindsight", "committed", "rolling", "committed_clim", "rolling_clim"]
const POLICIES = ["committed", "rolling", "committed_clim", "rolling_clim"]
const LABEL = Dict("hindsight" => "Hindsight", "committed" => "Committed", "rolling" => "Rolling",
                   "committed_clim" => "Committed + clim.", "rolling_clim" => "Rolling + clim.",
                   "greedy_committed"   => "Greedy (committed)",
                   "greedy_allforecast" => "Greedy (all forecasts)")
const COLOR = Dict("hindsight" => :black, "committed" => :firebrick, "rolling" => :steelblue,
                   "committed_clim" => :orange, "rolling_clim" => :seagreen,
                   "greedy_committed" => :purple, "greedy_allforecast" => :magenta)

gfs_outbase(tag) = gfs_path("$(OUTBASE_PREFIX)_$(tag)")   # prefix for rolling_gfs_dirs / the download script
var_mode(c) = "var_s$(round(Int, c.slack*100))_m$(round(Int, c.margin*100))"

# analysis constants
const CALM_MS   = 5.0
const WINDY_MS  = 12.0
const FC_MAX_H  = 336.0
const WS_BINS   = [0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 40.0]
const WS_LABELS = ["0-3", "3-6", "6-9", "9-12", "12-15", "15+"]
const HIST_EDGES = collect(0.0:2.0:30.0)

# =============================================================================
# 1. Shared helpers
# =============================================================================


# ---------------------------------------------------------------- wind-source helpers
function reset_sources!()
    clear_gfs_forecast_source!(); clear_synthetic_wind!(); clear_climatology_blend!()
    clear_hourly_reference_source!()
    while _CLIMATOLOGY_ARCHIVE_DEPTH[] > 0
        clear_climatology_archive_source!()
    end
end
function with_archive(f)
    set_climatology_archive_source!(; dir = HISTORICAL_DIR)
    try f() finally clear_climatology_archive_source!() end
end
function in_clean_archive(f)
    reset_sources!()
    try with_archive(f) finally reset_sources!() end
end

# ---------------------------------------------------------------- run-status tables (run_status.csv)
const TIME   = Dict{Tuple{String,String,String},Float64}()   # (tag, mode, method)
const ENERGY = Dict{Tuple{String,String,String},Float64}()
const REACH  = Dict{Tuple{String,String,String},Bool}()

"""Read run_status.csv (last row per run wins) into TIME / ENERGY / REACH and return the table."""
function load_run_status!()
    p = tab("run_status.csv")
    isfile(p) || error("run_status.csv not found at $p -- run the pipeline first.")
    status = CSV.read(p, DataFrame; types = Dict(:tag => String))
    status = unique(status[end:-1:1, :], [:tag, :mode, :method])
    empty!(TIME); empty!(ENERGY); empty!(REACH)
    for r in eachrow(status)
        k = (String(r.tag), String(r.mode), String(r.method))
        TIME[k] = r.time_h; ENERGY[k] = r.energy_kwh; REACH[k] = r.reached_goal
    end
    return status
end
timeof(t, md, m)   = get(TIME,   (t, md, m), NaN)
energyof(t, md, m) = get(ENERGY, (t, md, m), NaN)
pen(t, md, m) = (h = timeof(t, md, "hindsight"); 100 * (timeof(t, md, m) - h) / h)

function log_status(tag, mode, method, time_h, energy, reached)
    row = DataFrame(tag=[tag], mode=[mode], method=[method], time_h=[time_h],
                    energy_kwh=[energy], reached_goal=[reached])
    p = tab("run_status.csv")
    CSV.write(p, row; append = isfile(p))
end

# ---------------------------------------------------------------- small helpers
tag2date(tag::String) = Date(tag, dateformat"yyyymmdd")
fmt(x; d = 2) = (x isa Real && !isnan(x)) ? string(round(x, digits = d)) : "n/a"
pad(x, n = 12) = rpad(x isa AbstractString ? x : fmt(x), n)
safeplot(f::Function, name::AbstractString) =
    try f() catch e; @warn "plot $name failed" exception = (e, catch_backtrace()) end
fmt_lon(v) = string(abs(round(Int, v)), "°", v < 0 ? "W" : "E")
fmt_lat(v) = string(abs(round(Int, v)), "°", v < 0 ? "S" : "N")
attempt(f, name) = try Base.invokelatest(f) catch e; println("\n!!! $name failed: ", sprint(showerror, e)); nothing end

# ---------------------------------------------------------------- per-leg table
# Wind is sampled at the boat's position at mid-leg time. Cumulative differences keep
# the partial landing segments of the rolling route.
function leg_table(stepper, windfn, grid_points, date, mode, method)
    n_rows, n_cols = size(grid_points)
    lons_1d = [grid_points[1, j][1] for j in 1:n_cols]
    lats_1d = [grid_points[i, 1][2] for i in 1:n_rows]
    rows = DataFrame(date=Date[], mode=String[], method=String[], wind_ms=Float64[], power_kw=Float64[],
                     dt_h=Float64[], energy_kwh=Float64[], dist_nm=Float64[], speed_kn=Float64[])
    st = stepper.steps
    for k in 2:length(st)
        dt = st[k].cum_time_s - st[k-1].cum_time_s
        dt <= 1e-6 && continue
        t_mid = (st[k].cum_time_s + st[k-1].cum_time_s) / 2
        pos   = state_at_time(stepper, t_mid)
        U, V  = windfn(t_mid)
        ws, _ = _wind_speed_dir(U, V, nearest_index(lats_1d, pos.lat), nearest_index(lons_1d, pos.lon))
        e = st[k].cum_energy_kwh - st[k-1].cum_energy_kwh
        d_nm = (st[k].cum_dist_km - st[k-1].cum_dist_km) / 1.852
        push!(rows, (date, mode, method, ws, e / (dt/3600), dt/3600, e, d_nm, d_nm / (dt/3600)))
    end
    return rows
end
stepper_wind(s) = t -> get_wind(s.wind, t)

# ---------------------------------------------------------------- geometry helpers
function snap_ocean(mask, i, j; maxr = 8)
    mask[i, j] && return (i, j)
    for r in 1:maxr, di in -r:r, dj in -r:r
        max(abs(di), abs(dj)) == r || continue
        ni, nj = i + di, j + dj
        (1 <= ni <= size(mask,1) && 1 <= nj <= size(mask,2) && mask[ni,nj]) && return (ni, nj)
    end
    error("no ocean cell near ($i,$j)")
end

function great_circle_cells_ll(gp, mask, si, gi, sll, ell; n = 400)
    nr, nc = size(gp)
    lons = [gp[1, j][1] for j in 1:nc]; lats = [gp[i, 1][2] for i in 1:nr]
    φ1, λ1 = deg2rad(sll[2]), deg2rad(sll[1]); φ2, λ2 = deg2rad(ell[2]), deg2rad(ell[1])
    Δ = 2 * asin(clamp(sqrt(sin((φ2-φ1)/2)^2 + cos(φ1)*cos(φ2)*sin((λ2-λ1)/2)^2), 0.0, 1.0))
    cells = Tuple{Int,Int}[si]
    for f in range(0.0, 1.0, length = n)[2:end-1]
        A = sin((1-f)*Δ)/sin(Δ); B = sin(f*Δ)/sin(Δ)
        x = A*cos(φ1)*cos(λ1) + B*cos(φ2)*cos(λ2)
        y = A*cos(φ1)*sin(λ1) + B*cos(φ2)*sin(λ2)
        z = A*sin(φ1) + B*sin(φ2)
        c = snap_ocean(mask, nearest_index(lats, rad2deg(atan(z, sqrt(x^2 + y^2)))),
                             nearest_index(lons, rad2deg(atan(y, x))))
        c != cells[end] && push!(cells, c)
    end
    cells[end] != gi && push!(cells, gi)
    return cells
end

# times (s since departure) at which the rolling run started a new plan
function replan_times_s(path::String)
    jldopen(path, "r") do f
        n = f["shared/n_days"]
        t = 0.0; ts = Float64[]
        for k in 1:n
            push!(ts, t)
            t += f["day_$k/landed_time_s"]
        end
        ts
    end
end

# stats helpers used by the analysis
function rank_avg(x::AbstractVector{<:Real})
    p = sortperm(x); r = zeros(length(x)); i = 1
    while i <= length(x)
        j = i
        while j < length(x) && x[p[j+1]] == x[p[i]]; j += 1; end
        r[p[i:j]] .= (i + j) / 2
        i = j + 1
    end
    return r
end
spearman(x, y) = (length(x) < 3 || std(x) == 0 || std(y) == 0) ? NaN : cor(rank_avg(x), rank_avg(y))

function boot_ci(x::AbstractVector{<:Real}; B = 10_000, seed = 1)
    n = length(x); n < 2 && return (NaN, NaN)
    rng = MersenneTwister(seed)
    ms = [mean(x[rand(rng, 1:n, n)]) for _ in 1:B]
    return (quantile(ms, 0.025), quantile(ms, 0.975))
end

function hav_km(lon1, lat1, lon2, lat2)
    R = 6371.0; p1, p2 = deg2rad(lat1), deg2rad(lat2)
    a = sin((p2 - p1) / 2)^2 + cos(p1) * cos(p2) * sin(deg2rad(lon2 - lon1) / 2)^2
    return 2R * asin(min(1.0, sqrt(a)))
end

# "var_s102_m10" -> (slack = 1.02, margin = 0.10); anything else -> nothing
function parse_mode(md::String)
    mt = match(r"^var_s(\d+)_m(\d+)$", md)
    mt === nothing && return nothing
    return (slack = parse(Int, mt.captures[1]) / 100, margin = parse(Int, mt.captures[2]) / 100)
end

bin_index(w)   = clamp(searchsortedlast(WS_BINS, w), 1, length(WS_BINS) - 1)
level_index(p) = argmin(abs.(LEVELS_ALL .- p))

function leg_metrics(df::DataFrame)
    w = df.dt_h; T_ = sum(w)
    return (hours = T_, dist_nm = sum(df.dist_nm), mean_speed_kn = sum(df.dist_nm) / T_,
            mean_wind_ms = sum(w .* df.wind_ms) / T_,
            frac_calm = sum(w[df.wind_ms .< CALM_MS]) / T_, frac_windy = sum(w[df.wind_ms .>= WINDY_MS]) / T_,
            energy_kwh = sum(df.energy_kwh), mean_power_kw = sum(df.energy_kwh) / T_)
end

function wind_hist(df)
    c = zeros(length(HIST_EDGES) - 1)
    for (w, dt) in zip(df.wind_ms, df.dt_h)
        k = clamp(searchsortedlast(HIST_EDGES, w), 1, length(c)); c[k] += dt
    end
    return c ./ sum(c)
end
tvd(p, q) = 0.5 * sum(abs.(p .- q))

# =============================================================================
# 2. CONVERGENCE STUDY  (grid step, radius, energy bin, heading ablation)
# =============================================================================
Base.@kwdef struct ConvCfg
    start_location = (-94.0, 28.5)
    end_location   = (-60.0, 30.0)
    pad_deg        = 5.0
    base_step      = 0.25
    base_R         = 4
    hop_deg        = 1.0
    const_kw       = 400.0
    var_levels     = [0.0, 400.0, 1200.0, 2000.0]
    max_energy     = 150_000.0          # non-binding for the constant-400 runs
    budget_slack   = 1.02               # variable budget = slack * energy of the 400 kW run
    min_speed      = 1.0
    start_date     = Date(2025, 11, 1)
    start_hhmm     = 1200
    base_bin_div   = 100
end

const WIND_MODES = (:wind, :nowind)
const ROUTES = Dict{Tuple,Vector{Tuple{Float64,Float64}}}()

function make_domain(cfg::ConvCfg, step::Float64)
    pad_cells = max(2, round(Int, cfg.pad_deg / step))
    grid_points, lats, lons, mask = _build_grid_and_mask(;
        start_location = cfg.start_location, end_location = cfg.end_location,
        lon_step = step, lat_step = step, padding = pad_cells)
    si = (nearest_index(lats, cfg.start_location[2]), nearest_index(lons, cfg.start_location[1]))
    gi = (nearest_index(lats, cfg.end_location[2]),   nearest_index(lons, cfg.end_location[1]))
    _check_endpoints(mask, si, gi)
    snap_s = getDistance(cfg.start_location..., grid_points[si...]...) / 1000
    snap_g = getDistance(cfg.end_location...,   grid_points[gi...]...) / 1000
    println("  endpoint snapping: start $(round(snap_s,digits=2)) km, goal $(round(snap_g,digits=2)) km")
    return (grid_points = grid_points, mask = mask, start_idx = si, goal_idx = gi,
            n_cells = length(grid_points), snap_km = max(snap_s, snap_g))
end

function solve_row(cfg::ConvCfg, dom; Rmax::Int, power_levels::Vector{Float64},
                   max_energy::Float64, bin_size::Float64,
                   zero_wind::Bool = false, key = nothing, heading_mode::Symbol = :full)
    ct = cfg.start_hhmm
    cd = day(cfg.start_date); cm = month(cfg.start_date); cy = year(cfg.start_date)
    diag = Dict{Symbol,Any}()
    res, wall = run_timed(() -> findDijkstraRoutefinal(
        dom.grid_points, dom.start_idx, dom.goal_idx, ct, cd, cm, cy,
        max_energy, power_levels, bin_size, cfg.min_speed;
        ocean_mask = dom.mask, Rmax = Rmax,
        force_band = 1, dynamic_bands = false, heading_mode = heading_mode,
        stepped_wind = true, time_aware_wind = true, accurate_stepped_all_bands = true,
        zero_wind = zero_wind, record_diagnostics = false, diagnostics = false,
        wind_cache_size = 100, diag_out = diag))
    path, cost, energy, power = res
    if isempty(path)
        return (Feasible = false, TimeH = NaN, DistKm = NaN, EnergyKwh = NaN,
                Waypoints = 0, WallS = wall, Confirmed = 0)
    end
    t, d, _, e = _route_stats_power(path, cost, energy, power)
    key === nothing || (ROUTES[key] = path)
    return (Feasible = true, TimeH = t, DistKm = d, EnergyKwh = e, Waypoints = length(path),
            WallS = wall, Confirmed = get(diag, :total_confirmed, 0))
end

function add_errors!(df::DataFrame)
    f = df.Feasible
    ref = any(f) ? df.TimeH[findlast(f)] : NaN
    df.ErrPct = 100 .* (df.TimeH .- ref) ./ ref
    prev = NaN
    df.DPrevPct = map(eachindex(df.TimeH)) do k
        v = df.Feasible[k] ? df.TimeH[k] : NaN
        o = isnan(prev) || isnan(v) ? NaN : 100 * (v - prev) / prev
        isnan(v) || (prev = v)
        o
    end
    return df
end
function add_errors_grouped(df::DataFrame)
    parts = DataFrame[]
    for w in unique(df.Wind), c in unique(df.Case)
        sub = df[(df.Wind .== w) .& (df.Case .== c), :]
        isempty(sub) || push!(parts, add_errors!(sub))
    end
    return vcat(parts...)
end

function print_table(name, df, xcol)
    println("\n=== $name ===")
    show(stdout, select(df, :Wind, :Case, xcol, :Feasible, :TimeH, :DistKm,
                        :EnergyKwh, :ErrPct, :DPrevPct, :WallS); allrows = true)
    println()
end

# ---- (a) grid step, constant 400 kW
function run_grid_step_small(cfg::ConvCfg = ConvCfg();
        steps = [2.0, 1.0, 0.5, 0.25, 0.2, 0.1], csv = "conv_grid_small.csv")
    rows = NamedTuple[]
    last_dom = nothing
    for step in sort(steps, rev = true)
        R = max(1, round(Int, cfg.hop_deg / step))
        println("\n--- grid step $(step)°  (Rmax=$R) ---")
        dom = make_domain(cfg, step)
        last_dom = dom
        for w in WIND_MODES
            println("  [$w]")
            r = solve_row(cfg, dom; Rmax = R, power_levels = [cfg.const_kw],
                          max_energy = cfg.max_energy, bin_size = cfg.max_energy / cfg.base_bin_div,
                          zero_wind = (w == :nowind), key = (:grid, w, "const400", step))
            push!(rows, merge((Wind = String(w), Case = "const400", Step = step, Rmax = R,
                               Cells = dom.n_cells, SnapKm = dom.snap_km), r))
        end
    end
    df = add_errors_grouped(DataFrame(rows))
    CSV.write(conv_out(csv), df); print_table("Grid step", df, :Step)
    return (df = df, dom = last_dom, xs = sort(steps, rev = true), case = "const400")
end

# ---- (b) radius, constant 400 kW
function run_radius_small(cfg::ConvCfg = ConvCfg();
        radii = [1, 2, 3, 4, 6, 8, 12, 18], csv = "conv_radius_small.csv")
    dom = make_domain(cfg, cfg.base_step)
    rows = NamedTuple[]
    for R in sort(radii)
        println("\n--- Rmax = $R ---")
        for w in WIND_MODES
            println("  [$w]")
            r = solve_row(cfg, dom; Rmax = R, power_levels = [cfg.const_kw],
                          max_energy = cfg.max_energy, bin_size = cfg.max_energy / cfg.base_bin_div,
                          zero_wind = (w == :nowind), key = (:radius, w, "const400", R))
            push!(rows, merge((Wind = String(w), Case = "const400", Rmax = R), r))
        end
    end
    df = add_errors_grouped(DataFrame(rows))
    CSV.write(conv_out(csv), df); print_table("Radius", df, :Rmax)
    return (df = df, dom = dom, xs = sort(radii), case = "const400")
end

# ---- (c) energy bin
function run_bin_small(cfg::ConvCfg = ConvCfg();
        divs = [10, 20, 50, 100, 200, 400, 800], csv = "conv_bin_small.csv", tol_pct = 0.1)
    dom = make_domain(cfg, cfg.base_step)
    rows = NamedTuple[]
    for w in WIND_MODES
        zw = (w == :nowind)
        for dv in sort(divs)
            println("\n--- [$w] const 400 kW, bins = $dv ---")
            r = solve_row(cfg, dom; Rmax = cfg.base_R, power_levels = [cfg.const_kw],
                          max_energy = cfg.max_energy, bin_size = cfg.max_energy / dv,
                          zero_wind = zw, key = (:bin, w, "const400", dv))
            push!(rows, merge((Wind = String(w), Case = "const400", BinDiv = dv), r))
        end
        ref = solve_row(cfg, dom; Rmax = cfg.base_R, power_levels = [cfg.const_kw],
                        max_energy = cfg.max_energy, bin_size = cfg.max_energy / cfg.base_bin_div,
                        zero_wind = zw)
        budget = cfg.budget_slack * ref.EnergyKwh
        println("\n[$w] 400 kW energy = $(round(ref.EnergyKwh,digits=1)) kWh -> variable budget $(round(budget,digits=1)) kWh")
        for dv in sort(divs)
            println("\n--- [$w] variable power, budget bins = $dv ---")
            r = solve_row(cfg, dom; Rmax = cfg.base_R, power_levels = cfg.var_levels,
                          max_energy = budget, bin_size = budget / dv,
                          zero_wind = zw, key = (:bin, w, "variable", dv))
            push!(rows, merge((Wind = String(w), Case = "variable", BinDiv = dv), r))
        end
    end
    df = add_errors_grouped(DataFrame(rows))
    CSV.write(conv_out(csv), df)
    print_table("Energy bin", df, :BinDiv)
    for w in WIND_MODES, c in ("const400", "variable")
        sub = df[(df.Wind .== String(w)) .& (df.Case .== c), :]
        ok = sub[sub.Feasible .&& abs.(sub.ErrPct) .<= tol_pct, :]
        isempty(ok) || println("  [$w/$c] coarsest bin count within $(tol_pct)%: $(minimum(ok.BinDiv)) bins")
    end
    return (df = df, dom = dom, xs = sort(divs), case = "variable")
end

# ---- plots
function plot_metric(g, r, b, cfg::ConvCfg, col::Symbol, ylabel::String, file::String)
    fig = CM.Figure(size = (1500, 800))
    specs = ((g.df, :Step,   "Grid step (°)",             cfg.base_step,    true,  true),
             (r.df, :Rmax,   "R_max (cells)",             cfg.base_R,       false, false),
             (b.df, :BinDiv, "Energy bins across budget", cfg.base_bin_div, true,  false))
    for (ri, w) in enumerate(WIND_MODES)
        axs = []
        for (k, (df, xc, xl, chosen, logx, rev)) in enumerate(specs)
            ax = CM.Axis(fig[ri, k]; xlabel = xl,
                         ylabel = k == 1 ? "$ylabel ($(w == :wind ? "wind" : "no wind"))" : "",
                         xscale = logx ? log10 : identity, xreversed = rev)
            for c in unique(df.Case)
                s = df[(df.Wind .== String(w)) .& (df.Case .== c) .& df.Feasible, :]
                isempty(s) && continue
                CM.lines!(ax, s[!, xc], s[!, col]; label = c)
                CM.scatter!(ax, s[!, xc], s[!, col])
            end
            CM.vlines!(ax, [chosen]; color = :red, linestyle = :dash)
            k == 3 && ri == 1 && CM.axislegend(ax; position = :rt)
            push!(axs, ax)
        end
        CM.linkyaxes!(axs...)
    end
    CM.save(conv_out(file), fig); println("Saved ", conv_out(file))
    return fig
end

const ROUTE_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231",
                      "#911eb7", "#00b5c8", "#f032e6", "#000000"]

function plot_routes(study::Symbol, st, cfg::ConvCfg, file::String, label_fmt = string)
    gp = st.dom.grid_points
    lons = [gp[1, j][1] for j in 1:size(gp, 2)]
    lats = [gp[i, 1][2] for i in 1:size(gp, 1)]
    n = length(st.xs)
    widths = n == 1 ? [3.0] : collect(range(6.0, 1.5, length = n))
    fig = CM.Figure(size = (1500, 650))
    for (c, w) in enumerate(WIND_MODES)
        ax = CM.Axis(fig[1, c]; title = "$study — $(w == :wind ? "with wind" : "no wind") ($(st.case))",
                     xlabel = "lon", ylabel = "lat", aspect = CM.DataAspect())
        CM.heatmap!(ax, lons, lats, Float32.(st.dom.mask)'; colormap = [:lightgray, :white])
        for (k, x) in enumerate(st.xs)
            p = get(ROUTES, (study, w, st.case, x), nothing)
            p === nothing && continue
            CM.lines!(ax, first.(p), last.(p); color = ROUTE_COLORS[mod1(k, length(ROUTE_COLORS))],
                      linewidth = widths[k], label = label_fmt(x))
        end
        CM.lines!(ax, [cfg.start_location[1], cfg.end_location[1]],
                      [cfg.start_location[2], cfg.end_location[2]];
                  color = :gray30, linestyle = :dot, linewidth = 2, label = "endpoint great circle")
        c == 2 && CM.axislegend(ax; position = :lb, labelsize = 11)
    end
    CM.save(conv_out(file), fig); println("Saved ", conv_out(file))
    return fig
end

# ---- heading ablation (all headings vs 8 compass headings), reuses the radius study
function run_ablation_small(cfg::ConvCfg, rad; csv = "conv_ablation_small.csv")
    dom, radii = rad.dom, rad.xs
    rows = NamedTuple[]
    for r in eachrow(rad.df)
        push!(rows, (Headings = "all", Wind = r.Wind, Rmax = r.Rmax, Feasible = r.Feasible,
                     TimeH = r.TimeH, DistKm = r.DistKm, WallS = r.WallS))
    end
    for R in radii, w in WIND_MODES
        println("\n--- 8 headings, Rmax=$R, [$w] ---")
        r = solve_row(cfg, dom; Rmax = R, power_levels = [cfg.const_kw],
                      max_energy = cfg.max_energy, bin_size = cfg.max_energy / cfg.base_bin_div,
                      zero_wind = (w == :nowind), heading_mode = :eight)
        push!(rows, (Headings = "eight", Wind = String(w), Rmax = R, Feasible = r.Feasible,
                     TimeH = r.TimeH, DistKm = r.DistKm, WallS = r.WallS))
    end
    df = DataFrame(rows)
    df.GapPct   = fill(NaN, nrow(df))     # vs all-headings at the largest R, same wind mode
    df.DPrevPct = fill(NaN, nrow(df))
    for w in unique(df.Wind)
        ref = df[(df.Headings .== "all") .& (df.Wind .== w) .& (df.Rmax .== maximum(radii)), :TimeH][1]
        for h in unique(df.Headings)
            idx = findall((df.Headings .== h) .& (df.Wind .== w))
            idx = idx[sortperm(df.Rmax[idx])]
            prev = NaN
            for i in idx
                df.Feasible[i] || continue
                df.GapPct[i] = 100 * (df.TimeH[i] - ref) / ref
                df.DPrevPct[i] = isnan(prev) ? NaN : 100 * (df.TimeH[i] - prev) / prev
                prev = df.TimeH[i]
            end
        end
    end
    CSV.write(conv_out(csv), df)
    println("\n=== Ablation ===")
    show(stdout, select(df, :Headings, :Wind, :Rmax, :TimeH, :DistKm, :GapPct, :DPrevPct, :WallS); allrows = true)
    println()
    return df
end

function plot_ablation_small(df::DataFrame, cfg::ConvCfg; file = "ablation_radius.png")
    fig = CM.Figure(size = (1400, 550))
    sty = Dict("all" => ("#4363d8", "all headings"), "eight" => ("#e6194b", "8 compass headings"))
    axs = []
    for (c, w) in enumerate(WIND_MODES)
        ax = CM.Axis(fig[1, c]; xlabel = "R_max (cells)",
                     ylabel = c == 1 ? "Time gap vs all headings, largest R_max (%)" : "",
                     title = w == :wind ? "With wind" : "No wind")
        for h in ("all", "eight")
            s = sort(df[(df.Headings .== h) .& (df.Wind .== String(w)) .& df.Feasible, :], :Rmax)
            isempty(s) && continue
            CM.lines!(ax, s.Rmax, s.GapPct; color = sty[h][1], linewidth = 3, label = sty[h][2])
            CM.scatter!(ax, s.Rmax, s.GapPct; color = sty[h][1], markersize = 9)
        end
        CM.vlines!(ax, [cfg.base_R]; color = :gray40, linestyle = :dash)
        c == 1 && CM.axislegend(ax; position = :rt)
        push!(axs, ax)
    end
    CM.linkyaxes!(axs...)
    CM.save(conv_out(file), fig); println("Saved ", conv_out(file))
    return fig
end

"""Full convergence study. Uses the 6-hourly archive. `ablation = true` adds the heading ablation."""
function run_convergence(cfg::ConvCfg = ConvCfg(); steps = [2.0, 1.0, 0.5, 0.25, 0.2, 0.1], ablation = true)
    empty!(ROUTES)
    in_clean_archive() do                       # warm-up (compilation), not recorded
        solve_row(cfg, make_domain(cfg, 1.0); Rmax = 1, power_levels = [cfg.const_kw],
                  max_energy = cfg.max_energy, bin_size = cfg.max_energy / 20)
    end
    g = in_clean_archive(() -> run_grid_step_small(cfg; steps = steps))
    r = in_clean_archive(() -> run_radius_small(cfg))
    b = in_clean_archive(() -> run_bin_small(cfg))

    plot_metric(g, r, b, cfg, :ErrPct, "Time error vs finest (%)", "convergence_time_error.png")
    plot_routes(:grid,   g, cfg, "routes_grid.png",   x -> "step $(x)°")
    plot_routes(:radius, r, cfg, "routes_radius.png", x -> "Rmax $x")
    plot_routes(:bin,    b, cfg, "routes_bin.png",    x -> "$x bins")
    if ablation
        abl = in_clean_archive(() -> run_ablation_small(cfg, r))
        plot_ablation_small(abl, cfg)
    end
    return (grid = g, radius = r, bin = b)
end

# =============================================================================
# 3. VERIFICATION vs the 2025 baseline solver
#    (hourly 2025 archive; run in its own process: it sets the hourly reference source)
# =============================================================================
split_path(y, m, d, hhmm) = joinpath(SPLIT_WIND_DIR, "$(y*10000 + m*100 + d)_$(lpad(hhmm, 4, '0')).grib")

const VCFG = ConvCfg(start_hhmm = 0, start_date = Date(2025, 11, 1))   # hourly archive is 2025 only
const SIMPLE_STEP = 0.5
const SIMPLE_R    = 2

function describe_simple(cfg::ConvCfg = VCFG; step = SIMPLE_STEP, R = SIMPLE_R)
    dom = make_domain(cfg, step)
    gc_km = getDistance(cfg.start_location..., cfg.end_location...) / 1000
    println("simplified example: $(cfg.start_location) -> $(cfg.end_location), great circle $(round(gc_km, digits = 0)) km, " *
            "step $(step)°, Rmax $R, grid $(size(dom.grid_points)), $(count(dom.mask)) ocean cells of $(dom.n_cells), " *
            "start $(cfg.start_date) $(lpad(cfg.start_hhmm, 4, '0'))")
    return dom
end

# ---- baseline helpers: nearest-grid wind, direction = where the wind blows FROM
function wind_at(f::WindField, lon::Float64, lat::Float64)
    n  = length(f.lons)
    dx = f.lons[2] - f.lons[1]
    dy = f.lats[2] - f.lats[1]                       # sign-agnostic
    lw = f.lons[1] + mod(lon - f.lons[1], 360.0)
    i  = round(Int, (lw - f.lons[1]) / dx) + 1
    i > n && (i = 1)
    j  = clamp(round(Int, (lat - f.lats[1]) / dy) + 1, 1, length(f.lats))
    u = Float64(f.u_raw[i, j]); v = Float64(f.v_raw[i, j])
    return sqrt(u^2 + v^2), mod(rad2deg(atan(u, v)) + 180.0, 360.0)
end

function edgeCost(lon1, lat1, lon2, lat2, wind_speed, wind_dir, P_kw)
    polar_file = joinpath(POLAR_DIR, "RisePol$(round(Int, P_kw))kW.csv")
    polar_info = readPolar(polar_file)
    t_sec, _, _, boat_speed = GetTravelTime(lon1, lat1, lon2, lat2, wind_speed, wind_dir, polar_info)
    return t_sec, P_kw * t_sec / 3600.0, boat_speed
end

energy_bin_array_index(e::Float64, bin_size::Float64) = Int(floor(e / bin_size)) + 1

function line_on_water(i::Int, j::Int, ni::Int, nj::Int, ocean_mask::AbstractMatrix{Bool}; n_checks_min::Int = 20)::Bool
    N = max(abs(ni - i), abs(nj - j))
    n_checks = max(N * 4, n_checks_min)
    for k in 1:n_checks
        t  = k / (n_checks + 1)
        ti = clamp(round(Int, i + t * (ni - i)), 1, size(ocean_mask, 1))
        tj = clamp(round(Int, j + t * (nj - j)), 1, size(ocean_mask, 2))
        !ocean_mask[ti, tj] && return false
    end
    return true
end

const BASE_RELOADS = Ref(0)
const SPLIT_GRID = Ref{Any}(nothing)

# The 2025 getWindAngleSpeedFromVals assumes a global 0.25° grid; the split files are regional,
# so the grid origin is read once from a split file.
function split_grid_info(path::String)
    GRIB.GribFile(path) do f
        for msg in f
            if msg["name"] == "10 metre U wind component"
                lons, lats, _ = data(msg)
                lu = sort(unique(lons)); la = sort(unique(lats))
                return (lon0 = lu[1], lat_top = la[end], dx = lu[2] - lu[1],
                        n_lon = length(lu), n_lat = length(la))
            end
        end
        error("no 10u message in $path")
    end
end

function regional_wind(u, v, lon::Float64, lat::Float64, g)
    lw = g.lon0 + mod(lon - g.lon0, 360.0)
    i  = clamp(round(Int, (lw - g.lon0) / g.dx) + 1, 1, g.n_lon)
    j  = clamp(round(Int, (g.lat_top - lat) / g.dx) + 1, 1, g.n_lat)
    uu = Float64(u[i, j]); vv = Float64(v[i, j])
    return sqrt(uu^2 + vv^2), mod(rad2deg(atan(uu, vv)) + 180.0, 360.0)
end

# ---- the 2025 baseline solver (algorithm unchanged; wind from split files)
function findDijkstraRouteRichPowerBin(
        grid_points::Matrix{Tuple{Float64, Float64}}, start_idx::Tuple{Int, Int}, goal_idx::Tuple{Int, Int},
        radius::Int, cur_time::Int, cur_day::Int, cur_month::Int, cur_year::Int,
        max_energy::Float64, power_levels::Vector{}, bin_size::Float64, min_speed::Float64,
        ocean_mask::Union{Nothing, Matrix{Bool}} = nothing)

    n_rows, n_cols = size(grid_points)
    previous_wind_file = ""
    u_vals = nothing; v_vals = nothing

    num_bins = ceil(Int, max_energy / bin_size) + 1
    start_bin = 1
    labels_time   = fill(Inf, n_rows, n_cols, num_bins)
    labels_energy = fill(Inf, n_rows, n_cols, num_bins)
    labels_power  = fill(-1.0, n_rows, n_cols, num_bins)
    labels_time[start_idx[1], start_idx[2], start_bin]   = 0.0
    labels_energy[start_idx[1], start_idx[2], start_bin] = 0.0
    prev = fill((Inf, Inf, Inf), n_rows, n_cols, num_bins)

    pq = PriorityQueue{Tuple{Int, Int, Int}, Float64}()
    enqueue!(pq, (start_idx[1], start_idx[2], start_bin) => 0.0)

    while !isempty(pq)
        (i, j, e_bin), cur_seconds = dequeue_pair!(pq)

        if (i, j) == goal_idx
            path_idxs = [(i, j, e_bin)]
            while true
                pi_, pj_, pe_ = prev[path_idxs[end]...]
                pi_ == Inf && break
                push!(path_idxs, (pi_, pj_, pe_))
            end
            path_idxs    = reverse(path_idxs)
            path_coords  = [grid_points[ii, jj] for (ii, jj, _) in path_idxs]
            node_costs   = [labels_time[ii, jj, eb] for (ii, jj, eb) in path_idxs]
            final_energy = [labels_energy[ii, jj, eb] for (ii, jj, eb) in path_idxs]
            power_used   = [labels_power[ii, jj, eb] for (ii, jj, eb) in path_idxs][2:end]
            push!(power_used, 0.0)
            return path_coords, node_costs, final_energy, power_used
        end

        lon1, lat1 = grid_points[i, j]
        cur_seconds = round(Int, cur_seconds)
        ny, nm, nd, ut = addSeconds(cur_time, cur_day, cur_month, cur_year, cur_seconds)
        ny, nm, nd, ut = roundTime(ut, nd, nm, ny)

        wind_file = split_path(ny, nm, nd, ut)
        if wind_file != previous_wind_file
            file = GRIB.GribFile(wind_file)
            u_vals, v_vals = extractWindValues(file, ut)
            previous_wind_file = wind_file
            destroy(file)
            BASE_RELOADS[] += 1
            println(Dates.format(now(), "HH:MM:SS"), "  wind ", ny, "-", lpad(nm, 2, '0'), "-",
                    lpad(nd, 2, '0'), " ", lpad(ut, 4, '0'), "  reload #", BASE_RELOADS[], "  queue=", length(pq))
        end
        wind_speed, wind_dir = regional_wind(u_vals, v_vals, lon1, lat1, SPLIT_GRID[])

        for (ni, nj) in neighbor_indices(i, j, radius, n_rows, n_cols)
            !ocean_mask[ni, nj] && continue
            !line_on_water(i, j, ni, nj, ocean_mask) && continue
            lon2, lat2 = grid_points[ni, nj]

            for P_kw in power_levels
                t_sec, change_E, boat_speed = edgeCost(lon1, lat1, lon2, lat2, wind_speed, wind_dir, P_kw)
                boat_speed < min_speed && continue

                new_time   = cur_seconds + t_sec
                new_energy = labels_energy[i, j, e_bin] + change_E
                new_energy > max_energy && continue

                new_bin = energy_bin_array_index(new_energy, bin_size)
                (new_bin > num_bins || new_bin < 1) && continue
                labels_time[ni, nj, new_bin] <= new_time && continue

                labels_time[ni, nj, new_bin]   = new_time
                labels_energy[ni, nj, new_bin] = new_energy
                labels_power[ni, nj, new_bin]  = P_kw
                if haskey(pq, (ni, nj, new_bin))
                    pq[(ni, nj, new_bin)] = new_time
                else
                    enqueue!(pq, (ni, nj, new_bin) => new_time)
                end
                prev[ni, nj, new_bin] = (i, j, e_bin)
            end
        end
    end
    println("No feasible path found within energy budget.")
    return [], [], [], []
end

# ---- wind source (hourly 2025 archive). Idempotent; the month cache is kept between calls.
function ensure_hourly!()
    _HOURLY_REFERENCE_SOURCE[] !== nothing && return nothing
    reset_sources!()
    set_hourly_reference_source!(; dir = HOURLY_REFERENCE_DIR)
    return nothing
end

function check_hourly_source(cfg::ConvCfg = VCFG)
    ensure_hourly!()
    pool = WindFieldPool(; max_cached = 2)
    ep = _ensureEpochRegistered!(pool, cfg.start_hhmm, day(cfg.start_date), month(cfg.start_date),
                                 year(cfg.start_date), 0)
    f = _getField!(pool, ep)
    println("u_raw size $(size(f.u_raw)); lons $(length(f.lons)), lats $(length(f.lats)) -> ",
            size(f.u_raw) == (length(f.lons), length(f.lats)) ? "[lon, lat] (expected)" : "UNEXPECTED layout")
    println("epoch $ep = $(Dates.unix2datetime(ep)) | lon $(extrema(f.lons)) lat $(extrema(f.lats)) | u range $(extrema(f.u_raw))")
    src = _HOURLY_REFERENCE_SOURCE[]
    ks = sort(collect(keys(src.month_cache[(year(cfg.start_date), month(cfg.start_date))])))
    println("snapshots this month: $(length(ks)), spacing (s): $(unique(diff(ks)))   (expect ~720 and [3600])")
    println("field size: $(round((sizeof(f.u_raw) + sizeof(f.v_raw)) / 2^20, digits = 1)) MiB/hour -> month cache ~ " *
            "$(round(length(ks) * (sizeof(f.u_raw) + sizeof(f.v_raw)) / 2^30, digits = 1)) GiB")
    return f
end

function check_split_wind(cfg::ConvCfg = VCFG)
    ensure_hourly!()
    pool = WindFieldPool(; max_cached = 2)
    ep = _ensureEpochRegistered!(pool, cfg.start_hhmm, day(cfg.start_date), month(cfg.start_date), year(cfg.start_date), 0)
    f = _getField!(pool, ep)
    p = split_path(year(cfg.start_date), month(cfg.start_date), day(cfg.start_date), cfg.start_hhmm)
    SPLIT_GRID[] = split_grid_info(p)
    println("split grid: ", SPLIT_GRID[])
    file = GRIB.GribFile(p)
    u, v = extractWindValues(file, cfg.start_hhmm); destroy(file)
    ds = 0.0; da = 0.0
    for lon in range(-97, -60, length = 40), lat in range(24, 32, length = 20)
        s1, a1 = wind_at(f, Float64(lon), Float64(lat))
        s2, a2 = regional_wind(u, v, Float64(lon), Float64(lat), SPLIT_GRID[])
        ds = max(ds, abs(s1 - s2)); da = max(da, abs(mod(a1 - a2 + 180, 360) - 180))
    end
    ok = ds < 1e-4 && da < 1e-3
    println("split-file wind vs pool wind_at: max |Δspeed| = $ds, max |Δdir| = $da  ", ok ? "PASS" : "FAIL")
    return ok
end

# wind_at must agree with the 2025 getWindAngleSpeedFromVals (lon ascending, lat from north)
function check_wind_convention(cfg::ConvCfg = VCFG)
    f = check_hourly_source(cfg)
    if !(length(f.lons) == 1441 && length(f.lats) == 721 && isapprox(f.lons[1], -180) && isapprox(f.lons[end], 180))
        println("field is not the global -180..180, 0.25° grid; compare wind_at against an old .grib by hand")
        return nothing
    end
    flip = f.lats[1] < f.lats[end]
    u = flip ? f.u_raw[:, end:-1:1] : f.u_raw
    v = flip ? f.v_raw[:, end:-1:1] : f.v_raw
    ds = 0.0; da = 0.0
    for lon in range(-97, -80, length = 15), lat in range(24, 31, length = 10)
        s1, a1 = wind_at(f, Float64(lon), Float64(lat))
        s2, a2 = getWindAngleSpeedFromVals(u, v, Float64(lon), Float64(lat))
        ds = max(ds, abs(s1 - s2)); da = max(da, abs(mod(a1 - a2 + 180, 360) - 180))
    end
    ok = ds < 1e-4 && da < 1e-3
    println("wind_at vs getWindAngleSpeedFromVals: max |Δspeed| = $ds, max |Δdir| = $da  ", ok ? "PASS" : "FAIL")
    return ok
end

# ---- solver wrappers
function run_baseline(cfg::ConvCfg, dom; R, levels, max_energy, bin_size)
    ensure_hourly!()
    SPLIT_GRID[] = split_grid_info(split_path(year(cfg.start_date), month(cfg.start_date),
                                              day(cfg.start_date), cfg.start_hhmm))
    BASE_RELOADS[] = 0
    res, wall = run_timed(() -> findDijkstraRouteRichPowerBin(
        dom.grid_points, dom.start_idx, dom.goal_idx, R,
        cfg.start_hhmm, day(cfg.start_date), month(cfg.start_date), year(cfg.start_date),
        max_energy, levels, bin_size, cfg.min_speed, dom.mask))
    println("baseline wind reloads: $(BASE_RELOADS[])")
    return (path = res[1], cost = res[2], energy = res[3], power = res[4], wall = wall, diag = nothing)
end

function solve_full(cfg::ConvCfg, dom; Rmax, power_levels, max_energy, bin_size,
                    stepped_wind = true, time_aware_wind = true, force_band = 1,
                    zero_wind = false, heading_mode = :full)
    ensure_hourly!()
    diag = Dict{Symbol,Any}()
    res, wall = run_timed(() -> findDijkstraRoutefinal(
        dom.grid_points, dom.start_idx, dom.goal_idx,
        cfg.start_hhmm, day(cfg.start_date), month(cfg.start_date), year(cfg.start_date),
        max_energy, power_levels, bin_size, cfg.min_speed;
        ocean_mask = dom.mask, Rmax = Rmax, force_band = force_band, dynamic_bands = false,
        heading_mode = heading_mode, stepped_wind = stepped_wind, time_aware_wind = time_aware_wind,
        accurate_stepped_all_bands = true, zero_wind = zero_wind,
        record_diagnostics = false, diagnostics = false, wind_cache_size = 100, diag_out = diag))
    return (path = res[1], cost = res[2], energy = res[3], power = res[4], wall = wall, diag = diag)
end

function vrow(name, r)
    isempty(r.path) && return (Variant = name, Feasible = false, TimeH = NaN, DistKm = NaN,
        EnergyKwh = NaN, Waypoints = 0, WallS = r.wall, Confirmed = 0, WestHops = 0)
    t, d, _, e = _route_stats_power(r.path, r.cost, r.energy, r.power)
    pidx = r.diag === nothing ? nothing : r.diag[:path_idxs]
    west = pidx === nothing ? 0 : count(k -> pidx[k+1][2] < pidx[k][2], 1:length(pidx)-1)
    conf = r.diag === nothing ? 0 : get(r.diag, :total_confirmed, 0)
    return (Variant = name, Feasible = true, TimeH = t, DistKm = d, EnergyKwh = e,
            Waypoints = length(r.path), WallS = r.wall, Confirmed = conf, WestHops = west)
end

# ---- replay on the arc-sampled, time-aware cost model
function replay_solver_path(r)
    ensure_hourly!()
    st = replay_route_real_wind(r.diag, r.path, r.power; planned_time_s = r.cost[end])
    s = st.steps[end]
    return (TimeH = s.cum_time_s / 3600, DistKm = s.cum_dist_km, EnergyKwh = s.cum_energy_kwh)
end

function replay_cells(cfg::ConvCfg, dom, cells, powers, nb; planned_s)
    ensure_hourly!()
    gp = dom.grid_points
    diag = Dict{Symbol,Any}(:grid_points => gp, :ocean_mask => dom.mask,
        :path_idxs => [(i, j, 0) for (i, j) in cells],
        :cur_time => cfg.start_hhmm, :cur_day => day(cfg.start_date),
        :cur_month => month(cfg.start_date), :cur_year => year(cfg.start_date), :nb_flat => nb)
    st = replay_route_real_wind(diag, [gp[i, j] for (i, j) in cells], powers; planned_time_s = planned_s)
    s = st.steps[end]
    return (TimeH = s.cum_time_s / 3600, DistKm = s.cum_dist_km, EnergyKwh = s.cum_energy_kwh)
end

# ---- baseline cost estimate
function time_edgecost(cfg::ConvCfg, dom, levels; reps = 300)
    gp = dom.grid_points; i, j = dom.start_idx
    lon1, lat1 = gp[i, j]; lon2, lat2 = gp[i, min(j + 2, size(gp, 2))]
    edgeCost(lon1, lat1, lon2, lat2, 8.0, 90.0, levels[end])          # compile + warm file cache
    t = @elapsed for _ in 1:reps
        edgeCost(lon1, lat1, lon2, lat2, 8.0, 90.0, levels[end])
    end
    return t / reps
end

# ESTIMATE: seconds = pops x candidate hops per state (baseline: only nj >= j) x power levels x cost per evaluation
function baseline_estimate(t_eval, confirmed, nb_flat, mask, K)
    cnt = [count(e -> e[2] >= j, nb_flat[i, j]) for i in axes(nb_flat, 1), j in axes(nb_flat, 2) if mask[i, j]]
    return confirmed * mean(cnt) * K * t_eval
end

# ---- TEST A: baseline vs improved (same hourly wind)
function run_baseline_vs_improved(cfg::ConvCfg = VCFG; step = SIMPLE_STEP, R = SIMPLE_R,
        levels = [cfg.const_kw], max_energy = cfg.max_energy,
        bin_size = max_energy / cfg.base_bin_div, csv = "ver_baseline_vs_improved.csv",
        baseline_limit_min = 600)
    ensure_hourly!()
    dom = make_domain(cfg, step)
    solve_full(cfg, make_domain(cfg, 2.0); Rmax = 1, power_levels = levels,
               max_energy = max_energy, bin_size = max_energy / 20)            # warm-up

    variants = [
        ("improved: 2 samples, wind frozen at departure",     (stepped_wind = false, time_aware_wind = false, force_band = 1)),
        ("improved: arc sampling, wind frozen at departure",  (stepped_wind = true,  time_aware_wind = false, force_band = 1)),
        ("improved: arc sampling, time-aware",                (stepped_wind = true,  time_aware_wind = true,  force_band = 1)),
        ("improved: production bands (calibrated at t0)",     (stepped_wind = true,  time_aware_wind = true,  force_band = nothing)),
    ]
    imp = Dict{String,Any}()
    for (name, kw) in variants
        println("\n--- $name ---")
        imp[name] = solve_full(cfg, dom; Rmax = R, power_levels = levels,
                               max_energy = max_energy, bin_size = bin_size, kw...)
    end

    ref = imp[variants[3][1]]
    t_eval = time_edgecost(cfg, dom, levels)
    est = baseline_estimate(t_eval, ref.diag[:total_confirmed], ref.diag[:nb_flat], dom.mask, length(levels))
    if est / 60 > baseline_limit_min
        println("Skipping the baseline run: estimated $(round(est / 60, digits = 1)) min > limit $(baseline_limit_min) min.")
        return nothing
    end

    base = run_baseline(cfg, dom; R, levels, max_energy, bin_size)
    rows = NamedTuple[vrow("baseline (2025 solver)", base)]
    for (name, _) in variants
        push!(rows, vrow(name, imp[name]))
    end
    df = DataFrame(rows)
    df.DTimePct = 100 .* (df.TimeH .- df.TimeH[1]) ./ df.TimeH[1]
    df.Speedup = df.WallS[1] ./ df.WallS

    # prediction accuracy: solver's own time vs the same route replayed on hourly wind
    gp = dom.grid_points; n_rows, n_cols = size(gp)
    nb = build_neighbour_cache(gp, dom.mask, n_rows, n_cols, R, 27_800.0)
    lookup = Dict(gp[i, j] => (i, j) for i in 1:n_rows, j in 1:n_cols)
    replay_h = Float64[replay_cells(cfg, dom, [lookup[c] for c in base.path], base.power[1:end-1], nb;
                                    planned_s = base.cost[end]).TimeH]
    for (name, _) in variants
        push!(replay_h, replay_solver_path(imp[name]).TimeH)
    end
    df.ReplayH = replay_h
    df.PredErrPct = 100 .* (df.ReplayH .- df.TimeH) ./ df.TimeH

    CSV.write(ver_out(csv), df); show(stdout, df; allrows = true); println()
    return df
end

# ---- TEST B: zero-wind oracle (machinery is exact)
function zero_wind_oracle(cfg::ConvCfg = VCFG; step = SIMPLE_STEP, R = SIMPLE_R)
    dom = make_domain(cfg, step)
    r = solve_full(cfg, dom; Rmax = R, power_levels = [cfg.const_kw],
            max_energy = cfg.max_energy, bin_size = cfg.max_energy / cfg.base_bin_div, zero_wind = true)
    ws, wa, bs = readPolar(joinpath(POLAR_DIR, "RisePol$(round(Int, cfg.const_kw))kW.csv"))
    lut = build_polar_lut((ws, wa, bs))
    v = [Float64(lut_lookup(lut, 0.0, Float64(a))) for a in 0.0:15.0:180.0]
    spread = (maximum(v) - minimum(v)) / minimum(v)
    spread > 1e-6 && @warn "no-wind speed depends on TWA ($spread); oracle below is not valid"

    n_rows, n_cols = size(dom.grid_points)
    nb = build_neighbour_cache(dom.grid_points, dom.mask, n_rows, n_cols, R, 27_800.0)
    D = fill(Inf, n_rows, n_cols); D[dom.start_idx...] = 0.0
    pq = PriorityQueue{Tuple{Int,Int},Float64}(); pq[dom.start_idx] = 0.0
    while !isempty(pq)
        (i, j), d = dequeue_pair!(pq)
        (i, j) == dom.goal_idx && break
        for (ni, nj, dist, _, _, _) in nb[i, j]
            nd = d + dist
            if nd < D[ni, nj]
                D[ni, nj] = nd; pq[(ni, nj)] = nd
            end
        end
    end
    t_oracle = D[dom.goal_idx...] / v[1]
    rel = abs(r.cost[end] - t_oracle) / t_oracle
    println("zero-wind: solver $(round(r.cost[end] / 3600, digits = 4)) h, " *
            "independent Dijkstra $(round(t_oracle / 3600, digits = 4)) h, rel. diff $rel  " *
            (rel < 1e-6 ? "PASS" : "FAIL"))
    return rel
end

# ---- TEST C: independent baselines + replay consistency
great_circle_cells(cfg::ConvCfg, dom; n = 400) =
    great_circle_cells_ll(dom.grid_points, dom.mask, dom.start_idx, dom.goal_idx,
                          cfg.start_location, cfg.end_location; n = n)

function run_independent_baselines(cfg::ConvCfg = VCFG; step = SIMPLE_STEP, R = SIMPLE_R,
                                   csv = "ver_baselines.csv")
    ensure_hourly!()
    dom = make_domain(cfg, step)
    big = 2_000_000.0            # non-binding; with one power level the label order ignores bin width
    rows = NamedTuple[]
    checks = Tuple{String,Bool}[]

    sols = Dict{Float64,Any}()
    for P in (cfg.const_kw, 1200.0, 2000.0)
        sols[P] = solve_full(cfg, dom; Rmax = R, power_levels = [P], max_energy = big, bin_size = big / 10)
        push!(rows, vrow("constant $(Int(P)) kW, optimal route", sols[P]))
    end
    B = cfg.budget_slack * rows[1].EnergyKwh
    vs = solve_full(cfg, dom; Rmax = R, power_levels = cfg.var_levels, max_energy = B, bin_size = B / cfg.base_bin_div)
    push!(rows, vrow("variable power, optimal route", vs))

    n_rows, n_cols = size(dom.grid_points)
    nb = build_neighbour_cache(dom.grid_points, dom.mask, n_rows, n_cols, R, 27_800.0)
    cells = great_circle_cells(cfg, dom)
    n_land = count(k -> !line_on_water_geo(cells[k][1], cells[k][2], cells[k+1][1], cells[k+1][2], dom.mask;
                                           oversample = 10), 1:length(cells)-1)
    gc = replay_cells(cfg, dom, cells, fill(cfg.const_kw, length(cells) - 1), nb;
                      planned_s = 2 * sols[cfg.const_kw].cost[end])
    push!(rows, (Variant = "great circle, 400 kW ($n_land hops cross land)", Feasible = true, TimeH = gc.TimeH,
                 DistKm = gc.DistKm, EnergyKwh = gc.EnergyKwh, Waypoints = length(cells),
                 WallS = NaN, Confirmed = 0, WestHops = 0))

    for (name, r) in (("constant 400 kW", sols[cfg.const_kw]), ("variable power", vs))
        rp = replay_solver_path(r)
        gap = 100 * (rp.TimeH * 3600 - r.cost[end]) / r.cost[end]
        println("replay vs solver, $name: solver $(round(r.cost[end]/3600, digits = 3)) h, " *
                "replay $(round(rp.TimeH, digits = 3)) h, gap $(round(gap, digits = 4))%")
        name == "constant 400 kW" && push!(checks, ("replay == solver (constant power, |gap| < 0.05%)", abs(gap) < 0.05))
    end
    t(P) = rows[findfirst(r -> startswith(r.Variant, "constant $(Int(P))"), rows)].TimeH
    push!(checks, ("optimal 400 kW <= great circle 400 kW", t(400.0) <= gc.TimeH * (1 + 1e-4)))
    push!(checks, ("time(2000) <= time(1200) <= time(400)", t(2000.0) <= t(1200.0) <= t(400.0)))
    push!(checks, ("variable <= constant 400 (budget 1.02 E400, 0.5% tol)", rows[4].TimeH <= t(400.0) * 1.005))

    df = DataFrame(rows)
    df.PctVsVariable = 100 .* (df.TimeH .- rows[4].TimeH) ./ rows[4].TimeH
    CSV.write(ver_out(csv), df); show(stdout, df; allrows = true); println()
    for (n, ok) in checks
        println(rpad(n, 62), ok ? "PASS" : "FAIL")
    end
    return df
end

# ---- verification drivers
function run_verification_all(; step = SIMPLE_STEP, R = SIMPLE_R)
    describe_simple(; step, R)
    println("\n===== 0. wind source =====")
    check_wind_convention() === false && error("wind_at disagrees with getWindAngleSpeedFromVals; fix before running Test A")
    println("\n===== B. zero-wind oracle ====="); zero_wind_oracle(; step, R)
    println("\n===== C. independent baselines ====="); run_independent_baselines(; step, R)
    println("\n===== A. baseline vs improved ====="); return run_baseline_vs_improved(; step, R)
end

"""Variable-power baseline-vs-improved on the small Gulf domain. Budget = slack x the 400 kW energy."""
function run_verification_variable(; step = SIMPLE_STEP, R = SIMPLE_R)
    attempt(check_split_wind, "split wind")
    attempt(() -> begin
        dom = make_domain(VCFG, step)
        c400 = solve_full(VCFG, dom; Rmax = R, power_levels = [VCFG.const_kw],
                          max_energy = 2_000_000.0, bin_size = 200_000.0)
        B = VCFG.budget_slack * c400.energy[end]
        println("variable budget = $(VCFG.budget_slack) x $(round(c400.energy[end], digits = 1)) kWh = $(round(B, digits = 1)) kWh")
        run_baseline_vs_improved(VCFG; step = step, R = R, levels = VCFG.var_levels,
                                 max_energy = B, bin_size = B / VCFG.base_bin_div,
                                 csv = "ver_variable_baseline_vs_improved.csv")
    end, "variable power: baseline vs improved (small Gulf domain)")
end

"""Full Houston->Bay-of-Biscay-side domain: improved solver wall time vs ESTIMATED baseline time."""
function run_verification_full_estimate()
    cfg = ConvCfg(start_hhmm = 0, start_date = Date(2025, 11, 1), end_location = END_LOCATION,
                  pad_deg = 5.0, max_energy = 600_000.0)
    dom = make_domain(cfg, 0.25)
    solve_full(cfg, make_domain(cfg, 2.0); Rmax = 1, power_levels = [400.0],
               max_energy = 600_000.0, bin_size = 30_000.0)               # warm-up
    r400 = solve_full(cfg, dom; Rmax = 4, power_levels = [400.0], max_energy = 600_000.0, bin_size = 6_000.0)
    est400 = baseline_estimate(time_edgecost(cfg, dom, [400.0]), r400.diag[:total_confirmed],
                               r400.diag[:nb_flat], dom.mask, 1) / 3600
    println("[c400] improved wall: ", round(r400.wall, digits = 1), " s | baseline estimate: ", round(est400, digits = 1),
            " h | speed-up (estimate): ~", round(est400 * 3600 / r400.wall, digits = 0), "x")

    B = 1.02 * r400.energy[end]
    rv = solve_full(cfg, dom; Rmax = 4, power_levels = LEVELS_ALL, max_energy = B, bin_size = B / 100)
    estv = baseline_estimate(time_edgecost(cfg, dom, LEVELS_ALL), rv.diag[:total_confirmed],
                             rv.diag[:nb_flat], dom.mask, length(LEVELS_ALL)) / 3600
    println("[variable] improved wall: ", round(rv.wall, digits = 1), " s | baseline estimate: ", round(estv, digits = 1),
            " h | speed-up (estimate): ~", round(estv * 3600 / rv.wall, digits = 0), "x")
end

# =============================================================================
# 4. ANALYSIS OF SAVED RUNS
#
# Needs run_status.csv and legs_*.csv from the pipeline. Run in a separate process
# from the pipeline / convergence / verification.
#   Part A  tables + plots from CSV/JLD2 only (time penalty, variable vs c400, energy,
#           leg metrics, rolling diagnostics)
#   Part B  policy maps + progress plots (needs the routing code)
#   Greedy  optional greedy comparison (needs greedy_route.jl)
# =============================================================================

function load_leg_tables(modes, tags)
    leg = Dict{Tuple{String,String,String},DataFrame}()    # (tag, mode, method)
    for md in modes, t in tags, m in vcat(METHODS, ["greedy_committed", "greedy_allforecast"])
        f = tab("legs_$(md)_$(m)_$(t).csv")
        isfile(f) || continue
        df = CSV.read(f, DataFrame)
        nrow(df) > 0 && (leg[(t, md, m)] = df)
    end
    return leg
end

function dense_pts(coords; k = 8)
    pts = Tuple{Float64,Float64}[]
    for a in 1:length(coords)-1
        (lo1, la1), (lo2, la2) = coords[a], coords[a+1]
        for s in 0:k-1
            f = s / k; push!(pts, (lo1 + f * (lo2 - lo1), la1 + f * (la2 - la1)))
        end
    end
    push!(pts, coords[end]); return pts
end
function plan_dev_km(newc, oldc)
    od = dense_pts(oldc)
    d = [minimum(hav_km(lo, la, o[1], o[2]) for o in od) for (lo, la) in newc]
    return (mean(d), maximum(d))
end

function rolling_diag(path::String)
    isfile(path) || return nothing
    rows = DataFrame(day = Int[], date = Date[], hours_sailed = Float64[], cycle_age_days = Int[],
                     lead_h = Float64[], beyond_336h = Bool[], plan_dev_mean_km = Float64[], plan_dev_max_km = Float64[])
    local reached, total_h, total_e
    jldopen(path, "r") do f
        n = f["shared/n_days"]
        reached = f["shared/reached_goal"]; total_h = f["shared/total_time_s"] / 3600
        total_e = f["shared/total_energy_kwh"]
        d1 = f["day_1/date"]; start_dt = DateTime(d1); last_cycle = d1 + Day(NDAYS - 1)
        clock_s = 0.0; prev = nothing
        for k in 1:n
            g = "day_$k"
            ts = f["$g/landed_time_s"]; coords = f["$g/fc_path_coords"]
            clock = start_dt + Millisecond(round(Int, clock_s * 1000))
            cyc = min(Date(clock), last_cycle)
            lead_h = Dates.value(clock - DateTime(cyc)) / 3.6e6
            dm, dx = prev === nothing ? (NaN, NaN) : plan_dev_km(coords, prev)
            push!(rows, (k, f["$g/date"], ts / 3600, Dates.value(Date(clock) - cyc), lead_h, lead_h > FC_MAX_H, dm, dx))
            clock_s += ts; prev = coords
        end
    end
    return (rows = rows, reached = reached, total_h = total_h, total_e = total_e)
end

# ---------------------------------------------------------------- Part A
function run_analysis_part_a()
    status    = load_run_status!()
    modes     = sort(unique(String.(status.mode)))
    var_modes = [md for md in modes if parse_mode(md) !== nothing]
    tags      = sort(unique(String.(status.tag)))
    mode_tags(md) = sort(unique([t for (t, mm, m) in keys(TIME) if mm == md && m == "hindsight"]))

    println("Modes found: ", join(modes, ", "))
    println("Variable-power modes: ", isempty(var_modes) ? "none yet" : join(var_modes, ", "))
    println("Windows: ", join(tags, ", "))
    isempty(setdiff(modes, [C400_MODE], var_modes)) ||
        println("Ignoring unrecognised modes: ", join(setdiff(modes, [C400_MODE], var_modes), ", "))

    LEG = load_leg_tables(modes, tags)

    # ---- 1. time penalty vs hindsight, per mode
    pen_rows = DataFrame(tag = String[], mode = String[], method = String[], time_h = Float64[],
                         energy_kwh = Float64[], reached = Bool[], penalty_pct = Float64[])
    for md in modes, t in mode_tags(md), m in METHODS
        isnan(timeof(t, md, m)) && continue
        push!(pen_rows, (t, md, m, timeof(t, md, m), energyof(t, md, m), get(REACH, (t, md, m), false), pen(t, md, m)))
    end
    CSV.write(ap("penalties_all_modes.csv"), pen_rows)

    function report_paired(name, v)
        v = filter(!isnan, v)
        isempty(v) && return println("  ", rpad(name, 44), "no data")
        lo, hi = boot_ci(v)
        println("  ", rpad(name, 44), "mean ", pad(mean(v), 8), "median ", pad(median(v), 8),
                "CI [", fmt(lo), ", ", fmt(hi), "]  wins ", count(>(0), v), "/", length(v))
    end

    summ   = DataFrame(mode = String[], policy = String[], n = Int[], mean_pen = Float64[], ci_lo = Float64[],
                       ci_hi = Float64[], median_pen = Float64[], min_pen = Float64[], max_pen = Float64[], n_dnf = Int[])
    paired = DataFrame(mode = String[], tag = String[], replan_gain_pp = Float64[], replan_recovered_pct = Float64[],
                       clim_gain_committed_pp = Float64[], clim_gain_rolling_pp = Float64[])
    for md in modes
        tg = mode_tags(md); isempty(tg) && continue
        println("\n=== 1. [$md] time penalty vs that mode's hindsight (%) ===")
        println(pad("window", 11), pad("hind (h)", 10), join([pad(LABEL[m], 19) for m in POLICIES]))
        for t in tg
            println(pad(t, 11), pad(timeof(t, md, "hindsight"), 10),
                    join([pad(fmt(pen(t, md, m)) * (get(REACH, (t, md, m), true) ? "" : " DNF"), 19) for m in POLICIES]))
        end
        for m in POLICIES
            v = filter(!isnan, [pen(t, md, m) for t in tg]); isempty(v) && continue
            lo, hi = boot_ci(v)
            push!(summ, (md, LABEL[m], length(v), mean(v), lo, hi, median(v), minimum(v), maximum(v),
                         count(t -> !get(REACH, (t, md, m), true), tg)))
        end
        pc  = [pen(t, md, "committed") for t in tg];      pr  = [pen(t, md, "rolling") for t in tg]
        pcc = [pen(t, md, "committed_clim") for t in tg]; prc = [pen(t, md, "rolling_clim") for t in tg]
        for (k, t) in enumerate(tg)
            push!(paired, (md, t, pc[k] - pr[k], pc[k] > 0 ? 100 * (pc[k] - pr[k]) / pc[k] : NaN,
                           pc[k] - pcc[k], pr[k] - prc[k]))
        end
        println("  paired (percentage points of penalty; positive = the second option is better)")
        report_paired("replanning gain (committed - rolling)", pc .- pr)
        report_paired("climatology gain, committed", pc .- pcc)
        report_paired("climatology gain, rolling", pr .- prc)
        length(tg) < 8 && println("  NOTE: n = $(length(tg)) windows; CIs are very wide, quote per-window values and win counts.")
    end
    CSV.write(ap("summary_all_modes.csv"), summ)
    CSV.write(ap("paired_all_modes.csv"), paired)
    println("\n", summ)

    # ---- 2. variable vs constant power, same policy
    cv = DataFrame(tag = String[], var_mode = String[], method = String[], c400_h = Float64[], var_h = Float64[],
                   saved_h = Float64[], saved_pct = Float64[], c400_pen_pct = Float64[], var_pen_pct = Float64[])
    for vm in var_modes, t in mode_tags(vm), m in METHODS
        tc = timeof(t, C400_MODE, m); tv = timeof(t, vm, m)
        (isnan(tc) || isnan(tv)) && continue
        push!(cv, (t, vm, m, tc, tv, tc - tv, 100 * (tc - tv) / tc, pen(t, C400_MODE, m), pen(t, vm, m)))
    end
    CSV.write(ap("variable_vs_c400.csv"), cv)
    println("\n=== 2. Variable power vs constant 400 kW (positive saved = variable is faster, same policy) ===")
    if isempty(cv)
        println("  no variable-power runs with a c400 counterpart yet")
    else
        println(pad("var mode", 14), pad("window", 10), pad("policy", 20), pad("c400 h", 9), pad("var h", 9),
                pad("saved h", 9), pad("saved %", 9))
        for r in eachrow(cv)
            println(pad(r.var_mode, 14), pad(r.tag, 10), pad(LABEL[r.method], 20), pad(r.c400_h, 9), pad(r.var_h, 9),
                    pad(r.saved_h, 9), pad(r.saved_pct, 9))
        end
        cv_sum = combine(groupby(cv, [:var_mode, :method]), nrow => :n, :saved_pct => mean => :mean_saved_pct,
                         :saved_pct => median => :median_saved_pct, :saved_pct => (x -> count(>(0), x)) => :wins)
        CSV.write(ap("variable_vs_c400_summary.csv"), cv_sum)
        println("\n", cv_sum)
    end

    # ---- 3a. realised energy vs budget (budget = slack x energy of the c400 hindsight route)
    bud = DataFrame(tag = String[], mode = String[], method = String[], budget_kwh = Float64[], energy_kwh = Float64[],
                    used_pct_of_budget = Float64[], over_budget = Bool[], time_h = Float64[])
    for md in var_modes
        pm = parse_mode(md)
        for t in mode_tags(md), m in METHODS
            e = energyof(t, md, m); isnan(e) && continue
            budget = pm.slack * energyof(t, C400_MODE, "hindsight")
            push!(bud, (t, md, m, budget, e, 100 * e / budget, e > budget * (1 + 1e-6), timeof(t, md, m)))
        end
    end
    CSV.write(ap("energy_vs_budget.csv"), bud)
    println("\n=== 3a. Realised energy vs budget (committed replays measure energy, they do not enforce the budget) ===")
    if isempty(bud)
        println("  no variable-power runs yet")
    else
        println(pad("mode", 14), pad("window", 10), pad("policy", 20), pad("budget kWh", 12), pad("used kWh", 12),
                pad("% of budget", 12), "over?")
        for r in eachrow(bud)
            println(pad(r.mode, 14), pad(r.tag, 10), pad(LABEL[r.method], 20), pad(r.budget_kwh, 12), pad(r.energy_kwh, 12),
                    pad(r.used_pct_of_budget, 12), r.over_budget ? "OVER" : "")
        end
    end

    # ---- 3b. plan margin vs no margin (same slack)
    mg = DataFrame(margin_mode = String[], base_mode = String[], tag = String[], method = String[],
                   time_base_h = Float64[], time_margin_h = Float64[], dtime_h = Float64[], dtime_pct = Float64[],
                   energy_base_kwh = Float64[], energy_margin_kwh = Float64[], budget_kwh = Float64[],
                   over_base = Bool[], over_margin = Bool[])
    for mm in var_modes
        pm = parse_mode(mm)
        pm.margin > 0 || continue
        bm = "var_s$(round(Int, 100 * pm.slack))_m0"
        if !(bm in modes)
            println("\n3b: no no-margin run $bm to compare $mm against"); continue
        end
        for t in mode_tags(mm), m in METHODS
            tb = timeof(t, bm, m); tmv = timeof(t, mm, m)
            (isnan(tb) || isnan(tmv)) && continue
            eb = energyof(t, bm, m); emv = energyof(t, mm, m)
            budget = pm.slack * energyof(t, C400_MODE, "hindsight")
            push!(mg, (mm, bm, t, m, tb, tmv, tmv - tb, 100 * (tmv - tb) / tb, eb, emv, budget,
                       eb > budget * (1 + 1e-6), emv > budget * (1 + 1e-6)))
        end
    end
    CSV.write(ap("margin_vs_no_margin.csv"), mg)
    println("\n=== 3b. Plan margin vs no margin ===")
    if isempty(mg)
        println("  no margin runs yet (looking for var_sXXX_mYY with YY > 0 next to var_sXXX_m0)")
    else
        mg_sum = combine(groupby(mg, [:margin_mode, :method]), nrow => :n, :dtime_pct => mean => :mean_dtime_pct,
                         :over_base => sum => :n_over_without_margin, :over_margin => sum => :n_over_with_margin)
        CSV.write(ap("margin_vs_no_margin_summary.csv"), mg_sum)
        println(mg_sum)
        println("A margin is worth it if it removes budget overruns (n_over) for a small time cost (mean_dtime_pct).")
    end

    # ---- 3c. where the energy is used (pooled over windows)
    bins_tbl = DataFrame(mode = String[], method = String[], wind_bin = String[], hours = Float64[], energy_kwh = Float64[],
                         mean_power_kw = Float64[], time_share_pct = Float64[], energy_share_pct = Float64[])
    lvl_tbl  = DataFrame(mode = String[], method = String[], power_kw = Float64[], time_share_pct = Float64[],
                         energy_share_pct = Float64[], mean_wind_ms = Float64[])
    pw_tbl   = DataFrame(mode = String[], method = String[], mean_power_kw = Float64[], mean_power_calm_kw = Float64[],
                         mean_power_windy_kw = Float64[], frac_time_engine_off = Float64[], spearman_power_wind = Float64[])
    for md in var_modes, m in METHODS
        dfs = [LEG[(t, md, m)] for t in tags if haskey(LEG, (t, md, m))]
        isempty(dfs) && continue
        pool = vcat(dfs...)
        Tt = sum(pool.dt_h); Et = sum(pool.energy_kwh)
        bi = bin_index.(pool.wind_ms)
        for b in 1:length(WS_LABELS)
            mk = bi .== b
            hrs = sum(pool.dt_h[mk]); en = sum(pool.energy_kwh[mk])
            push!(bins_tbl, (md, m, WS_LABELS[b], hrs, en, hrs > 0 ? en / hrs : NaN, 100 * hrs / Tt, 100 * en / Et))
        end
        li = level_index.(pool.power_kw)
        for (k, P) in enumerate(LEVELS_ALL)
            mk = li .== k; any(mk) || continue
            hrs = sum(pool.dt_h[mk])
            push!(lvl_tbl, (md, m, P, 100 * hrs / Tt, 100 * sum(pool.energy_kwh[mk]) / Et,
                            sum(pool.dt_h[mk] .* pool.wind_ms[mk]) / hrs))
        end
        calm = pool.wind_ms .< CALM_MS; windy = pool.wind_ms .>= WINDY_MS
        pwr(mk) = any(mk) ? sum(pool.energy_kwh[mk]) / sum(pool.dt_h[mk]) : NaN
        push!(pw_tbl, (md, m, Et / Tt, pwr(calm), pwr(windy), sum(pool.dt_h[pool.power_kw .< 1.0]) / Tt,
                       spearman(pool.wind_ms, pool.power_kw)))
    end
    CSV.write(ap("energy_by_wind_bin.csv"), bins_tbl)
    CSV.write(ap("power_level_shares.csv"), lvl_tbl)
    CSV.write(ap("power_vs_wind.csv"), pw_tbl)
    println("\n=== 3c. Where the energy is used (pooled over windows) ===")
    if isempty(pw_tbl)
        println("  no variable-power leg tables yet")
    else
        println("Power vs wind (Spearman < 0 = more power in lighter wind):"); println(pw_tbl)
        println("\nTime and energy share by power level:"); println(lvl_tbl)
    end

    # ---- 4. leg metrics, all modes
    legm = DataFrame(tag = String[], mode = String[], method = String[], legs_h = Float64[], status_h = Float64[],
                     dist_nm = Float64[], mean_speed_kn = Float64[], mean_wind_ms = Float64[], frac_calm = Float64[],
                     frac_windy = Float64[], mean_power_kw = Float64[], tvd_vs_hindsight = Float64[])
    for md in modes, t in tags, m in METHODS
        haskey(LEG, (t, md, m)) || continue
        lmx = leg_metrics(LEG[(t, md, m)])
        tvv = (m != "hindsight" && haskey(LEG, (t, md, "hindsight"))) ?
              tvd(wind_hist(LEG[(t, md, m)]), wind_hist(LEG[(t, md, "hindsight")])) : NaN
        push!(legm, (t, md, m, lmx.hours, timeof(t, md, m), lmx.dist_nm, lmx.mean_speed_kn, lmx.mean_wind_ms,
                     lmx.frac_calm, lmx.frac_windy, lmx.mean_power_kw, tvv))
    end
    CSV.write(ap("leg_metrics_all_modes.csv"), legm)
    println("\n=== 4. Leg metrics ===")
    println(select(legm, :tag, :mode, :method, :dist_nm, :mean_speed_kn, :mean_wind_ms, :frac_calm, :tvd_vs_hindsight))
    badl = legm[abs.(legm.legs_h .- legm.status_h) .> 0.05, :]
    nrow(badl) > 0 && (println("\nWARNING: leg-table time differs from run_status time for:");
                       println(select(badl, :tag, :mode, :method, :legs_h, :status_h)))

    # ---- 5. rolling diagnostics (straight from the .jld2)
    diag_sum = DataFrame(tag = String[], mode = String[], method = String[], reached = Bool[], n_replans = Int[],
                         voyage_days = Float64[], mean_h_per_replan = Float64[], replans_on_stale_cycle = Int[],
                         replans_beyond_336h = Int[], mean_plan_dev_km = Float64[], max_plan_dev_km = Float64[],
                         mean_power_kw = Float64[])
    for md in modes, t in tags, (m, fn) in (("rolling", "rolling_replan_$(md)_$(t).jld2"),
                                            ("rolling_clim", "rolling_replan_climatology_$(md)_$(t).jld2"))
        isfile(out(fn)) || continue
        try
            d = rolling_diag(out(fn)); d === nothing && continue
            rr = d.rows
            pdm = filter(!isnan, rr.plan_dev_mean_km); pdx = filter(!isnan, rr.plan_dev_max_km)
            push!(diag_sum, (t, md, m, d.reached, nrow(rr), d.total_h / 24, mean(rr.hours_sailed),
                             count(>(0), rr.cycle_age_days), count(rr.beyond_336h),
                             isempty(pdm) ? NaN : mean(pdm), isempty(pdx) ? NaN : maximum(pdx), d.total_e / d.total_h))
        catch e
            @warn "could not read $fn" exception = e
        end
    end
    CSV.write(ap("rolling_diagnostics_all_modes.csv"), diag_sum)
    println("\n=== 5. Rolling diagnostics ==="); println(diag_sum)

    # ---- 6. plots
    safeplot("penalty_by_window") do
        mds = [md for md in modes if !isempty(mode_tags(md))]
        isempty(mds) && return
        fig = CM.Figure(size = (1200, 380 * length(mds)))
        for (r, md) in enumerate(mds)
            tg = mode_tags(md); nt = length(tg)
            ax = CM.Axis(fig[r, 1]; xticks = (1:nt, tg), ylabel = "penalty vs hindsight (%)",
                         title = "$md: time penalty by window and policy")
            w = 0.2
            for (k, m) in enumerate(POLICIES)
                y = [pen(t, md, m) for t in tg]; ok = .!isnan.(y); any(ok) || continue
                xs = collect(1:nt) .+ (k - 2.5) * w
                CM.barplot!(ax, xs[ok], y[ok]; width = w, color = COLOR[m], label = LABEL[m])
            end
            CM.hlines!(ax, [0.0]; color = :gray50, linestyle = :dash)
            r == 1 && CM.axislegend(ax; position = :lt)
        end
        CM.save(ap("penalty_by_window.png"), fig)
    end

    safeplot("variable_vs_c400") do
        isempty(cv) && return
        fig = CM.Figure(size = (1100, 420 * length(var_modes)))
        for (r, vm) in enumerate(var_modes)
            sub = cv[cv.var_mode .== vm, :]; isempty(sub) && continue
            ax = CM.Axis(fig[r, 1]; xticks = (1:length(METHODS), [LABEL[m] for m in METHODS]),
                         ylabel = "time saved vs c400 (%)", title = "$vm vs constant 400 kW (same policy)")
            for (k, m) in enumerate(METHODS)
                s = sub[sub.method .== m, :]; isempty(s) && continue
                CM.barplot!(ax, [k], [mean(s.saved_pct)]; color = COLOR[m], width = 0.6)
                CM.scatter!(ax, fill(k, nrow(s)), s.saved_pct; color = :black, markersize = 8)
            end
            CM.hlines!(ax, [0.0]; color = :gray50, linestyle = :dash)
        end
        CM.save(ap("variable_vs_c400.png"), fig)
    end

    safeplot("energy_vs_budget") do
        isempty(bud) && return
        mds = unique(bud.mode)
        fig = CM.Figure(size = (1200, 380 * length(mds)))
        for (r, md) in enumerate(mds)
            sub = bud[bud.mode .== md, :]; tg = sort(unique(sub.tag)); nt = length(tg)
            ax = CM.Axis(fig[r, 1]; xticks = (1:nt, tg), ylabel = "realised energy (% of budget)",
                         title = "$md: energy use against the budget (red = 100%)")
            for (k, m) in enumerate(METHODS)
                y = [(s = sub[(sub.tag .== t) .& (sub.method .== m), :]; nrow(s) == 0 ? NaN : s.used_pct_of_budget[1]) for t in tg]
                ok = .!isnan.(y); any(ok) || continue
                xs = collect(1:nt) .+ (k - 3) * 0.15
                CM.barplot!(ax, xs[ok], y[ok]; width = 0.15, color = COLOR[m], label = LABEL[m])
            end
            CM.hlines!(ax, [100.0]; color = :red, linestyle = :dash)
            r == 1 && CM.axislegend(ax; position = :lb)
        end
        CM.save(ap("energy_vs_budget.png"), fig)
    end

    for (name, ycol, ylab) in (("power_by_wind", :mean_power_kw, "mean engine power (kW)"),
                               ("energy_share_by_wind", :energy_share_pct, "share of the voyage's energy (%)"))
        safeplot(name) do
            isempty(bins_tbl) && return
            mds = unique(bins_tbl.mode)
            fig = CM.Figure(size = (700 * length(mds), 520)); axs = []
            for (c, md) in enumerate(mds)
                ax = CM.Axis(fig[1, c]; xticks = (1:length(WS_LABELS), WS_LABELS), xlabel = "wind speed at vessel (m/s)",
                             ylabel = c == 1 ? ylab : "", title = md)
                for m in METHODS
                    s = bins_tbl[(bins_tbl.mode .== md) .& (bins_tbl.method .== m), :]; isempty(s) && continue
                    xs = Int[findfirst(==(b), WS_LABELS) for b in s.wind_bin]
                    y = s[!, ycol]; ok = .!isnan.(y); any(ok) || continue
                    CM.scatterlines!(ax, xs[ok], y[ok]; color = COLOR[m], label = LABEL[m])
                end
                c == 1 && CM.axislegend(ax; position = :rt)
                push!(axs, ax)
            end
            CM.linkyaxes!(axs...)
            CM.save(ap("$(name).png"), fig)
        end
    end

    for md in modes
        safeplot("progress_$md") do
            tg = mode_tags(md); isempty(tg) && return
            nc = min(3, length(tg)); nr = cld(length(tg), nc)
            fig = CM.Figure(size = (520 * nc, 380 * nr))
            for (n, t) in enumerate(tg)
                ax = CM.Axis(fig[div(n - 1, nc) + 1, mod(n - 1, nc) + 1]; title = "$md  $t",
                             xlabel = "elapsed (days)", ylabel = "distance sailed (nm)")
                for m in METHODS
                    haskey(LEG, (t, md, m)) || continue
                    df = LEG[(t, md, m)]
                    CM.lines!(ax, vcat(0.0, cumsum(df.dt_h)) ./ 24, vcat(0.0, cumsum(df.dist_nm));
                              color = COLOR[m], linestyle = m == "hindsight" ? :dot : :solid, label = LABEL[m])
                end
                n == 1 && CM.axislegend(ax; position = :lt, labelsize = 10)
            end
            CM.save(ap("progress_$(md).png"), fig)
        end
    end
    println("\nPart A done. Files in $ADIR")
end

# ---------------------------------------------------------------- Part B
"""Map of every policy's track for one window and power mode, with replan points."""
function policy_map(tag::String; mode = C400_MODE)
    hind = load_hindsight_route(out("hindsight_$(mode)_$(tag).jld2"); rebuild_wind = false, rmax = RMAX)
    tracks     = Dict{String,Vector{Tuple{Float64,Float64}}}()
    replan_pts = Dict{String,Vector{Tuple{Float64,Float64}}}()
    tracks["hindsight"] = collect(hind.path)

    for (m, f) in (("committed",      "committed_$(mode)_$(tag).jld2"),
                   ("committed_clim", "committed_climatology_$(mode)_$(tag).jld2"))
        try
            tracks[m] = collect(load_committed_run(out(f); rebuild_wind = false).forecast.path)
        catch e
            @warn "[$tag] track for $m unavailable" exception = e
        end
    end

    for (m, f) in (("rolling",      "rolling_replan_$(mode)_$(tag).jld2"),
                   ("rolling_clim", "rolling_replan_climatology_$(mode)_$(tag).jld2"))
        try
            reset_sources!()
            result = load_rolling_replan(out(f))
            rs   = build_rolling_stepper(result)
            Tend = rs.steps[end].cum_time_s
            tracks[m] = [(s = state_at_time(rs, t); (Float64(s.lon), Float64(s.lat)))
                         for t in range(0.0, Tend; length = 300)]
            rts = replan_times_s(out(f))[2:end]
            replan_pts[m] = [(s = state_at_time(rs, min(t, Tend)); (Float64(s.lon), Float64(s.lat))) for t in rts]
        catch e
            @warn "[$tag] track for $m unavailable" exception = e
        end
    end
    reset_sources!()

    gp   = hind.grid_points
    lons = [gp[1, j][1] for j in 1:size(gp, 2)]
    lats = [gp[i, 1][2] for i in 1:size(gp, 1)]
    allp = reduce(vcat, values(tracks))
    pad_deg = 3.0
    x0, x1 = extrema(first.(allp)); y0, y1 = extrema(last.(allp))

    fig = CM.Figure(size = (1600, 850), backgroundcolor = :white)
    ax = CM.Axis(fig[1, 1];
                 title = "Routing strategies for " * Dates.format(tag2date(tag), "dd/mm/yyyy") *
                         (mode == C400_MODE ? " (constant 400 kW)" : " (variable power)"),
                 titlesize = 22, titlefont = :bold, titlealign = :left,
                 xtickformat = vs -> fmt_lon.(vs), ytickformat = vs -> fmt_lat.(vs),
                 aspect = CM.DataAspect(),
                 xgridcolor = (:white, 0.5), ygridcolor = (:white, 0.5),
                 topspinevisible = false, rightspinevisible = false)

    CM.heatmap!(ax, lons, lats, Float32.(hind.ocean_mask)';
                colormap = [CM.RGB(0.86, 0.80, 0.64), CM.RGB(0.84, 0.92, 0.98)],
                colorrange = (0, 1), interpolate = false)

    for m in METHODS
        haskey(tracks, m) || continue
        p = tracks[m]
        h = timeof(tag, mode, m)
        lab = m == "hindsight" ? "$(LABEL[m])  ($(fmt(h, d = 1)) h)" :
              "$(LABEL[m])  ($(fmt(h, d = 1)) h, +$(fmt(pen(tag, mode, m), d = 1))%)"
        CM.lines!(ax, first.(p), last.(p); color = (:white, 0.9), linewidth = m == "hindsight" ? 7.5 : 5.5)
        CM.lines!(ax, first.(p), last.(p); color = COLOR[m], label = lab,
                  linewidth = m == "hindsight" ? 4.5 : 3.0, linestyle = m == "hindsight" ? :dot : :solid)
    end

    for m in ("rolling", "rolling_clim")
        haskey(replan_pts, m) || continue
        rp = replan_pts[m]
        CM.scatter!(ax, first.(rp), last.(rp); color = COLOR[m], markersize = 13,
                    marker = :circle, strokewidth = 2.5, strokecolor = :white)
    end
    CM.scatter!(ax, [NaN], [NaN]; color = :gray40, markersize = 13, marker = :circle,
                strokewidth = 2.5, strokecolor = :white, label = "Replan point")

    sp = tracks["hindsight"][1]; ep = tracks["hindsight"][end]
    CM.scatter!(ax, [sp[1]], [sp[2]]; color = :limegreen, marker = :star5, markersize = 30,
                strokewidth = 2.5, strokecolor = :black, label = "Start")
    CM.scatter!(ax, [ep[1]], [ep[2]]; color = :gold, marker = :star5, markersize = 30,
                strokewidth = 2.5, strokecolor = :black, label = "End")
    CM.text!(ax, sp[1], sp[2]; text = "  Start", align = (:left, :top),  offset = (8, -8),
             fontsize = 16, font = :bold, color = :black)
    CM.text!(ax, ep[1], ep[2]; text = "End  ",   align = (:right, :top), offset = (-8, -8),
             fontsize = 16, font = :bold, color = :black)

    CM.xlims!(ax, x0 - pad_deg, x1 + pad_deg)
    CM.ylims!(ax, y0 - pad_deg, y1 + pad_deg)
    CM.axislegend(ax; position = :lt, framecolor = :gray60, backgroundcolor = (:white, 0.92),
                  labelsize = 15, patchsize = (40, 14), padding = (10, 10, 8, 8))
    suffix = mode == C400_MODE ? "" : "_var"
    CM.save(ap("best_route_map_policies$(suffix)_$(tag).png"), fig; px_per_unit = 2)
    println("Saved map for $tag ($mode)")
end

"""Part B driver: a policy map per (mode, window) and a progress grid per mode."""
function run_part_b(modes::Vector{String}, tags::Vector{String})
    load_run_status!()
    CM.activate!()      # the routing code loads GLMakie, so switch back

    for md in modes, t in tags
        try
            Base.invokelatest(policy_map, t; mode = md)
        catch e
            @warn "map for $t ($md) failed" exception = (e, catch_backtrace())
        end
    end

    for md in modes
        fig = CM.Figure(size = (1560, 760))
        for (n, t) in enumerate(tags)
            ax = CM.Axis(fig[div(n - 1, 3) + 1, mod(n - 1, 3) + 1];
                         title = Dates.format(tag2date(t), "yyyy-mm-dd"),
                         xlabel = "elapsed (days)", ylabel = "distance sailed (nm)")
            for m in METHODS
                f = tab("legs_$(md)_$(m)_$(t).csv")
                isfile(f) || (@warn "missing $f"; continue)
                df = CSV.read(f, DataFrame)
                CM.lines!(ax, vcat(0.0, cumsum(df.dt_h)) ./ 24, vcat(0.0, cumsum(df.dist_nm));
                          color = COLOR[m], linestyle = m == "hindsight" ? :dot : :solid, label = LABEL[m])
            end
            n == 1 && CM.axislegend(ax; position = :lt, labelsize = 10)
        end
        CM.save(ap(md == C400_MODE ? "progress_c400_grid.png" : "progress_$(md)_grid.png"), fig)
    end
    println("Part B done. Files in $ADIR")
end

# ---------------------------------------------------------------- Greedy (needs greedy_route.jl)
"""
Greedy router for one window at constant 400 kW, scored on the same real wind as every
other method. Writes legs_c400_greedy_{committed,allforecast}_<tag>.csv and logs to run_status.csv.
"""
function run_greedy_for_window(tag::String; mode = C400_MODE)
    d0   = tag2date(tag)
    hind = load_hindsight_route(out("hindsight_$(mode)_$(tag).jld2"); rebuild_wind = false, rmax = RMAX)
    h = hind.costs[end] / 3600
    for (label, committed) in (("greedy_committed", true), ("greedy_allforecast", false))
        try
            reset_sources!()
            set_climatology_archive_source!(; dir = HISTORICAL_DIR)
            gdirs, gdates = rolling_gfs_dirs(gfs_outbase(tag), d0, NDAYS, CYCLE_HOUR)
            g = run_greedy_route(
                grid_points = hind.grid_points, ocean_mask = hind.ocean_mask,
                start_idx = hind.start_idx, goal_idx = hind.goal_idx,
                cur_time = hind.diag[:cur_time], cur_day = hind.diag[:cur_day],
                cur_month = hind.diag[:cur_month], cur_year = hind.diag[:cur_year],
                max_energy = C400_MAX_E, power_kw = CONST_KW, min_speed = MIN_SPEED, Rmax = RMAX,
                power_levels = [CONST_KW], gfs_dirs = gdirs, gfs_init_dates = gdates,
                cycle_hour = CYCLE_HOUR, verify_path = hind.diag[:path_idxs], committed = committed)
            if g.stepper === nothing
                @warn "$label did not reach the goal for $tag"
                continue
            end
            leg = leg_table(g.stepper, t -> get_wind(g.stepper.wind, t), hind.grid_points, d0, mode, label)
            CSV.write(tab("legs_$(mode)_$(label)_$(tag).csv"), leg)
            th = g.costs[end] / 3600
            log_status(tag, mode, label, th, g.energy[end], true)
            println("[$tag] $(LABEL[label]): $(round(th, digits = 1)) h  (+$(round(100 * (th - h) / h, digits = 1))% vs hindsight $(round(h, digits = 1)) h)")
        catch e
            @warn "$label failed for $tag" exception = (e, catch_backtrace())
        finally
            reset_sources!()
        end
    end
end

# =============================================================================
# 5. SAMPLED ROUTES + INTERACTIVE VARIABLE-POWER EXPLORER
#    run_sampling first (own process; writes SDIR/samples_<mode>_<method>_<tag>.csv),
#    then launch_variable_power_explorer (GLMakie, reads those CSVs).
# =============================================================================
const N_SAMP = 300
csvpath(md, m, tag) = joinpath(SDIR, "samples_$(md)_$(m)_$(tag).csv")

# wind at the vessel's position/time, heading from the track 30 min ahead, TWA = angle between them
function sample_route(stepper, windfn, gp; n = N_SAMP)
    n_rows, n_cols = size(gp)
    lons = [gp[1, j][1] for j in 1:n_cols]; lats = [gp[i, 1][2] for i in 1:n_rows]
    Tend = stepper.steps[end].cum_time_s
    df = DataFrame(t_h = Float64[], lon = Float64[], lat = Float64[], wind_ms = Float64[],
                   wind_from_deg = Float64[], heading_deg = Float64[], twa_deg = Float64[],
                   speed_kn = Float64[], power_kw = Float64[], energy_kwh = Float64[])
    last_head = 0.0
    for t in range(0.0, Tend; length = n)
        s  = state_at_time(stepper, t)
        s2 = state_at_time(stepper, min(t + 1800.0, Tend))
        if abs(s2.lon - s.lon) + abs(s2.lat - s.lat) > 1e-6
            last_head = getBoatAngle(s.lon, s.lat, s2.lon, s2.lat)
        end
        U, V = windfn(t)
        ws, wd = _wind_speed_dir(U, V, nearest_index(lats, s.lat), nearest_index(lons, s.lon))
        twa = abs(mod(wd - last_head + 180.0, 360.0) - 180.0)
        push!(df, (t / 3600, s.lon, s.lat, ws, wd, last_head, twa, s.speed_kn, s.power_kw, s.energy_kwh))
    end
    return df
end

function load_and_sample(tag, mode, method)
    if method == "hindsight"
        h = load_hindsight_route(out("hindsight_$(mode)_$(tag).jld2"); rebuild_wind = true, rmax = RMAX)
        return sample_route(h.stepper, t -> get_wind(h.stepper.wind, t), h.grid_points)
    elseif method in ("committed", "committed_clim")
        f = method == "committed" ? "committed_$(mode)_$(tag).jld2" : "committed_climatology_$(mode)_$(tag).jld2"
        c = load_committed_run(out(f)); a = c.actual
        return sample_route(a, t -> get_wind(a.wind, t), c.forecast.grid_points)
    else
        f = method == "rolling" ? "rolling_replan_$(mode)_$(tag).jld2" : "rolling_replan_climatology_$(mode)_$(tag).jld2"
        r = load_rolling_replan(out(f))
        rs, offs = build_rolling_stepper(r), _rolling_day_offsets(r)
        return sample_route(rs, t -> (w = _rolling_wind_at(r, offs, t); (w[1], w[2])), r.grid_points)
    end
end

function run_sampling(tags::Vector{String}, modes::Vector{String})
    for tag in tags
        todo = [(md, m) for md in modes for m in METHODS if !isfile(csvpath(md, m, tag))]
        isempty(todo) && (println("[$tag] all sampled"); continue)
        reset_sources!(); set_climatology_archive_source!(; dir = HISTORICAL_DIR)   # fresh month cache per window
        for (md, m) in todo
            try
                CSV.write(csvpath(md, m, tag), load_and_sample(tag, md, m))
                println("[$tag] sampled $md $m")
            catch e
                @warn "[$tag] could not sample $md $m" exception = (e, catch_backtrace())
            end
        end
        reset_sources!(); GC.gc()
    end
    println("Done. CSVs in $SDIR")
end

# value of a sampled column at time t (hours); step = hold previous sample (power, directions)
function sample_val(df, col, t; step = false)
    ts = df.t_h; y = df[!, col]
    t >= ts[end] && return y[end]
    t <= ts[1] && return y[1]
    k = searchsortedlast(ts, t)
    (step || ts[k+1] <= ts[k]) && return y[k]
    return y[k] + (t - ts[k]) / (ts[k+1] - ts[k]) * (y[k+1] - y[k])
end

const EXPL_ROUTES = [
    (m = "hindsight",      name = "Hindsight",         color = :black,     style = :dot),
    (m = "committed",      name = "Committed",         color = :firebrick, style = :dash),
    (m = "rolling",        name = "Rolling",           color = :steelblue, style = :solid),
    (m = "committed_clim", name = "Committed + clim.", color = :orange,    style = :dash),
    (m = "rolling_clim",   name = "Rolling + clim.",   color = :seagreen,  style = :solid),
]
const EXPL_PANELS = [
    (col = :wind_ms,  label = "wind (m/s)"),
    (col = :twa_deg,  label = "TWA (deg)"),
    (col = :speed_kn, label = "speed (kn)"),
    (col = :power_kw, label = "power (kW)"),
]

function launch_variable_power_explorer(tags::Vector{String};
        mode_no_margin = "var_s102_m0", mode_margin = "var_s102_m10", slack = 1.02, margin = 0.10, step_h = 3.0)
    GM.activate!()
    mode_of = Dict(false => mode_no_margin, true => mode_margin)
    nan_pt = GM.Point2f(NaN, NaN)

    D = Dict{Tuple{String,String,String},DataFrame}()
    for tag in tags, md in values(mode_of), r in EXPL_ROUTES
        f = csvpath(md, r.m, tag)
        isfile(f) && (D[(tag, md, r.m)] = CSV.read(f, DataFrame))
    end
    isempty(D) && error("No sample CSVs in $SDIR -- run the sampling step first.")
    println("Loaded $(length(D)) sampled runs")
    tags_ok = [t for t in tags if any(k -> k[1] == t, keys(D))]
    tmax_all = ceil(maximum(df.t_h[end] for df in values(D)))

    st = load_run_status!()
    function budget_of(tag)
        s = st[(st.tag .== tag) .& (st.mode .== C400_MODE) .& (st.method .== "hindsight"), :]
        return nrow(s) == 0 ? NaN : slack * s.energy_kwh[1]
    end

    # ocean background from any saved hindsight run
    mask = nothing
    for tag in tags_ok
        f = out("hindsight_$(C400_MODE)_$(tag).jld2")
        isfile(f) || continue
        try
            jldopen(f, "r") do jf
                gp = jf["grid_points"]; om = jf["ocean_mask"]
                mask = (lons = [gp[1, j][1] for j in 1:size(gp, 2)],
                        lats = [gp[i, 1][2] for i in 1:size(gp, 1)], z = permutedims(Float32.(om)))
            end
            break
        catch e
            @warn "could not read mask from $f" exception = e
        end
    end

    fig = GM.Figure(size = (1900, 1400))
    title_text = GM.Observable("")
    GM.Label(fig[0, 1], title_text; fontsize = 20, font = :bold, halign = :left, tellwidth = false)

    ctrl = fig[1, 1] = GM.GridLayout()
    GM.Label(ctrl[1, 1], "Window:"; halign = :right)
    menu = GM.Menu(ctrl[1, 2]; options = collect(zip([Dates.format(tag2date(t), "yyyy-mm-dd") for t in tags_ok], tags_ok)))
    GM.Label(ctrl[1, 3], "Plan margin $(round(Int, 100margin))%:"; halign = :right)
    tg_margin = GM.Toggle(ctrl[1, 4]; active = false)
    margin_text = GM.Observable("no margin ($mode_no_margin)")
    GM.Label(ctrl[1, 5], margin_text; halign = :left)
    route_visible = [GM.Observable(true) for _ in EXPL_ROUTES]
    for (k, r) in enumerate(EXPL_ROUTES)
        t = GM.Toggle(ctrl[1, 6 + 2 * (k - 1)]; active = true)
        GM.Label(ctrl[1, 7 + 2 * (k - 1)], r.name; color = r.color, halign = :left, fontsize = 12)
        GM.on(t.active) do a; route_visible[k][] = a; end
    end

    main = fig[2, 1] = GM.GridLayout()
    ax_map = GM.Axis(main[1:length(EXPL_PANELS), 1]; xlabel = "lon", ylabel = "lat",
                     title = "Track (white triangle = wind blowing toward)")
    mask === nothing || GM.heatmap!(ax_map, mask.lons, mask.lats, mask.z;
                                    colormap = [GM.RGB(0.82, 0.75, 0.55), GM.RGB(0.85, 0.92, 0.97)],
                                    colorrange = (0, 1), interpolate = false)
    ax_p = [GM.Axis(main[p, 2]; ylabel = EXPL_PANELS[p].label, xlabel = p == length(EXPL_PANELS) ? "elapsed (h)" : "")
            for p in eachindex(EXPL_PANELS)]
    GM.linkxaxes!(ax_p...)

    map_pts    = [GM.Observable([nan_pt]) for _ in EXPL_ROUTES]
    pts        = [[GM.Observable([nan_pt]) for _ in EXPL_PANELS] for _ in EXPL_ROUTES]
    marker_pos = [GM.Observable(nan_pt) for _ in EXPL_ROUTES]
    wind_rot   = [GM.Observable(0.0) for _ in EXPL_ROUTES]
    tmark      = GM.Observable(0.0)

    for (k, r) in enumerate(EXPL_ROUTES)
        GM.lines!(ax_map, map_pts[k]; color = r.color, linewidth = 2.5, linestyle = r.style,
                  label = r.name, visible = route_visible[k])
        GM.scatter!(ax_map, marker_pos[k]; color = r.color, markersize = 24, marker = :star5,
                    strokewidth = 2, strokecolor = :white, visible = route_visible[k])
        GM.scatter!(ax_map, marker_pos[k]; color = :white, markersize = 12, marker = :utriangle,
                    rotation = wind_rot[k], strokewidth = 1, strokecolor = :black, visible = route_visible[k])
        for p in eachindex(EXPL_PANELS)
            if EXPL_PANELS[p].col == :power_kw
                GM.stairs!(ax_p[p], pts[k][p]; color = r.color, linewidth = 1.5, visible = route_visible[k])
            else
                GM.lines!(ax_p[p], pts[k][p]; color = r.color, linewidth = 2, linestyle = r.style,
                          visible = route_visible[k])
            end
        end
    end
    GM.axislegend(ax_map; position = :lb, labelsize = 10, backgroundcolor = (:white, 0.85))
    for ax in ax_p
        GM.vlines!(ax, tmark; color = :black, linestyle = :dash, linewidth = 1.2)
    end
    GM.colsize!(main, 1, GM.Relative(0.40)); GM.colsize!(main, 2, GM.Relative(0.60))

    stats = fig[3, 1] = GM.GridLayout()
    time_text = GM.Observable("")
    GM.Label(stats[1, 1:length(EXPL_ROUTES)], time_text; tellwidth = false, halign = :left, fontsize = 13, font = :bold)
    route_texts = [GM.Observable("") for _ in EXPL_ROUTES]
    for k in eachindex(EXPL_ROUTES)
        GM.Label(stats[2, k], route_texts[k]; tellwidth = false, halign = :left, valign = :top,
                 fontsize = 11, justification = :left, color = EXPL_ROUTES[k].color, visible = route_visible[k])
        GM.colsize!(stats, k, GM.Relative(1.0 / length(EXPL_ROUTES)))
    end

    cur_tag    = GM.Observable(tags_ok[1])
    cur_margin = GM.Observable(false)
    t_end_view = GM.Observable(1.0)
    tcur       = GM.Observable(0.0)

    function update!(t)
        tmark[] = t
        tag = cur_tag[]; md = mode_of[cur_margin[]]; b = budget_of(tag)
        time_text[] = "elapsed: $(round(t, digits = 1)) h  ($(round(t / 24, digits = 1)) days)   |   budget $(round(b, digits = 0)) kWh" *
                      (cur_margin[] ? "   |   planner holds back $(round(Int, 100margin))%" : "")
        for (k, r) in enumerate(EXPL_ROUTES)
            df = get(D, (tag, md, r.m), nothing)
            if df === nothing
                marker_pos[k][] = nan_pt
                route_texts[k][] = "$(r.name)\nno run available"
                continue
            end
            e   = sample_val(df, :energy_kwh, t)
            wdf = sample_val(df, :wind_from_deg, t; step = true)
            marker_pos[k][] = GM.Point2f(sample_val(df, :lon, t), sample_val(df, :lat, t))
            wind_rot[k][]   = -deg2rad(mod(wdf + 180.0, 360.0))
            dnf  = get(REACH, (tag, md, r.m), true) ? "" : "  [DNF - ran out]"
            done = t >= df.t_h[end]
            route_texts[k][] = "$(r.name)$dnf\n" *
                (done ? "voyage ended at $(round(df.t_h[end], digits = 0)) h\n" : "under way\n") *
                "energy $(round(e, digits = 0)) kWh ($(round(100e / b, digits = 1))% of budget)\n" *
                "speed $(round(sample_val(df, :speed_kn, t), digits = 1)) kn | power $(round(sample_val(df, :power_kw, t; step = true), digits = 0)) kW\n" *
                "wind $(round(sample_val(df, :wind_ms, t), digits = 1)) m/s from $(round(wdf, digits = 0))° | " *
                "TWA $(round(sample_val(df, :twa_deg, t; step = true), digits = 0))°"
        end
    end

    function reload!()
        tag = cur_tag[]; md = mode_of[cur_margin[]]; tend = 1.0
        for (k, r) in enumerate(EXPL_ROUTES)
            df = get(D, (tag, md, r.m), nothing)
            if df === nothing
                map_pts[k][] = [nan_pt]
                for p in eachindex(EXPL_PANELS); pts[k][p][] = [nan_pt]; end
            else
                map_pts[k][] = GM.Point2f.(df.lon, df.lat)
                for (p, pn) in enumerate(EXPL_PANELS); pts[k][p][] = GM.Point2f.(df.t_h, df[!, pn.col]); end
                tend = max(tend, df.t_h[end])
            end
        end
        t_end_view[] = tend
        title_text[] = "Variable-power explorer  -  window $(Dates.format(tag2date(tag), "yyyy-mm-dd"))  -  $md  (budget = $(slack) x c400 hindsight energy)"
        margin_text[] = cur_margin[] ? "$(round(Int, 100margin))% margin ($mode_margin)" : "no margin ($mode_no_margin)"
        GM.autolimits!(ax_map)
        for ax in ax_p; GM.autolimits!(ax); end
        GM.xlims!(ax_p[1], 0, tend)
        GM.ylims!(ax_p[2], 0, 180)
        tcur[] = 0.0
        update!(0.0)
    end

    controls = fig[4, 1] = GM.GridLayout()
    is_playing = GM.Observable(false)
    btn_prev  = GM.Button(controls[1, 1], label = "◀ Prev")
    btn_next  = GM.Button(controls[1, 2], label = "Next ▶")
    btn_reset = GM.Button(controls[1, 3], label = "⟲ Reset")
    btn_play  = GM.Button(controls[1, 4], label = "▶ Play")
    scrub = GM.Slider(controls[1, 5:7], range = 0.0:1.0:tmax_all, startvalue = 0.0)
    GM.Label(controls[1, 8], GM.lift(t -> "t = $(round(t, digits = 1)) h", tcur))

    syncing = Ref(false)
    GM.on(tcur) do t
        update!(t)
        syncing[] = true
        GM.set_close_to!(scrub, t)
        syncing[] = false
    end
    GM.on(scrub.value) do v
        syncing[] && return
        tcur[] = clamp(v, 0.0, t_end_view[])
    end
    GM.on(btn_prev.clicks)  do _; tcur[] = max(0.0, tcur[] - step_h); end
    GM.on(btn_next.clicks)  do _; tcur[] = min(t_end_view[], tcur[] + step_h); end
    GM.on(btn_reset.clicks) do _; tcur[] = 0.0; end
    GM.on(btn_play.clicks)  do _
        is_playing[] = !is_playing[]
        btn_play.label = is_playing[] ? "⏸ Pause" : "▶ Play"
    end
    play_timer = Ref{Union{Nothing,Timer}}(nothing)
    GM.on(is_playing) do playing
        play_timer[] === nothing || (close(play_timer[]); play_timer[] = nothing)
        if playing
            play_timer[] = Timer(0.1; interval = 0.1) do _
                if tcur[] < t_end_view[]
                    tcur[] = min(t_end_view[], tcur[] + step_h)
                else
                    is_playing[] = false
                    btn_play.label = "▶ Play"
                end
            end
        end
    end
    GM.on(menu.selection) do s; cur_tag[] = s; reload!(); end
    GM.on(tg_margin.active) do a; cur_margin[] = a; reload!(); end

    GM.rowsize!(fig.layout, 0, GM.Fixed(34))
    GM.rowsize!(fig.layout, 1, GM.Fixed(46))
    GM.rowsize!(fig.layout, 2, GM.Relative(0.66))
    GM.rowsize!(fig.layout, 3, GM.Fixed(120))
    GM.rowsize!(fig.layout, 4, GM.Fixed(60))

    reload!()
    screen = display(fig)
    isinteractive() || wait(screen)
    return fig
end

# =============================================================================
# 6. CLIMATOLOGY-MATCH SCORING
#    Which historical windows best match the climatological wind distribution along each
#    month's climatology-optimal corridor.
# =============================================================================
const CLIMATOLOGY_BY_MONTH_PATH = out("climatology_by_month.jld2")

# Raw (speed x direction) counts from one day's (U,V) snapshots inside corridor_mask, on the
# SAME bin edges as the pooled climatology rose.
function day_rose_from_snapshots(uv_list::Vector{Tuple{Matrix{Float32},Matrix{Float32}}},
                                 corridor_mask::Matrix{Bool}, ws_edges::Vector{Float32}, dir_edges::Vector{Float32})
    n_ws, n_dir = length(ws_edges) - 1, length(dir_edges) - 1
    counts = zeros(Float64, n_ws, n_dir)
    n_obs = 0
    n_rows, n_cols = size(corridor_mask)
    ws_max = ws_edges[end]
    for (U, V) in uv_list
        @inbounds for j in 1:n_cols, i in 1:n_rows
            corridor_mask[i, j] || continue
            u = U[i, j]; v = V[i, j]
            ws = sqrt(u^2 + v^2)
            ws >= ws_max && (ws = prevfloat(ws_max))
            wdir = mod(rad2deg(atan(u, v)) + 180f0, 360f0)
            wi = clamp(searchsortedlast(ws_edges, ws), 1, n_ws)
            di = clamp(searchsortedlast(dir_edges, Float32(wdir)), 1, n_dir)
            counts[wi, di] += 1.0
            n_obs += 1
        end
    end
    return counts, n_obs
end

# Corridor = ocean cells within corridor_km of the ACTUAL climatology-optimal route (not the great
# circle, which can cut across land). Matrix{Bool}, not BitMatrix: pooled_climatology_rose requires it.
function build_route_corridor_mask(grid_points::Matrix{Tuple{Float64,Float64}}, ocean_mask::Matrix{Bool},
                                   route_idxs::Vector{Tuple{Int,Int}}, corridor_km::Float64)
    n_rows, n_cols = size(grid_points)
    route_pts = [grid_points[i, j] for (i, j) in route_idxs]
    mask = fill(false, n_rows, n_cols)
    for j in 1:n_cols, i in 1:n_rows
        !ocean_mask[i, j] && continue
        lon0, lat0 = grid_points[i, j]
        mask[i, j] = minimum(getDistance(lon0, lat0, rp[1], rp[2]) / 1000.0 for rp in route_pts) <= corridor_km
    end
    return mask
end

# Rolling windows of per-day scores. agg: :mean, :worst or :sum. NOTE: for a fixed window length
# sum = mean * window_days, so :sum ranks identically to :mean (the sum columns are still reported).
function rolling_window_climatology_scores(rows::DataFrame; window_days::Int = 18, step_days::Int = 1, agg::Symbol = :mean)
    agg in (:mean, :worst, :sum) || error("agg must be :mean, :worst, or :sum, got $agg")
    nrow(rows) == 0 && error("rows is empty")
    r = sort(rows, :date)

    jsd_std = std(r.jsd); wsd_std = std(r.wasserstein_speed)
    (jsd_std == 0 || wsd_std == 0) && error("zero variance in jsd or wasserstein_speed -- can't z-score")
    combined_z = (r.jsd .- mean(r.jsd)) ./ jsd_std .+ (r.wasserstein_speed .- mean(r.wasserstein_speed)) ./ wsd_std
    date_to_row = Dict(d => k for (k, d) in enumerate(r.date))

    res = DataFrame(window_start = Date[], window_end = Date[], n_days = Int[],
                    mean_jsd = Float64[], mean_wasserstein = Float64[], mean_combined_z = Float64[],
                    sum_jsd = Float64[], sum_wasserstein = Float64[], sum_combined_z = Float64[],
                    worst_day_combined_z = Float64[], worst_day_date = Date[])
    length(r.date) < window_days && error("only $(length(r.date)) scored days, need window_days=$window_days")

    candidate_starts = r.date[1]:Day(step_days):(r.date[end] - Day(window_days - 1))
    n_skipped_gap = 0
    for ws in candidate_starts
        window_dates = [ws + Day(k) for k in 0:window_days-1]
        all(d -> haskey(date_to_row, d), window_dates) || (n_skipped_gap += 1; continue)
        idxs = [date_to_row[d] for d in window_dates]
        w_jsd = r.jsd[idxs]; w_wsd = r.wasserstein_speed[idxs]; w_cz = combined_z[idxs]
        worst_k = argmax(w_cz)
        push!(res, (ws, ws + Day(window_days - 1), window_days, mean(w_jsd), mean(w_wsd), mean(w_cz),
                    sum(w_jsd), sum(w_wsd), sum(w_cz), w_cz[worst_k], window_dates[worst_k]))
    end
    n_skipped_gap > 0 && println("skipped $n_skipped_gap / $(length(candidate_starts)) candidate window(s) due to gaps")
    isempty(res) && error("no complete $window_days-day window without a gap")
    sort!(res, agg == :mean ? :mean_combined_z : agg == :sum ? :sum_combined_z : :worst_day_combined_z)
    return res
end

"""
Score every archived day against its own month's climatological rose, roll into windows,
and validate the best windows with a real-wind replay.
Reference grid/start/goal come from the saved c400 rolling run of `reference_tag`;
validation GFS runs are expected under `gfs_base`_<yyyymmdd>_00z.
"""
function run_climatology_match(; reference_tag = RUN_TAGS[1], corridor_km = 150.0, window_days = 18,
                                power_kw = 1200.0, top_k = 5, gfs_base = gfs_path("gfs_rolling"))
    result = load_rolling_replan(out("rolling_replan_$(C400_MODE)_$(reference_tag).jld2"))
    start_idx, goal_idx = result.start_idx, result.goal_idx
    grid_points, ocean_mask = result.grid_points, result.ocean_mask

    all_paths = [joinpath(HISTORICAL_DIR, f) for f in readdir(HISTORICAL_DIR) if endswith(f, ".grib")]
    isempty(all_paths) && error("No .grib files found in $HISTORICAL_DIR")
    month_files = sort([(parse(Int, m.captures[1]), parse(Int, m.captures[2]))
                        for p in all_paths for m in [match(r"(\d{4})-(\d{2})\.grib$", basename(p))] if m !== nothing])

    climatologies = ensure_climatologies(last.(month_files))
    ref_field = _reference_field(all_paths[1])
    interp = build_interpolator(grid_points, ref_field)
    n_rows, n_cols = size(grid_points)
    nb_flat = build_neighbour_cache(grid_points, ocean_mask, n_rows, n_cols, RMAX, 27_800.0)

    # per-month reference route, corridor and pooled rose (nothing shared across months)
    month_next_hop = Dict{Int, Matrix{Tuple{Int,Int}}}()
    month_corridor = Dict{Int, Matrix{Bool}}()
    month_rose     = Dict{Int, NamedTuple}()
    for m in sort(collect(keys(climatologies)))
        println("Building month $m's reference route + corridor...")
        next_hop = build_climatology_next_hop(; climatology = climatologies[m], grid_points, ocean_mask,
                                              nb_flat, power_kw, goal_idx)
        route_idxs = reconstruct_climatology_route(next_hop, start_idx, goal_idx)
        corridor = build_route_corridor_mask(grid_points, ocean_mask, route_idxs, corridor_km)
        month_next_hop[m] = next_hop
        month_corridor[m] = corridor
        month_rose[m]     = pooled_climatology_rose(climatologies[m], corridor)
        println("  month $m: $(length(route_idxs)) waypoints, corridor = $(count(corridor)) / $(count(ocean_mask)) ocean cells")
    end

    # score every day against its own month
    rows = DataFrame(date = Date[], year = Int[], month = Int[], n_obs = Int[], jsd = Float64[], wasserstein_speed = Float64[])
    progress = Progress(length(month_files); desc = "Scoring days", showspeed = true)
    for (yr, mo) in month_files
        if haskey(month_rose, mo)
            rose = month_rose[mo]; corridor = month_corridor[mo]
            by_day = load_month_days(HISTORICAL_DIR, yr, mo, interp, n_rows, n_cols)
            for (d, uv_list) in sort(collect(by_day); by = x -> x[1])
                probs, n_obs = day_rose_from_snapshots(uv_list, corridor, rose.ws_edges, rose.dir_edges)
                n_obs == 0 && continue
                push!(rows, (d, yr, mo, n_obs, jensen_shannon_distance(probs, rose.probs),
                             wasserstein1_speed_marginal(probs, rose.probs, rose.ws_edges)))
            end
        end
        ProgressMeter.next!(progress)
    end
    sort!(rows, :date)
    CSV.write(match_out("climatology_match_distributional.csv"), rows)
    println("\nScored $(nrow(rows)) days")
    println("NOTE: z-scores are pooled over all months; check per-month jsd spread if a few months dominate the top windows.")

    scores = Dict(agg => rolling_window_climatology_scores(rows; window_days, agg) for agg in (:mean, :worst, :sum))
    CSV.write(match_out("climatology_window_scores_$(window_days)d.csv"),       scores[:mean])
    CSV.write(match_out("climatology_window_scores_$(window_days)d_worst.csv"), scores[:worst])
    CSV.write(match_out("climatology_window_scores_$(window_days)d_sum.csv"),   scores[:sum])
    println("\n--- TOP 10 best-matching $(window_days)-day windows (mean) ---")
    println(first(scores[:mean], 10))

    # validate: a window's start month decides which month's route it is validated with
    ws_mean = scores[:mean]
    span_start = minimum(ws_mean.window_start[1:min(top_k, nrow(ws_mean))])
    span_end   = maximum(ws_mean.window_end[1:min(top_k, nrow(ws_mean))])
    gfs_dirs, gfs_init_dates = rolling_gfs_dirs(gfs_base, span_start, Dates.value(span_end - span_start) + 1, CYCLE_HOUR)
    gfs_dir_lookup = Dict(zip(gfs_init_dates, gfs_dirs))
    gfs_dirs_for_date(d::Date) = haskey(gfs_dir_lookup, d) ? gfs_dir_lookup[d] :
        error("No GFS directory for $d under $gfs_base -- download it or adjust gfs_base.")

    parts = DataFrame[]
    for m in unique(Dates.month.(first(ws_mean, top_k).window_start))
        haskey(month_next_hop, m) || error("No climatology-optimal route built for month $m")
        mw = ws_mean[Dates.month.(ws_mean.window_start) .== m, :]
        isempty(mw) && continue
        push!(parts, validate_climatology_match_windows(mw; top_k = min(top_k, nrow(mw)),
                next_hop = month_next_hop[m], start_idx, goal_idx, power_kw,
                grid_points, ocean_mask, nb_flat, gfs_dirs_for_date))
    end
    if !isempty(parts)
        validated = sort!(vcat(parts...), :window_score)
        CSV.write(match_out("climatology_window_validation_top$(top_k).csv"), first(validated, top_k))
    end
    println("Climatology match done. Files in $CLIMMATCH_DIR")
end

# =============================================================================
# 7. MAIN PIPELINE  (select windows -> [download GFS] -> five methods per window)
# =============================================================================
const MAX_HOUR      = 336
const PARALLEL_JOBS = 12
const BASH_SCRIPT   = "./download_gfs_rolling_series.sh"

function gfs_ready(tag)
    dirs, _ = rolling_gfs_dirs(gfs_outbase(tag), tag2date(tag), NDAYS, CYCLE_HOUR)
    return all(isdir, dirs)
end

function select_windows()
    sel = DataFrame(window_start = [tag2date(t) for t in RUN_TAGS], tag = RUN_TAGS)
    sel.gfs_ready = gfs_ready.(sel.tag)
    CSV.write(tab("selected_windows.csv"), sel)
    return sel
end

function bash_exe()
    Sys.iswindows() || return "bash"
    haskey(ENV, "BASH_EXE") && return ENV["BASH_EXE"]
    for c in ("C:\\Program Files\\Git\\bin\\bash.exe", "C:\\Program Files (x86)\\Git\\bin\\bash.exe")
        isfile(c) && return c
    end
    error("Set BASH_EXE to Git Bash's bash.exe")
end

function download_missing(sel)
    for r in eachrow(sel)
        gfs_ready(r.tag) && continue
        println("Downloading missing GFS data for $(r.tag)")
        # script args: start date, n days, cycle, max forecast hour, output prefix, parallel jobs
        run(`$(bash_exe()) $BASH_SCRIPT $(r.tag) $NDAYS 00 $MAX_HOUR $(gfs_outbase(r.tag)) $PARALLEL_JOBS`)
    end
end

# climatology_by_month.jld2: load if present, build only the missing months, re-save
function ensure_climatologies(months)
    path = CLIMATOLOGY_BY_MONTH_PATH
    clims = isfile(path) ? load_climatologies_by_month(path) : Dict{Int,Climatology}()
    need = [m for m in sort(unique(months)) if !haskey(clims, m)]
    if !isempty(need)
        println("Building climatologies for months $need ...")
        r = build_climatologies_by_month(; start_location = START_LOCATION,
                end_location = END_LOCATION, months = need, save_path = nothing)
        merge!(clims, r.climatologies)
        save_climatologies_by_month(path, clims)
    end
    return clims
end

# ---------------- all five methods for one window + power mode ----------------
function run_date_mode(d, tag, mode; levels, max_energy, bin_size, methods = METHODS, plan_margin = 0.0)
    gfs_dirs, gfs_init_dates = rolling_gfs_dirs(gfs_outbase(tag), d, NDAYS, CYCLE_HOUR)
    m = month(d)
    legs_file(method) = tab("legs_$(mode)_$(method)_$(tag).csv")
    done(method) = isfile(legs_file(method))
    common = (start_location = START_LOCATION, end_location = END_LOCATION, power_levels = levels,
              bin_size = bin_size, start_date = d, start_hhmm = 0,
              max_energy = max_energy, min_speed = MIN_SPEED)
    plan_energy = max_energy * (1 - plan_margin)
    common_plan = merge(common, (max_energy = plan_energy,))

    # ---- hindsight
    if "hindsight" in methods && !done("hindsight")
        reset_sources!()
        h = run_hindsight_route(; common..., Rmax = RMAX)
        save_hindsight_route(out("hindsight_$(mode)_$(tag).jld2"), h)
        with_archive() do
            CSV.write(legs_file("hindsight"),
                leg_table(h.stepper, stepper_wind(h.stepper), h.grid_points, d, mode, "hindsight"))
        end
        log_status(tag, mode, "hindsight", h.costs[end]/3600, h.energy[end], true)
    end

    # ---- forecast, committed
    if "committed" in methods && !done("committed")
        reset_sources!()
        c = run_single_forecast_committed(; gfs_dir = gfs_dirs[1],
                init_year = year(d), init_month = month(d), init_day = day(d), cycle_hour = CYCLE_HOUR,
                common_plan..., Rmax = RMAX)
        save_committed_run(out("committed_$(mode)_$(tag).jld2"), c)
        with_archive() do
            CSV.write(legs_file("committed"),
                leg_table(c.actual, stepper_wind(c.actual), c.forecast.grid_points, d, mode, "committed"))
        end
        log_status(tag, mode, "committed", c.actual.steps[end].cum_time_s/3600,
                   c.actual.steps[end].cum_energy_kwh, true)
    end

    # ---- rolling
    if "rolling" in methods && !done("rolling")
        reset_sources!()
        r = run_rolling_replan(; gfs_dirs, gfs_init_dates, cycle_hour = CYCLE_HOUR, common...,
                Rmax = RMAX, plan_margin = plan_margin, max_days = NDAYS + 14)
        save_rolling_replan(out("rolling_replan_$(mode)_$(tag).jld2"), r)
        rs, offs = build_rolling_stepper(r), _rolling_day_offsets(r)
        with_archive() do
            wf = t -> (w = _rolling_wind_at(r, offs, t); (w[1], w[2]))
            CSV.write(legs_file("rolling"), leg_table(rs, wf, r.grid_points, d, mode, "rolling"))
        end
        log_status(tag, mode, "rolling", r.total_time_s/3600, r.total_energy_kwh, r.reached_goal)
    end

    # grid info for the climatology runs, taken from the saved committed run
    needs_clim = any(x -> x in methods && !done(x), ("committed_clim", "rolling_clim"))
    needs_clim || return nothing
    fc = load_committed_run(out("committed_$(mode)_$(tag).jld2"); rebuild_wind = false).forecast
    cache_mode = startswith(mode, "var_") ? "var" : mode
    cache_path = out("climatology_blend_cache_$(cache_mode)_month$(m).jld2")

    function neighbour_cache_for_fc()
        n_rows, n_cols = size(fc.grid_points)
        return build_neighbour_cache(fc.grid_points, fc.ocean_mask, n_rows, n_cols, fc.rmax,
                                     27_800.0; heading_mode = :full)
    end

    if "rolling_clim" in methods && !isfile(cache_path)
        reset_sources!()
        prepare_climatology_blend!(fc.grid_points, fc.ocean_mask, neighbour_cache_for_fc(), levels, MIN_SPEED;
            months = [m], full_forecast_days = FULL_FORECAST_DAYS, cache_path = cache_path)
        clear_climatology_blend!()
    end

    # ---- forecast + climatology, committed
    if "committed_clim" in methods && !done("committed_clim")
        reset_sources!()
        prepare_climatology_blend!(fc.grid_points, fc.ocean_mask, neighbour_cache_for_fc(), levels, MIN_SPEED;
            months = [m], full_forecast_days = FULL_FORECAST_DAYS, cache_path = cache_path)
        fcc = run_forecast_route(; gfs_dir = fc.gfs_dir, init_year = fc.gfs_init_year,
                init_month = fc.gfs_init_month, init_day = fc.gfs_init_day,
                cycle_hour = fc.gfs_cycle_hour, available_hours = fc.gfs_available_hours,
                start_location = START_LOCATION, end_location = END_LOCATION,
                power_levels = levels, max_energy = plan_energy, bin_size = bin_size,
                min_speed = MIN_SPEED, Rmax = fc.rmax, start_date = d, start_hhmm = 0)
        clear_gfs_forecast_source!()
        actual = with_archive() do
            replay_route_real_wind(fcc.diag, fcc.path, fcc.power; planned_time_s = fcc.costs[end])
        end
        save_committed_run(out("committed_climatology_$(mode)_$(tag).jld2"), (forecast = fcc, actual = actual))
        with_archive() do
            CSV.write(legs_file("committed_clim"),
                leg_table(actual, stepper_wind(actual), fcc.grid_points, d, mode, "committed_clim"))
        end
        log_status(tag, mode, "committed_clim", actual.steps[end].cum_time_s/3600,
                   actual.steps[end].cum_energy_kwh, true)
        clear_climatology_blend!()
    end

    # ---- rolling + climatology
    if "rolling_clim" in methods && !done("rolling_clim")
        reset_sources!()
        r = run_rolling_replan(; gfs_dirs, gfs_init_dates, cycle_hour = CYCLE_HOUR, common...,
                Rmax = fc.rmax, plan_margin = plan_margin, max_days = NDAYS + 14,
                use_climatology_blend = true, climatology_cache_path = cache_path,
                climatology_full_forecast_days = FULL_FORECAST_DAYS)
        save_rolling_replan(out("rolling_replan_climatology_$(mode)_$(tag).jld2"), r)
        rs, offs = build_rolling_stepper(r), _rolling_day_offsets(r)
        with_archive() do
            wf = t -> (w = _rolling_wind_at(r, offs, t); (w[1], w[2]))
            CSV.write(legs_file("rolling_clim"), leg_table(rs, wf, r.grid_points, d, mode, "rolling_clim"))
        end
        log_status(tag, mode, "rolling_clim", r.total_time_s/3600, r.total_energy_kwh, r.reached_goal)
        reset_sources!()
    end
    return nothing
end

"""
Run every selected window: constant 400 kW first (if `run_c400`), then each variable-power
config in `variable_configs` (budget = slack x the c400 hindsight energy, planner holds back `margin`).
Finished methods are skipped (a method is done when its legs_*.csv exists), so reruns resume.
"""
function run_pipeline(; only_tag = nothing, run_c400::Bool = true,
                        variable_configs = NamedTuple[], download::Bool = false)
    sel = select_windows()
    only_tag === nothing || (sel = sel[sel.tag .== only_tag, :])
    isempty(sel) && error("No windows selected")

    if download
        download_missing(sel)
        sel.gfs_ready = gfs_ready.(sel.tag)
    end
    missing_gfs = sel[.!sel.gfs_ready, :]
    if !isempty(missing_gfs)
        println("\nSkipping $(nrow(missing_gfs)) window(s) with no local GFS data:"); println(missing_gfs[:, [:tag]])
        sel = sel[sel.gfs_ready, :]
    end
    isempty(sel) && error("No windows have local GFS data -- nothing to run.")
    println("\nWindows to run ($(nrow(sel))):"); println(sel[:, [:tag, :window_start]])

    ensure_climatologies(month.(sel.window_start))
    failed = DataFrame(tag = String[], mode = String[], error = String[])
    function attempt_window(d, tag, mode, f)
        println("\n##### $d [$mode] #####")
        try
            f()
        catch err
            @warn "Window $d failed [$mode]" exception = (err, catch_backtrace())
            push!(failed, (tag, mode, sprint(showerror, err)))
            CSV.write(tab("failed_windows.csv"), failed)
        end
        GC.gc()
    end

    if run_c400
        println("\n########## $C400_MODE ##########")
        for r in eachrow(sel)
            attempt_window(r.window_start, r.tag, C400_MODE, () ->
                run_date_mode(r.window_start, r.tag, C400_MODE; levels = [CONST_KW],
                              max_energy = C400_MAX_E, bin_size = C400_BIN, plan_margin = 0.0))
        end
    end

    for cfg in variable_configs
        mode = var_mode(cfg)
        println("\n########## CONFIG $mode ##########")
        for r in eachrow(sel)
            attempt_window(r.window_start, r.tag, mode, () -> begin
                e_B = load_hindsight_route(out("hindsight_$(C400_MODE)_$(r.tag).jld2");
                                           rebuild_wind = false, rmax = RMAX).energy[end]
                println("budget: $(round(e_B * cfg.slack, digits=1)) kWh, planner holds back $(100*cfg.margin)%")
                run_date_mode(r.window_start, r.tag, mode; levels = LEVELS_ALL,
                              max_energy = e_B * cfg.slack, bin_size = e_B / 100, plan_margin = cfg.margin)
            end)
        end
    end
    return nothing
end

# =============================================================================
# 8. RUN SETTINGS  -- choose what runs; nothing else executes on include
#
# Process rules:
#   * pipeline / convergence / verification each want their OWN Julia process.
#   * analysis_a / analysis_b / greedy read what the pipeline wrote; do not combine them with
#     the pipeline in one process (the pipeline leaves wind sources / GLMakie state behind).
#   * explorer opens a GLMakie window; run it last or alone.
# =============================================================================
const VAR_CONFIGS = [
    (slack = 1.02, margin = 0.00),   # budget = 1.02 x c400 energy
    (slack = 1.02, margin = 0.10),   # same budget, planner holds back 10%. (For a 1.10 budget use slack = 1.10, margin = 0.00)
]

const RUN = (
    # ---- main pipeline
    pipeline            = true,
    only_tag            = nothing,         # e.g. "20221003" to run a single window
    run_c400            = true,
    run_variable        = false,           # variable-power runs for every config in VAR_CONFIGS
    download_missing    = false,           # fetch GFS for windows whose folders are absent

    # ---- studies (each in its own process)
    convergence         = false,
    convergence_ablation = true,
    verification        = :none,           # :none | :all | :variable | :full_estimate

    # ---- analysis of saved runs
    analysis_a          = false,           # tables + plots from run_status.csv / legs_*.csv
    analysis_b          = false,           # policy maps + progress grids
    analysis_b_modes    = [C400_MODE],     # add var_mode(c) entries for variable-power maps
    greedy              = false,           # greedy comparison (include("greedy_route.jl"))
    greedy_tags         = RUN_TAGS,

    # ---- sampling + explorer for variable-power runs
    sampling            = false,
    explorer            = false,
    explorer_tags       = RUN_TAGS,

    # ---- climatology-match scoring
    climatology_match   = false,
    climatology_match_reference_tag = RUN_TAGS[1],
)

function main(R = RUN)
    heavy = R.pipeline || R.convergence || R.verification !== :none
    (heavy && (R.analysis_a || R.analysis_b || R.greedy || R.sampling || R.explorer || R.climatology_match)) &&
        error("Run the pipeline / convergence / verification in their own process, separate from the analysis steps.")
    R.pipeline && (R.convergence || R.verification !== :none) &&
        error("Run the pipeline, convergence and verification in separate processes.")
    R.verification in (:none, :all, :variable, :full_estimate) ||
        error("RUN.verification must be :none, :all, :variable or :full_estimate")

    if R.pipeline
        Base.invokelatest(run_pipeline; only_tag = R.only_tag, run_c400 = R.run_c400,
                          variable_configs = R.run_variable ? VAR_CONFIGS : NamedTuple[],
                          download = R.download_missing)
    end
    R.convergence && Base.invokelatest(run_convergence; ablation = R.convergence_ablation)
    R.verification === :all           && Base.invokelatest(run_verification_all)
    R.verification === :variable      && Base.invokelatest(run_verification_variable)
    R.verification === :full_estimate && Base.invokelatest(run_verification_full_estimate)

    R.analysis_a && Base.invokelatest(run_analysis_part_a)
    R.analysis_b && Base.invokelatest(run_part_b, String.(R.analysis_b_modes), RUN_TAGS)
    if R.greedy
        include("greedy_route.jl")
        for t in R.greedy_tags
            Base.invokelatest(run_greedy_for_window, t)
        end
    end

    var_modes = [var_mode(c) for c in VAR_CONFIGS]
    R.sampling && Base.invokelatest(run_sampling, String.(R.explorer_tags), var_modes)
    R.explorer && Base.invokelatest(launch_variable_power_explorer, String.(R.explorer_tags);
                                    mode_no_margin = var_modes[1], mode_margin = var_modes[end],
                                    slack = VAR_CONFIGS[1].slack, margin = VAR_CONFIGS[end].margin)
    R.climatology_match && in_clean_archive(() ->
        Base.invokelatest(run_climatology_match; reference_tag = R.climatology_match_reference_tag))
    return nothing
end

get(ENV, "ROUTING_SKIP_MAIN", "") == "1" || main()


