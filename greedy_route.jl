# =============================================================================
# greedy_route.jl
#
# Adds a genuinely simple greedy router alongside the existing Dijkstra
# solvers, plus a way to compare it against a hindsight-optimal run using
# the SAME launch_simulator explorer route_simulator.jl already has for
# comparing different physical paths (not launch_forecast_actual_explorer --
# that one requires both steppers to share the same waypoint sequence,
# which hindsight and greedy won't).
#
# Include order:
include("routing.jl")
include("route_ship.jl")
include("route_simulator.jl")
# =============================================================================
# Information-matched greedy router.
#   decide : hop costs come from the GFS forecast cycle of the current real day
#   execute: the chosen hop is sailed under REAL wind (climatology-archive source)
# =============================================================================

# Temporarily set BOTH global wind-source refs, restore them afterwards.
function _with_wind_source(f::Function, gfs_src, clim_src)
    old_g = _GFS_FORECAST_SOURCE[]
    old_c = _CLIMATOLOGY_ARCHIVE_SOURCE[]
    _GFS_FORECAST_SOURCE[]        = gfs_src
    _CLIMATOLOGY_ARCHIVE_SOURCE[] = clim_src
    try
        return f()
    finally
        _GFS_FORECAST_SOURCE[]        = old_g
        _CLIMATOLOGY_ARCHIVE_SOURCE[] = old_c
    end
end

# Same NamedTuple set_gfs_forecast_source! builds, without the printing.
function _make_gfs_source(dir::String, init_date::Date, cycle_hour::Int, available_hours::Vector{Int})
    init_epoch = toEpochSeconds(cycle_hour * 100, Dates.day(init_date),
                                Dates.month(init_date), Dates.year(init_date))
    cache_dir = joinpath(dir, "jld2_cache"); mkpath(cache_dir)
    return (dir = dir, cycle = cycle_hour, init_epoch = init_epoch,
            available_hours = sort(available_hours), cache_dir = cache_dir)
end

# One wind source + its own pool/interpolator/uv cache + its cost functions.
mutable struct GreedyWindCtx
    gfs_src   ::Any
    clim_src  ::Any
    pool      ::WindFieldPool
    interp    ::GridInterpolator
    uv        ::UVGridCache
    cost      ::Function      # (dist, bearing, arc_flats, elapsed_s) -> (t_s, e_kwh) or (Inf, Inf)
    execute   ::Function      # (dist, bearing, arc_flats, elapsed_s) -> (t_s, e_kwh, vmin)
    reg_until ::Int           # epochs registered up to this many elapsed seconds
end

# cost   = exactly what findDijkstraRoutefinal does (stepped, time-aware, feasibility test)
# execute = no feasibility rejection; same integration as replay_route_real_wind
function make_edge_costs(nb_flat, wind_pool, interp, uv_cache, n_rows::Int, n_flat::Int,
                         cur_time::Int, cur_day::Int, cur_month::Int, cur_year::Int,
                         power_kw::Float64, levels::Vector{Float64}, polar,
                         min_speed::Float64, max_arc::Int)
    k = findfirst(==(power_kw), levels)
    k === nothing && error("make_edge_costs: power_kw=$power_kw is not in power_levels=$levels")
    n_power = length(levels)
    route_start_abs = toEpochSeconds(cur_time, cur_day, cur_month, cur_year)

    ws_buf  = Vector{Float32}(undef, max_arc)
    twa_buf = Vector{Float32}(undef, max_arc)
    speed_tmp = Vector{Float32}(undef, n_power)
    tp_buf = Vector{Float64}(undef, n_power)
    ep_buf = Vector{Float64}(undef, n_power)
    vminp_buf = Vector{Float64}(undef, n_power)
    cursor = WindTimeCursor()

    cost = function (dist::Float64, bearing::Float64, arc_flats::Vector{Int}, elapsed_int::Int)
        n_samp   = length(arc_flats)
        seg_dist = dist / n_samp
        t_start_abs = Float64(route_start_abs + elapsed_int)
        _fill_edge_wind_time_aware!(ws_buf, twa_buf, cursor, wind_pool, interp, uv_cache,
                                     n_rows, n_flat, arc_flats, bearing, t_start_abs, seg_dist,
                                     polar.lut_ref, route_start_abs, cur_time, cur_day, cur_month, cur_year)
        ok = _stepped_edge_all_powers!(tp_buf, ep_buf, vminp_buf, speed_tmp, polar.slut,
                                        ws_buf, twa_buf, n_samp, seg_dist, levels, min_speed)
        (ok && vminp_buf[k] >= min_speed) || return (Inf, Inf)
        return (tp_buf[k], ep_buf[k])
    end

    execute = function (dist::Float64, bearing::Float64, arc_flats::Vector{Int}, elapsed_int::Int)
        n_samp   = length(arc_flats)
        seg_dist = dist / n_samp
        t_start_abs = Float64(route_start_abs + elapsed_int)
        _fill_edge_wind_time_aware!(ws_buf, twa_buf, cursor, wind_pool, interp, uv_cache,
                                     n_rows, n_flat, arc_flats, bearing, t_start_abs, seg_dist,
                                     polar.lut_pk, route_start_abs, cur_time, cur_day, cur_month, cur_year)
        t = 0.0; vmin = Inf
        for s in 1:n_samp
            v = Float64(lut_lookup(polar.lut_pk, ws_buf[s], twa_buf[s]))
            vmin = min(vmin, v)
            t += seg_dist / max(v, 0.5)          # same floor replay_route_real_wind uses
        end
        return (t, power_kw * t / 3600.0, vmin)
    end

    return (cost = cost, execute = execute)
end

"""
    findGreedyRoute(...; gfs_dirs, gfs_init_dates, ...) -> (path_coords, node_costs, final_energy, power_used)

No lookahead, no backtracking. Candidates are neighbours strictly closer to the goal by
wind-free OCEAN-PATH distance. Choice = lowest FORECAST time per metre of progress (forecast of
the current real day). The chosen hop is then sailed under REAL wind; node_costs / final_energy
are real. `verify_path` (e.g. hindsight.diag[:path_idxs]) is re-costed with the real-wind
execution model and printed; it should reproduce the hindsight time.
"""
function findGreedyRoute(
    grid_points  ::Matrix{Tuple{Float64,Float64}},
    start_idx    ::Tuple{Int,Int},
    goal_idx     ::Tuple{Int,Int},
    cur_time     ::Int,
    cur_day      ::Int,
    cur_month    ::Int,
    cur_year     ::Int,
    max_energy   ::Float64,
    power_kw     ::Float64,
    min_speed    ::Float64;
    gfs_dirs       ::Vector{String},
    gfs_init_dates ::Vector{Date},
    cycle_hour     ::Int = 0,
    available_hours::Vector{Int} = vcat(collect(0:120), collect(123:3:336)),
    power_levels   ::Vector{Float64} = [power_kw],
    ocean_mask     ::Union{Nothing, Matrix{Bool}} = nothing,
    Rmax           ::Int     = 4,
    heading_mode   ::Symbol  = :full,
    diagnostics    ::Bool    = true,
    verify_path    ::Union{Nothing,Vector{Tuple{Int,Int,Int}}} = nothing,
    committed::Bool = false,
    clim_dir       ::String  = "./ClimatologyOceanData/route",
    diag_out ::Union{Nothing,Dict{Symbol,Any}} = nothing
)
    length(gfs_dirs) == length(gfs_init_dates) ||
        error("findGreedyRoute: gfs_dirs and gfs_init_dates must have the same length")
    n_rows, n_cols = size(grid_points)
    ocean_mask === nothing && (ocean_mask = trues(n_rows, n_cols))
    n_flat = n_rows * n_cols
    t_total_ns = time_ns()

    # 1. Neighbour graph (same builder/settings as the Dijkstra solver)
    nb_flat = build_neighbour_cache(grid_points, ocean_mask, n_rows, n_cols, Rmax,
                                     27_800.0; heading_mode = heading_mode)
    println("Neighbour cache built (Rmax=$Rmax, heading_mode=$heading_mode)")
    max_arc = 2
    for col in nb_flat, entry in col
        max_arc = max(max_arc, length(entry[5]))
    end

    # 2. Wind-free ocean-path distance to goal (backward Dijkstra on the reversed graph)
    rev = [Tuple{Int,Float64}[] for _ in 1:n_flat]
    for j in 1:n_cols, i in 1:n_rows
        f = flat_idx(i, j, n_rows)
        for (ni, nj, dist, _b, _a, _r) in nb_flat[i, j]
            push!(rev[flat_idx(ni, nj, n_rows)], (f, dist))
        end
    end
    D  = fill(Inf, n_flat)
    gf = flat_idx(goal_idx[1], goal_idx[2], n_rows)
    D[gf] = 0.0
    heap = BinaryMinHeap{Tuple{Float64,Int}}()
    push!(heap, (0.0, gf))
    while !isempty(heap)
        d, f = pop!(heap)
        d > D[f] && continue
        for (g, w) in rev[f]
            nd = d + w
            if nd < D[g]
                D[g] = nd
                push!(heap, (nd, g))
            end
        end
    end
    rev = nothing
    d_start = D[flat_idx(start_idx[1], start_idx[2], n_rows)]
    isfinite(d_start) || error("findGreedyRoute: goal is unreachable from start on this neighbour graph.")
    println("Ocean-path distance start -> goal: $(round(d_start / 1000, digits=1)) km")

    # 3. Polar tables (built once, shared by every wind context)
    levels = sort(power_levels)
    ws1, wa1, bs1 = readPolar("./Polars/RisePol$(round(Int, levels[end]))kW.csv")
    ws2, wa2, bs2 = readPolar("./Polars/RisePol$(round(Int, power_kw))kW.csv")
    polar = (lut_ref = build_polar_lut((ws1, wa1, bs1)),
             lut_pk  = build_polar_lut((ws2, wa2, bs2)),
             slut    = build_stacked_lut(levels))

    # 4. Wind contexts. REAL uses the climatology archive; FORECAST is built per day.
    clim_src = _CLIMATOLOGY_ARCHIVE_SOURCE[]
    clim_src === nothing &&
        (clim_src = (dir = clim_dir, month_cache = Dict{Tuple{Int,Int}, Dict{Int,WindField}}()))

    function build_ctx(gfs_src, csrc, e_from::Int, raw_cache::Int, uv_cap::Int)
        pool = WindFieldPool(; max_cached = raw_cache)
        interp = _with_wind_source(gfs_src, csrc) do
            for e in (e_from - 6 * 3600):3600:(e_from + 72 * 3600)
                _ensureEpochRegistered!(pool, cur_time, cur_day, cur_month, cur_year, e)
            end
            ep0 = _ensureEpochRegistered!(pool, cur_time, cur_day, cur_month, cur_year, e_from)
            build_interpolator(grid_points, _getField!(pool, ep0))
        end
        uvc = UVGridCache(n_flat; max_cached = uv_cap)
        fns = make_edge_costs(nb_flat, pool, interp, uvc, n_rows, n_flat,
                              cur_time, cur_day, cur_month, cur_year,
                              power_kw, levels, polar, min_speed, max_arc)
        return GreedyWindCtx(gfs_src, csrc, pool, interp, uvc, fns.cost, fns.execute,
                             e_from + 72 * 3600)
    end

    # keep >= 72 h of epochs registered ahead of the current elapsed time
    function ensure_window!(ctx::GreedyWindCtx, t_needed::Int)
        ctx.reg_until >= t_needed + 72 * 3600 && return
        target = t_needed + 96 * 3600
        _with_wind_source(ctx.gfs_src, ctx.clim_src) do
            for e in (ctx.reg_until + 3600):3600:target
                _ensureEpochRegistered!(ctx.pool, cur_time, cur_day, cur_month, cur_year, e)
            end
        end
        ctx.reg_until = target
    end

    real_ctx = build_ctx(nothing, clim_src, 0, 24, 90)

    # 5. Optional: re-cost a reference path under REAL wind with the execution model
    if verify_path !== nothing
        t_chk = _with_wind_source(nothing, clim_src) do
            tc = 0.0
            for k in 1:length(verify_path)-1
                i, j   = verify_path[k][1],   verify_path[k][2]
                ni, nj = verify_path[k+1][1], verify_path[k+1][2]
                idx = findfirst(x -> x[1] == ni && x[2] == nj, nb_flat[i, j])
                if idx === nothing
                    println("Same-cost check: reference edge ($i,$j)->($ni,$nj) not in this graph -- Rmax/heading_mode mismatch?")
                    return NaN
                end
                _, _, dist, bearing, arc_flats, _ = nb_flat[i, j][idx]
                ensure_window!(real_ctx, round(Int, tc))
                ts, _, _ = real_ctx.execute(dist, bearing, arc_flats, round(Int, tc))
                tc += ts
            end
            tc
        end
        println("Same-cost check: reference path re-costed under real wind = " *
                "$(round(t_chk / 3600, digits=3)) h  (should equal the hindsight time)")
    end

    # 6. Diagnostics dicts (same keys build_stepper2 expects)
    node_band_confirm     = Dict{Tuple{Int,Int,Int}, Int8}()
    node_relax_ns_confirm = Dict{Tuple{Int,Int,Int}, Float64}()
    node_epoch_confirm    = Dict{Tuple{Int,Int,Int}, Int}()
    node_confirm_order    = Dict{Tuple{Int,Int,Int}, Int}()
    visit_counter = Ref(0)
    function record_visit!(i::Int, j::Int, epoch::Int, relax_ns::Float64)
        key = (i, j, 1)
        visit_counter[] += 1
        node_band_confirm[key]     = Int8(1)
        node_relax_ns_confirm[key] = relax_ns
        node_epoch_confirm[key]    = epoch
        node_confirm_order[key]    = visit_counter[]
    end

    # 7. The greedy walk
    ci, cj = start_idx
    t = 0.0; e = 0.0; path_len_m = 0.0
    t_pred_sum = 0.0            # forecast's predicted time for the hops actually taken
    n_below_min = 0             # hops the forecast said feasible but real wind made < min_speed
    path_idxs2   = [start_idx]
    node_costs   = [0.0]
    final_energy = [0.0]
    fc_ctx = nothing
    fc_day = 0

    while (ci, cj) != goal_idx
        t0 = time_ns()
        elapsed_int = round(Int, t)
        day_k = committed ? 1 : 1 + fld(elapsed_int, 86400)
        if day_k > length(gfs_dirs)
            println("Greedy route: voyage ran past the last downloaded forecast day " *
                    "($(length(gfs_dirs))) -- stopping.")
            return [], [], [], []
        end
        if day_k != fc_day
            gsrc   = _make_gfs_source(gfs_dirs[day_k], gfs_init_dates[day_k], cycle_hour, available_hours)
            fc_ctx = build_ctx(gsrc, nothing, (day_k - 1) * 86400, 8, 100)
            fc_day = day_k
        end
        ensure_window!(fc_ctx, elapsed_int)
        ensure_window!(real_ctx, elapsed_int)

        d_cur = D[flat_idx(ci, cj, n_rows)]

        # ---- DECIDE with the forecast ----
        best = _with_wind_source(fc_ctx.gfs_src, nothing) do
            b = (score = Inf, ni = 0, nj = 0, dist = 0.0, bearing = 0.0, arcs = Int[], t_fc = 0.0)
            for (ni, nj, dist, bearing, arc_flats, _r) in nb_flat[ci, cj]
                progress = d_cur - D[flat_idx(ni, nj, n_rows)]
                progress > 1.0 || continue                      # strictly closer by sea
                t_sec, e_kwh = fc_ctx.cost(dist, bearing, arc_flats, elapsed_int)
                isfinite(t_sec) || continue
                (e + e_kwh) > max_energy && continue
                score = t_sec / progress                          # forecast s per metre of progress
                if score < b.score
                    b = (score = score, ni = ni, nj = nj, dist = dist, bearing = bearing,
                         arcs = arc_flats, t_fc = t_sec)
                end
            end
            b
        end

        if best.ni == 0
            println("Greedy route: at ($ci,$cj) no closer neighbour is feasible under the " *
                    "forecast (min_speed or energy budget) -- stopping.")
            return [], [], [], []
        end

        # ---- EXECUTE under real wind ----
        t_real, e_real, vmin, ep_real = _with_wind_source(nothing, clim_src) do
            ex = real_ctx.execute(best.dist, best.bearing, best.arcs, elapsed_int)
            ep = _ensureEpochRegistered!(real_ctx.pool, cur_time, cur_day, cur_month, cur_year, elapsed_int)
            (ex[1], ex[2], ex[3], ep)
        end
        record_visit!(ci, cj, ep_real, Float64(time_ns() - t0))

        if e + e_real > max_energy
            println("Greedy route: real-wind energy would exceed the budget at ($ci,$cj) -- stopping.")
            return [], [], [], []
        end
        vmin < min_speed && (n_below_min += 1)

        t += t_real; e += e_real; path_len_m += best.dist; t_pred_sum += best.t_fc
        ci, cj = best.ni, best.nj
        push!(path_idxs2, (ci, cj)); push!(node_costs, t); push!(final_energy, e)
    end
    record_visit!(goal_idx[1], goal_idx[2], real_ctx.pool.sorted_epochs[end], 0.0)

    t_total = (time_ns() - t_total_ns) / 1e9

    stats = (hops = length(path_idxs2) - 1, path_km = path_len_m / 1000, shortest_km = d_start / 1000,
             detour = path_len_m / d_start, forecast_pred_h = t_pred_sum / 3600, real_h = t / 3600,
             below_min_hops = n_below_min, forecast_days_used = fc_day)

    if diag_out !== nothing
        diag_out[:path_idxs]             = [(ii, jj, 1) for (ii, jj) in path_idxs2]
        diag_out[:node_band_confirm]     = node_band_confirm
        diag_out[:node_relax_ns_confirm] = node_relax_ns_confirm
        diag_out[:node_epoch_confirm]    = node_epoch_confirm
        diag_out[:node_confirm_order]    = node_confirm_order
        diag_out[:total_confirmed]       = visit_counter[]
        diag_out[:total_loop_time_ns]    = time_ns() - t_total_ns
        diag_out[:nb_flat]               = nb_flat
        diag_out[:wind_pool]             = real_ctx.pool      # REAL wind, for the explorer
        diag_out[:uv_cache]              = real_ctx.uv
        diag_out[:interp]                = real_ctx.interp
        diag_out[:grid_points]           = grid_points
        diag_out[:cur_time]              = cur_time
        diag_out[:cur_day]               = cur_day
        diag_out[:cur_month]             = cur_month
        diag_out[:cur_year]              = cur_year
        diag_out[:ocean_mask]            = ocean_mask
        diag_out[:power_kw]              = power_kw
        diag_out[:node_band_final]       = fill(Int8(1), n_rows, n_cols)
        diag_out[:stats]                 = stats
    end

    path_coords = [grid_points[ii, jj] for (ii, jj) in path_idxs2]
    power_used  = fill(power_kw, length(path_idxs2))
    power_used[end] = 0.0

    if diagnostics
        println("\n=== Greedy route found (forecast decisions, real-wind execution) ===")
        println("  waypoints        : $(length(path_coords))")
        println("  time (real wind) : $(round(node_costs[end]/3600, digits=2)) h")
        println("  forecast-predicted time for the same hops : $(round(stats.forecast_pred_h, digits=2)) h")
        println("  energy           : $(round(final_energy[end], digits=1)) kWh / $(max_energy) kWh budget")
        println("  path length      : $(round(stats.path_km, digits=1)) km  ($(round(stats.detour, digits=3)) x the shortest ocean path)")
        println("  hops below min_speed under real wind : $(stats.below_min_hops)")
        println("  forecast days used : $(stats.forecast_days_used)")
        println("  wall time        : $(round(t_total, digits=2)) s")
    end

    return path_coords, node_costs, final_energy, power_used
end

function run_greedy_route(;
    grid_points::Matrix{Tuple{Float64,Float64}},
    ocean_mask ::Matrix{Bool},
    start_idx  ::Tuple{Int,Int},
    goal_idx   ::Tuple{Int,Int},
    cur_time::Int, cur_day::Int, cur_month::Int, cur_year::Int,
    max_energy::Float64,
    power_kw  ::Float64,
    min_speed ::Float64,
    gfs_dirs       ::Vector{String},
    gfs_init_dates ::Vector{Date},
    cycle_hour     ::Int = 0,
    available_hours::Vector{Int} = vcat(collect(0:120), collect(123:3:336)),
    Rmax      ::Int = 4,
    power_levels::Vector{Float64} = [power_kw],
    committed::Bool = false,
    verify_path::Union{Nothing,Vector{Tuple{Int,Int,Int}}} = nothing
)
    _GFS_FORECAST_SOURCE[] === nothing ||
        error("run_greedy_route: call clear_gfs_forecast_source!() first (this function manages both sources itself).")

    diag = Dict{Symbol,Any}()
    path, costs, energy, power = findGreedyRoute(
        grid_points, start_idx, goal_idx,
        cur_time, cur_day, cur_month, cur_year,
        max_energy, power_kw, min_speed;
        gfs_dirs = gfs_dirs, gfs_init_dates = gfs_init_dates,
        cycle_hour = cycle_hour, available_hours = available_hours,
        power_levels = power_levels, ocean_mask = ocean_mask, Rmax = Rmax,
        committed = committed, verify_path = verify_path, diag_out = diag
    )

    if isempty(path)
        return (path = path, costs = costs, energy = energy, power = power, diag = diag,
                stepper = nothing, stats = nothing, grid_points = grid_points, ocean_mask = ocean_mask)
    end

    stepper = build_stepper2(path, costs, energy, power, diag, grid_points, ocean_mask)
    return (path = path, costs = costs, energy = energy, power = power, diag = diag,
            stepper = stepper, stats = diag[:stats], grid_points = grid_points, ocean_mask = ocean_mask)
end

# =============================================================================
# compare_hindsight_vs_greedy -- plain-text/CSV fallback, no GLMakie needed.
# =============================================================================
function compare_hindsight_vs_greedy(hindsight_costs::Vector{Float64}, hindsight_energy::Vector{Float64},
                                      hindsight_path::Vector{Tuple{Float64,Float64}},
                                      greedy_result; csv_path::String = "hindsight_vs_greedy.csv")
    h_time_h = isempty(hindsight_costs) ? NaN : hindsight_costs[end] / 3600
    h_energy = isempty(hindsight_energy) ? NaN : hindsight_energy[end]
    g_time_h = isempty(greedy_result.costs) ? NaN : greedy_result.costs[end] / 3600
    g_energy = isempty(greedy_result.energy) ? NaN : greedy_result.energy[end]
 
    println("\n=== Hindsight-optimal vs. greedy ===")
    println(rpad("", 14), rpad("Hindsight", 14), rpad("Greedy", 14), "Greedy / Hindsight")
    println(rpad("Time (h)", 14), rpad(round(h_time_h, digits=2), 14),
            rpad(round(g_time_h, digits=2), 14), round(g_time_h / h_time_h, digits=3))
    println(rpad("Energy (kWh)", 14), rpad(round(h_energy, digits=1), 14),
            rpad(round(g_energy, digits=1), 14),
            isfinite(h_energy) && h_energy != 0 ? round(g_energy / h_energy, digits=3) : "n/a")
    println(rpad("Waypoints", 14), rpad(length(hindsight_path), 14), length(greedy_result.path))
 
    n = max(length(hindsight_path), length(greedy_result.path))
    rows = DataFrame(idx = 1:n,
        hindsight_lon = [i <= length(hindsight_path) ? hindsight_path[i][1] : missing for i in 1:n],
        hindsight_lat = [i <= length(hindsight_path) ? hindsight_path[i][2] : missing for i in 1:n],
        greedy_lon    = [i <= length(greedy_result.path) ? greedy_result.path[i][1] : missing for i in 1:n],
        greedy_lat    = [i <= length(greedy_result.path) ? greedy_result.path[i][2] : missing for i in 1:n])
    CSV.write(csv_path, rows)
    println("\nWaypoint comparison written to $csv_path")
 
    return (hindsight_time_h = h_time_h, hindsight_energy_kwh = h_energy,
            greedy_time_h = g_time_h, greedy_energy_kwh = g_energy)
end

# =============================================================================
# launch_hindsight_vs_greedy_explorer -- map + per-route wind panes + stats +
# route toggles + transport controls. Same layout as
# launch_five_route_comparison_explorer, just two routes and no wind-error plots.
#
# Uses helpers already defined elsewhere in your codebase: state_at_time,
# _draw_route_wind_zoom!, _wind_speed_dir, get_wind, nearest_index.
#
# Wind is loaded lazily from the steppers' own wind pools while you scrub, so this
# turns the climatology archive source ON and deliberately leaves it on. Don't call
# clear_climatology_archive_source!() until you've closed the window.
# =============================================================================
function launch_hindsight_vs_greedy_explorer(hindsight_stepper::RouteStepper2,
                                              greedy_stepper::RouteStepper2;
                                              n_frames::Int = 150, zoom_cells::Int = 12)
    set_climatology_archive_source!()

    grid_points = hindsight_stepper.grid_points
    ocean_mask  = hindsight_stepper.ocean_mask
    n_rows, n_cols = size(ocean_mask)
    lons_1d = [grid_points[1, j][1] for j in 1:n_cols]
    lats_1d = [grid_points[i, 1][2] for i in 1:n_rows]

    routes = [
        (name = "Hindsight", stepper = hindsight_stepper, color = :black,     style = :solid),
        (name = "Greedy",    stepper = greedy_stepper,    color = :firebrick, style = :dash),
    ]
    n_routes = length(routes)
    t_end = maximum(r.stepper.steps[end].cum_time_s for r in routes)

    # position + real wind for one route at global elapsed time t_s.
    # Wind lookup time is clamped to that route's own end, so a route that has
    # already finished never asks its wind pool for epochs it never registered.
    function route_state(rs::RouteStepper2, t_s::Float64)
        s  = state_at_time(rs, t_s)
        tt = min(t_s, rs.steps[end].cum_time_s)
        U, V = get_wind(rs.wind, tt)
        i = nearest_index(lats_1d, s.lat)
        j = nearest_index(lons_1d, s.lon)
        return (state = s, U = U, V = V, i = i, j = j)
    end

    # ---------------- figure ----------------
    fig = GLMakie.Figure(size = (1700, 1000))
    GLMakie.Label(fig[0, 1:2], "Hindsight vs Greedy — Explorer";
                  fontsize = 22, font = :bold, halign = :left, tellwidth = false)

    top_row = fig[1, 1] = GLMakie.GridLayout()

    ax_map = GLMakie.Axis(top_row[1, 1]; xlabel = "lon", ylabel = "lat")
    mask_z = permutedims(Float32.(ocean_mask))
    GLMakie.heatmap!(ax_map, lons_1d, lats_1d, mask_z;
                      colormap = [RGB(0.82, 0.75, 0.55), RGB(0.75, 0.87, 0.95)],
                      colorrange = (0, 1), interpolate = false)

    route_visible = [GLMakie.Observable(true) for _ in 1:n_routes]
    marker_pos = [GLMakie.Observable(GLMakie.Point2f(r.stepper.steps[1].lon, r.stepper.steps[1].lat))
                  for r in routes]
    for (k, r) in enumerate(routes)
        lons = [s.lon for s in r.stepper.steps]; lats = [s.lat for s in r.stepper.steps]
        GLMakie.lines!(ax_map, lons, lats; color = r.color, linewidth = 2.5, linestyle = r.style,
                        label = r.name, visible = route_visible[k])
        GLMakie.scatter!(ax_map, marker_pos[k]; color = r.color, markersize = 24, marker = :star5,
                          strokewidth = 2, strokecolor = :white, visible = route_visible[k])
    end
    GLMakie.scatter!(ax_map, [routes[1].stepper.steps[1].lon], [routes[1].stepper.steps[1].lat];
                      color = :lime, markersize = 14, marker = :diamond, label = "start")
    GLMakie.scatter!(ax_map, [routes[1].stepper.steps[end].lon], [routes[1].stepper.steps[end].lat];
                      color = :red, markersize = 14, marker = :diamond, label = "goal")
    GLMakie.axislegend(ax_map; position = :lb, labelsize = 9, patchsize = (12, 8),
                        framevisible = true, backgroundcolor = (:white, 0.85), margin = (4, 4, 4, 4))

    # wind panes: one per route, stacked, shared colour scale
    wind_col = top_row[1, 2] = GLMakie.GridLayout()
    ax_wind = [GLMakie.Axis(wind_col[k, 1]; xlabel = "lon", ylabel = "lat",
                             title = "$(r.name) — actual wind") for (k, r) in enumerate(routes)]
    wind_range = GLMakie.Observable((0.0, 15.0))
    GLMakie.Colorbar(wind_col[n_routes + 1, 1]; colorrange = wind_range, colormap = :viridis,
                      label = "wind speed (m/s)", vertical = false)

    # stats: one column per route
    stats_grid = fig[2, 1] = GLMakie.GridLayout()
    time_text = GLMakie.Observable("")
    GLMakie.Label(stats_grid[1, 1:n_routes], time_text; tellwidth = false, halign = :left,
                  fontsize = 13, font = :bold)
    route_texts = [GLMakie.Observable("") for _ in 1:n_routes]
    for k in 1:n_routes
        GLMakie.Label(stats_grid[2, k], route_texts[k]; tellwidth = false, halign = :left,
                      valign = :top, fontsize = 12, justification = :left,
                      visible = route_visible[k])
        GLMakie.colsize!(stats_grid, k, GLMakie.Relative(1.0 / n_routes))
    end

    # route toggles: slim strip to the right of the map
    toggle_col = fig[1, 2] = GLMakie.GridLayout()
    GLMakie.Label(toggle_col[1, 1]; text = "Routes", fontsize = 13, font = :bold, halign = :left)
    for (k, r) in enumerate(routes)
        t = GLMakie.Toggle(toggle_col[k + 1, 1], active = true)
        GLMakie.Label(toggle_col[k + 1, 2], r.name; fontsize = 11, color = r.color, halign = :left)
        on(t.active) do a
            route_visible[k][] = a
        end
    end

    # ---------------- update logic ----------------
    function fmt_route(name, s, ws, wd)
        done = s.clamped == :end ? "  (finished)" : ""
        return "$(name)$(done)\n" *
               "$(round(s.cum_time_s / 3600, digits=1)) h elapsed\n" *
               "$(round(s.dist_km, digits=0)) km  |  $(round(s.energy_kwh, digits=0)) kWh\n" *
               "$(round(s.speed_kn, digits=1)) kn  |  $(round(s.power_kw, digits=0)) kW\n" *
               "wind: $(round(ws, digits=1)) m/s @ $(round(wd, digits=0))°"
    end

    function update!(t_s::Float64)
        time_text[] = "elapsed time: $(round(t_s / 3600, digits=1)) h  ($(round(t_s / 86400, digits=1)) days)"

        rws = [route_state(r.stepper, t_s) for r in routes]

        max_ws = 1.0
        speeds = Vector{Tuple{Float64,Float64}}(undef, n_routes)
        for k in 1:n_routes
            ws, wd = _wind_speed_dir(rws[k].U, rws[k].V, rws[k].i, rws[k].j)
            speeds[k] = (ws, wd)
            max_ws = max(max_ws, ws)
        end
        wind_range[] = (0.0, max_ws + 5.0)

        for (k, r) in enumerate(routes)
            rw = rws[k]
            marker_pos[k][] = GLMakie.Point2f(rw.state.lon, rw.state.lat)
            route_texts[k][] = fmt_route(r.name, rw.state, speeds[k][1], speeds[k][2])

            _draw_route_wind_zoom!(ax_wind[k], rw.U, rw.V, rw.i, rw.j, grid_points, ocean_mask,
                                    wind_range; zoom_cells = zoom_cells,
                                    title = "$(r.name) — actual wind")
            GLMakie.scatter!(ax_wind[k], [rw.state.lon], [rw.state.lat];
                              color = r.color, markersize = 14, marker = :star5)
        end
    end

    # ---------------- transport controls ----------------
    controls = fig[3, 1:2] = GLMakie.GridLayout()
    current_t  = GLMakie.Observable(0.0)
    is_playing = GLMakie.Observable(false)

    btn_prev  = GLMakie.Button(controls[1, 1], label = "◀ Prev")
    btn_next  = GLMakie.Button(controls[1, 2], label = "Next ▶")
    btn_reset = GLMakie.Button(controls[1, 3], label = "⟲ Reset")
    btn_play  = GLMakie.Button(controls[1, 4], label = "▶ Play")
    scrub = GLMakie.Slider(controls[1, 5:7], range = range(0.0, t_end, length = n_frames), startvalue = 0.0)
    GLMakie.Label(controls[1, 8], GLMakie.lift(t -> "t = $(round(t/3600, digits=1)) h", current_t))

    step_dt = t_end / n_frames
    playback_interval = 0.1

    on(btn_prev.clicks)  do _; current_t[] = max(0.0, current_t[] - step_dt); end
    on(btn_next.clicks)  do _; current_t[] = min(t_end, current_t[] + step_dt); end
    on(btn_reset.clicks) do _; current_t[] = 0.0; end
    on(btn_play.clicks)  do _
        is_playing[] = !is_playing[]
        btn_play.label = is_playing[] ? "⏸ Pause" : "▶ Play"
    end
    on(scrub.value) do v; current_t[] = clamp(v, 0.0, t_end); end
    on(current_t)   do t; update!(t); end

    play_timer = Ref{Union{Nothing, Timer}}(nothing)
    on(is_playing) do playing
        if play_timer[] !== nothing
            close(play_timer[]); play_timer[] = nothing
        end
        if playing
            play_timer[] = Timer(playback_interval; interval = playback_interval) do _
                if current_t[] < t_end
                    current_t[] = min(t_end, current_t[] + step_dt)
                else
                    is_playing[] = false
                    btn_play.label = "▶ Play"
                end
            end
        end
    end

    # ---------------- sizing ----------------
    # ---------------- sizing (all relative, so it scales with any window size) ----------------
    GLMakie.colsize!(fig.layout, 1, GLMakie.Relative(0.92))
    GLMakie.colsize!(fig.layout, 2, GLMakie.Relative(0.08))
    GLMakie.colsize!(top_row, 1, GLMakie.Relative(0.68))
    GLMakie.colsize!(top_row, 2, GLMakie.Relative(0.32))

    # wind panes: two axes + shared colourbar
    GLMakie.rowsize!(wind_col, 1, GLMakie.Relative(0.44))
    GLMakie.rowsize!(wind_col, 2, GLMakie.Relative(0.44))
    GLMakie.rowsize!(wind_col, 3, GLMakie.Relative(0.12))

    GLMakie.rowsize!(fig.layout, 0, GLMakie.Relative(0.05))   # title
    GLMakie.rowsize!(fig.layout, 1, GLMakie.Relative(0.60))   # map + wind panes
    GLMakie.rowsize!(fig.layout, 2, GLMakie.Relative(0.17))   # stats
    GLMakie.rowsize!(fig.layout, 3, GLMakie.Relative(0.09))   # controls

    update!(0.0)
    GLMakie.display(fig)
    return fig
end