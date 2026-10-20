using DataStructures
using Statistics
using CairoMakie
using Colors: RGB
using JLD2
using GLMakie
using Printf

"""

Builds (or loads from `cache_path` if present) the Climatology and its
per-edge cost cache, then calls set_climatology_blend!. Do this ONCE per
grid/graph -- both are static w.r.t. day/epoch, so in run_rolling_replan
call this before the day loop starts, not inside it. Pass `cache_path` to
skip the (potentially slow) climatology scan + edge-cost precompute on
every subsequent run against the same grid.
"""
function prepare_climatology_blend!(
    grid_points::Matrix{Tuple{Float64,Float64}}, ocean_mask::Matrix{Bool},
    nb_flat::Matrix{Vector{Tuple{Int,Int,Float64,Float64,Vector{Int},Int8}}},
    power_levels::Vector{Float64}, min_speed::Float64;
    historical_dir::String = _climatology_dir(),
    historical_paths::Union{Nothing,Vector{String}} = nothing,
    months::Union{Nothing,Vector{Int}} = nothing,
    month_of_path::Union{Nothing,Vector{Int}} = nothing,
    full_forecast_days::Float64 = 14.0,
    n_ws::Int = 18, ws_max::Float32 = 30f0, n_dir::Int = 24,
    reliability_threshold::Float64 = 0.9,
    cache_path::Union{Nothing,String} = nothing, verbose::Bool = true
)
    plevels_f = Float64.(sort(power_levels))

    if cache_path !== nothing && isfile(cache_path)
        climatology, clim_edge_cache = jldopen(cache_path, "r") do f
            n_rows, n_cols = size(grid_points)
            clim = Climatology(f["roses"], n_rows, n_cols, f["months"])
            (clim, f["clim_edge_cache"])
        end
        verbose && println("Loaded climatology blend cache from $cache_path")
    else
        paths = historical_paths === nothing ?
            [joinpath(historical_dir, f) for f in readdir(historical_dir) if endswith(f, ".grib")] :
            historical_paths
        isempty(paths) && error("prepare_climatology_blend!: no historical files found in $historical_dir")
        # NEW: auto-derive month_of_path from filenames when a months filter
        # is given but month_of_path wasn't supplied explicitly -- matches
        # the "YYYY-MM.grib" naming used by the ./ClimatologyOceanData/route
        # bulk historical archive (e.g. "2006-01.grib" .. "2025-12.grib").
        if months !== nothing && month_of_path === nothing
            month_of_path = Vector{Int}(undef, length(paths))
            for (k, p) in enumerate(paths)
                m = match(r"(\d{4})-(\d{2})", basename(p))
                m === nothing && error("prepare_climatology_blend!: couldn't parse a YYYY-MM " *
                    "month/year out of filename \"$(basename(p))\" -- pass month_of_path " *
                    "explicitly if your historical archive uses different naming.")
                mo = parse(Int, m.captures[2])
                (1 <= mo <= 12) || error("prepare_climatology_blend!: parsed out-of-range " *
                    "month $mo from filename \"$(basename(p))\"")
                month_of_path[k] = mo
            end
            n_years_matched = count(==(months[1]), month_of_path)  # quick sanity signal for the common single-month case
            verbose && println("Derived month_of_path from filenames: $(length(paths)) files total, " *
                "$(count(m -> m in months, month_of_path)) match months=$months")
        end

        ref_field = _reference_field(paths[1])
        interp    = build_interpolator(grid_points, ref_field)
        climatology = build_climatology(paths, grid_points, interp, ocean_mask;
                                         n_ws, ws_max, n_dir, months, month_of_path)
        slut = build_stacked_lut(plevels_f)
        clim_edge_cache = build_climatology_edge_cache(nb_flat, ocean_mask, climatology, slut,
                                plevels_f, min_speed; reliability_threshold, verbose)
        if cache_path !== nothing
            jldopen(cache_path, "w") do f
                f["roses"] = climatology.roses; f["months"] = climatology.months
                f["clim_edge_cache"] = clim_edge_cache
            end
            verbose && println("Saved climatology blend cache to $cache_path")
        end
    end

    set_climatology_blend!(; climatology, clim_edge_cache, full_forecast_days)
    return (climatology = climatology, clim_edge_cache = clim_edge_cache)
end


# =============================================================================
# forecast_vs_actual.jl
#
# Requires gfs_forecast_source_patch.jl already applied to routing.jl.
#
# Workflow:
#   fc     = run_forecast_route(...)                       # solves using GFS
#   actual = replay_route_real_wind(fc.diag, fc.path, fc.power)   # ERA5 replay
#   rows   = compare_forecast_vs_actual(fc.stepper, actual)
#
# WHAT "actual" MEANS HERE (open-loop replay):
#   Same grid-node sequence AND same power-level-per-leg as the forecast
#   solve chose. Only the wind sampled along each edge's arc is swapped from
#   GFS-forecast to real ERA5, which changes local boat speed and therefore
#   how long each leg actually takes and how much energy it actually burns.
#   Since the edges themselves never change, "where you'd actually be" at a
#   given elapsed time is a point further along (or behind) the SAME
#   polyline the plan uses -- this measures schedule drift along the route,
#   not a sideways/geographic deviation from it. A genuine deviation
#   comparison would mean re-solving with real wind as ground truth instead.
# =============================================================================

using Dates
using DataFrames
using CSV
using CairoMakie
using JLD2: jldopen, jldsave

# NEIGHBOUR_TARGET_SPACING_M is defined in route_simulator.jl -- make sure
# that file is included before this one, since load_forecast_route uses it.

# -----------------------------------------------------------------------
# 1. Solve using the GFS forecast
# -----------------------------------------------------------------------
"""
    run_forecast_route(; gfs_dir, init_year, init_month, init_day, cycle_hour,
                          available_hours, start_location, end_location,
                          start_date, start_hhmm, kwargs...)

Builds the grid/mask, points the loader at the downloaded GFS run, solves
with findDijkstraRoutefinal, and returns everything needed for the replay
step. Leaves the GFS source SET on return (does not clear it) -- call
clear_gfs_forecast_source!() yourself right before replay_route_real_wind,
per the ordering note in gfs_forecast_source_patch.jl. wind_cache_size
defaults to covering every distinct forecast hour so nothing gets evicted
before you're done with it (e.g. if you later open the interactive
visualizer on `stepper`).
"""
function run_forecast_route(;
        gfs_dir::String,
        init_year::Int, init_month::Int, init_day::Int, cycle_hour::Int,
        available_hours::Vector{Int} = vcat(collect(0:120), collect(123:3:336)),
        start_location, end_location,
        lon_step::Float64 = 0.25, lat_step::Float64 = 0.25, padding::Int = 20,
        power_levels::Vector{Float64} = [0.0, 400.0, 1200.0, 2000.0],
        max_energy::Float64 = 150_000.0, bin_size::Float64 = 1000.0, min_speed::Float64 = 1.0,
        start_date::Date, start_hhmm::Int,
        Rmax::Int = 4,
        wind_cache_size::Int = length(vcat(collect(0:120), collect(123:3:336))) + 4,
        kwargs...
    )
    grid_points, latitudes, longitudes, ocean_mask =
        _build_grid_and_mask(; start_location, end_location, lon_step, lat_step, padding)

    start_idx = (nearest_index(latitudes,  start_location[2]),
                 nearest_index(longitudes, start_location[1]))
    goal_idx  = (nearest_index(latitudes,  end_location[2]),
                 nearest_index(longitudes, end_location[1]))
    _check_endpoints(ocean_mask, start_idx, goal_idx)

    set_gfs_forecast_source!(; dir = gfs_dir, init_year, init_month, init_day,
                               cycle_hour, available_hours)

    ct = start_hhmm
    cd = day(start_date); cm = month(start_date); cy = year(start_date)

    diag = Dict{Symbol,Any}()
    (path, costs, energy, power), t_wall = run_timed(() ->
        findDijkstraRoutefinal(
            grid_points, start_idx, goal_idx, ct, cd, cm, cy,
            max_energy, power_levels, bin_size, min_speed;
            ocean_mask         = ocean_mask,
            Rmax               = Rmax,
            wind_cache_size    = wind_cache_size,
            record_diagnostics = true,
            diag_out           = diag,
            kwargs...
        )
    )

    isempty(path) && error("Forecast route infeasible within the given energy budget.")

    stepper = build_stepper2(path, costs, energy, power, diag, grid_points, ocean_mask)

    println("Forecast route solved: $(length(path)) waypoints, " *
            "time=$(round(costs[end]/3600, digits=2))h, " *
            "energy=$(round(energy[end], digits=1)) kWh, wall=$(round(t_wall, digits=2))s")

    return (stepper = stepper, diag = diag, path = path, costs = costs, energy = energy,
            power = power, grid_points = grid_points, ocean_mask = ocean_mask, wall_s = t_wall,
            rmax = Rmax, gfs_dir = gfs_dir, gfs_init_year = init_year,
            gfs_init_month = init_month, gfs_init_day = init_day, gfs_cycle_hour = cycle_hour,
            gfs_available_hours = available_hours)
end

# -----------------------------------------------------------------------
# Save / reload — skip re-running the (expensive) solve entirely.
#
# Same reasoning as save_steppers/load_steppers in route_simulator.jl:
# does NOT serialize wind_pool/interp/nb_flat (wind_pool's LRU only holds
# whatever was cached at save time, so dumping it verbatim could go stale
# or incomplete for a step you scrub to later; nb_flat is a cheap ~2s
# geometry pass, not worth storing) or labels_power (a ~60MB 3D array on
# a grid this size, and nothing downstream of run_forecast_route actually
# reads it — build_stepper2 takes power_used directly, not diag[:labels_power]).
# -----------------------------------------------------------------------

"""
    save_forecast_route(filepath, fc)

Persists everything needed to skip re-running run_forecast_route. `fc` is
whatever run_forecast_route returned. `filepath` should end in `.jld2`.
"""
function save_forecast_route(filepath::String, fc::NamedTuple)
    diag = fc.diag
    jldopen(filepath, "w") do f
        f["path"]        = fc.path
        f["costs"]       = fc.costs
        f["energy"]      = fc.energy
        f["power"]       = fc.power
        f["grid_points"] = fc.grid_points
        f["ocean_mask"]  = fc.ocean_mask
        f["wall_s"]      = fc.wall_s

        f["path_idxs"]             = diag[:path_idxs]
        f["cur_time"]              = diag[:cur_time]
        f["cur_day"]               = diag[:cur_day]
        f["cur_month"]             = diag[:cur_month]
        f["cur_year"]              = diag[:cur_year]
        f["node_band_confirm"]     = diag[:node_band_confirm]
        f["node_relax_ns_confirm"] = diag[:node_relax_ns_confirm]
        f["node_epoch_confirm"]    = diag[:node_epoch_confirm]
        f["node_confirm_order"]    = diag[:node_confirm_order]
        f["total_confirmed"]       = diag[:total_confirmed]
        f["total_loop_time_ns"]    = diag[:total_loop_time_ns]

        f["rmax"]                = fc.rmax
        f["gfs_dir"]             = fc.gfs_dir
        f["gfs_init_year"]       = fc.gfs_init_year
        f["gfs_init_month"]      = fc.gfs_init_month
        f["gfs_init_day"]        = fc.gfs_init_day
        f["gfs_cycle_hour"]      = fc.gfs_cycle_hour
        f["gfs_available_hours"] = fc.gfs_available_hours
    end
    println("Saved forecast route to $filepath (nb_flat/wind_pool rebuilt on load, not stored)")
end

"""
    load_forecast_route(filepath; rebuild_wind = true) -> fc

Reloads a forecast route saved by save_forecast_route WITHOUT re-running the
solver. Rebuilds nb_flat (fast geometry pass) and, if rebuild_wind=true, a
fresh GFS wind pool + interpolator (only needed if you'll open the
interactive stepper's zoom pane on fc.stepper) by re-pointing at the same
gfs_dir/init settings used originally -- those GFS files must still be on
disk for that. Set rebuild_wind=false to skip that entirely (e.g. if you
only want fc.path/costs/energy/power/diag for replay_route_real_wind,
which builds its own separate real-wind pool and never touches this one).
"""
function load_forecast_route(filepath::String; rebuild_wind::Bool = true)
    data = jldopen(filepath, "r") do f
        (
            path = f["path"], costs = f["costs"], energy = f["energy"], power = f["power"],
            grid_points = f["grid_points"], ocean_mask = f["ocean_mask"], wall_s = f["wall_s"],
            path_idxs = f["path_idxs"],
            cur_time = f["cur_time"], cur_day = f["cur_day"],
            cur_month = f["cur_month"], cur_year = f["cur_year"],
            node_band_confirm = f["node_band_confirm"],
            node_relax_ns_confirm = f["node_relax_ns_confirm"],
            node_epoch_confirm = f["node_epoch_confirm"],
            node_confirm_order = f["node_confirm_order"],
            total_confirmed = f["total_confirmed"],
            total_loop_time_ns = f["total_loop_time_ns"],
            rmax = f["rmax"], gfs_dir = f["gfs_dir"],
            gfs_init_year = f["gfs_init_year"], gfs_init_month = f["gfs_init_month"],
            gfs_init_day = f["gfs_init_day"], gfs_cycle_hour = f["gfs_cycle_hour"],
            gfs_available_hours = f["gfs_available_hours"]
        )
    end

    n_rows, n_cols = size(data.grid_points)
    nb_flat = build_neighbour_cache(data.grid_points, data.ocean_mask, n_rows, n_cols,
                                     data.rmax, NEIGHBOUR_TARGET_SPACING_M)

    diag = Dict{Symbol,Any}(
        :path_idxs => data.path_idxs, :grid_points => data.grid_points,
        :ocean_mask => data.ocean_mask, :cur_time => data.cur_time, :cur_day => data.cur_day,
        :cur_month => data.cur_month, :cur_year => data.cur_year, :nb_flat => nb_flat,
        :node_band_confirm => data.node_band_confirm,
        :node_relax_ns_confirm => data.node_relax_ns_confirm,
        :node_epoch_confirm => data.node_epoch_confirm,
        :node_confirm_order => data.node_confirm_order,
        :total_confirmed => data.total_confirmed,
        :total_loop_time_ns => data.total_loop_time_ns
    )

    stepper = nothing
    if rebuild_wind
        set_gfs_forecast_source!(; dir = data.gfs_dir, init_year = data.gfs_init_year,
                                   init_month = data.gfs_init_month, init_day = data.gfs_init_day,
                                   cycle_hour = data.gfs_cycle_hour,
                                   available_hours = data.gfs_available_hours)
        wind_pool  = WindFieldPool(; max_cached = length(data.gfs_available_hours) + 4)
        init_ep    = _ensureEpochRegistered!(wind_pool, data.cur_time, data.cur_day,
                                              data.cur_month, data.cur_year, 0)
        init_field = _getField!(wind_pool, init_ep)
        for hh in data.gfs_available_hours
            hh == 0 && continue   # already fetched as init_ep
            ep = _ensureEpochRegistered!(wind_pool, data.cur_time, data.cur_day,
                                          data.cur_month, data.cur_year, hh * 3600)
            _getField!(wind_pool, ep)   # force fetch now, while GFS source is still set
        end
        interp     = build_interpolator(data.grid_points, init_field)
        clear_gfs_forecast_source!()   # don't leave it set as a side effect of loading

        diag[:wind_pool] = wind_pool
        diag[:interp]    = interp
        stepper = build_stepper2(data.path, data.costs, data.energy, data.power, diag,
                                  data.grid_points, data.ocean_mask)
    end

    println("Loaded forecast route from $filepath" *
            (rebuild_wind ? "" : " (stepper not rebuilt -- rebuild_wind=false)"))

    return (stepper = stepper, diag = diag, path = data.path, costs = data.costs,
            energy = data.energy, power = data.power, grid_points = data.grid_points,
            ocean_mask = data.ocean_mask, wall_s = data.wall_s,
            rmax = data.rmax, gfs_dir = data.gfs_dir,
            gfs_init_year = data.gfs_init_year, gfs_init_month = data.gfs_init_month,
            gfs_init_day = data.gfs_init_day, gfs_cycle_hour = data.gfs_cycle_hour,
            gfs_available_hours = data.gfs_available_hours)
end

# -----------------------------------------------------------------------
# 2. Replay the SAME route/power schedule through real ERA5 wind
# -----------------------------------------------------------------------
"""
    replay_route_real_wind(diag, path_coords, power_used;
                            wind_cache_size = 24, target_spacing_m = 27_800.0,
                            max_elapsed_s = nothing)

Walks the forecast route's exact waypoint sequence and power schedule,
re-evaluating actual local boat speed sample-by-sample from real ERA5 wind.

If `max_elapsed_s` is provided (e.g. 24*3600 for rolling replay), the replay
stops internally as soon as that much real time has elapsed, creating a
synthetic endpoint at exactly `max_elapsed_s`. This avoids calculating the
entire route when only the first day will be used.

MUST be called with the GFS forecast source cleared (clear_gfs_forecast_source!())
so the default ERA5 loader is used.
"""
function replay_route_real_wind(
        diag::Dict, path_coords, power_used;
        wind_cache_size::Int = 24,
        target_spacing_m::Float64 = 27_800.0,
        max_elapsed_s::Union{Float64,Nothing} = nothing,
        planned_time_s::Union{Float64,Nothing} = nothing,
        stop_at_waypoint::Bool = false
    )
    _GFS_FORECAST_SOURCE[] === nothing ||
        error("replay_route_real_wind: call clear_gfs_forecast_source!() first -- " *
              "this needs the default ERA5 loader, not the GFS one.")

    grid_points = diag[:grid_points]
    ocean_mask  = diag[:ocean_mask]
    path_idxs   = diag[:path_idxs]
    cur_time    = diag[:cur_time]; cur_day = diag[:cur_day]
    cur_month   = diag[:cur_month]; cur_year = diag[:cur_year]

    n_rows, n_cols = size(grid_points)
    n_flat = n_rows * n_cols
    n_legs = length(path_idxs) - 1
    route_start_abs = toEpochSeconds(cur_time, cur_day, cur_month, cur_year)

    # Pre-registration window based on max_elapsed_s (for rolling) or 
    # conservative estimate from distance (for full replay)
    if max_elapsed_s !== nothing
        # For rolling replay: only need data for 24h + small buffer
        max_hours_est = ceil(Int, max_elapsed_s / 3600) + 2
    else
        # For full replay: use conservative distance/speed estimate
        total_dist_km = sum(getDistance(path_coords[k][1], path_coords[k][2],
                                         path_coords[k+1][1], path_coords[k+1][2]) / 1000.0
                             for k in 1:n_legs)
        max_hours_est = if planned_time_s !== nothing
            ceil(Int, planned_time_s / 3600 * 1.25) + 12
        else
            ceil(Int, total_dist_km / (1.0 * 1.852) * 1.5) + 24
        end
        
    end

    wind_pool = WindFieldPool(; max_cached = wind_cache_size)
    init_ep   = _ensureEpochRegistered!(wind_pool, cur_time, cur_day, cur_month, cur_year, 0)
    for hh in 1:max_hours_est
        _ensureEpochRegistered!(wind_pool, cur_time, cur_day, cur_month, cur_year, hh * 3600)
    end
    init_field = _getField!(wind_pool, init_ep)
    interp     = build_interpolator(grid_points, init_field)
    uv_cache   = UVGridCache(n_flat; max_cached = length(wind_pool.sorted_epochs) + 2)

    # Polar LUTs only for the power levels this route actually used.
    used_powers = sort(unique(filter(p -> p >= 0.0, power_used)))
    lut_cache = Dict{Float64, PolarLUT}()
    for P_kw in used_powers
        p_int = round(Int, P_kw)
        ws, wa, bs = readPolar("$(POLAR_DIR_REF[])/RisePol$(p_int)kW.csv")
        lut_cache[P_kw] = build_polar_lut((ws, wa, bs))
    end

    cursor = WindTimeCursor()
    
    # If max_elapsed_s is provided, we'll dynamically grow steps as needed
    steps = StepInfo[]
    
    i0, j0, e0 = path_idxs[1]
    lon0, lat0 = path_coords[1]
    push!(steps, StepInfo(i0, j0, e0, lon0, lat0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          Int8(0), -1, 0.0, 1, 0.0))

    t_now = 0.0
    e_now = 0.0
    cum_dist_km = 0.0

    for k in 1:n_legs
        i, j, _   = path_idxs[k]
        ni, nj, e_next = path_idxs[k+1]
        lon1, lat1 = path_coords[k]
        lon2, lat2 = path_coords[k+1]

        dist_m  = getDistance(lon1, lat1, lon2, lat2)
        bearing = getBoatAngle(lon1, lat1, lon2, lat2)
        P_kw    = power_used[k]

        if P_kw < 0.0
            error("replay_route_real_wind: unexpected sentinel power on leg $k -- " *
                "this should be unreachable now that climatology uses cost blending, " *
                "not a terminal jump. Check findDijkstraRoutefinal wasn't reverted.")
        end

        lut     = lut_cache[P_kw]

        n_samp   = max(1, round(Int, dist_m / target_spacing_m))
        seg_dist = dist_m / n_samp
        bearing32 = Float32(bearing)
        t_start_abs = Float64(route_start_abs) + t_now

        t_leg = 0.0
        
        # Track position within this leg for potential partial endpoint
        leg_lon_start = lon1
        leg_lat_start = lat1

        for s in 1:n_samp
            alpha = (s - 0.5) / n_samp
            si = clamp(round(Int, i + alpha * (ni - i)), 1, n_rows)
            sj = clamp(round(Int, j + alpha * (nj - j)), 1, n_cols)
            flat_ix = (sj - 1) * n_rows + si

            u, v = _cursor_uv_at_time!(cursor, wind_pool, interp, uv_cache,
                                        n_flat, flat_ix, t_start_abs + t_leg,
                                        route_start_abs, cur_time, cur_day, cur_month, cur_year)
            ws   = sqrt(u^2 + v^2)
            wdir = mod(rad2deg(atan(u, v)) + 180f0, 360f0)
            twa  = abs(mod(wdir - bearing32 + 180f0, 360f0) - 180f0)

            v_local = max(Float64(lut_lookup(lut, ws, twa)), 0.5)  # floor avoids div-by-zero
            dt = seg_dist / v_local
            

            # old partial-endpoint behaviour, now only when stop_at_waypoint is false
            if !stop_at_waypoint && max_elapsed_s !== nothing && t_now + t_leg + dt > max_elapsed_s
                remaining = max_elapsed_s - (t_now + t_leg)
                
                # Partial segment: calculate position at exactly max_elapsed_s
                partial_dist = v_local * remaining
                fraction = partial_dist / seg_dist
                
                # Calculate the start point of this sample
                alpha_start = (s - 1.0) / n_samp
                lon_samp_start = lon1 + alpha_start * (lon2 - lon1)
                lat_samp_start = lat1 + alpha_start * (lat2 - lat1)
                
                # Interpolate to the cutoff point
                alpha_end = s / n_samp
                lon_samp_end = lon1 + alpha_end * (lon2 - lon1)
                lat_samp_end = lat1 + alpha_end * (lat2 - lat1)

                lon_partial = lon_samp_start + fraction * (lon_samp_end - lon_samp_start)
                lat_partial = lat_samp_start + fraction * (lat_samp_end - lat_samp_start)
                
                # Update energy and time
                t_leg += remaining
                e_now += P_kw * remaining / 3600.0
                t_now += t_leg
                
                # Calculate partial leg distance
                partial_leg_dist_km = (seg_dist * (s - 1 + fraction)) / 1000.0
                cum_dist_km += partial_leg_dist_km
                
                # Create the synthetic endpoint step
                leg_spd_kn = t_leg > 0 ? (partial_leg_dist_km / 1.852) / (t_leg / 3600.0) : 0.0
                push!(steps, StepInfo(ni, nj, e_next, lon_partial, lat_partial,
                                       t_now, e_now, cum_dist_km, partial_leg_dist_km, 
                                       t_leg, leg_spd_kn, P_kw,
                                       Int8(0), -1, 0.0, length(steps) + 1, 0.0))
                
                # Build and return the truncated stepper
                wind_cache_obj = WindCache(wind_pool, interp, cur_time, cur_day, cur_month, cur_year)
                println("Real-wind replay stopped at $(round(t_now/3600, digits=2))h of $(round(max_elapsed_s/3600, digits=2))h")
                println("  Position: (lon=$(round(lon_partial, digits=3)), lat=$(round(lat_partial, digits=3)))")
                
                return RouteStepper2(steps, grid_points, ocean_mask, diag[:nb_flat], 
                                     wind_cache_obj, length(steps))
            end
            
            t_leg += dt
            e_now += P_kw * dt / 3600.0
        end

        t_now += t_leg
        leg_dist_km = dist_m / 1000.0
        cum_dist_km += leg_dist_km
        leg_spd_kn = t_leg > 0 ? (leg_dist_km / 1.852) / (t_leg / 3600.0) : 0.0

        if t_leg / 3600.0 > 5.0   # flag any single leg eating more than 5h
            println("  Leg $k: $(round(t_leg/3600,digits=1))h  " *
                     "dist=$(round(leg_dist_km,digits=1))km  " *
                     "P=$(P_kw)kW  avg_spd=$(round(leg_spd_kn,digits=2))kn")
        end

        push!(steps, StepInfo(ni, nj, e_next, lon2, lat2,
                               t_now, e_now, cum_dist_km, leg_dist_km, t_leg, leg_spd_kn, P_kw,
                               Int8(0), -1, 0.0, length(steps) + 1, 0.0))
        # Stop at the first waypoint reached at or after the replan interval.
        if stop_at_waypoint && max_elapsed_s !== nothing &&
        t_now >= max_elapsed_s

            wind_cache_obj = WindCache(wind_pool, interp, cur_time, cur_day, cur_month, cur_year)

            println("Real-wind replay stopped at first waypoint after interval: " *
                    "$(round(t_now / 3600, digits=2))h " *
                    "(limit $(round(max_elapsed_s / 3600, digits=2))h)")

            return RouteStepper2(
                steps,
                grid_points,
                ocean_mask,
                diag[:nb_flat],
                wind_cache_obj,
                length(steps)
            )
        end
    end

    wind_cache_obj = WindCache(wind_pool, interp, cur_time, cur_day, cur_month, cur_year)
    println("Real-wind replay done: time=$(round(t_now/3600, digits=2))h, " *
            "energy=$(round(e_now, digits=1)) kWh")

    return RouteStepper2(steps, grid_points, ocean_mask, diag[:nb_flat], wind_cache_obj,
                          length(steps))
end

# -----------------------------------------------------------------------
# 3. Position/state at an arbitrary elapsed time (extends the existing
#    dijkstra_state_at_time pattern in route_simulator.jl to also report
#    lon/lat, since here BOTH routes share the same physical waypoints and
#    the interesting comparison is where-along-the-track each one is).
# -----------------------------------------------------------------------
function state_at_time(rs::RouteStepper2, t_s::Float64)
    steps = rs.steps
    n = length(steps)
    if n == 1 || t_s <= steps[1].cum_time_s
        s = steps[1]
        return (lon = s.lon, lat = s.lat, dist_km = s.cum_dist_km,
                energy_kwh = s.cum_energy_kwh, cum_time_s = s.cum_time_s,
                speed_kn = 0.0, power_kw = 0.0, clamped = :start)
    end
    if t_s >= steps[end].cum_time_s
        s = steps[end]
        return (lon = s.lon, lat = s.lat, dist_km = s.cum_dist_km,
                energy_kwh = s.cum_energy_kwh, cum_time_s = s.cum_time_s,
                speed_kn = s.leg_speed_kn, power_kw = s.leg_power_kw, clamped = :end)
    end

    times = [st.cum_time_s for st in steps]
    k = clamp(searchsortedfirst(times, t_s), 2, n)
    s0, s1 = steps[k-1], steps[k]
    dt = s1.cum_time_s - s0.cum_time_s
    f  = dt > 0 ? (t_s - s0.cum_time_s) / dt : 0.0

    lon = s0.lon + f * (s1.lon - s0.lon)
    lat = s0.lat + f * (s1.lat - s0.lat)
    dist_km    = s0.cum_dist_km    + f * (s1.cum_dist_km    - s0.cum_dist_km)
    energy_kwh = s0.cum_energy_kwh + f * (s1.cum_energy_kwh - s0.cum_energy_kwh)
    return (lon = lon, lat = lat, dist_km = dist_km, energy_kwh = energy_kwh,
        cum_time_s = s0.cum_time_s + f * (s1.cum_time_s - s0.cum_time_s),
        speed_kn = s1.leg_speed_kn, power_kw = s1.leg_power_kw, clamped = :mid)
end

# -----------------------------------------------------------------------
# 4. Comparison report + plot
# -----------------------------------------------------------------------
"""
    compare_forecast_vs_actual(forecast_rs, actual_rs; n_samples=48,
                                save_csv=true, csv_name="forecast_vs_actual.csv",
                                out_png="forecast_vs_actual.png")

Samples both routes at matched elapsed times and reports, at each sample,
the along-track offset (great-circle distance between the two routes'
positions at that same elapsed time -- see the module docstring for what
this does and doesn't capture) and the cumulative energy difference.
Prints a summary and saves a CSV + a two-panel plot.
"""
function compare_forecast_vs_actual(forecast_rs::RouteStepper2, actual_rs::RouteStepper2;
                                     n_samples::Int = 48, save_csv::Bool = true,
                                     csv_name::String = "forecast_vs_actual.csv",
                                     out_png::String = "forecast_vs_actual.png")
    t_end = max(forecast_rs.steps[end].cum_time_s, actual_rs.steps[end].cum_time_s)
    ts = range(0.0, t_end; length = n_samples)

    rows = DataFrame(TimeH = Float64[], PlannedLon = Float64[], PlannedLat = Float64[],
                      ActualLon = Float64[], ActualLat = Float64[],
                      AlongTrackOffsetKm = Float64[], EnergyDiffKwh = Float64[])

    for t in ts
        f = state_at_time(forecast_rs, t)
        a = state_at_time(actual_rs, t)
        offset_km = getDistance(f.lon, f.lat, a.lon, a.lat) / 1000.0
        push!(rows, (t / 3600, f.lon, f.lat, a.lon, a.lat, offset_km, a.energy_kwh - f.energy_kwh))
    end

    save_csv && (CSV.write(csv_name, rows); println("Saved $csv_name"))

    planned_time_h = forecast_rs.steps[end].cum_time_s / 3600
    actual_time_h  = actual_rs.steps[end].cum_time_s / 3600
    planned_energy = forecast_rs.steps[end].cum_energy_kwh
    actual_energy  = actual_rs.steps[end].cum_energy_kwh

    println("\n=== Forecast plan vs actual (real ERA5 wind) replay ===")
    println("  Planned total time:   $(round(planned_time_h, digits=2)) h")
    println("  Actual total time:    $(round(actual_time_h, digits=2)) h  " *
            "($(round(100*(actual_time_h-planned_time_h)/planned_time_h, digits=1))% vs plan)")
    println("  Planned total energy: $(round(planned_energy, digits=1)) kWh")
    println("  Actual total energy:  $(round(actual_energy, digits=1)) kWh  " *
            "($(round(100*(actual_energy-planned_energy)/planned_energy, digits=1))% vs plan)")
    println("  Max along-track offset at any matched time: " *
            "$(round(maximum(rows.AlongTrackOffsetKm), digits=1)) km")
    println("\n  NOTE: this is an OPEN-LOOP replay (same edges/power schedule as the forecast")
    println("  route, re-timed under real wind). 'Offset' = how far ahead/behind schedule the")
    println("  boat would actually be along the SAME path -- not a sideways deviation from it.")

    # ---- plot: along-track offset + cumulative energy, vs elapsed time ----
    fig = CairoMakie.Figure(size = (1200, 500))
    ax1 = CairoMakie.Axis(fig[1, 1]; xlabel = "elapsed time (h)", ylabel = "along-track offset (km)",
                           title = "Actual position vs planned position (same path)")
    CairoMakie.lines!(ax1, rows.TimeH, rows.AlongTrackOffsetKm; color = :firebrick)
    CairoMakie.hlines!(ax1, [0.0]; color = :gray60, linestyle = :dash)

    ax2 = CairoMakie.Axis(fig[1, 2]; xlabel = "elapsed time (h)", ylabel = "cumulative energy diff (kWh)",
                           title = "Actual − planned energy")
    CairoMakie.lines!(ax2, rows.TimeH, rows.EnergyDiffKwh; color = :steelblue)
    CairoMakie.hlines!(ax2, [0.0]; color = :gray60, linestyle = :dash)

    CairoMakie.save(out_png, fig)
    println("Saved $out_png")
    display(fig)

    return rows
end


using Dates
using JLD2
using CairoMakie
using Colors: RGB

# =============================================================================
# rolling_replan.jl
#
# Rolling (receding-horizon) daily re-forecast experiment.
#
# Answers: "if the boat re-plans every day against a FRESH GFS forecast
# cycle -- rather than committing once to the day-0 forecast for the whole
# voyage -- how much closer does it get to what would have been possible
# with perfect foreknowledge of the real wind (ERA5)?"
#
# Three routes get compared, all scored under REAL (ERA5) wind:
#   1. rolling replan        -- run_rolling_replan            (this file)
#   2. single-forecast-committed -- run_single_forecast_committed (this file,
#      thin wrapper around run_forecast_route + replay_route_real_wind)
#   3. hindsight (perfect foreknowledge) -- run_hindsight_route (this file,
#      solves directly against ERA5, i.e. the best any forecast could do)
#
# DEPENDENCIES / INCLUDE ORDER -- include this file AFTER:
#   read_write_data.jl / times_and_dates.jl  (getDistance, getBoatAngle,
#       toEpochSeconds, DotEnv/CDSAPI plumbing, etc.)
#   wind.jl        (WindFieldPool, set_gfs_forecast_source!,
#                    clear_gfs_forecast_source!, _GFS_FORECAST_SOURCE,
#                    findDijkstraRoutefinal, build_neighbour_cache,
#                    build_interpolator, _ensureEpochRegistered!, _getField!)
#   the main routing file that defines _build_grid_and_mask, _check_endpoints,
#       nearest_index (used throughout the existing codebase already)
#   route_simulator.jl   (StepInfo, RouteStepper2, WindCache, get_wind,
#                          build_stepper2, NEIGHBOUR_TARGET_SPACING_M)
#   forecast_actual_explorer.jl  (run_forecast_route, replay_route_real_wind,
#                                  state_at_time, save/load_forecast_route)
#
# DATA PATH NOTE: replay_route_real_wind / run_hindsight_route read ERA5
# through the DEFAULT per-timestamp branch of _loadWindField
# (./OceanData/$(ut)-$(nd)$(monthWord(nm))$(ny).grib, one file per
# date+time, downloaded on demand via _download_wind_file! if missing).
# This is a DIFFERENT archive from the monthly-batched climatology archive
# under ./ClimatologyOceanData/route (bulk_download_climatology_batched) --
# the two don't share files, so make sure the per-timestamp ERA5 data your
# backtest window needs is actually available (or downloadable on demand)
# before running this.
# =============================================================================

# -----------------------------------------------------------------------
# 0. Helper: build gfs_dirs/gfs_init_dates from the naming convention
#    written by download_gfs_rolling_series.sh, so you don't hand-type them.
# -----------------------------------------------------------------------
"""
    rolling_gfs_dirs(outbase, start_date, num_days, cycle_hour) -> (dirs, dates)

Builds the `gfs_dirs` / `gfs_init_dates` arrays matching the directory
naming convention written by download_gfs_rolling_series.sh
(`OUTBASE_YYYYMMDD_CCz`), in date order.
"""
function rolling_gfs_dirs(outbase::String, start_date::Date, num_days::Int, cycle_hour::Int)
    dates = [start_date + Day(d) for d in 0:num_days-1]
    dirs  = [string(outbase, "_", Dates.format(dt, "yyyymmdd"), "_", lpad(cycle_hour, 2, '0'), "z")
             for dt in dates]
    return dirs, dates
end

# -----------------------------------------------------------------------
# 1. Per-day log
# -----------------------------------------------------------------------
struct RollingDayLog
    day_index         ::Int
    date              ::Date
    cycle_hour        ::Int
    start_idx         ::Tuple{Int,Int}
    start_energy_used ::Float64
    fc_path_idxs      ::Vector{Tuple{Int,Int,Int}}
    fc_path_coords    ::Vector{Tuple{Float64,Float64}}
    fc_costs          ::Vector{Float64}
    fc_energy         ::Vector{Float64}
    fc_power          ::Vector{Float64}
    actual_stepper    ::RouteStepper2   # real-wind replay up to daily cutoff (24h)
    landed_idx        ::Tuple{Int,Int}
    landed_lon        ::Float64
    landed_lat        ::Float64
    landed_dist_km    ::Float64
    landed_time_s     ::Float64         # real elapsed seconds actually used THIS DAY
    landed_energy_kwh ::Float64         # real energy used THIS DAY (not cumulative)
    reached_goal      ::Bool
end

# -----------------------------------------------------------------------
# 2. Nearest-ocean-cell snap, for landing the boat on the grid after a
#    real-wind replay interpolation.
# -----------------------------------------------------------------------
function _nearest_ocean_index(ocean_mask::Matrix{Bool}, latitudes::Vector{Float64},
                               longitudes::Vector{Float64}, lon::Float64, lat::Float64;
                               max_radius::Int = 8)
    i0 = nearest_index(latitudes, lat)
    j0 = nearest_index(longitudes, lon)
    n_rows, n_cols = size(ocean_mask)
    if 1 <= i0 <= n_rows && 1 <= j0 <= n_cols && ocean_mask[i0, j0]
        return (i0, j0)
    end
    for r in 1:max_radius
        for di in -r:r, dj in -r:r
            max(abs(di), abs(dj)) == r || continue
            ni, nj = i0 + di, j0 + dj
            (1 <= ni <= n_rows && 1 <= nj <= n_cols) || continue
            ocean_mask[ni, nj] && return (ni, nj)
        end
    end
    error("_nearest_ocean_index: no ocean cell found within $max_radius cells of " *
          "(lon=$lon, lat=$lat) -- the real-wind replay landed the boat far from any " *
          "ocean cell on the grid; increase max_radius or inspect the replay.")
end

# -----------------------------------------------------------------------
# 3. The rolling-replan driver
# -----------------------------------------------------------------------
"""
    run_rolling_replan(; gfs_dirs, gfs_init_dates, cycle_hour, available_hours,
                          start_location, end_location, start_date, start_hhmm,
                          power_levels, max_energy, bin_size, min_speed, Rmax,
                          replan_interval_s = 24*3600.0, max_days = length(gfs_dirs),
                          kwargs...)

Each iteration:
  1. Points the loader at that day's GFS cycle (`gfs_dirs[day]` /
     `gfs_init_dates[day]`) and solves findDijkstraRoutefinal from the
     boat's ACTUAL current grid cell (not the original start) to the goal,
     with the ENERGY BUDGET REDUCED by whatever has really been spent so far.
  2. Replays that whole plan through REAL ERA5 wind (replay_route_real_wind)
     to find out what actually would have happened.
  3. Truncates that replay at `replan_interval_s` (default 24h) of REAL
     elapsed time -- or at the goal, if the plan is real-wind-feasible in
     under that -- and snaps the resulting position to the nearest ocean
     grid cell. That becomes next iteration's start.

Stops when the goal is reached, `max_days` is exhausted, or a day's solve
comes back infeasible within the remaining energy budget.

`gfs_dirs` and `gfs_init_dates` must be the same length and in date order
(see `rolling_gfs_dirs`). `replan_interval_s` other than exactly 24h will
make the calendar-day advance (`cur_date += Day(1)`) drift out of sync with
real elapsed time -- fine for testing, but you should adjust the date
advance yourself in that case rather than relying on the default here.

Returns a NamedTuple with `day_logs::Vector{RollingDayLog}` plus shared
grid/goal/summary fields. Extra `kwargs` are passed through to
findDijkstraRoutefinal (e.g. `dynamic_bands`, `heading_mode`, ...).
"""
function run_rolling_replan(;
        gfs_dirs::Vector{String},
        gfs_init_dates::Vector{Date},
        cycle_hour::Int = 0,
        available_hours::Vector{Int} = vcat(collect(0:120), collect(123:3:336)),
        start_location, end_location,
        lon_step::Float64 = 0.25, lat_step::Float64 = 0.25, padding::Int = 20,
        power_levels::Vector{Float64} = [0.0, 400.0, 1200.0, 2000.0],
        max_energy::Float64 = 150_000.0, bin_size::Float64 = 1000.0, min_speed::Float64 = 1.0,
        start_date::Date, start_hhmm::Int = 0,
        Rmax::Int = 4,
        replan_interval_s::Float64 = 24 * 3600.0,
        max_days::Int = 100,
        max_iterations::Int = 3 * max_days,
        clock_resolution_min::Int = 1,
        wind_cache_size::Int = length(available_hours) + 4,
        replay_wind_cache_size::Int = 24,
        verbose::Bool = true,
        use_climatology_blend::Bool = false,
        climatology_cache_path::Union{Nothing,String} = nothing,
        climatology_full_forecast_days::Float64 = 14.0,
        plan_margin::Float64 = 0.0,
        kwargs...
    )
    length(gfs_dirs) == length(gfs_init_dates) ||
        error("run_rolling_replan: gfs_dirs and gfs_init_dates must have matching length.")
    isempty(gfs_dirs) && error("run_rolling_replan: gfs_dirs is empty -- nothing to run.")

    grid_points, latitudes, longitudes, ocean_mask =
        _build_grid_and_mask(; start_location, end_location, lon_step, lat_step, padding)
    start_idx = (nearest_index(latitudes,  start_location[2]),
                 nearest_index(longitudes, start_location[1]))
    goal_idx  = (nearest_index(latitudes,  end_location[2]),
                 nearest_index(longitudes, end_location[1]))
    _check_endpoints(ocean_mask, start_idx, goal_idx)

    n_rows, n_cols = size(grid_points)
    nb_flat_shared = build_neighbour_cache(grid_points, ocean_mask, n_rows, n_cols, Rmax, 27_800.0)

    if use_climatology_blend
        climatology_cache_path === nothing &&
            error("run_rolling_replan: use_climatology_blend=true requires climatology_cache_path")
        prepare_climatology_blend!(grid_points, ocean_mask, nb_flat_shared, power_levels, min_speed;
                                    cache_path = climatology_cache_path,
                                    full_forecast_days = climatology_full_forecast_days)
    end

    set_climatology_archive_source!()

    t0    = DateTime(start_date) + Hour(start_hhmm ÷ 100) + Minute(start_hhmm % 100)
    clock = t0
    cur_idx = start_idx
    total_energy = 0.0
    total_time_s = 0.0
    reached_goal = false
    day_logs = RollingDayLog[]

    for iter in 1:max_iterations
        if total_time_s >= max_days * 86400.0
            verbose && println("Reached max sailing time of $max_days day(s) -- stopping.")
            break
        end

        cur_date = Date(clock)
        cur_hhmm = Dates.hour(clock) * 100 + Dates.minute(clock)

        # Most recent forecast cycle issued on or before today; if the clock has run
        # past the last downloaded cycle, keep reusing the last one.
        cyc = findlast(<=(cur_date), gfs_init_dates)
        if cyc === nothing
            verbose && println("No GFS cycle on or before $cur_date -- stopping.")
            break
        end
        gfs_dir = gfs_dirs[cyc]
        init_dt = gfs_init_dates[cyc]
        if verbose && init_dt < cur_date
            println("  (using cycle $init_dt, $(Dates.value(cur_date - init_dt)) day(s) old)")
        end
        gfs_dir = gfs_dirs[cyc]
        init_dt = gfs_init_dates[cyc]
        # --- stop cleanly (DNF) instead of crashing in the solver ---
        if cyc == length(gfs_init_dates) &&
           clock >= DateTime(init_dt) + Hour(cycle_hour) + Hour(maximum(available_hours))
            verbose && println("  Last forecast cycle's horizon exhausted -- stopping (DNF).")
            break
        end
        left = max_energy - total_energy
        if left <= 0.0 && !(0.0 in power_levels)
            verbose && println("  No fuel and no 0 kW level -- stopping (DNF).")
            break
        end

        verbose && println("\n=== Rolling replan: iter $iter  (clock=$clock, cycle=$init_dt, " *
                            "position=$(grid_points[cur_idx...]), " *
                            "energy_used=$(round(total_energy, digits=1))/$(max_energy) kWh) ===")

        set_gfs_forecast_source!(; dir = gfs_dir, init_year = year(init_dt),
                                   init_month = month(init_dt), init_day = day(init_dt),
                                   cycle_hour = cycle_hour, available_hours = available_hours)

        diag = Dict{Symbol,Any}()
        remaining_energy = max_energy * (1 - plan_margin) - total_energy
        remaining_energy < min(bin_size, left) && (remaining_energy = left)   # reserve used up: use what's really left
        remaining_energy = max(remaining_energy, 1.0)                          # DP needs a positive budget
        bin_today = remaining_energy >= 20 * bin_size ? bin_size : remaining_energy / 20
        verbose && bin_today < bin_size &&
            println("  Low fuel: budget $(round(remaining_energy, digits=1)) kWh, bin $(round(bin_today, digits=2)) kWh")
        local path, costs, energy, power
        try
            path, costs, energy, power = findDijkstraRoutefinal(
                grid_points, cur_idx, goal_idx,
                cur_hhmm, day(cur_date), month(cur_date), year(cur_date),
                remaining_energy, power_levels, bin_today, min_speed;
                ocean_mask = ocean_mask, Rmax = Rmax, wind_cache_size = wind_cache_size,
                record_diagnostics = true, diag_out = diag, kwargs...
            )
        finally
            clear_gfs_forecast_source!()
        end

        if isempty(path)
            verbose && println("  No feasible forecast plan from here within remaining energy -- stopping.")
            break
        end

        actual_full = replay_route_real_wind(diag, path, power;
                                             wind_cache_size = replay_wind_cache_size,
                                             max_elapsed_s   = replan_interval_s,
                                             stop_at_waypoint = true)

        last_step = actual_full.steps[end]
        if length(actual_full.steps) < 2 || last_step.cum_time_s <= 0
            verbose && println("  Replay made no progress -- stopping.")
            break
        end

        day_reached_goal = length(actual_full.steps) == length(path)
        landed_idx = day_reached_goal ? goal_idx : (last_step.i, last_step.j)

        push!(day_logs, RollingDayLog(
            iter, cur_date, cycle_hour, cur_idx, total_energy,
            diag[:path_idxs], path, costs, energy, power,
            actual_full, landed_idx, last_step.lon, last_step.lat, last_step.cum_dist_km,
            last_step.cum_time_s,          # <<< time to the waypoint, NOT 24 h
            last_step.cum_energy_kwh, day_reached_goal
        ))

        verbose && println("  Replay stopped after $(round(last_step.cum_time_s / 3600, digits=2))h at " *
                           "grid cell $(landed_idx)" * (day_reached_goal ? "  -- GOAL REACHED" : ""))

        total_time_s += last_step.cum_time_s     # <<< only the time actually sailed
        total_energy  = day_logs[end].start_energy_used + last_step.cum_energy_kwh
        cur_idx       = landed_idx
        reached_goal  = day_reached_goal
        reached_goal && break

        # <<< clock moves forward by the time actually sailed, not by 24 h
        clock = round(t0 + Millisecond(round(Int, total_time_s * 1000)), Minute(clock_resolution_min))
    end

    if verbose
        reached_goal ?
            println("\nRolling replan reached the goal: total time=$(round(total_time_s/3600, digits=2))h, " *
                    "energy=$(round(total_energy, digits=1))kWh") :
            println("\nRolling replan did NOT reach the goal (sailed $(round(total_time_s/86400, digits=2)) days).")
    end
    clear_climatology_blend!()
    clear_climatology_archive_source!()
    return (day_logs = day_logs, reached_goal = reached_goal,
            total_time_s = total_time_s, total_energy_kwh = total_energy,
            grid_points = grid_points, ocean_mask = ocean_mask,
            latitudes = latitudes, longitudes = longitudes,
            start_idx = start_idx, goal_idx = goal_idx,
            max_energy = max_energy, replan_interval_s = replan_interval_s, rmax = Rmax)
end

# -----------------------------------------------------------------------
# 4. Baselines: single-forecast-committed, and hindsight (perfect foreknowledge)
# -----------------------------------------------------------------------
"""
    run_single_forecast_committed(; kwargs...) -> (forecast, actual)

Baseline for comparison against the rolling replan: solve ONCE against the
day-0 forecast for the whole voyage, then replay that single committed plan
through real ERA5 wind for its entire length -- i.e. what actually happens
if you never re-plan. `kwargs` are passed straight to `run_forecast_route`
(gfs_dir, init_year/month/day, cycle_hour, start_location, end_location,
start_date, start_hhmm, power_levels, max_energy, bin_size, min_speed,
Rmax, ...).
"""
function run_single_forecast_committed(; kwargs...)
    fc = run_forecast_route(; kwargs...)
    clear_gfs_forecast_source!()
    set_climatology_archive_source!()   # NEW
    actual = replay_route_real_wind(fc.diag, fc.path, fc.power; planned_time_s = fc.costs[end])
    clear_climatology_archive_source!()   # NEW
    return (forecast = fc, actual = actual)
end

"""
    run_hindsight_route(; start_location, end_location, start_date, start_hhmm,
                           power_levels, max_energy, bin_size, min_speed, Rmax, kwargs...)

Solves directly against REAL ERA5 wind (no forecast source set) -- the best
any forecast, however good, could possibly do. This is the ceiling that
both the rolling replan and the single-forecast-committed baseline are
measured against. Requires clear_gfs_forecast_source!() to already be the
active state (errors otherwise, since mixing GFS-source and ERA5-solve
would silently give a route that doesn't mean what it looks like it means).
"""
function run_hindsight_route(;
        start_location, end_location,
        lon_step::Float64 = 0.25, lat_step::Float64 = 0.25, padding::Int = 20,
        power_levels::Vector{Float64} = [0.0, 400.0, 1200.0, 2000.0],
        max_energy::Float64 = 150_000.0, bin_size::Float64 = 1000.0, min_speed::Float64 = 1.0,
        start_date::Date, start_hhmm::Int,
        Rmax::Int = 4, wind_cache_size::Int = 24,
        kwargs...
    )
    _GFS_FORECAST_SOURCE[] === nothing ||
        error("run_hindsight_route: call clear_gfs_forecast_source!() first -- this route must " *
              "be solved against the default ERA5 loader, not a GFS forecast source.")
    set_climatology_archive_source!()   # NEW

    grid_points, latitudes, longitudes, ocean_mask =
        _build_grid_and_mask(; start_location, end_location, lon_step, lat_step, padding)
    start_idx = (nearest_index(latitudes, start_location[2]), nearest_index(longitudes, start_location[1]))
    goal_idx  = (nearest_index(latitudes, end_location[2]),   nearest_index(longitudes, end_location[1]))
    _check_endpoints(ocean_mask, start_idx, goal_idx)

    diag = Dict{Symbol,Any}()
    path, costs, energy, power = findDijkstraRoutefinal(
        grid_points, start_idx, goal_idx,
        start_hhmm, day(start_date), month(start_date), year(start_date),
        max_energy, power_levels, bin_size, min_speed;
        ocean_mask = ocean_mask, Rmax = Rmax, wind_cache_size = wind_cache_size,
        record_diagnostics = true, diag_out = diag, kwargs...
    )
    isempty(path) && error("Hindsight (real-wind) route infeasible within the given energy budget.")

    stepper = build_stepper2(path, costs, energy, power, diag, grid_points, ocean_mask)
    println("Hindsight (real ERA5 wind, perfect foreknowledge) route solved: " *
            "$(length(path)) waypoints, time=$(round(costs[end] / 3600, digits=2))h, " *
            "energy=$(round(energy[end], digits=1)) kWh")

    clear_climatology_archive_source!()   # NEW
    return (stepper = stepper, diag = diag, path = path, costs = costs, energy = energy,
            power = power, grid_points = grid_points, ocean_mask = ocean_mask,
            start_idx = start_idx, goal_idx = goal_idx)
end

# -----------------------------------------------------------------------
# 5. Stitch the day-by-day rolling-replan logs into ONE continuous
#    RouteStepper2 -- what was actually sailed, under real wind.
# -----------------------------------------------------------------------

function build_rolling_stepper(result::NamedTuple)
    isempty(result.day_logs) && error("build_rolling_stepper: no day logs to stitch.")

    steps = StepInfo[]
    t_off = 0.0; e_off = 0.0; d_off = 0.0

    for log in result.day_logs
        for (idx, s) in enumerate(log.actual_stepper.steps)
            idx == 1 && !isempty(steps) && continue     # same waypoint as previous day's landing
            s.cum_time_s > log.landed_time_s + 1e-6 && continue
            push!(steps, StepInfo(s.i, s.j, s.e_bin, s.lon, s.lat,
                                   s.cum_time_s + t_off, s.cum_energy_kwh + e_off,
                                   s.cum_dist_km + d_off, s.leg_dist_km, s.leg_time_s,
                                   s.leg_speed_kn, s.leg_power_kw, s.band, s.epoch,
                                   s.relax_ms, length(steps) + 1, s.solver_time_s))
        end
        t_off = steps[end].cum_time_s
        e_off = steps[end].cum_energy_kwh
        d_off = steps[end].cum_dist_km
    end

    first_log = result.day_logs[1]
    return RouteStepper2(steps, result.grid_points, result.ocean_mask,
                          first_log.actual_stepper.nb_flat, first_log.actual_stepper.wind,
                          length(steps))
end

# -----------------------------------------------------------------------
# 6. Comparison report + plot
# -----------------------------------------------------------------------
"""
    compare_rolling_vs_committed(result, committed_stepper;
                                  hindsight_stepper = nothing,
                                  out_png = "rolling_vs_committed.png")

Prints a summary table and saves a plot comparing the rolling-replan track
(built via build_rolling_stepper) against the single-forecast-committed
track and, optionally, the hindsight (perfect-foreknowledge) track.
`committed_stepper` / `hindsight_stepper` must already carry REAL (ERA5)
cumulative time/energy -- i.e. pass `run_single_forecast_committed(...).actual`
and `run_hindsight_route(...).stepper`, not forecast-clock steppers.
"""
function compare_rolling_vs_committed(result::NamedTuple, committed_stepper::RouteStepper2;
                                       hindsight_stepper::Union{Nothing,RouteStepper2} = nothing,
                                       out_png::String = "rolling_vs_committed.png")
    rolling_stepper = build_rolling_stepper(result)

    rt_h = rolling_stepper.steps[end].cum_time_s / 3600
    re_k = rolling_stepper.steps[end].cum_energy_kwh
    ct_h = committed_stepper.steps[end].cum_time_s / 3600
    ce_k = committed_stepper.steps[end].cum_energy_kwh

    println("\n=== Rolling replan vs single-forecast-committed (both under REAL wind) ===")
    println(rpad("", 24), rpad("Time (h)", 14), rpad("Energy (kWh)", 14))
    println(rpad("Rolling replan", 24), rpad(string(round(rt_h, digits=2)), 14),
            rpad(string(round(re_k, digits=1)), 14))
    println(rpad("Single-forecast", 24), rpad(string(round(ct_h, digits=2)), 14),
            rpad(string(round(ce_k, digits=1)), 14))

    if hindsight_stepper !== nothing
        ht_h = hindsight_stepper.steps[end].cum_time_s / 3600
        he_k = hindsight_stepper.steps[end].cum_energy_kwh
        println(rpad("Hindsight (perfect)", 24), rpad(string(round(ht_h, digits=2)), 14),
                rpad(string(round(he_k, digits=1)), 14))

        gap_committed = 100 * (ct_h - ht_h) / ht_h
        gap_rolling   = 100 * (rt_h - ht_h) / ht_h
        println("\nTime penalty vs hindsight-optimal:")
        println("  single-forecast: +$(round(gap_committed, digits=1))%")
        println("  rolling replan:  +$(round(gap_rolling, digits=1))%")
        println("(Rolling replan should sit at or below single-forecast's gap -- that " *
                "difference is the measured value of re-planning with fresh forecasts.)")
    end

    fig = CairoMakie.Figure(size = (1300, 950))
    ax = CairoMakie.Axis(fig[1, 1]; title = "Rolling replan vs single-forecast-committed (real wind)",
                          xlabel = "lon", ylabel = "lat")

    n_rows, n_cols = size(result.ocean_mask)
    lons = [result.grid_points[1, j][1] for j in 1:n_cols]
    lats = [result.grid_points[i, 1][2] for i in 1:n_rows]
    mask_z = permutedims(Float32.(result.ocean_mask))
    CairoMakie.heatmap!(ax, lons, lats, mask_z;
                         colormap = [RGB(0.82, 0.75, 0.55), RGB(0.93, 0.93, 0.93)],
                         colorrange = (0, 1), interpolate = false)

    r_lons = [s.lon for s in rolling_stepper.steps]; r_lats = [s.lat for s in rolling_stepper.steps]
    c_lons = [s.lon for s in committed_stepper.steps]; c_lats = [s.lat for s in committed_stepper.steps]
    CairoMakie.lines!(ax, r_lons, r_lats; color = :steelblue, linewidth = 2.5, label = "rolling replan")
    CairoMakie.lines!(ax, c_lons, c_lats; color = :firebrick, linewidth = 2.5, linestyle = :dash,
                       label = "single-forecast-committed")

    if hindsight_stepper !== nothing
        h_lons = [s.lon for s in hindsight_stepper.steps]; h_lats = [s.lat for s in hindsight_stepper.steps]
        CairoMakie.lines!(ax, h_lons, h_lats; color = :black, linewidth = 1.5, linestyle = :dot,
                           label = "hindsight (perfect foreknowledge)")
    end

    for day_log in result.day_logs
        CairoMakie.scatter!(ax, [day_log.landed_lon], [day_log.landed_lat];
                             color = :orange, markersize = 8, marker = :circle)
    end
    CairoMakie.scatter!(ax, Float64[], Float64[]; color = :orange, markersize = 8,
                         marker = :circle, label = "daily replan point")

    CairoMakie.scatter!(ax, [rolling_stepper.steps[1].lon], [rolling_stepper.steps[1].lat];
                         color = :lime, markersize = 14, marker = :diamond, label = "start")
    CairoMakie.scatter!(ax, [rolling_stepper.steps[end].lon], [rolling_stepper.steps[end].lat];
                         color = :red, markersize = 14, marker = :diamond, label = "goal")
    CairoMakie.axislegend(ax; position = :lb)

    CairoMakie.save(out_png, fig)
    println("Saved $out_png")

    return (rolling_stepper = rolling_stepper, rolling_time_h = rt_h, rolling_energy_kwh = re_k,
            committed_time_h = ct_h, committed_energy_kwh = ce_k)
end

# -----------------------------------------------------------------------
# 7. Save / reload -- skip re-running every day's solve + replay.
#
# Same reasoning as save_steppers/save_forecast_route elsewhere: does NOT
# serialize WindFieldPool objects (a day's cache only holds whatever was
# loaded at save time) or nb_flat (cheap geometry pass, rebuilt on load).
# -----------------------------------------------------------------------
"""
    save_rolling_replan(path, result)

Persists a run_rolling_replan result, minus the per-day WindFieldPool/
neighbour-cache objects (rebuilt cheaply on load).
"""
function save_rolling_replan(path::String, result::NamedTuple)
    jldopen(path, "w") do f
        f["shared/grid_points"]       = result.grid_points
        f["shared/ocean_mask"]        = result.ocean_mask
        f["shared/start_idx"]         = result.start_idx
        f["shared/goal_idx"]          = result.goal_idx
        f["shared/max_energy"]        = result.max_energy
        f["shared/replan_interval_s"] = result.replan_interval_s
        f["shared/rmax"]              = result.rmax
        f["shared/reached_goal"]      = result.reached_goal
        f["shared/total_time_s"]      = result.total_time_s
        f["shared/total_energy_kwh"]  = result.total_energy_kwh
        f["shared/n_days"]            = length(result.day_logs)

        for (k, log) in enumerate(result.day_logs)
            g = "day_$k"
            f["$g/day_index"]         = log.day_index
            f["$g/date"]              = log.date
            f["$g/cycle_hour"]        = log.cycle_hour
            f["$g/start_idx"]         = log.start_idx
            f["$g/start_energy_used"] = log.start_energy_used
            f["$g/fc_path_idxs"]      = log.fc_path_idxs
            f["$g/fc_path_coords"]    = log.fc_path_coords
            f["$g/fc_costs"]          = log.fc_costs
            f["$g/fc_energy"]         = log.fc_energy
            f["$g/fc_power"]          = log.fc_power
            f["$g/actual_steps"]      = log.actual_stepper.steps
            f["$g/wind_ct"]           = log.actual_stepper.wind.ct
            f["$g/wind_cd"]           = log.actual_stepper.wind.cd
            f["$g/wind_cm"]           = log.actual_stepper.wind.cm
            f["$g/wind_cy"]           = log.actual_stepper.wind.cy
            f["$g/landed_idx"]        = log.landed_idx
            f["$g/landed_lon"]        = log.landed_lon
            f["$g/landed_lat"]        = log.landed_lat
            f["$g/landed_dist_km"]    = log.landed_dist_km
            f["$g/landed_time_s"]     = log.landed_time_s
            f["$g/landed_energy_kwh"] = log.landed_energy_kwh
            f["$g/reached_goal"]      = log.reached_goal
        end
    end
    println("Saved rolling replan ($(length(result.day_logs)) day(s)) to $path " *
            "(wind pools/nb_flat rebuilt on load, not stored)")
end

"""
    load_rolling_replan(path; wind_cache_size = 24) -> result

Reloads a rolling-replan run saved by save_rolling_replan, without
re-running any solve or replay. Rebuilds one lightweight ERA5
WindFieldPool + interpolator per day (only needed for that day's
actual_stepper zoom pane) and a single shared neighbour cache at the
saved Rmax.
"""
function load_rolling_replan(path::String; wind_cache_size::Int = 24)
    day_logs = RollingDayLog[]
    local grid_points, ocean_mask, start_idx, goal_idx
    local max_energy, replan_interval_s, rmax, reached_goal, total_time_s, total_energy_kwh

    jldopen(path, "r") do f
        grid_points       = f["shared/grid_points"]
        ocean_mask        = f["shared/ocean_mask"]
        start_idx         = f["shared/start_idx"]
        goal_idx          = f["shared/goal_idx"]
        max_energy        = f["shared/max_energy"]
        replan_interval_s = f["shared/replan_interval_s"]
        rmax              = f["shared/rmax"]
        reached_goal      = f["shared/reached_goal"]
        total_time_s      = f["shared/total_time_s"]
        total_energy_kwh  = f["shared/total_energy_kwh"]
        n_days            = f["shared/n_days"]

        n_rows, n_cols = size(grid_points)
        nb_flat = build_neighbour_cache(grid_points, ocean_mask, n_rows, n_cols, rmax,
                                         NEIGHBOUR_TARGET_SPACING_M)

        set_climatology_archive_source!()
        for k in 1:n_days
            g = "day_$k"
            steps = f["$g/actual_steps"]
            ct = f["$g/wind_ct"]; cd = f["$g/wind_cd"]; cm = f["$g/wind_cm"]; cy = f["$g/wind_cy"]

            wind_pool  = WindFieldPool(; max_cached = wind_cache_size)
            init_ep    = _ensureEpochRegistered!(wind_pool, ct, cd, cm, cy, 0)
            init_field = _getField!(wind_pool, init_ep)
            interp     = build_interpolator(grid_points, init_field)
            wind       = WindCache(wind_pool, interp, ct, cd, cm, cy)

            actual_stepper = RouteStepper2(steps, grid_points, ocean_mask, nb_flat, wind, length(steps))

            push!(day_logs, RollingDayLog(
                f["$g/day_index"], f["$g/date"], f["$g/cycle_hour"], f["$g/start_idx"],
                f["$g/start_energy_used"], f["$g/fc_path_idxs"], f["$g/fc_path_coords"],
                f["$g/fc_costs"], f["$g/fc_energy"], f["$g/fc_power"], actual_stepper,
                f["$g/landed_idx"], f["$g/landed_lon"], f["$g/landed_lat"],
                f["$g/landed_dist_km"], f["$g/landed_time_s"], f["$g/landed_energy_kwh"],
                f["$g/reached_goal"]
            ))
        end
        clear_climatology_archive_source!()
    end


    println("Loaded rolling replan ($(length(day_logs)) day(s)) from $path")
    return (day_logs = day_logs, reached_goal = reached_goal, total_time_s = total_time_s,
            total_energy_kwh = total_energy_kwh, grid_points = grid_points,
            ocean_mask = ocean_mask, latitudes = nothing, longitudes = nothing,
            start_idx = start_idx, goal_idx = goal_idx, max_energy = max_energy,
            replan_interval_s = replan_interval_s, rmax = rmax)
end

# -----------------------------------------------------------------------
# Save / reload run_single_forecast_committed's result.
#
# committed = (forecast = fc, actual = actual). fc is exactly what
# run_forecast_route returns; actual is a RouteStepper2 from
# replay_route_real_wind against the SAME path/power schedule under real
# (ERA5) wind. Saved together so load_committed_run gives back a matching
# pair without re-solving or re-replaying either half.
#
# Same reasoning as save_forecast_route: no WindFieldPool/nb_flat stored,
# rebuilt on load (nb_flat from grid_points/ocean_mask/rmax, wind pools
# fresh from disk -- GFS files for fc.stepper, default ERA5 loader for
# actual).
# -----------------------------------------------------------------------

"""
    save_committed_run(path, committed)

Persists a run_single_forecast_committed(...) result -- both the forecast
plan (fc) and its real-wind replay (actual) -- so it can be reloaded
without re-solving or re-replaying.
"""
function save_committed_run(path::String, committed::NamedTuple)
    fc     = committed.forecast
    actual = committed.actual
    diag   = fc.diag

    jldopen(path, "w") do f
        # --- forecast half (same fields as save_forecast_route) ---
        f["fc/path"]        = fc.path
        f["fc/costs"]       = fc.costs
        f["fc/energy"]      = fc.energy
        f["fc/power"]       = fc.power
        f["fc/grid_points"] = fc.grid_points
        f["fc/ocean_mask"]  = fc.ocean_mask
        f["fc/wall_s"]      = fc.wall_s

        f["fc/path_idxs"]             = diag[:path_idxs]
        f["fc/cur_time"]              = diag[:cur_time]
        f["fc/cur_day"]               = diag[:cur_day]
        f["fc/cur_month"]             = diag[:cur_month]
        f["fc/cur_year"]              = diag[:cur_year]
        f["fc/node_band_confirm"]     = diag[:node_band_confirm]
        f["fc/node_relax_ns_confirm"] = diag[:node_relax_ns_confirm]
        f["fc/node_epoch_confirm"]    = diag[:node_epoch_confirm]
        f["fc/node_confirm_order"]    = diag[:node_confirm_order]
        f["fc/total_confirmed"]       = diag[:total_confirmed]
        f["fc/total_loop_time_ns"]    = diag[:total_loop_time_ns]

        f["fc/rmax"]                = fc.rmax
        f["fc/gfs_dir"]             = fc.gfs_dir
        f["fc/gfs_init_year"]       = fc.gfs_init_year
        f["fc/gfs_init_month"]      = fc.gfs_init_month
        f["fc/gfs_init_day"]        = fc.gfs_init_day
        f["fc/gfs_cycle_hour"]      = fc.gfs_cycle_hour
        f["fc/gfs_available_hours"] = fc.gfs_available_hours

        # --- actual (real-wind replay) half ---
        f["actual/steps"] = actual.steps
        f["actual/ct"]     = actual.wind.ct
        f["actual/cd"]     = actual.wind.cd
        f["actual/cm"]     = actual.wind.cm
        f["actual/cy"]     = actual.wind.cy
    end
    println("Saved committed run (forecast + real-wind replay) to $path " *
            "(wind pools/nb_flat rebuilt on load, not stored)")
end

"""
    load_committed_run(path; rebuild_wind = true, wind_cache_size = 24) -> committed

Reloads a save_committed_run(...) file WITHOUT re-solving or re-replaying.
Set rebuild_wind = false to skip rebuilding both WindCaches (e.g. if you
only need path/costs/energy for a non-visual comparison) -- in that case
fc.stepper and actual will both come back with stepper = nothing / a
RouteStepper2 whose .wind field is unusable for zoom-pane lookups.
Requires the original GFS files (for fc) and the ERA5 archive under
./OceanData (for actual) still on disk if rebuild_wind = true.
"""
function load_committed_run(path::String; rebuild_wind::Bool = true, wind_cache_size::Int = 24)
    data = jldopen(path, "r") do f
        (
            fc_path = f["fc/path"], fc_costs = f["fc/costs"], fc_energy = f["fc/energy"],
            fc_power = f["fc/power"], fc_grid_points = f["fc/grid_points"],
            fc_ocean_mask = f["fc/ocean_mask"], fc_wall_s = f["fc/wall_s"],
            fc_path_idxs = f["fc/path_idxs"],
            fc_cur_time = f["fc/cur_time"], fc_cur_day = f["fc/cur_day"],
            fc_cur_month = f["fc/cur_month"], fc_cur_year = f["fc/cur_year"],
            fc_node_band_confirm = f["fc/node_band_confirm"],
            fc_node_relax_ns_confirm = f["fc/node_relax_ns_confirm"],
            fc_node_epoch_confirm = f["fc/node_epoch_confirm"],
            fc_node_confirm_order = f["fc/node_confirm_order"],
            fc_total_confirmed = f["fc/total_confirmed"],
            fc_total_loop_time_ns = f["fc/total_loop_time_ns"],
            fc_rmax = f["fc/rmax"], fc_gfs_dir = f["fc/gfs_dir"],
            fc_gfs_init_year = f["fc/gfs_init_year"], fc_gfs_init_month = f["fc/gfs_init_month"],
            fc_gfs_init_day = f["fc/gfs_init_day"], fc_gfs_cycle_hour = f["fc/gfs_cycle_hour"],
            fc_gfs_available_hours = f["fc/gfs_available_hours"],
            actual_steps = f["actual/steps"],
            actual_ct = f["actual/ct"], actual_cd = f["actual/cd"],
            actual_cm = f["actual/cm"], actual_cy = f["actual/cy"]
        )
    end

    n_rows, n_cols = size(data.fc_grid_points)
    nb_flat = build_neighbour_cache(data.fc_grid_points, data.fc_ocean_mask, n_rows, n_cols,
                                     data.fc_rmax, NEIGHBOUR_TARGET_SPACING_M)

    fc_diag = Dict{Symbol,Any}(
        :path_idxs => data.fc_path_idxs, :grid_points => data.fc_grid_points,
        :ocean_mask => data.fc_ocean_mask, :cur_time => data.fc_cur_time, :cur_day => data.fc_cur_day,
        :cur_month => data.fc_cur_month, :cur_year => data.fc_cur_year, :nb_flat => nb_flat,
        :node_band_confirm => data.fc_node_band_confirm,
        :node_relax_ns_confirm => data.fc_node_relax_ns_confirm,
        :node_epoch_confirm => data.fc_node_epoch_confirm,
        :node_confirm_order => data.fc_node_confirm_order,
        :total_confirmed => data.fc_total_confirmed,
        :total_loop_time_ns => data.fc_total_loop_time_ns
    )

    fc_stepper = nothing
    actual_stepper = nothing
    if rebuild_wind
        # forecast half: rebuild against the original GFS source
        set_gfs_forecast_source!(; dir = data.fc_gfs_dir, init_year = data.fc_gfs_init_year,
                                   init_month = data.fc_gfs_init_month, init_day = data.fc_gfs_init_day,
                                   cycle_hour = data.fc_gfs_cycle_hour,
                                   available_hours = data.fc_gfs_available_hours)
        fc_wind_pool = WindFieldPool(; max_cached = length(data.fc_gfs_available_hours) + 4)
        fc_init_ep = _ensureEpochRegistered!(fc_wind_pool, data.fc_cur_time, data.fc_cur_day,
                                              data.fc_cur_month, data.fc_cur_year, 0)
        fc_init_field = _getField!(fc_wind_pool, fc_init_ep)
        for hh in data.fc_gfs_available_hours
            hh == 0 && continue   # already fetched as fc_init_ep
            ep = _ensureEpochRegistered!(fc_wind_pool, data.fc_cur_time, data.fc_cur_day,
                                          data.fc_cur_month, data.fc_cur_year, hh * 3600)
            _getField!(fc_wind_pool, ep)   # force fetch now, while GFS source is still set
        end
        fc_interp = build_interpolator(data.fc_grid_points, fc_init_field)
        clear_gfs_forecast_source!()

        fc_diag[:wind_pool] = fc_wind_pool
        fc_diag[:interp]    = fc_interp
        fc_stepper = build_stepper2(data.fc_path, data.fc_costs, data.fc_energy, data.fc_power,
                                     fc_diag, data.fc_grid_points, data.fc_ocean_mask)

        # actual half: rebuild against the CLIMATOLOGY archive (6-hourly
        # batched), not the per-timestamp ERA5 default -- this must match
        # whatever source was active when the run was originally solved/
        # replayed, or a reload silently falls back to per-day CDS downloads.
        _GFS_FORECAST_SOURCE[] === nothing ||
            error("load_committed_run: expected clear_gfs_forecast_source!() state for the " *
                  "'actual' (real-wind) half -- internal ordering bug.")
        set_climatology_archive_source!()
        actual_wind_pool = WindFieldPool(; max_cached = wind_cache_size)
        actual_init_ep = _ensureEpochRegistered!(actual_wind_pool, data.actual_ct, data.actual_cd,
                                                  data.actual_cm, data.actual_cy, 0)
        actual_init_field = _getField!(actual_wind_pool, actual_init_ep)
        actual_interp = build_interpolator(data.fc_grid_points, actual_init_field)
        actual_wind = WindCache(actual_wind_pool, actual_interp, data.actual_ct, data.actual_cd,
                                 data.actual_cm, data.actual_cy)
        clear_climatology_archive_source!()
        actual_stepper = RouteStepper2(data.actual_steps, data.fc_grid_points, data.fc_ocean_mask,
                                        nb_flat, actual_wind, length(data.actual_steps))
    end

    println("Loaded committed run from $path" *
             (rebuild_wind ? "" : " (steppers not rebuilt -- rebuild_wind=false)"))

    fc = (stepper = fc_stepper, diag = fc_diag, path = data.fc_path, costs = data.fc_costs,
          energy = data.fc_energy, power = data.fc_power, grid_points = data.fc_grid_points,
          ocean_mask = data.fc_ocean_mask, wall_s = data.fc_wall_s,
          rmax = data.fc_rmax, gfs_dir = data.fc_gfs_dir,
          gfs_init_year = data.fc_gfs_init_year, gfs_init_month = data.fc_gfs_init_month,
          gfs_init_day = data.fc_gfs_init_day, gfs_cycle_hour = data.fc_gfs_cycle_hour,
          gfs_available_hours = data.fc_gfs_available_hours)

    return (forecast = fc, actual = actual_stepper)
end

# -----------------------------------------------------------------------
# Save / reload run_hindsight_route's result.
#
# Structurally the same as run_forecast_route's output (path, costs,
# energy, power, diag) minus any forecast-source bookkeeping, since
# hindsight is solved directly against the default ERA5 loader. Rebuild
# on load follows the same "no GFS source" branch replay_route_real_wind
# and run_hindsight_route itself require.
# -----------------------------------------------------------------------

"""
    save_hindsight_route(path, hindsight)

Persists a run_hindsight_route(...) result so it can be reloaded without
re-solving.
"""
function save_hindsight_route(path::String, hindsight::NamedTuple)
    diag = hindsight.diag
    jldopen(path, "w") do f
        f["path"]        = hindsight.path
        f["costs"]       = hindsight.costs
        f["energy"]      = hindsight.energy
        f["power"]       = hindsight.power
        f["grid_points"] = hindsight.grid_points
        f["ocean_mask"]  = hindsight.ocean_mask
        f["start_idx"]   = hindsight.start_idx
        f["goal_idx"]    = hindsight.goal_idx

        f["path_idxs"]             = diag[:path_idxs]
        f["cur_time"]              = diag[:cur_time]
        f["cur_day"]               = diag[:cur_day]
        f["cur_month"]             = diag[:cur_month]
        f["cur_year"]              = diag[:cur_year]
        f["node_band_confirm"]     = diag[:node_band_confirm]
        f["node_relax_ns_confirm"] = diag[:node_relax_ns_confirm]
        f["node_epoch_confirm"]    = diag[:node_epoch_confirm]
        f["node_confirm_order"]    = diag[:node_confirm_order]
        f["total_confirmed"]       = diag[:total_confirmed]
        f["total_loop_time_ns"]    = diag[:total_loop_time_ns]

        # Rmax isn't returned by run_hindsight_route directly, but nb_flat
        # is deterministic from grid_points/ocean_mask/Rmax -- the caller
        # must have used SOME Rmax, default to the function's own default
        # (4) unless they tell us otherwise via the rmax kwarg below.
    end
    println("Saved hindsight route to $path (wind pool/nb_flat rebuilt on load, not stored)")
end

"""
    load_hindsight_route(path; rebuild_wind = true, rmax = 4, wind_cache_size = 24) -> hindsight

Reloads a save_hindsight_route(...) file WITHOUT re-solving. `rmax` must
match whatever Rmax was actually passed to the original run_hindsight_route
call (default 4, matching that function's own default) -- it's used only
to rebuild nb_flat, which isn't itself stored. Requires
clear_gfs_forecast_source!() state and the ERA5 archive under
./OceanData still on disk if rebuild_wind = true.
"""
function load_hindsight_route(path::String; rebuild_wind::Bool = true, rmax::Int = 4,
                               wind_cache_size::Int = 24)
    data = jldopen(path, "r") do f
        (
            path = f["path"], costs = f["costs"], energy = f["energy"], power = f["power"],
            grid_points = f["grid_points"], ocean_mask = f["ocean_mask"],
            start_idx = f["start_idx"], goal_idx = f["goal_idx"],
            path_idxs = f["path_idxs"],
            cur_time = f["cur_time"], cur_day = f["cur_day"],
            cur_month = f["cur_month"], cur_year = f["cur_year"],
            node_band_confirm = f["node_band_confirm"],
            node_relax_ns_confirm = f["node_relax_ns_confirm"],
            node_epoch_confirm = f["node_epoch_confirm"],
            node_confirm_order = f["node_confirm_order"],
            total_confirmed = f["total_confirmed"],
            total_loop_time_ns = f["total_loop_time_ns"]
        )
    end

    n_rows, n_cols = size(data.grid_points)
    nb_flat = build_neighbour_cache(data.grid_points, data.ocean_mask, n_rows, n_cols,
                                     rmax, NEIGHBOUR_TARGET_SPACING_M)

    diag = Dict{Symbol,Any}(
        :path_idxs => data.path_idxs, :grid_points => data.grid_points,
        :ocean_mask => data.ocean_mask, :cur_time => data.cur_time, :cur_day => data.cur_day,
        :cur_month => data.cur_month, :cur_year => data.cur_year, :nb_flat => nb_flat,
        :node_band_confirm => data.node_band_confirm,
        :node_relax_ns_confirm => data.node_relax_ns_confirm,
        :node_epoch_confirm => data.node_epoch_confirm,
        :node_confirm_order => data.node_confirm_order,
        :total_confirmed => data.total_confirmed,
        :total_loop_time_ns => data.total_loop_time_ns
    )

    stepper = nothing
    set_climatology_archive_source!()
    if rebuild_wind
        _GFS_FORECAST_SOURCE[] === nothing ||
            error("load_hindsight_route: call clear_gfs_forecast_source!() first -- hindsight " *
                  "must be rebuilt against the default ERA5 loader, not a GFS forecast source.")
        wind_pool  = WindFieldPool(; max_cached = wind_cache_size)
        init_ep    = _ensureEpochRegistered!(wind_pool, data.cur_time, data.cur_day,
                                              data.cur_month, data.cur_year, 0)
        init_field = _getField!(wind_pool, init_ep)
        interp     = build_interpolator(data.grid_points, init_field)

        diag[:wind_pool] = wind_pool
        diag[:interp]    = interp
        stepper = build_stepper2(data.path, data.costs, data.energy, data.power, diag,
                                  data.grid_points, data.ocean_mask)
    end
    clear_climatology_archive_source!()

    println("Loaded hindsight route from $path" *
             (rebuild_wind ? "" : " (stepper not rebuilt -- rebuild_wind=false)"))

    return (stepper = stepper, diag = diag, path = data.path, costs = data.costs,
            energy = data.energy, power = data.power, grid_points = data.grid_points,
            ocean_mask = data.ocean_mask, start_idx = data.start_idx, goal_idx = data.goal_idx)
end

# -----------------------------------------------------------------------
# Lazily rebuilds a forecast WindCache for one rolling-replan day, from
# the original GFS files. Registers just enough hours to cover that
# day's real elapsed duration (landed_time_s), same reasoning as
# run_forecast_route's wind_cache_size. Cached per day_index so repeated
# scrubbing within the same day doesn't re-touch disk.
# -----------------------------------------------------------------------

function _gfs_cycle_for(gfs_dirs_list::Vector{String}, d::Date)
    cands = Tuple{Date,String}[]
    for dir in gfs_dirs_list
        m = match(r"(\d{8})_\d{2}z$", dir)
        m === nothing && continue
        push!(cands, (Date(m.captures[1], dateformat"yyyymmdd"), dir))
    end
    isempty(cands) && error("no parsable GFS dirs")
    sort!(cands; by = first)
    k = findlast(c -> c[1] <= d, cands)
    k === nothing && error("no GFS cycle on or before $d")
    return cands[k]            # (init_date, dir)
end

function _get_rolling_forecast_wind_cache!(
    cache::Dict{Int,WindCache}, gfs_dirs_list::Vector{String}, available_hours::Vector{Int},
    hhmm::Int, log::RollingDayLog; wind_cache_size::Int = 30
)
    haskey(cache, log.day_index) && return cache[log.day_index]

    init_dt, gfs_dir = _gfs_cycle_for(gfs_dirs_list, log.date)

    w = log.actual_stepper.wind        # this iteration's own start clock
    set_gfs_forecast_source!(; dir = gfs_dir, init_year = year(init_dt),
                               init_month = month(init_dt), init_day = day(init_dt),
                               cycle_hour = log.cycle_hour, available_hours = available_hours)

    wind_pool = WindFieldPool(; max_cached = wind_cache_size)
    max_hours_needed = ceil(Int, log.landed_time_s / 3600) + 3
    init_ep = _ensureEpochRegistered!(wind_pool, w.ct, w.cd, w.cm, w.cy, 0)
    init_field = _getField!(wind_pool, init_ep)
    for hh in 1:max_hours_needed
        ep = _ensureEpochRegistered!(wind_pool, w.ct, w.cd, w.cm, w.cy, hh * 3600)
        _getField!(wind_pool, ep)
    end
    interp = build_interpolator(log.actual_stepper.grid_points, init_field)
    clear_gfs_forecast_source!()

    wc = WindCache(wind_pool, interp, w.ct, w.cd, w.cm, w.cy)
    cache[log.day_index] = wc
    return wc
end

function _rolling_forecast_wind_at(
    result::NamedTuple, day_offsets::Vector{Float64}, fc_cache::Dict{Int,WindCache},
    gfs_dirs_list::Vector{String}, available_hours::Vector{Int}, hhmm::Int, t_s::Float64
)
    day_logs = result.day_logs
    k = clamp(searchsortedlast(day_offsets, t_s), 1, length(day_logs))
    log = day_logs[k]
    local_t = clamp(t_s - day_offsets[k], 0.0, log.landed_time_s)
    wc = _get_rolling_forecast_wind_cache!(fc_cache, gfs_dirs_list, available_hours, hhmm, log)
    U, V = get_wind(wc, local_t)
    return U, V
end

@inline function _angular_diff(a::Float64, b::Float64)
    d = mod(a - b + 180.0, 360.0) - 180.0
    return abs(d)
end

@inline function _wind_vector_error(u1::Float64, v1::Float64, u2::Float64, v2::Float64)
    return sqrt((u1 - u2)^2 + (v1 - v2)^2)
end

# -----------------------------------------------------------------------
# Per-day wind lookup for the STITCHED rolling stepper.
#
# rolling_stepper.wind only ever holds day 1's WindCache (see
# build_rolling_stepper's docstring/comment) -- using it directly for a
# time on day k>1 would silently return day 1's wind mislabeled as day k's.
# This instead finds which day_log a given GLOBAL elapsed time falls into
# and queries THAT day's own actual_stepper.wind with the LOCAL (day-
# relative) elapsed time, exactly matching how replay_route_real_wind
# built each day's WindCache in the first place.
# -----------------------------------------------------------------------
function _rolling_day_offsets(result::NamedTuple)
    offsets = Float64[]
    off = 0.0
    for log in result.day_logs
        push!(offsets, off)
        off += log.landed_time_s
    end
    return offsets   # offsets[k] = global elapsed seconds at the START of day k
end

function _rolling_wind_at(result::NamedTuple, day_offsets::Vector{Float64}, t_s::Float64)
    day_logs = result.day_logs
    k = clamp(searchsortedlast(day_offsets, t_s), 1, length(day_logs))
    local_t = clamp(t_s - day_offsets[k], 0.0, day_logs[k].landed_time_s)
    U, V = get_wind(day_logs[k].actual_stepper.wind, local_t)
    return U, V, k, local_t
end

# -----------------------------------------------------------------------
# Position + wind field for any of the three routes at a given GLOBAL
# elapsed time. Dispatches to the right WindCache (day-resolved for
# rolling, single-cache for the other two).
# -----------------------------------------------------------------------
function _route_wind_and_pos(
    kind::Symbol, t_s::Float64,
    rolling_stepper::RouteStepper2, committed_stepper::RouteStepper2,
    hindsight_stepper::Union{Nothing,RouteStepper2},
    result::NamedTuple, day_offsets::Vector{Float64},
    lats_1d::Vector{Float64}, lons_1d::Vector{Float64}
)
    if kind == :rolling
        s = state_at_time(rolling_stepper, t_s)
        U, V, _, _ = _rolling_wind_at(result, day_offsets, t_s)
    elseif kind == :committed
        s = state_at_time(committed_stepper, t_s)
        U, V = get_wind(committed_stepper.wind, t_s)
    elseif kind == :hindsight
        hindsight_stepper === nothing && error("_route_wind_and_pos: no hindsight_stepper supplied")
        s = state_at_time(hindsight_stepper, t_s)
        U, V = get_wind(hindsight_stepper.wind, t_s)
    else
        error("_route_wind_and_pos: unknown kind $kind")
    end
    i = nearest_index(lats_1d, s.lat)
    j = nearest_index(lons_1d, s.lon)
    return (state = s, U = U, V = V, i = i, j = j)
end

@inline function _wind_speed_dir(U::Matrix{Float32}, V::Matrix{Float32}, i::Int, j::Int)
    u, v = U[i, j], V[i, j]
    ws   = sqrt(u^2 + v^2)
    wdir = mod(rad2deg(atan(u, v)) + 180f0, 360f0)
    return Float64(ws), Float64(wdir)
end

# -----------------------------------------------------------------------
# Small local wind heatmap + quiver, centered on (i,j). Same visual
# language as draw_zoom! in route_simulator.jl, just parameterized on an
# already-fetched U,V pair so callers control which route/time it's for.
# -----------------------------------------------------------------------
function _draw_route_wind_zoom!(ax, U::Matrix{Float32}, V::Matrix{Float32}, i::Int, j::Int,
                                 grid_points::Matrix{Tuple{Float64,Float64}}, ocean_mask::Matrix{Bool},
                                 wind_range; zoom_cells::Int = 12, title::String = "")
    GLMakie.empty!(ax)
    n_rows, n_cols = size(ocean_mask)
    i_lo, i_hi = max(1, i - zoom_cells), min(n_rows, i + zoom_cells)
    j_lo, j_hi = max(1, j - zoom_cells), min(n_cols, j + zoom_cells)

    lons_sub = [grid_points[i, jj][1] for jj in j_lo:j_hi]
    lats_sub = [grid_points[ii, j][2] for ii in i_lo:i_hi]
    Usub = U[i_lo:i_hi, j_lo:j_hi]; Vsub = V[i_lo:i_hi, j_lo:j_hi]
    speed_sub = sqrt.(Usub .^ 2 .+ Vsub .^ 2)

    GLMakie.heatmap!(ax, lons_sub, lats_sub, permutedims(speed_sub);
                      colormap = :viridis, colorrange = wind_range[])

    step_q = max(1, zoom_cells ÷ 5)
    qlons = Float64[]; qlats = Float64[]; qu = Float64[]; qv = Float64[]
    for ii in i_lo:step_q:i_hi, jj in j_lo:step_q:j_hi
        lo, la = grid_points[ii, jj]
        push!(qlons, lo); push!(qlats, la)
        push!(qu, U[ii, jj]); push!(qv, V[ii, jj])
    end
    if !isempty(qlons)
        GLMakie.arrows!(ax, qlons, qlats, qu .* 0.03, qv .* 0.03;
                         color = :white, linewidth = 1.0, arrowsize = 6)
    end

    lon0, lat0 = grid_points[i, j]
    GLMakie.scatter!(ax, [lon0], [lat0]; color = :magenta, markersize = 14, marker = :star5)
    ax.title = title
end

# -----------------------------------------------------------------------
# Wind-exposure divergence: Total Variation Distance between the
# histograms of wind speed each route has actually sailed through, up to
# a given point in the voyage. TVD(P,Q) = 0.5 * sum(|P_i - Q_i|), 0 =
# identical exposure so far, 1 = fully disjoint. This deliberately
# compares EXPERIENCED wind (each boat's own position/time), not wind at
# a shared point -- that's the meaningful comparison for "did routing
# strategy X sail through different weather than strategy Y."
# -----------------------------------------------------------------------
function _tvd(p::Vector{Float64}, q::Vector{Float64})
    return 0.5 * sum(abs.(p .- q))
end

# -----------------------------------------------------------------------
# 2D (speed × direction) wind-exposure histogram. Direction matters now:
# two routes both seeing 15 m/s only count as "similar exposure" if
# they're seeing it from similar angles, not just the same magnitude.
# Direction bins wrap around (0/360 are the same heading) -- edges run
# 0:dir_bin_size:360 and a sample of exactly 360 is folded into the last
# bin via mod, same convention used elsewhere in this codebase
# (mod(..., 360f0)).
# -----------------------------------------------------------------------
function _histogram_probs_2d(
    ws_samples::AbstractVector{Float64}, dir_samples::AbstractVector{Float64},
    ws_edges::Vector{Float64}, dir_edges::Vector{Float64}
)
    n_ws  = length(ws_edges) - 1
    n_dir = length(dir_edges) - 1
    counts = zeros(Float64, n_ws, n_dir)
    @inbounds for k in eachindex(ws_samples)
        ws = ws_samples[k]; wd = dir_samples[k]
        (isnan(ws) || isnan(wd)) && continue
        wd_wrapped = mod(wd, 360.0)
        wi = clamp(searchsortedlast(ws_edges, ws), 1, n_ws)
        di = clamp(searchsortedlast(dir_edges, wd_wrapped), 1, n_dir)
        counts[wi, di] += 1.0
    end
    s = sum(counts)
    return s > 0 ? vec(counts ./ s) : vec(counts)   # flatten to 1D for TVD
end



function launch_rolling_comparison_explorer(
    result::NamedTuple, committed_stepper::RouteStepper2, committed_forecast_stepper::RouteStepper2;
    hindsight_stepper::Union{Nothing,RouteStepper2} = nothing,
    n_frames::Int = 150, ws_bin_size::Float64 = 2.0, ws_max::Float64 = 30.0,
    dir_bin_size::Float64 = 30.0,
    rolling_gfs_dirs_list::Union{Nothing,Vector{String}} = nothing,
    rolling_start_hhmm::Int = 0,
    rolling_available_hours::Vector{Int} = vcat(collect(0:120), collect(123:3:336)),
    committed_forecast_max_hours::Float64 = 336.0
)
    rolling_stepper = build_rolling_stepper(result)
    has_hindsight = hindsight_stepper !== nothing
    has_rolling_forecast = rolling_gfs_dirs_list !== nothing
    day_offsets = _rolling_day_offsets(result)
    rolling_fc_cache = Dict{Int,WindCache}()

    t_end = max(rolling_stepper.steps[end].cum_time_s, committed_stepper.steps[end].cum_time_s,
                has_hindsight ? hindsight_stepper.steps[end].cum_time_s : 0.0)

    grid_points = result.grid_points
    ocean_mask  = result.ocean_mask
    n_rows, n_cols = size(ocean_mask)
    lons_1d = [grid_points[1, j][1] for j in 1:n_cols]
    lats_1d = [grid_points[i, 1][2] for i in 1:n_rows]
    committed_fc_max_s = committed_forecast_max_hours * 3600.0

    # ------------------------------------------------------------
    # Precompute: forecast-vs-actual wind series for committed and
    # rolling (needed for the one TVD panel we keep). Cross-route
    # actual-vs-actual exposure series/TVD removed -- not rendered.
    # ------------------------------------------------------------
    rolling_end_s   = rolling_stepper.steps[end].cum_time_s
    committed_end_s = committed_stepper.steps[end].cum_time_s
    ts_grid = collect(range(0.0, t_end, length = n_frames))
    println("Precomputing forecast-vs-actual wind series ($n_frames samples)...")

    committed_ws = Vector{Float64}(undef, n_frames); committed_wd = Vector{Float64}(undef, n_frames)
    committed_u  = Vector{Float64}(undef, n_frames); committed_v  = Vector{Float64}(undef, n_frames)  # NEW
    committed_fc_ws = fill(NaN, n_frames); committed_fc_wd = fill(NaN, n_frames)
    committed_fc_u  = fill(NaN, n_frames); committed_fc_v  = fill(NaN, n_frames)  # NEW
    committed_fc_available = falses(n_frames)

    rolling_ws = Vector{Float64}(undef, n_frames); rolling_wd = Vector{Float64}(undef, n_frames)
    rolling_u  = Vector{Float64}(undef, n_frames); rolling_v  = Vector{Float64}(undef, n_frames)  # NEW
    rolling_fc_ws = fill(NaN, n_frames); rolling_fc_wd = fill(NaN, n_frames)
    rolling_fc_u  = fill(NaN, n_frames); rolling_fc_v  = fill(NaN, n_frames)  # NEW
    rolling_fc_available = falses(n_frames)

    for (idx, t) in enumerate(ts_grid)
        wr = _route_wind_and_pos(:rolling, t, rolling_stepper, committed_stepper, hindsight_stepper,
                                result, day_offsets, lats_1d, lons_1d)
        rolling_ws[idx], rolling_wd[idx] = _wind_speed_dir(wr.U, wr.V, wr.i, wr.j)
        rolling_u[idx], rolling_v[idx] = Float64(wr.U[wr.i, wr.j]), Float64(wr.V[wr.i, wr.j])   # NEW

        wc = _route_wind_and_pos(:committed, t, rolling_stepper, committed_stepper, hindsight_stepper,
                                result, day_offsets, lats_1d, lons_1d)
        committed_ws[idx], committed_wd[idx] = _wind_speed_dir(wc.U, wc.V, wc.i, wc.j)
        committed_u[idx], committed_v[idx] = Float64(wc.U[wc.i, wc.j]), Float64(wc.V[wc.i, wc.j])  # NEW

        if t <= committed_fc_max_s && t <= committed_end_s
            Ufc, Vfc = get_wind(committed_forecast_stepper.wind, t)
            committed_fc_ws[idx], committed_fc_wd[idx] = _wind_speed_dir(Ufc, Vfc, wc.i, wc.j)
            committed_fc_u[idx], committed_fc_v[idx] = Float64(Ufc[wc.i, wc.j]), Float64(Vfc[wc.i, wc.j])  # NEW
            committed_fc_available[idx] = true
        end

        if has_rolling_forecast && t <= rolling_end_s
            Ufc, Vfc = _rolling_forecast_wind_at(result, day_offsets, rolling_fc_cache,
                                                rolling_gfs_dirs_list, rolling_available_hours,
                                                rolling_start_hhmm, t)
            rolling_fc_ws[idx], rolling_fc_wd[idx] = _wind_speed_dir(Ufc, Vfc, wr.i, wr.j)
            rolling_fc_u[idx], rolling_fc_v[idx] = Float64(Ufc[wr.i, wr.j]), Float64(Vfc[wr.i, wr.j])  # NEW
            rolling_fc_available[idx] = true
        end
    end
    println("  done")

    ws_edges  = collect(0.0:ws_bin_size:ws_max)
    dir_edges = collect(0.0:dir_bin_size:360.0)

    # Plain wind error, both components:
    #   speed error  = forecast_ws - actual_ws     (signed, m/s)
    #   dir error    = angular_diff(forecast_wd, actual_wd)   (unsigned, deg -
    #                  "how far off was the heading", direction has no
    #                  natural sign the way speed does)
    # Direction is undefined near calm wind (an m/s-scale vector has no
    # stable heading) -- forecast-verification practice masks direction
    # error below a small speed floor rather than let calm-wind noise
    # dominate the direction-error series.
    MIN_WS_FOR_DIR_ERROR = 5.0

    committed_spd_err = fill(NaN, n_frames)
    rolling_spd_err   = fill(NaN, n_frames)
    committed_dir_err = fill(NaN, n_frames)
    rolling_dir_err   = fill(NaN, n_frames)
    committed_vec_err = fill(NaN, n_frames)  # NEW
    rolling_vec_err   = fill(NaN, n_frames)  # NEW
    committed_cum_spd_mae = fill(NaN, n_frames)
    rolling_cum_spd_mae   = fill(NaN, n_frames)
    committed_cum_dir_mae = fill(NaN, n_frames)
    rolling_cum_dir_mae   = fill(NaN, n_frames)
    committed_cum_vec_mae = fill(NaN, n_frames)  # NEW
    rolling_cum_vec_mae   = fill(NaN, n_frames)  # NEW

    for i in 1:n_frames
        if committed_fc_available[i]
            committed_spd_err[i] = committed_fc_ws[i] - committed_ws[i]
            committed_vec_err[i] = _wind_vector_error(committed_fc_u[i], committed_fc_v[i],
                                                        committed_u[i], committed_v[i])   # NEW
            if committed_ws[i] >= MIN_WS_FOR_DIR_ERROR && committed_fc_ws[i] >= MIN_WS_FOR_DIR_ERROR
                committed_dir_err[i] = _angular_diff(committed_fc_wd[i], committed_wd[i])
            end
        end
        if has_rolling_forecast && rolling_fc_available[i]
            rolling_spd_err[i] = rolling_fc_ws[i] - rolling_ws[i]
            rolling_vec_err[i] = _wind_vector_error(rolling_fc_u[i], rolling_fc_v[i],
                                                    rolling_u[i], rolling_v[i])        # NEW
            if rolling_ws[i] >= MIN_WS_FOR_DIR_ERROR && rolling_fc_ws[i] >= MIN_WS_FOR_DIR_ERROR
                rolling_dir_err[i] = _angular_diff(rolling_fc_wd[i], rolling_wd[i])
            end
        end
    end

    function _running_cumsum!(cum::Vector{Float64}, err::Vector{Float64})
        running_sum = 0.0
        for i in eachindex(err)
            !isnan(err[i]) && (running_sum += abs(err[i]))
            cum[i] = running_sum
        end
    end
    _running_cumsum!(committed_cum_spd_mae, committed_spd_err)
    _running_cumsum!(committed_cum_dir_mae, committed_dir_err)
    _running_cumsum!(committed_cum_vec_mae, committed_vec_err)   # NEW
    if has_rolling_forecast
        _running_cumsum!(rolling_cum_spd_mae, rolling_spd_err)
        _running_cumsum!(rolling_cum_dir_mae, rolling_dir_err)
        _running_cumsum!(rolling_cum_vec_mae, rolling_vec_err)    # NEW
    end

    # ------------------------------------------------------------
    # Figure layout:
    #   row 0: title
    #   row 1: MAP (large, left) | two wind panes stacked (smaller, right)
    #   row 2: stats (full width)
    #   row 3: forecast-vs-actual TVD (full width)
    #   row 4: transport controls (full width)
    # ------------------------------------------------------------
    fig = GLMakie.Figure(size = (1900, 1500))
    GLMakie.Label(fig[0, 1], "Rolling Replan vs Committed vs Hindsight — Explorer";
                  fontsize = 22, font = :bold, halign = :left, tellwidth = false)

    top_row = fig[1, 1] = GLMakie.GridLayout()

    ax_map = GLMakie.Axis(top_row[1, 1]; xlabel = "lon", ylabel = "lat")
    mask_z = permutedims(Float32.(ocean_mask))
    GLMakie.heatmap!(ax_map, lons_1d, lats_1d, mask_z;
                      colormap = [RGB(0.82, 0.75, 0.55), RGB(0.75, 0.87, 0.95)],
                      colorrange = (0, 1), interpolate = false)

    r_lons = [s.lon for s in rolling_stepper.steps];   r_lats = [s.lat for s in rolling_stepper.steps]
    c_lons = [s.lon for s in committed_stepper.steps]; c_lats = [s.lat for s in committed_stepper.steps]
    GLMakie.lines!(ax_map, r_lons, r_lats; color = :steelblue, linewidth = 2.5, label = "rolling replan")
    GLMakie.lines!(ax_map, c_lons, c_lats; color = :firebrick, linewidth = 2.5, linestyle = :dash,
                   label = "single-forecast-committed")
    if has_hindsight
        h_lons = [s.lon for s in hindsight_stepper.steps]; h_lats = [s.lat for s in hindsight_stepper.steps]
        GLMakie.lines!(ax_map, h_lons, h_lats; color = :black, linewidth = 1.5, linestyle = :dot,
                       label = "hindsight (perfect foreknowledge)")
    end
    for day_log in result.day_logs
        GLMakie.scatter!(ax_map, [day_log.landed_lon], [day_log.landed_lat];
                          color = :orange, markersize = 8, marker = :circle)
    end
    GLMakie.scatter!(ax_map, Float64[], Float64[]; color = :orange, markersize = 8,
                      marker = :circle, label = "daily replan point")
    GLMakie.scatter!(ax_map, [r_lons[1]], [r_lats[1]]; color = :lime, markersize = 14,
                      marker = :diamond, label = "start")
    GLMakie.scatter!(ax_map, [r_lons[end]], [r_lats[end]]; color = :red, markersize = 14,
                      marker = :diamond, label = "goal")

    # markers made large + white-haloed so they're unambiguous when the
    # scrub bar moves them
    rolling_marker   = GLMakie.Observable(GLMakie.Point2f(r_lons[1], r_lats[1]))
    committed_marker = GLMakie.Observable(GLMakie.Point2f(c_lons[1], c_lats[1]))
    GLMakie.scatter!(ax_map, rolling_marker;   color = :steelblue, markersize = 26, marker = :star5,
                      strokewidth = 2, strokecolor = :white)
    GLMakie.scatter!(ax_map, committed_marker; color = :firebrick, markersize = 26, marker = :star5,
                      strokewidth = 2, strokecolor = :white)
    hindsight_marker = if has_hindsight
        h0 = GLMakie.Observable(GLMakie.Point2f(hindsight_stepper.steps[1].lon, hindsight_stepper.steps[1].lat))
        GLMakie.scatter!(ax_map, h0; color = :black, markersize = 22, marker = :star5,
                          strokewidth = 2, strokecolor = :white)
        h0
    else
        nothing
    end
    GLMakie.axislegend(ax_map; position = :lb, labelsize = 9, patchsize = (12, 8),
                        framevisible = true, backgroundcolor = (:white, 0.85),
                        margin = (4, 4, 4, 4))

    # -- two smaller wind panes stacked to the right of the map: rolling
    #    and committed's own actual-wind neighbourhood. (Hindsight's wind
    #    pane is dropped -- its track still shows on the map, but it's
    #    rarely the boat you're inspecting closely.) --
    wind_col = top_row[1, 2] = GLMakie.GridLayout()
    ax_wind_r = GLMakie.Axis(wind_col[1, 1]; xlabel = "lon", ylabel = "lat", title = "Rolling — actual wind")
    ax_wind_c = GLMakie.Axis(wind_col[2, 1]; xlabel = "lon", ylabel = "lat", title = "Committed — actual wind")
    wind_range = GLMakie.Observable((0.0, 15.0))
    GLMakie.Colorbar(wind_col[3, 1]; colorrange = wind_range, colormap = :viridis,
                      label = "wind speed (m/s)", vertical = false)

    # -- stats side by side, one column per route, instead of one tall
    #    stacked text block -- frees vertical room for the charts below --
    ncols_stats = has_hindsight ? 3 : 2
    stats_grid = fig[2, 1] = GLMakie.GridLayout()
    time_text = GLMakie.Observable("")
    GLMakie.Label(stats_grid[1, 1:ncols_stats], time_text; tellwidth = false, halign = :left,
                  fontsize = 13, font = :bold)
    rolling_text   = GLMakie.Observable("")
    committed_text = GLMakie.Observable("")
    GLMakie.Label(stats_grid[2, 1], rolling_text;   tellwidth = false, halign = :left,
                  valign = :top, fontsize = 12, justification = :left)
    GLMakie.Label(stats_grid[2, 2], committed_text; tellwidth = false, halign = :left,
                  valign = :top, fontsize = 12, justification = :left)
    hindsight_text = GLMakie.Observable("")
    has_hindsight && GLMakie.Label(stats_grid[2, 3], hindsight_text; tellwidth = false, halign = :left,
                  valign = :top, fontsize = 12, justification = :left)

    # -- 2x2 quad: rows = speed / direction, cols = instantaneous / cumulative.
    #    Separate single-unit panels rather than dual-axis overlays --
    #    matches standard forecast-verification practice (speed RMSE and
    #    direction RMSE are always reported as separate series, e.g.
    #    JMA DeepMedcast Fig. 11) and is far easier to read at a glance. --
    ts_h = ts_grid ./ 3600
    bottom_row = fig[3, 1] = GLMakie.GridLayout()

    # -- row 1: vector error (primary metric) --
    ax_vec_err = GLMakie.Axis(bottom_row[1, 1]; title = "Vector wind error (forecast − actual)",
                            ylabel = "error (m/s)")
    GLMakie.hlines!(ax_vec_err, [0.0]; color = :gray50, linestyle = :dash)
    GLMakie.lines!(ax_vec_err, ts_h, committed_vec_err; color = :firebrick, label = "committed")
    has_rolling_forecast && GLMakie.lines!(ax_vec_err, ts_h, rolling_vec_err; color = :steelblue, label = "rolling")
    GLMakie.axislegend(ax_vec_err; position = :rt, labelsize = 10)

    ax_vec_cum = GLMakie.Axis(bottom_row[1, 2]; title = "Cumulative total vector error", ylabel = "Σ|error| (m/s)")
    GLMakie.lines!(ax_vec_cum, ts_h, committed_cum_vec_mae; color = :firebrick)
    has_rolling_forecast && GLMakie.lines!(ax_vec_cum, ts_h, rolling_cum_vec_mae; color = :steelblue)

    # -- row 2: speed error (diagnostic) --
    ax_spd_err = GLMakie.Axis(bottom_row[2, 1]; title = "Speed error (forecast − actual)",
                        ylabel = "error (m/s)")
    GLMakie.hlines!(ax_spd_err, [0.0]; color = :gray50, linestyle = :dash)
    GLMakie.lines!(ax_spd_err, ts_h, committed_spd_err; color = :firebrick)
    has_rolling_forecast && GLMakie.lines!(ax_spd_err, ts_h, rolling_spd_err; color = :steelblue)

    ax_spd_cum = GLMakie.Axis(bottom_row[2, 2]; title = "Cumulative total speed error", ylabel = "Σ|error| (m/s)")
    GLMakie.lines!(ax_spd_cum, ts_h, committed_cum_spd_mae; color = :firebrick)
    has_rolling_forecast && GLMakie.lines!(ax_spd_cum, ts_h, rolling_cum_spd_mae; color = :steelblue)

    # -- row 3: direction error (diagnostic, masked below MIN_WS_FOR_DIR_ERROR) --
    ax_dir_err = GLMakie.Axis(bottom_row[3, 1]; title = "Direction error (forecast − actual, wind ≥ $(MIN_WS_FOR_DIR_ERROR) m/s)",
                            xlabel = "elapsed time (h)", ylabel = "error (°)")
    GLMakie.hlines!(ax_dir_err, [0.0]; color = :gray50, linestyle = :dash)
    GLMakie.lines!(ax_dir_err, ts_h, committed_dir_err; color = :firebrick)
    has_rolling_forecast && GLMakie.lines!(ax_dir_err, ts_h, rolling_dir_err; color = :steelblue)

    ax_dir_cum = GLMakie.Axis(bottom_row[3, 2]; title = "Cumulative total direction error", xlabel = "elapsed time (h)")
    GLMakie.lines!(ax_dir_cum, ts_h, committed_cum_dir_mae; color = :firebrick)
    has_rolling_forecast && GLMakie.lines!(ax_dir_cum, ts_h, rolling_cum_dir_mae; color = :steelblue)

    GLMakie.linkxaxes!(ax_vec_err, ax_vec_cum, ax_spd_err, ax_spd_cum)

    if committed_fc_max_s < t_end
        GLMakie.vlines!(ax_vec_err, [committed_fc_max_s / 3600]; color = :firebrick, linestyle = :dot,
                        label = "committed forecast horizon")
    end
    fva_marker_x = GLMakie.Observable(0.0)
    for ax in (ax_vec_err, ax_vec_cum, ax_spd_err, ax_spd_cum)
        GLMakie.vlines!(ax, fva_marker_x; color = :black, linestyle = :dash, linewidth = 1.2)
    end
    GLMakie.rowsize!(bottom_row, 1, GLMakie.Relative(1/3))
    GLMakie.rowsize!(bottom_row, 2, GLMakie.Relative(1/3))
    GLMakie.rowsize!(bottom_row, 3, GLMakie.Relative(1/3))
    GLMakie.colsize!(bottom_row, 1, GLMakie.Relative(0.5))
    GLMakie.colsize!(bottom_row, 2, GLMakie.Relative(0.5))

    # ------------------------------------------------------------
    # Update logic
    # ------------------------------------------------------------
    function fmt_route(name::String, s, ws::Float64, wdir::Float64)
        return "$(name)\n" *
               "$(round(s.dist_km, digits=0)) km  |  $(round(s.energy_kwh, digits=0)) kWh\n" *
               "$(round(s.speed_kn, digits=1)) kn  |  $(round(s.power_kw, digits=0)) kW\n" *
               "wind: $(round(ws, digits=1)) m/s @ $(round(wdir, digits=0))°"
    end

    function fmt_forecast_err(available::Bool, ws_a::Float64, wd_a::Float64,
                            ws_f::Float64, wd_f::Float64)
        available || return "forecast: unavailable (beyond forecast horizon)"
        spd_err = ws_f - ws_a
        calm = ws_a < MIN_WS_FOR_DIR_ERROR || ws_f < MIN_WS_FOR_DIR_ERROR
        dir_str = calm ? "calm conditions (dir n/a)" : "$(round(_angular_diff(wd_f, wd_a), digits=0))°"
        ws_str  = calm ? "$(round(ws_f, digits=1)) m/s (calm) → $(round(ws_a, digits=1)) m/s (calm)" :
                        "$(round(ws_f, digits=1)) m/s @ $(round(wd_f, digits=0))° → " *
                        "actual $(round(ws_a, digits=1)) m/s @ $(round(wd_a, digits=0))°"
        return "forecast: $ws_str\nerror: $(round(spd_err, digits=1)) m/s, $dir_str"
    end

    function update!(t_s::Float64)
        wr = _route_wind_and_pos(:rolling, t_s, rolling_stepper, committed_stepper, hindsight_stepper,
                                  result, day_offsets, lats_1d, lons_1d)
        wc = _route_wind_and_pos(:committed, t_s, rolling_stepper, committed_stepper, hindsight_stepper,
                                  result, day_offsets, lats_1d, lons_1d)
        rolling_marker[]   = GLMakie.Point2f(wr.state.lon, wr.state.lat)
        committed_marker[] = GLMakie.Point2f(wc.state.lon, wc.state.lat)

        ws_r, wd_r = _wind_speed_dir(wr.U, wr.V, wr.i, wr.j)
        ws_c, wd_c = _wind_speed_dir(wc.U, wc.V, wc.i, wc.j)

        max_ws = max(ws_r, ws_c, 1.0)
        ws_h = NaN; wd_h = NaN
        wh = nothing
        if has_hindsight
            wh = _route_wind_and_pos(:hindsight, t_s, rolling_stepper, committed_stepper, hindsight_stepper,
                                      result, day_offsets, lats_1d, lons_1d)
            hindsight_marker[] = GLMakie.Point2f(wh.state.lon, wh.state.lat)
            ws_h, wd_h = _wind_speed_dir(wh.U, wh.V, wh.i, wh.j)
            max_ws = max(max_ws, ws_h)
        end
        wind_range[] = (0.0, max_ws + 5.0)

        _draw_route_wind_zoom!(ax_wind_r, wr.U, wr.V, wr.i, wr.j, grid_points, ocean_mask, wind_range;
                                zoom_cells = 12, title = "Rolling — actual wind")
        GLMakie.scatter!(ax_wind_r, [wr.state.lon], [wr.state.lat]; color = :steelblue, markersize = 14, marker = :star5)

        if wc.state.clamped != :end
            _draw_route_wind_zoom!(ax_wind_c, wc.U, wc.V, wc.i, wc.j, grid_points, ocean_mask, wind_range;
                                    zoom_cells = 12, title = "Committed — actual wind")
            GLMakie.scatter!(ax_wind_c, [wc.state.lon], [wc.state.lat]; color = :firebrick, markersize = 14, marker = :star5)
        end
        GLMakie.scatter!(ax_wind_c, [wc.state.lon], [wc.state.lat]; color = :firebrick, markersize = 14, marker = :star5)

        time_text[] = "elapsed time: $(round(t_s / 3600, digits=1)) h  ($(round(t_s / 86400, digits=1)) days)"

        f_idx = clamp(searchsortedfirst(ts_grid, t_s), 1, n_frames)

        r_lines = String[fmt_route("Rolling", wr.state, ws_r, wd_r)]
        if has_rolling_forecast
            Ufc, Vfc = _rolling_forecast_wind_at(result, day_offsets, rolling_fc_cache,
                                                  rolling_gfs_dirs_list, rolling_available_hours,
                                                  rolling_start_hhmm, t_s)
            ws_rf, wd_rf = _wind_speed_dir(Ufc, Vfc, wr.i, wr.j)
            push!(r_lines, fmt_forecast_err(true, ws_r, wd_r, ws_rf, wd_rf))
            push!(r_lines, isnan(rolling_cum_vec_mae[f_idx]) ? "cum. error: n/a" :
                "cum. vector error: $(round(rolling_cum_vec_mae[f_idx], digits=1)) m/s  " *
                "(spd $(round(rolling_cum_spd_mae[f_idx], digits=1)) m/s, dir $(round(rolling_cum_dir_mae[f_idx], digits=0))°)")
        end
        rolling_text[] = join(r_lines, "\n")

        c_lines = String[fmt_route("Committed", wc.state, ws_c, wd_c)]
        if t_s <= committed_fc_max_s
            Ufc, Vfc = get_wind(committed_forecast_stepper.wind, t_s)
            ws_cf, wd_cf = _wind_speed_dir(Ufc, Vfc, wc.i, wc.j)
            push!(c_lines, fmt_forecast_err(true, ws_c, wd_c, ws_cf, wd_cf))
        else
            push!(c_lines, fmt_forecast_err(false, ws_c, wd_c, 0.0, 0.0))
        end
        push!(c_lines, isnan(committed_cum_vec_mae[f_idx]) ? "cum. error: n/a" :
            "cum. vector error: $(round(committed_cum_vec_mae[f_idx], digits=1)) m/s  " *
            "(spd $(round(committed_cum_spd_mae[f_idx], digits=1)) m/s, dir $(round(committed_cum_dir_mae[f_idx], digits=0))°)")
        committed_text[] = join(c_lines, "\n")

        has_hindsight && (hindsight_text[] = fmt_route("Hindsight", wh.state, ws_h, wd_h))

        fva_marker_x[] = t_s / 3600
    end

    # ------------------------------------------------------------
    # Transport controls
    # ------------------------------------------------------------
    controls = fig[4, 1] = GLMakie.GridLayout()
    current_t = GLMakie.Observable(0.0)
    is_playing = GLMakie.Observable(false)

    btn_prev  = GLMakie.Button(controls[1, 1], label = "◀ Prev")
    btn_next  = GLMakie.Button(controls[1, 2], label = "Next ▶")
    btn_reset = GLMakie.Button(controls[1, 3], label = "⟲ Reset")
    btn_play  = GLMakie.Button(controls[1, 4], label = "▶ Play")

    scrub = GLMakie.Slider(controls[1, 5:7], range = ts_grid, startvalue = 0.0)
    GLMakie.Label(controls[1, 8], GLMakie.lift(t -> "t = $(round(t/3600, digits=1)) h", current_t))

    step_dt = t_end / n_frames
    playback_interval = 0.1   # fixed frame interval for ▶ Play (was previously the speed slider)

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

    # explicit sizing so nothing gets squeezed by auto-layout -- done
    # last, once every row/col actually exists in the layout
    GLMakie.colsize!(top_row, 1, GLMakie.Relative(0.68))   # map: wide
    GLMakie.colsize!(top_row, 2, GLMakie.Relative(0.32))   # wind panes: narrow
    GLMakie.rowsize!(wind_col, 1, GLMakie.Relative(0.46))  # rolling wind pane
    GLMakie.rowsize!(wind_col, 2, GLMakie.Relative(0.46))  # committed wind pane
    GLMakie.rowsize!(wind_col, 3, GLMakie.Relative(0.08))  # shared colorbar

    for c in 1:ncols_stats
        GLMakie.colsize!(stats_grid, c, GLMakie.Relative(1.0 / ncols_stats))
    end

    GLMakie.rowsize!(fig.layout, 0, GLMakie.Fixed(36))      # title
    GLMakie.rowsize!(fig.layout, 1, GLMakie.Relative(0.40)) # map + wind panes
    GLMakie.rowsize!(fig.layout, 2, GLMakie.Fixed(90))      # stats (side by side now, shorter)
    GLMakie.rowsize!(fig.layout, 3, GLMakie.Relative(0.45)) # 3x2 vector/speed/direction error grid
    GLMakie.rowsize!(fig.layout, 4, GLMakie.Fixed(100))      # controls (scrub bar etc.)

    update!(0.0)
    GLMakie.display(fig)
    return fig
end



function launch_five_route_comparison_explorer(
    rolling_result::NamedTuple,
    rolling_climatology_result::NamedTuple,
    committed_stepper::RouteStepper2,
    committed_forecast_stepper::RouteStepper2,
    committed_climatology_stepper::RouteStepper2,
    committed_climatology_forecast_stepper::RouteStepper2;
    hindsight_stepper::Union{Nothing,RouteStepper2} = nothing,
    n_frames::Int = 150,
    rolling_gfs_dirs_list::Union{Nothing,Vector{String}} = nothing,
    rolling_start_hhmm::Int = 0,
    rolling_available_hours::Vector{Int} = vcat(collect(0:120), collect(123:3:336)),
    committed_forecast_max_hours::Float64 = 336.0,
    committed_climatology_forecast_max_hours::Float64 = 336.0
)
    has_hindsight = hindsight_stepper !== nothing
    has_rolling_forecast = rolling_gfs_dirs_list !== nothing

    rolling_stepper      = build_rolling_stepper(rolling_result)
    rolling_clim_stepper = build_rolling_stepper(rolling_climatology_result)
    rolling_day_offsets      = _rolling_day_offsets(rolling_result)
    rolling_clim_day_offsets = _rolling_day_offsets(rolling_climatology_result)
    rolling_fc_cache      = Dict{Int,WindCache}()
    rolling_clim_fc_cache = Dict{Int,WindCache}()

    grid_points = rolling_result.grid_points
    ocean_mask  = rolling_result.ocean_mask
    n_rows, n_cols = size(ocean_mask)
    lons_1d = [grid_points[1, j][1] for j in 1:n_cols]
    lats_1d = [grid_points[i, 1][2] for i in 1:n_rows]

    committed_fc_max_s      = committed_forecast_max_hours * 3600.0
    committed_clim_fc_max_s = committed_climatology_forecast_max_hours * 3600.0
    rolling_end_s      = rolling_stepper.steps[end].cum_time_s
    rolling_clim_end_s = rolling_clim_stepper.steps[end].cum_time_s
    committed_end_s      = committed_stepper.steps[end].cum_time_s
    committed_clim_end_s = committed_climatology_stepper.steps[end].cum_time_s

    routes = [
        (key = :rolling,        name = "Rolling",          stepper = rolling_stepper,              color = :steelblue, style = :solid),
        (key = :committed,      name = "Committed",        stepper = committed_stepper,            color = :firebrick, style = :dash),
        (key = :rolling_clim,   name = "Rolling (clim)",   stepper = rolling_clim_stepper,         color = :seagreen,  style = :solid),
        (key = :committed_clim, name = "Committed (clim)", stepper = committed_climatology_stepper, color = :orange,   style = :dash),
    ]
    if has_hindsight
        push!(routes, (key = :hindsight, name = "Hindsight", stepper = hindsight_stepper,
                       color = :black, style = :dot))
    end
    n_routes = length(routes)
    t_end = maximum(r.stepper.steps[end].cum_time_s for r in routes)
    forecast_kinds = (:rolling, :committed, :rolling_clim, :committed_clim)

    # ------------------------------------------------------------
    # Position + real-wind lookup, per route kind
    # ------------------------------------------------------------
    function real_wind_at(kind::Symbol, t_s::Float64)
        if kind == :rolling
            s = state_at_time(rolling_stepper, t_s)
            U, V, _, _ = _rolling_wind_at(rolling_result, rolling_day_offsets, t_s)
        elseif kind == :rolling_clim
            s = state_at_time(rolling_clim_stepper, t_s)
            U, V, _, _ = _rolling_wind_at(rolling_climatology_result, rolling_clim_day_offsets, t_s)
        elseif kind == :committed
            s = state_at_time(committed_stepper, t_s)
            U, V = get_wind(committed_stepper.wind, t_s)
        elseif kind == :committed_clim
            s = state_at_time(committed_climatology_stepper, t_s)
            U, V = get_wind(committed_climatology_stepper.wind, t_s)
        elseif kind == :hindsight
            s = state_at_time(hindsight_stepper, t_s)
            U, V = get_wind(hindsight_stepper.wind, t_s)
        else
            error("real_wind_at: unknown kind $kind")
        end
        i = nearest_index(lats_1d, s.lat)
        j = nearest_index(lons_1d, s.lon)
        return (state = s, U = U, V = V, i = i, j = j)
    end

    function forecast_wind_at(kind::Symbol, t_s::Float64)
        if kind == :rolling
            (has_rolling_forecast && t_s <= rolling_end_s) || return nothing
            return _rolling_forecast_wind_at(rolling_result, rolling_day_offsets, rolling_fc_cache,
                                              rolling_gfs_dirs_list, rolling_available_hours,
                                              rolling_start_hhmm, t_s)
        elseif kind == :rolling_clim
            (has_rolling_forecast && t_s <= rolling_clim_end_s) || return nothing
            return _rolling_forecast_wind_at(rolling_climatology_result, rolling_clim_day_offsets,
                                              rolling_clim_fc_cache, rolling_gfs_dirs_list,
                                              rolling_available_hours, rolling_start_hhmm, t_s)
        elseif kind == :committed
            (t_s <= committed_fc_max_s && t_s <= committed_end_s) || return nothing
            return get_wind(committed_forecast_stepper.wind, t_s)
        elseif kind == :committed_clim
            (t_s <= committed_clim_fc_max_s && t_s <= committed_clim_end_s) || return nothing
            return get_wind(committed_climatology_forecast_stepper.wind, t_s)
        else
            return nothing
        end
    end

    MIN_WS_FOR_DIR_ERROR = 5.0

    # ------------------------------------------------------------
    # Precompute forecast-vs-actual series for the four forecast kinds
    # ------------------------------------------------------------
    ts_grid = collect(range(0.0, t_end, length = n_frames))
    ts_h = ts_grid ./ 3600
    println("Precomputing forecast-vs-actual wind series ($n_frames samples x $(length(forecast_kinds)) routes)...")

    spd_err = Dict{Symbol,Vector{Float64}}(k => fill(NaN, n_frames) for k in forecast_kinds)
    dir_err = Dict{Symbol,Vector{Float64}}(k => fill(NaN, n_frames) for k in forecast_kinds)
    vec_err = Dict{Symbol,Vector{Float64}}(k => fill(NaN, n_frames) for k in forecast_kinds)
    cum_spd = Dict{Symbol,Vector{Float64}}(k => fill(NaN, n_frames) for k in forecast_kinds)
    cum_dir = Dict{Symbol,Vector{Float64}}(k => fill(NaN, n_frames) for k in forecast_kinds)
    cum_vec = Dict{Symbol,Vector{Float64}}(k => fill(NaN, n_frames) for k in forecast_kinds)

    for k in forecast_kinds
        running_spd = 0.0; running_dir = 0.0; running_vec = 0.0
        for (idx, t) in enumerate(ts_grid)
            wr = real_wind_at(k, t)
            ws_a, wd_a = _wind_speed_dir(wr.U, wr.V, wr.i, wr.j)
            u_a = Float64(wr.U[wr.i, wr.j]); v_a = Float64(wr.V[wr.i, wr.j])

            fc = forecast_wind_at(k, t)
            if fc !== nothing
                Ufc, Vfc = fc
                ws_f, wd_f = _wind_speed_dir(Ufc, Vfc, wr.i, wr.j)
                u_f = Float64(Ufc[wr.i, wr.j]); v_f = Float64(Vfc[wr.i, wr.j])

                spd_err[k][idx] = ws_f - ws_a
                vec_err[k][idx] = _wind_vector_error(u_f, v_f, u_a, v_a)
                if ws_a >= MIN_WS_FOR_DIR_ERROR && ws_f >= MIN_WS_FOR_DIR_ERROR
                    dir_err[k][idx] = _angular_diff(wd_f, wd_a)
                end
            end

            !isnan(spd_err[k][idx]) && (running_spd += abs(spd_err[k][idx]))
            !isnan(dir_err[k][idx]) && (running_dir += abs(dir_err[k][idx]))
            !isnan(vec_err[k][idx]) && (running_vec += abs(vec_err[k][idx]))
            cum_spd[k][idx] = running_spd
            cum_dir[k][idx] = running_dir
            cum_vec[k][idx] = running_vec
        end
    end
    println("  done")

    # ------------------------------------------------------------
    # Figure layout — mirrors the 3-route version's proportions.
    #
    #   row 0: title
    #   row 1: [ MAP (0.68) | wind panes stacked (0.32) ]   <- like 3-route
    #   row 2: stats (one column per route)
    #   row 3: error grid (3 rows x 2 cols)
    #   row 4: transport controls
    #
    # Toggles live in a slim right-hand strip that shares row 1 with
    # the top_row, so they don't steal width from the map.
    # ------------------------------------------------------------
    fig = GLMakie.Figure(size = (1900, 1500))
    GLMakie.Label(fig[0, 1:2], "Five-Route Comparison Explorer";
                  fontsize = 22, font = :bold, halign = :left, tellwidth = false)

    top_row = fig[1, 1] = GLMakie.GridLayout()

    ax_map = GLMakie.Axis(top_row[1, 1]; xlabel = "lon", ylabel = "lat")
    mask_z = permutedims(Float32.(ocean_mask))
    GLMakie.heatmap!(ax_map, lons_1d, lats_1d, mask_z;
                      colormap = [RGB(0.82, 0.75, 0.55), RGB(0.75, 0.87, 0.95)],
                      colorrange = (0, 1), interpolate = false)

    route_visible = [GLMakie.Observable(true) for _ in 1:n_routes]
    marker_pos    = [GLMakie.Observable(GLMakie.Point2f(r.stepper.steps[1].lon, r.stepper.steps[1].lat))
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

    # Wind panes: rolling + committed (kept identical to the 3-route version;
    # the two climatology routes are visible on the map but don't get their
    # own wind pane, matching that function's "drop the extras" choice).
    wind_col = top_row[1, 2] = GLMakie.GridLayout()
    ax_wind_r = GLMakie.Axis(wind_col[1, 1]; xlabel = "lon", ylabel = "lat",
                             title = "Rolling — actual wind")
    ax_wind_c = GLMakie.Axis(wind_col[2, 1]; xlabel = "lon", ylabel = "lat",
                             title = "Committed — actual wind")
    wind_range = GLMakie.Observable((0.0, 15.0))
    GLMakie.Colorbar(wind_col[3, 1]; colorrange = wind_range, colormap = :viridis,
                      label = "wind speed (m/s)", vertical = false)

    # ---- stats row: one column per route ----
    stats_grid = fig[2, 1] = GLMakie.GridLayout()
    time_text = GLMakie.Observable("")
    GLMakie.Label(stats_grid[1, 1:n_routes], time_text; tellwidth = false, halign = :left,
                  fontsize = 13, font = :bold)
    route_texts = [GLMakie.Observable("") for _ in 1:n_routes]
    for k in 1:n_routes
        GLMakie.Label(stats_grid[2, k], route_texts[k]; tellwidth = false, halign = :left,
                      valign = :top, fontsize = 11, justification = :left,
                      visible = route_visible[k])
        GLMakie.colsize!(stats_grid, k, GLMakie.Relative(1.0 / n_routes))
    end

    # ---- error grid: rows = vector/speed/direction, cols = instant/cumulative ----
    bottom_row = fig[3, 1] = GLMakie.GridLayout()
    kind_of_route = Dict(r.key => k for (k, r) in enumerate(routes))

    function err_axis(row, col, title, ylabel, series, cumulative::Bool)
        ax = GLMakie.Axis(bottom_row[row, col]; title = title, ylabel = ylabel)
        !cumulative && GLMakie.hlines!(ax, [0.0]; color = :gray50, linestyle = :dash)
        for k in forecast_kinds
            ridx = kind_of_route[k]
            GLMakie.lines!(ax, ts_h, series[k]; color = routes[ridx].color,
                            label = routes[ridx].name, visible = route_visible[ridx])
        end
        return ax
    end

    ax_vec_err = err_axis(1, 1, "Vector wind error (forecast − actual)", "error (m/s)", vec_err, false)
    ax_vec_cum = err_axis(1, 2, "Cumulative total vector error", "Σ|error| (m/s)", cum_vec, true)
    ax_spd_err = err_axis(2, 1, "Speed error (forecast − actual)", "error (m/s)", spd_err, false)
    ax_spd_cum = err_axis(2, 2, "Cumulative total speed error", "Σ|error| (m/s)", cum_spd, true)
    GLMakie.axislegend(ax_vec_err; position = :rt, labelsize = 9)
    GLMakie.linkxaxes!(ax_vec_err, ax_vec_cum, ax_spd_err, ax_spd_cum)

    fva_marker_x = GLMakie.Observable(0.0)
    for ax in (ax_vec_err, ax_vec_cum, ax_spd_err, ax_spd_cum)
        GLMakie.vlines!(ax, fva_marker_x; color = :black, linestyle = :dash, linewidth = 1.2)
    end
    GLMakie.rowsize!(bottom_row, 1, GLMakie.Relative(0.5))
    GLMakie.rowsize!(bottom_row, 2, GLMakie.Relative(0.5))
    GLMakie.colsize!(bottom_row, 1, GLMakie.Relative(0.5))
    GLMakie.colsize!(bottom_row, 2, GLMakie.Relative(0.5))

    # ---- toggles: slim right strip sharing row 1 with top_row ----
    toggle_col = fig[1, 2] = GLMakie.GridLayout()
    GLMakie.Label(toggle_col[1, 1]; text = "Routes", fontsize = 13, font = :bold, halign = :left)
    for (k, r) in enumerate(routes)
        t = GLMakie.Toggle(toggle_col[k + 1, 1], active = true)
        GLMakie.Label(toggle_col[k + 1, 2], r.name; fontsize = 11, color = r.color, halign = :left)
        on(t.active) do a
            route_visible[k][] = a
        end
    end

    # ------------------------------------------------------------
    # Update logic
    # ------------------------------------------------------------
    function fmt_route(name, s, ws, wd, fc_line)
        base = "$(name)\n$(round(s.cum_time_s/3600, digits=1)) h elapsed\n" *
               "$(round(s.dist_km, digits=0)) km  |  $(round(s.energy_kwh, digits=0)) kWh\n" *
               "$(round(s.speed_kn, digits=1)) kn  |  $(round(s.power_kw, digits=0)) kW"
        return base * "\nwind: $(round(ws, digits=1)) m/s @ $(round(wd, digits=0))°" * fc_line
    end

    function update!(t_s::Float64)
        time_text[] = "elapsed time: $(round(t_s / 3600, digits=1)) h  ($(round(t_s / 86400, digits=1)) days)"
        f_idx = clamp(searchsortedfirst(ts_grid, t_s), 1, n_frames)

        max_ws = 1.0
        wr = nothing; wc = nothing
        for (k, r) in enumerate(routes)
            rw = real_wind_at(r.key, t_s)
            marker_pos[k][] = GLMakie.Point2f(rw.state.lon, rw.state.lat)
            ws, wd = _wind_speed_dir(rw.U, rw.V, rw.i, rw.j)
            max_ws = max(max_ws, ws)
            r.key == :rolling && (wr = rw)
            r.key == :committed && (wc = rw)

            fc_line = ""
            if r.key in forecast_kinds
                cs = round(cum_vec[r.key][f_idx], digits=1)
                fc = forecast_wind_at(r.key, t_s)
                if fc !== nothing
                    ws_f, wd_f = _wind_speed_dir(fc[1], fc[2], rw.i, rw.j)
                    fc_line = "\nforecast: $(round(ws_f, digits=1)) m/s @ $(round(wd_f, digits=0))°" *
                              "\ncum. vector err: $(isnan(cs) ? "n/a" : "$(cs) m/s")"
                else
                    fc_line = "\nforecast: unavailable"
                end
            end
            route_texts[k][] = fmt_route(r.name, rw.state, ws, wd, fc_line)
        end

        wind_range[] = (0.0, max_ws + 5.0)

        if wr !== nothing
            _draw_route_wind_zoom!(ax_wind_r, wr.U, wr.V, wr.i, wr.j, grid_points, ocean_mask, wind_range;
                                    zoom_cells = 12, title = "Rolling — actual wind")
            GLMakie.scatter!(ax_wind_r, [wr.state.lon], [wr.state.lat];
                              color = :steelblue, markersize = 14, marker = :star5)
        end
        if wc !== nothing
            _draw_route_wind_zoom!(ax_wind_c, wc.U, wc.V, wc.i, wc.j, grid_points, ocean_mask, wind_range;
                                    zoom_cells = 12, title = "Committed — actual wind")
            GLMakie.scatter!(ax_wind_c, [wc.state.lon], [wc.state.lat];
                              color = :firebrick, markersize = 14, marker = :star5)
        end

        fva_marker_x[] = t_s / 3600
    end

    # ------------------------------------------------------------
    # Transport controls
    # ------------------------------------------------------------
    controls = fig[4, 1:2] = GLMakie.GridLayout()
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

    # ---- sizing: mirrors 3-route proportions + slim toggle strip ----
    GLMakie.colsize!(fig.layout, 1, GLMakie.Relative(0.92))
    GLMakie.colsize!(fig.layout, 2, GLMakie.Relative(0.08))
    GLMakie.colsize!(top_row, 1, GLMakie.Relative(0.82))
    GLMakie.colsize!(top_row, 2, GLMakie.Relative(0.18))
    GLMakie.rowsize!(wind_col, 1, GLMakie.Relative(0.40))
    GLMakie.rowsize!(wind_col, 2, GLMakie.Relative(0.40))
    GLMakie.rowsize!(wind_col, 3, GLMakie.Relative(0.20))

    GLMakie.rowsize!(fig.layout, 0, GLMakie.Fixed(36))      # title
    GLMakie.rowsize!(fig.layout, 1, GLMakie.Relative(0.40)) # map + wind panes
    GLMakie.rowsize!(fig.layout, 2, GLMakie.Fixed(90))      # stats
    GLMakie.rowsize!(fig.layout, 3, GLMakie.Relative(0.45)) # error grid
    GLMakie.rowsize!(fig.layout, 4, GLMakie.Fixed(100))     # controls

    update!(0.0)
    GLMakie.display(fig)
    return fig
end


"""
    save_climatology(path, climatology::Climatology)

Persists just the wind-rose climatology (roses + months) -- no grid_points,
no ocean_mask, no DP output. Use this right after build_climatology(...) so
the GRIB archive never has to be re-scanned for the same climatology again.
"""
function save_climatology(path::String, climatology::Climatology)
    jldopen(path, "w") do f
        f["roses"]  = climatology.roses
        f["months"] = climatology.months
    end
    println("Saved climatology to $path")
end

"""
    load_climatology(path) -> Climatology
"""
function load_climatology(path::String)
    local roses, months
    jldopen(path, "r") do f
        roses  = f["roses"]
        months = f["months"]
    end
    n_rows, n_cols = size(roses)
    return Climatology(roses, n_rows, n_cols, months)
end

"""
    run_november_climatology_uv(; start_location, end_location, historical_dir,
                                   historical_paths = nothing, month_of_path,
                                   lon_step = 0.25, lat_step = 0.25, padding = 20,
                                   n_ws = 18, ws_max = 30f0, n_dir = 24,
                                   save_climatology_path = "climatology_november.jld2")

Top-level convenience wrapper, mirroring the setup half of
run_climatology_isochrones (same grid/interpolator construction) but
stopping right after the climatology + flattened expected-uv are built --
no backward DP, no isochrone plot, since this is purely a wind SOURCE for
blending, not a routing result in its own right.

`month_of_path` is REQUIRED here (unlike build_climatology, which allows
`months=nothing` to mean "pool everything") -- you're deliberately asking
for a November-only climatology, so there is no sensible default for which
calendar month each historical file belongs to. This codebase has no
standardized file-naming-to-month convention yet, so it isn't derived
automatically; build it yourself from whatever your historical_dir's
filenames/directory structure actually encode.

CRITICAL: `start_location`, `end_location`, `lon_step`, `lat_step`, `padding`
here MUST exactly match whatever you later pass to run_forecast_route /
run_blended_forecast_committed. Eu_flat/Ev_flat are indexed by this
function's own grid_points; a mismatched grid silently blends the wrong
cells together with no error. run_blended_forecast_committed below does a
best-effort size check (length(Eu_flat) vs the solved route's own grid) to
catch a GROSS mismatch (different grid dimensions entirely), but two grids
of the same n_rows x n_cols built from different bounding boxes would still
pass that check while blending nonsense -- keep these settings identical by
construction (e.g. define them once and pass the same NamedTuple to both
calls) rather than relying on the size check to save you.
"""
function run_november_climatology_uv(;
        start_location, end_location,
        lon_step::Float64 = 0.25, lat_step::Float64 = 0.25, padding::Int = 20,
        historical_dir::String = _climatology_dir(),
        historical_paths::Union{Nothing,Vector{String}} = nothing,
        month_of_path::Vector{Int},
        n_ws::Int = 18, ws_max::Float32 = 30f0, n_dir::Int = 24,
        save_climatology_path::Union{Nothing,String} = "climatology_november.jld2"
    )
    grid_points, latitudes, longitudes, ocean_mask =
        _build_grid_and_mask(; start_location, end_location, lon_step, lat_step, padding)

    paths = historical_paths === nothing ?
        [joinpath(historical_dir, f) for f in readdir(historical_dir) if endswith(f, ".grib")] :
        historical_paths
    isempty(paths) && error("No historical .grib files found in $historical_dir.")
    length(paths) == length(month_of_path) ||
        error("month_of_path must have one entry per historical path -- got " *
              "$(length(month_of_path)) entries vs $(length(paths)) paths.")

    n_nov = count(==(11), month_of_path)
    n_nov == 0 && error("month_of_path contains no entries == 11 -- nothing to build a " *
                         "November climatology from. Check the month labels you supplied.")

    ref_field = _reference_field(paths[1])
    interp    = build_interpolator(grid_points, ref_field)

    println("Building November climatology from $n_nov / $(length(paths)) source files...")
    climatology = build_climatology(
        paths, grid_points, interp, ocean_mask;
        n_ws, ws_max, n_dir, months = [11], month_of_path
    )
    save_climatology_path !== nothing && save_climatology(save_climatology_path, climatology)

    return (climatology = climatology, 
            grid_points = grid_points, ocean_mask = ocean_mask,
            latitudes = latitudes, longitudes = longitudes)
end


# -----------------------------------------------------------------------
# 1b. PER-MONTH climatologies -- one climatology per calendar month
#     (NOT a single pooled full-year climatology), so a test day is always
#     scored against its own month by default, and CAN be checked against
#     every other month too (see score_climatology_match_all_months below)
#     to answer "what if November just isn't representative this year".
# -----------------------------------------------------------------------
"""
    month_from_filename(path) -> Int

Parses the calendar month from this archive's own "YYYY-MM.grib" naming
convention. Errors clearly if a filename doesn't match, rather than
silently miscategorizing it.
"""
function month_from_filename(path::String)
    m = match(r"(\d{4})-(\d{2})\.grib$", basename(path))
    m === nothing && error("month_from_filename: '$(basename(path))' doesn't match the " *
                            "expected YYYY-MM.grib naming convention -- adjust this regex if " *
                            "your archive is named differently.")
    return parse(Int, m.captures[2])
end

"""
    save_climatologies_by_month(path, climatologies::Dict{Int,Climatology})
    load_climatologies_by_month(path) -> Dict{Int,Climatology}

Persists/reloads a whole set of monthly climatologies (roses+months only,
same as save_climatology/load_climatology) in one file, keyed by month
number 1..12.
"""
function save_climatologies_by_month(path::String, climatologies::Dict{Int,Climatology})
    jldopen(path, "w") do f
        for (m, clim) in climatologies
            f["month_$(m)/roses"]  = clim.roses
            f["month_$(m)/months"] = clim.months
        end
        f["months_present"] = sort(collect(keys(climatologies)))
    end
    println("Saved $(length(climatologies)) monthly climatologies to $path")
end

function load_climatologies_by_month(path::String)
    climatologies = Dict{Int,Climatology}()
    jldopen(path, "r") do f
        months_present = f["months_present"]
        for m in months_present
            roses  = f["month_$(m)/roses"]
            months = f["month_$(m)/months"]
            n_rows, n_cols = size(roses)
            climatologies[m] = Climatology(roses, n_rows, n_cols, months)
        end
    end
    println("Loaded climatologies for months: $(sort(collect(keys(climatologies))))")
    return climatologies
end

"""
    build_climatologies_by_month(; start_location, end_location, historical_dir,
                                    historical_paths = nothing, months = 1:12,
                                    lon_step = 0.25, lat_step = 0.25, padding = 20,
                                    n_ws = 18, ws_max = 30f0, n_dir = 24,
                                    save_path = "climatology_by_month.jld2")

Builds ONE climatology per calendar month in `months` (default all 12),
each restricted to just that month's own files -- e.g. only the ~20
"YYYY-11.grib" files feed month 11's climatology, only the ~20 "YYYY-02"
files feed month 2's, etc. This is NOT a single pooled full-year
climatology; each month's Climatology is completely independent, matching
run_november_climatology_uv's approach just generalized to every month.
The GRIB archive is only listed/month-parsed once and the grid/interpolator
are built once and reused across all 12 builds.

Returns (climatologies::Dict{Int,Climatology}, grid_points, ocean_mask,
latitudes, longitudes, interp).
"""
function build_climatologies_by_month(;
        start_location, end_location,
        lon_step::Float64 = 0.25, lat_step::Float64 = 0.25, padding::Int = 20,
        historical_dir::String = _climatology_dir(),
        historical_paths::Union{Nothing,Vector{String}} = nothing,
        months::AbstractVector{Int} = 1:12,
        n_ws::Int = 18, ws_max::Float32 = 30f0, n_dir::Int = 24,
        save_path::Union{Nothing,String} = "climatology_by_month.jld2"
    )
    grid_points, latitudes, longitudes, ocean_mask =
        _build_grid_and_mask(; start_location, end_location, lon_step, lat_step, padding)

    all_paths = historical_paths === nothing ?
        [joinpath(historical_dir, f) for f in readdir(historical_dir) if endswith(f, ".grib")] :
        historical_paths
    isempty(all_paths) && error("No historical .grib files found in $historical_dir.")

    all_months = month_from_filename.(all_paths)

    ref_field = _reference_field(all_paths[1])
    interp    = build_interpolator(grid_points, ref_field)

    climatologies = Dict{Int,Climatology}()
    for m in months
        m_paths = all_paths[all_months .== m]
        if isempty(m_paths)
            @warn "No files found for month $m -- skipping."
            continue
        end
        println("\n--- Month $m: $(length(m_paths)) files ---")
        climatologies[m] = build_climatology(
            m_paths, grid_points, interp, ocean_mask;
            n_ws, ws_max, n_dir, months = [m], month_of_path = fill(m, length(m_paths))
        )
    end

    save_path !== nothing && save_climatologies_by_month(save_path, climatologies)

    return (climatologies = climatologies, grid_points = grid_points, ocean_mask = ocean_mask,
            latitudes = latitudes, longitudes = longitudes, interp = interp)
end

# -----------------------------------------------------------------------
# 2. Fixed-power climatological route: build once, replay for any date.
# -----------------------------------------------------------------------
"""
    build_climatology_next_hop(; climatology, grid_points, ocean_mask, nb_flat,
                                  power_kw, power_levels, min_speed, goal_idx)

Builds the FIXED-power 2D backward DP (climatology_backward_dp, not the
energy-aware 3D version -- constant power means no budget dimension is
needed at all here). `power_kw` must be one of `power_levels`. Returns
`next_hop`, which does NOT depend on the date -- build it once and reuse
it for every candidate day's replay in step 3.
"""
function build_climatology_next_hop(;
        climatology::Climatology, grid_points::Matrix{Tuple{Float64,Float64}},
        ocean_mask::Matrix{Bool},
        nb_flat::Matrix{Vector{Tuple{Int,Int,Float64,Float64,Vector{Int},Int8}}},
        power_kw::Float64, power_levels::Vector{Float64} = [0.0, 400.0, 1200.0, 2000.0],
        min_speed::Float64 = 1.0, goal_idx::Tuple{Int,Int}
    )
    power_levels_sorted = sort(power_levels)
    power_kw in power_levels_sorted ||
        error("build_climatology_next_hop: power_kw=$power_kw not found in " *
              "power_levels=$power_levels_sorted -- pick one of those values, or add power_kw " *
              "to power_levels (needs a matching ./Polars/RisePol<power_kw>kW.csv file).")
    power_idx = findfirst(==(power_kw), power_levels_sorted)
    slut = build_stacked_lut(power_levels_sorted)
    V, next_hop = climatology_backward_dp(
        grid_points, ocean_mask, nb_flat, climatology, slut, power_idx, min_speed, goal_idx
    )
    return next_hop
end

"""
    reconstruct_climatology_route(next_hop, start_idx, goal_idx) -> Vector{Tuple{Int,Int}}

Walks next_hop from start_idx to goal_idx, per climatology_backward_dp's
own docstring. Errors clearly (rather than looping forever) if start_idx
was never settled, or if a cycle is somehow hit.
"""
function reconstruct_climatology_route(next_hop::Matrix{Tuple{Int,Int}},
                                        start_idx::Tuple{Int,Int}, goal_idx::Tuple{Int,Int})
    path_idxs = Tuple{Int,Int}[start_idx]
    cur = start_idx
    max_steps = length(next_hop) + 10   # generous cycle guard
    steps = 0
    while cur != goal_idx
        cur = next_hop[cur...]
        cur == (0, 0) && error("reconstruct_climatology_route: unreachable from $start_idx -- " *
                                "next_hop hit (0,0) before reaching goal (this start cell was " *
                                "never settled by the backward DP, e.g. no feasible edges at " *
                                "this power level).")
        push!(path_idxs, cur)
        steps += 1
        steps > max_steps && error("reconstruct_climatology_route: exceeded $max_steps hops -- " *
                                    "likely a next_hop cycle; this should not happen for a " *
                                    "correctly-built backward DP.")
    end
    return path_idxs
end

"""
    replay_climatology_route_real_wind(next_hop, start_idx, goal_idx, power_kw;
                                        grid_points, ocean_mask, nb_flat,
                                        hhmm, day, month, year, wind_cache_size = 24)

Reconstructs the fixed-power climatological route from start_idx and
replays it through REAL ERA5 wind at CONSTANT power_kw, reusing
replay_route_real_wind exactly as the forecast-committed pipeline does --
just fed a climatology-derived path and a constant power vector instead of
a live-solver path/power schedule. e_bin in the synthesized path_idxs is
unused (kept 0) since there's no budget dimension here. Requires
clear_gfs_forecast_source!() state.
"""
function replay_climatology_route_real_wind(
        next_hop::Matrix{Tuple{Int,Int}}, start_idx::Tuple{Int,Int}, goal_idx::Tuple{Int,Int},
        power_kw::Float64;
        grid_points::Matrix{Tuple{Float64,Float64}}, ocean_mask::Matrix{Bool},
        nb_flat::Matrix{Vector{Tuple{Int,Int,Float64,Float64,Vector{Int},Int8}}},
        hhmm::Int, day::Int, month::Int, year::Int,
        wind_cache_size::Int = 24
    )
    route_idxs = reconstruct_climatology_route(next_hop, start_idx, goal_idx)
    path_idxs   = [(i, j, 0) for (i, j) in route_idxs]
    path_coords = [grid_points[i, j] for (i, j) in route_idxs]
    power_used  = fill(power_kw, length(route_idxs) - 1)

    diag = Dict{Symbol,Any}(
        :grid_points => grid_points, :ocean_mask => ocean_mask, :path_idxs => path_idxs,
        :cur_time => hhmm, :cur_day => day, :cur_month => month, :cur_year => year,
        :nb_flat => nb_flat
    )
    return replay_route_real_wind(diag, path_coords, power_used; wind_cache_size)
end


# =============================================================================
# 2. Pooled climatology rose over the corridor, per month -- combines each
#    cell's WindRose (weighted by n_obs) into one rose on the shared bins.
# =============================================================================
function pooled_climatology_rose(clim::Climatology, corridor_mask::Matrix{Bool})
    ref = nothing
    for j in 1:clim.n_cols, i in 1:clim.n_rows
        (!corridor_mask[i, j]) && continue
        r = clim.roses[i, j]
        r.n_obs == 0 && continue
        ref = r; break
    end
    ref === nothing && error("pooled_climatology_rose: no climatology data in corridor")

    n_ws, n_dir = size(ref.probs)
    total_counts = zeros(Float64, n_ws, n_dir)
    for j in 1:clim.n_cols, i in 1:clim.n_rows
        (!corridor_mask[i, j]) && continue
        r = clim.roses[i, j]
        r.n_obs == 0 && continue
        total_counts .+= Float64.(r.probs) .* r.n_obs
    end
    s = sum(total_counts)
    s == 0 && error("pooled_climatology_rose: zero total mass in corridor")
    return (ws_edges = ref.ws_edges, dir_edges = ref.dir_edges, probs = total_counts ./ s)
end


# =============================================================================
# 4. Distribution distances
# =============================================================================
function jensen_shannon_distance(P::AbstractArray{<:Real}, Q::AbstractArray{<:Real})
    p = vec(Float64.(P)); q = vec(Float64.(Q))
    sp = sum(p); sq = sum(q)
    (sp == 0 || sq == 0) && return NaN
    p ./= sp; q ./= sq
    m = 0.5 .* (p .+ q)
    kl = 0.0
    @inbounds for k in eachindex(p)
        p[k] > 0 && m[k] > 0 && (kl += 0.5 * p[k] * log(p[k]/m[k]))
        q[k] > 0 && m[k] > 0 && (kl += 0.5 * q[k] * log(q[k]/m[k]))
    end
    return sqrt(max(kl, 0.0))
end

"1D Wasserstein-1 on the speed marginal (closed form: area between CDFs)."
function wasserstein1_speed_marginal(P::Matrix{<:Real}, Q::Matrix{<:Real}, ws_edges::Vector{Float32})
    p_speed = vec(sum(P, dims = 2)); q_speed = vec(sum(Q, dims = 2))
    sp = sum(p_speed); sq = sum(q_speed)
    (sp == 0 || sq == 0) && return NaN
    p_speed ./= sp; q_speed ./= sq
    cdf_p = cumsum(p_speed); cdf_q = cumsum(q_speed)
    widths = diff(Float64.(ws_edges))
    return sum(abs.(cdf_p .- cdf_q) .* widths)
end

# =============================================================================
# 5. Per-month cache: group each batch file's 6-hourly snapshots by day
#    (interpolated onto grid once), so re-scanning days in the same month
#    doesn't re-read/re-interpolate the GRIB file repeatedly.
# =============================================================================
function load_month_days(historical_dir::String, year::Int, month::Int, interp::GridInterpolator,
                          n_rows::Int, n_cols::Int)
    path = joinpath(historical_dir, "$(year)-$(lpad(month,2,'0')).grib")
    isfile(path) || return Dict{Date, Vector{Tuple{Matrix{Float32},Matrix{Float32}}}}()

    snaps = _load_uv_dated(path)   # (dataDate, time, WindField)
    by_day = Dict{Date, Vector{Tuple{Matrix{Float32},Matrix{Float32}}}}()
    n_flat = n_rows * n_cols
    for (dataDate, time_, field) in snaps
        yr = dataDate ÷ 10000; mo = (dataDate ÷ 100) % 100; dy = dataDate % 100
        d = Date(yr, mo, dy)
        u_buf = Vector{Float32}(undef, n_flat); v_buf = Vector{Float32}(undef, n_flat)
        precompute_uv_grid!(u_buf, v_buf, field, interp)
        push!(get!(by_day, d, Tuple{Matrix{Float32},Matrix{Float32}}[]),
              (reshape(u_buf, n_rows, n_cols), reshape(v_buf, n_rows, n_cols)))
    end
    return by_day
end

"""
    validate_climatology_match_windows(window_scores::DataFrame;
        top_k = 5, next_hop, start_idx, goal_idx, power_kw,
        grid_points, ocean_mask, nb_flat, gfs_dirs_for_date,
        max_energy = 2_000_000.0, bin_size = 20_000.0, hhmm = 0)

Ground-truth check for the top `top_k` windows from
`rolling_window_climatology_scores`. The distributional match score is a
proxy for "will a climatology route do well here" -- this is the step
that actually confirms it, by:

  1. Reconstructing the fixed-power climatology-optimal route from
     `next_hop` (build once via `build_climatology_next_hop`, reused
     across every window here) and replaying it through REAL wind
     starting at each window's start date
     (`replay_climatology_route_real_wind`).
  2. Separately solving + replaying a forecast-committed route for the
     SAME start date (`run_single_forecast_committed`), as the
     "informed" baseline to compare against.

Requires `clear_gfs_forecast_source!()` state to be current before this
is called (it toggles the GFS source internally per call, same
requirement as `run_constant_power_experiment`).

Returns a DataFrame: window_start, window_score, climatology_time_h,
forecast_time_h, actual_transit_days, climatology_vs_forecast_h
(negative => climatology route was faster/comparable under real wind for
this window -- the actual claim being tested). `actual_transit_days` vs
the `window_days` used to build `window_scores` is worth checking too: if
the real-wind transit runs well past the scored window, part of the
voyage happened in conditions that were never scored.
"""
function validate_climatology_match_windows(
        window_scores::DataFrame;
        top_k::Int = 5,
        next_hop::Matrix{Tuple{Int,Int}}, start_idx::Tuple{Int,Int}, goal_idx::Tuple{Int,Int},
        power_kw::Float64,
        grid_points::Matrix{Tuple{Float64,Float64}}, ocean_mask::Matrix{Bool},
        nb_flat::Matrix{Vector{Tuple{Int,Int,Float64,Float64,Vector{Int},Int8}}},
        gfs_dirs_for_date::Function,
        max_energy::Float64 = 2_000_000.0, bin_size::Float64 = 20_000.0,
        hhmm::Int = 0
    )
    nrow(window_scores) == 0 &&
        error("validate_climatology_match_windows: window_scores is empty")

    out = DataFrame(window_start = Date[], window_score = Float64[],
                     climatology_time_h = Float64[], forecast_time_h = Float64[],
                     actual_transit_days = Float64[], climatology_vs_forecast_h = Float64[])

    score_col = hasproperty(window_scores, :mean_combined_z) ? :mean_combined_z :
                :worst_day_combined_z

    for row in first(eachrow(window_scores), min(top_k, nrow(window_scores)))
        d = row.window_start
        s = row[score_col]
        println("\n=== Validating window starting $d (score=$(round(s, digits=3))) ===")

        clear_gfs_forecast_source!()
        actual_clim = replay_climatology_route_real_wind(
            next_hop, start_idx, goal_idx, power_kw;
            grid_points, ocean_mask, nb_flat,
            hhmm, day = Dates.day(d), month = Dates.month(d), year = Dates.year(d)
        )
        clim_time_h = actual_clim.steps[end].cum_time_s / 3600

        fc_committed = run_single_forecast_committed(
            gfs_dir = gfs_dirs_for_date(d),
            init_year = Dates.year(d), init_month = Dates.month(d), init_day = Dates.day(d),
            cycle_hour = 0,
            start_location = grid_points[start_idx...], end_location = grid_points[goal_idx...],
            start_date = d, start_hhmm = hhmm,
            power_levels = [power_kw], max_energy = max_energy, bin_size = bin_size
        )
        fc_time_h = fc_committed.actual.steps[end].cum_time_s / 3600

        push!(out, (d, s, clim_time_h, fc_time_h, clim_time_h / 24, clim_time_h - fc_time_h))

        println("  climatology real-wind transit: $(round(clim_time_h / 24, digits=1)) days " *
                "($(round(clim_time_h, digits=1)) h)")
        println("  forecast-committed real-wind:  $(round(fc_time_h / 24, digits=1)) days " *
                "($(round(fc_time_h, digits=1)) h)")
        println("  Δ (climatology - forecast): $(round(clim_time_h - fc_time_h, digits=1)) h" *
                (clim_time_h <= fc_time_h ? "  (climatology matched/beat forecast)" : ""))
    end

    return out
end

