# =============================================================================
# climatology_routing.jl
#
# Builds a wind-rose climatology per grid cell from many years of historical
# wind fields, then runs a backward Dijkstra from a fixed goal over the
# *static* expected-cost graph this induces. The result is a value function
# V (expected time-to-goal, SECONDS) for every ocean cell, plus a next-hop
# pointer that reconstructs the optimal-in-expectation route from any start.
#
# Layout
#   0. Includes / small helpers
#   1. GRIB loading (_load_uv_dated, _load_uv_for_climatology_grib, _reference_field)
#   2. Wind-rose climatology (WindRose, Climatology, build_climatology)
#   3. draw_wind_rose!
#   4. Expected edge costs from a wind rose
#   5. Fixed-power backward DP (2D)
#   6. Climatology blend (edge cache, cache I/O, set_climatology_blend!)
#   7. Energy-aware backward DP (3D) + route reconstruction
#   8. Great-circle reference, plots, save/load, run_climatology_isochrones*
#   9. Interactive explorers
#
# Requires (loaded before this file):
#   routing.jl           WindField, GridInterpolator, StackedLUT, NbCache, BucketQueue2,
#                        build_interpolator, precompute_uv_grid!, build_neighbour_cache,
#                        build_stacked_lut, stacked_lookup!, _rotate_to_pm180
#   route_simulator.jl   RouteStepper2 (used in explorer signatures)
#   read_write_data.jl   getDistance, getBoatAngle
#   grid helpers         _build_grid_and_mask, _check_endpoints, nearest_index
# =============================================================================

using DataStructures
using Statistics
using Dates
using GRIB
using CairoMakie
using Colors: RGB
using JLD2
using GLMakie
using Printf

@inline function climatology_uses_forecast(elapsed_seconds::Real; full_forecast_days::Float64 = 14.0)::Bool
    return elapsed_seconds < full_forecast_days * 86400.0
end

"""Archive directory: CLIMATOLOGY_ARCHIVE_DIR_REF[] if the project defines it, else the old default."""
_climatology_dir() =
    isdefined(@__MODULE__, :CLIMATOLOGY_ARCHIVE_DIR_REF) ?
        getfield(@__MODULE__, :CLIMATOLOGY_ARCHIVE_DIR_REF)[] : "./ClimatologyOceanData/route"

# =============================================================================
# 1. GRIB loading
# =============================================================================

# Build a WindField ([lon, lat] layout, ascending axes, -180..180) from raw GRIB arrays,
# by explicit coordinate lookup (no reliance on GRIB scan order).
function _windfield_from_arrays(lons, lats, uvals, vvals)::WindField
    lats_unique = sort(unique(lats))
    lons_unique = sort(unique(lons))
    n_lats, n_lons = length(lats_unique), length(lons_unique)
    lat_row = Dict{eltype(lats_unique), Int}(la => i for (i, la) in enumerate(lats_unique))
    lon_col = Dict{eltype(lons_unique), Int}(lo => j for (j, lo) in enumerate(lons_unique))

    u_mat = Array{Float32}(undef, n_lons, n_lats)
    v_mat = Array{Float32}(undef, n_lons, n_lats)
    @inbounds for k in eachindex(lons)
        i = lat_row[lats[k]]; j = lon_col[lons[k]]
        u_mat[j, i] = Float32(uvals[k]); v_mat[j, i] = Float32(vvals[k])
    end

    lons_f = Float64.(lons_unique)
    if maximum(lons_f) > 180.0                      # 0..360 file -> -180..180
        lons_f, u_mat, v_mat = _rotate_to_pm180(lons_f, u_mat, v_mat)
    end
    return WindField(Float64.(lats_unique), lons_f, u_mat, v_mat, n_lats, 0)
end

"""
    _load_uv_dated(path; verbose=false) -> Vector{(dataDate, time, WindField)}

One entry per (dataDate, time) group in the GRIB file, sorted. Every group must
contain both the 10 m U and V messages. The WindField epoch is 0 (callers that
need a real epoch compute it from dataDate/time).
"""
function _load_uv_dated(path::String; verbose::Bool = false)
    endswith(path, "manifest.jld2") && error("Skipping manifest file")
    GRIB.GribFile(path) do file
        msgs_by_key = Dict{Tuple{Int,Int}, Dict{String,Any}}()
        for msg in file
            if verbose
                println("GRIB: ", msg["name"], " ", msg["shortName"], " ", msg["dataDate"], " ", msg["time"])
            end
            key = (msg["dataDate"], msg["time"])
            d = get!(msgs_by_key, key, Dict{String,Any}())
            nm = msg["name"]
            if nm == "10 metre U wind component";      d["u"] = msg
            elseif nm == "10 metre V wind component";  d["v"] = msg
            end
        end
        isempty(msgs_by_key) && error("$path: no (dataDate,time) groups found in this GRIB file")

        triples = Tuple{Int,Int,WindField}[]
        for key in sort(collect(keys(msgs_by_key)))
            d = msgs_by_key[key]
            (haskey(d, "u") && haskey(d, "v")) ||
                error("$path: missing U and/or V message for timestamp $key " *
                      "(file has $(length(msgs_by_key)) timestamp(s))")
            lons, lats, uvals = data(d["u"])
            _, _, vvals = data(d["v"])
            push!(triples, (key[1], key[2], _windfield_from_arrays(lons, lats, uvals, vvals)))
        end
        return triples
    end
end

"""All snapshots in a GRIB file as a Vector{WindField} (epoch left at 0)."""
_load_uv_for_climatology_grib(path::String; verbose::Bool = false) =
    WindField[t[3] for t in _load_uv_dated(path; verbose)]

"""
Grid-defining WindField from the FIRST u/v message pair only. Use this instead of
`first(_load_uv_for_climatology_grib(path))`, which parses every snapshot in the file.
"""
function _reference_field(path::String)::WindField
    GRIB.GribFile(path) do file
        u = v = nothing
        for msg in file
            nm = msg["name"]
            nm == "10 metre U wind component" && (u = msg)
            nm == "10 metre V wind component" && (v = msg)
            (u !== nothing && v !== nothing) && break
        end
        (u === nothing || v === nothing) && error("$path: no 10 m U/V message pair found")
        lons, lats, uvals = data(u)
        _, _, vvals = data(v)
        return _windfield_from_arrays(lons, lats, uvals, vvals)
    end
end

# =============================================================================
# 2. Wind-rose climatology per cell
# =============================================================================

struct WindRose
    ws_edges  ::Vector{Float32}   # length n_ws+1 (m/s)
    dir_edges ::Vector{Float32}   # length n_dir+1 (deg, 0..360); direction the wind blows FROM
    probs     ::Matrix{Float32}   # n_ws x n_dir, sums to 1 (all-zero if no data / land)
    n_obs     ::Int
end

struct Climatology
    roses     ::Matrix{WindRose}
    n_rows    ::Int
    n_cols    ::Int
    months    ::Union{Nothing, Vector{Int}}   # nothing = all-year pooled
end

@inline function _bin_index(x::Float32, edges::Vector{Float32})::Int
    return clamp(searchsortedlast(edges, x), 1, length(edges) - 1)
end

"""
    build_climatology(historical_paths, grid_points, interp, ocean_mask; ...)

One linear pass over the archive; every snapshot in every file is folded in
(a "month file" contributes all its timestamps). `months`/`month_of_path` filter
at FILE level. Memory is O(grid x bins): counts are UInt32.
"""
function build_climatology(
    historical_paths::Vector{String},
    grid_points ::Matrix{Tuple{Float64,Float64}},
    interp      ::GridInterpolator,
    ocean_mask  ::Matrix{Bool};
    n_ws::Int = 18, ws_max::Float32 = 30f0, n_dir::Int = 24,
    months::Union{Nothing,Vector{Int}} = nothing,
    month_of_path::Union{Nothing,Vector{Int}} = nothing,
    load_field::Function = _load_uv_for_climatology_grib
)
    n_rows, n_cols = size(grid_points)
    ws_edges  = Float32.(range(0f0, ws_max, length = n_ws + 1))
    dir_edges = Float32.(range(0f0, 360f0, length = n_dir + 1))

    counts = zeros(UInt32, n_ws, n_dir, n_rows, n_cols)
    n_obs  = zeros(Int, n_rows, n_cols)

    n_flat = n_rows * n_cols
    u_buf  = Vector{Float32}(undef, n_flat)
    v_buf  = Vector{Float32}(undef, n_flat)

    n_used = 0
    skipped_paths = String[]
    for (k, path) in enumerate(historical_paths)
        if months !== nothing
            month_of_path === nothing && error("months filter given but month_of_path not supplied")
            month_of_path[k] in months || continue
        end

        local fields
        try
            fields = load_field(path)
        catch e
            push!(skipped_paths, path)
            @warn "Skipping unreadable/malformed climatology snapshot file" path = path exception = e
            continue
        end

        for field in fields
            precompute_uv_grid!(u_buf, v_buf, field, interp)
            n_used += 1
            @inbounds for j in 1:n_cols, i in 1:n_rows
                ocean_mask[i, j] || continue
                f = (j - 1) * n_rows + i
                u = u_buf[f]; v = v_buf[f]
                ws = sqrt(u^2 + v^2)
                ws >= ws_max && (ws = prevfloat(ws_max))          # clamp into last bin
                wdir = mod(rad2deg(atan(u, v)) + 180f0, 360f0)    # FROM direction
                counts[_bin_index(ws, ws_edges), _bin_index(Float32(wdir), dir_edges), i, j] += 0x00000001
                n_obs[i, j] += 1
            end
        end

        k % 200 == 0 && println("  climatology: $k / $(length(historical_paths)) files scanned " *
                                "($n_used snapshots used so far)")
    end
    if !isempty(skipped_paths)
        println("WARNING: skipped $(length(skipped_paths)) / $(length(historical_paths)) file(s). First few:")
        foreach(p -> println("    ", p), first(skipped_paths, 5))
        println("  Pass an explicit historical_paths list if the directory holds non-snapshot files.")
    end
    println("Climatology built from $n_used snapshots" *
            (months === nothing ? " (all-year)" : " (months=$months)"))

    roses = Matrix{WindRose}(undef, n_rows, n_cols)
    @inbounds for j in 1:n_cols, i in 1:n_rows
        if !ocean_mask[i, j] || n_obs[i, j] == 0
            roses[i, j] = WindRose(ws_edges, dir_edges, zeros(Float32, n_ws, n_dir), 0)
        else
            p = Float32.(counts[:, :, i, j]) ./ Float32(n_obs[i, j])
            roses[i, j] = WindRose(ws_edges, dir_edges, p, n_obs[i, j])
        end
    end
    return Climatology(roses, n_rows, n_cols, months)
end

# =============================================================================
# 3. Wind rose drawing (stacked polar wedges; 0 deg = north = up, clockwise)
# =============================================================================
function draw_wind_rose!(ax, rose::WindRose)
    GLMakie.empty!(ax)
    if rose.n_obs == 0
        GLMakie.text!(ax, [0.0], [0.0]; text = ["no data\n(land / uncovered cell)"],
                      align = (:center, :center), fontsize = 14)
        return
    end
    n_ws, n_dir = size(rose.probs)
    speed_colors = GLMakie.cgrad(:viridis, n_ws, categorical = true)

    for di in 1:n_dir
        θ0 = deg2rad(90 - rose.dir_edges[di])        # compass bearing -> math angle
        θ1 = deg2rad(90 - rose.dir_edges[di+1])
        r_cum = 0.0
        for wi in 1:n_ws
            p = rose.probs[wi, di]
            p <= 0f0 && continue
            r0 = r_cum; r1 = r_cum + p
            θs = range(θ0, θ1, length = 6)
            outer = [GLMakie.Point2f(r1 * cos(θ), r1 * sin(θ)) for θ in θs]
            inner = [GLMakie.Point2f(r0 * cos(θ), r0 * sin(θ)) for θ in reverse(θs)]
            GLMakie.poly!(ax, vcat(outer, inner); color = speed_colors[wi], strokewidth = 0.3,
                          strokecolor = (:black, 0.15))
            r_cum = r1
        end
    end
    max_r = maximum(sum(rose.probs, dims = 1))
    for frac in 0.1:0.1:ceil(max_r * 10) / 10
        θs = range(0, 2π, length = 60)
        GLMakie.lines!(ax, [frac * cos(θ) for θ in θs], [frac * sin(θ) for θ in θs];
                       color = (:gray40, 0.3), linewidth = 0.5)
    end
    GLMakie.text!(ax, [0.0],  [1.05max_r];  text = ["N"], align = (:center, :bottom), fontsize = 12)
    GLMakie.text!(ax, [0.0],  [-1.05max_r]; text = ["S"], align = (:center, :top),    fontsize = 12)
    GLMakie.text!(ax, [1.05max_r],  [0.0];  text = ["E"], align = (:left, :center),   fontsize = 12)
    GLMakie.text!(ax, [-1.05max_r], [0.0];  text = ["W"], align = (:right, :center),  fontsize = 12)
end

# =============================================================================
# 4. Expected edge cost from a wind rose
#
# Expectation is CONDITIONED on feasible bins (boat speed >= min_speed): an edge
# whose feasible probability mass is >= reliability_threshold is priced as if the
# infeasible remainder never happens (optimistic by design); below the threshold
# the edge is Inf.
# =============================================================================

"""Returns (expected_time_seconds, feasible_probability_mass) for ONE power level."""
function expected_edge_time(
    rose::WindRose, bearing::Float64, dist_m::Float64,
    slut::StackedLUT, power_idx::Int, min_speed::Float64;
    reliability_threshold::Float64 = 0.9
)
    rose.n_obs == 0 && return (Inf, 0.0)

    n_ws, n_dir = size(rose.probs)
    bearing32 = Float32(bearing)
    speed_scratch = Vector{Float32}(undef, slut.n_power)

    exp_time_feas = 0.0
    feas_p = 0.0
    @inbounds for di in 1:n_dir
        dmid = 0.5f0 * (rose.dir_edges[di] + rose.dir_edges[di+1])
        twa  = abs(mod(dmid - bearing32 + 180f0, 360f0) - 180f0)    # depends on di only
        for wi in 1:n_ws
            p = rose.probs[wi, di]
            p <= 0f0 && continue
            wmid = 0.5f0 * (rose.ws_edges[wi] + rose.ws_edges[wi+1])
            stacked_lookup!(speed_scratch, slut, wmid, twa)
            vboat = Float64(speed_scratch[power_idx])
            if vboat >= min_speed
                exp_time_feas += Float64(p) * (dist_m / vboat)
                feas_p += Float64(p)
            end
        end
    end
    return (feas_p >= reliability_threshold && feas_p > 0.0) ? (exp_time_feas / feas_p, feas_p) :
                                                                (Inf, feas_p)
end

"""Returns (exp_time[k] s, exp_energy[k] kWh, feasible_mass[k]) for every power level k."""
function expected_edge_costs_all_powers(
    rose::WindRose, bearing::Float64, dist_m::Float64,
    slut::StackedLUT, plevels_f::Vector{Float64}, min_speed::Float64;
    reliability_threshold::Float64 = 0.9
)
    n_power = slut.n_power
    exp_time_feas   = zeros(Float64, n_power)
    exp_energy_feas = zeros(Float64, n_power)
    feas_p          = zeros(Float64, n_power)
    rose.n_obs == 0 && return (fill(Inf, n_power), fill(Inf, n_power), feas_p)

    n_ws, n_dir = size(rose.probs)
    bearing32 = Float32(bearing)
    speed_scratch = Vector{Float32}(undef, n_power)

    @inbounds for di in 1:n_dir
        dmid = 0.5f0 * (rose.dir_edges[di] + rose.dir_edges[di+1])
        twa  = abs(mod(dmid - bearing32 + 180f0, 360f0) - 180f0)
        for wi in 1:n_ws
            p = rose.probs[wi, di]
            p <= 0f0 && continue
            wmid = 0.5f0 * (rose.ws_edges[wi] + rose.ws_edges[wi+1])
            stacked_lookup!(speed_scratch, slut, wmid, twa)
            for k in 1:n_power
                vboat = Float64(speed_scratch[k])
                if vboat >= min_speed
                    t = dist_m / vboat
                    exp_time_feas[k]   += Float64(p) * t
                    exp_energy_feas[k] += Float64(p) * plevels_f[k] * t / 3600.0
                    feas_p[k]          += Float64(p)
                end
            end
        end
    end

    exp_time   = Vector{Float64}(undef, n_power)
    exp_energy = Vector{Float64}(undef, n_power)
    @inbounds for k in 1:n_power
        if feas_p[k] >= reliability_threshold && feas_p[k] > 0.0
            exp_time[k]   = exp_time_feas[k]   / feas_p[k]
            exp_energy[k] = exp_energy_feas[k] / feas_p[k]
        else
            exp_time[k] = Inf; exp_energy[k] = Inf
        end
    end
    return (exp_time, exp_energy, feas_p)
end

# =============================================================================
# 5. Fixed-power backward Dijkstra (2D)
# =============================================================================

"""
    climatology_backward_dp(grid_points, ocean_mask, nb_flat, climatology, slut,
                            power_idx, min_speed, goal_idx; reliability_threshold, verbose)
        -> (V, next_hop)

V[i,j]        expected time to goal in SECONDS (Inf if unreachable); /3600 for hours.
next_hop[i,j] the cell to move to next; (0,0) at the goal / unreached cells.

Edge cost i->n depends only on the climatology at i, so this is an ordinary
static single-destination shortest-path problem (no epochs, no wind pool).
"""
function climatology_backward_dp(
    grid_points::Matrix{Tuple{Float64,Float64}}, ocean_mask::Matrix{Bool},
    nb_flat::NbCache, climatology::Climatology, slut::StackedLUT,
    power_idx::Int, min_speed::Float64, goal_idx::Tuple{Int,Int};
    reliability_threshold::Float64 = 0.9, verbose::Bool = true
)
    n_rows, n_cols = size(grid_points)

    rev_adj = Matrix{Vector{Tuple{Int,Int,Float64}}}(undef, n_rows, n_cols)
    for i in 1:n_rows, j in 1:n_cols
        rev_adj[i, j] = Tuple{Int,Int,Float64}[]
    end

    n_edges = 0
    t0 = time()
    for i in 1:n_rows, j in 1:n_cols
        ocean_mask[i, j] || continue
        rose = climatology.roses[i, j]
        rose.n_obs == 0 && continue
        for (ni, nj, dist, bearing, _arc, _ring) in nb_flat[i, j]
            ocean_mask[ni, nj] || continue
            t_exp, _ = expected_edge_time(rose, bearing, dist, slut, power_idx, min_speed;
                                          reliability_threshold)
            isinf(t_exp) && continue
            push!(rev_adj[ni, nj], (i, j, t_exp))
            n_edges += 1
        end
    end
    verbose && println("Reverse adjacency built: $n_edges edges in $(round(time()-t0, digits=1))s")

    V        = fill(Inf, n_rows, n_cols)
    next_hop = fill((0, 0), n_rows, n_cols)
    visited  = falses(n_rows, n_cols)

    pq = PriorityQueue{Tuple{Int,Int}, Float64}()
    V[goal_idx...] = 0.0
    pq[goal_idx] = 0.0

    n_settled = 0
    while !isempty(pq)
        u, du = peek(pq)
        dequeue!(pq)
        visited[u...] && continue
        visited[u...] = true
        n_settled += 1

        for (p1, p2, cost) in rev_adj[u...]
            visited[p1, p2] && continue
            cand = du + cost
            if cand < V[p1, p2]
                V[p1, p2]        = cand
                next_hop[p1, p2] = u
                pq[(p1, p2)]     = cand
            end
        end
    end
    verbose && println("Backward DP settled $n_settled / $(count(ocean_mask)) ocean cells (V in seconds)")
    return V, next_hop
end

# =============================================================================
# 6. Climatology blend: per-edge expected cost cache + global blend hook
# =============================================================================

"""Expected (time s, energy kWh) per power level for one edge. Fixed-size, so it is isbits."""
struct ClimEdgeCost{N}
    t_exp::NTuple{N,Float64}
    e_exp::NTuple{N,Float64}
end

"""
    build_climatology_edge_cache(nb_flat, ocean_mask, climatology, slut, plevels_f, min_speed; ...)

Expected cost for EVERY edge in `nb_flat`, indexed identically (cache[i,j][k] <->
nb_flat[i,j][k]); the solver zips by position. `nb_flat`, power levels, `min_speed`
and `reliability_threshold` must match what the consuming solves use. Use
`climatology_cache_meta` + `save_climatology_blend_cache` so a stale cache is caught.
"""
function build_climatology_edge_cache(
    nb_flat::NbCache, ocean_mask::Matrix{Bool}, climatology::Climatology, slut::StackedLUT,
    plevels_f::Vector{Float64}, min_speed::Float64;
    reliability_threshold::Float64 = 0.9, verbose::Bool = true
)
    return _build_edge_cache(Val(length(plevels_f)), nb_flat, ocean_mask, climatology, slut,
                             plevels_f, min_speed, reliability_threshold, verbose)
end

function _build_edge_cache(::Val{N}, nb_flat, ocean_mask, climatology, slut, plevels_f,
                           min_speed, reliability_threshold, verbose) where {N}
    n_rows, n_cols = size(nb_flat)
    cache = Matrix{Vector{ClimEdgeCost{N}}}(undef, n_rows, n_cols)
    inf_edge = ClimEdgeCost{N}(ntuple(_ -> Inf, N), ntuple(_ -> Inf, N))

    t0 = time(); n_edges = 0
    for i in 1:n_rows, j in 1:n_cols
        buf = nb_flat[i, j]
        col = Vector{ClimEdgeCost{N}}(undef, length(buf))
        cache[i, j] = col
        (!ocean_mask[i, j] || isempty(buf)) && continue
        rose = climatology.roses[i, j]
        for (k, (_ni, _nj, dist, bearing, _arc, _ring)) in enumerate(buf)
            if rose.n_obs == 0
                col[k] = inf_edge
                continue
            end
            t_k, e_k, _ = expected_edge_costs_all_powers(rose, bearing, dist, slut, plevels_f,
                                                         min_speed; reliability_threshold)
            col[k] = ClimEdgeCost{N}(NTuple{N,Float64}(t_k), NTuple{N,Float64}(e_k))
            n_edges += 1
        end
    end
    verbose && println("Climatology edge cache built: $n_edges edges in $(round(time()-t0, digits=1))s")
    return cache
end

# ---- cache metadata + I/O ------------------------------------------------------

"""Fingerprint of everything a cached edge table depends on."""
function climatology_cache_meta(nb_flat::NbCache, power_levels::Vector{Float64},
                                min_speed::Float64, reliability_threshold::Float64)
    return (grid_size = size(nb_flat),
            n_edges   = sum(length(c) for c in nb_flat),
            rmax      = Int(maximum((e[6] for c in nb_flat for e in c); init = Int8(0))),
            power_levels = Float64.(sort(power_levels)),
            min_speed = min_speed,
            reliability_threshold = reliability_threshold)
end

function save_climatology_blend_cache(path::String, climatology::Climatology, clim_edge_cache, meta::NamedTuple)
    jldopen(path, "w") do f
        f["roses"] = climatology.roses; f["months"] = climatology.months
        f["clim_edge_cache"] = clim_edge_cache
        f["meta"] = meta
    end
    println("Saved climatology blend cache to $path")
end

"""Load a cache written by save_climatology_blend_cache; errors if `expect` differs from the stored meta."""
function load_climatology_blend_cache(path::String, expect::NamedTuple)
    jldopen(path, "r") do f
        haskey(f, "meta") ||
            error("$path has no metadata (written by an older version) -- delete it and rebuild.")
        meta = f["meta"]
        bad = [string(k, ": cache=", repr(get(meta, k, missing)), " expected=", repr(expect[k]))
               for k in keys(expect) if get(meta, k, missing) != expect[k]]
        isempty(bad) ||
            error("Stale climatology cache $path -- delete it and rebuild:\n  " * join(bad, "\n  "))
        roses = f["roses"]
        n_rows, n_cols = size(roses)
        return (climatology = Climatology(roses, n_rows, n_cols, f["months"]),
                clim_edge_cache = f["clim_edge_cache"])
    end
end

const _CLIMATOLOGY_BLEND = Ref{Union{Nothing,NamedTuple}}(nothing)

"""
    set_climatology_blend!(; climatology, clim_edge_cache, full_forecast_days = 14.0)

Enables lead-time cost blending in findDijkstraRoutefinal (and everything built on
it). GLOBAL state: always pair with clear_climatology_blend!() (reset_sources! does).
The cache must come from build_climatology_edge_cache for the SAME grid/Rmax/heading set.
"""
function set_climatology_blend!(; climatology::Climatology, clim_edge_cache::AbstractMatrix,
                                  full_forecast_days::Float64 = 14.0)
    _CLIMATOLOGY_BLEND[] = (climatology = climatology, clim_edge_cache = clim_edge_cache,
                            full_forecast_days = full_forecast_days)
    return nothing
end
clear_climatology_blend!() = (_CLIMATOLOGY_BLEND[] = nothing; nothing)

# =============================================================================
# 7. Energy-aware backward DP (3D): state = (i, j, remaining-budget bin)
# =============================================================================

"""
Expected time/energy between ANY two grid nodes (not necessarily neighbours), using
the rose at `start_idx` and the direct bearing/distance to `end_idx`. Diagnostic.
"""
function average_edge_cost(
    climatology::Climatology, slut::StackedLUT, grid_points::Matrix{Tuple{Float64,Float64}},
    start_idx::Tuple{Int,Int}, end_idx::Tuple{Int,Int}, power_idx::Int, min_speed::Float64;
    reliability_threshold::Float64 = 0.9, verbose::Bool = true
)
    rose = climatology.roses[start_idx...]
    lon1, lat1 = grid_points[start_idx...]
    lon2, lat2 = grid_points[end_idx...]
    dist    = getDistance(lon1, lat1, lon2, lat2)
    bearing = getBoatAngle(lon1, lat1, lon2, lat2)
    t_exp, feas = expected_edge_time(rose, bearing, dist, slut, power_idx, min_speed;
                                     reliability_threshold)
    verbose && println("$(start_idx) -> $(end_idx): $(round(dist/1000, digits=1)) km @ bearing " *
                       "$(round(bearing, digits=1))deg -> expected $(round(t_exp/3600, digits=2)) h " *
                       "(feasible mass $(round(100*feas, digits=1))%)")
    return t_exp, feas
end

function _build_energy_reverse_adjacency(
    grid_points::Matrix{Tuple{Float64,Float64}}, ocean_mask::Matrix{Bool}, nb_flat::NbCache,
    climatology::Climatology, slut::StackedLUT, plevels_f::Vector{Float64},
    min_speed::Float64, bin_size::Float64;
    reliability_threshold::Float64 = 0.9, verbose::Bool = true
)
    n_rows, n_cols = size(grid_points)
    n_power = length(plevels_f)
    rev_adj = Matrix{Vector{Tuple{Int,Int,Int8,Float64,Int32,Float64}}}(undef, n_rows, n_cols)
    for i in 1:n_rows, j in 1:n_cols
        rev_adj[i, j] = Tuple{Int,Int,Int8,Float64,Int32,Float64}[]
    end

    n_edges = 0
    t0 = time()
    for i in 1:n_rows, j in 1:n_cols
        ocean_mask[i, j] || continue
        rose = climatology.roses[i, j]
        rose.n_obs == 0 && continue
        for (ni, nj, dist, bearing, _arc, _ring) in nb_flat[i, j]
            ocean_mask[ni, nj] || continue
            t_k, e_k, _ = expected_edge_costs_all_powers(rose, bearing, dist, slut, plevels_f,
                                                         min_speed; reliability_threshold)
            for k in 1:n_power
                isinf(t_k[k]) && continue
                # ceil: the budget is never undercounted (round() let sub-bin edges ride free
                # and drifted over hundreds of legs). Keep bin_size well below per-edge energy.
                ebin = ceil(Int32, e_k[k] / bin_size)
                push!(rev_adj[ni, nj], (i, j, Int8(k), t_k[k], ebin, e_k[k]))
                n_edges += 1
            end
        end
    end
    verbose && println("Energy-aware reverse adjacency: $n_edges (edge,power) entries in $(round(time()-t0, digits=1))s")
    return rev_adj
end

"""
    climatology_backward_dp_energy(grid_points, ocean_mask, nb_flat, climatology, slut,
                                   power_levels, min_speed, goal_idx, max_energy, bin_size; ...)
        -> (V, E, next_hop, power_levels_sorted)

V[i,j,b]        expected seconds to goal from (i,j) with remaining budget bin b (b = num_bins ~ full).
E[i,j,b]        summed expected energy along that route (kWh, un-binned).
next_hop[i,j,b] (ni, nj, new_bin, power_idx) -- power_idx indexes power_levels_sorted.
"""
function climatology_backward_dp_energy(
    grid_points::Matrix{Tuple{Float64,Float64}}, ocean_mask::Matrix{Bool}, nb_flat::NbCache,
    climatology::Climatology, slut::StackedLUT, power_levels::Vector{Float64},
    min_speed::Float64, goal_idx::Tuple{Int,Int}, max_energy::Float64, bin_size::Float64;
    reliability_threshold::Float64 = 0.9,
    bq_bucket_width::Float64 = 600.0, bq_n_buckets::Int = 50_000, verbose::Bool = true
)
    n_rows, n_cols = size(grid_points)
    power_levels_sorted = sort(power_levels)
    plevels_f = Float64.(power_levels_sorted)
    num_bins  = ceil(Int, max_energy / bin_size) + 1

    n_ocean = count(ocean_mask)
    verbose && println("Energy-aware DP: $n_ocean ocean cells x $num_bins budget bins " *
                       "= ~$(round(n_ocean * num_bins / 1e6, digits=1))M states")

    rev_adj = _build_energy_reverse_adjacency(grid_points, ocean_mask, nb_flat, climatology, slut,
                                              plevels_f, min_speed, bin_size;
                                              reliability_threshold, verbose)

    V        = fill(Inf, n_rows, n_cols, num_bins)
    E        = fill(Inf, n_rows, n_cols, num_bins)
    next_hop = fill((0, 0, 0, Int8(0)), n_rows, n_cols, num_bins)

    pq = BucketQueue2(; bucket_width = bq_bucket_width, n_buckets = bq_n_buckets)
    gi, gj = goal_idx
    for b in 1:num_bins
        V[gi, gj, b] = 0.0; E[gi, gj, b] = 0.0
        bq_push!(pq, 0.0, Int32(gi), Int32(gj), Int32(b))
    end

    n_settled = 0
    t0 = time()
    while !bq_isempty(pq)
        cur_cost, i32, j32, b32 = bq_pop!(pq)
        i = Int(i32); j = Int(j32); b = Int(b32)
        cur_cost > V[i, j, b] && continue
        n_settled += 1

        for (pi, pj, power_idx, t, ebd, e_val) in rev_adj[i, j]
            pred_b = b + ebd
            pred_b > num_bins && continue
            cand = cur_cost + t
            if cand < V[pi, pj, pred_b]
                V[pi, pj, pred_b]        = cand
                E[pi, pj, pred_b]        = E[i, j, b] + e_val
                next_hop[pi, pj, pred_b] = (i, j, b, Int8(power_idx))
                bq_push!(pq, cand, Int32(pi), Int32(pj), Int32(pred_b))
            end
        end

        verbose && n_settled % 500_000 == 0 &&
            println("  settled $n_settled states in $(round(time()-t0, digits=1))s")
    end
    verbose && println("Energy-aware backward DP settled $n_settled states in $(round(time()-t0, digits=1))s " *
                       "(re-relaxation means this can exceed the number of distinct states)")
    return V, E, next_hop, power_levels_sorted
end

"""
    reconstruct_climatology_route_energy(next_hop, start_idx, goal_idx, num_bins)
        -> (route_idxs::Vector{(i,j,bin)}, powers_used::Vector{Int8})

Walks the 3D next_hop from (start, full budget). `powers_used[k]` is the power-level index
(into power_levels_sorted) for the leg route_idxs[k] -> route_idxs[k+1]. These are exactly
the inputs launch_climatology_explorer expects.
"""
function reconstruct_climatology_route_energy(next_hop::Array{Tuple{Int,Int,Int,Int8},3},
                                              start_idx::Tuple{Int,Int}, goal_idx::Tuple{Int,Int},
                                              num_bins::Int)
    route  = Tuple{Int,Int,Int}[(start_idx[1], start_idx[2], num_bins)]
    powers = Int8[]
    cur = route[1]
    max_steps = length(next_hop) + 10
    while (cur[1], cur[2]) != goal_idx
        ni, nj, nb, p = next_hop[cur...]
        (ni == 0 && nj == 0) &&
            error("reconstruct_climatology_route_energy: start unreachable within the energy budget " *
                  "(next_hop hit (0,0) at $cur).")
        cur = (ni, nj, nb)
        push!(route, cur); push!(powers, p)
        length(route) > max_steps && error("reconstruct_climatology_route_energy: cycle in next_hop")
    end
    return route, powers
end

# =============================================================================
# 8. Great-circle reference, plots, save/load, drivers
# =============================================================================

@inline function _nearest_ocean_cell_with_data(
    climatology::Climatology, ocean_mask::Matrix{Bool}, i::Int, j::Int; max_radius::Int = 6
)
    n_rows, n_cols = size(ocean_mask)
    if 1 <= i <= n_rows && 1 <= j <= n_cols && ocean_mask[i, j] && climatology.roses[i, j].n_obs > 0
        return (i, j)
    end
    for r in 1:max_radius, di in -r:r, dj in -r:r
        max(abs(di), abs(dj)) == r || continue
        ni, nj = i + di, j + dj
        (1 <= ni <= n_rows && 1 <= nj <= n_cols) || continue
        ocean_mask[ni, nj] && climatology.roses[ni, nj].n_obs > 0 && return (ni, nj)
    end
    return nothing
end

"""
    great_circle_expected_time(...) -> seconds

Walks the great circle between start and goal (snapping samples to the nearest ocean cell
with data) and sums expected hop times at ONE power level. Returns Inf if any hop is
infeasible: dropping such hops would make the direct line look faster than it can be sailed.
Hops between snapped samples are not checked for crossing land or for exceeding Rmax.
"""
function great_circle_expected_time(
    grid_points::Matrix{Tuple{Float64,Float64}},
    latitudes::Vector{Float64}, longitudes::Vector{Float64},
    climatology::Climatology, ocean_mask::Matrix{Bool},
    slut::StackedLUT, power_idx::Int, min_speed::Float64,
    start_idx::Tuple{Int,Int}, goal_idx::Tuple{Int,Int};
    n_points::Int = 200, reliability_threshold::Float64 = 0.9
)
    lon1, lat1 = grid_points[start_idx...]
    lon2, lat2 = grid_points[goal_idx...]
    φ1, λ1 = deg2rad(lat1), deg2rad(lon1)
    φ2, λ2 = deg2rad(lat2), deg2rad(lon2)
    Δ = 2 * asin(clamp(sqrt(sin((φ2-φ1)/2)^2 + cos(φ1)*cos(φ2)*sin((λ2-λ1)/2)^2), 0.0, 1.0))

    total_time_s = 0.0
    n_snapped = 0; n_nodata = 0; n_infeasible = 0
    prev_idx = start_idx
    for f in range(0.0, 1.0, length = n_points)[2:end]
        if Δ < 1e-9
            lon, lat = lon2, lat2
        else
            A = sin((1-f)*Δ) / sin(Δ);  B = sin(f*Δ) / sin(Δ)
            x = A*cos(φ1)*cos(λ1) + B*cos(φ2)*cos(λ2)
            y = A*cos(φ1)*sin(λ1) + B*cos(φ2)*sin(λ2)
            z = A*sin(φ1) + B*sin(φ2)
            lat = rad2deg(atan(z, sqrt(x^2 + y^2)))
            lon = rad2deg(atan(y, x))
        end
        raw_idx = (nearest_index(latitudes, lat), nearest_index(longitudes, lon))
        idx = _nearest_ocean_cell_with_data(climatology, ocean_mask, raw_idx...)
        if idx === nothing
            n_nodata += 1
            continue
        end
        idx != raw_idx && (n_snapped += 1)
        idx == prev_idx && continue

        rose = climatology.roses[prev_idx...]
        lon_p, lat_p = grid_points[prev_idx...]
        lon_c, lat_c = grid_points[idx...]
        t_exp, _ = expected_edge_time(rose, getBoatAngle(lon_p, lat_p, lon_c, lat_c),
                                      getDistance(lon_p, lat_p, lon_c, lat_c),
                                      slut, power_idx, min_speed; reliability_threshold)
        isfinite(t_exp) ? (total_time_s += t_exp) : (n_infeasible += 1)
        prev_idx = idx
    end

    if n_snapped > 0 || n_nodata > 0 || n_infeasible > 0
        println("  great-circle reference: $n_snapped/$(n_points-1) sample(s) snapped to the nearest ocean cell, " *
                "$n_nodata skipped (no ocean data nearby), $n_infeasible infeasible hop(s)." *
                (n_snapped + n_nodata > (n_points-1) ÷ 4 ?
                 "  NOTE: large fraction -- the direct line likely crosses land." : ""))
    end
    return n_infeasible > 0 ? Inf : total_time_s
end

"""Expected hours-to-goal vs remaining energy budget at a fixed start, straight from the 3D V."""
function plot_time_vs_energy_budget(V::Array{Float64,3}, start_idx::Tuple{Int,Int}, bin_size::Float64)
    num_bins = size(V, 3)
    budgets_kwh = [(b - 1) * bin_size for b in 1:num_bins]
    times_h     = [V[start_idx..., b] / 3600 for b in 1:num_bins]

    fig = CairoMakie.Figure(size = (900, 550))
    ax  = CairoMakie.Axis(fig[1, 1]; title = "Expected time-to-goal vs energy budget",
                          xlabel = "remaining energy budget (kWh)", ylabel = "expected time (hours)")
    finite = isfinite.(times_h)
    CairoMakie.lines!(ax, budgets_kwh[finite], times_h[finite]; color = :steelblue, linewidth = 2.5)
    CairoMakie.scatter!(ax, budgets_kwh[finite], times_h[finite]; color = :steelblue, markersize = 5)
    return fig
end

"""Prints a live-solver vs climatology-optimal comparison for one forecast day."""
function compare_live_vs_climatology(
    path_live::Vector, node_costs_live::Vector, final_energy_live::Vector,
    route_idxs_clim::Vector{Tuple{Int,Int,Int}}, powers_clim::Vector{Int8},
    power_levels_sorted::Vector{Float64},
    V_clim::Array{Float64,3}, grid_points::Matrix{Tuple{Float64,Float64}},
    start_idx::Tuple{Int,Int}, num_bins::Int
)
    live_time_h = isempty(node_costs_live) ? NaN : node_costs_live[end] / 3600
    clim_time_h = isempty(route_idxs_clim) ? NaN : V_clim[start_idx..., num_bins] / 3600

    println("\n=== Live forecast solver vs. climatology-optimal (same power_levels) ===")
    println(rpad("", 20), rpad("Time (h)", 12), rpad("Waypoints", 12))
    println(rpad("Live (forecast)", 20), rpad(string(round(live_time_h, digits=1)), 12), rpad(string(length(path_live)), 12))
    println(rpad("Climatology", 20), rpad(string(round(clim_time_h, digits=1)), 12), rpad(string(length(route_idxs_clim)), 12))
    if !isnan(live_time_h) && !isnan(clim_time_h)
        gap_pct = 100 * (clim_time_h - live_time_h) / live_time_h
        println("\nLive route is $(round(abs(gap_pct), digits=1))% " * (gap_pct > 0 ? "faster" : "slower") *
                " than the climatology-optimal route.")
        println("(Large gap => the forecast had exploitable structure; small => the live route rediscovered the climatological corridor.)")
    end
end

"""Isochrone contours of V (seconds; must be 2D -- pass V[:,:,num_bins] for the energy-aware V)."""
function plot_climatology_isochrones(
    V::Matrix{Float64}, grid_points::Matrix{Tuple{Float64,Float64}}, ocean_mask::Matrix{Bool};
    band_days::AbstractVector{<:Real} = 1:12,
    start_idx::Union{Nothing,Tuple{Int,Int}} = nothing,
    goal_idx::Union{Nothing,Tuple{Int,Int}} = nothing,
    title::String = "Climatology isochrones (days to goal)"
)
    n_rows, n_cols = size(grid_points)
    lons = [grid_points[1, j][1] for j in 1:n_cols]
    lats = [grid_points[i, 1][2] for i in 1:n_rows]

    Vd_plot = V ./ 3600.0 ./ 24.0
    Vd_plot[.!ocean_mask]    .= NaN
    Vd_plot[isinf.(Vd_plot)] .= NaN

    fig = CairoMakie.Figure(size = (1300, 950))
    ax  = CairoMakie.Axis(fig[1, 1]; title = title, xlabel = "lon", ylabel = "lat")
    CairoMakie.heatmap!(ax, lons, lats, permutedims(Float32.(ocean_mask));
                        colormap = [RGB(0.82, 0.75, 0.55), RGB(0.93, 0.93, 0.93)],
                        colorrange = (0, 1), interpolate = false)

    lo_band, hi_band = extrema(band_days)
    try
        CairoMakie.contour!(ax, lons, lats, permutedims(Vd_plot); levels = collect(band_days),
                            colormap = :viridis, colorrange = (lo_band, hi_band),
                            linewidth = 1.8, labels = true, labelsize = 11)
    catch
        CairoMakie.contour!(ax, lons, lats, permutedims(Vd_plot); levels = collect(band_days),
                            colormap = :viridis, colorrange = (lo_band, hi_band), linewidth = 1.8)
    end
    CairoMakie.Colorbar(fig[1, 2]; colormap = :viridis, limits = (lo_band, hi_band),
                        label = "days to goal (isochrone band)")

    if start_idx !== nothing
        lon0, lat0 = grid_points[start_idx...]
        CairoMakie.scatter!(ax, [lon0], [lat0]; color = :lime, markersize = 14, marker = :diamond, label = "start")
    end
    if goal_idx !== nothing
        lon1, lat1 = grid_points[goal_idx...]
        CairoMakie.scatter!(ax, [lon1], [lat1]; color = :red, markersize = 14, marker = :diamond, label = "goal")
    end
    (start_idx !== nothing || goal_idx !== nothing) && CairoMakie.axislegend(ax; position = :lb)
    return fig
end

"""Saves everything needed to replot a fixed-power run (V is the whole answer; no single route)."""
function save_climatology_run(path::String, V::Matrix{Float64}, next_hop::Matrix{Tuple{Int,Int}},
                              climatology::Climatology,
                              grid_points::Matrix{Tuple{Float64,Float64}}, ocean_mask::Matrix{Bool},
                              start_idx::Tuple{Int,Int}, goal_idx::Tuple{Int,Int})
    jldopen(path, "w") do f
        f["V"] = V; f["next_hop"] = next_hop
        f["climatology_roses"] = climatology.roses; f["climatology_months"] = climatology.months
        f["grid_points"] = grid_points; f["ocean_mask"] = ocean_mask
        f["start_idx"] = start_idx; f["goal_idx"] = goal_idx
    end
    println("Saved climatology run to $path")
end

function load_climatology_run(path::String)
    local V, next_hop, roses, months, grid_points, ocean_mask, start_idx, goal_idx
    jldopen(path, "r") do f
        V = f["V"]; next_hop = f["next_hop"]
        roses = f["climatology_roses"]; months = f["climatology_months"]
        grid_points = f["grid_points"]; ocean_mask = f["ocean_mask"]
        start_idx = f["start_idx"]; goal_idx = f["goal_idx"]
    end
    n_rows, n_cols = size(grid_points)
    return (V = V, next_hop = next_hop, climatology = Climatology(roses, n_rows, n_cols, months),
            grid_points = grid_points, ocean_mask = ocean_mask, start_idx = start_idx, goal_idx = goal_idx)
end

# shared setup for the two run_climatology_isochrones* drivers
function _climatology_setup(; start_location, end_location, lon_step, lat_step, padding, Rmax,
                              heading_mode, historical_dir, historical_paths, months, month_of_path,
                              n_ws, ws_max, n_dir)
    grid_points, latitudes, longitudes, ocean_mask =
        _build_grid_and_mask(; start_location, end_location, lon_step, lat_step, padding)
    start_idx = (nearest_index(latitudes, start_location[2]), nearest_index(longitudes, start_location[1]))
    goal_idx  = (nearest_index(latitudes, end_location[2]),   nearest_index(longitudes, end_location[1]))
    _check_endpoints(ocean_mask, start_idx, goal_idx)
    n_rows, n_cols = size(grid_points)

    paths = historical_paths === nothing ?
        [joinpath(historical_dir, f) for f in readdir(historical_dir) if endswith(f, ".grib")] :
        historical_paths
    isempty(paths) && error("No historical .grib files found in $historical_dir.")
    println("Found $(length(paths)) historical snapshots")

    println("\n--- Building interpolator ---")
    interp = build_interpolator(grid_points, _reference_field(paths[1]))

    println("--- Building neighbour cache ---")
    t_nb = @elapsed nb_flat = build_neighbour_cache(grid_points, ocean_mask, n_rows, n_cols, Rmax,
                                                    27_800.0; heading_mode)
    println("  done in $(round(t_nb, digits=1))s")

    println("--- Building climatology ---")
    t_clim = @elapsed climatology = build_climatology(paths, grid_points, interp, ocean_mask;
                                                      n_ws, ws_max, n_dir, months, month_of_path)
    println("  done in $(round(t_clim, digits=1))s")

    return (grid_points = grid_points, latitudes = latitudes, longitudes = longitudes,
            ocean_mask = ocean_mask, start_idx = start_idx, goal_idx = goal_idx,
            nb_flat = nb_flat, climatology = climatology, n_paths = length(paths))
end

"""
Fixed-power climatology run: climatology -> backward DP -> isochrone plot. No route is
reconstructed; V[start_idx] (printed) is the expected time-to-goal. NOTE: `power_idx`
defaults to the HIGHEST level (2000 kW), unlike the 400 kW pipeline -- set it explicitly.
"""
function run_climatology_isochrones(;
    start_location   = (-94.0, 28.5),
    end_location     = (-5.5,  49.5),
    lon_step         = 0.25,
    lat_step         = 0.25,
    padding          = 20,
    Rmax             = 4,
    heading_mode     = :full,
    power_levels     = [0.0, 400.0, 1200.0, 2000.0],
    power_idx        = length(power_levels),
    min_speed        = 1.0,
    historical_dir   = _climatology_dir(),
    historical_paths = nothing,
    months           = nothing,
    month_of_path    = nothing,
    n_ws             = 18,
    ws_max           = 30f0,
    n_dir            = 24,
    reliability_threshold = 0.9,
    band_days        = 1:12,
    save_path        = "climatology_run.jld2",
    plot_path        = "climatology_isochrones.png"
)
    CairoMakie.activate!()
    S = _climatology_setup(; start_location, end_location, lon_step, lat_step, padding, Rmax,
                             heading_mode, historical_dir, historical_paths, months, month_of_path,
                             n_ws, ws_max, n_dir)

    power_levels_sorted = sort(power_levels)
    slut      = build_stacked_lut(power_levels_sorted)
    power_idx = clamp(power_idx, 1, length(power_levels_sorted))
    println("Routing at power level $(power_levels_sorted[power_idx]) kW (index $power_idx of $(length(power_levels_sorted)))")

    println("--- Backward DP from goal ---")
    t_dp = @elapsed V, next_hop = climatology_backward_dp(
        S.grid_points, S.ocean_mask, S.nb_flat, S.climatology, slut, power_idx, min_speed, S.goal_idx;
        reliability_threshold)
    println("  done in $(round(t_dp, digits=1))s")

    expected_time_h = V[S.start_idx...]
    if isinf(expected_time_h)
        println("Start is NOT reachable from the goal-side DP (no feasible climatological path).")
        expected_time_h = NaN
    else
        expected_time_h /= 3600
        println("Expected time-to-goal from start: $(round(expected_time_h, digits=1)) h ($(round(expected_time_h / 24, digits=1)) days)")
    end

    save_path !== nothing && save_climatology_run(save_path, V, next_hop, S.climatology, S.grid_points,
                                                  S.ocean_mask, S.start_idx, S.goal_idx)
    if plot_path !== nothing
        CairoMakie.save(plot_path, plot_climatology_isochrones(V, S.grid_points, S.ocean_mask;
                        band_days, start_idx = S.start_idx, goal_idx = S.goal_idx))
        println("Saved $plot_path")
    end

    return (V = V, next_hop = next_hop, climatology = S.climatology,
            grid_points = S.grid_points, ocean_mask = S.ocean_mask,
            start_idx = S.start_idx, goal_idx = S.goal_idx,
            expected_time_h = expected_time_h, n_snapshots = S.n_paths,
            Rmax = Rmax, power_idx = power_idx)
end

"""
Energy-aware run: remaining-budget dimension so power is chosen per edge. Outputs the
full-tank isochrone plot, the great-circle comparison and the time-vs-budget curve.
Choose `bin_size` well below typical per-edge energy (edge energy is rounded UP to whole bins).
"""
function run_climatology_isochrones_energy(;
    start_location   = (-94.0, 28.5),
    end_location     = (-5.5,  49.5),
    lon_step         = 0.25,
    lat_step         = 0.25,
    padding          = 20,
    Rmax             = 4,
    heading_mode     = :full,
    power_levels     = [0.0, 400.0, 1200.0, 2000.0],
    max_energy       = 150_000.0,
    bin_size         = 1000.0,
    min_speed        = 1.0,
    historical_dir   = _climatology_dir(),
    historical_paths = nothing,
    months           = nothing,
    month_of_path    = nothing,
    n_ws             = 18,
    ws_max           = 30f0,
    n_dir            = 24,
    reliability_threshold = 0.9,
    bq_bucket_width  = 600.0,
    bq_n_buckets     = 50_000,
    band_days        = 1:12,
    save_path        = "climatology_run_energy.jld2",
    plot_path        = "climatology_isochrones_energy.png",
    budget_plot_path = "climatology_time_vs_budget.png"
)
    CairoMakie.activate!()
    S = _climatology_setup(; start_location, end_location, lon_step, lat_step, padding, Rmax,
                             heading_mode, historical_dir, historical_paths, months, month_of_path,
                             n_ws, ws_max, n_dir)

    slut = build_stacked_lut(sort(power_levels))

    println("--- Energy-aware backward DP from goal ---")
    t_dp = @elapsed V, E, next_hop, power_levels_sorted = climatology_backward_dp_energy(
        S.grid_points, S.ocean_mask, S.nb_flat, S.climatology, slut, power_levels, min_speed, S.goal_idx,
        max_energy, bin_size; reliability_threshold, bq_bucket_width, bq_n_buckets)
    println("  done in $(round(t_dp, digits=1))s")

    num_bins = size(V, 3)
    expected_time_h = V[S.start_idx..., num_bins]
    if isinf(expected_time_h)
        println("Start is NOT reachable within max_energy=$max_energy.")
        expected_time_h = NaN
    else
        expected_time_h /= 3600
        println("Expected time-to-goal from start (full tank): $(round(expected_time_h, digits=1)) h ($(round(expected_time_h / 24, digits=1)) days)")
    end

    println("\n--- Great-circle reference ---")
    gc_time_h = great_circle_expected_time(
        S.grid_points, S.latitudes, S.longitudes, S.climatology, S.ocean_mask, slut,
        length(power_levels_sorted), min_speed, S.start_idx, S.goal_idx; reliability_threshold) / 3600
    if !isnan(expected_time_h) && isfinite(gc_time_h) && gc_time_h > 0
        detour_pct = 100 * (expected_time_h - gc_time_h) / gc_time_h
        println("Great circle (direct line, top power):  $(round(gc_time_h, digits=1)) h")
        println("Climatology-optimal (full tank):         $(round(expected_time_h, digits=1)) h")
        println("=> routing " * (detour_pct < 0 ? "SAVES" : "COSTS") * " $(round(abs(detour_pct), digits=1))% vs the direct line.")
    else
        println("Great circle is not sailable at top power (or not computable) -- see counts above.")
    end

    if save_path !== nothing
        jldopen(save_path, "w") do f
            f["V"] = V; f["next_hop"] = next_hop
            f["climatology_roses"] = S.climatology.roses; f["climatology_months"] = S.climatology.months
            f["power_levels_sorted"] = power_levels_sorted
            f["grid_points"] = S.grid_points; f["ocean_mask"] = S.ocean_mask
            f["start_idx"] = S.start_idx; f["goal_idx"] = S.goal_idx
            f["bin_size"] = bin_size; f["max_energy"] = max_energy; f["gc_time_h"] = gc_time_h
        end
        println("Saved energy-aware climatology run to $save_path")
    end
    if plot_path !== nothing
        CairoMakie.save(plot_path, plot_climatology_isochrones(V[:, :, num_bins], S.grid_points, S.ocean_mask;
                        band_days, start_idx = S.start_idx, goal_idx = S.goal_idx,
                        title = "Climatology isochrones (full energy budget)"))
        println("Saved $plot_path")
    end
    if budget_plot_path !== nothing
        CairoMakie.save(budget_plot_path, plot_time_vs_energy_budget(V, S.start_idx, bin_size))
        println("Saved $budget_plot_path")
    end

    return (V = V, E = E, next_hop = next_hop, power_levels_sorted = power_levels_sorted,
            climatology = S.climatology, slut = slut,
            grid_points = S.grid_points, ocean_mask = S.ocean_mask,
            latitudes = S.latitudes, longitudes = S.longitudes,
            start_idx = S.start_idx, goal_idx = S.goal_idx,
            expected_time_h = expected_time_h, gc_time_h = gc_time_h,
            bin_size = bin_size, max_energy = max_energy, num_bins = num_bins)
end

function load_climatology_run_energy(path::String)
    local V, next_hop, roses, months, power_levels_sorted
    local grid_points, ocean_mask, start_idx, goal_idx, bin_size, max_energy, gc_time_h
    jldopen(path, "r") do f
        V = f["V"]; next_hop = f["next_hop"]
        roses = f["climatology_roses"]; months = f["climatology_months"]
        power_levels_sorted = f["power_levels_sorted"]
        grid_points = f["grid_points"]; ocean_mask = f["ocean_mask"]
        start_idx = f["start_idx"]; goal_idx = f["goal_idx"]
        bin_size = f["bin_size"]; max_energy = f["max_energy"]; gc_time_h = f["gc_time_h"]
    end
    n_rows, n_cols = size(grid_points)
    return (V = V, next_hop = next_hop, power_levels_sorted = power_levels_sorted,
            climatology = Climatology(roses, n_rows, n_cols, months),
            grid_points = grid_points, ocean_mask = ocean_mask,
            start_idx = start_idx, goal_idx = goal_idx,
            bin_size = bin_size, max_energy = max_energy, gc_time_h = gc_time_h)
end

# =============================================================================
# 9. Interactive explorers
# =============================================================================

# ---------------------------------------------------------------------------
# Per-leg route data for the energy-aware route, precomputed so scrubbing is a lookup.
# Leg time is read off V by Bellman consistency (V[pred] - V[succ]) rather than
# recomputed, so it always matches what the DP solved.
# ---------------------------------------------------------------------------
struct ClimLegInfo
    i::Int; j::Int; bin::Int
    lon::Float64; lat::Float64
    cum_time_s::Float64
    cum_dist_km::Float64
    leg_dist_km::Float64
    leg_time_s::Float64
    leg_speed_kn::Float64
    power_kw::Float64
end

const KM_PER_NM_CLIM = 1.852

function build_climatology_legs(
    route_idxs::Vector{Tuple{Int,Int,Int}}, powers_used::Vector{Int8},
    power_levels_sorted::Vector{Float64}, grid_points::Matrix{Tuple{Float64,Float64}},
    V::Array{Float64,3}
)
    n = length(route_idxs)
    legs = Vector{ClimLegInfo}(undef, n)
    cum_dist = 0.0
    V_start = V[route_idxs[1]...]
    for k in 1:n
        i, j, b = route_idxs[k]
        lon, lat = grid_points[i, j]
        cum_time = V_start - V[i, j, b]
        if k == 1
            leg_dist = leg_t = leg_spd = pow = 0.0
        else
            lon0, lat0 = grid_points[route_idxs[k-1][1], route_idxs[k-1][2]]
            leg_dist = getDistance(lon0, lat0, lon, lat) / 1000.0
            leg_t    = V[route_idxs[k-1]...] - V[i, j, b]
            leg_spd  = leg_t > 0 ? (leg_dist / KM_PER_NM_CLIM) / (leg_t / 3600) : 0.0
            pow      = power_levels_sorted[Int(powers_used[k-1])]
        end
        cum_dist += leg_dist
        legs[k] = ClimLegInfo(i, j, b, lon, lat, cum_time, cum_dist, leg_dist, leg_t, leg_spd, pow)
    end
    return legs
end

# State of a live-solver route at an arbitrary elapsed time (linear interpolation between
# its bracketing steps; clamped outside its time range).
function dijkstra_state_at_time(rs::RouteStepper2, t_s::Float64)
    steps = rs.steps
    n = length(steps)
    if n == 1 || t_s <= steps[1].cum_time_s
        s = steps[1]
        return (dist_km = s.cum_dist_km, energy_kwh = s.cum_energy_kwh,
                speed_kn = 0.0, power_kw = 0.0, clamped = :start)
    end
    if t_s >= steps[end].cum_time_s
        s = steps[end]
        return (dist_km = s.cum_dist_km, energy_kwh = s.cum_energy_kwh,
                speed_kn = s.leg_speed_kn, power_kw = s.leg_power_kw, clamped = :end)
    end
    times = [s.cum_time_s for s in steps]
    k = clamp(searchsortedfirst(times, t_s), 2, n)
    s0, s1 = steps[k-1], steps[k]
    dt = s1.cum_time_s - s0.cum_time_s
    f  = dt > 0 ? (t_s - s0.cum_time_s) / dt : 0.0
    return (dist_km = s0.cum_dist_km + f * (s1.cum_dist_km - s0.cum_dist_km),
            energy_kwh = s0.cum_energy_kwh + f * (s1.cum_energy_kwh - s0.cum_energy_kwh),
            speed_kn = s1.leg_speed_kn, power_kw = s1.leg_power_kw, clamped = :mid)
end

# Click handler shared by both explorers: map click -> nearest ocean cell.
function _on_map_click!(f::Function, ax_map, lons_1d, lats_1d, ocean_mask)
    GLMakie.on(GLMakie.events(ax_map.scene).mousebutton) do event
        (event.button == GLMakie.Mouse.left && event.action == GLMakie.Mouse.press) || return
        GLMakie.is_mouseinside(ax_map.scene) || return
        pos = GLMakie.mouseposition(ax_map.scene)
        lon, lat = pos[1], pos[2]
        (minimum(lons_1d) <= lon <= maximum(lons_1d) && minimum(lats_1d) <= lat <= maximum(lats_1d)) || return
        i = nearest_index(lats_1d, lat); j = nearest_index(lons_1d, lon)
        ocean_mask[i, j] || return
        f((i, j))
    end
end

"""
    launch_climatology_explorer(V, next_hop, route_idxs, powers_used, power_levels_sorted,
                                climatology, grid_points, ocean_mask, start_idx, goal_idx,
                                bin_size; dijkstra_route = nothing)

Interactive explorer for an energy-aware climatology run. Get `route_idxs`/`powers_used`
from `reconstruct_climatology_route_energy(next_hop, start_idx, goal_idx, size(V,3))`.

Map: land/ocean, optional V heatmap, route coloured by power level. Click any ocean cell to
see its wind rose and its time-vs-budget curve. Bottom: step through the route leg by leg.
With `dijkstra_route` (a RouteStepper2 from build_stepper2) the live-solver route is overlaid
(coloured by power, toggleable) and a comparison panel shows full-route totals plus a live
"same elapsed time" readout (distance / energy / speed / power for both routes).

Two energy numbers are reported for the climatology route and are NOT interchangeable:
  budget-tracked  (bin_start - bin_now) * bin_size : what the DP checked against max_energy.
  Σ P·leg_time    : expected-time based; includes the feasibility conditioning of the rose.
"""
function launch_climatology_explorer(
    V::Array{Float64,3}, next_hop::Array{Tuple{Int,Int,Int,Int8},3},
    route_idxs::Vector{Tuple{Int,Int,Int}}, powers_used::Vector{Int8},
    power_levels_sorted::Vector{Float64}, climatology::Climatology,
    grid_points::Matrix{Tuple{Float64,Float64}}, ocean_mask::Matrix{Bool},
    start_idx::Tuple{Int,Int}, goal_idx::Tuple{Int,Int}, bin_size::Float64;
    dijkstra_route::Union{Nothing,RouteStepper2} = nothing
)   
    GLMakie.activate!()
    isempty(route_idxs) && error("launch_climatology_explorer: route_idxs is empty -- nothing to step through.")
    has_dijkstra = dijkstra_route !== nothing
    has_dijkstra && isempty(dijkstra_route.steps) &&
        error("launch_climatology_explorer: dijkstra_route has no steps.")

    legs = build_climatology_legs(route_idxs, powers_used, power_levels_sorted, grid_points, V)
    num_bins = size(V, 3)
    n_rows, n_cols = size(grid_points)
    lons_1d = [grid_points[1, j][1] for j in 1:n_cols]
    lats_1d = [grid_points[i, 1][2] for i in 1:n_rows]

    fig = GLMakie.Figure(size = (2000, 1420))
    GLMakie.Label(fig[0, 1:3], "Climatology Route Explorer"; fontsize = 22, font = :bold,
                  halign = :left, tellwidth = false)

    ax_map    = GLMakie.Axis(fig[1, 1]; title = "Route + climatology", xlabel = "lon", ylabel = "lat")
    ax_rose   = GLMakie.Axis(fig[1, 2]; title = "Wind rose (selected cell)", aspect = 1)
    GLMakie.hidedecorations!(ax_rose)
    ax_budget = GLMakie.Axis(fig[1, 3]; title = "Time vs remaining budget (selected cell)",
                             xlabel = "budget (kWh)", ylabel = "hours to goal")
    GLMakie.colsize!(fig.layout, 1, GLMakie.Relative(0.5))
    GLMakie.colsize!(fig.layout, 2, GLMakie.Relative(0.25))
    GLMakie.colsize!(fig.layout, 3, GLMakie.Relative(0.25))
    GLMakie.rowsize!(fig.layout, 3, GLMakie.Fixed(260))    # room for the multi-line stats text

    GLMakie.heatmap!(ax_map, lons_1d, lats_1d, permutedims(Float32.(ocean_mask));
                     colormap = [RGB(0.82, 0.75, 0.55), RGB(0.93, 0.93, 0.93)],
                     colorrange = (0, 1), interpolate = false)

    show_V = GLMakie.Observable(true)
    V_plot = V[:, :, num_bins] ./ 3600.0
    V_plot[.!ocean_mask] .= NaN
    V_plot[isinf.(V_plot)] .= NaN
    v_heatmap = GLMakie.heatmap!(ax_map, lons_1d, lats_1d, permutedims(V_plot);
                                 colormap = (:viridis, 0.4), visible = show_V)
    GLMakie.Colorbar(fig[2, 1], v_heatmap, label = "expected hours to goal (full tank)", vertical = false)

    n_power = length(power_levels_sorted)
    power_palette = GLMakie.cgrad(:thermal, max(n_power, 2), categorical = true)
    for k in 1:length(powers_used)
        GLMakie.lines!(ax_map, [legs[k].lon, legs[k+1].lon], [legs[k].lat, legs[k+1].lat];
                       color = power_palette[Int(powers_used[k])], linewidth = 3.0)
    end
    for p in sort(unique(powers_used))
        GLMakie.scatter!(ax_map, Float64[], Float64[]; color = power_palette[Int(p)],
                         markersize = 12, label = "$(power_levels_sorted[Int(p)]) kW")
    end
    GLMakie.scatter!(ax_map, [legs[1].lon],   [legs[1].lat];   color = :lime, markersize = 16, marker = :diamond, label = "climatology start")
    GLMakie.scatter!(ax_map, [legs[end].lon], [legs[end].lat]; color = :red,  markersize = 16, marker = :diamond, label = "climatology goal")

    # ---- optional live-solver overlay (dashed, coloured by power) ----
    dijkstra_visible     = GLMakie.Observable(has_dijkstra)
    dijkstra_line_ref    = Ref{Any}(nothing)
    dijkstra_markers_ref = Any[]

    # steps[1].leg_power_kw is 0 by construction (no incoming leg): borrow step 2's value
    dijkstra_power = has_dijkstra ? [s.leg_power_kw for s in dijkstra_route.steps] : Float64[]
    length(dijkstra_power) > 1 && (dijkstra_power[1] = dijkstra_power[2])
    dijkstra_power_range = if !isempty(dijkstra_power)
        lo, hi = extrema(dijkstra_power); hi > lo ? (lo, hi) : (lo - 1.0, hi + 1.0)
    else
        (0.0, 1.0)
    end
    has_dijkstra && GLMakie.Colorbar(fig[2, 2:3], limits = dijkstra_power_range, colormap = :thermal,
                                     label = "Dijkstra power along route (kW)", vertical = false)

    function draw_dijkstra_route!()
        dijkstra_line_ref[] === nothing || (delete!(ax_map, dijkstra_line_ref[]); dijkstra_line_ref[] = nothing)
        foreach(m -> delete!(ax_map, m), dijkstra_markers_ref); empty!(dijkstra_markers_ref)
        (has_dijkstra && dijkstra_visible[]) || return
        d_lons = [s.lon for s in dijkstra_route.steps]; d_lats = [s.lat for s in dijkstra_route.steps]
        dijkstra_line_ref[] = GLMakie.lines!(ax_map, d_lons, d_lats; color = dijkstra_power,
            colormap = :thermal, colorrange = dijkstra_power_range, linestyle = :dash, linewidth = 3.0,
            label = "Dijkstra (dashed, coloured by power)")
        push!(dijkstra_markers_ref,
              GLMakie.scatter!(ax_map, [d_lons[1]], [d_lats[1]]; color = :black, markersize = 11, marker = :rect, label = "Dijkstra start"),
              GLMakie.scatter!(ax_map, [d_lons[end]], [d_lats[end]]; color = :black, markersize = 11, marker = :utriangle, label = "Dijkstra goal"))
    end

    legend_ref = Ref{Any}(nothing)
    function refresh_legend!()
        legend_ref[] === nothing || (delete!(legend_ref[]); legend_ref[] = nothing)
        legend_ref[] = GLMakie.axislegend(ax_map; position = :lb, labelsize = 10)
    end

    current_leg = GLMakie.Observable(1)
    marker_pos  = GLMakie.Observable(GLMakie.Point2f(legs[1].lon, legs[1].lat))
    GLMakie.scatter!(ax_map, marker_pos; color = :magenta, markersize = 20, marker = :star5,
                     label = "climatology position")

    function refresh_selected!(idx::Tuple{Int,Int})
        rose = climatology.roses[idx...]
        draw_wind_rose!(ax_rose, rose)
        ax_rose.title = "Wind rose at ($(idx[1]),$(idx[2]))  --  n_obs=$(rose.n_obs)"
        GLMakie.empty!(ax_budget)
        budgets = [(b-1) * bin_size for b in 1:num_bins]
        times_h = [V[idx..., b] / 3600 for b in 1:num_bins]
        finite = isfinite.(times_h)
        if any(finite)
            GLMakie.lines!(ax_budget, budgets[finite], times_h[finite]; color = :steelblue, linewidth = 2)
            GLMakie.scatter!(ax_budget, budgets[finite], times_h[finite]; color = :steelblue, markersize = 4)
        else
            GLMakie.text!(ax_budget, [0.0], [0.0]; text = ["unreachable from here\n(within max budget)"],
                          align = (:center, :center))
        end
    end
    _on_map_click!(refresh_selected!, ax_map, lons_1d, lats_1d, ocean_mask)

    # ---- per-leg stats ----
    stats_text = GLMakie.Observable("")
    function update_stats!(k::Int)
        leg = legs[k]
        stats_text[] = @sprintf(
            "waypoint %d / %d\ncumulative time: %.1f h\ncumulative dist: %.1f km\nthis leg: %.1f km in %.0f s -> %.1f kn\npower this leg: %.0f kW\nnode: (%d,%d)  budget bin: %d",
            k, length(legs), leg.cum_time_s / 3600, leg.cum_dist_km,
            leg.leg_dist_km, leg.leg_time_s, leg.leg_speed_kn, leg.power_kw, leg.i, leg.j, leg.bin)
    end
    GLMakie.Label(fig[3, 1], stats_text; tellwidth = false, halign = :left, fontsize = 13, justification = :left)

    # ---- route comparison panel ----
    comparison_panel = fig[3, 2:3] = GLMakie.GridLayout(tellheight = false)
    GLMakie.Label(comparison_panel[1, 1:2], "ROUTE COMPARISON"; fontsize = 14, font = :bold, halign = :left)

    lc = legs[end]
    clim_energy_budget_kwh = (legs[1].bin - lc.bin) * bin_size
    clim_totals_text = @sprintf(
        "CLIMATOLOGY (full route)\ntotal time:   %.2f h\ntotal dist:   %.1f km\nenergy (budget-tracked): %.1f kWh  (bin %d→%d)\navg speed:    %.2f kn\nwaypoints:    %d",
        lc.cum_time_s / 3600, lc.cum_dist_km, clim_energy_budget_kwh, legs[1].bin, lc.bin,
        lc.cum_time_s > 0 ? (lc.cum_dist_km / KM_PER_NM_CLIM) / (lc.cum_time_s / 3600) : 0.0, length(legs))

    dijkstra_totals_text = if has_dijkstra
        ds = dijkstra_route.steps[end]
        @sprintf("DIJKSTRA (full route)\ntotal time:    %.2f h\ntotal dist:    %.1f km\ntotal energy:  %.1f kWh\navg speed:     %.2f kn\nwaypoints:     %d",
                 ds.cum_time_s / 3600, ds.cum_dist_km, ds.cum_energy_kwh,
                 ds.cum_time_s > 0 ? (ds.cum_dist_km / KM_PER_NM_CLIM) / (ds.cum_time_s / 3600) : 0.0,
                 length(dijkstra_route.steps))
    else
        "DIJKSTRA (full route)\nno dijkstra_route supplied --\npass one to compare"
    end
    GLMakie.Label(comparison_panel[2, 1], clim_totals_text;     tellwidth = false, halign = :left, valign = :top, fontsize = 12, justification = :left)
    GLMakie.Label(comparison_panel[2, 2], dijkstra_totals_text; tellwidth = false, halign = :left, valign = :top, fontsize = 12, justification = :left)
    GLMakie.colsize!(comparison_panel, 1, GLMakie.Relative(0.5))
    GLMakie.colsize!(comparison_panel, 2, GLMakie.Relative(0.5))

    matched_time_text = GLMakie.Observable(has_dijkstra ? "" : "(no Dijkstra route to match against)")
    GLMakie.Label(comparison_panel[3, 1:2], matched_time_text; tellwidth = false, halign = :left,
                  valign = :top, fontsize = 12, justification = :left)

    function update_comparison!(k::Int)
        has_dijkstra || return
        leg = legs[k]
        t_s = leg.cum_time_s
        d = dijkstra_state_at_time(dijkstra_route, t_s)
        clamp_note = d.clamped == :start ? "  (Dijkstra route not yet started)" :
                     d.clamped == :end   ? "  (Dijkstra route already finished)" : ""
        d_speed = d.clamped == :start ? "n/a" : @sprintf("%.2f kn", d.speed_kn)
        d_power = d.clamped == :start ? "n/a" : @sprintf("%.0f kW", d.power_kw)
        clim_e = (legs[1].bin - leg.bin) * bin_size       # exact bin-tracked energy so far
        matched_time_text[] = @sprintf(
            "AT SAME ELAPSED TIME (%.2f h)%s\n  climatology: %.1f km, %.1f kWh (budget-tracked), %.0f kW\n  dijkstra:    %.1f km, %.1f kWh, %s, %s\n  Δ (clim−dij): %.1f km, %.1f kWh",
            t_s / 3600, clamp_note, leg.cum_dist_km, clim_e, leg.power_kw,
            d.dist_km, d.energy_kwh, d_speed, d_power,
            leg.cum_dist_km - d.dist_km, clim_e - d.energy_kwh)
    end

    # ---- controls ----
    controls = fig[4, 1:3] = GLMakie.GridLayout()
    btn_prev = GLMakie.Button(controls[1, 1], label = "◀ Prev")
    btn_next = GLMakie.Button(controls[1, 2], label = "Next ▶")
    scrub    = GLMakie.Slider(controls[1, 3], range = 1:length(legs), startvalue = 1)
    GLMakie.Label(controls[1, 4], GLMakie.lift(k -> "waypoint $k / $(length(legs))", current_leg))
    v_toggle = GLMakie.Toggle(controls[1, 5], active = true)
    GLMakie.Label(controls[1, 6], "show V heatmap")

    if has_dijkstra
        dijkstra_toggle = GLMakie.Toggle(controls[1, 7], active = true)
        GLMakie.Label(controls[1, 8], "show Dijkstra route")
        GLMakie.on(dijkstra_toggle.active) do a
            dijkstra_visible[] = a
            draw_dijkstra_route!(); refresh_legend!()
        end
    end

    GLMakie.on(btn_prev.clicks) do _; current_leg[] = max(1, current_leg[] - 1); end
    GLMakie.on(btn_next.clicks) do _; current_leg[] = min(length(legs), current_leg[] + 1); end
    GLMakie.on(scrub.value)     do v; current_leg[] = clamp(round(Int, v), 1, length(legs)); end
    GLMakie.on(v_toggle.active) do a; show_V[] = a; end          # heatmap visibility is bound to show_V

    GLMakie.on(current_leg) do k
        leg = legs[k]
        marker_pos[] = GLMakie.Point2f(leg.lon, leg.lat)
        update_stats!(k); update_comparison!(k)
        refresh_selected!((leg.i, leg.j))
    end

    draw_dijkstra_route!(); refresh_legend!()
    update_stats!(1); update_comparison!(1); refresh_selected!(start_idx)

    GLMakie.display(fig)
    return fig
end

"""RouteStepper2 position at elapsed time `t_s` (linear interpolation, clamped)."""
function _position_at_time(rs::RouteStepper2, t_s::Float64)
    steps = rs.steps
    n = length(steps)
    (n == 1 || t_s <= steps[1].cum_time_s) && return (steps[1].lon, steps[1].lat)
    t_s >= steps[end].cum_time_s && return (steps[end].lon, steps[end].lat)
    times = [s.cum_time_s for s in steps]
    k = clamp(searchsortedfirst(times, t_s), 2, n)
    s0, s1 = steps[k-1], steps[k]
    dt = s1.cum_time_s - s0.cum_time_s
    f  = dt > 0 ? (t_s - s0.cum_time_s) / dt : 0.0
    return (s0.lon + f * (s1.lon - s0.lon), s0.lat + f * (s1.lat - s0.lat))
end

"""
    launch_climatology_isochrone_explorer(V, climatology, grid_points, ocean_mask,
                                          start_idx, goal_idx; band_days = 1:12, dijkstra_route = nothing)

`V` must be 2D (seconds); for the energy-aware V pass a budget slice such as V[:,:,num_bins].
A day slider highlights one isochrone; click a cell for its wind rose. With `dijkstra_route`,
its position at slider-time is marked and a status line says whether it is ahead of or behind
the climatological pace (a Bellman-value comparison: exact only along the DP's own corridor,
a directional signal elsewhere).
"""
function launch_climatology_isochrone_explorer(
    V::Matrix{Float64}, climatology::Climatology,
    grid_points::Matrix{Tuple{Float64,Float64}}, ocean_mask::Matrix{Bool},
    start_idx::Tuple{Int,Int}, goal_idx::Tuple{Int,Int};
    band_days::AbstractVector{<:Real} = 1:12,
    dijkstra_route::Union{Nothing,RouteStepper2} = nothing
)
    GLMakie.activate!()
    has_dijkstra = dijkstra_route !== nothing
    has_dijkstra && isempty(dijkstra_route.steps) &&
        error("launch_climatology_isochrone_explorer: dijkstra_route has no steps.")

    n_rows, n_cols = size(grid_points)
    lons_1d = [grid_points[1, j][1] for j in 1:n_cols]
    lats_1d = [grid_points[i, 1][2] for i in 1:n_rows]

    Vd_plot = V ./ 3600.0 ./ 24.0
    Vd_plot[.!ocean_mask] .= NaN
    Vd_plot[isinf.(Vd_plot)] .= NaN
    Vd_plot_t = permutedims(Vd_plot)

    lo_band, hi_band = extrema(band_days)
    expected_total_h = isfinite(V[start_idx...]) ? V[start_idx...] / 3600 : NaN

    fig = GLMakie.Figure(size = (1900, 1150))
    GLMakie.Label(fig[0, 1:2], "Climatology Isochrone Explorer"; fontsize = 22, font = :bold,
                  halign = :left, tellwidth = false)
    ax_map  = GLMakie.Axis(fig[1, 1]; title = "Isochrone frontier", xlabel = "lon", ylabel = "lat")
    ax_rose = GLMakie.Axis(fig[1, 2]; title = "Wind rose (selected cell)", aspect = 1)
    GLMakie.hidedecorations!(ax_rose)
    GLMakie.colsize!(fig.layout, 1, GLMakie.Relative(0.68))
    GLMakie.colsize!(fig.layout, 2, GLMakie.Relative(0.32))

    GLMakie.heatmap!(ax_map, lons_1d, lats_1d, permutedims(Float32.(ocean_mask));
                     colormap = [RGB(0.82, 0.75, 0.55), RGB(0.93, 0.93, 0.93)],
                     colorrange = (0, 1), interpolate = false)
    v_heatmap = GLMakie.heatmap!(ax_map, lons_1d, lats_1d, Vd_plot_t; colormap = (:viridis, 0.35))
    GLMakie.Colorbar(fig[2, 1], v_heatmap, label = "days to goal", vertical = false)
    GLMakie.contour!(ax_map, lons_1d, lats_1d, Vd_plot_t; levels = collect(band_days),
                     color = (:gray40, 0.35), linewidth = 0.8)

    GLMakie.scatter!(ax_map, [grid_points[start_idx...][1]], [grid_points[start_idx...][2]];
                     color = :lime, markersize = 14, marker = :diamond, label = "start")
    GLMakie.scatter!(ax_map, [grid_points[goal_idx...][1]], [grid_points[goal_idx...][2]];
                     color = :red, markersize = 14, marker = :diamond, label = "goal")

    slider_days   = GLMakie.Observable(Float64(lo_band))
    highlight_ref = Ref{Any}(nothing)
    function draw_highlight!()
        highlight_ref[] === nothing || (delete!(ax_map, highlight_ref[]); highlight_ref[] = nothing)
        highlight_ref[] = GLMakie.contour!(ax_map, lons_1d, lats_1d, Vd_plot_t;
                                           levels = [slider_days[]], color = :orange, linewidth = 3.0)
    end

    dijkstra_marker_ref = Ref{Any}(nothing)
    if has_dijkstra
        d_lons = [s.lon for s in dijkstra_route.steps]; d_lats = [s.lat for s in dijkstra_route.steps]
        d_powers = [s.leg_power_kw for s in dijkstra_route.steps]
        length(d_powers) > 1 && (d_powers[1] = d_powers[2])
        prange = extrema(d_powers)
        prange = prange[2] > prange[1] ? prange : (prange[1] - 1.0, prange[2] + 1.0)
        GLMakie.lines!(ax_map, d_lons, d_lats; color = d_powers, colormap = :thermal, colorrange = prange,
                       linestyle = :dash, linewidth = 2.5, label = "live-solver route")
        dijkstra_marker_ref[] = GLMakie.scatter!(ax_map, [d_lons[1]], [d_lats[1]]; color = :magenta,
                                                 markersize = 18, marker = :star5, label = "live route @ slider time")
    end

    draw_highlight!()                                   # before the legend so it is in the plot set
    GLMakie.axislegend(ax_map; position = :lb, labelsize = 10)

    function refresh_rose!(idx::Tuple{Int,Int})
        rose = climatology.roses[idx...]
        draw_wind_rose!(ax_rose, rose)
        ax_rose.title = "Wind rose at ($(idx[1]),$(idx[2]))  --  n_obs=$(rose.n_obs)"
    end
    _on_map_click!(refresh_rose!, ax_map, lons_1d, lats_1d, ocean_mask)

    status_text = GLMakie.Observable("")
    GLMakie.Label(fig[2, 2], status_text; tellwidth = false, halign = :left, valign = :top,
                  fontsize = 13, justification = :left)

    function update_status!()
        d = slider_days[]
        base = "Isochrone day: $(round(d, digits=1)) ($(round(d * 24, digits=0))h)\n" *
               "Expected total time-to-goal from start: " *
               (isnan(expected_total_h) ? "unreachable" :
                "$(round(expected_total_h, digits=1)) h ($(round(expected_total_h/24, digits=1)) days)")
        if !has_dijkstra
            status_text[] = base * "\n\n(no live-solver route supplied to compare against)"
            return
        end

        lon, lat = _position_at_time(dijkstra_route, d * 24 * 3600)
        if dijkstra_marker_ref[] !== nothing
            dijkstra_marker_ref[][1] = [lon]; dijkstra_marker_ref[][2] = [lat]
        end
        i = nearest_index(lats_1d, lat); j = nearest_index(lons_1d, lon)
        v_here_h = ocean_mask[i, j] ? V[i, j] / 3600 : NaN

        pace_note = if isnan(expected_total_h) || isnan(v_here_h) || isinf(v_here_h)
            "(pace comparison unavailable -- unreachable cell or start)"
        else
            implied_total_h = d * 24 + v_here_h
            delta_h = implied_total_h - expected_total_h
            "Live route at day $(round(d, digits=1)): climatology says $(round(v_here_h, digits=1))h still needed " *
            "(implied total $(round(implied_total_h, digits=1))h vs expected $(round(expected_total_h, digits=1))h)\n" *
            "=> $(round(abs(delta_h), digits=1))h $(delta_h < 0 ? "AHEAD of" : "BEHIND") typical climatological pace"
        end
        status_text[] = base * "\n\n" * pace_note
    end

    controls = fig[3, 1:2] = GLMakie.GridLayout()
    GLMakie.Label(controls[1, 1], "Isochrone day:"; fontsize = 13, font = :bold, halign = :right)
    day_slider = GLMakie.Slider(controls[1, 2], range = lo_band:0.5:hi_band, startvalue = lo_band)
    GLMakie.Label(controls[1, 3], GLMakie.lift(d -> "$(round(d, digits=1)) days", day_slider.value))

    GLMakie.on(day_slider.value) do d
        slider_days[] = d
        draw_highlight!(); update_status!()
    end

    update_status!()
    GLMakie.display(fig)
    return fig
end