# =============================================================================
# routing.jl  --  time- and energy-aware Dijkstra weather routing
#
# Layout
#   1. Types
#   2. Wind-source configuration (synthetic / GFS forecast / archives)
#   3. Epoch registry and wind-field loading
#   4. Bilinear interpolation and the routing-grid UV cache
#   5. Polar look-up tables (single and stacked over power levels)
#   6. Edge-cost integration along arc samples
#   7. Bucket queue
#   8. Geometry and neighbour cache
#   9. Radius bands and calibration helpers
#  10. The solver: findDijkstraRoutefinal
#
# External dependencies (defined in other files, resolved at call time):
#   read_write_data.jl / wind.jl / times_and_dates.jl:
#       getDistance, getBoatAngle, readPolar, twa_and_boat_speed,
#       addSeconds, roundTime, monthWord, timeString, getWindData
#   climatology_routing.jl:      _CLIMATOLOGY_BLEND, climatology_uses_forecast, _load_uv_dated   (must return [lon, lat] arrays)
#
# NOTE: this file now owns the climatology-archive depth counter and the
# unified _loadWindField / _ensureEpochRegistered!. The "OVERRIDE" copies at
# the bottom of the climatology file can be deleted.
# =============================================================================

using CSV
using DataFrames
using DataStructures
using Dates
using GRIB
using JLD2

include("read_write_data.jl")
include("wind.jl")
include("times_and_dates.jl")

const DIRECTIONS = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1)
]

# =============================================================================
# 1. Types
# =============================================================================

struct BilinearWeights
    i0::Int32
    i1::Int32
    j0::Int32
    j1::Int32
    wx::Float32
    wy::Float32
end

"""Raw wind snapshot. `u_raw`/`v_raw` are laid out (n_lons, n_lats)."""
struct WindField
    lats::Vector{Float64}
    lons::Vector{Float64}
    u_raw::Matrix{Float32}
    v_raw::Matrix{Float32}
    n_lats::Int
    epoch::Int
end

struct GridInterpolator
    weights::Matrix{BilinearWeights}
    n_rows::Int
    n_cols::Int
end

"""The two UV snapshots (on the routing grid) bracketing the current node."""
mutable struct TemporalWindPair
    u0::Vector{Float32}
    v0::Vector{Float32}
    u1::Vector{Float32}
    v1::Vector{Float32}
    t0_epoch::Int
    t1_epoch::Int
    ready::Bool
end

mutable struct WindFieldPool
    sorted_epochs::Vector{Int}
    epoch_to_file::Dict{Int, String}
    epoch_to_timeinfo::Dict{Int, Tuple{Int,Int,Int,Int}}
    loaded_fields::OrderedDict{Int, WindField}   # LRU of raw fields
    max_cached::Int
    grid_lats::Union{Nothing, Vector{Float64}}
    grid_lons::Union{Nothing, Vector{Float64}}
    grid_n_lats::Int
end

function WindFieldPool(; max_cached::Int = 6)
    return WindFieldPool(Int[], Dict{Int,String}(), Dict{Int,Tuple{Int,Int,Int,Int}}(),
                         OrderedDict{Int,WindField}(), max_cached, nothing, nothing, 0)
end

"""One neighbour edge: (ni, nj, dist_m, bearing_deg, arc_sample_flat_indices, ring)."""
const NbEntry = Tuple{Int,Int,Float64,Float64,Vector{Int},Int8}
const NbCache = Matrix{Vector{NbEntry}}

function toEpochSeconds(use_time::Int, day::Int, month::Int, year::Int)::Int
    return round(Int, Dates.datetime2unix(DateTime(year, month, day, use_time ÷ 100, use_time % 100)))
end

# =============================================================================
# 2. Wind-source configuration
#
# At most one source is active; _loadWindField and _ensureEpochRegistered!
# dispatch on them in the order: synthetic, GFS, hourly reference,
# climatology archive, then the default per-timestamp ERA5 files.
# =============================================================================

# ---- synthetic uniform wind (isolates geometry from wind variation) ---------
const _SYNTHETIC_WIND_UV = Ref{Union{Nothing,Tuple{Float32,Float32}}}(nothing)

set_synthetic_wind!(u::Real, v::Real) = (_SYNTHETIC_WIND_UV[] = (Float32(u), Float32(v)); nothing)
clear_synthetic_wind!()               = (_SYNTHETIC_WIND_UV[] = nothing; nothing)

# ---- GFS forecast run -------------------------------------------------------
const _GFS_FORECAST_SOURCE = Ref{Union{Nothing,NamedTuple}}(nothing)

"""
    set_gfs_forecast_source!(; dir, init_year, init_month, init_day, cycle_hour,
                               available_hours = vcat(0:120, 123:3:336))

Point the loader at a downloaded GFS run. `available_hours` must match what was
actually downloaded. Any route solved with this active must start exactly at the
run's init time, since epochs are forecast-hour offsets from it.
"""
function set_gfs_forecast_source!(;
        dir::String, init_year::Int, init_month::Int, init_day::Int, cycle_hour::Int,
        available_hours::Vector{Int} = vcat(collect(0:120), collect(123:3:336)))
    init_epoch = toEpochSeconds(cycle_hour * 100, init_day, init_month, init_year)
    cache_dir  = joinpath(dir, "jld2_cache")
    mkpath(cache_dir)
    _GFS_FORECAST_SOURCE[] = (dir = dir, cycle = cycle_hour, init_epoch = init_epoch,
                              available_hours = sort(available_hours), cache_dir = cache_dir)
    println("GFS forecast source set: $dir (init $(init_year)-$(init_month)-$(init_day) " *
            "$(cycle_hour)Z, $(length(available_hours)) forecast hours)")
    return nothing
end
clear_gfs_forecast_source!() = (_GFS_FORECAST_SOURCE[] = nothing; nothing)

"""Snap an epoch to the nearest actually-downloaded forecast hour."""
function _gfs_snap_hour(src::NamedTuple, epoch::Real)::Int
    elapsed_h = (epoch - src.init_epoch) / 3600.0
    best = src.available_hours[1]
    best_diff = abs(elapsed_h - best)
    for h in src.available_hours
        d = abs(elapsed_h - h)
        d < best_diff && (best = h; best_diff = d)
    end
    return best
end

_gfs_file_for_hour(src::NamedTuple, h::Int)::String =
    joinpath(src.dir, "gfs.t$(lpad(src.cycle,2,'0'))z.pgrb2.0p25.f$(lpad(h,3,'0')).uv10m.grib2")

# ---- 6-hourly monthly-batched climatology archive (nestable) ----------------
const _CLIMATOLOGY_ARCHIVE_SOURCE = Ref{Union{Nothing,NamedTuple}}(nothing)
const _CLIMATOLOGY_ARCHIVE_DEPTH  = Ref(0)

"""Reference-counted: nested set/clear pairs are safe; the source is dropped when depth hits 0."""
function set_climatology_archive_source!(; dir::String = CLIMATOLOGY_ARCHIVE_DIR_REF[])
    if _CLIMATOLOGY_ARCHIVE_SOURCE[] === nothing
        _GFS_FORECAST_SOURCE[] === nothing ||
            error("set_climatology_archive_source!: a GFS source is active -- call clear_gfs_forecast_source!() first.")
        isdir(dir) || error("set_climatology_archive_source!: directory not found: $dir")
        _CLIMATOLOGY_ARCHIVE_SOURCE[] =
            (dir = dir, month_cache = Dict{Tuple{Int,Int}, Dict{Int,WindField}}())
        println("Climatology batch-archive wind source set: $dir (6-hourly monthly-batched GRIB)")
    end
    _CLIMATOLOGY_ARCHIVE_DEPTH[] += 1
    return nothing
end

function clear_climatology_archive_source!()
    _CLIMATOLOGY_ARCHIVE_DEPTH[] = max(0, _CLIMATOLOGY_ARCHIVE_DEPTH[] - 1)
    _CLIMATOLOGY_ARCHIVE_DEPTH[] == 0 && (_CLIMATOLOGY_ARCHIVE_SOURCE[] = nothing)
    return nothing
end

# ---- hourly reference archive ------------------------------------------------
const _HOURLY_REFERENCE_SOURCE = Ref{Union{Nothing,NamedTuple}}(nothing)

function set_hourly_reference_source!(; dir::String = "./ClimatologyOceanData/hourly_reference")
    _GFS_FORECAST_SOURCE[]        === nothing || error("clear_gfs_forecast_source!() first")
    _CLIMATOLOGY_ARCHIVE_SOURCE[] === nothing || error("clear_climatology_archive_source!() first")
    isdir(dir) || error("hourly_reference dir not found: $dir")

    manifest_path = joinpath(dir, "manifest.jld2")
    isfile(manifest_path) || error("hourly_reference: manifest.jld2 not found in $dir")
    ptm = jldopen(manifest_path, "r") do f
        haskey(f, "path_to_month") || error("hourly_reference: manifest has no path_to_month")
        f["path_to_month"]
    end

    # manifest keys mix / and \ (written on Windows): normalise each to (year, month)
    available = Set{Tuple{Int,Int}}()
    for k in keys(ptm)
        base = basename(replace(String(k), '\\' => '/'))          # "2025-01-hourly.grib"
        mm = match(r"^(\d{4})-(\d{2})-hourly\.grib$", base)
        mm === nothing && continue
        push!(available, (parse(Int, mm.captures[1]), parse(Int, mm.captures[2])))
    end
    isempty(available) && error("hourly_reference: manifest contained no YYYY-MM-hourly.grib entries")

    _HOURLY_REFERENCE_SOURCE[] = (dir = dir, available = available,
                                  month_cache = Dict{Tuple{Int,Int}, Dict{Int,WindField}}())
    println("Hourly-reference wind source set: $dir ($(length(available)) months, " *
            "$(minimum(available)) .. $(maximum(available)))")
    return nothing
end
clear_hourly_reference_source!() = (_HOURLY_REFERENCE_SOURCE[] = nothing; nothing)

# =============================================================================
# 3. Epoch registry and wind-field loading
# =============================================================================

# ---- epoch registry ----------------------------------------------------------
function _register_epoch!(pool::WindFieldPool, ep::Int, file::String,
                          timeinfo::Tuple{Int,Int,Int,Int})::Int
    haskey(pool.epoch_to_file, ep) && return ep
    pool.epoch_to_file[ep]     = file
    pool.epoch_to_timeinfo[ep] = timeinfo
    insert!(pool.sorted_epochs, searchsortedfirst(pool.sorted_epochs, ep), ep)
    return ep
end

_snap_epoch(approx::Real, step_s::Int)::Int = round(Int, approx / step_s) * step_s

"""
Register (if new) and return the wind epoch for `elapsed_seconds` after the
route start, snapped to whatever grid the active source actually has.
"""
function _ensureEpochRegistered!(pool::WindFieldPool,
        base_time::Int, base_day::Int, base_month::Int, base_year::Int, elapsed_seconds::Int)

    approx_epoch() = toEpochSeconds(base_time, base_day, base_month, base_year) + elapsed_seconds

    if (src = _GFS_FORECAST_SOURCE[]) !== nothing
        h = _gfs_snap_hour(src, approx_epoch())      # only real downloaded hours become epochs
        return _register_epoch!(pool, src.init_epoch + h * 3600,
                                _gfs_file_for_hour(src, h), (base_time, base_day, base_month, base_year))
    elseif _HOURLY_REFERENCE_SOURCE[] !== nothing
        return _register_epoch!(pool, _snap_epoch(approx_epoch(), 3600), "", (0, 0, 0, 0))
    elseif _CLIMATOLOGY_ARCHIVE_SOURCE[] !== nothing
        # Unix epoch 0 is 00Z, so multiples of 21600 s land exactly on 00/06/12/18Z.
        return _register_epoch!(pool, _snap_epoch(approx_epoch(), 21600), "", (0, 0, 0, 0))
    end

    # default: per-timestamp ERA5 files
    ny, nm, nd, ut = addSeconds(base_time, base_day, base_month, base_year, elapsed_seconds)
    ny, nm, nd, ut = roundTime(ut, nd, nm, ny)
    ep = toEpochSeconds(ut, nd, nm, ny)
    haskey(pool.epoch_to_file, ep) && return ep
    return _register_epoch!(pool, ep, "./OceanData/$(ut)-$(nd)$(monthWord(nm))$(ny).grib",
                            (ut, nd, nm, ny))
end

function get_epoch_for_bucket!(hour_to_epoch::Dict{Int,Int}, wind_pool::WindFieldPool,
        cur_time::Int, cur_day::Int, cur_month::Int, cur_year::Int,
        rounded_hour_bucket::Int, elapsed_int::Int)::Int
    ep = get(hour_to_epoch, rounded_hour_bucket, -1)
    ep != -1 && return ep
    ep = _ensureEpochRegistered!(wind_pool, cur_time, cur_day, cur_month, cur_year, elapsed_int)
    hour_to_epoch[rounded_hour_bucket] = ep
    return ep
end

# ---- shared loading helpers --------------------------------------------------
function _adopt_grid!(pool::WindFieldPool, lats::Vector{Float64}, lons::Vector{Float64})
    if pool.grid_lats === nothing
        pool.grid_lats   = lats
        pool.grid_lons   = lons
        pool.grid_n_lats = length(lats)
    end
    return nothing
end

"""Build (n_lons, n_lats) u/v matrices by explicit coordinate lookup (no reliance on GRIB order)."""
function _build_uv_matrix_indexed(lats_raw, lons_raw, u_vals, v_vals)
    lats_unique = sort(unique(lats_raw))   # ascending
    lons_unique = sort(unique(lons_raw))   # ascending
    lat_idx = Dict{eltype(lats_unique), Int}(la => i for (i, la) in enumerate(lats_unique))
    lon_idx = Dict{eltype(lons_unique), Int}(lo => j for (j, lo) in enumerate(lons_unique))

    u_mat = Array{Float32}(undef, length(lons_unique), length(lats_unique))
    v_mat = Array{Float32}(undef, length(lons_unique), length(lats_unique))
    @inbounds for k in eachindex(lons_raw)
        i = lon_idx[lons_raw[k]]
        j = lat_idx[lats_raw[k]]
        u_mat[i, j] = Float32(u_vals[k])
        v_mat[i, j] = Float32(v_vals[k])
    end
    return lats_unique, lons_unique, u_mat, v_mat
end

"""Read 10u/10v (lats, lons, u, v) from a GRIB file; throws if either is missing."""
function _read_uv_grib(path::String)
    lats_raw = lons_raw = u_vals = v_vals = nothing
    GRIB.GribFile(path) do f
        for msg in f
            sname = msg["shortName"]
            if sname == "10u"
                lats_raw = msg["latitudes"]; lons_raw = msg["longitudes"]; u_vals = msg["values"]
            elseif sname == "10v"
                v_vals = msg["values"]
            end
            !isnothing(u_vals) && !isnothing(v_vals) && break
        end
    end
    (isnothing(u_vals) || isnothing(v_vals)) && error("Missing 10u/10v shortName in $path")
    return lats_raw, lons_raw, u_vals, v_vals
end

"""Rotate a 0..360 longitude axis to -180..180 (reordering the u/v rows to match)."""
function _rotate_to_pm180(lons360::Vector{Float64}, u360::Matrix{Float32}, v360::Matrix{Float32})
    split_idx = findfirst(l -> l >= 180.0, lons360)
    split_idx === nothing && return lons360, u360, v360
    lons = vcat(lons360[split_idx:end] .- 360.0, lons360[1:split_idx-1])
    u    = vcat(u360[split_idx:end, :], u360[1:split_idx-1, :])
    v    = vcat(v360[split_idx:end, :], v360[1:split_idx-1, :])
    return lons, u, v
end

function _read_wind_cache!(pool::WindFieldPool, jld2_path::String)
    return jldopen(jld2_path, "r") do f
        _adopt_grid!(pool, f["grid_lats"], f["grid_lons"])
        (f["u_raw"]::Matrix{Float32}, f["v_raw"]::Matrix{Float32})
    end
end

_write_wind_cache(jld2_path::String, pool::WindFieldPool, u_mat, v_mat) =
    jldsave(jld2_path; u_raw = u_mat, v_raw = v_mat,
            grid_lats = pool.grid_lats, grid_lons = pool.grid_lons)

# ---- synthetic ---------------------------------------------------------------
function _synthetic_field(pool::WindFieldPool, epoch::Int)::WindField
    u_const, v_const = _SYNTHETIC_WIND_UV[]
    if pool.grid_lats === nothing    # whole-globe grid; resolution is irrelevant for a constant field
        _adopt_grid!(pool, collect(-90.0:1.0:90.0), collect(-180.0:1.0:179.0))
    end
    n_lons = length(pool.grid_lons)
    return WindField(pool.grid_lats, pool.grid_lons,
                     fill(u_const, n_lons, pool.grid_n_lats), fill(v_const, n_lons, pool.grid_n_lats),
                     pool.grid_n_lats, epoch)
end

# ---- GFS ---------------------------------------------------------------------
function _load_gfs_field(pool::WindFieldPool, src::NamedTuple, epoch::Int)::WindField
    h = round(Int, (epoch - src.init_epoch) / 3600.0)
    grib_path = _gfs_file_for_hour(src, h)
    jld2_path = joinpath(src.cache_dir, "f$(lpad(h,3,'0')).jld2")   # keyed by forecast hour

    if isfile(jld2_path)
        u_mat, v_mat = _read_wind_cache!(pool, jld2_path)
    else
        isfile(grib_path) ||
            error("GFS file missing for forecast hour f$h: $grib_path -- re-run " *
                  "download_gfs_14day_wind.sh or check available_hours.")
        lats_raw, lons_raw, u_vals, v_vals = _read_uv_grib(grib_path)
        lats, lons360, u360, v360 = _build_uv_matrix_indexed(lats_raw, lons_raw, u_vals, v_vals)
        lons, u_mat, v_mat = _rotate_to_pm180(lons360, u360, v360)
        _adopt_grid!(pool, lats, lons)
        _write_wind_cache(jld2_path, pool, u_mat, v_mat)
    end
    return WindField(pool.grid_lats, pool.grid_lons, u_mat, v_mat, pool.grid_n_lats, epoch)
end

# ---- monthly-batched archives (climatology 6-hourly, hourly reference) --------
function _month_snapshots_by_epoch(path::String)::Dict{Int,WindField}
    by_epoch = Dict{Int,WindField}()
    for (dataDate, time_, field) in _load_uv_dated(path)    # (dataDate, time, WindField)
        ep = toEpochSeconds(time_, dataDate % 100, (dataDate ÷ 100) % 100, dataDate ÷ 10000)
        by_epoch[ep] = WindField(field.lats, field.lons, field.u_raw, field.v_raw, field.n_lats, ep)
    end
    return by_epoch
end

function _snapshot_field!(pool::WindFieldPool, source::NamedTuple, epoch::Int,
                          path::String, label::String)::WindField
    dt = Dates.unix2datetime(epoch)
    key = (Dates.year(dt), Dates.month(dt))
    by_epoch = get!(source.month_cache, key) do
        isfile(path) || error("$label: no batch file for $(key[1])-$(lpad(key[2],2,'0')) ($path)")
        d = _month_snapshots_by_epoch(path)
        println("$label: loaded $(length(d)) snapshots for $(key[1])-$(lpad(key[2],2,'0'))")
        d
    end
    haskey(by_epoch, epoch) ||
        error("$label: no snapshot at epoch $epoch ($dt) -- a snapping bug or a genuine archive gap.")
    field = by_epoch[epoch]
    _adopt_grid!(pool, field.lats, field.lons)
    return field
end

function _load_climatology_archive_field(pool::WindFieldPool, source::NamedTuple, epoch::Int)::WindField
    dt = Dates.unix2datetime(epoch)
    path = joinpath(source.dir, "$(Dates.year(dt))-$(lpad(Dates.month(dt),2,'0')).grib")
    return _snapshot_field!(pool, source, epoch, path, "Climatology archive")
end

function _load_hourly_reference_field(pool::WindFieldPool, source::NamedTuple, epoch::Int)::WindField
    dt = Dates.unix2datetime(epoch)
    y, m = Dates.year(dt), Dates.month(dt)
    (y, m) in source.available ||
        error("hourly_reference: month $y-$(lpad(m,2,'0')) not in manifest " *
              "(available: $(sort(collect(source.available))))")
    path = joinpath(source.dir, "$(y)-$(lpad(m,2,'0'))-hourly.grib")
    return _snapshot_field!(pool, source, epoch, path, "hourly_reference")
end

# ---- default per-timestamp ERA5 ----------------------------------------------
function _download_wind_file!(path, year, month, day, use_time)
    str_time = timeString(use_time)
    getWindData(path, year, month, day, [str_time], [90, -180, -90, 180])
    (isfile(path) && filesize(path) > 1024) ||
        error("Download failed or returned an empty/near-empty file: $path " *
              "(requested $year-$month-$day $str_time -- check this date/hour is published in your ERA5 source)")
    println("Downloaded: $path")
end

function _load_era5_field(pool::WindFieldPool, path::String, epoch::Int,
                          use_time::Int, day::Int, month::Int, year::Int)::WindField
    jld2_path = replace(path, ".grib" => ".jld2")

    if isfile(jld2_path)
        u_mat, v_mat = _read_wind_cache!(pool, jld2_path)
    else
        isfile(path) || _download_wind_file!(path, year, month, day, use_time)
        # A corrupt, partial, or u/v-less file gets one re-download before we give up.
        lats_raw, lons_raw, u_vals, v_vals = try
            _read_uv_grib(path)
        catch e
            @warn "Corrupt/incomplete GRIB detected, re-downloading: $path" exception = e
            rm(path; force = true)
            _download_wind_file!(path, year, month, day, use_time)
            _read_uv_grib(path)
        end
        lats, lons, u_mat, v_mat = _build_uv_matrix_indexed(lats_raw, lons_raw, u_vals, v_vals)
        _adopt_grid!(pool, lats, lons)
        _write_wind_cache(jld2_path, pool, u_mat, v_mat)
        println("Converted to JLD2: $jld2_path")
    end
    return WindField(pool.grid_lats, pool.grid_lons, u_mat, v_mat, pool.grid_n_lats, epoch)
end

# ---- dispatcher --------------------------------------------------------------
function _loadWindField(pool::WindFieldPool, path::String, epoch::Int,
                        use_time::Int, day::Int, month::Int, year::Int)::WindField
    _SYNTHETIC_WIND_UV[]          !== nothing && return _synthetic_field(pool, epoch)
    _GFS_FORECAST_SOURCE[]        !== nothing && return _load_gfs_field(pool, _GFS_FORECAST_SOURCE[], epoch)
    _HOURLY_REFERENCE_SOURCE[]    !== nothing && return _load_hourly_reference_field(pool, _HOURLY_REFERENCE_SOURCE[], epoch)
    _CLIMATOLOGY_ARCHIVE_SOURCE[] !== nothing && return _load_climatology_archive_field(pool, _CLIMATOLOGY_ARCHIVE_SOURCE[], epoch)
    return _load_era5_field(pool, path, epoch, use_time, day, month, year)
end

"""LRU fetch of a raw wind field."""
function _getField!(pool::WindFieldPool, epoch::Int)::WindField
    if haskey(pool.loaded_fields, epoch)
        field = pop!(pool.loaded_fields, epoch)
        pool.loaded_fields[epoch] = field          # move to most-recent
        return field
    end
    length(pool.loaded_fields) >= pool.max_cached &&
        delete!(pool.loaded_fields, first(keys(pool.loaded_fields)))

    ut, d, m, y = pool.epoch_to_timeinfo[epoch]
    field = _loadWindField(pool, pool.epoch_to_file[epoch], epoch, ut, d, m, y)
    pool.loaded_fields[epoch] = field
    return field
end

# =============================================================================
# 4. Bilinear interpolation and the routing-grid UV cache
# =============================================================================

function build_interpolator(grid_points::Matrix{Tuple{Float64,Float64}}, field::WindField)::GridInterpolator
    n_rows, n_cols = size(grid_points)
    weights = Matrix{BilinearWeights}(undef, n_rows, n_cols)
    lons_g, lats_g = field.lons, field.lats
    n_lats, n_lons = field.n_lats, length(lons_g)
    lon_wrap_span = lons_g[1] + 360.0 - lons_g[n_lons]     # gap across the antimeridian

    @inbounds for j in 1:n_cols, i in 1:n_rows
        lon, lat = grid_points[i, j]

        i0 = searchsortedlast(lons_g, lon)
        local i1::Int32
        local wx::Float32
        if i0 < 1                                  # west of first column: wrap
            i0 = n_lons; i1 = 1
            wx = Float32((lon + 360.0 - lons_g[n_lons]) / lon_wrap_span)
        elseif i0 >= n_lons                        # east of last column: wrap
            i0 = n_lons; i1 = 1
            wx = Float32((lon - lons_g[n_lons]) / lon_wrap_span)
        else
            i1 = i0 + 1
            wx = Float32((lon - lons_g[i0]) / (lons_g[i1] - lons_g[i0]))
        end
        wx = clamp(wx, 0f0, 1f0)

        jl0 = clamp(searchsortedlast(lats_g, lat), 1, n_lats - 1)
        jl1 = jl0 + 1
        wy  = Float32((lat - lats_g[jl0]) / (lats_g[jl1] - lats_g[jl0]))

        weights[i, j] = BilinearWeights(Int32(i0), Int32(i1), Int32(jl0), Int32(jl1), wx, wy)
    end
    return GridInterpolator(weights, n_rows, n_cols)
end

"""Interpolate a raw field onto the routing grid; output is column-major flat (flat = (j-1)*n_rows + i)."""
function precompute_uv_grid!(out_u::Vector{Float32}, out_v::Vector{Float32},
                             field::WindField, interp::GridInterpolator)
    n_rows, n_cols = interp.n_rows, interp.n_cols
    u_raw, v_raw = field.u_raw, field.v_raw

    @inbounds for j in 1:n_cols, i in 1:n_rows
        w = interp.weights[i, j]
        wx = w.wx; wy = w.wy
        w00 = (1f0 - wx) * (1f0 - wy)
        w10 =        wx  * (1f0 - wy)
        w01 = (1f0 - wx) *        wy
        w11 =        wx  *        wy

        flat = (j - 1) * n_rows + i
        out_u[flat] = w00 * u_raw[w.i0, w.j0] + w10 * u_raw[w.i1, w.j0] +
                      w01 * u_raw[w.i0, w.j1] + w11 * u_raw[w.i1, w.j1]
        out_v[flat] = w00 * v_raw[w.i0, w.j0] + w10 * v_raw[w.i1, w.j0] +
                      w01 * v_raw[w.i0, w.j1] + w11 * v_raw[w.i1, w.j1]
    end
    return nothing
end

"""LRU cache of routing-grid (u, v) per epoch."""
mutable struct UVGridCache
    entries   ::OrderedDict{Int, Tuple{Vector{Float32}, Vector{Float32}}}
    max_cached::Int
    n_flat    ::Int
    hits      ::Int
    misses    ::Int
end

UVGridCache(n_flat::Int; max_cached::Int = 256) =
    UVGridCache(OrderedDict{Int, Tuple{Vector{Float32}, Vector{Float32}}}(), max_cached, n_flat, 0, 0)

"""Insert into the cache. Copies by default: never alias buffers that are mutated in place elsewhere (wind_pair.u0/v0)."""
function _uv_cache_put!(cache::UVGridCache, epoch::Int, u::Vector{Float32}, v::Vector{Float32};
                        copy_data::Bool = true)
    if haskey(cache.entries, epoch)
        delete!(cache.entries, epoch)
    elseif length(cache.entries) >= cache.max_cached
        delete!(cache.entries, first(keys(cache.entries)))
    end
    cache.entries[epoch] = copy_data ? (Base.copy(u), Base.copy(v)) : (u, v)
    return nothing
end

function get_uv!(cache::UVGridCache, pool::WindFieldPool, interp::GridInterpolator,
                 epoch::Int)::Tuple{Vector{Float32}, Vector{Float32}}
    if haskey(cache.entries, epoch)
        uv = pop!(cache.entries, epoch)
        cache.entries[epoch] = uv                  # LRU touch
        cache.hits += 1
        return uv
    end
    cache.misses += 1
    u = Vector{Float32}(undef, cache.n_flat)
    v = Vector{Float32}(undef, cache.n_flat)
    precompute_uv_grid!(u, v, _getField!(pool, epoch), interp)
    _uv_cache_put!(cache, epoch, u, v; copy_data = false)    # fresh arrays, no copy needed
    return u, v
end

function prewarm_uv_cache!(cache::UVGridCache, pool::WindFieldPool, interp::GridInterpolator;
                           epochs::Union{Nothing, AbstractVector{Int}} = nothing,
                           progress_every::Int = 24)
    to_load = epochs === nothing ? pool.sorted_epochs : collect(epochs)
    n = length(to_load)
    n == 0 && return
    println("Pre-warming UV cache for $n epochs (cap=$(cache.max_cached))...")
    t0 = time()
    for (k, ep) in enumerate(to_load)
        get_uv!(cache, pool, interp, ep)
        if progress_every > 0 && (k == n || k % progress_every == 0)
            println("  UV cache: $k / $n epochs  " *
                    "(raw GRIB in RAM=$(length(pool.loaded_fields))/$(pool.max_cached))")
        end
    end
    println("UV cache warm: $(length(cache.entries)) epochs in $(round(time()-t0, digits=1))s")
end

# =============================================================================
# 5. Polar look-up tables
# =============================================================================

struct PolarLUT
    ws_vals      ::Vector{Float32}
    twa_vals     ::Vector{Float32}
    speeds       ::Matrix{Float32}     # n_ws x n_twa
    ws_step_inv  ::Float32
    twa_step_inv ::Float32
end

function build_polar_lut(polar_info; n_ws::Int = 200, n_twa::Int = 361)::PolarLUT
    ws_max   = Float32(polar_info[1][end])
    ws_vals  = Float32.(range(0.0, ws_max, length = n_ws))
    twa_vals = Float32.(range(0.0, 180.0,  length = n_twa))

    speeds = Matrix{Float32}(undef, n_ws, n_twa)
    for (ti, twa_val) in enumerate(twa_vals), (wi, ws) in enumerate(ws_vals)
        _, spd = twa_and_boat_speed(Float64(ws), Float64(twa_val), 0.0, polar_info)
        speeds[wi, ti] = Float32(spd)
    end
    return PolarLUT(ws_vals, twa_vals, speeds,
                    1f0 / (ws_vals[2] - ws_vals[1]), 1f0 / (twa_vals[2] - twa_vals[1]))
end

const CLIMATOLOGY_ARCHIVE_DIR_REF = Ref("./ClimatologyOceanData/route")
const POLAR_DIR_REF = Ref("./Polars")

function load_polar_lut(P_kw::Real; kwargs...)::PolarLUT
    return build_polar_lut(readPolar("$(POLAR_DIR_REF[])/RisePol$(round(Int, P_kw))kW.csv"); kwargs...)
end

"""Bilinear boat-speed lookup. Float32 method is the hot-path one; Float64 forwards to it."""
@inline function lut_lookup(lut::PolarLUT, wind_speed::Float32, twa::Float32)::Float32
    n_ws, n_twa = length(lut.ws_vals), length(lut.twa_vals)

    wi_f = clamp(wind_speed * lut.ws_step_inv, 0f0, Float32(n_ws  - 1))
    ti_f = clamp(twa        * lut.twa_step_inv, 0f0, Float32(n_twa - 1))
    wi = min(unsafe_trunc(Int, wi_f) + 1, n_ws  - 1)
    ti = min(unsafe_trunc(Int, ti_f) + 1, n_twa - 1)
    αw = wi_f - Float32(wi - 1)
    αt = ti_f - Float32(ti - 1)

    return (1f0 - αw) * (1f0 - αt) * lut.speeds[wi,     ti    ] +
                  αw  * (1f0 - αt) * lut.speeds[wi + 1, ti    ] +
           (1f0 - αw) *        αt  * lut.speeds[wi,     ti + 1] +
                  αw  *        αt  * lut.speeds[wi + 1, ti + 1]
end
@inline lut_lookup(lut::PolarLUT, wind_speed::Float64, twa::Float64)::Float32 =
    lut_lookup(lut, Float32(wind_speed), Float32(twa))

"""Polars for every power level on a common (ws, twa) grid, stacked on the 3rd axis (ascending power)."""
struct StackedLUT
    ws_vals      ::Vector{Float32}
    twa_vals     ::Vector{Float32}
    speeds       ::Array{Float32,3}   # n_ws x n_twa x n_power
    ws_step_inv  ::Float32
    twa_step_inv ::Float32
    n_power      ::Int
end

"""Stack already-built LUTs (avoids re-reading and re-tabulating the polars)."""
function build_stacked_lut(luts::Vector{PolarLUT})::StackedLUT
    n_p   = length(luts)
    n_ws  = length(luts[1].ws_vals)
    n_twa = length(luts[1].twa_vals)

    ws_max_common = minimum(Float64(l.ws_vals[end]) for l in luts)
    ws_vals  = Float32.(range(0.0, ws_max_common, length = n_ws))
    twa_vals = luts[1].twa_vals

    speeds = Array{Float32,3}(undef, n_ws, n_twa, n_p)
    for k in 1:n_p, ti in 1:n_twa, wi in 1:n_ws
        speeds[wi, ti, k] = lut_lookup(luts[k], Float64(ws_vals[wi]), Float64(twa_vals[ti]))
    end
    return StackedLUT(ws_vals, twa_vals, speeds,
                      1f0 / (ws_vals[2] - ws_vals[1]), 1f0 / (twa_vals[2] - twa_vals[1]), n_p)
end

build_stacked_lut(power_levels::Vector{Float64}; n_ws::Int = 200, n_twa::Int = 361) =
    build_stacked_lut(PolarLUT[load_polar_lut(P; n_ws, n_twa) for P in power_levels])

"""Fill out[1..n_power] with boat speed at every power level; bilinear indices computed once."""
@inline function stacked_lookup!(out::Vector{Float32}, slut::StackedLUT,
                                 wind_speed::Float32, twa::Float32)
    n_ws, n_twa = length(slut.ws_vals), length(slut.twa_vals)

    wi_f = clamp(wind_speed * slut.ws_step_inv, 0f0, Float32(n_ws  - 1))
    ti_f = clamp(twa        * slut.twa_step_inv, 0f0, Float32(n_twa - 1))
    wi = min(unsafe_trunc(Int, wi_f) + 1, n_ws  - 1)
    ti = min(unsafe_trunc(Int, ti_f) + 1, n_twa - 1)
    αw = wi_f - Float32(wi - 1)
    αt = ti_f - Float32(ti - 1)

    w00 = (1f0 - αw) * (1f0 - αt);  w10 = αw * (1f0 - αt)
    w01 = (1f0 - αw) *        αt ;  w11 = αw *        αt

    @inbounds for k in 1:slut.n_power
        out[k] = w00 * slut.speeds[wi,   ti,   k] + w10 * slut.speeds[wi+1, ti,   k] +
                 w01 * slut.speeds[wi,   ti+1, k] + w11 * slut.speeds[wi+1, ti+1, k]
    end
end

# =============================================================================
# 6. Edge-cost integration
#
# Each edge is walked sample by sample: take the local (time-interpolated)
# wind, look up boat speed at that heading for every power level, and
# accumulate dt = seg_dist / v and de = P * dt / 3600 (forward Euler).
# =============================================================================

"""
Integrate every power level over one edge in a single pass over its samples.
Returns false as soon as even the FASTEST level (index n_power) is below
`min_speed` at some sample -- boat speed is non-decreasing in power, so no level
can then be feasible. ASSUMES levels are sorted ascending.
Outputs: t_buf (s), e_buf (kWh), vmin_buf (slowest sample speed) per level.
"""
@inline function _stepped_edge_all_powers!(
        t_buf::Vector{Float64}, e_buf::Vector{Float64}, vmin_buf::Vector{Float64},
        speed_tmp::Vector{Float32}, slut::StackedLUT,
        ws_buf::Vector{Float32}, twa_buf::Vector{Float32},
        n_samp::Int, seg_dist::Float64, plevels_f::Vector{Float64}, min_speed::Float64)::Bool
    n_power = slut.n_power
    @inbounds for k in 1:n_power
        t_buf[k] = 0.0; e_buf[k] = 0.0; vmin_buf[k] = Inf
    end
    @inbounds for s in 1:n_samp
        stacked_lookup!(speed_tmp, slut, ws_buf[s], twa_buf[s])
        Float64(speed_tmp[n_power]) < min_speed && return false
        for k in 1:n_power
            vboat       = Float64(speed_tmp[k])
            vmin_buf[k] = min(vmin_buf[k], vboat)
            t_seg       = seg_dist / vboat
            t_buf[k]   += t_seg
            e_buf[k]   += plevels_f[k] * t_seg * (1.0 / 3600.0)
        end
    end
    return true
end

@inline function _feasibility_from_vmin!(feasible_buf::Vector{Bool}, vmin_buf::Vector{Float64},
                                         edge_feasible::Bool, min_speed::Float64, n_power::Int)
    @inbounds for k in 1:n_power
        feasible_buf[k] = edge_feasible && vmin_buf[k] >= min_speed
    end
    return nothing
end

# ---- wind sample -> (wind speed, true wind angle) ------------------------------
@inline function _wind_to_ws_twa(u::Float32, v::Float32, bearing32::Float32)::Tuple{Float32,Float32}
    wind_dir = mod(rad2deg(atan(u, v)) + 180f0, 360f0)        # direction the wind blows FROM
    return sqrt(u^2 + v^2), abs(mod(wind_dir - bearing32 + 180f0, 360f0) - 180f0)
end

@inline function _frozen_uv(wp::TemporalWindPair, flat_ix::Int, omt::Float32, t_frac::Float32)
    @inbounds if wp.ready
        return (omt * wp.u0[flat_ix] + t_frac * wp.u1[flat_ix],
                omt * wp.v0[flat_ix] + t_frac * wp.v1[flat_ix])
    else
        return (wp.u0[flat_ix], wp.v0[flat_ix])
    end
end

"""Fraction of the way between wind_pair's two snapshots at absolute time `abs_now`."""
@inline function _bracket_fraction(wp::TemporalWindPair, abs_now::Int)::Tuple{Float32,Float32}
    wp.ready || return (1f0, 0f0)                       # (omt, t_frac)
    dt_ep  = Float32(wp.t1_epoch - wp.t0_epoch)
    t_frac = dt_ep > 0f0 ? clamp(Float32(abs_now - wp.t0_epoch) / dt_ep, 0f0, 1f0) : 0f0
    return (1f0 - t_frac, t_frac)
end

# ---- time-aware sampling: re-fetch the epoch bracket mid-edge -------------------
"""Holds the bracket containing the sample currently being evaluated along an edge."""
mutable struct WindTimeCursor
    ep0::Int
    ep1::Int
    u0::Vector{Float32}
    v0::Vector{Float32}
    u1::Vector{Float32}
    v1::Vector{Float32}
    has_bracket::Bool
end
WindTimeCursor() = WindTimeCursor(-1, -1, Float32[], Float32[], Float32[], Float32[], false)

# `n_flat` is unused but kept so existing callers (replay_route_real_wind) still match.
@inline function _cursor_uv_at_time!(
        cursor::WindTimeCursor, wind_pool::WindFieldPool, interp::GridInterpolator,
        uv_grid_cache::UVGridCache, n_flat::Int, flat_ix::Int, t_abs::Float64,
        route_start_abs::Int, cur_time::Int, cur_day::Int, cur_month::Int, cur_year::Int
    )::Tuple{Float32,Float32}
    if !cursor.has_bracket || t_abs < Float64(cursor.ep0) || t_abs >= Float64(cursor.ep1)
        elapsed_est = round(Int, t_abs - route_start_abs)
        _ensureEpochRegistered!(wind_pool, cur_time, cur_day, cur_month, cur_year, elapsed_est)
        sorted = wind_pool.sorted_epochs
        idx = clamp(searchsortedlast(sorted, floor(Int, t_abs)), 1, length(sorted) - 1)
        ep0 = sorted[idx]; ep1 = sorted[idx + 1]

        cursor.u0, cursor.v0 = get_uv!(uv_grid_cache, wind_pool, interp, ep0)
        cursor.u1, cursor.v1 = get_uv!(uv_grid_cache, wind_pool, interp, ep1)
        cursor.ep0 = ep0; cursor.ep1 = ep1
        cursor.has_bracket = true
    end

    dt  = Float64(cursor.ep1 - cursor.ep0)
    tf  = Float32(dt > 0 ? clamp((t_abs - cursor.ep0) / dt, 0.0, 1.0) : 0.0)
    omt = 1f0 - tf
    @inbounds begin
        u = omt * cursor.u0[flat_ix] + tf * cursor.u1[flat_ix]
        v = omt * cursor.v0[flat_ix] + tf * cursor.v1[flat_ix]
    end
    return u, v
end

"""
Time-aware fill, ONE arc walk per edge. The local clock advances using the speed
of the fastest power level (`lut_ref`) evaluated on the wind just sampled, so it
does not drift as it would with a single departure-time speed estimate.
"""
@inline function _fill_edge_wind_time_aware!(
        ws_buf::Vector{Float32}, twa_buf::Vector{Float32},
        cursor::WindTimeCursor, wind_pool::WindFieldPool, interp::GridInterpolator,
        uv_grid_cache::UVGridCache, n_flat::Int, arc_flats::Vector{Int}, bearing::Float64,
        t_start_abs::Float64, seg_dist::Float64, lut_ref::PolarLUT,
        route_start_abs::Int, cur_time::Int, cur_day::Int, cur_month::Int, cur_year::Int)
    bearing32 = Float32(bearing)
    t_now = t_start_abs
    @inbounds for k in eachindex(arc_flats)
        u, v = _cursor_uv_at_time!(cursor, wind_pool, interp, uv_grid_cache, n_flat,
                                   arc_flats[k], t_now, route_start_abs,
                                   cur_time, cur_day, cur_month, cur_year)
        ws, twa = _wind_to_ws_twa(u, v, bearing32)
        ws_buf[k] = ws; twa_buf[k] = twa
        t_now += seg_dist / max(Float64(lut_lookup(lut_ref, ws, twa)), 0.5)
    end
end

"""Fast path: wind frozen at the node's departure bracket. Use when edges are short relative to the GRIB step."""
@inline function _fill_edge_wind_frozen!(
        ws_buf::Vector{Float32}, twa_buf::Vector{Float32}, wind_pair::TemporalWindPair,
        arc_flats::Vector{Int}, bearing::Float64, omt::Float32, t_frac::Float32)
    bearing32 = Float32(bearing)
    @inbounds for k in eachindex(arc_flats)
        u, v = _frozen_uv(wind_pair, arc_flats[k], omt, t_frac)
        ws_buf[k], twa_buf[k] = _wind_to_ws_twa(u, v, bearing32)
    end
end

"""Cheapest path: two samples at the 1/4 and 3/4 points of the arc (slots 1 and 2)."""
@inline function _fill_edge_wind_coarse2!(
        ws_buf::Vector{Float32}, twa_buf::Vector{Float32}, wind_pair::TemporalWindPair,
        arc_flats::Vector{Int}, bearing::Float64, omt::Float32, t_frac::Float32)
    n_full    = length(arc_flats)
    bearing32 = Float32(bearing)
    ix1 = arc_flats[max(1, (n_full + 2) ÷ 4)]
    ix2 = arc_flats[min(n_full, (3 * n_full + 2) ÷ 4)]
    u, v = _frozen_uv(wind_pair, ix1, omt, t_frac)
    ws_buf[1], twa_buf[1] = _wind_to_ws_twa(u, v, bearing32)
    u, v = _frozen_uv(wind_pair, ix2, omt, t_frac)
    ws_buf[2], twa_buf[2] = _wind_to_ws_twa(u, v, bearing32)
end

# =============================================================================
# 7. Bucket queue (used by the climatology DP; the main solver uses BinaryMinHeap)
# =============================================================================

mutable struct BucketQueue2
    buckets     ::Vector{Vector{Tuple{Float64,Int32,Int32,Int32}}}
    n_buckets   ::Int
    bucket_width::Float64
    inv_bw      ::Float64
    current     ::Int
    size        ::Int
end

function BucketQueue2(; bucket_width::Float64 = 600.0, n_buckets::Int = 50_000)
    buckets = [Tuple{Float64,Int32,Int32,Int32}[] for _ in 1:n_buckets]
    return BucketQueue2(buckets, n_buckets, bucket_width, 1.0 / bucket_width, 1, 0)
end

@inline function bq_push!(bq::BucketQueue2, cost::Float64, i::Int32, j::Int32, e::Int32)
    b = (unsafe_trunc(Int, cost * bq.inv_bw) % bq.n_buckets) + 1
    push!(bq.buckets[b], (cost, i, j, e))
    bq.size += 1
end

@inline bq_isempty(bq::BucketQueue2) = bq.size == 0

function bq_pop!(bq::BucketQueue2)
    n = bq.n_buckets
    c = bq.current
    while isempty(bq.buckets[c])
        c = c == n ? 1 : c + 1
    end
    bq.current = c
    entry = pop!(bq.buckets[c])
    bq.size -= 1
    return entry
end

# =============================================================================
# 8. Geometry and neighbour cache
# =============================================================================

@inline flat_idx(i::Int, j::Int, n_rows::Int) = (j - 1) * n_rows + i

"""Sampled land check along a grid line; also forbids slipping between two diagonally adjacent land cells."""
function line_on_water_geo(i1::Int, j1::Int, i2::Int, j2::Int,
                           ocean_mask::Matrix{Bool}; oversample::Int = 3)
    n_rows, n_cols = size(ocean_mask)
    n_samp = max(2, oversample * max(abs(i2 - i1), abs(j2 - j1)) + 1)

    (i1 < 1 || i1 > n_rows || j1 < 1 || j1 > n_cols) && return false
    !ocean_mask[i1, j1] && return false

    i_prev, j_prev = i1, j1
    for k in 1:n_samp
        α = k / n_samp
        i = round(Int, i1 + α * (i2 - i1))
        j = round(Int, j1 + α * (j2 - j1))

        (i < 1 || i > n_rows || j < 1 || j > n_cols) && return false
        !ocean_mask[i, j] && return false

        if i != i_prev && j != j_prev            # diagonal step: both orthogonal cells must be water
            (!ocean_mask[i_prev, j] || !ocean_mask[i, j_prev]) && return false
        end
        i_prev, j_prev = i, j
    end
    return true
end

"""Half-plane neighbours (dj >= 0) within `radius`; used by the 2025 baseline solver only."""
function neighbor_indices(i::Int, j::Int, radius::Int, n_rows::Int, n_cols::Int)
    nbrs = Tuple{Int,Int}[]
    for di in -radius:radius, dj in 0:radius
        ni, nj = i + di, j + dj
        if (di != 0 || dj != 0) && (1 ≤ ni ≤ n_rows) && (1 ≤ nj ≤ n_cols)
            push!(nbrs, (ni, nj))
        end
    end
    return nbrs
end

"""
    build_neighbour_cache(grid_points, ocean_mask, n_rows, n_cols, Rmax, target_spacing_m;
                          heading_mode = :full)

One flat edge list per ocean node, ordered ring r = 1 .. Rmax. `:full` uses every
cell on each ring; `:eight` uses only the compass headings scaled by r.

ORDER MATTERS: climatology edge caches are indexed positionally against these
lists, so the offset generators below must not be reordered.
"""
function build_neighbour_cache(grid_points::Matrix{Tuple{Float64,Float64}},
                               ocean_mask::Matrix{Bool}, n_rows::Int, n_cols::Int,
                               Rmax::Int, target_spacing_m::Float64;
                               heading_mode::Symbol = :full)::NbCache
    heading_mode in (:full, :eight) ||
        error("heading_mode must be :full or :eight, got $heading_mode")

    nb_flat = NbCache(undef, n_rows, n_cols)
    for i in 1:n_rows, j in 1:n_cols
        nb_flat[i, j] = NbEntry[]
    end

    for i in 1:n_rows, j in 1:n_cols
        !ocean_mask[i, j] && continue
        lon1, lat1 = grid_points[i, j]
        buf = nb_flat[i, j]
        for r in 1:Rmax
            offsets = heading_mode === :eight ?
                ((ddi * r, ddj * r) for (ddi, ddj) in DIRECTIONS) :
                ((di, dj) for di in -r:r, dj in -r:r
                          if !(di == 0 && dj == 0) && max(abs(di), abs(dj)) == r)

            for (di, dj) in offsets
                ni, nj = i + di, j + dj
                (1 <= ni <= n_rows && 1 <= nj <= n_cols) || continue
                !ocean_mask[ni, nj] && continue
                !line_on_water_geo(i, j, ni, nj, ocean_mask; oversample = 10) && continue

                lon2, lat2 = grid_points[ni, nj]
                dist    = getDistance(lon1, lat1, lon2, lat2)
                bearing = getBoatAngle(lon1, lat1, lon2, lat2)
                n_samp  = max(1, round(Int, dist / target_spacing_m))
                arc_flats = Vector{Int}(undef, n_samp)
                for k in 1:n_samp
                    alpha = (k - 0.5) / n_samp
                    si = clamp(round(Int, i + alpha * (ni - i)), 1, n_rows)
                    sj = clamp(round(Int, j + alpha * (nj - j)), 1, n_cols)
                    arc_flats[k] = (sj - 1) * n_rows + si
                end
                push!(buf, (ni, nj, dist, bearing, arc_flats, Int8(r)))
            end
        end
    end
    return nb_flat
end

"""Index in each node's edge list where the outer ring (r == Rmax) starts."""
function build_outer_start(nb_flat::NbCache, Rmax::Int)::Matrix{Int32}
    outer = Matrix{Int32}(undef, size(nb_flat)...)
    for idx in CartesianIndices(nb_flat)
        buf = nb_flat[idx]
        k = findfirst(e -> e[6] == Rmax, buf)
        outer[idx] = Int32(k === nothing ? length(buf) + 1 : k)
    end
    return outer
end

# =============================================================================
# 9. Radius bands and calibration
#
# Each ocean node is assigned one of two states per wind epoch:
#   band == 1     near land OR high wind/direction gradient -> full ring range 1..Rmax
#   band == Rmax  far from land AND calm, steady wind       -> outer ring only
# "Near land" never caps hop length; it only adds short-radius headings.
# =============================================================================

"""Static distance-to-land (km), by a two-pass chamfer sweep in cells times the cell size."""
function distance_to_land_km(grid_points::Matrix{Tuple{Float64,Float64}}, ocean_mask::Matrix{Bool})
    n_rows, n_cols = size(grid_points)
    d = fill(Inf, n_rows, n_cols)
    for i in 1:n_rows, j in 1:n_cols
        !ocean_mask[i,j] && (d[i,j] = 0.0; continue)
        i > 1 && (d[i,j] = min(d[i,j], d[i-1,j] + 1.0))
        j > 1 && (d[i,j] = min(d[i,j], d[i,j-1] + 1.0))
    end
    for i in n_rows:-1:1, j in n_cols:-1:1
        !ocean_mask[i,j] && (d[i,j] = 0.0; continue)
        i < n_rows && (d[i,j] = min(d[i,j], d[i+1,j] + 1.0))
        j < n_cols && (d[i,j] = min(d[i,j], d[i,j+1] + 1.0))
    end
    lon1, lat1 = grid_points[1, 1]
    lon2, lat2 = grid_points[min(2, n_rows), min(2, n_cols)]
    cell_km = sqrt(((lat2 - lat1) * 111.0)^2 +
                   ((lon2 - lon1) * 111.0 * cos(deg2rad((lat1 + lat2) / 2)))^2)
    return d .* cell_km
end

"""Central-difference wind-speed gradient (m/s per cell) and wind-direction gradient (deg per cell); zero on the border."""
function _wind_gradients(u_flat::Vector{Float32}, v_flat::Vector{Float32}, n_rows::Int, n_cols::Int)
    spd = Matrix{Float64}(undef, n_rows, n_cols)
    cx  = Matrix{Float64}(undef, n_rows, n_cols)
    cy  = Matrix{Float64}(undef, n_rows, n_cols)
    @inbounds for j in 1:n_cols, i in 1:n_rows
        f = flat_idx(i, j, n_rows)
        u = Float64(u_flat[f]); v = Float64(v_flat[f])
        s = sqrt(u^2 + v^2)
        spd[i, j] = s
        inv_s = 1.0 / max(s, 1e-3)
        cx[i, j] = u * inv_s          # unit wind vector, so differences measure rotation
        cy[i, j] = v * inv_s
    end

    sg = zeros(n_rows, n_cols)
    dg = zeros(n_rows, n_cols)
    @inbounds for i in 2:n_rows-1, j in 2:n_cols-1
        sg[i, j] = sqrt(((spd[i+1, j] - spd[i-1, j]) / 2)^2 + ((spd[i, j+1] - spd[i, j-1]) / 2)^2)
        mx = sqrt(((cx[i+1, j] - cx[i-1, j]) / 2)^2 + ((cy[i+1, j] - cy[i-1, j]) / 2)^2)
        my = sqrt(((cx[i, j+1] - cx[i, j-1]) / 2)^2 + ((cy[i, j+1] - cy[i, j-1]) / 2)^2)
        dg[i, j] = rad2deg(sqrt(mx^2 + my^2))
    end
    return sg, dg
end

"""
Recompute `node_band` from the current UV field and the static `dist_km`.
calm = 0.5*(distance to land) + 0.25*(1 - speed gradient) + 0.25*(1 - direction gradient),
each normalised to [0,1]; calm >= calm_threshold selects the outer-ring-only band.
"""
function compute_radius_bands!(node_band::Matrix{Int8},
        u_flat::Vector{Float32}, v_flat::Vector{Float32},
        dist_km::Matrix{Float64}, ocean_mask::Matrix{Bool}, n_rows::Int, n_cols::Int,
        land_norm_km::Float64, wind_grad_max::Float64, dir_grad_max::Float64, Rmax::Int;
        calm_threshold::Float64 = 0.5)
    sg, dg = _wind_gradients(u_flat, v_flat, n_rows, n_cols)
    inv_l = 1.0 / land_norm_km
    inv_g = 1.0 / wind_grad_max
    inv_d = 1.0 / dir_grad_max
    @inbounds for i in 1:n_rows, j in 1:n_cols
        !ocean_mask[i, j] && continue
        calm = 0.5  * clamp(dist_km[i, j] * inv_l, 0.0, 1.0) +
               0.25 * (1.0 - clamp(sg[i, j] * inv_g, 0.0, 1.0)) +
               0.25 * (1.0 - clamp(dg[i, j] * inv_d, 0.0, 1.0))
        node_band[i, j] = calm >= calm_threshold ? Int8(Rmax) : Int8(1)
    end
    return nothing
end

"""Collect distance-to-land and wind-gradient samples over many epochs, to choose land_norm_km / wind_grad_max / dir_grad_max from data."""
function collect_calibration_stats(wind_pool::WindFieldPool, interp::GridInterpolator,
        dist_km::Matrix{Float64}, ocean_mask::Matrix{Bool}, epochs::Vector{Int})
    n_rows, n_cols = size(dist_km)
    n_flat = n_rows * n_cols
    u_buf = Vector{Float32}(undef, n_flat)
    v_buf = Vector{Float32}(undef, n_flat)
    all_sg = Float64[]
    all_dg = Float64[]

    for ep in epochs
        precompute_uv_grid!(u_buf, v_buf, _getField!(wind_pool, ep), interp)
        sg, dg = _wind_gradients(u_buf, v_buf, n_rows, n_cols)
        for i in 2:n_rows-1, j in 2:n_cols-1
            ocean_mask[i, j] || continue
            push!(all_sg, sg[i, j]); push!(all_dg, dg[i, j])
        end
    end
    all_dist_ocean = [dist_km[i, j] for i in 1:n_rows, j in 1:n_cols if ocean_mask[i, j]]
    return all_dist_ocean, all_sg, all_dg
end

function pct(v::Vector{<:Real}, p::Real)
    s = sort(v)
    return s[clamp(round(Int, p / 100 * length(s)), 1, length(s))]
end

report_percentiles(name::String, v::Vector{<:Real}) =
    println("$name: ", ["$(p)%=$(round(pct(v, p), digits=3))" for p in (50, 75, 90, 95, 99)])

# =============================================================================
# 10. The solver
#
# Labels are (node, energy bin). Each label stores the earliest arrival time with
# that cumulative energy; a Dijkstra pass over (time) with an energy-bin
# dominance rule yields the fastest route within `max_energy`.
# =============================================================================

_haversine_km(lon1, lat1, lon2, lat2) = begin
    dlon = deg2rad(lon2 - lon1); dlat = deg2rad(lat2 - lat1)
    a = sin(dlat / 2)^2 + cos(deg2rad(lat1)) * cos(deg2rad(lat2)) * sin(dlon / 2)^2
    6371.0 * 2 * atan(sqrt(a), sqrt(1 - a))
end

_secs_since(t0_ns) = (time_ns() - t0_ns) / 1e9

"""Per-solve counters and hierarchical timers (nanoseconds)."""
mutable struct SolverStats
    confirmed::Int; stale_pops::Int; queue_pushes::Int; epoch_changes::Int
    total_edges::Int; band_rebuilds::Int
    full_band_nodes::Int; outer_band_nodes::Int
    t_relax_ns::Float64; t_epoch_ns::Float64; t_uv_ns::Float64
    t_queue_ns::Float64; t_bands_ns::Float64; t_loop_ns::Float64
end
SolverStats() = SolverStats(0, 0, 1, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # 1 = the start push

"""Reused per-edge buffers."""
struct EdgeScratch
    ws::Vector{Float32};  twa::Vector{Float32}
    speed_tmp::Vector{Float32}
    t::Vector{Float64};   e::Vector{Float64};  vmin::Vector{Float64}
    feasible::Vector{Bool}
end
EdgeScratch(max_arc_samples::Int, n_power::Int) =
    EdgeScratch(Vector{Float32}(undef, max_arc_samples), Vector{Float32}(undef, max_arc_samples),
                Vector{Float32}(undef, n_power), Vector{Float64}(undef, n_power),
                Vector{Float64}(undef, n_power), Vector{Float64}(undef, n_power),
                Vector{Bool}(undef, n_power))

"""Everything the forecast-wind edge integrator needs that is constant for one solve."""
struct EdgeContext
    wind_pool::WindFieldPool;  interp::GridInterpolator;  uv_cache::UVGridCache
    cursor::WindTimeCursor;    wind_pair::TemporalWindPair
    slut::StackedLUT;          lut_ref::PolarLUT
    plevels_f::Vector{Float64}; min_speed::Float64
    route_start_abs::Int; cur_time::Int; cur_day::Int; cur_month::Int; cur_year::Int
    n_rows::Int; n_flat::Int
    time_aware::Bool; zero_wind::Bool
end

"""Fill sc.t / sc.e / sc.feasible for one edge from wind. Returns whether any power level is feasible."""
@inline function _integrate_forecast_edge!(sc::EdgeScratch, ctx::EdgeContext,
        arc_flats::Vector{Int}, dist::Float64, bearing::Float64,
        t_start_abs::Float64, omt::Float32, t_frac::Float32, use_stepped::Bool)::Bool
    n_samp   = length(arc_flats)
    seg_dist = dist / n_samp
    n_power  = ctx.slut.n_power

    if ctx.zero_wind                                   # one sample of 0/0
        sc.ws[1] = 0f0; sc.twa[1] = 0f0
        n_used, seg = 1, dist
    elseif use_stepped
        if ctx.time_aware
            _fill_edge_wind_time_aware!(sc.ws, sc.twa, ctx.cursor, ctx.wind_pool, ctx.interp,
                ctx.uv_cache, ctx.n_flat, arc_flats, bearing, t_start_abs, seg_dist, ctx.lut_ref,
                ctx.route_start_abs, ctx.cur_time, ctx.cur_day, ctx.cur_month, ctx.cur_year)
        else
            _fill_edge_wind_frozen!(sc.ws, sc.twa, ctx.wind_pair, arc_flats, bearing, omt, t_frac)
        end
        n_used, seg = n_samp, seg_dist
    else                                               # coarse 2-sample integration
        _fill_edge_wind_coarse2!(sc.ws, sc.twa, ctx.wind_pair, arc_flats, bearing, omt, t_frac)
        n_used, seg = 2, dist / 2
    end

    ok = _stepped_edge_all_powers!(sc.t, sc.e, sc.vmin, sc.speed_tmp, ctx.slut, sc.ws, sc.twa,
                                   n_used, seg, ctx.plevels_f, ctx.min_speed)
    _feasibility_from_vmin!(sc.feasible, sc.vmin, ok, ctx.min_speed, n_power)
    return ok
end

"""Past the forecast horizon: use the climatology's expected (time, energy) directly (no forecast sampling)."""
@inline function _load_climatology_edge!(sc::EdgeScratch, clim_edge, n_power::Int)::Bool
    any_feasible = false
    @inbounds for k in 1:n_power
        ct = clim_edge.t_exp[k]
        ok = isfinite(ct)
        sc.feasible[k] = ok
        if ok
            sc.t[k] = ct
            sc.e[k] = clim_edge.e_exp[k]
            any_feasible = true
        end
    end
    return any_feasible
end

"""Load the epoch bracket (epoch, next epoch) into `wp`, reusing the old t1 buffer as the new t0 when it matches."""
function _advance_wind_pair!(wp::TemporalWindPair, uv_cache::UVGridCache,
                             pool::WindFieldPool, interp::GridInterpolator, epoch::Int)
    sorted = pool.sorted_epochs
    idx = searchsortedfirst(sorted, epoch)
    if idx < length(sorted)
        next_ep = sorted[idx + 1]
        if wp.ready && wp.t1_epoch == epoch
            wp.u0, wp.u1 = wp.u1, wp.u0
            wp.v0, wp.v1 = wp.v1, wp.v0
        else
            u0, v0 = get_uv!(uv_cache, pool, interp, epoch)
            copy!(wp.u0, u0); copy!(wp.v0, v0)
        end
        wp.t0_epoch = epoch
        u1, v1 = get_uv!(uv_cache, pool, interp, next_ep)
        copy!(wp.u1, u1); copy!(wp.v1, v1)
        wp.t1_epoch = next_ep
        wp.ready    = true
    else
        u0, v0 = get_uv!(uv_cache, pool, interp, epoch)
        copy!(wp.u0, u0); copy!(wp.v0, v0)
        wp.t0_epoch = epoch
        wp.ready    = false
    end
    return nothing
end

function _reconstruct_path(prev_i, prev_j, prev_e, goal_idx, best_bin)
    path_idxs = Tuple{Int,Int,Int}[]
    ci, cj, ce = goal_idx[1], goal_idx[2], best_bin
    while true
        push!(path_idxs, (ci, cj, ce))
        pi_ = Int(prev_i[ci, cj, ce]); pj_ = Int(prev_j[ci, cj, ce]); pe_ = Int(prev_e[ci, cj, ce])
        (pi_ == 0 && pj_ == 0) && break
        ci, cj, ce = pi_, pj_, pe_
    end
    return reverse!(path_idxs)
end

"""
    findDijkstraRoutefinal(grid_points, start_idx, goal_idx, cur_time, cur_day,
                           cur_month, cur_year, max_energy, power_levels, bin_size,
                           min_speed; kwargs...)
        -> (path_coords, node_costs_s, cumulative_energy_kwh, power_used_kw)

Returns four empty vectors if no route fits the energy budget. Key keywords:
- `Rmax`, `heading_mode`        hop radius and heading set
- `stepped_wind`, `time_aware_wind`, `accurate_stepped_all_bands`
                                edge-integration fidelity (arc samples / mid-edge epoch / all bands)
- `force_band`, `dynamic_bands`, `calm_threshold`, `land_norm_km`, `wind_grad_max`, `dir_grad_max`
                                radius-band policy
- `zero_wind`                   isolate geometry from wind
- `record_diagnostics`, `diag_out`  per-confirmed-node records and a Dict filled for the stepper/replay
"""
function findDijkstraRoutefinal(
        grid_points    ::Matrix{Tuple{Float64, Float64}},
        start_idx      ::Tuple{Int, Int},
        goal_idx       ::Tuple{Int, Int},
        cur_time       ::Int,
        cur_day        ::Int,
        cur_month      ::Int,
        cur_year       ::Int,
        max_energy     ::Float64,
        power_levels   ::Vector{Float64},
        bin_size       ::Float64,
        min_speed      ::Float64;
        wind_cache_size::Int     = 4,                     # raw-field LRU size
        uv_cache_size  ::Union{Nothing,Int} = nothing,    # routing-grid UV epochs; default = voyage hours
        prewarm_uv     ::Bool    = false,
        time_aware_wind::Bool    = true,
        ocean_mask     ::Union{Nothing, Matrix{Bool}} = nothing,
        diagnostics    ::Bool    = true,
        Rmax           ::Int     = 4,
        land_norm_km   ::Float64 = 1317.25,
        wind_grad_max  ::Float64 = 1.004,
        dir_grad_max   ::Float64 = 16.34,
        calm_threshold ::Float64 = 0.5,
        force_band     ::Union{Nothing,Int} = nothing,
        dynamic_bands  ::Bool    = false,
        heading_mode   ::Symbol  = :full,
        detailed_setup_timing::Bool = true,
        stepped_wind   ::Bool    = true,
        accurate_stepped_all_bands::Bool = true,
        zero_wind      ::Bool    = false,
        record_diagnostics::Bool = true,
        diag_out ::Union{Nothing,Dict{Symbol,Any}} = nothing)

    n_rows, n_cols = size(grid_points)
    ocean_mask === nothing && (ocean_mask = fill(true, n_rows, n_cols))

    t_total_ns      = time_ns()
    route_start_abs = toEpochSeconds(cur_time, cur_day, cur_month, cur_year)
    setup_t = Dict{Symbol,Float64}()
    note_setup(name, t0) = (setup_t[name] = _secs_since(t0);
        detailed_setup_timing && println("  [setup] $(rpad(string(name), 18)): $(round(setup_t[name], digits=3))s"))

    # ---- 1. polar LUTs (sorted ascending: the early-exit relies on it) -------
    t0 = time_ns()
    power_sorted = sort(power_levels)
    n_power      = length(power_sorted)
    plevels_f    = Float64.(power_sorted)
    luts         = PolarLUT[load_polar_lut(P) for P in power_sorted]
    slut         = build_stacked_lut(luts)
    note_setup(:power_model, t0)
    println("Discrete power mode: $n_power fixed levels: $power_levels kW")

    # ---- 2. static distance to land -----------------------------------------
    t0 = time_ns()
    dist_km = distance_to_land_km(grid_points, ocean_mask)
    note_setup(:dist_km, t0)

    node_band = fill(Int8(Rmax), n_rows, n_cols)

    # ---- 3. wind pool, interpolator, initial bracket ------------------------
    t0 = time_ns()
    route_gc_km   = _haversine_km(grid_points[start_idx...]..., grid_points[goal_idx...]...)
    max_hours_est = ceil(Int, route_gc_km / min_speed * 1.5) + 2    # generous: km/min_speed over-estimates hours
    uv_cap        = uv_cache_size === nothing ? (max_hours_est + 2) : uv_cache_size

    wind_pool = WindFieldPool(; max_cached = wind_cache_size)
    init_ep   = _ensureEpochRegistered!(wind_pool, cur_time, cur_day, cur_month, cur_year, 0)
    for hh in 1:max_hours_est
        _ensureEpochRegistered!(wind_pool, cur_time, cur_day, cur_month, cur_year, hh * 3600)
    end
    note_setup(:wind_registry, t0)
    println("Pre-registered $(length(wind_pool.sorted_epochs)) epochs " *
            "(raw GRIB LRU=$wind_cache_size, UV cache cap=$uv_cap)")

    t0 = time_ns()
    interp = build_interpolator(grid_points, _getField!(wind_pool, init_ep))
    note_setup(:interpolator, t0)

    n_flat   = n_rows * n_cols
    uv_cache = UVGridCache(n_flat; max_cached = uv_cap)
    if prewarm_uv && !zero_wind
        t0 = time_ns()
        prewarm_uv_cache!(uv_cache, wind_pool, interp)
        note_setup(:uv_prewarm, t0)
    end

    wind_pair = TemporalWindPair(
        Vector{Float32}(undef, n_flat), Vector{Float32}(undef, n_flat),
        Vector{Float32}(undef, n_flat), Vector{Float32}(undef, n_flat), 0, 0, false)
    _advance_wind_pair!(wind_pair, uv_cache, wind_pool, interp, init_ep)

    # ---- initial radius bands -----------------------------------------------
    t0 = time_ns()
    if force_band !== nothing
        fb = Int8(force_band)
        @inbounds for i in 1:n_rows, j in 1:n_cols
            ocean_mask[i, j] && (node_band[i, j] = fb)
        end
    else
        compute_radius_bands!(node_band, wind_pair.u0, wind_pair.v0, dist_km, ocean_mask,
                              n_rows, n_cols, land_norm_km, wind_grad_max, dir_grad_max, Rmax;
                              calm_threshold = calm_threshold)
    end
    note_setup(:initial_bands, t0)

    # ---- 4. neighbour cache ---------------------------------------------------
    t0 = time_ns()
    nb_flat        = build_neighbour_cache(grid_points, ocean_mask, n_rows, n_cols, Rmax,
                                           27_800.0; heading_mode = heading_mode)
    nb_outer_start = build_outer_start(nb_flat, Rmax)
    note_setup(:neighbour_cache, t0)
    println("Neighbour cache built (Rmax=$Rmax, heading_mode=$heading_mode)")

    # ---- 5. labels, queue, scratch --------------------------------------------
    num_bins  = ceil(Int, max_energy / bin_size) + 1
    start_bin = 1

    labels_time   = fill(Inf,      n_rows, n_cols, num_bins)
    labels_energy = fill(Inf,      n_rows, n_cols, num_bins)
    labels_power  = fill(-1.0,     n_rows, n_cols, num_bins)
    queued_time   = fill(Inf,      n_rows, n_cols, num_bins)
    prev_i        = fill(Int32(0), n_rows, n_cols, num_bins)
    prev_j        = fill(Int32(0), n_rows, n_cols, num_bins)
    prev_e        = fill(Int32(0), n_rows, n_cols, num_bins)

    si, sj = start_idx
    labels_time[si, sj, start_bin]   = 0.0
    labels_energy[si, sj, start_bin] = 0.0
    queued_time[si, sj, start_bin]   = 0.0

    pq = BinaryMinHeap{Tuple{Float64,Int32,Int32,Int32}}()
    push!(pq, (0.0, Int32(si), Int32(sj), Int32(start_bin)))

    node_band_confirm     = Dict{Tuple{Int,Int,Int}, Int8}()
    node_relax_ns_confirm = Dict{Tuple{Int,Int,Int}, Float64}()
    node_epoch_confirm    = Dict{Tuple{Int,Int,Int}, Int}()
    node_confirm_order    = Dict{Tuple{Int,Int,Int}, Int}()

    max_arc_samples = max(2, maximum(length(e[5]) for col in nb_flat for e in col; init = 1))
    scratch = EdgeScratch(max_arc_samples, n_power)
    ctx = EdgeContext(wind_pool, interp, uv_cache, WindTimeCursor(), wind_pair, slut, luts[end],
                      plevels_f, min_speed, route_start_abs, cur_time, cur_day, cur_month, cur_year,
                      n_rows, n_flat, time_aware_wind, zero_wind)

    clim_cfg      = _CLIMATOLOGY_BLEND[]
    stats         = SolverStats()
    hour_to_epoch = Dict{Int,Int}()
    last_epoch    = -1
    start_minute  = cur_time % 100
    inv_bin_size  = 1.0 / bin_size
    setup_total_ns = time_ns() - t_total_ns

    # ---- 6. main loop ---------------------------------------------------------
    while !isempty(pq)
        t_iter = time_ns()

        t0 = time_ns()
        cur_cost, i32, j32, eb32 = pop!(pq)
        stats.t_queue_ns += time_ns() - t0

        i = Int(i32); j = Int(j32); e_bin = Int(eb32)

        if cur_cost > labels_time[i, j, e_bin]               # stale heap entry
            stats.stale_pops += 1
            stats.t_loop_ns  += time_ns() - t_iter
            continue
        end

        stats.confirmed += 1
        if stats.confirmed % 1_000_000 == 0
            println("Confirmed: $(stats.confirmed) | node=($i,$j,bin=$e_bin) | " *
                    "elapsed=$(round(cur_cost/3600, digits=1))h | " *
                    "UV=$(length(uv_cache.entries))/$uv_cap raw=$(length(wind_pool.loaded_fields))/$wind_cache_size")
        end

        (i, j) == goal_idx && break

        elapsed_int = round(Int, cur_cost)
        cur_energy  = labels_energy[i, j, e_bin]

        # -- wind epoch for this node's departure time --
        t0 = time_ns()
        hour_bucket = fld(start_minute * 60 + elapsed_int + 1800, 3600)
        epoch = get_epoch_for_bucket!(hour_to_epoch, wind_pool, cur_time, cur_day, cur_month,
                                      cur_year, hour_bucket, elapsed_int)
        stats.t_epoch_ns += time_ns() - t0

        if epoch != last_epoch
            t0 = time_ns()
            _advance_wind_pair!(wind_pair, uv_cache, wind_pool, interp, epoch)
            stats.t_uv_ns += time_ns() - t0
            ctx.cursor.has_bracket = false
            last_epoch = epoch
            stats.epoch_changes += 1

            if force_band === nothing && dynamic_bands
                t0 = time_ns()
                compute_radius_bands!(node_band, wind_pair.u0, wind_pair.v0, dist_km, ocean_mask,
                                      n_rows, n_cols, land_norm_km, wind_grad_max, dir_grad_max, Rmax;
                                      calm_threshold = calm_threshold)
                stats.t_bands_ns += time_ns() - t0
                stats.band_rebuilds += 1
            end
        end

        omt, t_frac = _bracket_fraction(wind_pair, route_start_abs + elapsed_int)

        band = Int(node_band[i, j])
        band == 1 ? (stats.full_band_nodes += 1) : (stats.outer_band_nodes += 1)
        if record_diagnostics
            node_band_confirm[(i, j, e_bin)]  = Int8(band)
            node_epoch_confirm[(i, j, e_bin)] = epoch
            node_confirm_order[(i, j, e_bin)] = stats.confirmed
        end

        use_stepped = stepped_wind && !zero_wind && (accurate_stepped_all_bands || band == 1)
        t_start_abs = Float64(route_start_abs + elapsed_int)
        past_horizon = clim_cfg !== nothing && !zero_wind &&
            !climatology_uses_forecast(elapsed_int; full_forecast_days = clim_cfg.full_forecast_days)

        # -- relax edges --
        t0 = time_ns()
        flat = nb_flat[i, j]
        lo   = band == 1 ? 1 : Int(nb_outer_start[i, j])
        nbrs = view(flat, lo:length(flat))
        clim_row = clim_cfg === nothing ? nothing :
                   view(clim_cfg.clim_edge_cache[i, j], lo:length(flat))
        stats.total_edges += length(nbrs)

        for (edge_ord, (ni, nj, dist, bearing, arc_flats, _ring)) in enumerate(nbrs)
            any_feasible = past_horizon ?
                _load_climatology_edge!(scratch, clim_row[edge_ord], n_power) :
                _integrate_forecast_edge!(scratch, ctx, arc_flats, dist, bearing,
                                          t_start_abs, omt, t_frac, use_stepped)
            any_feasible || continue

            for k in 1:n_power
                scratch.feasible[k] || continue

                new_time   = cur_cost + scratch.t[k]
                new_energy = cur_energy + scratch.e[k]
                new_energy > max_energy && continue

                new_bin = unsafe_trunc(Int, new_energy * inv_bin_size) + 1
                (new_bin < 1 || new_bin > num_bins) && continue
                new_time >= labels_time[ni, nj, new_bin] && continue

                labels_time[ni, nj, new_bin]   = new_time
                labels_energy[ni, nj, new_bin] = new_energy
                labels_power[ni, nj, new_bin]  = plevels_f[k]
                prev_i[ni, nj, new_bin]        = Int32(i)
                prev_j[ni, nj, new_bin]        = Int32(j)
                prev_e[ni, nj, new_bin]        = Int32(e_bin)

                if new_time < queued_time[ni, nj, new_bin]
                    queued_time[ni, nj, new_bin] = new_time
                    tq = time_ns()
                    push!(pq, (new_time, Int32(ni), Int32(nj), Int32(new_bin)))
                    stats.t_queue_ns += time_ns() - tq
                    stats.queue_pushes += 1
                end
            end
        end
        relax_ns = time_ns() - t0
        stats.t_relax_ns += relax_ns
        record_diagnostics && (node_relax_ns_confirm[(i, j, e_bin)] = relax_ns)

        stats.t_loop_ns += time_ns() - t_iter
    end

    # ---- 7. path reconstruction -----------------------------------------------
    t0 = time_ns()
    gi, gj    = goal_idx
    best_bin  = 0
    best_time = Inf
    for eb in 1:num_bins
        labels_time[gi, gj, eb] < best_time && (best_time = labels_time[gi, gj, eb]; best_bin = eb)
    end

    if best_bin == 0
        println("No feasible path found within energy budget.")
        return [], [], [], []
    end

    path_idxs    = _reconstruct_path(prev_i, prev_j, prev_e, goal_idx, best_bin)
    path_coords  = [grid_points[ii, jj]       for (ii, jj, _)  in path_idxs]
    node_costs   = [labels_time[ii, jj, eb]   for (ii, jj, eb) in path_idxs]
    final_energy = [labels_energy[ii, jj, eb] for (ii, jj, eb) in path_idxs]
    # power chosen on the edge INTO each node, shifted so power_used[k] is the leg k -> k+1
    power_used   = vcat([labels_power[ii, jj, eb] for (ii, jj, eb) in path_idxs][2:end], 0.0)
    t_path_ns    = time_ns() - t0

    if diag_out !== nothing
        merge!(diag_out, Dict{Symbol,Any}(
            :path_idxs => path_idxs, :nb_flat => nb_flat, :ocean_mask => ocean_mask,
            :node_band_confirm => node_band_confirm, :node_relax_ns_confirm => node_relax_ns_confirm,
            :node_epoch_confirm => node_epoch_confirm, :node_confirm_order => node_confirm_order,
            :node_band_final => node_band,
            # lets the simulator/replay rebuild the wind for any elapsed time on the route
            :wind_pool => wind_pool, :uv_cache => uv_cache, :interp => interp,
            :grid_points => grid_points,
            :cur_time => cur_time, :cur_day => cur_day, :cur_month => cur_month, :cur_year => cur_year,
            :labels_power => labels_power, :total_confirmed => stats.confirmed,
            :total_loop_time_ns => stats.t_loop_ns))
    end

    println("Path found: $(length(path_coords)) waypoints | " *
            "time=$(round(best_time/3600, digits=2))h | " *
            "energy=$(round(final_energy[end], digits=1)) kWh / $(max_energy) kWh budget")

    diagnostics && _print_solver_diagnostics(stats, setup_t, setup_total_ns, t_path_ns,
        time_ns() - t_total_ns, uv_cache, wind_pool, uv_cap, wind_cache_size, Rmax;
        detailed_setup_timing, prewarm_uv = prewarm_uv && !zero_wind, force_band, dynamic_bands,
        calm_threshold, stepped_wind, time_aware_wind, zero_wind)

    return path_coords, node_costs, final_energy, power_used
end

function _print_solver_diagnostics(stats::SolverStats, setup_t, setup_total_ns, t_path_ns, total_ns,
        uv_cache, wind_pool, uv_cap, raw_cap, Rmax;
        detailed_setup_timing, prewarm_uv, force_band, dynamic_bands, calm_threshold,
        stepped_wind, time_aware_wind, zero_wind)
    s(x) = round(x / 1e9, digits = 2)
    t_loop = stats.t_loop_ns
    known  = stats.t_relax_ns + stats.t_epoch_ns + stats.t_uv_ns + stats.t_queue_ns + stats.t_bands_ns
    c      = max(stats.confirmed, 1)

    println("\n=== Hierarchical timing breakdown (seconds) ===")
    println("  total wall time            : $(s(total_ns))")
    println("  ├─ setup (pre-loop)        : $(s(setup_total_ns))")
    if detailed_setup_timing
        for name in (:power_model, :dist_km, :wind_registry, :interpolator, :uv_prewarm, :initial_bands, :neighbour_cache)
            haskey(setup_t, name) && println("  │  ├─ $(rpad(string(name), 16)): $(round(setup_t[name], digits=2))")
        end
    end
    println("  ├─ while-loop total        : $(s(t_loop))")
    println("  │  ├─ edge relaxation      : $(s(stats.t_relax_ns))")
    println("  │  ├─ epoch bucket lookup  : $(s(stats.t_epoch_ns))")
    println("  │  ├─ UV rebuild           : $(s(stats.t_uv_ns))")
    println("  │  ├─ queue operations     : $(s(stats.t_queue_ns))")
    println("  │  ├─ radius band rebuild  : $(s(stats.t_bands_ns))s over $(stats.band_rebuilds) calls")
    println("  │  └─ loop overhead (other): $(s(t_loop - known))")
    println("  └─ path reconstruction     : $(s(t_path_ns))")
    println("\n  Sum of all parts          : $(s(setup_total_ns + t_loop + t_path_ns)) (should equal total)")

    println("\n=== Search efficiency ===")
    println("  confirmed states     : $(stats.confirmed)")
    println("  stale pops           : $(stats.stale_pops)")
    println("  queue pushes         : $(stats.queue_pushes)")
    println("  epoch changes        : $(stats.epoch_changes)")
    println("  stale/confirmed      : $(round(stats.stale_pops / c, digits=2))")
    println("  total edges relaxed  : $(stats.total_edges)  ($(round(stats.total_edges / c, digits=2)) edges/node avg)")

    println("\n=== Radius band distribution ===")
    println("  full range (1..$Rmax), near land/volatile : $(stats.full_band_nodes) nodes ($(round(100.0 * stats.full_band_nodes / c, digits=1))%)")
    println("  outer ring only ($Rmax), calm/open water   : $(stats.outer_band_nodes) nodes ($(round(100.0 * stats.outer_band_nodes / c, digits=1))%)")
    (force_band !== nothing || !dynamic_bands) && println("\n  (band logic was overridden -- see test hooks below)")

    uv_total = uv_cache.hits + uv_cache.misses
    uv_hit   = uv_total > 0 ? round(100.0 * uv_cache.hits / uv_total, digits = 1) : 100.0
    println("\n=== Test-hook state ===")
    println("  force_band       : $(force_band === nothing ? "off (dynamic)" : force_band)")
    println("  dynamic_bands    : $dynamic_bands")
    println("  calm_threshold   : $calm_threshold")
    println("  stepped_wind     : $stepped_wind")
    println("  time_aware_wind  : $time_aware_wind")
    println("  prewarm_uv       : $prewarm_uv")
    println("  zero_wind        : $zero_wind")
    println("  UV cache         : $(length(uv_cache.entries))/$uv_cap epochs, " *
            "hits=$(uv_cache.hits) misses=$(uv_cache.misses) ($uv_hit% hit rate)")
    println("  raw GRIB LRU     : $(length(wind_pool.loaded_fields))/$raw_cap")
    flush(stdout)
end