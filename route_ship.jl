using Dates
using CSV
using DataFrames
using Shapefile, LibGEOS, GeoInterface
using Statistics
using JLD2: jldopen, jldsave

include("discritise_domain.jl")
include("routing.jl")
include("plots.jl")


# =============================================================================
# ocean_mask_v2.jl
#
# Rasteriser that builds a navigable-water mask from GSHHG LAND polygons
# (not Natural Earth OCEAN polygons) and inverts appropriately. Feed this
# GSHHS_*_L1.shp (land), never ne_10m_ocean.shp.
#
# Pipeline actually implemented:
#   1. Rasterize the land layer -> land_mask.
#   2. Coastal buffering is done as a VECTOR operation in a projected
#      (metric) CRS via LibGEOS.buffer, not raster erosion in grid-index
#      space. Raster erosion on a lon/lat grid is anisotropic (a "1 cell"
#      step is a different physical distance in lat vs lon) and, at
#      buffer_km=50 (the old default), silently strangles anything
#      narrower than ~100 km — which includes literally every shipping
#      channel on Earth. Default here is 2.0 km, not 50.
#   3. Exclusion-zone polygons (ports, anchorages, restricted areas) are
#      rasterized and ANDed out of the final navigable mask as a proper
#      second pass (the original _rasterize_shape! never accepted a
#      `set_value` kwarg, so that branch threw as soon as
#      exclusion_shp_path was used — fixed here).
#   4. Optional DEPTH MASK from GEBCO bathymetry (netCDF), so a laden
#      ship with min_draft_m can't be routed over a shoal just because
#      the cell is nominally "wet." (Loader is still a stub — wire up
#      NCDatasets.jl before using depth_nc_path.)
#
# NOTE: mixed-resolution detail patches and explicit channel-override
# polygons are NOT implemented in this version. If you need
# Houston-Ship-Channel-level detail on top of a coarse global layer,
# that needs to be added back in as a real feature, not documented as
# if it already existed.
# =============================================================================

# ---------------------------------------------------------------------------
# Small helpers — raw polygon fill (used both for base rasterisation and,
# separately, for rasterising the already-buffered land ring set)
# ---------------------------------------------------------------------------
function _grid_cache_key(lats::Vector{Float64}, lons::Vector{Float64}, tag::String)::String
    params = string(
        tag, "_",
        round(lats[1],         digits = 8), "_",
        round(lats[end],       digits = 8), "_",
        round(lats[2]-lats[1], digits = 10), "_",
        round(lons[1],         digits = 8), "_",
        round(lons[end],       digits = 8), "_",
        round(lons[2]-lons[1], digits = 10)
    )
    return string(hash(params), base = 16)
end

function _extract_rings(shape)::Vector{Vector{Tuple{Float64,Float64}}}
    pts   = shape.points
    n_pts = length(pts)
    n_pts == 0 && return Vector{Tuple{Float64,Float64}}[]
    part_starts = vcat(Int.(shape.parts), n_pts)
    n_rings     = length(shape.parts)
    rings = Vector{Vector{Tuple{Float64,Float64}}}(undef, n_rings)
    for r in 1:n_rings
        i0 = part_starts[r] + 1
        i1 = part_starts[r + 1]
        rings[r] = [(Float64(pts[k].x), Float64(pts[k].y)) for k in i0:i1]
    end
    return rings
end

function _accumulate_crossings!(
    crossings::Vector{Vector{Float64}},
    lats::Vector{Float64},
    ring::Vector{Tuple{Float64,Float64}}
)
    n = length(ring)
    n < 3 && return
    for k in 1:n
        x1, y1 = ring[k]
        x2, y2 = ring[mod1(k + 1, n)]
        y1 == y2 && continue
        y_lo, y_hi = y1 < y2 ? (y1, y2) : (y2, y1)
        i_lo = searchsortedfirst(lats, y_lo)
        i_hi = searchsortedlast(lats,  y_hi)
        while i_hi >= i_lo && lats[i_hi] >= y_hi
            i_hi -= 1
        end
        i_lo > i_hi && continue
        inv_dy = 1.0 / (y2 - y1)
        for i in i_lo:i_hi
            t = (lats[i] - y1) * inv_dy
            push!(crossings[i], x1 + t * (x2 - x1))
        end
    end
end

function _fill_rows!(
    mask::BitMatrix,
    lons::Vector{Float64},
    crossings::Vector{Vector{Float64}};
    lon_lo::Float64 = -Inf,
    lon_hi::Float64 = Inf
)
    n_lats = length(crossings)
    @inbounds for i in 1:n_lats
        row = crossings[i]
        isempty(row) && continue
        sort!(row)
        k = 1
        while k + 1 <= length(row)
            seg_lo = max(row[k],   lon_lo)
            seg_hi = min(row[k+1], lon_hi)
            if seg_lo < seg_hi
                j_lo = searchsortedfirst(lons, seg_lo)
                j_hi = searchsortedlast(lons,  seg_hi)
                for j in j_lo:j_hi
                    mask[i, j] ⊻= true
                end
            end
            k += 2
        end
    end
end

# rasterize_shapes! ORs land polygons into `mask` (true = inside polygon).
# lon_lo/lon_hi restrict the fill to a bbox (kept for callers that want it;
# unused by the default single-resolution pipeline below).
function _rasterize_shapes!(mask::BitMatrix, lats::Vector{Float64},
                             lons::Vector{Float64}, shapes;
                             lon_lo::Float64 = -Inf, lon_hi::Float64 = Inf)
    for shape in shapes
        rings = _extract_rings(shape)
        isempty(rings) && continue
        crossings = [Float64[] for _ in 1:length(lats)]
        for ring in rings
            _accumulate_crossings!(crossings, lats, ring)
        end
        _fill_rows!(mask, lons, crossings; lon_lo = lon_lo, lon_hi = lon_hi)
    end
end

# ---------------------------------------------------------------------------
# Vector coastal buffer via LibGEOS — replaces raster-erosion buffering.
#
# Buffers LAND polygons outward by buffer_km in a locally-metric projection
# (simple equirectangular scaling around the domain's mean latitude — good
# enough for buffer widths of a few km to a few tens of km; for anything
# requiring survey-grade accuracy, project to UTM instead).
#
# Each shapefile *shape* can carry several rings (multiple disjoint land
# masses and/or holes such as inland lakes). Rings are grouped into proper
# (exterior, [holes...]) polygons using the shapefile ring-orientation
# convention (clockwise = exterior, counter-clockwise = hole) before being
# handed to GEOS, so a lake ring is not buffered as if it were its own
# separate island of land.
# ---------------------------------------------------------------------------
function _ring_signed_area(ring::Vector{Tuple{Float64,Float64}})::Float64
    n = length(ring)
    n < 3 && return 0.0
    a = 0.0
    for i in 1:n
        x1, y1 = ring[i]
        x2, y2 = ring[mod1(i + 1, n)]
        a += x1 * y2 - x2 * y1
    end
    return a / 2
end

function _close_ring(ring::Vector{Tuple{Float64,Float64}})
    (isempty(ring) || ring[1] == ring[end]) && return ring
    return vcat(ring, [ring[1]])
end

# Groups a shape's rings into (exterior, holes) polygons. Shapefile spec:
# outer rings are clockwise (negative signed area with the shoelace formula
# used above), inner (hole) rings are counter-clockwise (positive). Falls
# back to "every ring is its own exterior, no holes" if a shape has no
# clockwise ring at all (defensive — shouldn't happen for well-formed
# GSHHG data, but better than silently dropping geometry).
function _group_rings_into_polygons(rings::Vector{Vector{Tuple{Float64,Float64}}})
    polys = Vector{Tuple{Vector{Tuple{Float64,Float64}},Vector{Vector{Tuple{Float64,Float64}}}}}()
    exterior_idx = Int[]
    for (i, r) in enumerate(rings)
        length(r) >= 3 && _ring_signed_area(r) < 0 && push!(exterior_idx, i)
    end
    if isempty(exterior_idx)
        for r in rings
            length(r) >= 3 && push!(polys, (r, Vector{Vector{Tuple{Float64,Float64}}}()))
        end
        return polys
    end
    for (k, ext_i) in enumerate(exterior_idx)
        next_ext  = k < length(exterior_idx) ? exterior_idx[k + 1] : length(rings) + 1
        holes     = Vector{Vector{Tuple{Float64,Float64}}}()
        for j in (ext_i + 1):(next_ext - 1)
            length(rings[j]) >= 3 && push!(holes, rings[j])
        end
        push!(polys, (rings[ext_i], holes))
    end
    return polys
end

# Extracts exterior-ring point lists ([x,y] pairs, in whatever CRS the input
# geometry is in) from a LibGEOS buffer result, which may come back as
# either a Polygon or a MultiPolygon depending on whether nearby land
# features merged during buffering. Dispatches on the concrete LibGEOS
# geometry type instead of guessing from GeoInterface.coordinates' nested
# Vector{<:Vector} shape, which can't reliably distinguish the two cases
# (a plain Polygon's ring-of-points is itself a Vector{<:Vector}).
function _exterior_rings(geom)::Vector{Vector{Vector{Float64}}}
    if geom isa LibGEOS.Polygon
        coords = GeoInterface.coordinates(geom)   # Vector{ring}, ring = Vector{[x,y]}
        return isempty(coords) ? Vector{Vector{Float64}}[] : [coords[1]]
    elseif geom isa LibGEOS.MultiPolygon
        coords = GeoInterface.coordinates(geom)   # Vector{polygon}, polygon = Vector{ring}
        return [pc[1] for pc in coords if !isempty(pc)]
    else
        return Vector{Vector{Float64}}[]
    end
end

function _buffer_land_polygons(shapes, mean_lat_deg::Float64, buffer_km::Float64)
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * cos(deg2rad(mean_lat_deg))

    to_xy(lon, lat)  = (lon * m_per_deg_lon, lat * m_per_deg_lat)
    to_lonlat(x, y)  = (x / m_per_deg_lon, y / m_per_deg_lat)

    buffered_rings = Vector{Vector{Tuple{Float64,Float64}}}()

    for shape in shapes
        rings = _extract_rings(shape)
        isempty(rings) && continue

        for (ext, holes) in _group_rings_into_polygons(rings)
            ext_xy   = _close_ring([to_xy(p[1], p[2]) for p in ext])
            length(ext_xy) < 4 && continue  # need >=3 distinct points + closing point

            coords = Vector{Vector{Vector{Float64}}}()
            push!(coords, [[p[1], p[2]] for p in ext_xy])
            for h in holes
                h_xy = _close_ring([to_xy(p[1], p[2]) for p in h])
                length(h_xy) >= 4 && push!(coords, [[p[1], p[2]] for p in h_xy])
            end

            poly     = LibGEOS.Polygon(coords)
            buffered = LibGEOS.buffer(poly, buffer_km * 1000.0)

            for ext_ring_xy in _exterior_rings(buffered)
                push!(buffered_rings, [to_lonlat(c[1], c[2]) for c in ext_ring_xy])
            end
        end
    end
    return buffered_rings
end

function _rasterize_ring_set!(mask::BitMatrix, lats::Vector{Float64},
                               lons::Vector{Float64},
                               rings::Vector{Vector{Tuple{Float64,Float64}}})
    crossings = [Float64[] for _ in 1:length(lats)]
    for ring in rings
        _accumulate_crossings!(crossings, lats, ring)
    end
    _fill_rows!(mask, lons, crossings)
end

# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------
"""
    build_ocean_mask_v2(lats, lons; kwargs...)

Build a navigable-water mask on the given lat/lon grid.

# Required data
- `land_shp_path`      : path to a GSHHG land polygon shapefile, e.g.
                         "OceanData/GSHHS_i_L1.shp" ('i' = intermediate
                         resolution — good default for basin-scale routing;
                         use 'f' (full) only over small domains, it is
                         expensive at global/ocean-basin extent).

# Optional
- `exclusion_shp_paths`: Vector of shapefile paths (ports, anchorages,
                         restricted areas) rasterized and ANDed OUT of the
                         final navigable mask. Applied last.
- `coastal_buffer_km`  : vector buffer distance from land, in km. Default 2.0
                         (NOT 50 — 50 km will close every strait and channel
                         in a domain like the Gulf).
- `depth_nc_path`      : optional GEBCO netCDF bathymetry file.
- `min_draft_m`        : minimum water depth required (positive number of
                         metres), used with depth_nc_path.
- `cache_dir`          : where to store the JLD2 cache.

Returns `Matrix{Bool}`, true = navigable water.
"""
function build_ocean_mask_v2(
    lats::AbstractVector,
    lons::AbstractVector;
    land_shp_path::String,
    exclusion_shp_paths::Vector{String} = String[],
    coastal_buffer_km::Float64 = 2.0,
    depth_nc_path::Union{Nothing,String} = nothing,
    min_draft_m::Float64 = 0.0,
    cache_dir::String = "OceanData"
)::Matrix{Bool}

    lats_f = collect(Float64, lats)
    lons_f = collect(Float64, lons)

    key        = _grid_cache_key(lats_f, lons_f,
                    "v2_$(land_shp_path)_$(coastal_buffer_km)_$(min_draft_m)_" *
                    "$(join(exclusion_shp_paths, "|"))_$(something(depth_nc_path, "none"))")
    cache_path = joinpath(cache_dir, "ocean_mask_v2_$(key).jld2")

    if isfile(cache_path)
        print("Ocean mask v2: loading cache $(basename(cache_path))... ")
        flush(stdout)
        t0 = time()
        mask = jldopen(cache_path, "r") do f
            Matrix{Bool}(f["mask"])
        end
        n_ocean = count(mask); n_total = length(mask)
        println("done in $(round(time()-t0, digits=2)) s  " *
                "[$(n_ocean)/$(n_total) navigable " *
                "($(round(100.0*n_ocean/n_total, digits=1))%)]")
        return mask
    end

    println("Ocean mask v2: cache miss — building from GSHHG")
    println("  Grid : $(length(lats_f)) × $(length(lons_f))")
    println("  Land layer   : $land_shp_path")
    println("  Coastal buffer: $(coastal_buffer_km) km")
    flush(stdout)
    t_total = time()

    # -------------------------------------------------------------------
    # 1. Rasterize LAND layer -> land_mask (true = land)
    # -------------------------------------------------------------------
    land_mask = falses(length(lats_f), length(lons_f))
    table  = Shapefile.Table(land_shp_path)
    shapes = collect(Shapefile.shapes(table))
    println("  Loaded $(length(shapes)) land polygon(s).")
    _rasterize_shapes!(land_mask, lats_f, lons_f, shapes)

    n_land = count(land_mask)
    println("  Land cells: $n_land / $(length(land_mask)) " *
            "($(round(100*n_land/length(land_mask), digits=1))%)")

    # -------------------------------------------------------------------
    # 2. Vector buffer of land -> buffered_land_mask, done in a metric CRS
    # -------------------------------------------------------------------
    ocean_mask = .!land_mask
    if coastal_buffer_km > 0
        println("  Buffering land by $(coastal_buffer_km) km (vector, LibGEOS)...")
        mean_lat = mean(lats_f)
        buffered_rings = _buffer_land_polygons(shapes, mean_lat, coastal_buffer_km)
        buffered_land  = falses(size(land_mask)...)
        _rasterize_ring_set!(buffered_land, lats_f, lons_f, buffered_rings)
        ocean_mask = .!buffered_land
    end

    # -------------------------------------------------------------------
    # 3. Apply exclusion zones (ports, anchorages, restricted areas) —
    #    proper AND-out pass.
    # -------------------------------------------------------------------
    for excl_path in exclusion_shp_paths
        println("  Applying exclusion zone: $excl_path")
        table_e  = Shapefile.Table(excl_path)
        shapes_e = collect(Shapefile.shapes(table_e))
        excl_mask = falses(size(ocean_mask)...)
        _rasterize_shapes!(excl_mask, lats_f, lons_f, shapes_e)
        ocean_mask .&= .!excl_mask
    end

    # -------------------------------------------------------------------
    # 4. Optional depth filter (GEBCO bathymetry)
    # -------------------------------------------------------------------
    if depth_nc_path !== nothing && min_draft_m > 0
        println("  Applying depth mask: min_draft_m=$min_draft_m from $depth_nc_path")
        depth_m = _load_gebco_depth(depth_nc_path, lats_f, lons_f)
        ocean_mask .&= (depth_m .<= -min_draft_m)
    end

    result  = Matrix{Bool}(ocean_mask)
    n_ocean = count(result); n_total = length(result)
    println("Ocean mask v2 complete in $(round(time()-t_total, digits=1)) s — " *
            "$n_ocean/$n_total navigable ($(round(100.0*n_ocean/n_total, digits=1))%)")

    mkpath(cache_dir)
    jldsave(cache_path; mask = result)
    println("  Cache saved to $(basename(cache_path)).")
    return result
end

# ---------------------------------------------------------------------------
# GEBCO bathymetry loader (stub — wire up to NCDatasets.jl)
#
# GEBCO stores elevation in metres, negative = below sea level. Returns a
# matrix interpolated onto (lats, lons); values > 0 are land/above sea level.
# ---------------------------------------------------------------------------
function _load_gebco_depth(nc_path::String, lats::Vector{Float64}, lons::Vector{Float64})
    error("_load_gebco_depth is a stub — add `using NCDatasets` and implement " *
          "grid interpolation from the GEBCO netCDF onto (lats, lons). " *
          "GEBCO 2024 grid: https://www.gebco.net/data_products/gridded_bathymetry_data/")
end


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------
function run_timed(f)
    t0 = time()
    out = f()
    return out, time() - t0
end

function nearest_index(vals, target)
    return nearest_sorted_index(vals, target)
end

# ---------------------------------------------------------------------------
# Shared grid + mask builder
# ---------------------------------------------------------------------------
function _build_grid_and_mask(;
    start_location, end_location,
    lon_step, lat_step, padding,
    land_shp_path::String = "OceanData/GSHHS_i_L1.shp",
    coastal_buffer_km::Float64 = 2.0,
    exclusion_shp_paths::Vector{String} = String[],
    depth_nc_path::Union{Nothing,String} = nothing,
    min_draft_m::Float64 = 0.0
)
    println("=== Grid ===")
    flush(stdout)
    lon_min = min(start_location[1], end_location[1])
    lon_max = max(start_location[1], end_location[1])
    lat_min = min(start_location[2], end_location[2])
    lat_max = max(start_location[2], end_location[2])

    longitudes  = collect((lon_min - padding * lon_step) : lon_step :
                          (lon_max + padding * lon_step))
    latitudes   = collect((lat_min - padding * lat_step) : lat_step :
                          (lat_max + padding * lat_step))
    grid_points = [(lon, lat) for lat in latitudes, lon in longitudes]
    n_rows, n_cols = size(grid_points)
    println("  $(n_rows) rows × $(n_cols) cols = $(n_rows*n_cols) points")
    println("  Lon: $(longitudes[1])° → $(longitudes[end])°  (step $(lon_step)°, padding $(padding*lon_step)°)")
    println("  Lat: $(latitudes[1])° → $(latitudes[end])°  (step $(lat_step)°, padding $(padding*lat_step)°)")
    flush(stdout)

    println("\n=== Ocean mask ===")
    flush(stdout)
    ocean_mask = build_ocean_mask_v2(
        latitudes, longitudes;
        land_shp_path       = land_shp_path,
        exclusion_shp_paths = exclusion_shp_paths,
        coastal_buffer_km   = coastal_buffer_km,
        depth_nc_path       = depth_nc_path,
        min_draft_m         = min_draft_m
    )

    return grid_points, latitudes, longitudes, ocean_mask
end

# ---------------------------------------------------------------------------
# Stats / IO helpers for power-bin outputs
# ---------------------------------------------------------------------------

# Returns (total_time_h, total_dist_km, avg_speed_kmh, total_energy_kwh).
function _route_stats_power(path, costs, energy, power)

    total_time = isempty(costs) ? 0.0 : costs[end]/3600
    total_dist = length(path) > 1 ?
        sum(getDistance(
            path[i][1], path[i][2],
            path[i+1][1], path[i+1][2]
        ) for i in 1:length(path)-1)/1000 :
        0.0

    avg_speed = total_dist / max(total_time,1e-9)

    total_energy = isempty(energy) ? 0.0 : energy[end]

    avg_power = total_energy / max(total_time,1e-9)

    max_power = maximum(power)

    energy_per_km = total_energy / max(total_dist,1e-9)

    return (
        total_time,
        total_dist,
        avg_speed,
        total_energy,
        avg_power,
        max_power,
        energy_per_km
    )
end

function _check_endpoints(ocean_mask, start_idx, goal_idx)
    !ocean_mask[start_idx...] &&
        @warn "Start $(start_idx) is on land — no path will be found."
    !ocean_mask[goal_idx...] &&
        @warn "Goal $(goal_idx) is on land — no path will be found."
end

# Save power-bin route: one row per waypoint.
# power_kw[k] = engine setting on the leg *leaving* node k (0.0 at goal).
function _save_powerbin_csv(path, costs, energy, power, filename)
    isempty(path) && return
    CSV.write(filename, DataFrame(
        lon        = [p[1] for p in path],
        lat        = [p[2] for p in path],
        time_s     = costs,
        energy_kwh = energy,
        power_kw   = power,
    ))
    println("Saved $filename")
    flush(stdout)
end
