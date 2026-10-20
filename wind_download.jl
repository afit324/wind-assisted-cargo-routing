include("times_and_dates.jl")
include("read_write_data.jl")   # brings in getWindData, getDistance, DotEnv, CDSAPI, GRIB, etc.

using Dates
using OrderedCollections
using ProgressMeter
using JLD2

# =============================================================================
# batched_downloader.jl
#
# Wired to your actual getWindData/CDSAPI.retrieve implementation AND to
# the actual routing grid built by _build_grid_and_mask / run_dynamic_band_
# convergence_hifi -- not two small boxes around the endpoints. The
# algorithm needs wind coverage across the whole corridor the router might
# explore (start/end + padding), so the download area is derived from the
# SAME start_location/end_location/padding/step you pass to the router,
# via route_area() below, which mirrors _build_grid_and_mask's own bbox math.
#
# Two changes from the original bulk_download_climatology.jl:
#   1. Area: global default -> route_area(...) corridor. Still a large cut
#      (e.g. ~31deg x 98.5deg for the Houston-London route vs 180x360
#      global) even though it's not a tiny box.
#   2. Batching: one CDS request per (year, month) with day/time as JSON
#      LISTS, instead of one request per single (day, time). CDS's own
#      guidance is to batch by month for hourly/sub-daily ERA5, and since
#      CDS only processes one request per account at a time, cutting
#      request COUNT (not request size) is what buys the speed.
#
# Also requests `grid` explicitly matching your routing `step`, so if you
# ever run coarser tests (step=0.5 instead of 0.25) you're not silently
# still downloading at ERA5's native 0.25 default.
# =============================================================================

const MANIFEST_NAME = "manifest.jld2"

"""
    route_area(start_location, end_location, padding, step) -> [north, west, south, east]

Mirrors the exact bbox math in `_build_grid_and_mask` so the wind download
area is always consistent with the actual routing grid -- the algorithm
needs wind coverage along the whole corridor between the two points (plus
padding for the solver to explore around), not just small boxes at the
two endpoints. Call this with the SAME start_location/end_location/
padding/step you pass to run_dynamic_band_convergence_hifi.
"""
function route_area(start_location, end_location, padding::Real, step::Real)
    lon_min = min(start_location[1], end_location[1])
    lon_max = max(start_location[1], end_location[1])
    lat_min = min(start_location[2], end_location[2])
    lat_max = max(start_location[2], end_location[2])
    pad_deg = padding * step
    north = lat_max + pad_deg
    south = lat_min - pad_deg
    west  = lon_min - pad_deg
    east  = lon_max + pad_deg
    return [north, west, south, east]
end

"""
    getWindDataMonth(output_name, year, month, days, times, area, grid_step)

Sibling of getWindData that requests a WHOLE MONTH in one CDS API call:
`days` and `times` are passed as JSON lists rather than single values, so
CDS returns one multi-message GRIB file covering every (day, time) in the
month instead of you making one request per (day, time) pair. `grid_step`
is passed explicitly so download resolution always matches whatever
spatial `step` you're actually routing at, rather than silently relying
on ERA5's 0.25 default.

# Arguments
- `output_name`: output file path.
- `year`, `month`: Ints.
- `days`: Vector{Int}, e.g. collect(1:daysinmonth(...)).
- `times`: Vector{String}, e.g. ["00:00","06:00","12:00","18:00"].
- `area`: [north, west, south, east] -- use route_area(...) for this.
- `grid_step`: Real, degrees -- should match the `step` you route with.
"""
function getWindDataMonth(output_name::String, year::Int, month::Int,
                           days::Vector{Int}, times::Vector{String}, area,
                           grid_step::Real)
    DotEnv.load!()

    dataset = "reanalysis-era5-single-levels"
    day_list = "[" * join(["\"$(lpad(d,2,'0'))\"" for d in days], ", ") * "]"

    request = """{
        "product_type": ["reanalysis"],
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind"
        ],
        "year": ["$year"],
        "month": ["$(lpad(month,2,'0'))"],
        "day": $day_list,
        "time": $times,
        "data_format": "grib",
        "download_format": "unarchived",
        "area": $area,
        "grid": [$grid_step, $grid_step]
    }"""
    print("Request: ", request)

    CDSAPI.retrieve(dataset, request, output_name)
end

function _load_manifest(manifest_path::String)
    isfile(manifest_path) || return OrderedDict{String,Int}()
    return jldopen(manifest_path, "r") do f
        haskey(f, "path_to_month") ? f["path_to_month"] : OrderedDict{String,Int}()
    end
end

function _save_manifest(manifest_path::String, manifest::OrderedDict{String,Int})
    jldopen(manifest_path, "w") do f
        f["path_to_month"] = manifest
    end
end

"""
    bulk_download_climatology_batched(; start_location, end_location, padding, step,
                                        start_year, end_year, times_per_day, months=nothing,
                                        out_dir="./ClimatologyOceanData/route")

One request per calendar month, one continuous area covering the whole
routing corridor (derived from start_location/end_location/padding/step
via route_area, matching _build_grid_and_mask exactly) -- instead of one
request per single day+time, and instead of separate small boxes per
endpoint. Resumable: existing monthly files are skipped, manifest
(monthly-file -> month) is preserved/rebuilt.

`times_per_day`: set from resolution_tests.jl's subsampling_aliasing_test,
as "HH:MM" strings, e.g. ["00:00","06:00","12:00","18:00"] for 6-hourly.
"""
function bulk_download_climatology_batched(;
        start_location,
        end_location,
        padding::Real,
        step::Real,
        start_year::Int,
        end_year::Int,
        times_per_day::Vector{String} = ["00:00", "06:00", "12:00", "18:00"],
        months::Union{Nothing,Vector{Int}} = nothing,
        out_dir::String = "./ClimatologyOceanData/route"
    )
    start_year <= end_year || error("start_year must be <= end_year")
    area = route_area(start_location, end_location, padding, step)

    mkpath(out_dir)
    manifest_path = joinpath(out_dir, MANIFEST_NAME)

    year_months = [(y, m) for y in start_year:end_year for m in 1:12 if
                   (months === nothing || m in months)]
    total_requests = length(year_months)

    total_downloaded = 0
    total_skipped = 0
    total_failed  = 0

    println("📊 Batched Climatology Download Summary:")
    println("   Year range: $start_year – $end_year " *
            "($(end_year - start_year + 1) years)")
    println("   Cadence: $(length(times_per_day))x/day ($(join(times_per_day, ", ")))")
    println("   Month filter: $(months === nothing ? "none (all-year)" : months)")
    println("   Route corridor area [N,W,S,E]: $area  (from $start_location -> $end_location, " *
            "padding=$padding, step=$step)")
    println("   Grid resolution: $(step)° x $(step)°")
    println("   Total requests (one per calendar month): $total_requests")
    old_style_requests = total_requests * 30 * length(times_per_day)
    println("   (vs. ~$old_style_requests requests with one-file-per-timestamp)")
    println()

    manifest = _load_manifest(manifest_path)
    println("   Resuming with $(length(manifest)) manifest entries already recorded")
    println()

    progress = Progress(total_requests; desc = "Downloading months",
                         barglyphs = ProgressMeter.BarGlyphs("[=> ]"),
                         showspeed = true)

    manifest_dirty = 0
    for (y, m) in sort(year_months)
        n_days = daysinmonth(Date(y, m, 1))
        days = collect(1:n_days)

        month_file = joinpath(out_dir, "$(y)-$(lpad(m,2,'0')).grib")

        if isfile(month_file)
            total_skipped += 1
            if !haskey(manifest, month_file)
                manifest[month_file] = m
                manifest_dirty += 1
            end
            ProgressMeter.next!(progress)
            continue
        end

        try
            getWindDataMonth(month_file, y, m, days, times_per_day, area, step)

            if isfile(month_file)
                total_downloaded += 1
                manifest[month_file] = m
                manifest_dirty += 1
            else
                throw(error("File not created"))
            end
        catch e
            total_failed += 1
            println("\n   ✗ Failed: $(basename(month_file)) -- $e")
            if isfile(month_file)
                rm(month_file)
            end
        end

        ProgressMeter.next!(progress)

        if manifest_dirty >= 12
            _save_manifest(manifest_path, manifest)
            manifest_dirty = 0
        end
    end

    _save_manifest(manifest_path, manifest)

    println("\n" * "="^60)
    println("🎉 BATCHED DOWNLOAD COMPLETE")
    println("="^60)
    println("   📥 Successfully downloaded: $total_downloaded monthly files")
    println("   ⏭  Already existed (skipped): $total_skipped monthly files")
    if total_failed > 0
        println("   ❌ Failed: $total_failed monthly files -- re-run to retry, " *
                "completed months are skipped.")
    end
    println("   📊 Manifest entries: $(length(manifest)) (saved to $manifest_path)")
    println("   📁 Data saved to: $out_dir")
    println("="^60)

    return (downloaded = total_downloaded, skipped = total_skipped,
            failed = total_failed, manifest = manifest)
end

"""
    download_hourly_reference(; start_location, end_location, padding, step,
                                 year=2024, out_dir="./ClimatologyOceanData/hourly_reference")

One-off HOURLY download, for resolution testing ONLY -- not for the final
multi-decade archive. A single year at hourly cadence over the same route
corridor lets resolution_tests.jl compare true hourly wind against
subsampled 3h/6h/12h/etc versions of ITSELF, which the 6-hourly archive
can't do (6-hourly data has no finer truth to subsample from). Keep this
to 1 year (or even a handful of months) -- it's only for deciding cadence,
not for the distribution you'll actually use.

12 requests total (one per month, still batched), each ~24x the size of
the equivalent 6-hourly month, so this is noticeably heavier per request
than the main archive -- that's expected and fine for a one-off.
"""
function download_hourly_reference(;
        start_location,
        end_location,
        padding::Real,
        step::Real,
        year::Int = 2025,
        out_dir::String = "./ClimatologyOceanData/hourly_reference"
    )
    area = route_area(start_location, end_location, padding, step)
    hourly_times = ["$(lpad(h,2,'0')):00" for h in 0:23]

    mkpath(out_dir)
    manifest_path = joinpath(out_dir, MANIFEST_NAME)
    manifest = _load_manifest(manifest_path)

    println("📊 Hourly Reference Download (resolution testing only):")
    println("   Year: $year")
    println("   Area: $area")
    println("   Cadence: hourly (24x/day)")
    println("   Total requests: 12 (one per month)")
    println()

    progress = Progress(12; desc = "Downloading hourly reference months",
                         barglyphs = ProgressMeter.BarGlyphs("[=> ]"), showspeed = true)

    downloaded = 0; skipped = 0; failed = 0
    for m in 1:12
        n_days = daysinmonth(Date(year, m, 1))
        days = collect(1:n_days)
        month_file = joinpath(out_dir, "$(year)-$(lpad(m,2,'0'))-hourly.grib")

        if isfile(month_file)
            skipped += 1
            manifest[month_file] = m
            ProgressMeter.next!(progress)
            continue
        end

        try
            getWindDataMonth(month_file, year, m, days, hourly_times, area, step)
            if isfile(month_file)
                downloaded += 1
                manifest[month_file] = m
            else
                throw(error("File not created"))
            end
        catch e
            failed += 1
            println("\n   ✗ Failed: $(basename(month_file)) -- $e")
            isfile(month_file) && rm(month_file)
        end
        ProgressMeter.next!(progress)
    end

    _save_manifest(manifest_path, manifest)
    println("\n✅ Hourly reference download complete: $downloaded downloaded, " *
            "$skipped skipped, $failed failed. Saved to $out_dir")
    return (downloaded = downloaded, skipped = skipped, failed = failed)
end

function historical_paths_and_months(; out_dir::String = "./ClimatologyOceanData/route")
    manifest_path = joinpath(out_dir, MANIFEST_NAME)
    manifest = _load_manifest(manifest_path)
    isempty(manifest) && error("No manifest found at $manifest_path -- " *
                                "run bulk_download_climatology_batched first.")
    paths  = collect(keys(manifest))
    months = [manifest[p] for p in paths]
    return paths, months
end

# -----------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------
# Run resolution_tests.jl FIRST against your existing 3-year archive to
# pick times_per_day (from subsampling_aliasing_test) and start_year/
# end_year (from convergence_test). start_location/end_location/padding/
# step below should match whatever you actually pass to
# run_dynamic_band_convergence_hifi -- keep them in sync.

println("🔋 Disabling sleep mode during download...")
run(`powercfg /change standby-timeout-ac 0`)

println("🚀 Starting batched climatology archive download...")
println()

# For the resolution decision itself, run this FIRST (much smaller/faster
# than the full archive below) -- see resolution_tests.jl's hourly-vs-6h
# aliasing test, which needs this to exist:
#
println("🔋 Disabling sleep mode during download...")
run(`powercfg /change standby-timeout-ac 0`)

println("🚀 Starting batched climatology archive download...")
println()

bulk_download_climatology_batched(
    start_location = (-94.0, 28.5),
    end_location   = (-5.5,  49.5),
    padding        = 80,
    step           = 0.25,
    start_year     = 2006,
    end_year       = 2025,
    times_per_day  = ["00:00", "06:00", "12:00", "18:00"],
    months         = nothing
)

println("🔋 Restoring sleep mode...")
run(`powercfg /change standby-timeout-ac 30`)

println()
println("✅ Script execution complete!")