using JLD2, GRIB

function reconvert_all_jld2(grib_dir::String = "./OceanData")
    # --- Step 1: Delete all existing JLD2 files ---
    deleted = 0
    for fname in readdir(grib_dir)
        endswith(fname, ".jld2") || continue
        path = joinpath(grib_dir, fname)
        rm(path)
        println("Deleted: $fname")
        deleted += 1
    end
    println("Deleted $deleted JLD2 files.\n")

    # --- Step 2: Reconvert from GRIB with efficient storage ---
    converted = 0
    skipped   = 0
    for fname in readdir(grib_dir)
        endswith(fname, ".grib") || continue
        grib_path = joinpath(grib_dir, fname)
        jld2_path = replace(grib_path, ".grib" => ".jld2")

        u_raw = v_raw = lats_raw = lons_raw = nothing

        try
            GRIB.GribFile(grib_path) do f
                for msg in f
                    sname = msg["shortName"]
                    if sname == "10u"
                        lats_raw = msg["latitudes"]
                        lons_raw = msg["longitudes"]
                        u_raw    = msg["values"]
                    elseif sname == "10v"
                        v_raw = msg["values"]
                    end
                    !isnothing(u_raw) && !isnothing(v_raw) && break
                end
            end
        catch e
            @warn "Failed to read $fname: $e"
            skipped += 1
            continue
        end

        if isnothing(u_raw) || isnothing(v_raw)
            @warn "Missing u/v in $fname, skipping"
            skipped += 1
            continue
        end

        # Store only unique sorted lats/lons (721 + 1440 values)
        # NOT the full flat array (~1M values) — this is the key fix
        grid_lats = sort(unique(lats_raw))
        grid_lons = sort(unique(lons_raw))

        jldsave(jld2_path; u_raw, v_raw, grid_lats, grid_lons)
        println("Converted: $fname -> $(basename(jld2_path))")
        converted += 1
    end

    println("\nDone. Converted=$converted | Skipped=$skipped")
end

reconvert_all_jld2()

using GRIB, JLD2

grib_path = "./OceanData/1200-1Nov2024.grib"
jld2_path = "./OceanData/1200-1Nov2024.jld2"

# Load from GRIB
u_grib = Ref{Any}(nothing)
v_grib = Ref{Any}(nothing)

GRIB.GribFile(grib_path) do f
    for msg in f
        msg["shortName"] == "10u" && (u_grib[] = msg["values"])
        msg["shortName"] == "10v" && (v_grib[] = msg["values"])
        !isnothing(u_grib[]) && !isnothing(v_grib[]) && break
    end
end

# Load from JLD2
u_jld2 = jldopen(jld2_path, "r") do f f["u_raw"] end
v_jld2 = jldopen(jld2_path, "r") do f f["v_raw"] end

println("U match: ", u_grib[] == u_jld2)
println("V match: ", v_grib[] == v_jld2)