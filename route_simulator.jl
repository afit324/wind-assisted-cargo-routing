# =============================================================================
# route_simulator.jl  --  route steppers and interactive explorers
#
# Layout
#   1. Shared helpers (grid axes, ocean background, wind neighbourhood pane)
#   2. Wind reconstruction at an arbitrary elapsed time (WindCache / get_wind)
#   3. Stepper data model (StepInfo, RouteStepper2, build_stepper2)
#   4. Great-circle helper
#   5. Convergence plots
#   6. launch_simulator           (compare different variant/mode/Rmax runs)
#   7. save_steppers / load_steppers
#   8. launch_forecast_actual_explorer (one route, forecast clock vs real clock)
#
# Requires routing.jl (WindFieldPool, GridInterpolator, NbCache, get_epoch_for_bucket!,
# _getField!, precompute_uv_grid!, build_neighbour_cache, ...) and the diag_out keys
# findDijkstraRoutefinal fills (wind_pool, interp, cur_*, node_*_confirm, total_confirmed).
# =============================================================================

using GLMakie
using GeoMakie          # launch_simulator's overview map uses GeoMakie.GeoAxis
using Printf
using DataFrames
using JLD2

# RouteKey = (variant, mode, Rmax); variant is "wind" or "nowind".
const RouteKey = Tuple{String,String,Int}

const KM_PER_NM = 1.852
const NEIGHBOUR_TARGET_SPACING_M = 27_800.0   # must match the build_neighbour_cache calls in the solver

# =============================================================================
# 1. Shared helpers
# =============================================================================

"""1-D longitude and latitude axes of a regular grid_points matrix."""
grid_axes(grid_points) = ([grid_points[1, j][1] for j in axes(grid_points, 2)],
                          [grid_points[i, 1][2] for i in axes(grid_points, 1)])

"""Land (tan) / ocean (pale blue) background heatmap. Returns the plot so callers can delete it."""
function draw_ocean_background!(ax, grid_points, ocean_mask;
                                land = RGB(0.82, 0.75, 0.55), ocean = RGB(0.75, 0.87, 0.95))
    lons, lats = grid_axes(grid_points)
    # ocean_mask is [lat, lon]; heatmap! wants z[ix, iy] = (lon, lat)
    return GLMakie.heatmap!(ax, lons, lats, permutedims(Float32.(ocean_mask));
                            colormap = [land, ocean], colorrange = (0, 1), interpolate = false)
end

"""
Draw wind speed (heatmap) and direction (arrows) in a (2*zoom_cells+1)^2 window around
node (i, j). Updates `wind_range` and returns the window bounds (i_lo, i_hi, j_lo, j_hi).
"""
function draw_wind_neighbourhood!(ax, U::Matrix{Float32}, V::Matrix{Float32},
                                  grid_points, i::Int, j::Int, wind_range;
                                  zoom_cells::Int = 15)
    n_rows, n_cols = size(grid_points)
    i_lo, i_hi = max(1, i - zoom_cells), min(n_rows, i + zoom_cells)
    j_lo, j_hi = max(1, j - zoom_cells), min(n_cols, j + zoom_cells)

    lons_sub = [grid_points[i, jj][1] for jj in j_lo:j_hi]
    lats_sub = [grid_points[ii, j][2] for ii in i_lo:i_hi]
    speed_sub = sqrt.(U[i_lo:i_hi, j_lo:j_hi] .^ 2 .+ V[i_lo:i_hi, j_lo:j_hi] .^ 2)

    wind_range[] = (0.0, max(maximum(speed_sub), 1.0))
    GLMakie.heatmap!(ax, lons_sub, lats_sub, permutedims(speed_sub);
                     colormap = :viridis, colorrange = wind_range[])

    step_q = max(1, zoom_cells ÷ 6)
    qlons = Float64[]; qlats = Float64[]; qu = Float64[]; qv = Float64[]
    for ii in i_lo:step_q:i_hi, jj in j_lo:step_q:j_hi
        lo, la = grid_points[ii, jj]
        push!(qlons, lo); push!(qlats, la)
        push!(qu, U[ii, jj]); push!(qv, V[ii, jj])
    end
    isempty(qlons) || GLMakie.arrows!(ax, qlons, qlats, qu .* 0.03, qv .* 0.03;
                                      color = :white, linewidth = 1.0, arrowsize = 6)
    return (i_lo, i_hi, j_lo, j_hi)
end

_delete_plots!(ax, plots) = (for p in plots; p === nothing || delete!(ax, p); end; empty!(plots))

# =============================================================================
# 2. Wind reconstruction
#
# Rebuilds the time-interpolated u,v field at any elapsed_seconds along a route,
# reusing the wind_pool / interp the solver built (no new reads unless a new
# epoch is needed). The time fraction uses the same absolute clock as the solver
# (route_start_abs + elapsed vs each field's absolute epoch).
# =============================================================================

function wind_snapshot_at(wind_pool::WindFieldPool, interp::GridInterpolator,
                          cur_time::Int, cur_day::Int, cur_month::Int, cur_year::Int,
                          elapsed_seconds::Real)
    n_rows, n_cols = interp.n_rows, interp.n_cols
    n_flat = n_rows * n_cols

    elapsed_int = round(Int, elapsed_seconds)
    bucket      = fld((cur_time % 100) * 60 + elapsed_int + 1800, 3600)
    epoch = get_epoch_for_bucket!(Dict{Int,Int}(), wind_pool, cur_time, cur_day, cur_month,
                                  cur_year, bucket, elapsed_int)

    u0 = Vector{Float32}(undef, n_flat); v0 = Vector{Float32}(undef, n_flat)
    precompute_uv_grid!(u0, v0, _getField!(wind_pool, epoch), interp)

    abs_now = toEpochSeconds(cur_time, cur_day, cur_month, cur_year) + elapsed_int
    ep_idx  = searchsortedfirst(wind_pool.sorted_epochs, epoch)
    if ep_idx < length(wind_pool.sorted_epochs)
        next_ep = wind_pool.sorted_epochs[ep_idx + 1]
        u1 = Vector{Float32}(undef, n_flat); v1 = Vector{Float32}(undef, n_flat)
        precompute_uv_grid!(u1, v1, _getField!(wind_pool, next_ep), interp)

        dt    = Float64(next_ep - epoch)
        tfrac = dt > 0 ? clamp((Float64(abs_now) - epoch) / dt, 0.0, 1.0) : 0.0
        @inbounds for k in 1:n_flat
            u0[k] = Float32((1 - tfrac) * u0[k] + tfrac * u1[k])
            v0[k] = Float32((1 - tfrac) * v0[k] + tfrac * v1[k])
        end
    end
    return reshape(u0, n_rows, n_cols), reshape(v0, n_rows, n_cols), epoch
end

"""Memoised wind lookup, bucketed to 5 minutes (far finer than the GRIB cadence)."""
mutable struct WindCache
    pool::WindFieldPool
    interp::GridInterpolator
    ct::Int; cd::Int; cm::Int; cy::Int
    cache::Dict{Int, Tuple{Matrix{Float32}, Matrix{Float32}}}
end
WindCache(pool, interp, ct, cd, cm, cy) =
    WindCache(pool, interp, ct, cd, cm, cy, Dict{Int, Tuple{Matrix{Float32}, Matrix{Float32}}}())

function get_wind(wc::WindCache, elapsed_seconds::Real)
    key = round(Int, elapsed_seconds / 300)
    get!(wc.cache, key) do
        U, V, _ = wind_snapshot_at(wc.pool, wc.interp, wc.ct, wc.cd, wc.cm, wc.cy, key * 300)
        (U, V)
    end
end

# =============================================================================
# 3. Stepper data model (precomputed once so scrubbing is an array lookup)
# =============================================================================

struct StepInfo
    i::Int; j::Int; e_bin::Int
    lon::Float64; lat::Float64
    cum_time_s::Float64
    cum_energy_kwh::Float64
    cum_dist_km::Float64
    leg_dist_km::Float64
    leg_time_s::Float64
    leg_speed_kn::Float64
    leg_power_kw::Float64
    band::Int8
    epoch::Int
    relax_ms::Float64
    confirm_order::Int
    solver_time_s::Float64
end

mutable struct RouteStepper2
    steps::Vector{StepInfo}
    grid_points::Matrix{Tuple{Float64,Float64}}
    ocean_mask::Matrix{Bool}
    nb_flat::NbCache
    wind::WindCache
    total_confirmed::Int
end

band_label(b::Int8) = b == 1 ? "FULL (1..Rmax)" : b == 0 ? "unknown" : "OUTER-ring only"

"""
Build a stepper from exactly the four values findDijkstraRoutefinal returns plus its
`diag_out` dict, so the stepper always agrees with what the solver reported.
Solver time per step is apportioned by confirm order (an estimate, not a measurement).
"""
function build_stepper2(path_coords, node_costs, final_energy, power_used,
                        diag::Dict, grid_points::Matrix{Tuple{Float64,Float64}},
                        ocean_mask::Matrix{Bool})
    path_idxs       = diag[:path_idxs]
    total_confirmed = diag[:total_confirmed]
    total_solver_s  = get(diag, :total_loop_time_ns, 0.0) / 1e9

    n = length(path_idxs)
    steps = Vector{StepInfo}(undef, n)
    cum_dist = 0.0
    for k in 1:n
        i, j, e = path_idxs[k]
        lon, lat = path_coords[k]

        if k == 1
            leg_dist = leg_t = leg_spd = 0.0
        else
            lon0, lat0 = path_coords[k-1]
            leg_dist = getDistance(lon0, lat0, lon, lat) / 1000.0           # m -> km
            leg_t    = node_costs[k] - node_costs[k-1]
            leg_spd  = leg_t > 0 ? (leg_dist / KM_PER_NM) / (leg_t / 3600) : 0.0
        end
        cum_dist += leg_dist

        key   = (i, j, e)
        order = get(diag[:node_confirm_order], key, 0)
        solver_s = (total_confirmed > 0 && order > 0) ? total_solver_s * (order / total_confirmed) : 0.0

        steps[k] = StepInfo(i, j, e, lon, lat, node_costs[k], final_energy[k], cum_dist,
                            leg_dist, leg_t, leg_spd, k == 1 ? 0.0 : power_used[k-1],
                            get(diag[:node_band_confirm], key, Int8(0)),
                            get(diag[:node_epoch_confirm], key, -1),
                            get(diag[:node_relax_ns_confirm], key, 0.0) / 1e6,
                            order, solver_s)
    end

    wind = WindCache(diag[:wind_pool], diag[:interp],
                     diag[:cur_time], diag[:cur_day], diag[:cur_month], diag[:cur_year])
    return RouteStepper2(steps, grid_points, ocean_mask, diag[:nb_flat], wind, total_confirmed)
end

# =============================================================================
# 4. Great circle (spherical slerp), for the optional overlay
# =============================================================================

function great_circle_points(lon1::Float64, lat1::Float64, lon2::Float64, lat2::Float64; n::Int = 150)
    φ1, λ1 = deg2rad(lat1), deg2rad(lon1)
    φ2, λ2 = deg2rad(lat2), deg2rad(lon2)
    Δ = 2 * asin(clamp(sqrt(sin((φ2 - φ1) / 2)^2 + cos(φ1) * cos(φ2) * sin((λ2 - λ1) / 2)^2), 0.0, 1.0))
    Δ < 1e-9 && return [lon1, lon2], [lat1, lat2]

    lons = Vector{Float64}(undef, n); lats = Vector{Float64}(undef, n)
    sinΔ = sin(Δ)
    for (k, f) in enumerate(range(0.0, 1.0, length = n))
        A = sin((1 - f) * Δ) / sinΔ
        B = sin(f * Δ) / sinΔ
        x = A * cos(φ1) * cos(λ1) + B * cos(φ2) * cos(λ2)
        y = A * cos(φ1) * sin(λ1) + B * cos(φ2) * sin(λ2)
        z = A * sin(φ1) + B * sin(φ2)
        lons[k] = rad2deg(atan(y, x))
        lats[k] = rad2deg(atan(z, sqrt(x^2 + y^2)))
    end
    return lons, lats
end

# =============================================================================
# 5. Convergence plots (separate figure; "back" swaps the same window to the stepper)
# =============================================================================

function show_convergence_plots(results_wind::DataFrame, results_nowind::DataFrame, screen, main_fig)
    w = copy(results_wind);   w[!, :Variant] .= "Wind"
    nw = copy(results_nowind); nw[!, :Variant] .= "No-Wind"
    all_results = vcat(w, nw)
    feasible = all_results[all_results.Feasible, :]

    fig = GLMakie.Figure(size = (1400, 1050))
    panels = [(GLMakie.Axis(fig[1, 1]; title = "Time vs Rmax",      xlabel = "Rmax", ylabel = "Time (h)"),      :TimeH),
              (GLMakie.Axis(fig[1, 2]; title = "Distance vs Rmax",  xlabel = "Rmax", ylabel = "Distance (km)"), :DistKm),
              (GLMakie.Axis(fig[2, 1]; title = "Energy vs Rmax",    xlabel = "Rmax", ylabel = "Energy (kWh)"),  :EnergyKwh),
              (GLMakie.Axis(fig[2, 2]; title = "Waypoints vs Rmax", xlabel = "Rmax", ylabel = "Waypoints"),     :Waypoints)]

    colors  = Dict("Wind" => :blue,   "No-Wind" => :red)
    markers = Dict("Wind" => :circle, "No-Wind" => :diamond)

    for mode in sort(unique(feasible.Mode)), variant in ("Wind", "No-Wind")
        subset = sort(feasible[(feasible.Mode .== mode) .& (feasible.Variant .== variant), :], :Rmax)
        isempty(subset) && continue
        for (k, (ax, col)) in enumerate(panels)
            GLMakie.scatter!(ax, subset.Rmax, subset[!, col]; color = colors[variant],
                             marker = markers[variant], markersize = 12,
                             label = k == 1 ? "$variant · $mode" : nothing)
            GLMakie.lines!(ax, subset.Rmax, subset[!, col]; color = colors[variant],
                           linewidth = 1.5, linestyle = :dash)
        end
    end

    GLMakie.axislegend(panels[1][1], position = :rt)
    GLMakie.Label(fig[0, :], "Radius Convergence Analysis — Wind vs No-Wind"; fontsize = 18, font = :bold)
    GLMakie.Label(fig[3, 1:2], "Lines connect points for each (Variant, Mode) across Rmax; points are computed values.";
                  fontsize = 11, color = :gray50, halign = :center)
    btn_back = GLMakie.Button(fig[4, 1:2], label = "← Back to Route Simulator")
    on(btn_back.clicks) do _
        GLMakie.display(screen, main_fig)
    end
    return fig
end

# =============================================================================
# 6. launch_simulator
#
# Colour = Mode hue, lightness = Rmax (and a small variant offset); line style =
# variant (solid WIND, dashed NO-WIND). Visibility toggles AND together; "isolate"
# shows only the route being stepped.
# =============================================================================

const STEP_LABEL_STAGGER = [((8, 8), (:left, :bottom)), ((8, -8), (:left, :top)),
                            ((-8, 8), (:right, :bottom)), ((-8, -8), (:right, :top))]

function launch_simulator(steppers::Dict{RouteKey, RouteStepper2},
                          results_wind::DataFrame, results_nowind::DataFrame)
    isempty(steppers) && error("No steppers to display — nothing came back from run_radius_convergence_hifi.")

    run_labels = sort(collect(keys(steppers)))
    run_names  = [string(uppercase(l[1]) == "WIND" ? "WIND" : "NO-WIND", "  ·  ", l[2], "  ·  Rmax=", l[3])
                  for l in run_labels]
    name_to_key = Dict(zip(run_names, run_labels))
    max_steps = maximum(length(rs.steps) for rs in values(steppers))

    variants_present = sort(unique(first.(run_labels)))
    modes_present    = sort(unique(getindex.(run_labels, 2)))
    rmaxs_present    = sort(unique(last.(run_labels)))
    rmax_lo, rmax_hi = extrema(rmaxs_present)

    mode_palette = [RGB(0, 0.39, 0), RGB(0.5, 0, 0.5), RGB(1, 0.55, 0), RGB(0, 0.5, 0.5),
                    RGB(0.65, 0.16, 0.16), RGB(0, 0, 0.5)]
    mode_colors  = Dict(m => mode_palette[mod1(i, length(mode_palette))] for (i, m) in enumerate(modes_present))
    variant_style = Dict("wind" => :solid, "nowind" => :dash)

    # light tint of the mode hue at the smallest Rmax -> darker at the largest
    function gradient_color(variant, mode, rmax)
        base = get(mode_colors, mode, RGB(0.5, 0.5, 0.5))
        bias = variant == "wind" ? 0.15 : -0.15
        frac = rmax_hi > rmax_lo ? (rmax - rmax_lo) / (rmax_hi - rmax_lo) : 1.0
        lo = (clamp(base.r + 0.6, 0, 1), clamp(base.g + 0.6, 0, 1), clamp(base.b + 0.6, 0, 1))
        hi = (clamp(base.r * 0.6 + bias, 0, 1), clamp(base.g * 0.6 + bias, 0, 1), clamp(base.b * 0.6 + bias, 0, 1))
        return RGB((lo .+ frac .* (hi .- lo))...)
    end

    current_run  = GLMakie.Observable(run_labels[1])
    current_step = GLMakie.Observable(1)
    is_playing   = GLMakie.Observable(false)
    wind_range   = GLMakie.Observable((0.0, 15.0))
    get_rs() = steppers[current_run[]]

    # ---- figure: row 1 map | zoom | colorbar, row 2 toggles | stats, row 3 controls ----
    fig = GLMakie.Figure(size = (2400, 1450))
    screen = GLMakie.Screen()
    GLMakie.Label(fig[0, 1:3], "Route Simulator — Radius Convergence Explorer";
                  fontsize = 22, font = :bold, halign = :left, tellwidth = false)

    ax_route = GeoMakie.GeoAxis(fig[1, 1]; title = "Route Overview", dest = "+proj=longlat +datum=WGS84")
    ax_zoom  = GLMakie.Axis(fig[1, 2]; title = "Neighbourhood + wind", xlabel = "lon", ylabel = "lat")
    GLMakie.Colorbar(fig[1, 3]; colorrange = wind_range, colormap = :viridis, label = "wind speed (m/s)")
    GLMakie.colsize!(fig.layout, 1, GLMakie.Relative(0.58))
    GLMakie.colsize!(fig.layout, 2, GLMakie.Relative(0.37))
    GLMakie.colsize!(fig.layout, 3, GLMakie.Relative(0.05))

    # ---- visibility toggles ----
    toggle_panel = fig[2, 1] = GLMakie.GridLayout(tellheight = false)
    GLMakie.Label(toggle_panel[1, 1:6], "Route Visibility"; fontsize = 15, font = :bold, halign = :left)
    GLMakie.Label(toggle_panel[2, 1:6],
                  "colour: hue = Mode, lighter/darker = variant + Rmax   ·   line: solid = WIND, dashed = NO-WIND   ·   dotted black = great circle";
                  fontsize = 11, color = :gray30, halign = :left)

    variant_on = Dict{String,Any}(); mode_on = Dict{String,Any}(); rmax_on = Dict{Int,Any}()
    function add_toggle_row!(row, title, items, store; fmt = string, color = _ -> :black)
        GLMakie.Label(toggle_panel[row, 1], title; fontsize = 12, font = :bold, halign = :right)
        for (c, item) in enumerate(items)
            t = GLMakie.Toggle(fig; active = true)
            store[item] = t.active
            toggle_panel[row, 2c]   = t
            toggle_panel[row, 2c+1] = GLMakie.Label(fig, fmt(item); fontsize = 12, color = color(item))
        end
    end
    row = 3
    add_toggle_row!(row, "Type:", variants_present, variant_on; fmt = uppercase); row += 1
    if length(modes_present) > 1
        add_toggle_row!(row, "Mode:", modes_present, mode_on; color = m -> mode_colors[m]); row += 1
    else
        for m in modes_present; mode_on[m] = GLMakie.Observable(true); end
    end
    add_toggle_row!(row, "Rmax:", rmaxs_present, rmax_on); row += 1

    function add_switch!(row, title)
        sw = GLMakie.Toggle(fig; active = false)
        toggle_panel[row, 1] = GLMakie.Label(fig, title; fontsize = 12, font = :bold, halign = :right)
        toggle_panel[row, 2] = sw
        return sw
    end
    isolate_sw = add_switch!(row, "Isolate stepped route:");   row += 1
    gc_sw      = add_switch!(row, "Great circle overlay:");    row += 1
    grid_sw    = add_switch!(row, "Show grid:");               row += 1

    is_isolated() = isolate_sw.active[]
    is_route_visible(key::RouteKey) = variant_on[key[1]][] && mode_on[key[2]][] && rmax_on[key[3]][]
    final_visible(key::RouteKey) = is_isolated() ? key == current_run[] : is_route_visible(key)

    # ---- drawing ----
    route_plots   = Any[]                    # lines + start/end markers of all visible routes
    ocean_ref     = Ref{Any}(nothing)
    marker_ref    = Ref{Any}(nothing)
    gc_ref        = Ref{Any}(nothing)
    grid_ref      = Ref{Any}(nothing)
    legend_ref    = Ref{Any}(nothing)
    label_plots   = Any[]
    labeled_steps = Set{Int}()

    function draw_ocean_mask!(rs)
        ocean_ref[] === nothing || delete!(ax_route, ocean_ref[])
        ocean_ref[] = draw_ocean_background!(ax_route, rs.grid_points, rs.ocean_mask)
        GLMakie.translate!(ocean_ref[], 0, 0, 1)            # under grid (5) and routes (10)
    end

    function draw_all_routes!()
        _delete_plots!(ax_route, route_plots)
        for key in run_labels
            final_visible(key) || continue
            variant, mode, rmax = key
            steps = steppers[key].steps
            lons = [s.lon for s in steps]; lats = [s.lat for s in steps]
            line = GLMakie.lines!(ax_route, lons, lats; color = gradient_color(variant, mode, rmax),
                                  linewidth = 1.6, linestyle = get(variant_style, variant, :solid),
                                  label = string(variant, " · ", mode, " (R=", rmax, ")"))
            m1 = GLMakie.scatter!(ax_route, [lons[1]],   [lats[1]];   color = :green, markersize = 10, marker = :diamond)
            m2 = GLMakie.scatter!(ax_route, [lons[end]], [lats[end]]; color = :red,   markersize = 10, marker = :diamond)
            foreach(p -> GLMakie.translate!(p, 0, 0, 10), (line, m1, m2))
            append!(route_plots, (line, m1, m2))
        end
        legend_ref[] === nothing || (delete!(legend_ref[]); legend_ref[] = nothing)
        isempty(route_plots) || (legend_ref[] = GLMakie.axislegend(ax_route; position = :lb,
                                    framevisible = true, labelsize = 10, patchsize = (12, 6)))
    end

    function draw_position_marker!(rs)
        marker_ref[] === nothing || delete!(ax_route, marker_ref[])
        s = rs.steps[1]
        marker_ref[] = GLMakie.scatter!(ax_route, [s.lon], [s.lat]; color = :magenta,
                                        markersize = 20, marker = :diamond, label = "current position")
    end

    function draw_great_circle!(rs)
        gc_ref[] === nothing || (delete!(ax_route, gc_ref[]); gc_ref[] = nothing)
        gc_sw.active[] || return
        gl, gla = great_circle_points(rs.steps[1].lon, rs.steps[1].lat, rs.steps[end].lon, rs.steps[end].lat)
        gc_ref[] = GLMakie.lines!(ax_route, gl, gla; color = :black, linestyle = :dot,
                                  linewidth = 2.5, label = "great circle")
    end

    function draw_grid!(rs)
        grid_ref[] === nothing || (delete!(ax_route, grid_ref[]); grid_ref[] = nothing)
        grid_sw.active[] || return
        lons, lats = grid_axes(rs.grid_points)
        lon_lo, lon_hi = extrema(lons); lat_lo, lat_hi = extrema(lats)
        xs = Float64[]; ys = Float64[]                      # NaNs break one line object into segments
        for lon in lons; push!(xs, lon, lon, NaN);       push!(ys, lat_lo, lat_hi, NaN); end
        for lat in lats; push!(xs, lon_lo, lon_hi, NaN); push!(ys, lat, lat, NaN);       end
        step = round(abs(lons[2] - lons[1]), digits = 3)
        grid_ref[] = GLMakie.lines!(ax_route, xs, ys; color = (:gray70, 0.3), linewidth = 1.0,
                                    label = "grid ($(step)°)")
        GLMakie.translate!(grid_ref[], 0, 0, 5)
    end

    function clear_step_labels!()
        _delete_plots!(ax_route, label_plots)
        empty!(labeled_steps)
    end

    function add_step_label!(rs, k)                         # solver-time label, isolate mode only
        is_isolated() || return
        k in labeled_steps && return
        s = rs.steps[k]
        off, al = STEP_LABEL_STAGGER[(length(label_plots) % length(STEP_LABEL_STAGGER)) + 1]
        t = GLMakie.text!(ax_route, [s.lon], [s.lat]; text = [@sprintf("%.2fs", s.solver_time_s)],
                          fontsize = 7, color = :black, align = al, offset = off)
        GLMakie.translate!(t, 0, 0, 20)
        push!(label_plots, t); push!(labeled_steps, k)
    end

    function fit_view!()                                    # zoom the overview to the visible routes
        keys_vis = [k for k in run_labels if final_visible(k)]
        isempty(keys_vis) && (keys_vis = [current_run[]])
        lons = Float64[]; lats = Float64[]
        for k in keys_vis
            append!(lons, (s.lon for s in steppers[k].steps)); append!(lats, (s.lat for s in steppers[k].steps))
        end
        lon_lo, lon_hi = extrema(lons); lat_lo, lat_hi = extrema(lats)
        lon_pad = max(2.0, 0.12 * (lon_hi - lon_lo)); lat_pad = max(2.0, 0.12 * (lat_hi - lat_lo))
        GLMakie.xlims!(ax_route, lon_lo - lon_pad, lon_hi + lon_pad)
        GLMakie.ylims!(ax_route, lat_lo - lat_pad, lat_hi + lat_pad)
    end

    function draw_zoom!(rs, k; zoom_cells::Int = 15)
        GLMakie.empty!(ax_zoom)
        s = rs.steps[k]
        U, V = get_wind(rs.wind, s.cum_time_s)
        i_lo, i_hi, j_lo, j_hi = draw_wind_neighbourhood!(ax_zoom, U, V, rs.grid_points, s.i, s.j,
                                                          wind_range; zoom_cells)

        land = [rs.grid_points[ii, jj] for jj in j_lo:j_hi for ii in i_lo:i_hi if !rs.ocean_mask[ii, jj]]
        isempty(land) || GLMakie.scatter!(ax_zoom, first.(land), last.(land); color = (:khaki, 0.85), markersize = 6)

        buf = rs.nb_flat[s.i, s.j]
        if !isempty(buf)                                    # edges the solver used from this node in red
            Rmax_here = maximum(x[6] for x in buf)
            lon0, lat0 = rs.grid_points[s.i, s.j]
            for (ni, nj, _, _, _, r) in buf
                (i_lo <= ni <= i_hi && j_lo <= nj <= j_hi) || continue
                lon2, lat2 = rs.grid_points[ni, nj]
                used = s.band == 1 || r == Rmax_here
                GLMakie.lines!(ax_zoom, [lon0, lon2], [lat0, lat2];
                               color = used ? (:red, 0.9) : (:grey60, 0.3), linewidth = used ? 2.0 : 0.6)
            end
        end

        GLMakie.scatter!(ax_zoom, [s.lon], [s.lat]; color = :black, markersize = 16, marker = :star5)
        if k < length(rs.steps)
            nxt = rs.steps[k + 1]
            GLMakie.scatter!(ax_zoom, [nxt.lon], [nxt.lat]; color = :lime, markersize = 12)
        end
        ax_zoom.title = "Step $k/$(length(rs.steps)) — node ($(s.i),$(s.j))  band=$(band_label(s.band))"
    end

    # ---- stats ----
    stats_route_text  = GLMakie.Observable("")
    stats_solver_text = GLMakie.Observable("")
    stats_panel = fig[2, 2:3] = GLMakie.GridLayout(tellheight = false)
    GLMakie.Label(stats_panel[1, 1], "ROUTE / BOAT STATS"; fontsize = 14, font = :bold, halign = :left)
    GLMakie.Label(stats_panel[1, 2], "SOLVER DIAGNOSTICS"; fontsize = 14, font = :bold, halign = :left)
    GLMakie.Label(stats_panel[2, 1], stats_route_text;  tellwidth = false, halign = :left, valign = :top, fontsize = 13, justification = :left)
    GLMakie.Label(stats_panel[2, 2], stats_solver_text; tellwidth = false, halign = :left, valign = :top, fontsize = 13, justification = :left)
    GLMakie.colsize!(stats_panel, 1, GLMakie.Relative(0.5))
    GLMakie.colsize!(stats_panel, 2, GLMakie.Relative(0.5))

    function update_stats!(rs, k)
        s = rs.steps[k]
        stats_route_text[] = @sprintf(
            "cumulative time:   %.2f h\ncumulative dist:   %.1f km\ncumulative energy: %.1f kWh\navg speed so far:  %.2f kn\n\nthis leg: %.2f km in %.0f s → %.2f kn\nengine power (this leg): %.0f kW",
            s.cum_time_s / 3600, s.cum_dist_km, s.cum_energy_kwh,
            s.cum_time_s > 0 ? (s.cum_dist_km / KM_PER_NM) / (s.cum_time_s / 3600) : 0.0,
            s.leg_dist_km, s.leg_time_s, s.leg_speed_kn, s.leg_power_kw)
        stats_solver_text[] = @sprintf(
            "band:              %s\nwind epoch used:   %s\nrelax time @confirm: %.3f ms\nconfirm order:     %d / %d states\nneighbours cached: %d\ncumulative solver time: %.2f s",
            band_label(s.band), s.epoch == -1 ? "n/a" : string(s.epoch), s.relax_ms,
            s.confirm_order, rs.total_confirmed, length(rs.nb_flat[s.i, s.j]), s.solver_time_s)
    end

    # ---- transport controls ----
    controls = fig[3, 1:3] = GLMakie.GridLayout()
    GLMakie.Label(controls[1, 1], "Stepping through:"; fontsize = 13, font = :bold, halign = :right)
    menu      = GLMakie.Menu(controls[1, 2], options = run_names, default = run_names[1])
    btn_fit   = GLMakie.Button(controls[1, 3], label = "⤢ Fit View")
    btn_prev  = GLMakie.Button(controls[1, 4], label = "◀ Prev")
    btn_next  = GLMakie.Button(controls[1, 5], label = "Next ▶")
    btn_reset = GLMakie.Button(controls[1, 6], label = "⟲ Reset")
    btn_play  = GLMakie.Button(controls[1, 7], label = "▶ Play")
    speed_sl  = GLMakie.Slider(controls[1, 8], range = 0.1:0.1:3.0, startvalue = 0.5)
    GLMakie.Label(controls[1, 9], GLMakie.lift(v -> "speed: $(round(v, digits=1)) s/step", speed_sl.value))
    btn_convergence = GLMakie.Button(controls[1, 10], label = "Convergence Plots")
    scrub = GLMakie.Slider(controls[2, 1:8], range = 1:max_steps, startvalue = 1)
    GLMakie.Label(controls[2, 9], GLMakie.lift(k -> "step $k / $(length(get_rs().steps))", current_step))

    # ---- wiring ----
    redraw_routes!() = (draw_all_routes!(); draw_position_marker!(get_rs()); fit_view!())
    for store in (variant_on, mode_on, rmax_on), (_, obs) in store
        on(obs) do _; redraw_routes!(); end
    end
    on(isolate_sw.active) do _
        redraw_routes!(); clear_step_labels!()
        add_step_label!(get_rs(), current_step[])
    end
    on(gc_sw.active)   do _; draw_great_circle!(get_rs()); end
    on(grid_sw.active) do _; draw_grid!(get_rs()); end

    on(btn_prev.clicks)  do _; current_step[] = max(1, current_step[] - 1); end
    on(btn_next.clicks)  do _; current_step[] = min(length(get_rs().steps), current_step[] + 1); end
    on(btn_reset.clicks) do _; current_step[] = 1; end
    on(btn_fit.clicks)   do _; fit_view!(); end
    on(scrub.value) do v; current_step[] = clamp(round(Int, v), 1, length(get_rs().steps)); end
    on(menu.selection) do sel
        key = sel === nothing ? nothing : get(name_to_key, sel, nothing)
        key === nothing || (current_run[] = key)
    end
    on(btn_convergence.clicks) do _
        GLMakie.display(screen, show_convergence_plots(results_wind, results_nowind, screen, fig))
    end

    on(current_run) do key
        rs = steppers[key]
        draw_ocean_mask!(rs); draw_all_routes!(); draw_position_marker!(rs)
        draw_great_circle!(rs); draw_grid!(rs)
        is_isolated() && fit_view!()
        current_step[] = 1
        draw_zoom!(rs, 1); update_stats!(rs, 1)
        clear_step_labels!(); add_step_label!(rs, 1)
    end
    on(current_step) do k
        rs = get_rs()
        k = clamp(k, 1, length(rs.steps))
        if marker_ref[] !== nothing
            marker_ref[][1] = [rs.steps[k].lon]; marker_ref[][2] = [rs.steps[k].lat]
        end
        draw_zoom!(rs, k); update_stats!(rs, k); add_step_label!(rs, k)
    end

    play_timer = Ref{Union{Nothing,Timer}}(nothing)
    stop_timer!() = (play_timer[] === nothing || close(play_timer[]); play_timer[] = nothing)
    function start_timer!()
        stop_timer!()
        interval = speed_sl.value[]
        play_timer[] = Timer(interval; interval = interval) do _
            if current_step[] < length(get_rs().steps)
                current_step[] += 1
            else
                is_playing[] = false
            end
        end
    end
    on(btn_play.clicks) do _; is_playing[] = !is_playing[]; end
    on(is_playing) do playing
        btn_play.label = playing ? "⏸ Pause" : "▶ Play"
        playing ? start_timer!() : stop_timer!()
    end
    on(speed_sl.value) do _; is_playing[] && start_timer!(); end    # speed change takes effect immediately

    # ---- initial draw ----
    rs0 = get_rs()
    draw_ocean_mask!(rs0); draw_all_routes!(); draw_position_marker!(rs0)
    draw_great_circle!(rs0); draw_grid!(rs0); fit_view!()
    draw_zoom!(rs0, 1); update_stats!(rs0, 1); clear_step_labels!(); add_step_label!(rs0, 1)

    GLMakie.display(screen, fig)
    isinteractive() || wait(screen)
    return fig
end

# =============================================================================
# 7. Persisting / reloading a stepper set
#
# Not serialised: wind_pool/interp (the LRU only holds whatever was cached at save
# time, so a dump would silently go stale) and nb_flat (multi-GB per run, and fully
# determined by grid_points / ocean_mask / Rmax). Both are rebuilt on load. The
# domain (grid_points, ocean_mask) is written once, shared across runs. The summary
# DataFrames live in the CSVs written by run_radius_convergence_hifi.
# =============================================================================

"""
    save_steppers(path, steppers)

Persist a set of solved routes to a `.jld2` file. All steppers must share one
grid_points / ocean_mask (true for anything from run_radius_convergence_hifi).
"""
function save_steppers(path::String, steppers::Dict{RouteKey, RouteStepper2})
    isempty(steppers) && error("Nothing to save.")
    run_keys = collect(keys(steppers))
    first_rs = steppers[run_keys[1]]
    for key in run_keys
        rs = steppers[key]
        (size(rs.grid_points) == size(first_rs.grid_points) && rs.ocean_mask == first_rs.ocean_mask) ||
            error("save_steppers: grid_points/ocean_mask differ across runs — save mismatched domains to separate files.")
    end

    jldopen(path, "w") do f
        f["run_keys"]           = run_keys
        f["shared/grid_points"] = first_rs.grid_points
        f["shared/ocean_mask"]  = first_rs.ocean_mask
        for key in run_keys
            variant, mode, rmax = key
            g  = "run_$(variant)_$(mode)_$(rmax)"
            rs = steppers[key]
            f["$g/steps"]           = rs.steps
            f["$g/rmax"]            = rmax
            f["$g/total_confirmed"] = rs.total_confirmed
            f["$g/ct"] = rs.wind.ct; f["$g/cd"] = rs.wind.cd; f["$g/cm"] = rs.wind.cm; f["$g/cy"] = rs.wind.cy
        end
    end
    println("Saved $(length(steppers)) stepper run(s) to $path (neighbour caches rebuilt on load, not stored)")
end

"""
    load_steppers(path; wind_cache_size = 6) -> Dict{RouteKey, RouteStepper2}

Reload a set written by `save_steppers` without re-solving. The wind source that was
active when the routes were solved (ERA5 files, archive, ...) must be active again,
since the zoom pane reads wind live. The neighbour cache is rebuilt per Rmax with the
default `:full` heading set; routes solved with `heading_mode = :eight` will show the
wrong candidate edges in the zoom pane.
"""
function load_steppers(path::String; wind_cache_size::Int = 6)
    steppers = Dict{RouteKey, RouteStepper2}()
    jldopen(path, "r") do f
        grid_points = f["shared/grid_points"]
        ocean_mask  = f["shared/ocean_mask"]
        n_rows, n_cols = size(grid_points)
        nb_by_rmax = Dict{Int,NbCache}()                    # one cache per Rmax, shared across variant/mode

        for key in f["run_keys"]
            variant, mode, rmax = key
            g = "run_$(variant)_$(mode)_$(rmax)"
            ct = f["$g/ct"]; cd = f["$g/cd"]; cm = f["$g/cm"]; cy = f["$g/cy"]

            nb_flat = get!(nb_by_rmax, rmax) do
                build_neighbour_cache(grid_points, ocean_mask, n_rows, n_cols, rmax, NEIGHBOUR_TARGET_SPACING_M)
            end

            wind_pool = WindFieldPool(; max_cached = wind_cache_size)
            init_ep   = _ensureEpochRegistered!(wind_pool, ct, cd, cm, cy, 0)
            interp    = build_interpolator(grid_points, _getField!(wind_pool, init_ep))
            wind      = WindCache(wind_pool, interp, ct, cd, cm, cy)

            steppers[key] = RouteStepper2(f["$g/steps"], grid_points, ocean_mask, nb_flat, wind,
                                          f["$g/total_confirmed"])
        end
    end
    println("Loaded $(length(steppers)) stepper run(s) from $path")
    return steppers
end

# =============================================================================
# 8. launch_forecast_actual_explorer
#
# One route, two clocks. `actual_stepper` is a real-wind replay of `fc_stepper`'s own
# waypoints and power schedule, so at waypoint k the only difference is which clock you
# read: forecast-predicted vs real cumulative time/energy. The route is therefore drawn
# once (coloured by the shared power schedule) with one position marker.
# =============================================================================

function launch_forecast_actual_explorer(fc_stepper::RouteStepper2, actual_stepper::RouteStepper2)
    length(fc_stepper.steps) == length(actual_stepper.steps) ||
        error("launch_forecast_actual_explorer: steppers must share the same waypoint sequence " *
              "(actual should be a replay of fc's own path) -- got " *
              "$(length(fc_stepper.steps)) vs $(length(actual_stepper.steps)) steps.")

    steps_fc = fc_stepper.steps
    steps_ac = actual_stepper.steps
    n = length(steps_fc)
    grid_points = fc_stepper.grid_points

    fig = GLMakie.Figure(size = (2200, 1400))
    GLMakie.Label(fig[0, 1:3], "Forecast vs Actual (real-wind replay) Explorer";
                  fontsize = 22, font = :bold, halign = :left, tellwidth = false)

    ax_map  = GLMakie.Axis(fig[1, 1]; title = "Route (colored by planned power)", xlabel = "lon", ylabel = "lat")
    ax_zoom = GLMakie.Axis(fig[1, 2]; title = "Neighbourhood + wind", xlabel = "lon", ylabel = "lat")
    wind_range = GLMakie.Observable((0.0, 15.0))
    GLMakie.Colorbar(fig[1, 3]; colorrange = wind_range, colormap = :viridis, label = "wind speed (m/s)")
    GLMakie.colsize!(fig.layout, 1, GLMakie.Relative(0.55))
    GLMakie.colsize!(fig.layout, 2, GLMakie.Relative(0.40))
    GLMakie.colsize!(fig.layout, 3, GLMakie.Relative(0.05))

    draw_ocean_background!(ax_map, grid_points, fc_stepper.ocean_mask)

    lons   = [s.lon for s in steps_fc]
    lats   = [s.lat for s in steps_fc]
    powers = [s.leg_power_kw for s in steps_fc]
    prange = extrema(powers)
    prange = prange[2] > prange[1] ? prange : (prange[1] - 1.0, prange[2] + 1.0)

    GLMakie.lines!(ax_map, lons, lats; color = powers, colormap = :thermal, colorrange = prange, linewidth = 3.0)
    GLMakie.Colorbar(fig[2, 1], limits = prange, colormap = :thermal, label = "power (kW)", vertical = false)
    GLMakie.scatter!(ax_map, [lons[1]],   [lats[1]];   color = :lime, markersize = 14, marker = :diamond, label = "start")
    GLMakie.scatter!(ax_map, [lons[end]], [lats[end]]; color = :red,  markersize = 14, marker = :diamond, label = "goal")
    marker_pos = GLMakie.Observable(GLMakie.Point2f(lons[1], lats[1]))
    GLMakie.scatter!(ax_map, marker_pos; color = :magenta, markersize = 20, marker = :star5, label = "current waypoint")
    GLMakie.axislegend(ax_map; position = :lb, labelsize = 10)

    show_actual_wind = GLMakie.Observable(false)

    function draw_zoom!(k::Int)
        GLMakie.empty!(ax_zoom)
        s = steps_fc[k]
        wc  = show_actual_wind[] ? actual_stepper.wind : fc_stepper.wind
        t_s = show_actual_wind[] ? steps_ac[k].cum_time_s : s.cum_time_s
        U, V = get_wind(wc, t_s)
        draw_wind_neighbourhood!(ax_zoom, U, V, grid_points, s.i, s.j, wind_range)
        GLMakie.scatter!(ax_zoom, [s.lon], [s.lat]; color = :black, markersize = 16, marker = :star5)
        ax_zoom.title = "Step $k/$n — node ($(s.i),$(s.j)) — " *
                        (show_actual_wind[] ? "ACTUAL (ERA5)" : "FORECAST (GFS)") *
                        " wind @ t=$(round(t_s/3600, digits=1))h"
    end

    stats_text = GLMakie.Observable("")
    function update_stats!(k::Int)
        sf = steps_fc[k]; sa = steps_ac[k]
        stats_text[] = @sprintf(
            "waypoint %d / %d   node (%d,%d)\n\nFORECAST (GFS)\n  cum time:   %.2f h\n  cum energy: %.1f kWh\n  leg speed:  %.2f kn\n  leg power:  %.0f kW\n\nACTUAL (ERA5 replay)\n  cum time:   %.2f h\n  cum energy: %.1f kWh\n  leg speed:  %.2f kn\n  leg power:  %.0f kW\n\nΔ actual − forecast\n  time:   %+.2f h\n  energy: %+.1f kWh",
            k, n, sf.i, sf.j,
            sf.cum_time_s / 3600, sf.cum_energy_kwh, sf.leg_speed_kn, sf.leg_power_kw,
            sa.cum_time_s / 3600, sa.cum_energy_kwh, sa.leg_speed_kn, sa.leg_power_kw,
            (sa.cum_time_s - sf.cum_time_s) / 3600, sa.cum_energy_kwh - sf.cum_energy_kwh)
    end
    GLMakie.Label(fig[3, 1:3], stats_text; tellwidth = false, halign = :left, valign = :top,
                  fontsize = 13, justification = :left)

    controls = fig[4, 1:3] = GLMakie.GridLayout()
    btn_prev = GLMakie.Button(controls[1, 1], label = "◀ Prev")
    btn_next = GLMakie.Button(controls[1, 2], label = "Next ▶")
    scrub    = GLMakie.Slider(controls[1, 3], range = 1:n, startvalue = 1)
    current_step = GLMakie.Observable(1)
    GLMakie.Label(controls[1, 4], GLMakie.lift(k -> "waypoint $k / $n", current_step))
    wind_toggle = GLMakie.Toggle(controls[1, 5], active = false)
    GLMakie.Label(controls[1, 6], "zoom pane: show ACTUAL wind (off = forecast)")

    on(btn_prev.clicks) do _; current_step[] = max(1, current_step[] - 1); end
    on(btn_next.clicks) do _; current_step[] = min(n, current_step[] + 1); end
    on(scrub.value)     do v; current_step[] = clamp(round(Int, v), 1, n); end
    on(wind_toggle.active) do a
        show_actual_wind[] = a
        draw_zoom!(current_step[])
    end
    on(current_step) do k
        marker_pos[] = GLMakie.Point2f(steps_fc[k].lon, steps_fc[k].lat)
        draw_zoom!(k); update_stats!(k)
    end

    draw_zoom!(1); update_stats!(1)
    GLMakie.display(fig)
    return fig
end