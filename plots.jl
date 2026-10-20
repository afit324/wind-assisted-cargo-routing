using CSV
using DataFrames
using Plots
using Base
using DataStructures 
using GRIB
using CDSAPI
using CairoMakie
using GeoMakie


function plotPolar(wind_speeds::Vector{Float64}, true_wind_angles::Vector{Float64}, boat_speeds::Matrix{Float64})
    """
    Generates a polar plot of boat speed as a function of true wind angle for various wind speeds.

    Arguments:
    - `wind_speeds::Vector{Float64}`: Wind speeds (m/s) corresponding to each column of the boat_speeds matrix.
    - `true_wind_angles::Vector{Float64}`: True wind angles (degrees) corresponding to each row of the boat_speeds matrix.
    - `boat_speeds::Matrix{Float64}`: Matrix of boat speeds (m/s), where each row corresponds to a wind angle and each column to a wind speed.

    Displays:
    - A polar plot with labeled curves for each wind speed.

    """

    # Convert degrees to radians for plotting
    angles_rad = deg2rad.(true_wind_angles)

    # create plot
    plt = Plots.plot(
        proj = :polar,
        legend = :outerright,
        title = "RISE Polar No Power: Boat Speed (m/s) vs True Wind Angle (degrees)",
        titlefontsize = 8,                      # Smaller title font
        legendfontsize = 8,
        theta_direction = :clock
    )
    
    # Add curves for each wind speed
    for (i, ws) in enumerate(wind_speeds)
        Plots.plot!(angles_rad, boat_speeds[:, i], label = "Wind Speed $ws m/s")
    end

    display(plt)

end


function plotBoatVectors(wind_direction::Float64, wind_speed::Float64, boat_speed::Float64, target_angle::Float64)
    """
    Plots wind and boat velocity vectors on a polar-style compass diagram for visualizing their directions and magnitudes.

    Arguments:
    - `wind_direction::Float64`: Direction *from which* the wind is coming (in degrees, 0° = North, increases clockwise).
    - `wind_speed::Float64`: Wind speed in m/s.
    - `boat_speed::Float64`: Boat speed in m/s.
    - `target_angle::Float64`: Direction the boat is heading *towards* (in degrees, 0° = North, increases clockwise).

    Behavior:
    - Converts all angles to radians with 0 pointing north and increasing clockwise (nautical convention).

    Returns:
    - A `Plots.Plot` object showing the vectors and compass diagram.

    """

    # Convert degrees to radians (0° is north, clockwise rotation)
    wind_rad = deg2rad(mod(-(wind_direction) + 90, 360))
    target_rad = deg2rad(mod(-target_angle + 90, 360))

    # Fixed wind vector length from edge to center
    wind_len = 1.0  # fixed length
    boat_len = (boat_speed / wind_speed) * wind_len  # scaled relative to wind

    # Boat vector (from center outward)
    boat_x, boat_y = boat_len * cos(target_rad), boat_len * sin(target_rad)

    # Wind vector (from edge inward to center)
    wind_start_x, wind_start_y = cos(wind_rad), sin(wind_rad)
    wind_dx, wind_dy = -wind_len * cos(wind_rad), -wind_len * sin(wind_rad)


    # Plot unit circle
    θ = 0:pi/180:2pi
    unit_circle_x = cos.(θ)
    unit_circle_y = sin.(θ)

    plt = Plots.plot(unit_circle_x, unit_circle_y, 
        aspect_ratio=1,
        legend = false,
        linewidth=0.5,
        linecolor=:gray,
        framestyle=:none,
        xticks=false,
        yticks=false,
        xlims=(-1.5, 1.5), 
        ylims=(-1.5, 1.5),
        background_color=:white,
        title="Vessel and Wind Vectors",
        titlefontsize=10)

    # Add degree annotations around the compass
    for deg in 0:30:330
        rad = deg2rad(mod(-deg + 90, 360))
        x = 1.25 * cos(rad)
        y = 1.25 * sin(rad)
        Plots.annotate!(x, y, Plots.text("$(deg)°", 8, :gray))
    end

    # Add vectors
    Plots.quiver!([0], [0], quiver=([boat_x], [boat_y]), arrow=true, color=:blue, label="Boat", lw=2)
    Plots.quiver!([wind_start_x], [wind_start_y], quiver=([wind_dx], [wind_dy]), arrow=true, color=:orange, label="Wind", lw=2)
    # quiver!([0], [0], quiver=([1.2 * cos(target_rad)], [1.2 * sin(target_rad)]), arrow=true, color=:black, label="Target", lw=1.5, linestyle=:dash)

    # Add descriptive text
    Plots.annotate!(-1.8, -0.8, Plots.text("Vessel: $(round(target_angle, digits=1))°, $(round(boat_speed, digits=2)) m/s", :blue, 9))
    Plots.annotate!(-1.8, -1, Plots.text("Wind: $(round(wind_direction, digits=1))°, $(round(wind_speed, digits=2)) m/s", :orange, 9))

    # Add cardinal directions (N, E, S, W)
    cardinals = Dict("N" => 0, "E" => 90, "S" => 180, "W" => 270)
    for (label, deg) in cardinals
        rad = deg2rad(mod(-deg + 90, 360))
        x = 1.45 * cos(rad)
        y = 1.45 * sin(rad)
        Plots.annotate!(x, y, Plots.text(label, 10, :black, :bold))
    end

    return plt
end


function plotCurrentGird(grid_points, uos, vos)
    """
    Plots the current vectors (magnitude + direction) at given locations.

    Arguments:
    - `grid_points`: A list of tuples containing the latitude and longitude of the locations in order (long, lat).
    - `uos`: Zonal current (east-west component).
    - `vos`: Meridional current (north-south component).

    Returns:
    - A vector plot showing ocean current direction and speed
    """

    lons = [lon for (lon, lat) in grid_points]
    lats = [lat for (lon, lat) in grid_points]
    scale = 2.0

    # Plot route
    Plots.scatter(lons, lats, xlabel="Longitude", ylabel="Latitude", title="Route with Currents", legend=false)
    Plots.quiver!(lons, lats, quiver=(scale .* uos, scale .* vos))
end

function plotCurrentGirdWithColorBar(uo, vo, locations)
    """
    Plots the current vectors (magnitude + direction) at given locations with color bar for speed.

    Arguments:
    - `uo`: Zonal current (east-west component)
    - `vo`: Meridional current (north-south component)
    - `locations`: A list of (longitude, latitude) tuples

    Returns:
    - A vector plot showing ocean current direction and speed
    """

    lons = [long for (long, lat) in locations]
    lats = [lat for (long, lat) in locations]

    # Compute speed for color
    speeds = sqrt.(uo.^2 .+ vo.^2)

    fig = Figure(resolution=(800, 600))
    ax = Axis(fig[1, 1], title="Current Speed and Direction", xlabel="Longitude", ylabel="Latitude")

    arrows!(ax, lons, lats, uo, vo,
            arrowsize=5,
            linewidth=1.5,
            lengthscale = 1,
            color=speeds,
            colormap=:plasma,
            normalize=false)

    Colorbar(fig[1, 2], colormap=:plasma, limits=extrema(speeds), label="Speed (m/s)")
    # add dots for the locations
    CairoMakie.scatter!(ax, lons, lats, markersize=10, color=:black, label="Locations")
    fig
end


function plotPathWithWindMap(path, wind_speeds, wind_directions)
    """
    Plots a sailing path with wind vectors at each point on the path.
    
    # Arguments:
    - `path::Vector{Tuple{Float64, Float64}}`: (lon, lat) points of the route.
    - `wind_speeds::Vector{Float64}`: wind speed (m/s) at each path point.
    - `wind_directions::Vector{Float64}`: wind direction (degrees) at each path point.
    """
    # Extract coordinates
    lons = [p[1] for p in path]
    lats = [p[2] for p in path]
    
    # Compute arrow components (u = eastward, v = northward)
    # Note: wind direction = FROM direction, arrow points TO where wind is going.
    # Standard: u = -speed * sin(θ), v = -speed * cos(θ) for meteorological wind direction.
    # Adjust sign according to your convention.
    θ_rad = deg2rad.(wind_directions)
    us = wind_speeds .* sin.(θ_rad)   # eastward component
    vs = wind_speeds .* cos.(θ_rad)   # northward component
    
    # Determine map limits (add padding)
    min_lon = minimum(lons) - 5
    max_lon = maximum(lons) + 5
    min_lat = minimum(lats) - 5
    max_lat = maximum(lats) + 5
    
    fig = Figure()
    ga = GeoAxis(fig[1, 1];
        dest = "+proj=longlat +datum=WGS84",
        limits = ((min_lon, max_lon), (min_lat, max_lat))
    )
    
    # Background: Earth image + land polygons
    image!(ga, -180..180, -90..90, GeoMakie.earth() |> rotr90; interpolate = false)
    poly!(ga, GeoMakie.land(); color = :lightyellow, strokecolor = :black, strokewidth = 0.5)
    
    # Wind arrows along the path
    arrows!(ga, lons, lats, us, vs;
        arrowsize = 10,
        linewidth = 1.5,
        lengthscale = 0.3,
        color = wind_speeds,
        colormap = :plasma,
        normalize = false
    )
    
    # Sailing path line
    lines!(ga, lons, lats; color = :blue, linewidth = 2, label = "Sailing Path")
    
    # Start and end markers
    scatter!(ga, [lons[1]], [lats[1]]; markersize = 12, color = :green, label = "Start")
    text!(ga, lons[1], lats[1]; text = "Start", align = (:right, :bottom), fontsize = 11, color = :green, font = :bold)
    scatter!(ga, [lons[end]], [lats[end]]; markersize = 12, color = :red, label = "End")
    text!(ga, lons[end], lats[end]; text = "End", align = (:left, :top), fontsize = 11, color = :red, font = :bold)
    
    # Colorbar for wind speed
    Colorbar(fig[1, 2];
        colormap = :plasma,
        limits = extrema(wind_speeds),
        label = "Wind speed (m/s)",
        height = Relative(0.5)
    )
    
    # Optional legend
    # axislegend(ga; position = :rb)
    
    fig
end

function plotPathWindHeatMap(grid, path, wind_speeds, wind_directions)
    """
    Plots a sailing path with wind vectors on a grid of geographic points.

    # Arguments:
    - `grid::Vector{Vector{Tuple{Float64, Float64}}}`: A vector of vectors representing geographic points (longitude, latitude) in a grid format.
    - `path::Vector{Tuple{Float64, Float64}}`: A vector of tuples (longitude, latitude) representing the sailing path.
    - `wind_speeds::Vector{Vector{Float64}}`: A vector of wind speeds corresponding to the path points.
    - `wind_directions::Vector{Vector{Float64}}`: A vector of wind directions (in degrees) corresponding to the path points.
    """
    new_grid = vec(grid)
    wind_speeds = vec(wind_speeds)
    wind_directions = vec(wind_directions)
    xs = [Float32(lon) for (lon, lat) in new_grid]
    ys = [Float32(lat) for (lon, lat) in new_grid]
    minimum_x = minimum(xs)
    minimum_y = minimum(ys)             
    maximum_x = maximum(xs)
    maximum_y = maximum(ys)
    us  = [Float32(speed * cos(deg2rad(direction))) for (speed, direction) in zip(wind_speeds, wind_directions)]
    vs  = [Float32(speed * sin(deg2rad(direction))) for (speed, direction) in zip(wind_speeds, wind_directions)] 
    fig = Figure()
    ga = GeoAxis(
        fig[1, 1]; # any cell of the figure's layout
        dest = "+proj=longlat +datum=WGS84", # the CRS in which you want to plot,
        limits = ((minimum_x - 10, maximum_x+ 10), (minimum_y- 10, maximum_y+ 10)), # the limits of the plot
    )
    image!(ga, -180..180, -90..90, GeoMakie.earth() |> rotr90; interpolate = false)
    CairoMakie.contourf!(ga, xs, ys, wind_speeds; colormap=:plasma,
    xautolimits=false, yautolimits=false, transparency = true)
    lp = lines!(ga, path; color = :blue, linewidth = 2, label = "Sailing Path", xautolimits = false, yautolimits = false)
    arrows!(ga, xs, ys, us, vs,
                arrowsize=3,
                linewidth=1.0,
                lengthscale = 0.5,
                normalize=false)
    poly!(ga, GeoMakie.land(); color = :lightyellow, strokecolor = :black, strokewidth = 1)
    Colorbar(fig[1,2], colormap=:plasma, limits=extrema(wind_speeds), label="Speed (m/s)", height = Relative(0.5))
    axislegend(ga)
    fig
end

function plotPathWindHeatMapPowers(grid, path, wind_speeds, wind_directions, powers)
    """
    Plots a sailing path with wind vectors on a grid of geographic points.

    # Arguments:
    - `grid::Vector{Vector{Tuple{Float64, Float64}}}`: A vector of vectors representing geographic points (longitude, latitude) in a grid format.
    - `path::Vector{Tuple{Float64, Float64}}`: A vector of tuples (longitude, latitude) representing the sailing path.
    - `wind_speeds::Vector{Vector{Float64}}`: A vector of wind speeds corresponding to the path points.
    - `wind_directions::Vector{Vector{Float64}}`: A vector of wind directions (in degrees) corresponding to the path points.
    - `powers::Vector{Float64}`: A vector of power values (in kW) corresponding to the path points.
    """
    new_grid = vec(grid)
    wind_speeds = vec(wind_speeds)
    wind_directions = vec(wind_directions)
    xs = [Float32(lon) for (lon, lat) in new_grid]
    ys = [Float32(lat) for (lon, lat) in new_grid]
     # Path points
    path_xs = [Float32(lon) for (lon, lat) in path]
    path_ys = [Float32(lat) for (lon, lat) in path]

    minimum_x = minimum(xs)
    minimum_y = minimum(ys)             
    maximum_x = maximum(xs)
    maximum_y = maximum(ys)
    us  = [Float32(speed * cos(deg2rad(direction))) for (speed, direction) in zip(wind_speeds, wind_directions)]
    vs  = [Float32(speed * sin(deg2rad(direction))) for (speed, direction) in zip(wind_speeds, wind_directions)] 
    fig = Figure()
    ga = GeoAxis(
        fig[1, 1]; # any cell of the figure's layout
        dest = "+proj=longlat +datum=WGS84", # the CRS in which you want to plot,
        limits = ((minimum_x - 10, maximum_x+ 10), (minimum_y- 10, maximum_y+ 10)), # the limits of the plot
    )
    image!(ga, -180..180, -90..90, GeoMakie.earth() |> rotr90; interpolate = false)
    CairoMakie.contourf!(ga, xs, ys, wind_speeds; colormap=:plasma,
    xautolimits=false, yautolimits=false, transparency = true)
    poly!(ga, GeoMakie.land(); color = :lightyellow, strokecolor = :black, strokewidth = 1)
    lp = lines!(ga, path; color = :blue, linewidth = 2, label = "Sailing Path", xautolimits = false, yautolimits = false)
    CairoMakie.scatter!(ga, path_xs, path_ys;
        color = powers,
        colormap = :coolwarm,
        markersize = 12,
        strokewidth = 0.5,
        strokecolor = :black,
        label = "Power (kW)"
    )
    Colorbar(fig[1,2], colormap=:plasma, limits=extrema(wind_speeds), label="Speed (m/s)", height = Relative(0.5))
    Colorbar(fig[1,3], colormap=:coolwarm, limits=extrema(powers), label="Power (kW)", height = Relative(0.5))
    fig
end

function plotPathGridMap(grid, path)
    """
    Plots a sailing path on a grid of geographic points.

    # Arguments:
    - `grid::Vector{Vector{Tuple{Float64, Float64}}}`: A vector of vectors representing geographic points (longitude, latitude) in a grid format.
    - `path::Vector{Tuple{Float64, Float64}}`: A vector of tuples (longitude, latitude) representing the sailing path.
    """
    new_grid = reduce(vcat, grid)
    xs = [Float32(lon) for (lon, lat) in new_grid]
    ys = [Float32(lat) for (lon, lat) in new_grid]
    minimum_x = minimum(xs)
    minimum_y = minimum(ys)
    maximum_x = maximum(xs)
    maximum_y = maximum(ys)
    fig = Figure()
    ga = GeoAxis(
        fig[1, 1]; # any cell of the figure's layout
        dest = "+proj=longlat +datum=WGS84", # the CRS in which you want to plot,
        limits = ((minimum_x - 10, maximum_x+ 10), (minimum_y- 10, maximum_y+ 10)), # the limits of the plot
    )
    image!(ga, -180..180, -90..90, GeoMakie.earth() |> rotr90; interpolate = false)
    poly!(ga, GeoMakie.land(); color = :lightyellow, strokecolor = :black, strokewidth = 1)
    lp = lines!(ga, path; color = :blue, linewidth = 2, label = "Sailing Path", xautolimits = false, yautolimits = false)
    CairoMakie.scatter!(ga, xs, ys; color = :gray, markersize = 3, label = "Grid Points", xautolimits = false, yautolimits = false)
    axislegend(ga)
    fig
end

function plotPathsMap(path1, path2)
    """
    Plots two sailing path on a map, one smoothed, one path1.

    # Arguments:
    - `path1::Vector{Tuple{Float64, Float64}}`: A vector of tuples (longitude, latitude) representing the sailing path.
    - `path2::Vector{Tuple{Float64, Float64}}`: A vector of tuples (longitude, latitude) representing the smoothed sailing path.
    """
    xs = [Float32(lon) for (lon, lat) in path1]
    ys = [Float32(lat) for (lon, lat) in path1]
    minimum_x = minimum(xs)
    minimum_y = minimum(ys)
    maximum_x = maximum(xs)
    maximum_y = maximum(ys)
    fig = Figure()
    ga = GeoAxis(
        fig[1, 1]; # any cell of the figure's layout
        dest = "+proj=longlat +datum=WGS84", # the CRS in which you want to plot,
        limits = ((minimum_x - 10, maximum_x+ 10), (minimum_y- 10, maximum_y+ 10)), # the limits of the plot
    )
    image!(ga, -180..180, -90..90, GeoMakie.earth() |> rotr90; interpolate = false)
    poly!(ga, GeoMakie.land(); color = :lightyellow, strokecolor = :black, strokewidth = 1)
    lp = lines!(ga, path1; color = :blue, linewidth = 2, label = "Sailing Path", xautolimits = false, yautolimits = false)
    lp2 = lines!(ga, path2; color = :red, linewidth = 2, label = "Smooted Sailing Path", xautolimits = false, yautolimits = false)
    fig 
end

function plotHeatMapTimes(costs_vector, num_days_per_contour, path)
    """
    Plots the contours of time from the result of running the routing algorithm. Also plots the path.

    # Arguments:
    - `costs_vector` - the costs_vector outputted from the algorith, which is a array of tuples containg longitude, 
        lattitude and time to each that location e.g. (long, lat, time)
    - ` num_days_per_contour` - the number of days each contour represents
    - `path` - the path you want to visualise on the plot which is a array of tuples containg the longitude and latitide
        e.g. (long, lat)

    """
    days_in_seconds = 24 * 3600

    xs = [Float32(lon) for (lon, lat, time) in costs_vector]
    ys = [Float32(lat) for (lon, lat, time) in costs_vector]
    # times = [Float32(rem(time, days_in_seconds*num_days_per_contour)) for (lon, lat, time) in costs_vector]
    times = [Float32(time) for (lon, lat, time) in costs_vector]
    minimum_x = minimum(xs)
    minimum_y = minimum(ys)             
    maximum_x = maximum(xs)
    maximum_y = maximum(ys)
    fig = Figure()
    ga = GeoAxis(
        fig[1, 1]; # any cell of the figure's layout
        dest = "+proj=longlat +datum=WGS84", # the CRS in which you want to plot,
        limits = ((minimum_x - 10, maximum_x+ 10), (minimum_y- 10, maximum_y+ 10)), # the limits of the plot
    )
    image!(ga, -180..180, -90..90, GeoMakie.earth() |> rotr90; interpolate = false)
    CairoMakie.contourf!(ga, xs, ys, times; colormap=:plasma,
    xautolimits=false, yautolimits=false, transparency = true, levels = Int(round(maximum(times)/(days_in_seconds*num_days_per_contour))))
    lp = lines!(ga, path; color = :blue, linewidth = 2, label = "Sailing Path", xautolimits = false, yautolimits = false)
    poly!(ga, GeoMakie.land(); color = :lightyellow, strokecolor = :black, strokewidth = 1)
    fig
end

function plotHeatMapTimesLinesDistance(costs_vector, num_days_per_contour, path, const_wind_circular_contours)
    """
    Plots the contours of time from the result of running the routing algorithm. Also plots the path.

    # Arguments:
    - `costs_vector` - the costs_vector outputted from the algorith, which is a array of tuples containg longitude, 
        lattitude and time to each that location e.g. (long, lat, time)
    - ` num_days_per_contour` - the number of days each contour represents
    - `path` - the path you want to visualise on the plot which is a array of tuples containg the longitude and latitide
        e.g. (long, lat)

    """
    days_in_seconds = 24 * 3600


    xs = [Float32(lon) for (lon, lat, time) in costs_vector]
    ys = [Float32(lat) for (lon, lat, time) in costs_vector]
    times = [Float32(time) for (lon, lat, time) in costs_vector]
    xs_lines = [Float32(lon) for (lon, lat, time) in const_wind_circular_contours]
    ys_lines = [Float32(lat) for (lon, lat, time) in const_wind_circular_contours]
    times_lines = [time for (lon, lat, time) in const_wind_circular_contours]
    minimum_x = minimum(xs)
    minimum_y = minimum(ys)             
    maximum_x = maximum(xs)
    maximum_y = maximum(ys)
    fig = Figure()
    ga = GeoAxis(
        fig[1, 1]; # any cell of the figure's layout
        dest = "+proj=longlat +datum=WGS84", # the CRS in which you want to plot,
        limits = ((minimum_x - 10, maximum_x+ 10), (minimum_y- 10, maximum_y+ 10)), # the limits of the plot
    )
    image!(ga, -180..180, -90..90, GeoMakie.earth() |> rotr90; interpolate = false)
    CairoMakie.contourf!(ga, xs, ys, times; colormap=:plasma,
    xautolimits=false, yautolimits=false, transparency = true, levels = Int(round(maximum(times)/(days_in_seconds*num_days_per_contour))))
    CairoMakie.contour!(ga, xs_lines, ys_lines, times_lines; levels=10, color=:black, linewidth=1.0, linestyle=:dash, label="Constant Wind Circular Contours", xautolimits = false, yautolimits = false)
    lp = lines!(ga, path; color = :blue, linewidth = 2, label = "Sailing Path", xautolimits = false, yautolimits = false)
    poly!(ga, GeoMakie.land(); color = :lightyellow, strokecolor = :black, strokewidth = 1)
    fig
end


  


function plotTwoPathsPowerBin(
    path1, power1,
    path2, power2;
    label1 = "V1",
    label2 = "V2",
    title = "Route comparison with engine power"
)

    @assert length(path1) > 1
    @assert length(path2) > 1

    lons1 = first.(path1)
    lats1 = last.(path1)

    lons2 = first.(path2)
    lats2 = last.(path2)

    # Handle either node powers or segment powers
    seg_power1 = length(power1) == length(path1) ? power1[1:end-1] : power1
    seg_power2 = length(power2) == length(path2) ? power2[1:end-1] : power2

    @assert length(seg_power1) == length(path1)-1
    @assert length(seg_power2) == length(path2)-1

    pmin = min(minimum(seg_power1), minimum(seg_power2))
    pmax = max(maximum(seg_power1), maximum(seg_power2))

    if pmin == pmax
        pmax += 1e-9
    end

    all_lons = vcat(lons1, lons2)
    all_lats = vcat(lats1, lats2)

    fig = CairoMakie.Figure(size = (1400, 800))

    ga = GeoMakie.GeoAxis(
        fig[1, 1];
        dest = "+proj=longlat +datum=WGS84",
        title = title,
        limits = (
            (minimum(all_lons)-5, maximum(all_lons)+5),
            (minimum(all_lats)-5, maximum(all_lats)+5)
        )
    )

    CairoMakie.image!(
        ga,
        -180..180,
        -90..90,
        GeoMakie.earth() |> rotr90;
        interpolate = false
    )

    GeoMakie.poly!(
        ga,
        GeoMakie.land();
        color = :lightyellow,
        strokecolor = :black,
        strokewidth = 0.5
    )

    ############################
    # V1 Route
    ############################

    for k in 1:(length(path1)-1)

        CairoMakie.lines!(
            ga,
            [lons1[k], lons1[k+1]],
            [lats1[k], lats1[k+1]];
            color = seg_power1[k],
            colorrange = (pmin, pmax),
            colormap = :plasma,
            linewidth = 4,
            linestyle = :solid
        )

    end

    ############################
    # V2 Route
    ############################

    for k in 1:(length(path2)-1)

        CairoMakie.lines!(
            ga,
            [lons2[k], lons2[k+1]],
            [lats2[k], lats2[k+1]];
            color = seg_power2[k],
            colorrange = (pmin, pmax),
            colormap = :plasma,
            linewidth = 4,
            linestyle = :dash
        )

    end

    ############################
    # Start / End
    ############################

    CairoMakie.scatter!(
        ga,
        [lons1[1]],
        [lats1[1]];
        color = :green,
        markersize = 18
    )

    CairoMakie.scatter!(
        ga,
        [lons1[end]],
        [lats1[end]];
        color = :red,
        markersize = 18
    )

    ############################
    # Colourbar
    ############################

    CairoMakie.Colorbar(
        fig[1, 2];
        colormap = :plasma,
        limits = (pmin, pmax),
        label = "Engine power (kW)"
    )

    ############################
    # Layout sizing
    ############################

    CairoMakie.colsize!(fig.layout, 1, CairoMakie.Relative(0.92))
    CairoMakie.colsize!(fig.layout, 2, CairoMakie.Relative(0.08))

    ############################
    # Manual legend
    ############################

    v1_handle = CairoMakie.LineElement(
        color = :black,
        linewidth = 3,
        linestyle = :solid
    )

    v2_handle = CairoMakie.LineElement(
        color = :black,
        linewidth = 3,
        linestyle = :dash
    )

    start_handle = CairoMakie.MarkerElement(
        color = :green,
        marker = :circle
    )

    end_handle = CairoMakie.MarkerElement(
        color = :red,
        marker = :circle
    )

    legend = CairoMakie.Legend(
        fig,
        [v1_handle, v2_handle, start_handle, end_handle],
        [label1, label2, "Start", "End"];
        tellwidth = false,
        tellheight = false
    )

    CairoMakie.translate!(legend.blockscene, 30, 30, 100)

    fig[1,1] = ga

    return fig

end