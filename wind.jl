using CSV
using DataFrames
using Plots
using Base
using DataStructures 

include("read_write_data.jl")  # Include the read/write functions
include("geo_helpers.jl")  # Include the geo helper functions

@inline function nearest_sorted_index(vals::AbstractVector{<:Real}, target::Real)::Int
    n = length(vals)
    n == 0 && throw(ArgumentError("nearest_sorted_index: empty vals"))

    i = searchsortedfirst(vals, target)
    if i <= 1
        return 1
    elseif i > n
        return n
    end

    lo = i - 1
    hi = i
    return (target - vals[lo] <= vals[hi] - target) ? lo : hi
end


function boatSpeedClosest(wind_speeds::Vector{Float64}, wind_angles::Vector{Float64}, boat_speeds::Matrix{Float64}, wind_speed::Float64, twa::Float64)
    """

    Returns the boat speed by finding the closest match to a given wind speed and true wind angle (TWA) 
    from precomputed polar performance data.

    Arguments:
    - `wind_speeds::Vector{Float64}`: Vector of known wind speeds (in m/s) from the polar data.
    - `wind_angles::Vector{Float64}`: Vector of known true wind angles (in degrees) from the polar data.
    - `boat_speeds::Matrix{Float64}`: Matrix of boat speeds (in m/s), indexed by wind angle (rows) and wind speed (columns).
    - `wind_speed::Float64`: The current wind speed to match (in m/s).
    - `twa::Float64`: The current true wind angle to match (in degrees, can be any real number).

    Returns:
    - `boat_speed::Float64`: The boat speed (in m/s) corresponding to the closest matching wind speed and angle.

    """

    # Ensure TWA is within 0-360
    twa = mod(twa, 360.0)
    if twa > 180.0
        twa = 360.0 - twa
    end

    # Find the closest wind speed and true wind angle in the polar data
    closest_wind = nearest_sorted_index(wind_speeds, wind_speed)
    closest_twa  = nearest_sorted_index(wind_angles, twa)
    boat_speed   = boat_speeds[closest_twa, closest_wind]

    return boat_speed
end


function getBoatAngle(longitude1::Float64, latitude1::Float64, longitude2::Float64, latitude2::Float64)
    """

    Calculates the bearing (direction) from a starting point to a destination point on the Earth's surface.

    Arguments:
    - `longitude1::Float64`: Longitude of the starting point (in degrees).
    - `latitude1::Float64`: Latitude of the starting point (in degrees).
    - `longitude2::Float64`: Longitude of the destination point (in degrees).
    - `latitude2::Float64`: Latitude of the destination point (in degrees).

    Returns:
    - `TaregtAngle::Float64`: The bearing (in degrees) from the start point to the destination point,
        where 0° represents north and angles increase clockwise (i.e., compass-style).

    """

    lat1 = deg2rad(latitude1)
    lat2 = deg2rad(latitude2)
    del_lat = deg2rad(latitude2 - latitude1)
    del_long= deg2rad(longitude2 - longitude1)

    y = sin(del_long) * cos(lat2)
    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(del_long)

    # Calculate the angle in radians and convert to degrees
    target_angle = atan(y, x)
    target_angle = rad2deg(target_angle)  

    # Shift so 0° is at positive y-axis and increase clockwise
    target_angle = mod(target_angle, 360)

    return target_angle
end


function getTWA(wind_direction::Float64, boat_angle::Float64)
    """

    Calculates the True Wind Angle (TWA) given the wind direction and the boat's target angle.

    Arguments:
    - `wind_direction::Float64`: The direction from which the wind is coming (in degrees, where 0° is north and angles increase clockwise).
    - `boat_angle::Float64`: The target angle of the boat (in degrees, where 0° is north and angles increase clockwise).

    Returns:
    - `twa::Float64`: The True Wind Angle (TWA) in degrees. The TWA represents the angle between the boat's heading and the direction of the wind.

    """

     if wind_direction > boat_angle
        twa = 360-wind_direction + boat_angle
    else 
        twa = boat_angle - wind_direction
    end
    
    return twa
end

function boatSpeedInterp(wind_speeds::Vector{Float64}, wind_angles::Vector{Float64}, boat_speeds::Matrix{Float64}, wind_speed::Float64, twa::Float64)
    twa = mod(twa, 360.0)
    if twa > 180.0
        twa = 360.0 - twa
    end

    # Clamp to table bounds
    wind_speed = clamp(wind_speed, wind_speeds[1], wind_speeds[end])
    twa        = clamp(twa, wind_angles[1], wind_angles[end])

    # Find bounding indices for wind speed
    wi_high = searchsortedfirst(wind_speeds, wind_speed)
    wi_low  = max(wi_high - 1, 1)
    wi_high = min(wi_high, length(wind_speeds))

    # Find bounding indices for twa
    ai_high = searchsortedfirst(wind_angles, twa)
    ai_low  = max(ai_high - 1, 1)
    ai_high = min(ai_high, length(wind_angles))

    # If exactly on a grid point no interpolation needed
    if wi_low == wi_high && ai_low == ai_high
        return boat_speeds[ai_low, wi_low]
    end

    # Interpolation weights
    if wi_low == wi_high
        tw = 0.0
    else
        tw = (wind_speed - wind_speeds[wi_low]) / (wind_speeds[wi_high] - wind_speeds[wi_low])
    end

    if ai_low == ai_high
        ta = 0.0
    else
        ta = (twa - wind_angles[ai_low]) / (wind_angles[ai_high] - wind_angles[ai_low])
    end

    # Bilinear interpolation over the 4 surrounding polar table points
    v00 = boat_speeds[ai_low,  wi_low]
    v10 = boat_speeds[ai_high, wi_low]
    v01 = boat_speeds[ai_low,  wi_high]
    v11 = boat_speeds[ai_high, wi_high]

    return (1-ta) * (1-tw) * v00 +
               ta * (1-tw) * v10 +
           (1-ta) *     tw * v01 +
               ta *     tw * v11
end


@inline function twa_and_boat_speed(
    longitude1::Float64,
    latitude1::Float64,
    longitude2::Float64,
    latitude2::Float64,
    wind_speed::Float64,
    wind_direction::Float64,
    polar::Tuple,
)::Tuple{Float64,Float64}
    wind_speeds = polar[1]
    wind_angles = polar[2]
    boat_speeds = polar[3]

    boat_angle = getBoatAngle(longitude1, latitude1, longitude2, latitude2)
    twa = getTWA(wind_direction, boat_angle)
    boat_speed = boatSpeedClosest(wind_speeds, wind_angles, boat_speeds, wind_speed, twa)

    return twa, boat_speed
end

@inline function twa_and_boat_speed(
    wind_speed::Float64,
    wind_direction::Float64,
    boat_angle::Float64,
    polar::Tuple,
)::Tuple{Float64,Float64}
    wind_speeds = polar[1]
    wind_angles = polar[2]
    boat_speeds = polar[3]

    twa = getTWA(wind_direction, boat_angle)
    boat_speed = boatSpeedClosest(wind_speeds, wind_angles, boat_speeds, wind_speed, twa)

    return twa, boat_speed
end


function GetTravelTime(longitude1::Float64, latitude1::Float64, longitude2::Float64, latitude2::Float64, wind_speed::Float64, wind_direction::Float64, polar::Tuple)
    """

    Calculates the travel time between two geographical points based on wind speed and direction, using polar data to estimate boat speed.

    Arguments:
    - `longitude1::Float64`: Longitude of the starting point.
    - `latitude1::Float64`: Latitude of the starting point.
    - `longitude2::Float64`: Longitude of the destination point.
    - `latitude2::Float64`: Latitude of the destination point.
    - `wind_speed::Float64`: Wind speed (in m/s).
    - `wind_direction::Float64`: Wind direction (in degrees, where 0° is north and angles increase clockwise).
    - `polar::Tuple`: Tuple containing three vectors: first containing wind speeds (m/s), second containing true wind angles (degrees), 
      third containing corresponding boat speeds (m/s).

    Returns:
    - `time::Float64`: The estimated travel time (in seconds) between the starting point and the destination, based on the boat's speed and the distance.
    - `twa::Float64`: The True Wind Angle (TWA) (in degrees), which is the angle between the boat's heading and the direction of the wind.
    - `boat_angle::Float64`: The target angle for the boat (in degrees), calculated based on the starting and destination points.
    - `boat_speed::Float64`: The estimated boat speed (in m/s) corresponding to the wind speed and TWA.

    """
    
    # Find the distance in meters
    distance = getDistance(longitude1, latitude1, longitude2, latitude2)  

    # Calculate the target angle for the boat in degrees
    boat_angle = getBoatAngle(longitude1, latitude1, longitude2, latitude2)

    # read in the polar data 
    #wind_speeds, wind_angles, boat_speeds = readPolar(polar)
    wind_speeds = polar[1]
    wind_angles = polar[2]
    boat_speeds = polar[3]

    # Get the twa between 0 and 360 degrees
    twa = getTWA(wind_direction, boat_angle) 

    # symetry in the polar data means we need to check if TWA is greater than 180 degrees
    if twa > 180 
         twa_temp = 360 - twa
    else 
        twa_temp = twa
    end

    # Get the closest boat speed using the polar data, wind speed and TWA
    boat_speed = boatSpeedClosest(wind_speeds, wind_angles, boat_speeds, wind_speed, twa_temp)

    # Calculate travel time in seconds 
    time = distance / boat_speed  

    return time, twa, boat_angle, boat_speed
end

function getWeightedWind(lon::Float64, lat::Float64, wind_file::String, time::Int)
    # MAKE SURE FIRECTION IS CORRECT

    file = GRIB.GribFile(wind_file)
    u_vals, v_vals = extractWindValues(file, time)
    destroy(file)
    
    # Define grid parameters
    min_lon = -180
    max_lon = 180
    min_lat = -90
    max_lat = 90
    change_lon = 0.25
    change_lat = 0.25

    # Compute raw grid indices (floating point)
    x = (lon - min_lon) / change_lon + 1
    y = (max_lat - lat) / change_lat + 1  # Note the direction (lat decreases as y increases)

    if (isinteger(x) && isinteger(y))
        i = x
        j = y
        index = CartesianIndex(i, j)
        # get the u and v values at the index
        u = u_vals[index]
        v = v_vals[index]
    else 
        # Calculate index
        i = round(Int, x) + 1
        j = round(Int, y) + 1
        index = CartesianIndex(i, j)

        # Find surrounding 4 grid points
        i_low  = floor(Int, x)
        i_high = ceil(Int, x)
        j_low  = floor(Int, y)
        j_high = ceil(Int, y)

        # Actual lon/lat values at surrounding grid points
        lon_ll = min_lon + (i_low - 1) * change_lon
        lon_hl = min_lon + (i_high - 1) * change_lon
        lat_ul = max_lat - (j_low - 1) * change_lat
        lat_dl = max_lat - (j_high - 1) * change_lat

        points = [
            (i_low,  j_low,  lon_ll, lat_ul),  # Top-left
            (i_high, j_low,  lon_hl, lat_ul),  # Top-right
            (i_low,  j_high, lon_ll, lat_dl),  # Bottom-left
            (i_high, j_high, lon_hl, lat_dl)   # Bottom-right
        ]
        println(points)

        weights = []
        us = []
        vs = []

        for (ii, jj, grid_lon, grid_lat) in points
            d = sqrt((lon - grid_lon)^2 + (lat - grid_lat)^2)
            w = 1 / (d + 1e-6)  # avoid division by zero
            push!(weights, w)
            push!(us, u_vals[ii, jj])
            push!(vs, v_vals[ii, jj])
        end
        println("Weights: ", weights)
        println("U values: ", us)
        println("V values: ", vs)

        # Weighted average of u and v
        total_w = sum(weights)
        u = sum(w * val for (w, val) in zip(weights, us)) / total_w
        v = sum(w * val for (w, val) in zip(weights, vs)) / total_w
    end

    # calculate speed and direction
    speed = sqrt(u^2 + v^2)
    θ_rad = atan(u, v) 
    wind_dir = mod(rad2deg(θ_rad)+180, 360)

    return speed, wind_dir
end

