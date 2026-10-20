using CSV
using DataFrames
using Plots
using Base
using DataStructures 
using Dates
using GRIB
using Geodesics

include("read_write_data.jl")  
include("wind.jl")  
include("times_and_dates.jl") 


function checkTimeFromPath(path::Vector{Tuple{Float64, Float64}}, polar_file::String, cur_time::Int, cur_day::Int, cur_month::Int, cur_year::Int)

    """
    Calculates the time taken to travel along a given path, given the polars and the start date and time.

    # Arguments
    - `path::Vector{Tuple{Float64, Float64}}`: A vector of tuples representing the path, where each tuple contains (longitude, latitude).
    - `polar_file::String`: Path to the polar data file.
    - `cur_time::Int`: Current time in HHMM format (e.g., 1345 for 1:45 PM).
    - `cur_day::Int`: Current day of the month (1-31).
    - `cur_month::Int`: Current month (1-12).
    - `cur_year::Int`: Current year (e.g., 2025).

    # Returns
    - `times::Vector{Int}`: A vector of times in HHMM format for each point in the path.
    - `wind_info::Vector{Tuple{Float64, Float64}}`: A vector of tuples containing wind speed and direction for each point in the path.

    """

    # Load polar data
    wind_speeds, wind_angles, boat_speeds = readPolar(polar_file)
    polar_info = (wind_speeds, wind_angles, boat_speeds)
    twas = []
    boat_speeds = []

    times = [0]  # Initialize with zero for the first point
    wind_info = []

    total_dist = 0

    for i in 1:length(path)-1
        p1 = path[i]
        p2 = path[i+1]

        new_year, new_month, new_day, use_time = addSeconds(cur_time, cur_day, cur_month, cur_year, times[i])
        new_year, new_month, new_day, use_time = roundTime(use_time, new_day, new_month, new_year)
        word_month = monthWord(new_month)

        wind_file = "./OceanData/$(use_time)-$(new_day)$(word_month)$(new_year).grib"
        println("Processing segment $(i): From $(p1) to $(p2)")
        println("Wind file: $(wind_file)")

        if !isfile(wind_file)
            str_time = timeString(use_time)
            getWindData(wind_file, new_year, new_month, new_day, [str_time], [90, -180, -90, 180])
        end

        println(wind_file)
        file = GRIB.GribFile(wind_file)
        u_msg, v_msg = extractWindMessages(file, use_time)

        wind_speed, wind_dir = getWindAngleSpeedFromMessages(u_msg, v_msg, p1[1], p1[2])

        # Calculate travel time between p1 and p2
        t, twa, _, boat_speed = GetTravelTime(p1[1], p1[2], p2[1], p2[2],
                                   wind_speed, wind_dir, polar_info)
        push!(twas, twa)
        push!(boat_speeds, boat_speed)
        t = round(Int, t)  # Ensure t is an integer
        distance = getDistance(p1[1], p1[2], p2[1], p2[2])
        push!(times, times[i] + t)
        push!(wind_info, (wind_speed, wind_dir))
        #println("Segment $(i): Time from $(p1) to $(p2) is $(t) seconds, total time is $(times[i+1]) seconds.")
        #println("days to get to current point: $(times[i] / 86400)")
        #println("days to get to next point: $(t/86400) ")
        #println("Total time to get to next point: $(times[i+1] / 86400) days")
        #println("Distance from $(p1) to $(p2): $(distance) km")
        println("Wind speed: $(wind_speed) m/s, Wind direction: $(wind_dir) degrees, Boat speed: $(boat_speed) m/s, TWA: $(twa) degrees")
        total_dist = total_dist + distance

        destroy(file)  # Clean up the GRIB file to free resources
    end
    #println("Total distance traveled: $(total_dist) m")
    push!(wind_info, (0.0, 0.0))  # Add 0 values for wind at end of path
    push!(boat_speeds, 0.0)  # Add 0 value for boat speed at end of path
    push!(twas, 0.0)  # Add 0 value for twa at end of path
    println("Total distance traveled: $(total_dist / 1000) km")
    return times, wind_info, boat_speeds, twas

end

function getWindAlongPathPower(path::Vector{Tuple{Float64, Float64}}, powers::Vector{}, cur_time::Int, cur_day::Int, cur_month::Int, cur_year::Int)

    """
    Calculates the time taken to travel along a given path, given the polars and the start date and time.

    # Arguments
    - `path::Vector{Tuple{Float64, Float64}}`: A vector of tuples representing the path, where each tuple contains (longitude, latitude).
    - `powers::Vector{}`: A vector of power values corresponding to each segment of the path. (note that it is power to get to the point from previous point)
    - `cur_time::Int`: Current time in HHMM format (e.g., 1345 for 1:45 PM).
    - `cur_day::Int`: Current day of the month (1-31).
    - `cur_month::Int`: Current month (1-12).
    - `cur_year::Int`: Current year (e.g., 2025).

    # Returns
    - `times::Vector{Int}`: A vector of times in HHMM format for each point in the path.
    - `wind_info::Vector{Tuple{Float64, Float64}}`: A vector of tuples containing wind speed and direction for each point in the path.

    """

    times = [0]  # Initialize with zero for the first point
    wind_info = []
    twas = []
    boat_speed_data = []
    distances = []

    for i in 1:length(path)-1
        println("On segment $(i) of $(length(path)-1)")
        p1 = path[i]
        p2 = path[i+1]

        new_year, new_month, new_day, use_time = addSeconds(cur_time, cur_day, cur_month, cur_year, times[i])
        new_year, new_month, new_day, use_time = roundTime(use_time, new_day, new_month, new_year)
        word_month = monthWord(new_month)

        wind_file = "./OceanData/$(use_time)-$(new_day)$(word_month)$(new_year).grib"
        println("Processing segment $(i): From $(p1) to $(p2)")
        println("Wind file: $(wind_file)")

        if !isfile(wind_file)
            str_time = timeString(use_time)
            getWindData(wind_file, new_year, new_month, new_day, [str_time], [90, -180, -90, 180])
        end

        println(wind_file)
        file = GRIB.GribFile(wind_file)
        u_msg, v_msg = extractWindMessages(file, use_time)

        wind_speed, wind_dir = getWindAngleSpeedFromMessages(u_msg, v_msg, p1[1], p1[2])

        p = round(Int, powers[i])  # Ensure power is an integer

        # CHECK POWER MATCHES
        polar_file = "Polars/RisePol$(p)kW.csv"
        # Load polar data
        wind_speeds, wind_angles, boat_speeds = readPolar(polar_file)
        polar_info = (wind_speeds, wind_angles, boat_speeds)

        # Calculate travel time between p1 and p2
        t, twa, _, boat_speed = GetTravelTime(p1[1], p1[2], p2[1], p2[2],
                                   wind_speed, wind_dir, polar_info)
        t = round(Int, t)  # Ensure t is an integer

        dist = getDistance(p1[1], p1[2], p2[1], p2[2])
        push!(distances, dist)

        push!(twas, twa)
        push!(boat_speed_data, boat_speed)

        push!(times, times[i] + t)
        push!(wind_info, (wind_speed, wind_dir))
       
        destroy(file)  # Clean up the GRIB file to free resources
    end
    
    push!(wind_info, (0.0, 0.0))  # Add 0 values for wind at end of path
    push!(twas, 0.0)
    push!(boat_speed_data, 0.0)
    push!(distances, 0.0)
 
    return times, wind_info, boat_speed_data, twas, distances

end

function simulatedRouteChecker(path::Vector{Tuple{Float64, Float64}}, powers::Vector{}, cur_time::Int, cur_day::Int, cur_month::Int, cur_year::Int)
    """
    Simulates traveling along a given path, where wind conditions are updated hourly as you move along the route and are found based on a weighted sum.
    # Arguments
    - `path::Vector{Tuple{Float64, Float64}}`: A vector of tuples representing the path, where each tuple contains (longitude, latitude).
    - `powers::Vector{}`: A vector of power values corresponding to each segment of the path.
    - `cur_time::Int`: Current time in HHMM format (e.g., 1345 for 1:45 PM).
    - `cur_day::Int`: Current day of the month (1-31).
    - `cur_month::Int`: Current month (1-12).
    - `cur_year::Int`: Current year (e.g., 2025).

    # Returns
    - `times` - the time in seconds to reach each point along the path
    - `wind_info` - the wind speed and direction at each point along the path
    - `boat_speed_data` - the boat speed at each point along the path
    - `twas` - the true wind angle at each point along the path
    - `distances` - the distance traveled between each point along the path
    - `points` - the coordinates of each point along the path, when moving along hourly
    """

    # Initialize variables
    times = [] 
    wind_info = []
    twas = []
    boat_speed_data = []
    distances = []
    points = []
    wind_speed = 0.0
    wind_dir = 0.0
    polar_info = (Float64[], Float64[], Float64[])

    destination_reached = false
    current_point = path[1]
    next_point = path[2]

    time = 0
    current_power = powers[1]
    time_taken_in_hour = 0
    path_index = 1

    push!(points, current_point)
    push!(times, 0)
    push!(twas, 0.0)
    push!(boat_speed_data, 0.0)
    push!(distances, 0.0)

    a = 6371e3    # Earth's radius in meters
    f = 0.0        # assume a spherical Earth

    # Loop until destination is reached
    while destination_reached == false

        # Update wind data every hour
        if time_taken_in_hour == 0 
            
            new_year, new_month, new_day, use_time = addSeconds(cur_time, cur_day, cur_month, cur_year, time)
            new_year, new_month, new_day, use_time = roundTime(use_time, new_day, new_month, new_year)
            word_month = monthWord(new_month)

            wind_file = "./OceanData/$(use_time)-$(new_day)$(word_month)$(new_year).grib"
            println("Wind file: $(wind_file)")

            if !isfile(wind_file)
                str_time = timeString(use_time)
                getWindData(wind_file, new_year, new_month, new_day, [str_time], [90, -180, -90, 180])
            end

            println(wind_file)
            # Get weighted wind speed and direction at current point
            wind_speed, wind_dir = getWeightedWind(current_point[1], current_point[2], wind_file, use_time)

            polar_file = "Polars/RisePol$(current_power)kW.csv"

            wind_speeds, wind_angles, boat_speeds = readPolar(polar_file)
            polar_info = (wind_speeds, wind_angles, boat_speeds)
        end 

        # Calculate travel time between p1 and p2
        t, twa, boat_angle, boat_speed = GetTravelTime(current_point[1], current_point[2], next_point[1], next_point[2],
                                   wind_speed, wind_dir, polar_info)
        
        # if time to next point fits within the hour
        if t + time_taken_in_hour <= 3600
            time += round(Int, t)
            time_taken_in_hour += round(Int, t)
            push!(times, time)
            push!(points, next_point)
            push!(twas, twa)
            push!(boat_speed_data, boat_speed)
            push!(distances, t * boat_speed)
            push!(wind_info, (wind_speed, wind_dir))

            path_index += 1
            # check if destination reached
            if path_index >= length(path)
                destination_reached = true
            else
                # update current and next points
                current_point = next_point
                next_point = path[path_index+1]
                current_power = powers[path_index]
            end

        # if time to next point is greater than the hour
        else
            available_time = 3600 - time_taken_in_hour
            # determine the point reached in the available time
            dist = boat_speed * available_time
            distance, az, _ = Geodesics.inverse(deg2rad.((current_point[1], current_point[2], next_point[1], next_point[2]))..., a, f)
            lon_rad, lat_rad, _ = Geodesics.forward(deg2rad.((current_point[1], current_point[2]))..., az, dist, a, f)
            lat2 = normalize_latitude(rad2deg(lat_rad))
            lon2 = normalize_longitude(rad2deg(lon_rad))
            current_point = (lon2, lat2)

            # update data
            time += round(Int, available_time)
            time_taken_in_hour = 0
            push!(times, time)
            push!(points, current_point)
            push!(twas, twa)
            push!(boat_speed_data, boat_speed)
            push!(distances, dist)
            push!(wind_info, (wind_speed, wind_dir))
        end
        

    end
    push!(wind_info, (0.0, 0.0))

    return times, wind_info, boat_speed_data, twas, distances, points
end
