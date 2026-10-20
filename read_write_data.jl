using CSV
using DataFrames
using Plots
using Base
using DataStructures 
using Plots
using Shapefile
using LibGEOS
using GeoInterface
using GRIB
using CDSAPI
using CairoMakie
using DotEnv
using GeoJSON
using JSON
using GRIB

include("geo_helpers.jl")

function readPolar(fileName::String)
    """
    Reads a polar performance diagram from a CSV file and returns structured data for boat performance modeling.

    Arguments:
    - `fileName::String`: Path to the CSV file containing the polar data.

    Returns:
    - `wind_speeds::Vector{Float64}`: A vector of wind speeds (m/s), extracted from the first row (excluding the first cell).
    - `true_wind_angles::Vector{Float64}`: A vector of true wind angles (degrees), extracted from the first column (excluding the first row).
    - `boat_speeds::Matrix{Float64}`: A matrix of boat speeds (m/s), where each row corresponds to a wind angle and each column to a wind speed.

    Assumptions:
    - The CSV has no header.
    - The first row (after the first cell) contains wind speeds.
    - The first column (after the first row) contains wind angles.
    - The rest of the matrix contains boat speed values
    - The first cell (top-left) is ignored as it does not contain relevant data.

    """
    df_raw = CSV.read(fileName, DataFrame, header=false)

    # Extract wind speeds (row 1, skipping first cell)
    wind_speeds = Vector{Float64}(df_raw[1, 2:end])

    # Extract true wind angles (column 1, skipping first row)
    true_wind_angles = parse.(Float64, Vector(df_raw[2:end, 1]))

    # Extract boat speeds (as matrix)
    boat_speeds = df_raw[2:end, 2:end]
    boat_speeds = Matrix{Float64}(boat_speeds)

    return wind_speeds, true_wind_angles, boat_speeds
end


function readGeoJSON(filename::String)
    """
    Reads a GeoJSON file and returns the coordinates of the features.

    Arguments:
    - `filename::String`: Path to the GeoJSON file.

    Returns:
    - `coordinates::Vector{Tuple{Float32, Float32}}`: A vector of tuples containing the coordinates (longitude, latitude) of each feature in the GeoJSON file.
    """
    geojson = GeoJSON.read(filename)
    coordinates = []

    for feature in geojson.features
        if feature.geometry isa GeoJSON.Point
            push!(coordinates, feature.geometry.coordinates)
        elseif feature.geometry isa GeoJSON.LineString
            for coord in feature.geometry.coordinates
                push!(coordinates, coord)
            end
        end
    end

    return coordinates
end


function writeToGeoJSON(points, filename::String)
    """

    Writes a list of points to a GeoJSON file. The points are in the format (long, lat).

    # Arguments:
    - `points`: A list of tuples containing the longitude and latitude of the points in order (long, lat).
    - `filename`: The name of the output GeoJSON file.

    """
    # Create GeoJSON features
    features = [
        GeoJSON.Feature(geometry = GeoJSON.Point(coordinates = (Float32(lon), Float32(lat))), properties = Dict{Symbol, Any}()) for (lon, lat) in points
    ]
    # adds the features to a GeoJSON feature collection
    fc = GeoJSON.FeatureCollection(features=features)
    # write the feature collection to a GeoJSON file
    GeoJSON.write(filename, fc)

end

function writeToGeoJSONIndex(points, filename::String, start_index::Int, end_index::Int)
    """

    Writes a list of points to a GeoJSON file. The points are in the format (long, lat).

    # Arguments:
    - `points`: A list of tuples containing the longitude and latitude of the points in order (long, lat).
    - `filename`: The name of the output GeoJSON file.
    - `start_index`: The index of the start point in the points list.
    - `end_index`: The index of the end point in the points list.

    """
    # Create GeoJSON features
    features = [
        GeoJSON.Feature(geometry = GeoJSON.Point(coordinates = (Float32(lon), Float32(lat))), properties = Dict{Symbol, Any}(
            :start => start_index == start_index ? true : false,
            :end => end_index == end_index ? true : false,
        )) for (lon, lat) in points
    ]
    # adds the features to a GeoJSON feature collection
    fc = GeoJSON.FeatureCollection(features=features)
    # write the feature collection to a GeoJSON file
    GeoJSON.write(filename, fc)

end

function writeToGeoJsonPowerProperty(points, filename::String, powers)
    """

    Writes a list of points to a GeoJSON file with power as a property. The points are in the format (long, lat).

    # Arguments:
    - `points`: A list of tuples containing the longitude and latitude of the points in order (long, lat).
    - `filename`: The name of the output GeoJSON file.
    - `powers`: a list of powers that is the same length as points for the power at each point

    """
    # Create GeoJSON features with power property
    features = [
        GeoJSON.Feature(
            geometry = GeoJSON.Point(coordinates = (Float32(lon), Float32(lat))),
            properties = Dict{Symbol, Any}(
                :power => powers[i]
            )
        ) for (i, (lon, lat)) in enumerate(points)
    ]
    # flip the coordinates to (lon, lat) for GeoJSON and convert to Float32
    path = [(Float32(lon), Float32(lat)) for (lon, lat) in points]
    # Create GeoJSON features by making a LineString from the path
    push!(features, GeoJSON.Feature(geometry = GeoJSON.LineString(coordinates = path), properties = Dict{Symbol, Any}(
                :power => "path"
            )))
    
    # adds the features to a GeoJSON feature collection
    fc = GeoJSON.FeatureCollection(features=features)
    # write the feature collection to a GeoJSON file
    GeoJSON.write(filename, fc)
end

function writeToGeoJSONWindProperties(points, filename::String, wind_speeds, wind_directions)
    """

    Writes a list of points to a GeoJSON file with wind speed and direction properties. The points are in the format (long, lat).

    # Arguments:
    - `points`: A list of tuples containing the longitude and latitude of the points in order (long, lat).
    - `filename`: The name of the output GeoJSON file.
    - `wind_speeds`: A list of wind speeds corresponding to each point.
    - `wind_directions`: A list of wind directions corresponding to each point.

    """
    # converts wind speeds and directions to u and v components
    u_components = [Float32(speed * sin(deg2rad(direction-180))) for (speed, direction) in zip(wind_speeds, wind_directions)]
    v_components = [Float32(speed * cos(deg2rad(direction-180))) for (speed, direction) in zip(wind_speeds, wind_directions)]
    # Create GeoJSON features with wind properties
    features = [
        GeoJSON.Feature(
            geometry = GeoJSON.Point(coordinates = (Float32(lon), Float32(lat))),
            properties = Dict{Symbol, Any}(
                :wind_speed => wind_speeds[i],
                :wind_direction => wind_directions[i],
                :u_component => u_components[i],
                :v_component => v_components[i]
            )
        ) for (i, (lon, lat)) in enumerate(points)
    ]
    # adds the features to a GeoJSON feature collection
    fc = GeoJSON.FeatureCollection(features=features)
    # write the feature collection to a GeoJSON file
    GeoJSON.write(filename, fc)
end

function writeToGeoJSONLineString(path, filename::String)
    """

    Writes a list of points to a GeoJSON file as a LineString. The points are in the format (long, lat).

    # Arguments:
    - `path`: A list of tuples containing the longitude and latitude of the points in order (long, lat) that form a line.
    - `filename`: The name of the output GeoJSON file.

    """
    # flip the coordinates to (lon, lat) for GeoJSON and convert to Float32
    path = [(Float32(lon), Float32(lat)) for (lon, lat) in path]
    # Create GeoJSON features by making a LineString from the path
    features = [
        GeoJSON.Feature(geometry = GeoJSON.LineString(coordinates = path), properties = Dict{Symbol, Any}())
    ]
    # adds the features to a GeoJSON feature collection
    fc = GeoJSON.FeatureCollection(features=features)
    # write the feature collection to a GeoJSON file
    GeoJSON.write(filename, fc)
end

function seeGribData(filename, key)
    # allows for the user to see the data in the grib file
    GRIB.GribFile(filename) do f
        for msg in f
            println("key:" , GRIB.getindex(msg, key))
        end
    end
end

function getWindData(output_name::String, year::Int, month::Int, day::Int, times::Vector{String}, area)
    """

    Gets wind data from the Copernicus Climate Data Store

    # Arguments:
    - `output_name`: The name of the output file to save the data.
    - `year`: The year of the data (e.g. 2023).
    - `month`: The month of the data (e.g. 10).
    - `day`: The day of the data (e.g. 01).
    - `times`: The times of the data (e.g. ["00:00", "12:00"]).
    - `area`: The area of the data (e.g. [90, -180, -90, 180]).

    # Notes:
    You must have a .env file with the following variables as described in the ReadMe:
    - CDSAPI_URL: The URL of the Copernicus Climate Data Store API.
    - CDSAPI_KEY: The API key for the Copernicus Climate Data Store.

    """

    # Load environment variables from .env file
    DotEnv.load!()

    # sets up the request
    dataset = "reanalysis-era5-single-levels"
    request = """{
        "product_type": ["reanalysis"],
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind"
        ],
        "year": ["$year"],
        "month": ["$month"],
        "day": ["$day"],
        "time": $times,
        "data_format": "grib",
        "download_format": "unarchived",
        "area": $area
    }"""
    print("Request: ", request)

    # retrieve the data and save it to the output file
    CDSAPI.retrieve(dataset, request, output_name)
end


function getDataAtLongLatTime(filename::String, name::String, long, lat, time)
    """

    Gets the data at a given location and time from a GRIB file.

    # Arguments:
    - `filename`: The name of the GRIB file.
    - `name`: The name of the data to retrieve (e.g. "10 metre U wind component").
    - `long`: The longitude of the location.
    - `lat`: The latitude of the location.
    - `time`: The time of the data (e.g. 2100, 0100).
    

    # Returns:
    - `closest_value` - the value of the data at the closest location and time.

    """
    file = GRIB.GribFile(filename)

    for message in file
        if message["name"] == name && message["time"] == time
            longs, lats, values = data(message)

            # Combine longs and lats into tuples
            longlat = zip(longs, lats)

            # Compute distances
            distances = [getDistance(long, lat, lo, la) for (lo, la) in longlat]
            closest_index = argmin(distances)

            closest_lon = longs[closest_index]
            closest_lat = lats[closest_index]
            closest_value = values[closest_index]
            distance = getDistance(long, lat, closest_lon, closest_lat)

            destroy(file)

            println("Closest point to ($long, $lat) is ($closest_lon, $closest_lat) with value $closest_value at distance $distance meters")
            return closest_value
        end
    end
    

    destroy(file)

    return nothing
end

function getDataAtLongLatTimeMac(filename::String, name::String, long, lat, time)
    """
    Gets the data at a given location and time from a GRIB file on Mac.
    # Arguments:
    - `filename`: The name of the GRIB file.
    - `name`: The name of the data to retrieve (e.g. "10 metre U wind component").
    - `long`: The longitude of the location.
    - `lat`: The latitude of the location.
    - `time`: The time of the data (e.g. 2100, 0100).       
    # Returns:
    - `closest_value` - the value of the data at the closest location and time.
    # Notes:
    This function only works on Mac due to the way the GRIB library is set up.
    """
    index = Index(filename, "name")
    select!(index, "name", name)
    for message in index
        if message["time"] != time
            continue
        end
        longs, lats, values = data(message)

        # make long and lat into a tuple
        longlat = zip(longs, lats)

        # Find the closest point to the given longitude and latitude using get distance function
        distances = [getDistance(long, lat, longs, lats) for (longs, lats) in longlat]
        closest_index = argmin(distances)
        closest_lon = longs[closest_index]
        closest_lat = lats[closest_index]
        closest_value = values[closest_index]
        distance = getDistance(long, lat, closest_lon, closest_lat)
        println("Closest point to ($long, $lat) is ($closest_lon, $closest_lat) with value $closest_value at distance $distance meters")
        # Print the values at the closest point
        println("Values at ($closest_lon, $closest_lat):", closest_value)
        return closest_value
    end
end

function getWindAngleSpeedAtLocation(filename::String, long, lat, time)
    """

    Gets the wind speed and angle at a given location and time from a GRIB file.

    # Arguments:
    - `filename`: The name of the GRIB file.
    - `long`: The longitude of the location.
    - `lat`: The latitude of the location.
    - `time`: The time of the data (e.g. 2100, 0100).

    # Returns:
    - `speed`: The wind speed at the location in m/s.
    - `wind_dir`: The wind direction at the location in degrees (clockwise from the north).

    """
    # get the U and V components of the wind
    u = getDataAtLongLatTime(filename, "10 metre U wind component", long, lat, time)
    v = getDataAtLongLatTime(filename, "10 metre V wind component", long, lat, time)
    # calculate the wind speed
    speed = sqrt(u^2 + v^2)
    # calculate the wind angle
    θ_rad = atan(u, v)
    wind_dir = mod(rad2deg(θ_rad), 360)

    return speed, wind_dir
end


function getWindAngleSpeedAtLocations(filename::String, locations, time)
    """

    Gets the wind speed and angle at multiple locations and a given time from a GRIB file.

    # Arguments:
    - `filename`: The name of the GRIB file.
    - `locations`: A list of tuples containing the latitude and longitude of the locations in order (long, lat).
    - `time`: The time of the data (e.g. 2100, 0100).

    # Returns:
    - `speed`: The wind speed at the location in m/s.
    - `wind_dir`: The wind direction at the location in degrees (clockwise from the north).

    """
    speed = []
    angle = []
    # get the wind speed and angle at each location
    for (long, lat) in locations
        s, a = getWindAngleSpeedAtLocation(filename, long, lat, time)
        push!(speed, s)
        push!(angle, a)
    end
    return [speed, angle]
end

function extractWindMessages(file::GRIB.GribFile, time)
    """
    Extracts wind messages for a specific time from a GRIB file.
    # Arguments:
        - `file` (GRIB.GribFile): The GRIB file to extract messages from.
        - `time` (Int): The time for which to extract wind messages (e.g., 2100, 0100).
    # Returns:
        - `u_msg` (GRIB.GribMessage): The message containing the U wind component.
        - `v_msg` (GRIB.GribMessage): The message containing the V wind component.
    """
    u_msg = nothing
    v_msg = nothing

    for msg in file
        if msg["time"] == time
            if msg["name"] == "10 metre U wind component"
                u_msg = msg
            elseif msg["name"] == "10 metre V wind component"
                v_msg = msg
            end
        end
    end
    return u_msg, v_msg
end

function getWindAngleSpeedFromMessages(u_msg, v_msg, long, lat)
    """
    Gets the wind speed and angle at a specific location from U and V wind component messages.
    # Arguments:
        - `u_msg` (GRIB.GribMessage): The message containing the U wind component.
        - `v_msg` (GRIB.GribMessage): The message containing the V wind component.
        - `long` (Float64): The longitude of the location.
        - `lat` (Float64): The latitude of the location.
    # Returns:
        - `speed` (Float32): The wind speed at the location in m/s.
        - `wind_dir` (Float32): The wind direction at the location in degrees (direction wind coming from).
    """
    u_longs, u_lats, u_vals = data(u_msg)
    v_longs, v_lats, v_vals = data(v_msg)

    # Assume same grid, so find nearest once
    longlat = zip(u_longs, u_lats)
    closest_index = argmin([(getDistance(long, lat, lo, la)) for (lo, la) in longlat])

    u = u_vals[closest_index]
    v = v_vals[closest_index]

    speed = sqrt(u^2 + v^2)
    θ_rad = atan(u, v) 
    wind_dir = mod(rad2deg(θ_rad)+180, 360)

    return speed, wind_dir
end

function extractWindValues(file::GRIB.GribFile, time)
    """
    Extracts U and V wind component values from a GRIB file for a specific time.
    # Arguments:
        - `file` (GRIB.GribFile): The GRIB file to extract values
        - `time` (Int): The time for which to extract wind values (e.g., 2100, 0100).
    # Returns:
        - `u_vals`: The U wind component values.
        - `v_vals`: The V wind component values.
    """
    u_msg, v_msg = extractWindMessages(file, time)
    _, _, u_vals = data(u_msg)
    _, _, v_vals = data(v_msg)
    return u_vals, v_vals
end

function getWindAngleSpeedFromVals(u_vals, v_vals, long, lat)
    """ 
    Gets the wind speed and angle at a specific location from U and V wind component values. Uses the known grid to find the index.
    # Arguments:
        - `u_vals`: The U wind component values.
        - `v_vals`: The V wind component values.
        - `long` (Float64): The longitude of the location.
        - `lat` (Float64): The latitude of the location.
    # Returns:
        - `speed` (Float32): The wind speed at the location in m/s.
        - `wind_dir` (Float32): The wind direction at the location in degrees (direction wind coming from).
    # Notes:
        - Assumes a regular grid with:
            - Longitude range: -180 to 180
            - Latitude range: -90 to 90
            - Grid spacing: 0.25 degrees
    """
    # Define grid parameters
    min_long = -180
    max_long = 180
    min_lat = -90
    max_lat = 90
    change_long = 0.25
    change_lat = 0.25
    # Calculate index
    i = round(Int, (long - min_long) / change_long) + 1
    j = round(Int, (max_lat - lat) / change_lat) + 1
    index = CartesianIndex(i, j)

    # get the u and v values at the index
    u = u_vals[index]
    v = v_vals[index]

    # calculate speed and direction
    speed = sqrt(u^2 + v^2)
    θ_rad = atan(u, v) 
    wind_dir = mod(rad2deg(θ_rad)+180, 360)

    return speed, wind_dir  
end


function writeRouteDataToCsv(file_name::String, path_data::Vector{Tuple{Float64, Float64}}, cost_data::Vector{Float64}, wind_data::Vector{Tuple{Float64, Float64}}, boat_speed_data::Vector{Float64}, twas_data::Vector{Float64})
    """
    Writes route data to a CSV file.

    # Arguments:
    - `file_name::String`: The name of the output CSV file.
    - `path_data::Vector{Tuple{Float64, Float64}}`: A list of tuples containing the longitude and latitude of the points in the path.
    - `cost_data::Vector{Float64}`: A list of times in seconds associated with each point in the path.
    - `wind_data::Vector{Tuple{Float64, Float64}}`: A list of tuples containing the wind speed (m/s) and direction (degrees from north, wind is coming from)at each point in the path.
    - `boat_speed_data::Vector{Float64}`: A list of boat speeds (m/s) at each point in the path.
    - `twas_data::Vector{Float64}`: A list of true wind angles (degrees) at each point in the path.

    # Returns:
    - Writes the data to the CSV file.
    """

    df = DataFrame()

    df[!,:"longitude"] = [point[1] for point in path_data]
    df[!,:"latitude"] = [point[2] for point in path_data]
    df[!,:"wind_speed"] = [wind[1] for wind in wind_data]
    df[!,:"wind_direction"] = [wind[2] for wind in wind_data]
    df[!,:"boat_speed"] = [speed for speed in boat_speed_data]
    df[!,:"twa"] = [twa for twa in twas_data]
    df[!,:"time"] = [cost for cost in cost_data]
    df[!,:"days"] = [cost/86400 for cost in cost_data]

    # Write to CSV
    CSV.write(file_name, df)

end


function writeRouteDataPowerToCsv(file_name::String, path_data::Vector{Tuple{Float64, Float64}}, cost_data::Vector{Float64}, energy_data::Vector{} ,power_data::Vector{})
    """
    Writes route data to a CSV file.

    # Arguments:
    - `file_name::String`: The name of the output CSV file.
    - `path_data::Vector{Tuple{Float64, Float64}}`: A list of tuples containing the longitude and latitude of the points in the path.
    - `cost_data::Vector{Float64}`: A list of times in seconds associated with each point in the path.
    - `energy_data::Vector{}`: A list of energy levels (kWh) at each point in the path.
    - `power_data::Vector{Float64}`: A list of power levels (kW) at each point in the path.

    # Returns:
    - Writes the data to the CSV file.
    """

    df = DataFrame()

    df[!,:"longitude"] = [point[1] for point in path_data]
    df[!,:"latitude"] = [point[2] for point in path_data]
    df[!,:"time"] = [cost for cost in cost_data]
    df[!,:"days"] = [cost/86400 for cost in cost_data]
    df[!,:"energy_kWh"] = [energy for energy in energy_data]
    df[!,:"power_kW"] = [power for power in power_data]

    # Write to CSV
    CSV.write(file_name, df)

end


function readRouteDataFromCsv(file_name::String)
    """
    Reads route data from a CSV file.

    # Arguments:
    - `file_name::String`: The name of the input CSV file.

    # Returns:
    - 'path_data::Vector{Tuple{Float64, Float64}}': A list of tuples containing the longitude and latitude of the points in the path.
    - 'wind_data::Vector{Tuple{Float64, Float64}}': A list of tuples containing the wind speed (m/s) and direction (degrees from north, wind is coming from) at each point in the path.
    - 'boat_speed_data::Vector{Float64}': A list of boat speeds (m/s) at each point in the path.
    - 'twas_data::Vector{Float64}': A list of true wind angles (degrees) at each point in the path.
    - 'time_data::Vector{Float64}': A list of times in seconds associated with each point in the path.
    
    """

    df = CSV.read(file_name, DataFrame)

    # Create list of (longitude, latitude)
    path_data = [(row.longitude, row.latitude) for row in eachrow(df)]

    # Create list of (wind_speed, wind_direction)
    wind_data = [(row.wind_speed, row.wind_direction) for row in eachrow(df)]

    # Create list of boat speeds
    boat_speed_data = [row.boat_speed for row in eachrow(df)]

    # Create list of true wind angles
    twas_data = [row.twa for row in eachrow(df)]

    time_data = [row.time for row in eachrow(df)]

    return path_data, wind_data, boat_speed_data, twas_data, time_data
end
