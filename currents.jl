using NCDatasets
using Plots          # generate simple plots
using Dates 

include("geo_helpers.jl")  

function readNetCDF(filename::String)
    """    
    Reads a NetCDF file and returns the dataset.

    # Arguments:
        - `filename`` (String): Path to the NetCDF file.
    # Returns:
        - `Dataset``: The opened NetCDF dataset.
    """
    # Open the NetCDF file
    ds = Dataset(filename)
    return ds
end

function getData(dataset, var_name::String)
    """   
    Extracts data for a specific variable from the NetCDF dataset.
    Args:
        - `dataset`` (Dataset): The opened NetCDF dataset.
        - `var_name`` (String): The name of the variable to extract.
    Returns:
        - `Array`: The data for the specified variable.
    """
    # Extract data for a specific variable
    data = dataset[var_name][:]
    return data
end

function getLonLat(dataset)
    """ 
    Extracts longitude and latitude data from the NetCDF dataset.
    
    # Arguments:
        - `dataset` (Dataset): The opened NetCDF dataset.
    # Returns:
        - `lon` (Array): The longitude data.
        - `lat` (Array): The latitude data.
    """

    # Extract longitude and latitude data
    lon = dataset["longitude"][:]
    lat = dataset["latitude"][:]
    return lon, lat
end


function getUoVo(dataset)
    """
    Extracts uo and vo components from the NetCDF dataset.
    # Arguments:
        - `dataset` (Dataset): The opened NetCDF dataset.
    # Returns:
        - `uo` (Array): The uo component data.
        - `vo` (Array): The vo component data.
    """
    # Extract uo and vo data
    uo = dataset["uo"][:, :, 1, 1]
    vo = dataset["vo"][:, :, 1, 1]
    return uo, vo
end

function getUoVoAtLocation(dataset, location::Tuple{Float64, Float64})
    """
    Finds the closest uo and vo components to a specific geographic location in the dataset.
    # Arguments:
        - `dataset` (Dataset): The opened NetCDF dataset.
        - `location` (Tuple): A tuple containing the longitude and latitude of the location (lon, lat).
    # Returns:
        - `uo` (Float32): The uo component at the closest point.
        - `vo` (Float32): The vo component at the closest point.
    """

    lon, lat = getLonLat(dataset)
    uo, vo = getUoVo(dataset)

    min_dist = Inf
    closest_i = 0
    closest_j = 0

    for i in eachindex(lon)
        for j in eachindex(lat)
            # Skip if data is missing
            if ismissing(uo[i, j]) || ismissing(vo[i, j])
                continue
            end

            # Compute distance
            dist = getDistance(location[1], location[2], lon[i], lat[j])
            if dist < min_dist
                min_dist = dist
                closest_i = i
                closest_j = j
            end
        end
    end

    if closest_i == 0 || closest_j == 0
        error("No valid uo/vo data found near location: $location")
    end

    closest_lon = lon[closest_i]
    closest_lat = lat[closest_j]
    closest_uo = uo[closest_i, closest_j]
    closest_vo = vo[closest_i, closest_j]

    println("Closest point to $location is ($closest_lon, $closest_lat) with uo $closest_uo and vo $closest_vo at distance $min_dist meters")
    return closest_uo, closest_vo
end

function getCurrentSpeedDirectionAtLocation(dataset, location::Tuple{Float64, Float64})
    """
    Calculates the current speed and direction at a specific geographic location in the dataset.
    # Arguments:
        - `dataset` (Dataset): The opened NetCDF dataset.
        - `location` (Tuple): A tuple containing the longitude and latitude of the location (lon, lat).
    # Returns:
        - `speed` (Float32): The current speed at the closest point.
        - `direction` (Float32): The current direction (direction heading to) at the closest point in degrees.
    """
    uo, vo = getUoVoAtLocation(dataset, location)
    speed = sqrt(clamp(uo^2 + vo^2, 0.0, Inf))
    # calculate the wind angle
    θ_rad = atan(uo, vo)
    direction = mod(rad2deg(θ_rad), 360)
    return speed, direction
end