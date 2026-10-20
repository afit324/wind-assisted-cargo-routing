using Shapefile
using LibGEOS
using GeoInterface
using Geodesics

include("geo_helpers.jl")

function createGrid(start_location, end_location, longitude_step, latitude_step)
    """
    Creates a grid of points between two locations. The grid is a list of tuples containing 
    the latitude and longitude of the locations in order (long, lat).

    # Arguments:
    - `start_location`: A tuple containing the starting longitude and latitude (longitude, latitude).
    - `end_location`: A tuple containing the ending longitude and latitude (longitude, latitude).
    - `longitude_step`: The step size for longitude.
    - `latitude_step`: The step size for latitude.

    # Returns:
    - a list of tuples containing the longitude and latitude of the grid points in order (long, lat).
    """

    # Create a grid of points between startLocation and endLocation
    longitudes = start_location[1]:longitude_step:end_location[1]
    latitudes = start_location[2]:latitude_step:end_location[2]
    
    
    # Create a grid of points
    grid = [(lon, lat) for lat in latitudes for lon in longitudes]
    return grid
end

function loadOceanPolygons(filename::String)
    """

    Reads in a shapefile containing polygon areas of the ocean and returns a list of polygons.

    # Arguments:
    - `filename`: The path to the shapefile containing the ocean polygons.

    # Returns:
    - a list of polygons representing the ocean areas.

    """
    table = Shapefile.Table(filename)
    polygons = GeoInterface.convert(LibGEOS, Shapefile.shape(first(table)))
    return polygons
end

function isOcean(location::Tuple{Float64, Float64}, polygons)
    """

    Returns true if the given location is in the ocean, false otherwise.

    # Arguments:
    - `location`: A tuple containing the longitude and latitude of the location (longitude, latitude).
    - `polygons`: A list of polygons representing the ocean areas.

    # Returns:
    - true if the location is in the ocean, false otherwise.

    """
    # create a LibGEOS point from the location
    point = LibGEOS.readgeom(string("POINT(", location[1], " ", location[2], ")"))
    # check if the point is in the ocean
    return LibGEOS.contains(polygons, point)
end

function getOceanPointsMatrix(grid::Matrix{Tuple{Float64, Float64}}, polygons)
    """

    Returns a matrix of points that are in the ocean. Where each point is a tuple containing the latitude and longitude of the location (latitude, longitude).

    # Arguments:
    - `gird` : A matrix of tuples containing the longitude and latitude of the locations in order (long, lat).
    - `polygons`: A list of polygons representing the ocean areas.

    # Returns:
    - a matrix of points that are in the ocean. Where each point is a tuple containing the longitude and latitude of the location (longitude, latitude) and points not in ocean set to (NaN, NaN).

    """
    # Create a matrix to store the ocean points
    ocean_points = Matrix{Tuple{Float64, Float64}}(undef, size(grid))
    for i in 1:size(grid, 1), j in 1:size(grid, 2)
        if isOcean(grid[i, j], polygons)
            ocean_points[i, j] = grid[i, j]
        else
            ocean_points[i, j] = (NaN, NaN)  # or some other placeholder for non-ocean points
        end
    end
    return ocean_points
end

function getOceanPoints(grid, polygons)
    """

    Returns a list of lists of points that are in the ocean. Where each list in the list is a stage and each point is a tuple containing the latitude and longitude of the location (latitude, longitude).

    # Arguments:
    - `gird` : A list of lists of tuples containing the longitude and latitude of the locations in order (long, lat).
    - `polygons`: A list of polygons representing the ocean areas.

    # Returns:
    - a list of lists of points that are in the ocean. Where each list in the list is a stage and each point is a tuple containing the longitude and latitude of the location (longitude, latitude).

    """
    # Create a list to store the ocean points
    ocean_points = []
    # Iterate over each stage in the grid
    for stage in grid
        # Create a list to store the ocean points in the stage
        ocean_stage = []
        for point in stage
            # Check if the point is in the ocean
            if isOcean(point, polygons)
                # If the point is in the ocean, add it to the ocean stage
                push!(ocean_stage, point)
            end
        end
        # add the stage to the ocean points
        push!(ocean_points, ocean_stage)
    end
    return ocean_points
end

# Utility to find index of closest value
function closest_index(vec, val)
    idx = findmin(abs.(vec .- val))[2]
    return idx
end

function createExpandedGridMatrix(
    start_location::Tuple{Float64, Float64},
    end_location::Tuple{Float64, Float64},
    num_lon_points::Int,
    num_lat_points::Int,
    padding::Int
)
    """
    Creates an expanded grid of points between two locations with customizable padding.

    # Arguments:
    - `start_location`: (longitude, latitude) of original grid start.
    - `end_location`: (longitude, latitude) of original grid end.
    - `num_lon_points`: Number of longitude points between start and end location inclusive.
    - `num_lat_points`: Number of latitude points between start and end location inclusive.
    - `padding`: Number of extra points to add outside of the area of the start and end location.

    # Returns:
    - `grid`: Matrix of (longitude, latitude) tuples.
    - `start_index`: (row, col) index where original grid starts in expanded grid.
    - `end_index`: (row, col) index where original grid ends in expanded grid.
    """

    # Compute step sizes
    lat_step = (end_location[2] - start_location[2]) / (num_lat_points - 1)
    lon_step = (end_location[1] - start_location[1]) / (num_lon_points - 1)

    latitudes = range(
    start_location[2] - padding * lat_step,
    step = lat_step,
    length = num_lat_points + 2 * padding
    )
    longitudes = range(
        start_location[1] - padding * lon_step,
        step = lon_step,
        length = num_lon_points + 2 * padding
    )

    # Create grid
    grid = Matrix{Tuple{Float64, Float64}}(undef, length(latitudes), length(longitudes))
    for i in 1:length(latitudes), j in 1:length(longitudes)
        grid[i, j] = (normalize_longitude(longitudes[j]), normalize_latitude(latitudes[i]))
    end

    # Calculate start and end indices directly
    start_index = (1 + padding, 1 + padding)
    end_index = (padding + num_lat_points, padding + num_lon_points)

    return grid, start_index, end_index
end

function createExpandedGridMatrixDifferentPadding(
    start_location::Tuple{Float64, Float64},
    end_location::Tuple{Float64, Float64},
    num_lon_points::Int,
    num_lat_points::Int,
    padding_start_long::Int,
    padding_start_lat::Int,
    padding_end_long::Int,
    padding_end_lat::Int
)
    """
    Creates an expanded grid of points between two locations with customizable padding.

    # Arguments:
    - `start_location`: (longitude, latitude) of original grid start.
    - `end_location`: (longitude, latitude) of original grid end.
    - `num_lon_points`: Number of longitude points between start and end location inclusive.
    - `num_lat_points`: Number of latitude points between start and end location inclusive.
    - `padding_start_long`: Number of extra points to add before the start longitude.
    - `padding_start_lat`: Number of extra points to add before the start latitude.
    - `padding_end_long`: Number of extra points to add after the end longitude.
    - `padding_end_lat`: Number of extra points to add after the end latitude.

    # Returns:
    - `grid`: Matrix of (longitude, latitude) tuples.
    - `start_index`: (row, col) index where original grid starts in expanded grid.
    - `end_index`: (row, col) index where original grid ends in expanded grid.
    """

    # Compute step sizes
    lat_step = (end_location[2] - start_location[2]) / (num_lat_points - 1)
    lon_step = (end_location[1] - start_location[1]) / (num_lon_points - 1)

    latitudes = range(
    start_location[2] - padding_start_lat * lat_step,
    step = lat_step,
    length = num_lat_points + padding_start_lat + padding_end_lat
    )
    longitudes = range(
        start_location[1] - padding_start_long * lon_step,
        step = lon_step,
        length = num_lon_points + padding_start_long + padding_end_long
    )

    # Create grid
    grid = Matrix{Tuple{Float64, Float64}}(undef, length(latitudes), length(longitudes))
    for i in 1:length(latitudes), j in 1:length(longitudes)
        grid[i, j] = (normalize_longitude(longitudes[j]), normalize_latitude(latitudes[i]))
    end

    # Calculate start and end indices directly
    start_index = (1 + padding_start_lat, 1 + padding_start_long)
    end_index = (padding_start_lat + num_lat_points, padding_start_long + num_lon_points)

    return grid, start_index, end_index
end



function createGridMatrix(start_location::Tuple{Float64, Float64}, end_location::Tuple{Float64, Float64}, longitude_step::Int, latitude_step::Int)
    """ 
    Creates a grid of points between two locations. The grid is a matrix of tuples containing
    the latitude and longitude of the locations in order (long, lat).

    # Arguments:
    - `start_location`: A tuple containing the starting longitude and latitude (longitude, latitude).
    - `end_location`: A tuple containing the ending longitude and latitude (longitude, latitude).
    - `longitude_step`: The step size for longitude.
    - `latitude_step`: The step size for latitude.
    # Returns:
    - a matrix of tuples containing the longitude and latitude of the grid points in order (long, lat).
    """

    # Determine correct step directions
    lat_step = start_location[2] <= end_location[2] ? latitude_step : -latitude_step
    lon_step = start_location[1] <= end_location[1] ? longitude_step : -longitude_step

    # Generate coordinate vectors
    latitudes = collect(start_location[2]:lat_step:end_location[2])
    longitudes = collect(start_location[1]:lon_step:end_location[1])

    # Initialize matrix
    grid = Matrix{Tuple{Float64, Float64}}(undef, length(latitudes), length(longitudes))

    # Fill matrix so that grid[i, j] = (longitude, latitude)
    for i in 1:length(latitudes), j in 1:length(longitudes)
        grid[i, j] = (longitudes[j], latitudes[i])
    end

    return grid
end

# Function to create sailing grid along a great circle
function create_sailing_grid_gc(departure::Tuple{Float64, Float64},
                                destination::Tuple{Float64, Float64},
                                n_stages::Int,
                                n_points_in_stage::Int,
                                spacing_distance::Float64)
    """

    Creates a grid of points along a great circle route between two locations. Where the grid is a list of lists of tuples containing the longitude and latitude of the locations in order (long, lat).

    # Arguments:
    - `departure`: A tuple containing the starting longitude and latitude (longitude, latitude).
    - `destination`: A tuple containing the ending longitude and latitude (longitude, latitude).
    - `n_stages`: The number of stages in the grid.
    - `n_points_in_stage`: The number of points in each stage.
    - `spacing_distance`: The distance between points in the stage in meters.

    # Returns:
    - `grid` - a list of lists of points in the grid. Where each list in the list is a stage and each point is a tuple containing the longitude anf latitude of the location (long, lat).

    """
    

    # Unpack points
    lon1, lat1 = departure
    lon2, lat2 = destination

    # Compute the geodesic line
    a, f = Geodesics.EARTH_R_MAJOR_WGS84, Geodesics.F_WGS84;
    distance, az, _ = Geodesics.inverse(deg2rad.((lon1, lat1, lon2, lat2))..., a, f)

    # Distances along the geodesic for the columns
    distances = LinRange(0, distance, n_stages)

    # Storage for grid points
    grid = []

    for (i, s) in enumerate(distances)
        
        # First stage
        if i == 1
            push!(grid, [departure])
            continue
        end

        # Last stage
        if i == n_stages
            push!(grid, [destination])
            continue
        end

        # For intermediate stages
        lon_rad, lat_rad, az_now = Geodesics.forward(deg2rad.((lon1, lat1))..., az, s, a, f)

        # Make vertical spread
        offset_range = collect(-div(n_points_in_stage-1, 2):div(n_points_in_stage-1, 2))
        stage = []

        for offset in offset_range
            # Calculate the distance for the offset
            dist = offset * spacing_distance
            # Calculate the new point in the stage
            lon2_rad, lat2_rad, _ = Geodesics.forward(lon_rad, lat_rad, az_now + (π/2), dist, a, f)
            lat2 = normalize_latitude(rad2deg(lat2_rad))
            lon2 = normalize_longitude(rad2deg(lon2_rad))
            push!(stage, (lon2, lat2))
        end
        # Add the stage to the grid
        push!(grid, stage)
    end

    return grid
end

function create_grid_from_path(path, n_points_in_stage::Int, spacing_distance::Float64)
    """

    Creates a grid of based on an inputted path, where stages are located at points in path and halfway between points in path. Where the grid is a list of lists of points. Where each list in the list is a stage and each point is a long, lat location in order (long, lat).

    # Arguments:
    - `path`: A list of tuples containing the longitude and latitude of the locations in order (long, lat).
    - `n_points_in_stage`: The number of points in each stage.
    - `spacing_distance`: The distance between points in the stage.

    # Returns:
    - a list of lists of points in the grid. Where each list in the list is a stage and each point is a tuple containing the longitude and latitude of the location (long, lat).

    """

    # initlise the grid and add the first point
    grid = []
    push!(grid, [path[1]])

    # loop through the path
    for i in 2:length(path)
        # get the last point and current point in the path
        last_lon, last_lat = path[i-1]
        lon, lat = path[i]
        lon_rad, lat_rad = deg2rad.((lon, lat))

        # Compute the distance between the two points
        a, f = Geodesics.EARTH_R_MAJOR_WGS84, Geodesics.F_WGS84;
        distance, az_now, _ = Geodesics.inverse(deg2rad.((last_lon, last_lat))..., lon_rad, lat_rad, a, f)

        # create grid points halfway between the two points
        lon_half_rad, lat_half_rad, _ = Geodesics.forward(deg2rad.((last_lon, last_lat))..., az_now, distance/2, a, f)

        # create the offset range for the points in the stage
        offset_range = collect(-div(n_points_in_stage-1, 2):div(n_points_in_stage-1, 2))

        # initlise the stage
        stage_half = []
        stage_whole = []

        for offset in offset_range

            # calculate the distance for the offset
            dist = offset * spacing_distance

            # calculate the new point halfway between the two points
            lon_half, lat_half, _ = Geodesics.forward(lon_half_rad, lat_half_rad, az_now+(pi/2), dist, a, f)
            lat_half = normalize_latitude(rad2deg(lat_half))
            lon_half = normalize_longitude(rad2deg(lon_half))
            # add the point to the stage
            push!(stage_half, (lon_half, lat_half))

            # calculate the new point at the current point
            lon, lat, _ = Geodesics.forward(lon_rad, lat_rad, az_now+(pi/2), dist, a, f)
            lat = normalize_latitude(rad2deg(lat))
            lon = normalize_longitude(rad2deg(lon))
            # add the point to the stage
            push!(stage_whole, (lon, lat))

        end
        # add the halfway point to the stage
        push!(grid, stage_half)
        # add the current point to the stage but if it is the last point in the path, only add the last point
        if i != length(path)
            push!(grid, stage_whole)
        else
            push!(grid, [path[i]])
        end
    end
    return grid
end


function reduceGrid(grid, path, path_padding)
    """
    Reduces a grid of points to only include points that are within a certain distance of a given path.

    # Arguments:
    - `grid`: A matrix of tuples containing the longitude and latitude of the locations in order (long, lat).
    - `path`: A list of tuples containing the longitude and latitude of the path points in order (long, lat).
    - `path_padding`: The maximum distance from the path that a point in the grid can be to be included in the reduced grid in meters
    """

    for i in axes(grid, 1), j in axes(grid, 2)
        lon, lat = grid[i, j]
        # compute minimum distance to any path point
        mindist = minimum(getDistance(lon, lat, lonp, latp) for (lonp, latp) in path)
        if mindist <= path_padding
            grid[i, j] = grid[i, j]       # keep the point
        else
            grid[i, j] = (NaN, NaN)         # too far → set to missing
        end
    end
    return grid
end
