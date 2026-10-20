using Shapefile

function compute_bbox(points::Vector{Shapefile.Point})
    """

    Computes the bounding box of a list of points.

    # Arguments:
    - `points::Vector{Shapefile.Point}`: A vector of points.

    # Returns:
    - a shapefile rectangle representing the bounding box of the points.

    """
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    return Shapefile.Rect(minimum(xs), minimum(ys), maximum(xs), maximum(ys))
end

function normalize_longitude(lon::Float64)
    """
    Normalizes a longitude value to be within the range of -180 to 180 degrees.
    # Arguments:
    - `lon::Float64`: The longitude value to normalize.
    # Returns:
    - `lon::Float64`: The normalized longitude value.
    """
    lon = mod(lon + 180, 360) - 180
    return lon
end

function normalize_latitude(lat::Float64)
    """
    Normalizes a latitude value to be within the range of -90 to 90 degrees.
    # Arguments:
    - `lat::Float64`: The latitude value to normalize.
    # Returns:
    - `lat::Float64`: The normalized latitude value.
    """
    if lat > 90
        lat = 180 - lat
    elseif lat < -90
        lat = -180 - lat
    end
    return lat
end

function getDistance(longitude1, latitude1, longitude2, latitude2)
    """

    Calculates the great-circle distance between two geographic points on Earth using the Haversine formula.

    Arguments:
    - `longitude1::Float64`: Longitude of the first point.
    - `latitude1::Float64`: Latitude of the first point.
    - `longitude2::Float64`: Longitude of the second point.
    - `latitude2::Float64`: Latitude of the second point.

    Returns:
    - `distance::Float64`: The distance between the two points in meters.

    """

    # Haversine formula to calculate distance between two points on the Earth
    R = 6371e3  # Earth's radius in meters
    lat1 = deg2rad(latitude1)
    lat2 = deg2rad(latitude2)
    del_lat = deg2rad(latitude2 - latitude1)
    del_long= deg2rad(longitude2 - longitude1)

    a = sin(del_lat/2)^2 + cos(lat1) * cos(lat2) * sin(del_long/2)^2
    c = 2 * atan(sqrt(clamp(a, 0.0, 1.0)), sqrt(1-clamp(a, 0.0, 1.0)))

    distance = R * c  # Distance in meters
    return distance
end

function matrixToListOfLists(mat::AbstractMatrix{<:Tuple{<:Real, <:Real}})
    """
    Converts a matrix of tuples into a list of lists (rows),
    skipping any tuples that contain NaN values.

    # Arguments:
    - `mat`: A 2D matrix of tuples (e.g., Tuple{Float64, Float64}).

    # Returns:
    - A list of lists, where each inner list contains valid (non-NaN) tuples from that row.
    """
    return [[val for val in mat[i, :] if !(any(isnan, val))] for i in 1:size(mat, 1)]
end
