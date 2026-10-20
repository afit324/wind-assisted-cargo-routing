
using LinearAlgebra


function to_cartesian(lon::Float64, lat::Float64)
    """
    Convert spherical coordinates (longitude, latitude) to Cartesian coordinates (x, y, z).

    # Arguments:
    - `lon::Float64`: Longitude in radians.
    - `lat::Float64`: Latitude in radians.

    # Returns:
    - `cartesian_coords`: A vector of Cartesian coordinates [x, y, z].
    """
    x = cos(lat) * cos(lon)
    y = cos(lat) * sin(lon)
    z = sin(lat)
    return [x, y, z]
end


function great_circle_path(start::Tuple{Float64, Float64}, stop::Tuple{Float64, Float64}, n::Int)
    """
    Generate a great circle path between two points on the Earth.

    # Arguments:
    - `start::Tuple{Float64, Float64}`: A tuple representing the starting point (longitude, latitude).
    - `stop::Tuple{Float64, Float64}`: A tuple representing the ending point (longitude, latitude).
    - `n::Int`: The number of points to generate along the great circle path.

    # Returns:
    - `points`: A vector of tuples representing the points along the great circle path, each in the form (longitude, latitude).
    
    """

    # Convert degrees to radians
    lon1, lat1 = deg2rad(start[1]), deg2rad(start[2])
    lon2, lat2 = deg2rad(stop[1]), deg2rad(stop[2])

    p1 = to_cartesian(lon1, lat1)
    p2 = to_cartesian(lon2, lat2)

    # Angle between points
    ω = acos(clamp(dot(p1, p2), -1.0, 1.0))

    # If the points are the same, return copies of start
    if isapprox(ω, 0.0)
        return fill(start, n)
    end

    points = Vector{Tuple{Float64, Float64}}(undef, n)

    for i in 1:n
        t = (i - 1) / (n - 1)
        # Slerp formula
        part1 = sin((1 - t) * ω) / sin(ω) * p1
        part2 = sin(t * ω) / sin(ω) * p2
        interp = part1 + part2

        # Convert back to spherical (lon, lat in degrees)
        x, y, z = interp
        lon = atan(y, x)
        hyp = sqrt(x^2 + y^2)
        lat = atan(z, hyp)

        points[i] = (rad2deg(lon), rad2deg(lat))
    end

    return points
end
