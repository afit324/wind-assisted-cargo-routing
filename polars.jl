using Plots

include("plots.jl")

# Define the structure for the result
struct Result
    speed_over_ground_kn::Cdouble
    propulsion_power_kw::Cdouble
    propeller_rps::Cdouble
end

# Define constants for return values
const SUCCESS = 0
const ERROR = -1
const ERROR_REQUESTED_SPEED_OUT_OF_BOUNDS = -2
const ERROR_REQUESTED_POWER_OUT_OF_BOUNDS = -3
const ERROR_REQUESTED_RPS_OUT_OF_BOUNDS = -4
const ERROR_WIND_SPEED_OUT_OF_BOUNDS = -5
const ERROR_UNFEASIBLE_ENGINE_OPERATING_POINT = -6

# Load the shared library
const libcadet = "cadet_generic/cadet_generic.dll"  

# This cadet package contains tabulated performance indicators for a Generic Wind-Assisted Ship. 
# Interpolation within the tabulated data is performed by linear interpolation.


function getPerformanceFromSpeed(wind_angle::Float64, wind_speed::Float64,
                                    current_angle::Float64, current_speed::Float64,
                                    sog::Float64)
    """
    This function retrieves the performance data of a boat based on wind speed and angle, 
    current speed and angle, and speed over ground (SOG).

    Arguments:
    - `wind_angle::Float64`: The true wind angle in degrees, relative to the vessel bow (deg), wind is coming from this 
        direction. Positive to starboard, headwind is 0, internally wrapped to [-180, 180]).
    - `wind_speed::Float64`: The wind speed (kn).
    - 'current_angle::Float64': The sea current angle relative to the vessel bow (deg), the current is moving twoards 
        this direction. Postitive to starboard, following current is 0, internally wrapped to [-180, 180].
    - `current_speed::Float64`: The sea current speed in knots (kn).
    - `sog::Float64`: The speed over ground (m/s).

    Returns:
    - `ret::Cint`: Return code indicating success or error.
    - `result::Result`: A structure containing the performance data:
        - `speed_over_ground_kn::Cdouble`: Speed over ground in knots (kn).
        - `propulsion_power_kw::Cdouble`: Propulsion power in kilowatts (kW).
        - `propeller_rps::Cdouble`: Propeller revolutions per second (RPS).

    """
    # Create a result object to hold the performance data
    result = Ref(Result(0.0, 0.0, 0.0))

    # Call the C function to get performance data
    ret = ccall((:get_performance_from_speed, libcadet),
                Cint,
                (Cdouble, Cdouble, Cdouble, Cdouble, Cdouble, Ref{Result}),
                wind_angle, wind_speed, current_angle, current_speed, sog, result)
    return ret, result[]
end


function getPerformanceFromPower(wind_angle::Float64, wind_speed::Float64,
                                    current_angle::Float64, current_speed::Float64,
                                    power::Float64)
    """
    This function retrieves the performance data of a boat based on wind speed and angle, 
    current speed and angle, and propulsion power.

    Arguments:
    - `wind_angle::Float64`: The true wind angle in degrees, relative to the vessel bow (deg), wind is coming from this 
        direction. Positive to starboard, headwind is 0, internally wrapped to [-180, 180]).
    - `wind_speed::Float64`: The wind speed (m/s).
    - 'current_angle::Float64': The sea current angle relative to the vessel bow (deg), the current is moving twoards 
        this direction. Postitive to starboard, following current is 0, internally wrapped to [-180, 180].
    - `current_speed::Float64`: The sea current speed in knots (kn).
    - `power::Float64`: The propulsion power in kilowatts (kW).

     Returns:
    - `ret::Cint`: Return code indicating success or error.
    - `result::Result`: A structure containing the performance data:
        - `speed_over_ground_kn::Cdouble`: Speed over ground in knots (kn).
        - `propulsion_power_kw::Cdouble`: Propulsion power in kilowatts (kW).
        - `propeller_rps::Cdouble`: Propeller revolutions per second (RPS).
        
    """

    # Create a result object to hold the performance data
    result = Ref(Result(0.0, 0.0, 0.0))

    # Call the C function to get performance data
    ret = ccall((:get_performance_from_power, libcadet),
                Cint,
                (Cdouble, Cdouble, Cdouble, Cdouble, Cdouble, Ref{Result}),
                wind_angle, wind_speed, current_angle, current_speed, power, result)
    return ret, result[]
end


function getPerformanceFromRps(wind_angle::Float64, wind_speed::Float64,
                                  current_angle::Float64, current_speed::Float64,
                                  rps::Float64)
    """
    This function retrieves the performance data of a boat based on wind speed and angle,
    current speed and angle, and propeller revolutions per second (RPS).

    Arguments:
    - `wind_angle::Float64`: The true wind angle in degrees, relative to the vessel bow (deg), wind is coming from this 
        direction. Positive to starboard, headwind is 0, internally wrapped to [-180, 180]).
    - `wind_speed::Float64`: The wind speed (m/s).
    - 'current_angle::Float64': The sea current angle relative to the vessel bow (deg), the current is moving twoards 
        this direction. Postitive to starboard, following current is 0, internally wrapped to [-180, 180].
    - `current_speed::Float64`: The sea current speed in knots (kn).
    - `rps::Float64`: The propeller revolutions per second (RPS).

     Returns:
    - `ret::Cint`: Return code indicating success or error.
    - `result::Result`: A structure containing the performance data:
        - `speed_over_ground_kn::Cdouble`: Speed over ground in knots (kn).
        - `propulsion_power_kw::Cdouble`: Propulsion power in kilowatts (kW).
        - `propeller_rps::Cdouble`: Propeller revolutions per second (RPS).

    """
    # Create a result object to hold the performance data
    result = Ref(Result(0.0, 0.0, 0.0))

    # Call the C function to get performance data
    ret = ccall((:get_performance_from_rps, libcadet),
                Cint,
                (Cdouble, Cdouble, Cdouble, Cdouble, Cdouble, Ref{Result}),
                wind_angle, wind_speed, current_angle, current_speed, rps, result)
    return ret, result[]
end


function plotPolarRise(wind_speeds::Vector{Float64}, current_angle::Float64, current_speed::Float64, type::String, value::Float64)

    """
    This function generates a polar diagram for a given set of wind speeds and angles based on the sharded library provided by RISE.
    
    Arguments:
    - `wind_speeds::Vector{Float64}`: A vector of wind speeds (m/s).
    - 'current_angle::Float64': The sea current angle relative to the vessel bow (deg), the current is moving twoards 
        this direction. Postitive to starboard, following current is 0, internally wrapped to [-180, 180].
    - `current_speed::Float64`: The sea current speed in m/s (m/s).
    - `type`: A string indicating the type of performance data to retrieve. Options are "power" or "rps".
    - `value`: The value of power (kW) or RPS to use to get performance data.

    Returns:
    - A polar diagram plot showing the boat speed for each wind speed for true wind angles between 0 and 180 degrees.

    Note: We can retrieve data based on boat speed, however this is not implemented in this function since we want boat speed plotted.

    """

    # Create angles from 0 to 180 degrees
    angles = range(0, stop=360, length=361)

    # Initialize the boat speed matrix
    boat_speeds = zeros(length(angles), length(wind_speeds))

    # Change current to knots for input
    current_speed = current_speed / 0.53996  # Convert from m/s to knots
    
    # Get the boat speeds for each wind speed and angle
    for i in 1:length(wind_speeds)
        wind_speed = wind_speeds[i]

        for j in 1:length(angles)
            angle = angles[j]

            # Get the boat speed for the current wind speed and angle
            if type == "power"
                ret, res = getPerformanceFromPower(angle, wind_speed, current_angle, current_speed, value)
            elseif type == "rps"
                ret, res = getPerformanceFromRps(angle, wind_speed, current_angle, current_speed, value)
            else
                println("Invalid type. Please choose 'power' or 'rps'.")
                return
            end


            if ret == SUCCESS
                boat_speeds[j, i] = res.speed_over_ground_kn
                boat_speeds[j, i] = boat_speeds[j, i] * 0.53996  # Convert from knots to m/s
            else
                println("Error: $ret")
                return
            end
        end
    end
    
    # plot the polar diagram
    plotPolar(wind_speeds, angles, boat_speeds)

end


