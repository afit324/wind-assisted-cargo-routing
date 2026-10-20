#include <iostream>
#include "api.h"

int main()
{

    // Define the environmental conditions
    double true_wind_angle_deg = 0;
    double true_wind_speed_mps = 10;
    double sea_current_angle_deg = 242.1;
    double sea_current_speed_kn = 0 / 0.5144;

    // Create a Result struct for the output
    Result result;

    // And a return code for

    // Calculate the vessel performance using boat speed as input
    double boat_speed_kn = 9.0;
    if (get_performance_from_speed(true_wind_angle_deg, true_wind_speed_mps, sea_current_angle_deg, sea_current_speed_kn, boat_speed_kn, &result) < 0)
    {
        std::cerr << "Calling Cadet failed..." << std::endl;
    }
    else
    {
        std::cout << "TWS: " << true_wind_speed_mps << " TWA: " << true_wind_angle_deg << " Boat speed [kn]: " << boat_speed_kn << std::endl;
        std::cout << "Achieved boat speed [kn]: " << result.speed_over_ground_kn << " Required power [kW]: " << result.propulsion_power_kw << " Propeller rps [rev/s]: " << result.propeller_rps << std::endl;
    }

    // Calculate the vessel performance using a propulsion power as input
    double propulsion_power_kw = 100000;
    if (get_performance_from_power(true_wind_angle_deg, true_wind_speed_mps, sea_current_angle_deg, sea_current_speed_kn, propulsion_power_kw, &result) < 0)
    {
        std::cerr << "Calling Cadet failed..." << std::endl;
    }
    else
    {
        std::cout << "TWS: " << true_wind_speed_mps << " TWA: " << true_wind_angle_deg << " Propulsion_power [kW]: " << propulsion_power_kw << std::endl;
        std::cout << "Achieved boat speed [kn]: " << result.speed_over_ground_kn << " Required power [kW]: " << result.propulsion_power_kw << " Propeller rps [rev/s]: " << result.propeller_rps << std::endl;
    }

    // Calculate the vessel performance using a propeller rps as input
    double propeller_rps = 1.0;
    if (get_performance_from_rps(true_wind_angle_deg, true_wind_speed_mps, sea_current_angle_deg, sea_current_speed_kn, propeller_rps, &result) < 0)
    {
        std::cerr << "Calling Cadet failed..." << std::endl;
    }
    else
    {
        std::cout << "TWS: " << true_wind_speed_mps << " TWA: " << true_wind_angle_deg << " Propeller rps [rev/s]: " << propeller_rps << std::endl;
        std::cout << "Achieved boat speed [kn]: " << result.speed_over_ground_kn << " Required power [kW]: " << result.propulsion_power_kw << " Propeller rps [rev/s]: " << result.propeller_rps << std::endl;
    }

    return 0;
}
