#ifdef WIN32
#define CALL __cdecl
#define DLL __declspec(dllexport)
#else
#define CALL
#define DLL __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C"
{
#endif

    /// @mainpage Cadet Generic Wind-Assisted Ship
    ///
    /// This cadet package contains tabulated performance indicators for a Generic
	/// Wind-Assisted Ship. Interpolation within the tabulated data is performed by linear interpolation.
    ///

    enum ReturnCode
    {
        SUCCESS = 0,
        ERROR = -1,
        ERROR_REQUESTED_SPEED_OUT_OF_BOUNDS = -2,
        ERROR_REQUESTED_POWER_OUT_OF_BOUNDS = -3,
        ERROR_REQUESTED_RPS_OUT_OF_BOUNDS = -4,
        ERROR_WIND_SPEED_OUT_OF_BOUNDS = -5,
        ERROR_UNFEASIBLE_ENGINE_OPERATING_POINT = -6,

    };

#pragma pack(push, 1)

    /// @brief Result from interpolation at requested operating point
    typedef struct
    {
        double speed_over_ground_kn; // [kn] (speed over ground)
        double propulsion_power_kw;  // [kW]
        double propeller_rps;        // [rev/s]

    } Result;

#pragma pack(pop)

    /// @brief Interpolate in the tabulated performance polar for the specific inputted condition
    ///
    /// @param true_wind_angle_deg The true wind angle relative to the vessel bow (deg), wind is coming from this direction. Positive to starboard, headwind is 0, internally wrapped to [-180, 180].
    /// @param true_wind_speed_mps The true wind speed felt by the vessel (meters per second)
    /// @param sea_current_angle_deg The sea current angle relative to the vessel bow (deg), the current is moving twoards this direction. Postitive to starboard, following current is 0, internally wrapped to [-180, 180].
    /// @param sea_current_speed_kn The sea current speed (knots).
    /// @param speed_over_ground_kn The requested speed over ground [kn] of the vessel
    /// @param result Pointer to a Result struct that will be filled with the resulting, interpolated values on a successful call
    ///
    /// @returns Return code (ReturnCode)
    DLL ReturnCode get_performance_from_speed(double true_wind_angle_deg, double true_wind_speed_mps, double sea_current_angle_deg, double sea_current_speed_kn, double speed_over_ground_kn, Result *result);

    /// @brief Interpolate in the tabulated performance polar for the specific inputted condition
    ///
    /// @param true_wind_angle_deg The true wind angle relative to the vessel bow (deg). Positive to starboard, headwind is 0, internally wrapped to [-180, 180].
    /// @param true_wind_speed_mps The true wind speed felt by the vessel (meters per second)
    /// @param sea_current_angle_deg The sea current angle relative to the vessel bow (deg), the current is moving twoards this direction. Postitive to starboard, following current is 0, internally wrapped to [-180, 180].
    /// @param sea_current_speed_kn The sea current speed (knots).
    /// @param propulsion_power_kw The requested power [kW] delivery to the propulsion.
    /// @param result Pointer to a Result struct that will be filled with the resulting, interpolated values on a successful call
    ///
    /// @returns Return code (ReturnCode)
    DLL ReturnCode get_performance_from_power(double true_wind_angle_deg, double true_wind_speed_mps, double sea_current_angle_deg, double sea_current_speed_kn, double propulsion_power_kw, Result *result);

    /// @brief Interpolate in the tabulated performance polar for the specific inputted condition
    ///
    /// @param true_wind_angle_deg The true wind angle relative to the vessel bow (deg). Positive to starboard, headwind is 0, internally wrapped to [-180, 180].
    /// @param true_wind_speed_mps The apparent wind speed felt by the vessel (meters per second)
    /// @param sea_current_angle_deg The sea current angle relative to the vessel bow (deg), the current is moving twoards this direction. Postitive to starboard, following current is 0, internally wrapped to [-180, 180].
    /// @param sea_current_speed_kn The sea current speed (knots).
    /// @param propeller_rps The requested rps [rev/s] of the propeller(s)
    /// @param result Pointer to a Result struct that will be filled with the resulting, interpolated values on a successful call
    ///
    /// @returns Return code (ReturnCode)
    DLL ReturnCode get_performance_from_rps(double true_wind_angle_deg, double true_wind_speed_mps, double sea_current_angle_deg, double sea_current_speed_kn, double propeller_rps, Result *result);

#ifdef __cplusplus
}
#endif