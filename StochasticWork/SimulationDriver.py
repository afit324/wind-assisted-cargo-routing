import os
import json
import shutil

import experiments_gribs
import ActualWeatherSimulation

#####################################################

OUTPUT_DIR = "BenchmarkResults"

#####################################################

LATITUDE_GRIDS = [

    [-45,-43,-41,-39,-37],
    [-43,-41,-39,-37,-35],
    [-41,-39,-37,-35,-33],
    [-39,-37,-35,-33,-31],
    [-37,-35,-33,-31,-29],

]

LONGITUDE_GRIDS = [

    [100,110,120,130,140,150],
    [105,115,125,135,145,155],
    [110,120,130,140,150,160],
    [115,125,135,145,155,165],
    [120,130,140,150,160,170],

]

MEASURES = [

    "wind_speed",
    "u_abs",
    "v_abs",

]

#####################################################

benchmark = 1

os.makedirs(OUTPUT_DIR, exist_ok=True)

#####################################################

for measure in MEASURES:

    for lats in LATITUDE_GRIDS:

        for lons in LONGITUDE_GRIDS:

            print("="*70)
            print(f"Benchmark {benchmark}")
            print(measure)
            print(lats)
            print(lons)
            print("="*70)

            folder = os.path.join(
                OUTPUT_DIR,
                f"Benchmark_{benchmark:03d}"
            )

            os.makedirs(folder, exist_ok=True)

            ###################################################
            # Configure routing problem
            ###################################################

            experiments_gribs.configure_grid(
                lats,
                lons
            )

            ActualWeatherSimulation.configure_grid(
                lats,
                lons
            )

            experiments_gribs.set_cost_function(
                measure
            )

            ActualWeatherSimulation.ACTIVE_COST_FN = measure

            ###################################################
            # Run forecast optimisation
            ###################################################

            results = experiments_gribs.run_experiments()

            experiments_gribs.save_results(results)

            ###################################################
            # Evaluate with ERA5
            ###################################################

            ActualWeatherSimulation.run_simulation()

            ###################################################
            # Save metadata
            ###################################################

            with open(
                os.path.join(folder, "settings.json"),
                "w"
            ) as f:

                json.dump(

                    {

                        "measure": measure,
                        "latitudes": lats,
                        "longitudes": lons,

                    },

                    f,

                    indent=4

                )

            ###################################################
            # Move outputs
            ###################################################

            files = [

                "forecast_policy_routes.pkl",
                "actual_weather_results.csv",
                "actual_weather_arc_details.csv",

            ]

            for file in files:

                if os.path.exists(file):

                    shutil.move(
                        file,
                        os.path.join(folder,file)
                    )

            benchmark += 1

print("Finished.")