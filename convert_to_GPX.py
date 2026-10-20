import geopandas as gpd
from datetime import datetime, timedelta, timezone
import csv

# Convert GeoJSON to GPX format function
def convert_geojson_to_gpx(input_geojson, output_gpx):
    """
    Convert a GeoJSON file to GPX format.
    
    Arguments:
    - input_geojson: str, path to the input GeoJSON file.
    - output_gpx: str, path to the output GPX file.
    """
    # Read the GeoJSON file
    gdf = gpd.read_file(input_geojson)
    
    # Write to GPX format
    gdf.to_file(output_gpx, driver="GPX")

def csv_to_gpx(csv_file, gpx_file, start_date):
    gpx_header = """<?xml version="1.0" encoding="UTF-8"?>
    <gpx version="1.1" creator="CSVtoGPX" xmlns="http://www.topografix.com/GPX/1/1">
    <trk>
    <trkseg>
    """
    gpx_footer = """
    </trkseg>
    </trk>
    </gpx>
    """

    gpx_points = ""
    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            lat = row['latitude']
            lon = row['longitude']
            time = row['time']
            # add time in s to the start date
            time = start_date + timedelta(seconds=float(time))
            time = time.strftime("%Y-%m-%dT%H:%M:%SZ")
            gpx_points += f'<trkpt lat="{lat}" lon="{lon}"><time>{time}</time></trkpt>\n'

    with open(gpx_file, "w") as f:
        f.write(gpx_header + gpx_points + gpx_footer)

    print(f"Saved GPX to {gpx_file}")
