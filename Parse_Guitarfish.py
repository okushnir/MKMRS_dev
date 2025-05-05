import pandas as pd
import json
import hashlib
from datetime import datetime
from geopy.distance import geodesic

COASTLINE_POINTS = {
    "Haifa_Beach": (32.8323, 34.9706),
    "Tel_Aviv_Promenade": (32.0809, 34.7686),
    "Ashdod_Coast": (31.8032, 34.6436),
    "Eilat_Beach": (29.5581, 34.9519),
    "Netanya_Cliffs": (32.3281, 34.8561),
    "Hadera_Shore": (32.4421, 34.9036),
    "Ashkelon_Marina": (31.6688, 34.5721)
}

def define_unique_id(event_date, lat, lng, suffix, prefix="GF_EVT_"):
    base = f"{event_date.isoformat()}_{lat:.6f}_{lng:.6f}_{suffix}"
    return f"{prefix}{hashlib.sha1(base.encode()).hexdigest()[:12]}"

def nearest_coast_shore(lat, lng):
    return min(COASTLINE_POINTS.items(),
               key=lambda x: geodesic((lat, lng), x[1]).meters)[0]

def parse_guitar_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    recorded_by, start_date = None, None
    for line in lines:
        if line.startswith("START:"):
            start = json.loads(line[6:])
            recorded_by = start["owner"]
            start_date = datetime.strptime(start["stringDate"], "%d/%m/%Y %H:%M:%S")
            break

    location_lat, location_lng = None, None
    for line in lines:
        if line.startswith("LOCATION:"):
            loc = json.loads(line[9:])
            location_lat, location_lng = loc["latLng"]
            break

    yyyymmdd = start_date.strftime("%Y%m%d")
    field_number = f"GF-{yyyymmdd}-001"
    event_id = define_unique_id(start_date, location_lat, location_lng, field_number)

    # Core Events
    rhini_total = crab_total = jelly_total = 0
    for line in lines:
        if line.startswith("OBSERVATION:"):
            obs = json.loads(line[12:])
            rhini_total += obs.get("rhiniCount", 0)
            crab_total += obs.get("crabCount", 0)
            jelly_total += obs.get("jellyfishCount", 0)

    core = pd.DataFrame([{
        "eventID": event_id,
        "eventDate": start_date,
        "samplingProtocol": "Visual Survey",
        "habitat": "Nearshore Marine",
        "locationID": nearest_coast_shore(location_lat, location_lng),
        "decimalLatitude": location_lat,
        "decimalLongitude": location_lng,
        "geodeticDatum": "WGS84",
        "fieldNumber": field_number,
        "recordedBy": recorded_by,
        "remarks": f"Total rhini: {rhini_total}, crab: {crab_total}, jellyfish: {jelly_total}"
    }])

    # Occurrences
    occurrences = []
    count = 1
    for line in lines:
        if line.startswith("OBSERVATION:"):
            obs = json.loads(line[12:])
            lat, lng = obs["latLng"]
            oid = define_unique_id(start_date, lat, lng, f"{field_number}_{count:03d}", prefix="")
            occurrences.append({
                "occurrenceID": oid,
                "eventID": event_id,
                "scientificName": "Glaucostegus cemiculus",
                "taxonRank": "species",
                "individualCount": obs.get("rhiniCount", 0),
                "decimalLatitude": lat,
                "decimalLongitude": lng,
                "occurrenceStatus": "present",
                # "organismQuantityType": "individuals",
                # "organismQuantity": obs.get("rhiniCount", 0),
                "recordedBy": recorded_by,
                "occurrenceRemarks": f"Other counts - crab: {obs.get('crabCount', 0)}, jellyfish: {obs.get('jellyfishCount', 0)}"
            })
            count += 1
        elif line.startswith("LOCATION:"):
            loc = json.loads(line[9:])
            lat, lng = loc["latLng"]
            oid = define_unique_id(start_date, lat, lng, f"{field_number}_{count:03d}", prefix="")
            occurrences.append({
                "occurrenceID": oid,
                "eventID": event_id,
                "scientificName": "Glaucostegus cemiculus",
                "taxonRank": "species",
                "individualCount": 0,
                "decimalLatitude": lat,
                "decimalLongitude": lng,
                "occurrenceStatus": "absent",
                # "organismQuantityType": "individuals",
                # "organismQuantity": 0,
                "recordedBy": recorded_by,
                "occurrenceRemarks": "No observation recorded"
            })
            count += 1
    # --- Extended_MeasurementsOrFacts table ---
    env_mapping = {
        "envTemp": ("air temperature", "°C", "Thermometer"),
        "waterTemp": ("water temperature", "°C", "CTD sensor"),
        "waveHeight": ("wave height", "m", "Visual estimation"),
        "windDir": ("wind direction", "°C", "Weather station"),
        "windSpeed": ("wind speed", "m/s", "Weather station"),
    }

    measurements = []
    measurement_count = 1
    env_data_found = False  # <-- ADD THIS FLAG

    for line in lines:
        if line.startswith("ENV_DATA:") and not env_data_found:
            env_data_found = True  # <-- PROCESS ONLY THE FIRST BLOCK
            env_data = json.loads(line[len("ENV_DATA:"):])
            env_vals = env_data.get("environmentalDataValues", {})
            shared_mid = define_unique_id(start_date, location_lat, location_lng, f"MEAS_{measurement_count:03d}",
                                          prefix="")
            measurement_count += 1
            for key, (mtype, unit, method) in env_mapping.items():
                if key in env_vals:
                    measurements.append({
                        "measurementID": shared_mid,
                        "eventID": event_id,
                        "occurrenceID": None,
                        "measurementType": mtype,
                        "measurementValue": str(env_vals[key]),
                        "measurementUnit": unit,
                        "measurementMethod": method,
                        "measurementRemarks": None
                    })

    # Add totalDistance from END line
    for line in reversed(lines):
        if line.startswith("END:"):
            end_data = json.loads(line[len("END:"):])
            distance_val = str(end_data.get("totalDistance", "0"))
            mid = define_unique_id(start_date, location_lat, location_lng, f"MEAS_TOTALDIST", prefix="")
            measurements.append({
                "measurementID": mid,
                "eventID": event_id,
                "occurrenceID": None,
                "measurementType": "totalDistance",
                "measurementValue": distance_val,
                "measurementUnit": "kilometers",
                "measurementMethod": None,
                "measurementRemarks": "Extracted from END block"
            })
            break

    return (
        pd.DataFrame([core.iloc[0]]),
        pd.DataFrame(occurrences),
        pd.DataFrame(measurements)
    )

if __name__ == "__main__":
    dir_path = "/Users/odedkushnir/MKMRS/Guitarfish/Efrat/" 
    core_df, occ_df, meas_df = parse_guitar_file(dir_path + "20250311_0632_danarein.guitarfish.txt")
    core_df.to_csv(dir_path + "core_events.csv", index=False)
    occ_df.to_csv(dir_path + "occurrences.csv", index=False)
    meas_df.to_csv(dir_path + "measurements.csv", index=False)