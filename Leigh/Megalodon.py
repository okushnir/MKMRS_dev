import os
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
import csv

# Set up directories
Dir1 = os.path.expanduser("~/Leigh")
Dir2 = os.path.join(Dir1, "Delphis_raw")
Dir3 = Dir1


# Function to extract the date from filenames
def extract_date_from_filename(filename):
    try:
        return datetime.strptime(filename[9:17], "%Y%m%d").date()
    except ValueError:
        return None


# Get a list of all filenames in the folder
filenames = [f for f in os.listdir(Dir2) if os.path.isfile(os.path.join(Dir2, f))]

# Filter filenames that start with "transect_" and have 8 consecutive digits
valid_filenames = [f for f in filenames if re.match(r"^transect_\d{8}", f)]

# Extract dates and convert them to Date objects
dates = [extract_date_from_filename(f) for f in valid_filenames]
dates = [d for d in dates if d is not None]

# Find the latest date and its corresponding file
if dates:
    latest_date = max(dates)
    latest_index = dates.index(latest_date)
    Survey_file = valid_filenames[latest_index]
else:
    print("No valid transect files found.")
    exit()

print(f"Survey file selected: {Survey_file}")


# Load the survey file
def load_survey_file(file_path):
    return pd.read_csv(file_path, header=None, encoding="utf-8")


survey_file_path = os.path.join(Dir2, Survey_file)
# data_in = pd.read_csv(survey_file_path, header=None, encoding="utf-8")
with open(survey_file_path, encoding="utf-8") as file:
    reader = csv.reader(file)
    for i, row in enumerate(reader):
        print(row)
        if i > 10:  # Print only the first 10 rows for inspection
            break

# Process Delphis output into a list of grouped DataFrames
def process_delphis_output(data):
    data["categ"] = np.where(data[0].str.contains(r"\["), data[0], None)
    data["categ"] = data["categ"].fillna(method="ffill")
    grouped = data.groupby("categ")
    return {name: group.reset_index(drop=True) for name, group in grouped}


data_in_list = process_delphis_output(data_in)


# Extract "[Megalodon Capture]" data
def prepare_shark_data(data_in_list):
    if "[Megalodon Capture]" not in data_in_list:
        print("Megalodon Capture data not found.")
        return pd.DataFrame()

    Sharks = data_in_list["[Megalodon Capture]"].copy()
    Sharks.columns = Sharks.iloc[1]  # Set second row as column names
    Sharks = Sharks.iloc[2:].reset_index(drop=True)  # Remove unnecessary rows
    Sharks.columns = Sharks.columns.str.replace(" ", ".")  # Replace spaces with "."

    # Split "Lat.Lng" column into "Lat" and "Lon"
    if "Lat.Lng" in Sharks.columns:
        Sharks[["Lat", "Lon"]] = Sharks["Lat.Lng"].str.split(" ", expand=True).astype(float)
        Sharks.drop(columns=["Lat.Lng"], inplace=True)

    # Drop rows with all NAs
    Sharks.dropna(how="all", inplace=True)
    return Sharks


Sharks = prepare_shark_data(data_in_list)
print("Sharks DataFrame processed.")


# Save processed Sharks data
def save_dataframe(dataframe, file_name, directory):
    output_path = os.path.join(directory, file_name)
    dataframe.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")


save_dataframe(Sharks, "Processed_Sharks.csv", Dir3)

# Additional steps to replicate the rest of the R script can be added here...