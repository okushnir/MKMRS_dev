#!/usr/bin/env python
"""
@Created by: Biodaat; Oded Kushnir

"""
import argparse
import glob
import os
import numpy as np
import pandas as pd
import re
import sys


def extract_tables_with_filter(file_path, filter_table=None, filter_column=None):
    """
    Extracts multiple tables from a file where tables are separated by lines like '[marker]'.
    Optionally filters rows of a specific table based on non-empty values in a given column.

    Args:
        file_path (str): Path to the input file.
        filter_table (str): Name of the table to filter (marker).
        filter_column (str): Name of the column to filter for non-empty values.

    Returns:
        dict: A dictionary where keys are table names (from markers) and values are pandas DataFrames.
    """
    data_sections = {}
    current_marker = None
    current_lines = []

    # Read the file line by line
    with open(file_path, "r", encoding="utf-8-sig") as file:
        print(f"Processing file: {file_path}")
        for line in file:
            # Strip leading/trailing spaces and remove extra commas
            clean_line = line.split(",")[0].strip()

            # Check if the line contains a marker
            marker_match = re.match(r"\[\s*(.*?)\s*\]", clean_line, re.IGNORECASE)
            if marker_match:
                # Save the previous section, if any
                if current_marker:
                    print(f"Detected marker: {current_marker}, lines: {len(current_lines)}")
                    data_sections[current_marker] = current_lines

                # Extract the marker name
                current_marker = marker_match.group(1)
                current_lines = []  # Reset lines for this new section
            else:
                # Add the original line to the current section
                if current_marker:
                    current_lines.append(line.strip())

        # Store the last section after finishing the file
        if current_marker and current_lines:
            print(f"Detected marker: {current_marker}, lines: {len(current_lines)}")
            data_sections[current_marker] = current_lines

    # Function to convert raw lines to DataFrame
    def lines_to_dataframe(lines):
        if not lines:
            return pd.DataFrame()  # Return empty DataFrame for empty input

        header = lines[0].split(",")  # Extract header
        expected_columns = len(header)  # Determine the expected number of columns

        data = []
        for line in lines[1:]:  # Process data lines
            row = line.split(",")
            if len(row) == expected_columns:
                data.append(row)  # Add only rows with the correct column count
            else:
                print(f"Skipping row with mismatched columns: {row}")

        return pd.DataFrame(data, columns=header)

    # Convert each section into DataFrames
    dataframes = {marker: lines_to_dataframe(lines) for marker, lines in data_sections.items()}

    # Apply filtering to the specified table, if requested
    if filter_table and filter_column:
        if filter_table in dataframes:
            df = dataframes[filter_table]
            dataframes[filter_table] = df[df[filter_column].notna() & (df[filter_column] != "")]

    return dataframes


def process_directory(directory, filter_table, filter_column):
    """
    Processes all files in a directory, extracting and filtering data from a specific table.

    Args:
        directory (str): Path to the directory containing files.
        filter_table (str): Table name to extract (e.g., "Megalodon Capture").
        filter_column (str): Column name to filter rows by.

    Returns:
        pd.DataFrame: Combined DataFrame from all matching files.
    """
    combined_data = []
    boat_tbl = []

    # List all files in the directory
    delphis_lst = glob.glob(os.path.join(directory, '*'))
    for filename in delphis_lst:
        print(f"Processing file: {filename}")
        delphis_tbl = extract_tables_with_filter(filename, filter_table, filter_column)
        try:
            boat_tbl = delphis_tbl["Summery"]["Boat"]
        except KeyError:
            boat_tbl = delphis_tbl["Summary"]["Boat"]

        # Append filtered data if it exists
        if filter_table in delphis_tbl:
            combined_data.append(delphis_tbl[filter_table])
            if "Summery" in delphis_tbl:
                combined_data.append(delphis_tbl["Summery"]["Boat"])
            elif "Summary" in delphis_tbl:
                combined_data.append(delphis_tbl["Summary"]["Boat"])
    # Combine all DataFrames into a single one
    if combined_data:
        return pd.concat(combined_data, ignore_index=True)

    else:
        return pd.DataFrame()  # Return an empty DataFrame if no matching data


def parse_datetime(value):
    try:
        return pd.to_datetime(value, format='%d/%m/%Y %H:%M:%S')
    except ValueError:
        return pd.to_datetime(value, format='mixed')


def fill_site_based_on_date(df, date_col='Date', site_col='site'):
    """
    Fills NA values in the `site_col` based on rows with the same `date_col`.

    Parameters:
        df (pd.DataFrame): The DataFrame to process.
        date_col (str): The name of the date column.
        site_col (str): The name of the site column.

    Returns:
        pd.DataFrame: The updated DataFrame with NA values filled in `site_col`.
    """
    # Group by the date column
    grouped = df.groupby(date_col)

    # Define a function to fill NA values in the site column within each group
    def fill_na_in_group(group):
        # Find the first non-NA value, if it exists
        non_na_values = group[site_col].dropna()
        if not non_na_values.empty:
            non_na_value = non_na_values.iloc[0]
            group[site_col] = group[site_col].fillna(non_na_value)
        return group

    # Apply the function to each group
    return grouped.apply(fill_na_in_group, include_groups=False).reset_index(drop=True)


# Define the function to replace numeric values with "Yes"
def replace_numbers_with_yes(value):
    try:
        # Try to convert the value to a float
        float_value = float(value)
        return "Yes"
    except (ValueError, TypeError):
        # If conversion fails, return the original value
        return value


def main(args):
    delphis_dir = args.delphis_dir
    transmitters_unique = args.transmitters_unique
    tagging_dict_df = pd.read_csv(args.tagging_dict)
    tagging_list_df = pd.read_csv(args.tagging_list)
    output_file = args.output_file


    if args.readonly:
        print("Readonly mode enabled. Files in Delphis_raw and transmitters_unique will not be modified.")

    # Process the directory and extract filtered data
    final_table = process_directory(delphis_dir, "Megalodon Capture", "Spagetti")
    final_table["Boat"] = final_table[0].shift(-1)
    tagging_dict = dict(zip(tagging_dict_df.iloc[:, 0], tagging_dict_df.iloc[:, 1]))
    final_table = final_table.rename(columns=tagging_dict)
    tagging_lst = list(tagging_dict.values())
    final_table = final_table[tagging_lst]
    final_table["tag_date"] = final_table["tag_date"].replace("NaN", None)
    # final_table= final_table.dropna(subset="tag_date" ,axis=0)

    # Convert the column to datetime format (if not already)
    final_table['Datetime'] = final_table["tag_date"].apply(parse_datetime)
    # Extract only the date
    final_table['Date'] = final_table['Datetime'].dt.date

    final_table["site"] = final_table["site"].replace("NaN", None)
    final_table["site"] = final_table["site"].replace("None", None)
    final_table = fill_site_based_on_date(final_table, "Date", "site")
    final_table = final_table[tagging_lst]

    # Remove invalid characters
    with open(transmitters_unique, "rb") as f:
        content = f.read()


    # Process and clean transmitters file
    # Replace invalid bytes
    with open(transmitters_unique, "rb") as f:
        content = f.read()

    cleaned_content = content.decode("utf-8", errors="replace")

    # If not readonly, write cleaned content back to the original file
    if not args.readonly:
        with open(transmitters_unique, "w", encoding="utf-8") as f:
            f.write(cleaned_content)
    else:
        # Use the cleaned content directly without saving it back
        with open("temp_transmitters.csv", "w", encoding="utf-8") as temp_file:
            temp_file.write(cleaned_content)
        transmitters_unique = "temp_transmitters.csv"

    # Load cleaned file
    transmitters = pd.read_csv(transmitters_unique, sep=",", encoding="utf-8")
    transmitters = transmitters[["serial_number", "temp_id", "depth_id", "protocol","est_tag_life_days", "tag_turned_on_date"]]
    transmitters["tag_turned_on_date"] = pd.to_datetime(transmitters["tag_turned_on_date"], dayfirst=True)
    transmitters["acoustic_tag_removed_date"] = transmitters["tag_turned_on_date"] + pd.to_timedelta(transmitters["est_tag_life_days"], unit='days')
    transmitters = transmitters.rename(columns={"serial_number": "acoustic_tag_serial"})
    final_table = final_table.merge(transmitters, on="acoustic_tag_serial", how="left")
    final_table = final_table[final_table["visual_tag"] != "NA"]
    final_table["blood"] = final_table["blood"].apply(replace_numbers_with_yes)
    final_table["skin_mb"] = final_table["skin_mb"].apply(replace_numbers_with_yes)
    final_table["cloaca_mb"] = final_table["cloaca_mb"].apply(replace_numbers_with_yes)
    final_table["gills_mb"] = final_table["gills_mb"].apply(replace_numbers_with_yes)
    final_table["water_mb"] = final_table["water_mb"].apply(replace_numbers_with_yes)
    final_table["wound_mb"] = final_table["wound_mb"].apply(replace_numbers_with_yes)
    final_table["catcam"] = final_table["catcam"].apply(replace_numbers_with_yes)
    final_table["ultrasound_device"] = final_table["ultrasound_device"].apply(replace_numbers_with_yes)
    final_table[['latitude', 'longitude']] = final_table['longitude'].str.split(' ', expand=True)
    final_table["site"] = np.where(final_table["site"]=="Dakar-Raphi", "Hadera",
                                   np.where(final_table["site"] =="Adva Boston", "Ashdod Power Plant",
                                            final_table["site"]))

    # Ensure visual_tag is integer
    final_table.dropna(subset=["visual_tag"], inplace=True)
    final_table["visual_tag"] = final_table["visual_tag"].astype(int)
    final_table = final_table.sort_values(by="visual_tag", ascending=True)
    new_tagging_lst = list(tagging_list_df.iloc[:, 0])
    final_table = final_table[new_tagging_lst]
    # final_table = final_table.drop_duplicates(subset=["visual_tag"])
    # Save the final table to the output file
    if not final_table.empty:
        final_table.to_csv(output_file, index=False, float_format='%.0f')
        print(f"Combined table saved to {output_file}")
    else:
        print("No data matching the criteria was found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("delphis_dir", type=str, help="The path to the Delphis_raw directory")
    parser.add_argument("transmitters_unique", type=str, help="The path to the transmitters_unique")
    parser.add_argument("tagging_dict", type=str, help="The path to the tagging_dict.csv")
    parser.add_argument("tagging_list", type=str, help="The path to the tagging_list.csv")
    parser.add_argument("output_file", type=str, help="The path of the output file")
    parser.add_argument("--readonly", action="store_true", help="Prevent file overwriting")
    args = parser.parse_args(sys.argv[1:])
    main(args)
