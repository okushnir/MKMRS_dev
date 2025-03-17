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
import chardet

def detect_encoding(file_path):
    """Detects the encoding of a file by reading a small portion of it."""
    with open(file_path, "rb") as f:
        raw_data = f.read(100000)  # Read first 100KB for analysis
    result = chardet.detect(raw_data)
    return result["encoding"] if result["encoding"] else "utf-8"

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
    encoding = detect_encoding(file_path)  # Auto-detect encoding
    print(f"Detected encoding for {file_path}: {encoding}")  # Debugging output

    data_sections = {}
    current_marker = None
    current_lines = []

    # Read the file line by line
    with open(file_path, "r", encoding=encoding, errors="replace") as file:
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

    # List all files in the directory
    delphis_lst = glob.glob(os.path.join(directory, '*'))
    for filename in delphis_lst:
        print(f"Processing file: {filename}")
        delphis_tbl = extract_tables_with_filter(filename, filter_table, filter_column)

        # Append filtered data if it exists
        if filter_table in delphis_tbl:
            combined_data.append(delphis_tbl[filter_table])

    # Combine all DataFrames into a single one
    if combined_data:
        return pd.concat(combined_data, ignore_index=True)
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no matching data


def main(args):
    delphis_dir = args.delphis_dir
    transmitters_unique = args.transmitters_unique
    output_file = args.output_file

    if args.readonly:
        print("Readonly mode enabled. Files in Delphis_raw and transmitters_unique will not be modified.")

    # Process the directory and extract filtered data
    final_table = process_directory(delphis_dir, "Megalodon Capture", "Spagetti")
    final_table = final_table.rename(columns={"Spagetti": "visual_tag", "PITDec": "dec_tag_id", "Satellite Tag Type": "type",
                                             "Acoustic Serial": "acoustic_serial", "Acoustic ID": "acoustic_tag_id", "Specie": "species", "Gender": "sex", "Shark/Ray Name": "shark_name"})
    final_table = final_table[["visual_tag", "dec_tag_id", "type", "acoustic_serial", "acoustic_tag_id", "species", "sex", "shark_name"]]
    final_table["species"] = np.where(final_table["species"] == "Dusky", "C. obscurus",
                           np.where(final_table["species"] == "Sandbar", "C. plumbeus", final_table["species"]))

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
    transmitters = transmitters[["serial_number", "acoustic_tag_id", "temp_id", "depth_id"]]
    transmitters = transmitters.rename(columns={"serial_number": "acoustic_serial"})
    final_table = final_table.merge(transmitters, on="acoustic_serial", how="outer")
    final_table = final_table[
        ["visual_tag", "dec_tag_id", "type", "acoustic_serial", "acoustic_tag_id_x", "temp_id", "depth_id", "species",
         "sex", "shark_name"]]
    final_table = final_table.rename(columns={"acoustic_tag_id_x": "acoustic_tag_id"})
    final_table = final_table[final_table["visual_tag"] != "NA"]
    final_table["dec_tag_id"] = np.where(final_table["dec_tag_id"]=="NA", 0, final_table["dec_tag_id"])
    final_table = final_table.drop_duplicates(subset=["dec_tag_id"])

    # Convert the 'dec_tag_id' column to strings to avoid scientific notation
    final_table["dec_tag_id"] = final_table["dec_tag_id"].apply(
        lambda x: '{:.0f}'.format(float(x)) if pd.notnull(x) and str(x).strip() != '' and str(x).replace('.', '', 1).isdigit() else x)
    # Ensure visual_tag is integer
    final_table.dropna(subset=["visual_tag"], inplace=True)
    final_table["visual_tag"] = final_table["visual_tag"].astype(int)
    final_table["dec_tag_id"] = final_table["dec_tag_id"].astype(str)
    final_table["dec_tag_id"] = "'" + final_table["dec_tag_id"]
    final_table = final_table.sort_values(by="visual_tag", ascending=True)
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
    parser.add_argument("output_file", type=str, help="The path of the output file")
    parser.add_argument("--readonly", action="store_true", help="Prevent file overwriting")
    args = parser.parse_args(sys.argv[1:])
    main(args)
