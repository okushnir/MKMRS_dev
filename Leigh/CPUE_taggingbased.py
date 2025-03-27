#!/usr/bin/env python
"""
@Created by: Biodaat; Oded Kushnir
@Enhanced by: [Your Name]

A script for processing shark cpue data from multiple CSV files.
"""
import argparse
import glob
import os
import numpy as np
import pandas as pd
import re
import sys
import io


def read_file_with_encoding(file_path):
    """
    Reads a file trying multiple encodings until one works.

    Args:
        file_path (str): Path to the file to read

    Returns:
        tuple: (content_lines, encoding_used) or (None, None) if all fail
    """
    # Try these encodings in order
    encodings = ['utf-8-sig', 'cp1252', 'latin1', 'iso-8859-1']

    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as file:
                content = file.readlines()
                print(f"Successfully read file with encoding: {encoding}")
                return content, encoding
        except UnicodeDecodeError:
            print(f"Failed to read with encoding {encoding}, trying another...")

    # If all encodings fail
    print(f"ERROR: Could not read file {file_path} with any encoding")
    return None, None


def extract_sections(content_lines):
    """
    Extracts sections from content marked by [SectionName] headers.

    Args:
        content_lines (list): List of strings representing file content

    Returns:
        dict: Dictionary with section names as keys and lists of lines as values
    """
    sections = {}
    current_section = None
    current_lines = []

    for line in content_lines:
        line = line.strip()
        if not line:  # Skip empty lines
            continue

        # Check if this line contains a section marker
        section_match = re.match(r"\[\s*(.*?)\s*\]", line, re.IGNORECASE)
        if section_match:
            # Save previous section if it exists
            if current_section:
                print(f"Found section: {current_section} with {len(current_lines)} lines")
                sections[current_section] = current_lines

            # Start new section
            current_section = section_match.group(1)
            current_lines = []
        elif current_section:
            # Add line to current section
            current_lines.append(line)

    # Add the last section
    if current_section and current_lines:
        print(f"Found section: {current_section} with {len(current_lines)} lines")
        sections[current_section] = current_lines

    return sections


def parse_csv_section(lines):
    """
    Parse a section of CSV data into a DataFrame.

    Args:
        lines (list): List of strings containing CSV data

    Returns:
        pd.DataFrame: DataFrame containing the parsed data
    """
    if not lines:
        return pd.DataFrame()

    # Create a single string with newlines and parse with pandas
    csv_content = "\n".join(lines)

    try:
        # Try to parse with pandas
        df = pd.read_csv(
            pd.io.common.StringIO(csv_content),
            skip_blank_lines=True,
            on_bad_lines='warn'
        )

        # Rename unnamed columns
        unnamed_cols = [col for col in df.columns if 'Unnamed:' in str(col)]
        rename_dict = {col: f"column_{i}" for i, col in enumerate(unnamed_cols)}
        df = df.rename(columns=rename_dict)

        return df
    except Exception as e:
        print(f"Error parsing CSV section: {e}")
        # Fallback to manual parsing if pandas fails
        return manual_parse_csv(lines)


def manual_parse_csv(lines):
    """
    Manually parse CSV when pandas fails.

    Args:
        lines (list): List of strings containing CSV data

    Returns:
        pd.DataFrame: DataFrame containing the parsed data
    """
    if not lines:
        return pd.DataFrame()

    # Get header
    header = lines[0].split(',')
    header = [h.strip() for h in header]

    # Fix empty or duplicate headers
    fixed_header = []
    for i, h in enumerate(header):
        if not h:
            h = f"column_{i}"

        # Handle duplicates
        if h in fixed_header:
            j = 1
            while f"{h}_{j}" in fixed_header:
                j += 1
            h = f"{h}_{j}"

        fixed_header.append(h)

    # Process data rows
    data = []
    for line in lines[1:]:
        if not line.strip():  # Skip empty lines
            continue

        values = line.split(',')

        # Ensure consistent row length
        if len(values) < len(fixed_header):
            values.extend([''] * (len(fixed_header) - len(values)))
        elif len(values) > len(fixed_header):
            values = values[:len(fixed_header)]

        data.append(values)

    # Create DataFrame
    df = pd.DataFrame(data, columns=fixed_header)
    return df


def process_file(file_path, target_section=None, filter_column=None):
    """
    Process a single file, extracting specified sections.

    Args:
        file_path (str): Path to the file
        target_section (str): Section to extract (e.g., "Megalodon Capture")
        filter_column (str): Column to filter non-empty values by

    Returns:
        dict: Dictionary of DataFrames, with section names as keys
    """
    print(f"Processing file: {file_path}")
    content_lines, encoding = read_file_with_encoding(file_path)

    if not content_lines:
        return {}

    # Extract sections from content
    sections = extract_sections(content_lines)

    # Convert sections to DataFrames
    dataframes = {}
    for section_name, section_lines in sections.items():
        try:
            df = parse_csv_section(section_lines)

            # Add source file information
            df['source_file'] = os.path.basename(file_path)

            dataframes[section_name] = df
            print(f"  Parsed {section_name}: {df.shape[0]} rows, {df.shape[1]} columns")
        except Exception as e:
            print(f"  Error parsing section {section_name}: {e}")

    # Apply filtering if requested
    if target_section and filter_column and target_section in dataframes:
        df = dataframes[target_section]
        if filter_column in df.columns:
            # Filter rows with non-empty values in the column
            df_filtered = df[df[filter_column].notna() & (df[filter_column].astype(str) != "")]
            print(
                f"  Filtered {target_section} from {df.shape[0]} to {df_filtered.shape[0]} rows using {filter_column}")
            dataframes[target_section] = df_filtered

    return dataframes


def process_directory(directory, target_section, filter_column, capture_dict_df, gear_dict_df,
                      capture_mapping, gear_mapping, cpue_list_df):
    """
    Process all files in a directory.

    Args:
        directory (str): Directory path containing files to process
        target_section (str): Section to extract
        filter_column (str): Column to filter by

    Returns:
        pd.DataFrame: Combined DataFrame from all files
    """
    combined_data = []
    processed_files = 0
    final_cpue_data = pd.DataFrame()

    # List all files in the directory
    file_paths = glob.glob(os.path.join(directory, '*'))
    print(f"Found {len(file_paths)} files in directory")

    for file_path in file_paths:
        # Skip directories
        if os.path.isdir(file_path):
            continue

        # Process the file
        dataframes = process_file(file_path, target_section, filter_column)
        capture_data = dataframes["Megalodon Capture"]
        gear_data = dataframes["Megalodon Gear"]
        # print(gear_data.head().to_string())

        # Filter the DataFrame to keep only non-column_ columns
        columns_to_keep = [col for col in gear_data.columns if not col.startswith("column_") or not col[7:].isdigit()]
        # print(f"Columns to keep: {columns_to_keep}")
        gear_data = gear_data[columns_to_keep]
        gear_data = gear_data.dropna(subset="DateTime")

        if capture_data.empty:
            print("No capture data found. Exiting.")
            return
        if gear_data.empty:
            print("No gear data found. Exiting.")
            return

        # Map columns to expected names
        capture_data = capture_data.dropna(subset="DateTime")
        valid_capture_columns = capture_dict_df["Column name"].tolist()
        capture_data = capture_data[valid_capture_columns]
        capture_data = map_columns(capture_data, capture_mapping)

        capture_data["col_date"] = pd.to_datetime(capture_data["col_date"], errors='coerce')
        capture_data["col_date"] = capture_data["col_date"].dt.strftime('%d/%m/%Y')
        capture_data["col_time"] = pd.to_datetime(capture_data["col_time"], errors='coerce')
        capture_data["col_time"] = capture_data["col_time"].dt.strftime('%H:%M')
        capture_data["col_date_time"] = capture_data["col_date"] + " " + capture_data["col_time"]
        capture_data["col_date_time"] = pd.to_datetime(capture_data["col_date_time"], errors='coerce')
        capture_data = capture_data.drop(columns=["col_time", "col_date"])

        # Create a dictionary mapping Drumline Number to Bait On/Off from Collection rows
        bait_status_by_drumline = {}
        for _, row in gear_data[gear_data['Activity'] == 'Collection'].iterrows():
            if pd.notna(row['Drumline Number']):
                bait_status_by_drumline[row['Drumline Number']] = row['Bait On/Off']

        # Update Deployment rows with corresponding Bait On/Off values
        for i, row in gear_data.iterrows():
            if (row['Activity'] == 'Deployment' and
                    pd.notna(row['Drumline Number']) and
                    row['Drumline Number'] in bait_status_by_drumline):
                gear_data.at[i, 'Bait On/Off'] = bait_status_by_drumline[row['Drumline Number']]

        # Create a dictionary to store DateTime values from Collection activities
        datetime_by_drumline = {}
        for _, row in gear_data[gear_data['Activity'] == 'Collection'].iterrows():
            if pd.notna(row['Drumline Number']):
                datetime_by_drumline[row['Drumline Number']] = row['DateTime']

        # Add a new column col_date_time to the DataFrame
        gear_data['col_date_time'] = None

        # Update Deployment rows with corresponding DateTime values from Collection
        for i, row in gear_data.iterrows():
            if (row['Activity'] == 'Deployment' and
                    pd.notna(row['Drumline Number']) and
                    row['Drumline Number'] in datetime_by_drumline):
                gear_data.at[i, 'col_date_time'] = datetime_by_drumline[row['Drumline Number']]

        gear_data = gear_data[gear_data["Activity"] != "Collection"]
        gear_data = map_columns(gear_data, gear_mapping)

        # Handle longitude/latitude splitting
        if "dep_lon" in gear_data.columns:
            # Check if values contain spaces (indicating combined lat/long)
            needs_splitting = gear_data['dep_lon'].astype(str).str.contains(' ').any()

            if needs_splitting:
                try:
                    gear_data[['dep_lat', 'dep_lon']] = \
                        gear_data['dep_lon'].astype(str).str.split(' ', expand=True)
                    print("Split dep_lon column into dep_lat and dep_lon")
                except Exception as e:
                    print(f"Error splitting longitude column: {e}")

        valid_gear_columns = gear_dict_df["sharks_CPUE"].tolist()
        # gear_data = gear_data[valid_gear_columns]

        # Add missing 'comments' column if it doesn't exist
        if 'comments' in valid_gear_columns and 'comments' not in gear_data.columns:
            print(f"Adding missing 'comments' column to gear_data")
            gear_data['comments'] = ''  # Add empty comments column

        # Filter columns but handle missing ones
        valid_columns_present = [col for col in valid_gear_columns if col in gear_data.columns]
        gear_data = gear_data[valid_columns_present]

        cpue_data = capture_data.merge(gear_data, on="dep_drumline_number", how="outer")
        # Method 1: Using pandas' fillna function
        # cpue_data['col_date_time'] = cpue_data['col_date_time_y'].fillna(cpue_data['col_date_time_x'])

        # Method 2: Using pandas' combine_first function
        cpue_data['col_date_time'] = cpue_data['col_date_time_y'].combine_first(cpue_data['col_date_time_x'])
        cpue_data['dep_type'] = cpue_data['dep_type_y'].combine_first(cpue_data['dep_type_x'])
        cpue_data = cpue_data.drop(columns=["col_date_time_y", "col_date_time_x", "dep_type_y", "dep_type_x"])
        cpue_data["event"] = cpue_data["visual_tag"]

        # # Format tag_date before saving
        # if 'dep_date_time' in cpue_data.columns:
        #     try:
        #         # First ensure tag_date is in datetime format
        #         if not pd.api.types.is_datetime64_any_dtype(cpue_data['dep_date_time']):
        #             # If tag_date is still a string, convert it to datetime
        #             cpue_data['dep_date_time'] = cpue_data['dep_date_time'].apply(parse_datetime_safely)
        #
        #         # Format to dd/mm/yyyy hh:mm
        #         cpue_data['dep_date_time'] = cpue_data['dep_date_time'].dt.strftime('%d/%m/%Y %H:%M')
        #         print("dep_date_time column formatted as dd/mm/yyyy hh:mm")
        #     except Exception as e:
        #         print(f"Warning: Could not format dep_date_time column: {e}")
        #
        # if 'col_date_time' in cpue_data.columns:
        #     try:
        #         # First ensure tag_date is in datetime format
        #         if not pd.api.types.is_datetime64_any_dtype(cpue_data['col_date_time']):
        #             # If tag_date is still a string, convert it to datetime
        #             cpue_data['col_date_time'] = cpue_data['col_date_time'].apply(parse_datetime_safely)
        #
        #         # Format to dd/mm/yyyy hh:mm
        #         cpue_data['col_date_time'] = cpue_data['col_date_time'].dt.strftime('%d/%m/%Y %H:%M')
        #         print("col_date_time column formatted as dd/mm/yyyy hh:mm")
        #     except Exception as e:
        #         print(f"Warning: Could not format col_date_time column: {e}")

                # Save to file
        if not cpue_data.empty:
            try:
                columns_list = cpue_list_df["sharks_CPUE"].to_list()
                # Assuming your columns are named 'Lat' and 'Lng'
                cpue_data["site"] = cpue_data.apply(
                    lambda row: determine_site_from_coordinates(
                        float(row["dep_lat"]),
                        float(row["dep_lon"])
                    ),
                    axis=1
                )
                cpue_data = cpue_data[columns_list]
                print(f"Successfully created {len(cpue_data)} rows")
                final_cpue_data = pd.concat([final_cpue_data, cpue_data], ignore_index=True)
            except Exception as e:
                print(f"Error creating cpue_data: {e}")
        else:
            print("No data to save")

    return final_cpue_data


def map_columns(df, column_mapping, position_mapping_fallback=True):
    """
    Map DataFrame columns according to a dictionary.

    Args:
        df (pd.DataFrame): DataFrame to map
        column_mapping (dict): Dictionary mapping original column names to new names
        position_mapping_fallback (bool): Whether to try position-based mapping if name mapping fails

    Returns:
        pd.DataFrame: DataFrame with mapped columns
    """
    # First try mapping by column name
    found_columns = [col for col in column_mapping.keys() if col in df.columns]
    print(f"Found {len(found_columns)} matching columns out of {len(column_mapping)} expected")

    # If less than 50% of columns are found, try position-based mapping
    if position_mapping_fallback and len(found_columns) < len(column_mapping) * 0.5:
        print("Using position-based column mapping due to low match rate")

        # Get expected columns in order
        expected_cols = list(column_mapping.keys())
        expected_renamed = list(column_mapping.values())

        # Create position-based mapping for all columns
        position_mapping = {}
        for i, col in enumerate(df.columns):
            if i < len(expected_renamed):
                position_mapping[col] = expected_renamed[i]

        # Apply position mapping
        mapped_df = df.copy()
        mapped_df.columns = [position_mapping.get(col, col) for col in df.columns]
        return mapped_df
    else:
        # Apply standard mapping
        return df.rename(columns=column_mapping)


def parse_datetime_safely(value, day_first=True):
    """
    Safely parse a datetime string with day-first format preference.

    Args:
        value: Value to parse
        day_first: Whether to interpret dates as day first (European style)

    Returns:
        datetime or NaT: Parsed datetime or NaT if parsing fails
    """
    if pd.isna(value):
        return pd.NaT

    try:
        # Use dayfirst parameter to specify day/month/year preference
        return pd.to_datetime(value, dayfirst=day_first)
    except:
        return pd.NaT


def replace_numbers_with_yes(value):
    """
    Replace numeric values with "Yes".

    Args:
        value: Value to check

    Returns:
        "Yes" if numeric, original value otherwise
    """
    if pd.isna(value):
        return value

    try:
        float(value)
        return "Yes"
    except (ValueError, TypeError):
        return value


def deduplicate_data(df, key_column):
    """
    Remove duplicate rows based on a key column.

    Args:
        df (pd.DataFrame): DataFrame to deduplicate
        key_column (str): Column to use as the unique key

    Returns:
        pd.DataFrame: Deduplicated DataFrame
    """
    if key_column not in df.columns:
        print(f"Warning: Key column '{key_column}' not found for deduplication")
        return df

    # Count duplicate rows
    duplicates = df.duplicated(subset=[key_column], keep=False)
    duplicate_count = duplicates.sum()

    if duplicate_count == 0:
        print("No duplicates found")
        return df

    print(f"Found {duplicate_count} duplicate rows based on {key_column}")

    # Keep the most complete record for each key
    def count_non_null(row):
        return row.count()

    # Group by key and find the row with the most non-null values for each group
    grouped = df.groupby(key_column)
    indices = []

    for name, group in grouped:
        group['non_null_count'] = group.apply(count_non_null, axis=1)
        indices.append(group['non_null_count'].idxmax())

    deduped_df = df.loc[indices].copy()

    # Remove the temporary count column
    if 'non_null_count' in deduped_df.columns:
        deduped_df = deduped_df.drop(columns=['non_null_count'])

    print(f"After deduplication: {len(deduped_df)} unique rows")
    return deduped_df


def determine_site_from_coordinates(lat, lon):
    """
    Determine site name based on latitude and longitude coordinates.

    Parameters:
    lat (float): Latitude value
    lon (float): Longitude value

    Returns:
    str: Site name ('Hadera', 'Ashdod Power Plant', 'Ashkelon', 'Michmoret', or 'Unknown')
    """
    # Define approximate coordinate ranges for each site
    # Hadera coordinates (approximate boundaries)
    hadera_lat_range = (32.45, 32.48)
    hadera_lon_range = (34.87, 34.89)

    # Ashdod Power Plant coordinates (approximate boundaries)
    ashdod_lat_range = (31.75, 31.80)
    ashdod_lon_range = (34.60, 34.65)

    # Ashkelon coordinates (approximate boundaries)
    ashkelon_lat_range = (31.65, 31.70)
    ashkelon_lon_range = (34.54, 34.58)

    # Michmoret coordinates (approximate boundaries)
    michmoret_lat_range = (32.40, 32.44)
    michmoret_lon_range = (34.86, 34.89)

    # Check which site the coordinates belong to
    if hadera_lat_range[0] <= lat <= hadera_lat_range[1] and hadera_lon_range[0] <= lon <= hadera_lon_range[1]:
        return 'Hadera'
    elif ashdod_lat_range[0] <= lat <= ashdod_lat_range[1] and ashdod_lon_range[0] <= lon <= ashdod_lon_range[1]:
        return 'Ashdod Power Plant'
    elif ashkelon_lat_range[0] <= lat <= ashkelon_lat_range[1] and ashkelon_lon_range[0] <= lon <= ashkelon_lon_range[
        1]:
        return 'Ashkelon'
    elif michmoret_lat_range[0] <= lat <= michmoret_lat_range[1] and michmoret_lon_range[0] <= lon <= \
            michmoret_lon_range[1]:
        return 'Michmoret'
    else:
        return 'Unknown'


# Example usage:
# site = determine_site_from_coordinates(32.46, 34.88)  # Should return 'Hadera'
# site = determine_site_from_coordinates(31.78, 34.63)  # Should return 'Ashdod Power Plant'

def create_tag_dataframe():
    """
    Create a pandas DataFrame with tag data hardcoded.

    Returns:
        pd.DataFrame: DataFrame containing the tag data
    """

    # The tag data as a multi-line string
    tag_data = """visual_tag	dep_date_time	col_date_time	site	dep_lat	dep_lon	dep_type	dep_drumline_number	dep_longline_letter	dep_longline_number	event	dep_bait	dep_bait_on_off	dep_longline_bottom_surface	dep_bottom_salinity	dep_bottom_temperature	dep_sea_surface_salinity	dep_sea_surface_temperature	dep_handline_anchoring_motoring	comments
1	17/01/2017 0:00	17/01/2017 0:00	NA	NA	NA	NA	NA	NA	NA		NA	NA	NA	NA	NA	NA	NA	NA	
1	23/03/2017 0:00	23/03/2017 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
2	17/01/2017 0:00	17/01/2017 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
3	24/01/2017 0:00	24/01/2017 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
4	20/02/2017 0:00	20/02/2017 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
5	21/02/2017 0:00	21/02/2017 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
6	23/02/2017 0:00	23/02/2017 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
7	23/02/2017 0:00	23/02/2017 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
8	28/02/2017 0:00	28/02/2017 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
9	06/03/2017 0:00	06/03/2017 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
10	08/03/2017 0:00	08/03/2017 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
11	08/03/2017 0:00	08/03/2017 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
12	28/03/2017 0:00	28/03/2017 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
13	06/04/2017 0:00	06/04/2017 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
14	06/04/2017 0:00	06/04/2017 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
15	27/11/2017 0:00	27/11/2017 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
16	12/12/2017 0:00	12/12/2017 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
17	19/12/2017 0:00	19/12/2017 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
18	27/12/2017 0:00	27/12/2017 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
19	27/12/2017 0:00	27/12/2017 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
20	09/01/2018 0:00	09/01/2018 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
21	01/02/2018 0:00	01/02/2018 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
22	06/02/2018 0:00	06/02/2018 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
23	12/03/2018 0:00	12/03/2018 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
24	12/03/2018 0:00	12/03/2018 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
25	14/03/2018 0:00	14/03/2018 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
26	28/03/2018 0:00	28/03/2018 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
27	28/03/2018 0:00	28/03/2018 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
28	02/04/2018 0:00	02/04/2018 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
29	02/04/2018 0:00	02/04/2018 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
30	01/05/2018 0:00	01/05/2018 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
31	27/11/2018 0:00	27/11/2018 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
32	16/12/2018 0:00	16/12/2018 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
33	21/01/2019 0:00	21/01/2019 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
34	20/02/2019 0:00	20/02/2019 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
35	20/02/2019 0:00	20/02/2019 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
36	24/02/2019 0:00	24/02/2019 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
37	25/02/2019 0:00	25/02/2019 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
38	11/03/2019 0:00	11/03/2019 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
39	11/03/2019 0:00	11/03/2019 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
40	20/03/2019 0:00	20/03/2019 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
41	23/03/2016 0:00	22/03/2016 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
42	24/04/2019 0:00	24/04/2019 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
43	23/03/2016 0:00	23/03/2016 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
44	23/03/2016 0:00	23/03/2016 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
45	25/02/2016 0:00	25/02/2016 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
46	25/02/2016 0:00	25/02/2016 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
47	25/02/2016 0:00	25/02/2016 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
48	17/02/2016 0:00	17/02/2016 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
49	24/04/2019 0:00	24/04/2019 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
49	29/04/2019 0:00	29/04/2019 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
50	24/04/2019 0:00	24/04/2019 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
51	25/11/2019 0:00	25/11/2019 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
52	14/01/2020 0:00	14/01/2020 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
53	15/01/2020 0:00	15/01/2020 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
53	05/02/2020 0:00	05/02/2020 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
54	17/02/2020 0:00	17/02/2020 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
55	17/02/2020 0:00	17/02/2020 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
56	04/03/2020 0:00	04/03/2020 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
57	29/12/2020 0:00	29/12/2020 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
58	07/01/2021 0:00	07/01/2021 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
59	07/01/2021 0:00	07/01/2021 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
60	25/01/2021 0:00	25/01/2021 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
61	25/01/2021 0:00	25/01/2021 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
62	08/02/2021 0:00	08/02/2021 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
63	15/02/2021 0:00	15/02/2021 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
64	10/03/2021 0:00	10/03/2021 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
65	15/03/2021 0:00	15/03/2021 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
66	03/05/2021 0:00	03/05/2021 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	
67	03/05/2021 0:00	03/05/2021 0:00	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA"""

    try:
        # Use pandas to read the string as a tab-delimited file
        df = pd.read_csv(io.StringIO(tag_data), sep='\t')
        print(f"Successfully created tag DataFrame with {df.shape[0]} rows and {df.shape[1]} columns")

        # Replace 'NA' strings with actual NaN values
        df = df.replace('NA', np.nan)

        # # Convert date columns to datetime
        # date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        # for col in date_cols:
        #     df[col] = pd.to_datetime(df[col], errors='coerce')

        return df

    except Exception as e:
        print(f"Error creating tag DataFrame: {e}")
        return pd.DataFrame()


def main(args):
    """
    Main function to process shark cpue data.

    Args:
        args: Command line arguments
    """
    delphis_dir = args.delphis_dir
    capture_dict_file = args.capture_dict
    gear_dict_file = args.gear_dict
    cpue_list_file = args.cpue_list
    output_file = args.output_file
    readonly = args.readonly

    print(f"Starting processing with parameters:")
    print(f"  Delphis directory: {delphis_dir}")
    print(f"  capture dictionary: {capture_dict_file}")
    print(f"  gear dictionary: {gear_dict_file}")
    print(f"  cpue list: {cpue_list_file}")
    print(f"  Output file: {output_file}")
    print(f"  Readonly mode: {readonly}")

    # Load mapping files
    try:
        capture_dict_df = pd.read_csv(capture_dict_file)
        gear_dict_df = pd.read_csv(gear_dict_file)
        cpue_list_df = pd.read_csv(cpue_list_file)

        # Create mapping dictionary
        capture_mapping = dict(zip(capture_dict_df.iloc[:, 0], capture_dict_df.iloc[:, 1]))
        gear_mapping = dict(zip(gear_dict_df.iloc[:, 0], gear_dict_df.iloc[:, 1]))
    except Exception as e:
        print(f"Error loading mapping files: {e}")
        return

    # Process directory and extract data
    final_cpue_data = process_directory(delphis_dir, None, None, capture_dict_df, gear_dict_df, capture_mapping, gear_mapping, cpue_list_df)

    # Create the tag dataframe directly
    tag_data = create_tag_dataframe()

    if not tag_data.empty:
        # Merge with the CPUE data based on visual_tag
        final_cpue_data = pd.concat([final_cpue_data, tag_data], axis=0, ignore_index=True)
        print(f"Merged tag data with CPUE data")
        # Apply the function to your date columns
        final_cpue_data["dep_date_time"] = final_cpue_data["dep_date_time"].apply(
            lambda x: parse_datetime_safely(x, day_first=True))
        final_cpue_data["dep_date_time"] = final_cpue_data["dep_date_time"].dt.strftime('%d/%m/%Y %H:%M')

        final_cpue_data["col_date_time"] = final_cpue_data["col_date_time"].apply(
            lambda x: parse_datetime_safely(x, day_first=True))
        final_cpue_data["col_date_time"] = final_cpue_data["col_date_time"].dt.strftime('%d/%m/%Y %H:%M')
    try:
        final_cpue_data.to_csv(output_file, index=False, float_format='%.0f')
        print(f"Successfully saved {len(final_cpue_data)} rows to {output_file}")
    except Exception as e:
        print(f"Error saving output file: {e}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process shark cpue data")
    parser.add_argument("delphis_dir", type=str, help="Path to directory with Delphis files")
    parser.add_argument("capture_dict", type=str, help="Path to capture column mapping dictionary file")
    parser.add_argument("gear_dict", type=str, help="Path to gear column mapping dictionary file")
    parser.add_argument("cpue_list", type=str, help="Path to output column list file")
    parser.add_argument("output_file", type=str, help="Path to output file")
    parser.add_argument("--readonly", action="store_true", help="Do not modify input files")

    args = parser.parse_args(sys.argv[1:])
    main(args)