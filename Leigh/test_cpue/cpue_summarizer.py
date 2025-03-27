#!/usr/bin/env python
"""
@Created by: Built on Biodaat & Oded Kushnir's work
@Enhanced by: Claude 3.7 Sonnet

A script for summarizing shark CPUE data from multiple files in a directory.
This is a standalone script that combines all functionality without relying on imports.
"""
import argparse
import glob
import os
import numpy as np
import pandas as pd
import re
import sys
from datetime import datetime


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


def parse_datetime_safely(value):
    """
    Safely parse a datetime string with multiple format attempts.

    Args:
        value: Value to parse

    Returns:
        datetime or NaT: Parsed datetime or NaT if parsing fails
    """
    if pd.isna(value):
        return pd.NaT

    formats = ['%d/%m/%Y %H:%M:%S', '%Y-%m-%d %H:%M:%S', '%d-%m-%Y %H:%M:%S',
               '%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y']

    for fmt in formats:
        try:
            return pd.to_datetime(value, format=fmt)
        except (ValueError, TypeError):
            continue

    # If none of the formats work, try pandas' inference
    try:
        return pd.to_datetime(value)
    except:
        return pd.NaT


def determine_site_from_coordinates(lat, lon):
    """
    Determine site name based on latitude and longitude coordinates.

    Parameters:
    lat (float): Latitude value
    lon (float): Longitude value

    Returns:
    str: Site name ('Hadera', 'Ashdod Power Plant', or 'Unknown')
    """
    # Handle invalid coordinates
    try:
        lat = float(lat)
        lon = float(lon)
    except (ValueError, TypeError):
        return 'Unknown'

    # Define approximate coordinate ranges for each site
    # Hadera coordinates (approximate boundaries)
    hadera_lat_range = (32.45, 32.48)
    hadera_lon_range = (34.87, 34.89)

    # Ashdod Power Plant coordinates (approximate boundaries)
    ashdod_lat_range = (31.75, 31.80)
    ashdod_lon_range = (34.60, 34.65)

    # Check which site the coordinates belong to
    if hadera_lat_range[0] <= lat <= hadera_lat_range[1] and hadera_lon_range[0] <= lon <= hadera_lon_range[1]:
        return 'Hadera'
    elif ashdod_lat_range[0] <= lat <= ashdod_lat_range[1] and ashdod_lon_range[0] <= lon <= ashdod_lon_range[1]:
        return 'Ashdod Power Plant'
    else:
        return 'Unknown'


def process_all_files_in_directory(directory, target_section=None, filter_column=None):
    """
    Process all files in a directory and concatenate the results.

    Args:
        directory (str): Directory path containing files to process
        target_section (str): Section to extract
        filter_column (str): Column to filter by

    Returns:
        dict: Dictionary of concatenated DataFrames, with section names as keys
    """
    combined_sections = {}
    processed_files = 0

    # List all files in the directory
    file_paths = glob.glob(os.path.join(directory, '*'))
    print(f"Found {len(file_paths)} files in directory")

    for file_path in file_paths:
        # Skip directories
        if os.path.isdir(file_path):
            continue

        # Skip non-data files (e.g., mapping files, output files)
        if file_path.endswith('.csv') and any(
                x in file_path.lower() for x in ['dict', 'mapping', 'list', 'output', 'summary']):
            print(f"Skipping mapping/output file: {file_path}")
            continue

        # Process the file
        try:
            dataframes = process_file(file_path, target_section, filter_column)

            if dataframes:
                processed_files += 1

                # Add each section to the combined results
                for section_name, df in dataframes.items():
                    if section_name not in combined_sections:
                        combined_sections[section_name] = df
                    else:
                        combined_sections[section_name] = pd.concat([combined_sections[section_name], df],
                                                                    ignore_index=True)

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    print(f"Successfully processed {processed_files} files")

    # Report summary stats for each section
    for section_name, df in combined_sections.items():
        print(f"Combined {section_name}: {df.shape[0]} rows, {df.shape[1]} columns")

    return combined_sections


def generate_summary_report(cpue_data, output_summary_file):
    """
    Generate a summary report from the CPUE data.

    Args:
        cpue_data (pd.DataFrame): CPUE data to summarize
        output_summary_file (str): Path to save the summary report
    """
    # Create a copy to avoid modifying the original
    summary_df = cpue_data.copy()

    # Convert datetime columns to datetime type if they're not already
    date_cols = ['dep_date_time', 'col_date_time']
    for col in date_cols:
        if col in summary_df.columns:
            summary_df[col] = pd.to_datetime(summary_df[col], format='%d/%m/%Y %H:%M', errors='coerce')

    # Calculate soak time in hours (difference between collection and deployment times)
    if 'dep_date_time' in summary_df.columns and 'col_date_time' in summary_df.columns:
        summary_df['soak_time_hours'] = (summary_df['col_date_time'] - summary_df[
            'dep_date_time']).dt.total_seconds() / 3600

    # Summary by site
    site_summary = summary_df.groupby('site').agg(
        total_deployments=pd.NamedAgg(column='dep_drumline_number', aggfunc='count'),
        unique_capture_events=pd.NamedAgg(column='event', aggfunc=lambda x: x.notna().sum()),
        avg_soak_time=pd.NamedAgg(column='soak_time_hours', aggfunc='mean'),
    )

    # Summary by month/year
    if 'dep_date_time' in summary_df.columns:
        summary_df['year_month'] = summary_df['dep_date_time'].dt.strftime('%Y-%m')
        time_summary = summary_df.groupby('year_month').agg(
            total_deployments=pd.NamedAgg(column='dep_drumline_number', aggfunc='count'),
            unique_capture_events=pd.NamedAgg(column='event', aggfunc=lambda x: x.notna().sum()),
            avg_soak_time=pd.NamedAgg(column='soak_time_hours', aggfunc='mean'),
        )
    else:
        time_summary = pd.DataFrame()

    # Species summary if species column exists
    species_col = next((col for col in summary_df.columns if 'species' in col.lower()), None)
    if species_col:
        species_summary = summary_df.groupby([species_col, 'site']).size().reset_index(name='count')
    else:
        species_summary = pd.DataFrame()

    # Write summary to file
    with open(output_summary_file, 'w') as f:
        f.write(f"CPUE Summary Report\n")
        f.write(f"Generated on: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n\n")

        f.write(f"Total records processed: {len(summary_df)}\n")
        f.write(
            f"Date range: {summary_df['dep_date_time'].min().strftime('%d/%m/%Y')} to {summary_df['dep_date_time'].max().strftime('%d/%m/%Y')}\n\n")

        f.write("Summary by Site:\n")
        f.write(site_summary.to_string())
        f.write("\n\n")

        if not time_summary.empty:
            f.write("Summary by Month:\n")
            f.write(time_summary.to_string())
            f.write("\n\n")

        if not species_summary.empty:
            f.write("Summary by Species:\n")
            f.write(species_summary.to_string())
            f.write("\n\n")

        # Calculate CPUE per site
        f.write("CPUE by Site (Catch Per Unit Effort):\n")
        for site, group in summary_df.groupby('site'):
            total_effort_hours = group['soak_time_hours'].sum()
            total_catches = group['event'].notna().sum()
            cpue = total_catches / total_effort_hours if total_effort_hours > 0 else 0
            f.write(f"{site}: {total_catches} catches / {total_effort_hours:.2f} hours = {cpue:.4f} catches per hour\n")

    print(f"Summary report saved to {output_summary_file}")

    # Also save site summary as CSV for easy integration with other tools
    site_summary.reset_index().to_csv(output_summary_file.replace('.txt', '_site.csv'), index=False)

    if not time_summary.empty:
        time_summary.reset_index().to_csv(output_summary_file.replace('.txt', '_time.csv'), index=False)

    if not species_summary.empty:
        species_summary.to_csv(output_summary_file.replace('.txt', '_species.csv'), index=False)


def main(args):
    """
    Main function to process shark CPUE data and generate summary.

    Args:
        args: Command line arguments
    """
    delphis_dir = args.delphis_dir
    capture_dict_file = args.capture_dict
    gear_dict_file = args.gear_dict
    cpue_list_file = args.cpue_list
    output_file = args.output_file
    output_summary_file = args.output_summary if args.output_summary else output_file.replace('.csv', '_summary.txt')
    readonly = args.readonly

    print(f"Starting processing with parameters:")
    print(f"  Delphis directory: {delphis_dir}")
    print(f"  Capture dictionary: {capture_dict_file}")
    print(f"  Gear dictionary: {gear_dict_file}")
    print(f"  CPUE list: {cpue_list_file}")
    print(f"  Output file: {output_file}")
    print(f"  Summary file: {output_summary_file}")
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

    # Process all files in directory and extract data
    delphis_data = process_all_files_in_directory(delphis_dir, None, None)

    if "Megalodon Capture" not in delphis_data or "Megalodon Gear" not in delphis_data:
        print("Required sections not found in the processed files. Exiting.")
        return

    capture_data = delphis_data["Megalodon Capture"]
    gear_data = delphis_data["Megalodon Gear"]

    # Filter the DataFrame to keep only non-column_ columns
    columns_to_keep = [col for col in gear_data.columns if not col.startswith("column_") or not col[7:].isdigit()]
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
    gear_data = gear_data[valid_gear_columns]

    cpue_data = capture_data.merge(gear_data, on="dep_drumline_number", how="outer")

    # Combine columns that exist in both dataframes
    cpue_data['col_date_time'] = cpue_data['col_date_time_y'].combine_first(cpue_data['col_date_time_x'])
    cpue_data['dep_type'] = cpue_data['dep_type_y'].combine_first(cpue_data['dep_type_x'])
    cpue_data = cpue_data.drop(columns=["col_date_time_y", "col_date_time_x", "dep_type_y", "dep_type_x"])
    cpue_data["event"] = cpue_data["visual_tag"]

    # Format datetime columns
    for date_col in ['dep_date_time', 'col_date_time']:
        if date_col in cpue_data.columns:
            try:
                # Ensure column is in datetime format
                if not pd.api.types.is_datetime64_any_dtype(cpue_data[date_col]):
                    cpue_data[date_col] = cpue_data[date_col].apply(parse_datetime_safely)

                # Format to dd/mm/yyyy hh:mm
                cpue_data[date_col] = cpue_data[date_col].dt.strftime('%d/%m/%Y %H:%M')
                print(f"{date_col} column formatted as dd/mm/yyyy hh:mm")
            except Exception as e:
                print(f"Warning: Could not format {date_col} column: {e}")

    # Save to file
    if not cpue_data.empty:
        try:
            columns_list = cpue_list_df["sharks_CPUE"].to_list()

            # Determine site based on coordinates
            cpue_data["site"] = cpue_data.apply(
                lambda row: determine_site_from_coordinates(
                    row["dep_lat"] if pd.notna(row["dep_lat"]) else 0,
                    row["dep_lon"] if pd.notna(row["dep_lon"]) else 0
                ),
                axis=1
            )

            # Select columns and save the consolidated data
            final_cpue_data = cpue_data[columns_list]
            final_cpue_data.to_csv(output_file, index=False, float_format='%.0f')
            print(f"Successfully saved {len(final_cpue_data)} rows to {output_file}")

            # Generate summary report
            generate_summary_report(final_cpue_data, output_summary_file)

        except Exception as e:
            print(f"Error saving output file: {e}")
    else:
        print("No data to save")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and summarize shark CPUE data from multiple files")
    parser.add_argument("delphis_dir", type=str, help="Path to directory with Delphis files")
    parser.add_argument("capture_dict", type=str, help="Path to capture column mapping dictionary file")
    parser.add_argument("gear_dict", type=str, help="Path to gear column mapping dictionary file")
    parser.add_argument("cpue_list", type=str, help="Path to output column list file")
    parser.add_argument("output_file", type=str, help="Path to output file")
    parser.add_argument("--output_summary", type=str,
                        help="Path to summary output file (default: output_file_summary.txt)")
    parser.add_argument("--readonly", action="store_true", help="Do not modify input files")

    args = parser.parse_args()
    main(args)