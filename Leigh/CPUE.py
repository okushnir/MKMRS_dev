#!/usr/bin/env python
"""
@Created by: Biodaat; Oded Kushnir
@Enhanced by: [Your Name]

A script for processing shark CPUE (Catch Per Unit Effort) data from multiple CSV files.
Adapted from the Tagging.py script.
"""
import argparse
import glob
import os
import numpy as np
import pandas as pd
import re
import sys


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
    Also handles cases where there might not be sections but just CSV data.

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

    # If no sections found but file seems to be a CSV, create a default section
    if not sections and len(content_lines) > 1:
        # Check if the first line looks like CSV headers (contains commas)
        if ',' in content_lines[0]:
            print("No sections found, but file appears to be a CSV. Creating default CPUE section.")
            sections["CPUE"] = content_lines

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
        target_section (str): Section to extract (e.g., "Megalodon CPUE")
        filter_column (str): Column to filter non-empty values by

    Returns:
        dict: Dictionary of DataFrames, with section names as keys
    """
    # Skip files that don't likely contain CPUE data
    file_name = os.path.basename(file_path).lower()
    if file_name in ['cpue_dict.csv', 'cpue_list.csv']:
        print(f"Skipping metadata file: {file_path}")
        return {}
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


def process_directory(directory, target_section, filter_column=None):
    """
    Process all files in a directory.

    Args:
        directory (str): Directory path containing files to process
        target_section (str): Section to extract
        filter_column (str): Column to filter by

    Returns:
        pd.DataFrame: Combined DataFrame from all files
    """
    # First check if the directory contains our expected metadata files
    # If so, skip them when processing
    metadata_files = ['cpue_dict.csv', 'cpue_list.csv']
    skipfiles = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower() in metadata_files:
                skipfiles.append(os.path.join(root, file))
                print(f"Will skip metadata file: {file}")

    # If the directory only contains one CSV file that's not in our skipfiles,
    # we can try to process it directly
    csv_files = [f for f in glob.glob(os.path.join(directory, '*.csv')) if f not in skipfiles]
    if len(csv_files) == 1:
        print(f"Found a single CSV file: {csv_files[0]}")
        # Try to read it directly with pandas
        try:
            df = pd.read_csv(csv_files[0])
            df['source_file'] = os.path.basename(csv_files[0])
            print(f"Successfully read CSV file directly: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            print(f"Error reading CSV directly: {e}")
            # Continue with normal processing
            pass
    combined_data = []
    processed_files = 0

    # List all files in the directory
    file_paths = glob.glob(os.path.join(directory, '*'))
    print(f"Found {len(file_paths)} files in directory")

    for file_path in file_paths:
        # Skip directories
        if os.path.isdir(file_path):
            continue

        # Process the file
        dataframes = process_file(file_path, target_section, filter_column)

        # Extract the target section if it exists
        if target_section in dataframes:
            df = dataframes[target_section]
            combined_data.append(df)
            processed_files += 1

            # Try to add boat information if available
            try:
                if "Summary" in dataframes and isinstance(dataframes["Summary"], pd.DataFrame):
                    # This is a simplistic approach - adjust based on actual data structure
                    if "Boat" in dataframes["Summary"].columns:
                        boat_value = dataframes["Summary"]["Boat"].iloc[0]
                        df["Boat"] = boat_value
                elif "Summery" in dataframes and isinstance(dataframes["Summery"], pd.DataFrame):
                    # Check alternate spelling
                    if "Boat" in dataframes["Summery"].columns:
                        boat_value = dataframes["Summery"]["Boat"].iloc[0]
                        df["Boat"] = boat_value
            except Exception as e:
                print(f"  Could not extract boat information: {e}")

    # Combine all the data
    if combined_data:
        try:
            combined_df = pd.concat(combined_data, ignore_index=True)
            print(
                f"Combined data from {processed_files} files: {combined_df.shape[0]} rows, {combined_df.shape[1]} columns")
            return combined_df
        except Exception as e:
            print(f"Error combining data: {e}")

    print("No data found matching criteria")
    return pd.DataFrame()


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
               '%d/%m/%Y %H:%M', '%Y-%m-%d %H:%M', '%d-%m-%Y %H:%M',
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


def split_lat_lng(df, col_name):
    """
    Split a column containing "lat lng" format into separate latitude and longitude columns.

    Args:
        df (pd.DataFrame): DataFrame containing the column
        col_name (str): Name of the column to split

    Returns:
        pd.DataFrame: Modified DataFrame with split columns
    """
    if col_name not in df.columns:
        print(f"Warning: Column {col_name} not found for lat/lng splitting")
        return df

    result_df = df.copy()

    # Check if values contain spaces (indicating combined lat/long)
    needs_splitting = result_df[col_name].astype(str).str.contains(' ').any()

    if needs_splitting:
        try:
            # Split the column by space
            lat_lng_split = result_df[col_name].astype(str).str.split(' ', expand=True)

            # Assign to dep_lat and dep_lon
            result_df['dep_lat'] = pd.to_numeric(lat_lng_split[0], errors='coerce')
            result_df['dep_lon'] = pd.to_numeric(lat_lng_split[1], errors='coerce')

            print(f"Successfully split {col_name} into dep_lat and dep_lon columns")
        except Exception as e:
            print(f"Error splitting lat/lng column: {e}")

    return result_df


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


def main(args):
    """
    Main function to process shark CPUE data.

    Args:
        args: Command line arguments
    """
    delphis_dir = args.delphis_dir
    cpue_dict_file = args.cpue_dict
    cpue_list_file = args.cpue_list
    output_file = args.output_file
    readonly = args.readonly

    # Check if delphis_dir is actually a file instead of a directory
    if os.path.isfile(delphis_dir):
        print(f"Warning: {delphis_dir} is a file, not a directory.")
        print("Processing this file directly instead of looking for files in a directory.")
        # Create a temporary directory to hold this file
        dirpath = "temp_dir_for_file"
        os.makedirs(dirpath, exist_ok=True)
        # Copy the file to the temporary directory
        import shutil
        shutil.copy2(delphis_dir, os.path.join(dirpath, os.path.basename(delphis_dir)))
        # Update delphis_dir to point to this directory
        delphis_dir = dirpath

    print(f"Starting processing with parameters:")
    print(f"  Delphis directory: {delphis_dir}")
    print(f"  CPUE dictionary: {cpue_dict_file}")
    print(f"  CPUE list: {cpue_list_file}")
    print(f"  Output file: {output_file}")
    print(f"  Readonly mode: {readonly}")

    # Load mapping files
    try:
        cpue_dict_df = pd.read_csv(cpue_dict_file)
        cpue_list_df = pd.read_csv(cpue_list_file)

        # Create mapping dictionary
        column_mapping = dict(zip(cpue_dict_df.iloc[:, 0], cpue_dict_df.iloc[:, 1]))
    except Exception as e:
        print(f"Error loading mapping files: {e}")
        return

    # Process directory and extract data - try multiple possible section names
    section_names = ["Megalodon CPUE", "CPUE", "Megalodon Effort", "Effort"]

    cpue_data = pd.DataFrame()
    for section_name in section_names:
        print(f"Trying to find section: {section_name}")
        data = process_directory(delphis_dir, section_name)
        if not data.empty:
            cpue_data = data
            print(f"Found data in section: {section_name}")
            break

    if cpue_data.empty:
        print("No CPUE data found in any of the expected sections. Exiting.")
        return

    # Map columns to expected names
    cpue_data = map_columns(cpue_data, column_mapping)

    # Get expected columns from cpue list
    expected_columns = list(cpue_list_df.iloc[:, 0])

    # Add source_file for debugging if not in expected columns
    if 'source_file' in cpue_data.columns and 'source_file' not in expected_columns:
        expected_columns.append('source_file')

    # Process Lat Lng column if it exists
    if 'Lat Lng' in cpue_data.columns:
        cpue_data = split_lat_lng(cpue_data, 'Lat Lng')

    # Process datetime columns if they exist
    datetime_columns = ['dep_date_time', 'col_date_time']
    for col in datetime_columns:
        if col in cpue_data.columns:
            cpue_data[col] = cpue_data[col].replace(['NaN', 'None', ''], None)
            # Convert to datetime format
            cpue_data[col] = cpue_data[col].apply(parse_datetime_safely)
            # Format to dd/mm/yyyy hh:mm
            cpue_data[col] = cpue_data[col].dt.strftime('%d/%m/%Y %H:%M')
            print(f"{col} column formatted as dd/mm/yyyy hh:mm")

    # Process visual_tag if available
    if "visual_tag" in cpue_data.columns:
        # Convert visual_tag to integer if possible
        try:
            cpue_data["visual_tag"] = pd.to_numeric(cpue_data["visual_tag"], errors='coerce')
            cpue_data["visual_tag"] = cpue_data["visual_tag"].fillna(0).astype(int)
            print("Successfully converted visual_tag to integer")
        except Exception as e:
            print(f"Warning: Could not convert visual_tag to integer: {e}")

        # Remove duplicates based on visual_tag if it exists
        cpue_data = deduplicate_data(cpue_data, "visual_tag")

        # Sort by visual_tag
        cpue_data = cpue_data.sort_values(by="visual_tag")

    # Ensure all expected columns exist in the dataframe
    for col in expected_columns:
        if col not in cpue_data.columns:
            cpue_data[col] = None
            print(f"Added missing column: {col}")

    # Select only the columns in the expected order
    available_output_cols = [col for col in expected_columns if col in cpue_data.columns]

    if len(available_output_cols) < len(expected_columns):
        missing = set(expected_columns) - set(available_output_cols)
        print(f"Warning: {len(missing)} columns missing in final output: {missing}")

    final_table = cpue_data[available_output_cols].copy()

    # Save to file
    if not final_table.empty:
        try:
            final_table.to_csv(output_file, index=False, float_format='%.6f')
            print(f"Successfully saved {len(final_table)} rows to {output_file}")
        except Exception as e:
            print(f"Error saving output file: {e}")
    else:
        print("No data to save")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process shark CPUE data")
    parser.add_argument("delphis_dir", type=str, help="Path to directory with Delphis files")
    parser.add_argument("cpue_dict", type=str, help="Path to column mapping dictionary file")
    parser.add_argument("cpue_list", type=str, help="Path to output column list file")
    parser.add_argument("output_file", type=str, help="Path to output file")
    parser.add_argument("--readonly", action="store_true", help="Do not modify input files")

    args = parser.parse_args(sys.argv[1:])
    main(args)