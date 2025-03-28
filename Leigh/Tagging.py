#!/usr/bin/env python
"""
@Created by: Biodaat; Oded Kushnir
@Enhanced by: [Your Name]

A script for processing shark tagging data from multiple CSV files.
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


def process_directory(directory, target_section, filter_column):
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


def main(args):
    """
    Main function to process shark tagging data.

    Args:
        args: Command line arguments
    """
    delphis_dir = args.delphis_dir
    transmitters_file = args.transmitters_unique
    tagging_dict_file = args.tagging_dict
    tagging_list_file = args.tagging_list
    output_file = args.output_file
    readonly = args.readonly

    print(f"Starting processing with parameters:")
    print(f"  Delphis directory: {delphis_dir}")
    print(f"  Transmitters file: {transmitters_file}")
    print(f"  Tagging dictionary: {tagging_dict_file}")
    print(f"  Tagging list: {tagging_list_file}")
    print(f"  Output file: {output_file}")
    print(f"  Readonly mode: {readonly}")

    # Load mapping files
    try:
        tagging_dict_df = pd.read_csv(tagging_dict_file)
        tagging_list_df = pd.read_csv(tagging_list_file)

        # Create mapping dictionary
        column_mapping = dict(zip(tagging_dict_df.iloc[:, 0], tagging_dict_df.iloc[:, 1]))
    except Exception as e:
        print(f"Error loading mapping files: {e}")
        return

    # Process directory and extract data
    tagging_data = process_directory(delphis_dir, "Megalodon Capture", "Spagetti")

    if tagging_data.empty:
        print("No tagging data found. Exiting.")
        return

    # Map columns to expected names
    tagging_data = map_columns(tagging_data, column_mapping)

    # Get expected columns from tagging list
    expected_columns = list(tagging_list_df.iloc[:, 0])

    # Add source_file for debugging if not in expected columns
    if 'source_file' in tagging_data.columns and 'source_file' not in expected_columns:
        expected_columns.append('source_file')

    # Keep only columns that exist in the data
    available_columns = [col for col in expected_columns if col in tagging_data.columns]
    if len(available_columns) < len(expected_columns):
        missing = set(expected_columns) - set(available_columns)
        print(f"Warning: Missing {len(missing)} columns in data: {missing}")

    # Select only the available columns
    tagging_data = tagging_data[available_columns]

    # Process tag_date if available
    if 'tag_date' in tagging_data.columns:
        tagging_data['tag_date'] = tagging_data['tag_date'].replace(['NaN', 'None', ''], None)

        # Convert to datetime format
        tagging_data['Datetime'] = tagging_data['tag_date'].apply(parse_datetime_safely)

        # Keep the original tag_date in a temporary column for later formatting
        tagging_data['original_tag_date'] = tagging_data['tag_date']

        # Update tag_date with parsed datetime values
        tagging_data['tag_date'] = tagging_data['Datetime']

        # Extract date only for grouping operations
        tagging_data['Date'] = tagging_data['Datetime'].dt.date

    # Process site column if available
    if 'site' in tagging_data.columns:
        tagging_data['site'] = tagging_data['site'].replace(['NaN', 'None', ''], None)

        # Fill site based on date if possible
        if 'Date' in tagging_data.columns:
            # Group by date and forward-fill site values
            tagging_data = tagging_data.sort_values('Date')
            for date, group in tagging_data.groupby('Date'):
                non_null_sites = group['site'].dropna()
                if not non_null_sites.empty:
                    site_value = non_null_sites.iloc[0]
                    tagging_data.loc[tagging_data['Date'] == date, 'site'] = \
                        tagging_data.loc[tagging_data['Date'] == date, 'site'].fillna(site_value)

        # Map specific site names
        site_mapping = {
            'Dakar-Raphi': 'Hadera',
            'Adva Boston': 'Ashdod Power Plant'
        }

        for old_name, new_name in site_mapping.items():
            tagging_data.loc[tagging_data['site'] == old_name, 'site'] = new_name

    # Process transmitters file
    try:
        # Read and clean transmitters file
        transmitters_content, encoding = read_file_with_encoding(transmitters_file)

        if transmitters_content:
            # Create a clean file to read
            temp_file = "temp_transmitters.csv"
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(''.join(transmitters_content))

            # Read the cleaned file
            transmitters = pd.read_csv(temp_file)

            # Required columns
            required_cols = ["serial_number", "temp_id", "depth_id", "protocol",
                             "est_tag_life_days", "tag_turned_on_date"]

            # Check which required columns exist
            available_cols = [col for col in required_cols if col in transmitters.columns]

            if available_cols:
                # Select only available columns
                transmitters = transmitters[available_cols]

                # Process dates if available
                if "tag_turned_on_date" in transmitters.columns:
                    transmitters["tag_turned_on_date"] = pd.to_datetime(
                        transmitters["tag_turned_on_date"],
                        dayfirst=True,
                        errors='coerce'
                    )

                    if "est_tag_life_days" in transmitters.columns:
                        transmitters["acoustic_tag_removed_date"] = \
                            transmitters["tag_turned_on_date"] + \
                            pd.to_timedelta(transmitters["est_tag_life_days"], unit='days')

                # Rename serial_number to match tagging data
                if "serial_number" in transmitters.columns:
                    transmitters = transmitters.rename(columns={"serial_number": "acoustic_tag_serial"})

                    # Merge with tagging data if possible
                    if "acoustic_tag_serial" in tagging_data.columns:
                        tagging_data = tagging_data.merge(
                            transmitters,
                            on="acoustic_tag_serial",
                            how="left"
                        )

            # Clean up temp file
            if os.path.exists(temp_file) and not readonly:
                os.remove(temp_file)
    except Exception as e:
        print(f"Error processing transmitters file: {e}")

    # Handle special columns
    boolean_columns = ["blood", "skin_mb", "cloaca_mb", "gills_mb",
                       "water_mb", "wound_mb", "catcam", "ultrasound_device"]

    for col in boolean_columns:
        if col in tagging_data.columns:
            tagging_data[col] = tagging_data[col].apply(replace_numbers_with_yes)

    # Handle longitude/latitude splitting
    if "longitude" in tagging_data.columns:
        # Check if values contain spaces (indicating combined lat/long)
        needs_splitting = tagging_data['longitude'].astype(str).str.contains(' ').any()

        if needs_splitting:
            try:
                tagging_data[['latitude', 'longitude']] = \
                    tagging_data['longitude'].astype(str).str.split(' ', expand=True)
                print("Split longitude column into latitude and longitude")
            except Exception as e:
                print(f"Error splitting longitude column: {e}")

    # Filter out NA visual tags
    if "visual_tag" in tagging_data.columns:
        before_count = len(tagging_data)
        tagging_data = tagging_data[tagging_data["visual_tag"].astype(str) != "NA"]
        tagging_data = tagging_data.dropna(subset=["visual_tag"])
        after_count = len(tagging_data)

        if before_count > after_count:
            print(f"Removed {before_count - after_count} rows with NA or empty visual_tag")

        # Convert visual_tag to integer if possible
        try:
            tagging_data["visual_tag"] = tagging_data["visual_tag"].astype(int)
            print("Successfully converted visual_tag to integer")
        except Exception as e:
            print(f"Warning: Could not convert visual_tag to integer: {e}")

    # Remove duplicates
    if "visual_tag" in tagging_data.columns:
        tagging_data = deduplicate_data(tagging_data, "visual_tag")

        # Sort by visual_tag
        tagging_data = tagging_data.sort_values(by="visual_tag")

    # Prepare final output
    output_columns = list(tagging_list_df.iloc[:, 0])
    available_output_cols = [col for col in output_columns if col in tagging_data.columns]

    if len(available_output_cols) < len(output_columns):
        missing = set(output_columns) - set(available_output_cols)
        print(f"Warning: {len(missing)} columns missing in final output: {missing}")

    final_table = tagging_data[available_output_cols].copy()

    # Format tag_date before saving
    if 'tag_date' in final_table.columns:
        try:
            # First ensure tag_date is in datetime format
            if not pd.api.types.is_datetime64_any_dtype(final_table['tag_date']):
                # If tag_date is still a string, convert it to datetime
                final_table['tag_date'] = final_table['tag_date'].apply(parse_datetime_safely)

            # Format to dd/mm/yyyy hh:mm
            final_table['tag_date'] = final_table['tag_date'].dt.strftime('%d/%m/%Y %H:%M')
            print("tag_date column formatted as dd/mm/yyyy hh:mm")
        except Exception as e:
            print(f"Warning: Could not format tag_date column: {e}")

    # Save to file
    if not final_table.empty:
        try:
            final_table.to_csv(output_file, index=False, float_format='%.0f')
            print(f"Successfully saved {len(final_table)} rows to {output_file}")
        except Exception as e:
            print(f"Error saving output file: {e}")
    else:
        print("No data to save")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process shark tagging data")
    parser.add_argument("delphis_dir", type=str, help="Path to directory with Delphis files")
    parser.add_argument("transmitters_unique", type=str, help="Path to transmitters file")
    parser.add_argument("tagging_dict", type=str, help="Path to column mapping dictionary file")
    parser.add_argument("tagging_list", type=str, help="Path to output column list file")
    parser.add_argument("output_file", type=str, help="Path to output file")
    parser.add_argument("--readonly", action="store_true", help="Do not modify input files")

    args = parser.parse_args(sys.argv[1:])
    main(args)