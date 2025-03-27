#!/usr/bin/env python
"""
SimplifiedCPUE.py - A script for processing shark CPUE data from CSV files
"""
import argparse
import glob
import os
import pandas as pd
import re


def read_file(file_path):
    """Read a file with the appropriate encoding"""
    encodings = ['utf-8-sig', 'cp1252', 'latin1', 'iso-8859-1']

    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as file:
                return file.readlines(), encoding
        except UnicodeDecodeError:
            continue

    print(f"ERROR: Could not read file {file_path}")
    return None, None


def extract_sections(content):
    """Extract sections marked with [SectionName]"""
    sections = {}
    current_section = None
    current_lines = []

    for line in content:
        line = line.strip()
        if not line:
            continue

        section_match = re.match(r"\[\s*(.*?)\s*\]", line, re.IGNORECASE)
        if section_match:
            if current_section:
                sections[current_section] = current_lines

            current_section = section_match.group(1)
            current_lines = []
        elif current_section:
            current_lines.append(line)

    # Add the last section
    if current_section and current_lines:
        sections[current_section] = current_lines

    return sections


def parse_csv_section(lines):
    """Parse a section as CSV data"""
    if not lines:
        return pd.DataFrame()

    csv_content = "\n".join(lines)

    try:
        df = pd.read_csv(pd.io.common.StringIO(csv_content), skip_blank_lines=True, on_bad_lines='warn')
        return df
    except Exception as e:
        print(f"Error parsing CSV: {e}")
        return pd.DataFrame()


def process_files(directory):
    """Process all CSV files in the directory"""
    capture_data = []
    gear_data = []

    for file_path in glob.glob(os.path.join(directory, "*.csv")):
        print(f"Processing {file_path}")

        content, _ = read_file(file_path)
        if not content:
            continue

        sections = extract_sections(content)

        if "Megalodon Capture" in sections:
            df = parse_csv_section(sections["Megalodon Capture"])
            df['source_file'] = os.path.basename(file_path)
            capture_data.append(df)

        if "Megalodon Gear" in sections:
            df = parse_csv_section(sections["Megalodon Gear"])
            df['source_file'] = os.path.basename(file_path)
            gear_data.append(df)

    # Combine all data
    if capture_data:
        capture_df = pd.concat(capture_data, ignore_index=True)
    else:
        capture_df = pd.DataFrame()

    if gear_data:
        gear_df = pd.concat(gear_data, ignore_index=True)
    else:
        gear_df = pd.DataFrame()

    return capture_df, gear_df


def get_column_mapping(dict_file):
    """Create column mapping from dictionary file"""
    try:
        df = pd.read_csv(dict_file)
        # Assuming Column_No_in_output, Column name, sharks_CPUE are the columns
        mapping = {}

        for _, row in df.iterrows():
            if 'Column name' in df.columns and 'sharks_CPUE' in df.columns:
                source = row['Column name']
                target = row['sharks_CPUE']
                if pd.notna(source) and pd.notna(target):
                    mapping[source] = target

        return mapping
    except Exception as e:
        print(f"Error loading dictionary file: {e}")
        return {}


def get_output_columns(list_file):
    """Get output columns from list file"""
    try:
        df = pd.read_csv(list_file)
        if len(df.columns) > 0:
            return list(df.iloc[:, 0])
        return []
    except Exception as e:
        print(f"Error loading list file: {e}")
        return []


def combine_data(capture_df, gear_df, column_mapping, output_columns):
    """Combine data from capture and gear sections"""
    result_df = pd.DataFrame(columns=output_columns)

    if capture_df.empty:
        print("No capture data found")
        return result_df

    for idx, capture_row in capture_df.iterrows():
        result_row = {}
        source_file = capture_row.get('source_file', '')

        # Process date/time
        date_value = capture_row.get('DateTime')
        time_value = capture_row.get('TS Capture')

        if pd.notna(date_value):
            # Set deployment date/time
            try:
                dep_date = pd.to_datetime(date_value)
                result_row['dep_date_time'] = dep_date.strftime('%d/%m/%Y %H:%M')

                # Set collection date/time
                if pd.notna(time_value):
                    result_row['col_date_time'] = f"{dep_date.strftime('%d/%m/%Y')} {time_value}"
            except:
                pass

        # Copy location values if available
        for field in ['Lat', 'Long']:
            if field in capture_df.columns and pd.notna(capture_row.get(field)):
                target_field = 'dep_lat' if field == 'Lat' else 'dep_lon'
                result_row[target_field] = capture_row.get(field)

        # Process TagNumber if available
        if 'TagNumber' in capture_df.columns and pd.notna(capture_row.get('TagNumber')):
            try:
                result_row['visual_tag'] = int(float(capture_row.get('TagNumber')))
            except:
                result_row['visual_tag'] = capture_row.get('TagNumber')

        # Map other capture columns
        for source_col, target_col in column_mapping.items():
            if source_col in capture_df.columns and target_col in output_columns:
                value = capture_row.get(source_col)
                if pd.notna(value):
                    result_row[target_col] = value

        # Add gear data if available
        if not gear_df.empty:
            # Get gear rows from the same file
            file_gear = gear_df[gear_df['source_file'] == source_file]

            if not file_gear.empty:
                # Get the last non-NA values for relevant columns
                for gear_col in file_gear.columns:
                    if gear_col in column_mapping:
                        target_col = column_mapping[gear_col]
                        if target_col in output_columns and target_col not in result_row:
                            values = file_gear[gear_col].dropna()
                            if not values.empty:
                                result_row[target_col] = values.iloc[-1]

        # Add row to result
        result_df = pd.concat([result_df, pd.DataFrame([result_row])], ignore_index=True)

    # Ensure all columns exist
    for col in output_columns:
        if col not in result_df.columns:
            result_df[col] = None

    # Sort columns to match output_columns order
    result_df = result_df[output_columns]

    # Sort by visual_tag if available
    if 'visual_tag' in result_df.columns and not result_df.empty:
        try:
            result_df = result_df.sort_values(by='visual_tag')
        except:
            pass

    return result_df


def main():
    parser = argparse.ArgumentParser(description="Process shark CPUE data")
    parser.add_argument("delphis_dir", help="Directory with Delphis CSV files")
    parser.add_argument("cpue_dict", help="CPUE dictionary CSV file")
    parser.add_argument("cpue_list", help="CPUE output column list CSV file")
    parser.add_argument("output_file", help="Output CSV file path")

    args = parser.parse_args()

    print(f"Processing files from {args.delphis_dir}")
    print(f"Using dictionary {args.cpue_dict}")
    print(f"Using column list {args.cpue_list}")
    print(f"Output will be saved to {args.output_file}")

    # Get column mapping and output columns
    column_mapping = get_column_mapping(args.cpue_dict)
    output_columns = get_output_columns(args.cpue_list)

    print(f"Found {len(column_mapping)} column mappings")
    print(f"Found {len(output_columns)} output columns")

    # Process files
    capture_df, gear_df = process_files(args.delphis_dir)

    print(f"Processed {len(capture_df) if not capture_df.empty else 0} capture records")
    print(f"Processed {len(gear_df) if not gear_df.empty else 0} gear records")

    # Combine data
    result_df = combine_data(capture_df, gear_df, column_mapping, output_columns)

    # Save output
    if not result_df.empty:
        print(f"Generated {len(result_df)} records, saving to {args.output_file}")
        result_df.to_csv(args.output_file, index=False)
        print("Done!")
    else:
        print("No data to save")


if __name__ == "__main__":
    main()