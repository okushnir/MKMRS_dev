#!/usr/bin/env python
"""
@Created by: Biodaat; Oded Kushnir
@Adapted for CPUE Processing from tagged transect files

"""
import argparse
import glob
import os
import pandas as pd
import numpy as np
import re
import sys
import chardet


def detect_encoding(file_path):
    with open(file_path, "rb") as f:
        raw_data = f.read(100000)
    result = chardet.detect(raw_data)
    return result["encoding"] if result["encoding"] else "utf-8"


def extract_tables_with_filter(file_path):
    encoding = detect_encoding(file_path)
    print(f"Reading {file_path} with encoding {encoding}")

    data_sections = {}
    current_marker = None
    current_lines = []

    with open(file_path, "r", encoding=encoding, errors="replace") as file:
        for line in file:
            stripped_line = line.strip()
            first_token = stripped_line.split(",")[0]

            marker_match = re.match(r"\[\s*(.*?)\s*\]", first_token, re.IGNORECASE)
            if marker_match:
                if current_marker and current_lines:
                    data_sections[current_marker] = current_lines
                current_marker = marker_match.group(1).strip().lower()
                current_lines = []
            else:
                if current_marker:
                    current_lines.append(stripped_line)

        if current_marker and current_lines:
            data_sections[current_marker] = current_lines

    for marker, lines in data_sections.items():
        if not lines:
            continue
        header = lines[0].split(",")
        rows = [row.split(",") for row in lines[1:] if row.strip() and len(row.split(",")) == len(header)]
        data_sections[marker] = pd.DataFrame(rows, columns=header)

    return data_sections


def process_cpue_directory(delphis_dir, cpue_dict_file, cpue_list_file):
    rename_dict = pd.read_csv(cpue_dict_file).dropna()
    column_mapping = dict(zip(rename_dict["Column name"], rename_dict["sharks_CPUE"]))
    desired_columns = pd.read_csv(cpue_list_file)["sharks_CPUE"].dropna().tolist()

    combined_data = []
    delphis_lst = glob.glob(os.path.join(delphis_dir, '*'))

    for file_path in delphis_lst:
        print(f"Processing file: {file_path}")
        data = extract_tables_with_filter(file_path)

        capture_df = data.get("megalodon capture")
        gear_df = data.get("megalodon gear")

        summary_df = data.get("summary")
        if summary_df is None or summary_df.empty:
            summary_df = data.get("summery")

        if capture_df is None or gear_df is None:
            print("Missing capture or gear table, skipping.")
            continue

        boat_value = None
        if isinstance(summary_df, pd.DataFrame) and "Boat" in summary_df.columns:
            boat_value = summary_df.iloc[0]["Boat"]

        try:
            capture_df["event"] = capture_df["Drumline Number"].astype(str)
            gear_df["event"] = gear_df["Drumline Number"].astype(str)
            merged_df = pd.merge(capture_df, gear_df, on="event", suffixes=("_cap", "_gear"))
        except Exception as e:
            print(f"Merge failed in {file_path}: {e}")
            continue

        if merged_df.empty:
            print(f"No matching events to merge in: {file_path}")
            continue

        # Split Lat Lng column into dep_lat and dep_lon
        if "Lat Lng" in merged_df.columns:
            try:
                merged_df[["dep_lat", "dep_lon"]] = merged_df["Lat Lng"].str.split(" ", expand=True)
            except Exception as e:
                print(f"Failed to split Lat Lng in {file_path}: {e}")

        merged_df = merged_df.rename(columns=column_mapping)

        for col in desired_columns:
            if col not in merged_df.columns:
                merged_df[col] = ""

        if boat_value:
            merged_df["boat"] = boat_value
            if "boat" not in desired_columns:
                desired_columns.append("boat")

        merged_df.drop_duplicates(inplace=True)
        cpue_table = merged_df[desired_columns]
        combined_data.append(cpue_table)

    if combined_data:
        return pd.concat(combined_data, ignore_index=True)
    else:
        return pd.DataFrame(columns=desired_columns)


def main(args):
    final_cpue = process_cpue_directory(args.delphis_dir, args.cpue_dict, args.cpue_list)
    if not final_cpue.empty:
        if args.readonly:
            print("Readonly mode enabled. Skipping file writes.")
        else:
            final_cpue.to_csv(args.output_file, index=False)
            print(f"CPUE table saved to {args.output_file}")
    else:
        print("No CPUE data was generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("delphis_dir", type=str, help="Path to the Delphis_raw directory")
    parser.add_argument("cpue_dict", type=str, help="Path to the CPUE_dict.csv file")
    parser.add_argument("cpue_list", type=str, help="Path to the CPUE_list.csv file")
    parser.add_argument("output_file", type=str, help="Path for the output CPUE CSV file")
    parser.add_argument("--readonly", action="store_true", help="Enable readonly mode (no file writing)")
    args = parser.parse_args(sys.argv[1:])
    main(args)
