#!/usr/bin/env python3
"""
CSV Timestamp Combiner
Combines 'date' and 'time' columns from a CSV file into a single 'detection_timestamp_utc' column.
"""

import pandas as pd
import sys
from datetime import datetime
import argparse


def combine_datetime_columns(input_file, output_file=None):
    """
    Combines 'date' and 'time' columns into 'detection_timestamp_utc' column.

    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file (optional)

    Returns:
        pd.DataFrame: DataFrame with the new timestamp column
    """
    try:
        # Read the CSV file
        print(f"Reading CSV file: {input_file}")
        df = pd.read_csv(input_file)

        # Clean column names by stripping whitespace
        df.columns = df.columns.str.strip()
        print(f"Columns found: {list(df.columns)}")

        # Check if required columns exist
        if 'date' not in df.columns or 'time' not in df.columns:
            missing_cols = []
            if 'date' not in df.columns:
                missing_cols.append('date')
            if 'time' not in df.columns:
                missing_cols.append('time')
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Display info about the DataFrame
        print(f"Found {len(df)} rows with columns: {list(df.columns)}")

        # Show sample data
        print("\nSample data:")
        print(df[['date', 'time']].head())

        # Combine date and time columns
        print("\nCombining date and time columns...")

        # Handle different date/time formats automatically
        df['detection_timestamp_utc'] = pd.to_datetime(
            df['date'].astype(str) + ' ' + df['time'].astype(str),
            infer_datetime_format=True,
            errors='coerce'
        )

        # Check for any parsing errors
        null_timestamps = df['detection_timestamp_utc'].isnull().sum()
        if null_timestamps > 0:
            print(f"Warning: {null_timestamps} rows had timestamp parsing errors")
            print("Rows with parsing errors:")
            print(df[df['detection_timestamp_utc'].isnull()][['date', 'time']])

        # Display sample of combined timestamps
        print("\nSample combined timestamps:")
        print(df[['date', 'time', 'detection_timestamp_utc']].head())

        # Remove the original date and time columns
        print("\nRemoving original 'date' and 'time' columns...")
        df = df.drop(['date', 'time'], axis=1)

        # Show final columns
        print(f"Final columns: {list(df.columns)}")
        print(f"Final DataFrame shape: {df.shape}")

        # Save to output file if specified
        if output_file:
            print(f"\nSaving to: {output_file}")
            df.to_csv(output_file, index=False)
        else:
            # If no output file specified, save with '_with_timestamp' suffix
            output_file = input_file.replace('.csv', '_with_timestamp.csv')
            print(f"\nSaving to: {output_file}")
            df.to_csv(output_file, index=False)

        print("Done!")
        return df

    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Combine date and time columns in CSV file')
    parser.add_argument('input_file', help='Input CSV file path')
    parser.add_argument('-o', '--output', help='Output CSV file path (optional)')
    parser.add_argument('--preview', action='store_true', help='Preview only, do not save file')

    args = parser.parse_args()

    if args.preview:
        # Preview mode - don't save file
        df = combine_datetime_columns(args.input_file, output_file=None)
        print("\nPreview mode - file not saved")
    else:
        # Normal mode - save file
        combine_datetime_columns(args.input_file, args.output)


if __name__ == "__main__":
    main()