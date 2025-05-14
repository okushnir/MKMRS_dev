import pandas as pd
import numpy as np
from datetime import datetime


def check_table_date(df, column_name):
    # Print all unique values in the sample_date column
    unique_dates = df[column_name].unique()
    print("Number of unique date values:", len(unique_dates))
    print("Unique date values:")
    for date in sorted(unique_dates):
        print(f"'{date}' - Type: {type(date)}")

    # Check for potential problematic values
    print("\nChecking for potential problematic values:")
    for i, date in enumerate(df[column_name]):
        if pd.isna(date) or date == '' or str(date).isspace():
            print(f"Row {i}: Empty or NA value")
        elif not isinstance(date, str):
            print(f"Row {i}: Non-string value - {date}, Type: {type(date)}")


def standardize_date(date_str):
    """
    # Function to standardize date format to YYYY-MM-DD
    Args:
        date_str:

    Returns:

    """
    try:
        # First check if it's in YYYY/MM/DD format
        if len(date_str) == 10 and date_str[4] == '/' and date_str[7] == '/':
            year, month, day = date_str.split('/')
            return f"{year}-{month}-{day}"

        # Otherwise assume it's in DD/MM/YYYY format
        else:
            day, month, year = date_str.split('/')
            return f"{year}-{month}-{day}"
    except:
        # If any parsing error occurs, return the original string
        return date_str

    # Apply the date standardization
    # df['sample_date'] = df['sample_date'].apply(standardize_date)

# Read the CSV file
input_file = '/Users/odedkushnir/MKMRS/Chemistry/Tal/20250320_ODV_Database_ALL_Stations_modified for web upload.csv'
df = pd.read_csv(input_file)

# Step 1: Check for missing values
print(f"Total rows: {len(df)}")
print(f"Missing values in sample_date: {df['sample_date'].isna().sum()}")
print(f"Missing values in sample_time: {df['sample_time'].isna().sum()}")

# Step 2: Fill missing sample_date values from time_iso8601
if 'time_iso8601' in df.columns:
    # Create a mask for rows with missing sample_date
    date_mask = df['sample_date'].isna()

    if date_mask.any():
        print(f"Filling {date_mask.sum()} missing dates from time_iso8601")

        # Extract date part from time_iso8601 (format: yyyy-mm-ddThh:mm)
        # and convert to yyyy/mm/dd
        df.loc[date_mask, 'sample_date'] = df.loc[date_mask, 'time_iso8601'].apply(
            lambda x: datetime.strptime(x.split('T')[0], '%Y-%m-%d').strftime('%Y/%m/%d')
            if isinstance(x, str) and 'T' in x else np.nan
        )

    # Step 3: Fill missing sample_time values from time_iso8601
    time_mask = df['sample_time'].isna()

    if time_mask.any():
        print(f"Filling {time_mask.sum()} missing times from time_iso8601")

        # Extract time part from time_iso8601 (format: yyyy-mm-ddThh:mm)
        df.loc[time_mask, 'sample_time'] = df.loc[time_mask, 'time_iso8601'].apply(
            lambda x: x.split('T')[1][:5] if isinstance(x, str) and 'T' in x else np.nan
        )

# Step 4: Format existing sample_date values to dd/mm/yyyy
# First, make a copy of the date column to avoid modifying during iteration
date_values = df['sample_date'].copy()

# Initialize a list to track which rows were reformatted
reformatted_rows = []

# Try different common date formats and convert to dd/mm/yyyy
for idx, date_val in enumerate(date_values):
    if pd.notna(date_val) and isinstance(date_val, str):
        # Skip if already in yyyy/mm/dd format
        if len(date_val.split('/')) == 3 and len(date_val.split('/')[0]) == 4:
            continue

        try:
            # Try yyyy-mm-dd format
            if '-' in date_val:
                date_obj = datetime.strptime(date_val, '%Y-%m-%d')
                df.loc[idx, 'sample_date'] = date_obj.strftime('%Y/%m/%d')
                reformatted_rows.append(idx)
            # Try mm/dd/yyyy format
            elif '/' in date_val and len(date_val.split('/')[2]) == 4:
                date_obj = datetime.strptime(date_val, '%m/%d/%Y')
                df.loc[idx, 'sample_date'] = date_obj.strftime('%Y/%m/%d')
                reformatted_rows.append(idx)
            # Try other formats as needed
        except (ValueError, IndexError):
            # If format is unrecognized, leave it as is
            pass

print(f"Reformatted {len(reformatted_rows)} date values to yyyy/mm/dd format")

# Step 5: Ensure sample_time is in hh:mm format
# Make a copy of the time column
time_values = df['sample_time'].copy()
time_reformatted = 0
time_replaced = 0

for idx, time_val in enumerate(time_values):
    # Check if value is in proper hh:mm format
    is_proper_format = False

    if pd.notna(time_val):
        if isinstance(time_val, str):
            # Check if already in proper hh:mm format
            if len(time_val) == 5 and ':' in time_val and time_val.count(':') == 1:
                # Verify hour and minute parts are numeric
                parts = time_val.split(':')
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    is_proper_format = True

            # If not in proper format but has colons, try to fix
            elif ':' in time_val:
                try:
                    # For formats like hh:mm:ss, get just the hh:mm part
                    parts = time_val.split(':')
                    if len(parts) >= 2:
                        df.loc[idx, 'sample_time'] = f"{parts[0].zfill(2)}:{parts[1].zfill(2)}"
                        time_reformatted += 1
                        is_proper_format = True
                except Exception:
                    pass

        # Handle numeric values (like 36925, 43048, etc.) or any other improper format
        if not is_proper_format:
            # Replace with time from time_iso8601 if available
            if pd.notna(df.at[idx, 'time_iso8601']) and isinstance(df.at[idx, 'time_iso8601'], str):
                try:
                    # Extract time part from time_iso8601
                    iso_time = df.at[idx, 'time_iso8601'].split('T')[1][:5]
                    df.loc[idx, 'sample_time'] = iso_time
                    time_replaced += 1
                except (IndexError, AttributeError):
                    pass

print(f"Reformatted {time_reformatted} time values to hh:mm format")
print(f"Replaced {time_replaced} improper time values with times from time_iso8601")

# Step 6: Check for any remaining missing values
remaining_missing_dates = df['sample_date'].isna().sum()
remaining_missing_times = df['sample_time'].isna().sum()

print(f"Remaining missing dates: {remaining_missing_dates}")
print(f"Remaining missing times: {remaining_missing_times}")

# Step 7: Save the cleaned data
df["time_iso8601"] = df["time_iso8601"].fillna("")
df = df[df['time_iso8601']!=""]
df.to_csv(input_file.split(".")[0] + "_cleaned_station_data.csv", index=False)
print("Saved cleaned data to cleaned_station_data.csv")

# Print a sample of the result
print("\nSample of results (first 5 rows):")
print(df[['sample_date', 'sample_time', 'time_iso8601']].head(5))
