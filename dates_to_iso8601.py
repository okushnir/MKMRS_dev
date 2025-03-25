import pandas as pd
from datetime import datetime


# def convert_to_iso8601(df, columns_to_convert=None):
#     """
#     Convert date strings in specified columns to standard ISO 8601 format,
#     first removing the 'T' character that might be causing parsing issues.
#
#     Args:
#         df (pandas.DataFrame): The DataFrame containing the columns to convert
#         columns_to_convert (list, optional): List of column names to convert.
#                                            If None, converts all columns with string-like data.
#
#     Returns:
#         pandas.DataFrame: DataFrame with converted date columns
#     """
#     # Create a copy of the dataframe to avoid modifying the original
#     result_df = df.copy()
#
#     # If no specific columns are provided, try to convert all object/string columns
#     if columns_to_convert is None:
#         columns_to_convert = df.select_dtypes(include=['object']).columns
#
#     # Process each specified column
#     for col in columns_to_convert:
#         if col in df.columns:
#             try:
#                 # First replace 'T' with space in the datetime strings
#                 cleaned_dates = df[col].astype(str).str.replace('T', ' ')
#
#                 # Parse the datetime and convert back to standard ISO 8601 format
#                 result_df[col] = pd.to_datetime(cleaned_dates).dt.strftime('%Y-%m-%d %H:%M:%SZ', format="mixed")
#                 print(f"Successfully converted column '{col}'")
#             except Exception as e:
#                 print(f"Error converting column '{col}': {e}")
#                 try:
#                     # Try alternative approach - directly parse without cleaning
#                     result_df[col] = pd.to_datetime(df[col], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%SZ')
#                     print(f"Successfully converted column '{col}' with alternative method")
#                 except Exception as e2:
#                     print(f"All conversion attempts failed for column '{col}': {e2}")
#
#     return result_df


def convert_to_iso8601(df, column_name):
    """
    Converts a given column in a DataFrame to ISO 8601 format (YYYY-MM-DDTHH:MM:SSZ).

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column to convert.

    Returns:
        pd.DataFrame: DataFrame with the column converted to ISO 8601.
    """
    df[column_name] = pd.to_datetime(df[column_name], errors='coerce')  # Convert to datetime
    df[column_name] = df[column_name].dt.strftime('%Y-%m-%dT%H:%M:%S')  # Convert to ISO 8601 format
    return df


import pandas as pd


def fix_date_format(df, column_name):
    """
    Checks if dates in a given column are in MM/DD/YYYY format and converts them to DD/MM/YYYY
    based on whether the first number (day) is greater than 12.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column to check and convert.

    Returns:
        pd.DataFrame: The modified DataFrame with corrected date formats.
    """

    def convert_date(date_str):
        try:
            # Split the date to check format
            parts = date_str.split('/')
            if len(parts) == 3:
                month, day, year = parts
                # Convert to integers
                month, day = int(month), int(day)
                # If the day is greater than 12, it's actually DD/MM/YYYY, so swap
                if day > 12:
                    return f"{day:02d}/{month:02d}/{year}"
                else:
                    return date_str  # Keep as is if it's already correct
            return date_str  # If format is unknown, keep as is
        except:
            return date_str  # In case of errors, return the original value

    df[column_name] = df[column_name].astype(str).apply(convert_date)
    return df



def remove_empty_sample_date(df, column_name):
    """
    Removes rows where the specified column is empty, NaN, or contains only spaces.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column to check for empty values.

    Returns:
        pd.DataFrame: A DataFrame with empty rows removed.
    """
    # df = fix_date_format(df, column_name)
    # Convert the column to string, strip spaces, and remove rows with empty or NaN values
    df[column_name] = df[column_name].fillna("")
    blanks_df = df[df[column_name] == ""]
    blanks_df.to_csv('/Users/odedkushnir/MKMRS/Chemistry/Tal/20250320_ODV_Database_ALL_Stations_modified for web upload_blanks.csv', index=False)
    df = df[df[column_name] != ""]

    return df

def main():

    # Load the CSV file
    sample_df = pd.read_csv('/Users/odedkushnir/MKMRS/Chemistry/Tal/chemistry_db_ch1_sample_data.csv')
    print(sample_df["time_iso8601"])
    print("**********************")
    df = pd.read_csv('/Users/odedkushnir/MKMRS/Chemistry/Tal/20250320_ODV_Database_ALL_Stations_modified for web upload.csv')

    # df["time_iso8601"] = df["time_iso8601"].astype(str).str.replace('T', ' ')
    # df["time_iso8601"] = pd.to_datetime(df["time_iso8601"]).dt.strftime('%Y-%m-%d %H:%M:%SZ', format="mixed")
    convert_to_iso8601(df, "time_iso8601")
    # First, make sure the column is treated as a string
    df['time_iso8601'] = df['time_iso8601'].astype(str)

    # Then convert to datetime explicitly
    df['time_iso8601_dt'] = pd.to_datetime(df['time_iso8601'])

    # Now create the sample_date column (date only)
    df['sample_date'] = df['time_iso8601_dt'].dt.date

    # Create the sample_time column (HH:MM format)
    df['sample_time'] = df['time_iso8601_dt'].dt.strftime('%H:%M')

    # If you want to remove the intermediate datetime column
    df = df.drop('time_iso8601_dt', axis=1)

    df["sample_date"] = pd.to_datetime(df["time_iso8601"], errors='coerce')  # Convert to datetime
    df["sample_date"] = df["sample_date"].dt.strftime('%Y/%m/%d')
    # df = df.fillna("")
    # null_df = df[df["sample_date"] != ""]
    df = remove_empty_sample_date(df, "sample_date")
    # fix_date_format(df, "sample_date")
    # df["sample_date"] = pd.to_datetime(df["sample_date"], format='%d/%m/%Y').dt.strftime('%Y-%m-%d')


    # df.to_csv('/Users/odedkushnir/MKMRS/Chemistry/Tal/12022025_ODV_Database_ALL_Stations_for_upload_date_modification.csv', index=False)

    # df["time_iso8601"] = df["time_iso8601"].astype(str).str.replace('T', '_')
    # df["time_iso8601"] = df["time_iso8601"].astype(str) + "Z"
    # df["time_iso8601"] = pd.to_datetime(df["time_iso8601"], errors='coerce')
    # df["time_iso8601"] = df["time_iso8601"].astype(str).str.replace(' ', '_')

    # Convert back to the desired format
    # df["time_iso8601"] = pd.to_datetime(df["time_iso8601"].dt.strftime('%Y-%m-%d'), errors='coerce')

    # Save the modified file
    output_path = "/Users/odedkushnir/MKMRS/Chemistry/Tal/20250320_ODV_Database_ALL_Stations_modified for web upload_date_modification.csv"
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()