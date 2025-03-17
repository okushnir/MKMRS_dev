import pandas as pd
import re


def separate_megalodon(file_path):
    # Initialize empty lists to collect lines for each table
    capture_lines = []  # Will store lines under "[Megalodon Capture]"
    gear_lines = []  # Will store lines under "[Megalodon Gear]"
    current_section = None  # Tracks which table we're reading lines into

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if "[Megalodon Capture]" in line:
                current_section = "capture"
                continue  # Skip the marker line itself
            elif "[Megalodon Gear]" in line:
                current_section = "gear"
                continue  # Skip the marker line itself

            if current_section == "capture":
                capture_lines.append(line)
            elif current_section == "gear":
                gear_lines.append(line)

    # Function to convert raw lines to DataFrame
    def lines_to_dataframe(lines):
        """
        Converts lines into a pandas DataFrame, ensuring consistent column counts.
        Rows with mismatched column counts are ignored.

        Args:
            lines (list): List of strings representing the rows of the table.

        Returns:
            pd.DataFrame: DataFrame constructed from the lines.
        """
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

    # Convert the lines into DataFrames
    capture_df = lines_to_dataframe(capture_lines)
    gear_df = lines_to_dataframe(gear_lines)
    return capture_df, gear_df


def extract_tables_from_csv(file_path):
    """
    Extracts multiple tables from a CSV file where tables are separated by lines like '[marker]'.

    Args:
        file_path (str): Path to the input file.

    Returns:
        dict: A dictionary where keys are table names (from markers) and values are pandas DataFrames.
    """
    # Initialize variables
    data_sections = {}
    current_marker = None
    current_lines = []

    # Read the file line by line
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()

            # Check for marker lines: [something here]
            if re.match(r"\[.*?\]", line):
                # Store the previous section (if any)
                if current_marker:
                    data_sections[current_marker] = current_lines

                # Start a new section
                current_marker = line.strip("[]")  # Remove square brackets
                current_lines = []  # Reset lines for this new section
            else:
                # Add non-marker lines to the current section
                if current_marker:
                    current_lines.append(line)

        # Store the last section after finishing the file
        if current_marker and current_lines:
            data_sections[current_marker] = current_lines

    # Function to convert raw lines to DataFrame
    def lines_to_dataframe(lines):
        if not lines:  # Handle empty sections
            return pd.DataFrame()
        header = lines[0].split(",")  # First line as header
        data = [line.split(",") for line in lines[1:] if line]  # Remaining lines are data
        return pd.DataFrame(data, columns=header)

    # Convert each section into DataFrames
    dataframes = {marker: lines_to_dataframe(lines) for marker, lines in data_sections.items()}
    return dataframes


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
    data_sections = {}
    current_marker = None
    current_lines = []

    # Read the file line by line
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()

            # Check for marker lines: [something here]
            if re.match(r"\[.*?\]", line, re.IGNORECASE):
                if current_marker:
                    print(current_marker)
                    data_sections[current_marker] = current_lines
                current_marker = line.strip("[]")  # Remove square brackets
                current_lines = []  # Reset lines for this new section
            else:
                if current_marker:
                    current_lines.append(line)

        # Store the last section after finishing the file
        if current_marker and current_lines:
            data_sections[current_marker] = current_lines

    # Updated lines_to_dataframe function
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
