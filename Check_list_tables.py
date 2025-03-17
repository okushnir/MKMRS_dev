import pandas as pd


def check_for_blanks_date(df, column):
    """
    Checks if there are any blank ("") values in the specified column of a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The name of the column to check.

    Returns:
        bool: True if blank values are found, False otherwise.
    """

    # Ensure NaN values are replaced with empty strings
    df = df.fillna("")

    # Check if any values in the column are exactly ""
    return (df[column] == "").any()

def compare_table_columns(table1_columns, table2_columns):
    """
    Compare columns between two tables and report differences.

    Args:
        table1_columns (list): List of column names from the first table
        table2_columns (list): List of column names from the second table
    """
    # Convert lists to sets for easier comparison
    set1 = set(table1_columns)
    set2 = set(table2_columns)

    # Check if sets are equal
    if set1 == set2:
        print("The columns are matched")
    else:
        # Find columns unique to each table
        only_in_table1 = set1 - set2
        only_in_table2 = set2 - set1

        # Print columns unique to table1
        if only_in_table1:
            print("Columns only in Table 1:")
            for column in sorted(only_in_table1):
                print(f"- {column}")

        # Print columns unique to table2
        if only_in_table2:
            print("Columns only in Table 2:")
            for column in sorted(only_in_table2):
                print(f"- {column}")


def main():

    # # Example table columns
    # table1_columns = ["id", "name", "age", "email"]
    # table2_columns = ["id", "name", "phone", "email"]
    #
    # print("Example 1:")
    # compare_table_columns(table1_columns, table2_columns)
    #
    # # Another example
    # table3_columns = ["product_id", "price", "quantity", "category"]
    # table4_columns = ["product_id", "price", "quantity", "category"]
    #
    # print("\nExample 2:")
    # compare_table_columns(table3_columns, table4_columns)

    table1 = pd.read_csv('/Users/odedkushnir/MKMRS/Chemistry/Tal/chemistry_db_ch1_sample_data.csv')
    table2 = pd.read_csv('/Users/odedkushnir/MKMRS/Chemistry/Tal/12022025_ODV_Database_ALL_Stations_for_upload_date_modification.csv')
    compare_table_columns(table1, table2)

if __name__ == '__main__':
    main()