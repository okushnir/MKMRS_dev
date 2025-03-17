import pandas as pd

# Load the CSV file
df = pd.read_csv('/Users/odedkushnir/MKMRS/Peleg/fish_pathogens_data_26DEC24 2.csv')

# 1. Convert the date column
df['collection_date'] = pd.to_datetime(df['collection_date'], format='%d.%m.%y').dt.strftime('%Y-%m-%d')
df['necropsy_date'] = pd.to_datetime(df['necropsy_date'], format='%d.%m.%y').dt.strftime('%Y-%m-%d')
# 2.Columns to convert to integers
columns_to_convert = [
    "nnv", "strep", "photobac", "vibrio", "mycobacterium", "toxoplasma",
    "herpesvirus", "learedius", "contracaecum", "aeromonas",
    "citrobacter", "klebsiella", "proteus", "Shewanwlla_spp"]
#
#
for column in columns_to_convert:
    # Handle missing or invalid values by filling them with 0 (or any default value)
    df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)
#
# 3.Columns that might contain commas as thousand separators
columns_with_commas = ["weight_gr", "length_cm"]
for column in columns_with_commas:
        if column in df.columns:
            df[column] = df[column].replace({',': ''}, regex=True)
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

# Save the modified CSV
df.to_csv('/Users/odedkushnir/MKMRS/Peleg/fish_pathogens_data_26DEC24_2_dates.csv', index=False)