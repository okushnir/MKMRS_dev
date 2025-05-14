#!/usr/bin/env python3
"""
Noise reduction script for GLATOS detection data.
This script applies the false_detections filter to reduce noise in acoustic telemetry data.
"""

import os
import sys
import subprocess
from pathlib import Path


def run_r_script(r_code):
    """Execute R code using subprocess."""
    try:
        # Create temporary R script
        temp_script = "temp_noise_filter.R"
        with open(temp_script, 'w') as f:
            f.write(r_code)

        # Run R script
        result = subprocess.run(['Rscript', temp_script],
                                capture_output=True, text=True, encoding='utf-8')

        # Clean up
        os.remove(temp_script)

        if result.returncode == 0:
            print("R script executed successfully")
            if result.stdout:
                print(result.stdout)
        else:
            print("Error running R script:")
            print(result.stderr)

        return result

    except Exception as e:
        print(f"Error running R script: {e}")
        return None


def reduce_noise(input_file, output_file=None, time_filter=3600):
    """Reduce noise in GLATOS detection data by filtering false detections."""
    if output_file is None:
        output_file = input_file.replace('.csv', '_filtered.csv')

    print(f"Starting noise reduction...")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Time filter: {time_filter} seconds")

    r_code = f'''
    # Simple, robust noise reduction script
    print("=== GLATOS Noise Reduction ===")

    # Step 1: Check if GLATOS is installed
    if (!require(glatos, quietly = TRUE)) {{
        stop("GLATOS package is not installed. Please install it first.")
    }}

    # Step 2: Read the data
    print("\\nReading input file...")
    print("File: {input_file}")

    # Check if file exists
    if (!file.exists("{input_file}")) {{
        stop("Input file does not exist")
    }}

    # Read the CSV file
    dets <- read.csv("{input_file}")
    print(paste("Read", nrow(dets), "rows and", ncol(dets), "columns"))

    # Check essential columns
    if (!"detection_timestamp_utc" %in% colnames(dets)) {{
        stop("Column 'detection_timestamp_utc' not found")
    }}

    if (!"animal_id" %in% colnames(dets)) {{
        stop("Column 'animal_id' not found")
    }}

    # Convert timestamp to POSIXct if needed
    if (is.character(dets$detection_timestamp_utc)) {{
        print("Converting timestamp to POSIXct...")
        dets$detection_timestamp_utc <- as.POSIXct(dets$detection_timestamp_utc, 
                                                  format = "%Y-%m-%d %H:%M:%S", 
                                                  tz = "UTC")
    }}

    # Add GLATOS class
    class(dets) <- c("glatos_detections", "data.frame")

    # Step 3: Apply false detection filter
    print("\\nApplying false detection filter...")
    print(paste("Time filter:", {time_filter}, "seconds"))

    # Run the false_detections function
    dets_filtered <- false_detections(dets, tf = {time_filter})

    # Step 4: Check results
    if ("passed_filter" %in% colnames(dets_filtered)) {{
        # Count results
        total <- nrow(dets_filtered)
        passed <- sum(dets_filtered$passed_filter == TRUE, na.rm = TRUE)
        failed <- sum(dets_filtered$passed_filter == FALSE, na.rm = TRUE)

        print("\\nFiltering results:")
        print(paste("Total detections:", format(total, big.mark=",")))
        print(paste("Passed filter:", format(passed, big.mark=","), 
                    paste0("(", round(passed/total*100, 1), "%)")))
        print(paste("Failed filter:", format(failed, big.mark=","), 
                    paste0("(", round(failed/total*100, 1), "%)")))

        # Keep only detections that passed the filter
        final_data <- dets_filtered[dets_filtered$passed_filter == TRUE, ]

        print(paste("\\nFinal dataset:", format(nrow(final_data), big.mark=","), "detections"))
    }} else {{
        print("\\nWarning: No 'passed_filter' column created")
        print("Using original data without filtering")
        final_data <- dets_filtered
    }}

    # Step 5: Select only the required columns
    print("\\nSelecting essential columns...")
    required_columns <- c("animal_id", "protocol", "station", "detection_timestamp_utc")

    # Check which required columns exist
    available_columns <- colnames(final_data)
    missing_columns <- required_columns[!required_columns %in% available_columns]
    if (length(missing_columns) > 0) {{
        print(paste("Warning: Missing columns:", paste(missing_columns, collapse=", ")))
    }}

    # Select only the columns that exist and are required
    columns_to_keep <- required_columns[required_columns %in% available_columns]
    final_data <- final_data[, columns_to_keep, drop = FALSE]

    print(paste("Selected columns:", paste(columns_to_keep, collapse=", ")))
    print(paste("Final dataset dimensions:", nrow(final_data), "rows x", ncol(final_data), "columns"))

    # Step 6: Save the filtered data with only essential columns
    print("\\nSaving filtered data...")
    write.csv(final_data, "{output_file}", row.names = FALSE)

    # Verify the file was created
    if (file.exists("{output_file}")) {{
        file_info <- file.info("{output_file}")
        print("\\n✓ SUCCESS!")
        print(paste("Output file:", "{output_file}"))
        print(paste("File size:", format(file_info$size, big.mark=","), "bytes"))
        print(paste("Rows saved:", nrow(final_data)))
        print(paste("Columns saved:", ncol(final_data)))
        print(paste("Column names:", paste(colnames(final_data), collapse=", ")))

        # Quick summary
        print("\\nQuick summary of filtered data:")
        if ("animal_id" %in% colnames(final_data)) {{
            print(paste("Unique animals:", length(unique(final_data$animal_id))))
        }}
        if ("station" %in% colnames(final_data)) {{
            print(paste("Unique stations:", length(unique(final_data$station))))
        }}
    }} else {{
        print("\\n✗ ERROR: Output file was not created")
    }}

    print("\\n=== Noise Reduction Complete ===")
    '''

    # Run the R script
    run_r_script(r_code)


def main():
    if len(sys.argv) < 2:
        print("Usage: python noise_filter.py <input_file> [output_file] [time_filter]")
        print("Example: python noise_filter.py data.csv filtered_data.csv 3600")
        print("Output will contain only columns: animal_id, protocol, station, detection_timestamp_utc")
        return

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    time_filter = int(sys.argv[3]) if len(sys.argv) > 3 else 3600

    reduce_noise(input_file, output_file, time_filter)


if __name__ == "__main__":
    main()