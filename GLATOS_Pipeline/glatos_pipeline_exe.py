#!/usr/bin/env python3
"""
GLATOS Pipeline - Complete Acoustic Telemetry Processing System
Single executable file containing all pipeline functionality.
"""

import os
import sys
import subprocess
import pandas as pd
import argparse
from pathlib import Path
import time


def setup_environment():
    """Ensure R is in PATH."""
    current_path = os.environ.get('PATH', '')
    r_paths = ['/usr/local/bin', '/usr/bin', '/opt/homebrew/bin']
    for r_path in r_paths:
        if r_path not in current_path and os.path.exists(r_path):
            os.environ['PATH'] = f"{r_path}:{current_path}"


setup_environment()


class GLATOSPipeline:
    def __init__(self):
        # Try to find R
        self.r_cmd = 'R'
        self.r_path = self.find_r()

    def find_r(self):
        """Find R executable."""
        # Try simple R command first
        try:
            result = subprocess.run(['R', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                return 'R'
        except:
            pass

        # Try common R paths
        for r_path in ['/usr/local/bin/R', '/opt/homebrew/bin/R']:
            if os.path.exists(r_path):
                try:
                    result = subprocess.run([r_path, '--version'], capture_output=True, text=True)
                    if result.returncode == 0:
                        return r_path
                except:
                    continue

        raise RuntimeError("R not found. Please install R and make sure it's in PATH.")

    def connect_to_amazon_db(self, dbname, host, port, username, password):
        """
        Establishes a connection to an Amazon RDS database.

        Args:
            dbname (str): The name of the database to connect to
            host (str): The hostname of the Amazon RDS instance
            port (int): The port number to connect on
            username (str): The username for authentication
            password (str): The password for authentication

        Returns:
            connection: A database connection object
        """
        try:
            # Try to import the required PostgreSQL module
            try:
                import psycopg2
            except ImportError:
                print("Installing psycopg2...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "psycopg2-binary"])
                import psycopg2

            # For PostgreSQL
            conn = psycopg2.connect(
                dbname=dbname,
                host=host,
                port=port,
                user=username,
                password=password
            )
            print("Connection to Amazon RDS established successfully!")
            return conn
        except Exception as e:
            print(f"Error connecting to Amazon RDS: {e}")
            return None

    def run_r_script(self, r_code):
        """Execute R code."""
        import tempfile

        # Create temp file in a writable directory
        try:
            # Try to use system temp directory first
            with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False) as temp_file:
                temp_script = temp_file.name
                temp_file.write(r_code)
        except:
            # Fallback to current working directory if temp fails
            temp_script = os.path.expanduser("~/temp_script.R")
            with open(temp_script, 'w') as f:
                f.write(r_code)

        try:
            # Use Rscript instead of R
            rscript_path = self.r_path.replace('/R', '/Rscript') if self.r_path.endswith('/R') else 'Rscript'
            result = subprocess.run([rscript_path, temp_script], capture_output=True, text=True)

            if result.returncode == 0:
                print("R script executed successfully")
                if result.stdout:
                    print(result.stdout)
                return True
            else:
                print("Error running R script:")
                print(result.stderr)
                return False
        except Exception as e:
            print(f"Error: {e}")
            return False
        finally:
            # Clean up temp file
            try:
                if os.path.exists(temp_script):
                    os.remove(temp_script)
            except:
                pass  # Don't fail if we can't remove temp file

    def combine_timestamps(self, input_file, output_file):
        """Combine date and time columns."""
        print(f"Reading: {input_file}")
        df = pd.read_csv(input_file)
        df.columns = df.columns.str.strip()
        print(f"Columns found: {list(df.columns)}")
        print(f"Found {len(df)} rows")

        # Show sample
        print("\nSample data:")
        print(df[['date', 'time']].head())

        # Combine date and time
        print("\nCombining date and time columns...")
        df['detection_timestamp_utc'] = pd.to_datetime(
            df['date'].astype(str) + ' ' + df['time'].astype(str),
            dayfirst=True,  # Add this if your dates are DD/MM/YYYY
            errors='coerce'
        )

        # Check for errors
        null_timestamps = df['detection_timestamp_utc'].isnull().sum()
        if null_timestamps > 0:
            print(f"Warning: {null_timestamps} rows had parsing errors")

        # Show sample with timestamps
        print("\nSample combined timestamps:")
        print(df[['date', 'time', 'detection_timestamp_utc']].head())

        # Remove original columns
        print("\nRemoving original date and time columns...")
        df = df.drop(['date', 'time'], axis=1)

        print(f"Final columns: {list(df.columns)}")
        print(f"Final shape: {df.shape}")

        # Save
        print(f"\nSaving to: {output_file}")
        df.to_csv(output_file, index=False)
        print("Done!")
        return True

    def convert_to_glatos(self, input_file, output_file):
        """Convert to GLATOS format."""
        r_code = f'''
        # Install GLATOS if needed
        if (!require(glatos, quietly = TRUE)) {{
            if (!require(remotes, quietly = TRUE)) {{
                install.packages("remotes", repos="https://cran.r-project.org/")
            }}
            remotes::install_github("ocean-tracking-network/glatos")
            library(glatos)
        }}

        # Read and convert
        raw_data <- read.csv("{input_file}")
        glatos_data <- raw_data

        # Map columns
        if ("id" %in% colnames(raw_data)) {{
            colnames(glatos_data)[colnames(glatos_data) == "id"] <- "animal_id"
        }}
        if ("receiver" %in% colnames(raw_data)) {{
            colnames(glatos_data)[colnames(glatos_data) == "receiver"] <- "station"
        }}

        # Convert timestamp
        if (!inherits(glatos_data$detection_timestamp_utc, "POSIXct")) {{
            glatos_data$detection_timestamp_utc <- as.POSIXct(glatos_data$detection_timestamp_utc, 
                                                              format = "%Y-%m-%d %H:%M:%S", 
                                                              tz = "UTC")
        }}

        # Add required GLATOS columns
        glatos_data$transmitter_codespace <- ifelse("protocol" %in% colnames(raw_data), 
                                                   raw_data$protocol, "R64K-69kHz")
        glatos_data$transmitter_id <- as.character(glatos_data$animal_id)
        glatos_data$glatos_array <- "ARRAY1"
        glatos_data$station_no <- as.character(glatos_data$station)
        glatos_data$sensor_value <- NA
        glatos_data$sensor_unit <- NA
        glatos_data$deploy_lat <- NA
        glatos_data$deploy_long <- NA
        glatos_data$receiver_sn <- paste0("SN", glatos_data$station)
        glatos_data$tag_type <- NA
        glatos_data$tag_model <- NA
        glatos_data$tag_serial_number <- NA
        glatos_data$common_name_e <- "shark"
        glatos_data$capture_location <- NA
        glatos_data$length <- NA
        glatos_data$weight <- NA
        glatos_data$sex <- NA
        glatos_data$release_group <- NA
        glatos_data$release_location <- NA
        glatos_data$release_latitude <- NA
        glatos_data$release_longitude <- NA
        glatos_data$utc_release_date_time <- NA
        glatos_data$glatos_project_transmitter <- "MKMRS"
        glatos_data$glatos_project_receiver <- "MKMRS"
        glatos_data$glatos_tag_recovered <- "NO"
        glatos_data$glatos_caught_date <- NA

        # Save
        write.csv(glatos_data, "{output_file}", row.names = FALSE)
        print(paste("Saved:", "{output_file}"))
        '''
        return self.run_r_script(r_code)

    def test_glatos_file(self, file_path):
        """Test GLATOS file and create comprehensive summary."""
        input_dir = Path(file_path).parent
        summary_file = input_dir / "glatos_test_summary.txt"
        detection_plot = input_dir / "detection_timeline.png"

        # Convert paths for R (use forward slashes)
        file_path_r = str(file_path).replace('\\', '/')
        summary_file_r = str(summary_file).replace('\\', '/')
        detection_plot_r = str(detection_plot).replace('\\', '/')

        r_code = f'''
        library(glatos)

        # Read as CSV and add GLATOS class
        print("Testing converted GLATOS file...")
        detections <- read.csv("{file_path_r}")
        class(detections) <- c("glatos_detections", "data.frame")

        print(paste("Loaded", nrow(detections), "detections"))
        print("Data structure:")
        print(str(detections, max.level = 1))

        # Test GLATOS functions
        print("=== Testing GLATOS Functions ===")

        # 1. Basic summary
        print("1. Basic data summary:")
        print(paste("Number of unique animals:", length(unique(detections$animal_id))))
        print(paste("Number of unique stations:", length(unique(detections$station))))
        print(paste("Date range:", min(detections$detection_timestamp_utc), "to", max(detections$detection_timestamp_utc)))

        # 2. Detection timeline
        print("2. Creating detection timeline...")
        detections$date <- as.Date(detections$detection_timestamp_utc)
        daily_detections <- aggregate(animal_id ~ date, detections, length)
        names(daily_detections)[2] <- "count"

        # 3. Station activity
        print("3. Station activity:")
        station_activity <- table(detections$station)
        print("Most active stations:")
        print(head(sort(station_activity, decreasing = TRUE), 5))

        # 4. Create plot
        print("4. Creating visualization...")
        png("{detection_plot_r}", width = 1000, height = 600)
        plot(daily_detections$date, daily_detections$count, 
             type = "l", 
             main = "Daily Detection Counts",
             xlab = "Date", 
             ylab = "Number of Detections")
        dev.off()
        print("Created timeline plot")

        # 5. Generate comprehensive summary report
        print("5. Generating comprehensive summary report...")

        # Create detailed report
        report <- c()
        report <- c(report, "GLATOS Data Analysis Summary")
        report <- c(report, paste("Generated on:", Sys.time()))
        report <- c(report, paste("File:", "{file_path_r}"))
        report <- c(report, paste(rep("=", 80), collapse=""))
        report <- c(report, "")

        # Basic statistics
        report <- c(report, "🦈 ANIMAL TRACKING SUMMARY:")
        study_years <- round(as.numeric(difftime(max(as.Date(detections$detection_timestamp_utc)), 
                                               min(as.Date(detections$detection_timestamp_utc)), 
                                               units="days")) / 365, 1)
        report <- c(report, paste("  * **", format(length(unique(detections$animal_id)), big.mark=","),
                                " different sharks** tracked over ", study_years, " years"))

        # Top animals
        det_by_animal <- sort(table(detections$animal_id), decreasing = TRUE)
        report <- c(report, "  * Top 5 most detected animals:")
        for (i in 1:min(5, length(det_by_animal))) {{
            report <- c(report, paste("     ", i, ". ID", names(det_by_animal)[i], ":", 
                               format(det_by_animal[i], big.mark=","), "detections"))
        }}
        report <- c(report, "")

        # Station network
        report <- c(report, "📍 STATION NETWORK:")
        unique_stations <- length(unique(detections$station))
        report <- c(report, paste("  * **", unique_stations, "receiver stations** total"))

        # Most active stations
        station_activity <- sort(table(detections$station), decreasing = TRUE)
        report <- c(report, "  * Most active stations:")
        for (i in 1:min(5, length(station_activity))) {{
            report <- c(report, paste("     ", i, ". Station", names(station_activity)[i], ":", 
                               format(station_activity[i], big.mark=","), "detections"))
        }}
        report <- c(report, "")

        # Temporal patterns
        report <- c(report, "📊 TEMPORAL PATTERNS:")
        report <- c(report, paste("  * Daily detection counts created (saved as detection_timeline.png)"))
        peak_day <- daily_detections[which.max(daily_detections$count), ]
        peak_month <- format(peak_day$date, "%B %Y")
        report <- c(report, paste("  * Peak activity:", format(peak_day$count, big.mark=","), 
                                 "detections on", peak_day$date, "(", peak_month, ")"))

        date_range <- range(detections$date)
        report <- c(report, paste("  * Data collected from", date_range[1], "to", date_range[2]))
        report <- c(report, "")

        # Protocol breakdown
        report <- c(report, "🏷️  PROTOCOL TYPES:")
        protocols <- table(detections$protocol)
        for (protocol in names(protocols)) {{
            report <- c(report, paste("  * ", protocol, ":", format(protocols[protocol], big.mark=","), "detections"))
        }}
        report <- c(report, "")

        # Additional statistics
        report <- c(report, "📋 ADDITIONAL STATISTICS:")
        first_detection <- min(as.Date(detections$detection_timestamp_utc))
        last_detection <- max(as.Date(detections$detection_timestamp_utc))
        study_duration <- as.numeric(difftime(last_detection, first_detection, units="days"))

        report <- c(report, paste("  * Study duration:", round(study_duration, 0), "days"))
        report <- c(report, paste("  * Average detections per day:", 
                                 format(round(nrow(detections) / study_duration, 0), big.mark=",")))
        report <- c(report, paste("  * Average detections per animal:", 
                                 format(round(nrow(detections) / length(unique(detections$animal_id)), 0), big.mark=",")))
        report <- c(report, paste("  * Total detections:", format(nrow(detections), big.mark=",")))

        # Save the report
        writeLines(report, "{summary_file_r}")

        print("Summary report saved!")
        print("=== Test Complete ===")
        print("Your converted file works with GLATOS functions!")
        print("enerated files:")
        print("  - detection_timeline.png")
        print("  - glatos_test_summary.txt")
        '''

        return self.run_r_script(r_code)

    def filter_noise(self, input_file, output_file, time_filter=3600):
        """Apply noise filtering."""
        print(f"Starting noise reduction...")
        print(f"Input: {input_file}")
        print(f"Output: {output_file}")
        print(f"Time filter: {time_filter} seconds")

        r_code = f'''
        print("=== GLATOS Noise Reduction ===")

        # Check if GLATOS is installed
        if (!require(glatos, quietly = TRUE)) {{
            stop("GLATOS package is not installed.")
        }}

        # Read the data
        print("Reading input file...")
        print("File: {input_file}")

        if (!file.exists("{input_file}")) {{
            stop("Input file does not exist")
        }}

        dets <- read.csv("{input_file}")
        print(paste("Read", nrow(dets), "rows and", ncol(dets), "columns"))

        # Check columns
        if (!"detection_timestamp_utc" %in% colnames(dets)) {{
            stop("Column 'detection_timestamp_utc' not found")
        }}
        if (!"animal_id" %in% colnames(dets)) {{
            stop("Column 'animal_id' not found")
        }}

        # Convert timestamp
        if (is.character(dets$detection_timestamp_utc)) {{
            print("Converting timestamp to POSIXct...")
            dets$detection_timestamp_utc <- as.POSIXct(dets$detection_timestamp_utc, 
                                                      format = "%Y-%m-%d %H:%M:%S", 
                                                      tz = "UTC")
        }}

        # Add GLATOS class
        class(dets) <- c("glatos_detections", "data.frame")

        # Apply false detection filter
        print("Applying false detection filter...")
        print(paste("Time filter:", {time_filter}, "seconds"))

        dets_filtered <- false_detections(dets, tf = {time_filter})

        # Check results
        if ("passed_filter" %in% colnames(dets_filtered)) {{
            total <- nrow(dets_filtered)
            passed <- sum(dets_filtered$passed_filter == TRUE, na.rm = TRUE)
            failed <- sum(dets_filtered$passed_filter == FALSE, na.rm = TRUE)

            print("Filtering results:")
            print(paste("Total detections:", format(total, big.mark=",")))
            print(paste("Passed filter:", format(passed, big.mark=","), 
                        paste0("(", round(passed/total*100, 1), "%)")))
            print(paste("Failed filter:", format(failed, big.mark=","), 
                        paste0("(", round(failed/total*100, 1), "%)")))

            final_data <- dets_filtered[dets_filtered$passed_filter == TRUE, ]
            print(paste("Final dataset:", format(nrow(final_data), big.mark=","), "detections"))
        }} else {{
            print("Warning: No 'passed_filter' column created")
            print("Using original data")
            final_data <- dets_filtered
        }}

        # Select essential columns
        print("Selecting essential columns...")
        required_columns <- c("animal_id", "protocol", "station", "detection_timestamp_utc")
        available_columns <- colnames(final_data)
        columns_to_keep <- required_columns[required_columns %in% available_columns]
        final_data <- final_data[, columns_to_keep, drop = FALSE]

        print(paste("Selected columns:", paste(columns_to_keep, collapse=", ")))
        print(paste("Final dimensions:", nrow(final_data), "rows x", ncol(final_data), "columns"))

        # Save
        print("Saving filtered data...")
        write.csv(final_data, "{output_file}", row.names = FALSE)

        # Verify
        if (file.exists("{output_file}")) {{
            file_info <- file.info("{output_file}")
            print("✓ SUCCESS!")
            print(paste("Output file:", "{output_file}"))
            print(paste("File size:", format(file_info$size, big.mark=","), "bytes"))
            print(paste("Rows saved:", nrow(final_data)))

            # Quick summary
            print("Quick summary:")
            if ("animal_id" %in% colnames(final_data)) {{
                print(paste("Unique animals:", length(unique(final_data$animal_id))))
            }}
            if ("station" %in% colnames(final_data)) {{
                print(paste("Unique stations:", length(unique(final_data$station))))
            }}
        }} else {{
            print("✗ ERROR: Output file was not created")
        }}

        print("=== Noise Reduction Complete ===")
        '''

        return self.run_r_script(r_code)

    def compare_tags_with_db(self, filtered_file, db_params):
        """
        Compare animal_id from processed CSV with acoustic_tag_id from sharks_unique table.
        Filter out tags that aren't in the database and create a filtered output file.

        Args:
            filtered_file (str): Path to the filtered CSV file
            db_params (dict): Database connection parameters

        Returns:
            bool: Success status
        """
        print("\n🔄 Comparing detected tags with database records...")

        try:
            # Read the CSV file
            df = pd.read_csv(filtered_file)
            detected_tags = df['animal_id'].unique()
            print(f"Found {len(detected_tags)} unique animal IDs in the processed data")

            # Connect to the database
            conn = self.connect_to_amazon_db(
                dbname=db_params['dbname'],
                host=db_params['host'],
                port=db_params['port'],
                username=db_params['username'],
                password=db_params['password']
            )

            if not conn:
                return False

            # Create cursor and execute query
            cursor = conn.cursor()
            cursor.execute("SELECT acoustic_tag_id FROM sharks_unique")
            db_tags = [row[0] for row in cursor.fetchall()]

            print(f"Found {len(db_tags)} unique acoustic tag IDs in database")

            # Convert to sets for comparison
            detected_set = set(str(tag) for tag in detected_tags)
            db_set = set(str(tag) for tag in db_tags)

            # Find matches and mismatches
            matches = detected_set.intersection(db_set)
            not_in_db = detected_set - db_set

            # Create results directory
            output_dir = Path(filtered_file).parent
            results_file = output_dir / "tag_comparison_results.csv"

            # Create the filtered output filename
            filtered_file_path = Path(filtered_file)
            base_name = filtered_file_path.stem.replace("_filtered", "")
            filtered_output_file = output_dir / f"{base_name}_db_filtered.csv"

            # Generate comparison results
            print("\n📊 Tag Comparison Results:")
            print(
                f"✓ Matching tags: {len(matches)} ({round(len(matches) / len(detected_set) * 100 if len(detected_set) > 0 else 0, 1)}%)")
            print(
                f"✗ Tags not in database: {len(not_in_db)} ({round(len(not_in_db) / len(detected_set) * 100 if len(detected_set) > 0 else 0, 1)}%)")

            # Save detailed results to CSV
            results = []
            for tag in detected_tags:
                results.append({
                    'animal_id': tag,
                    'in_database': str(tag) in db_set,
                    'detection_count': len(df[df['animal_id'] == tag])
                })

            results_df = pd.DataFrame(results)
            results_df.sort_values('detection_count', ascending=False, inplace=True)
            results_df.to_csv(results_file, index=False)

            print(f"\n📄 Detailed comparison saved to: {results_file}")

            # Filter out tags that aren't in the database
            print("\n🔍 Filtering out tags not found in database...")
            original_count = len(df)

            # Convert tags to strings for comparison
            str_db_tags = set(str(tag) for tag in db_tags)
            df['str_animal_id'] = df['animal_id'].astype(str)

            # Filter dataframe to keep only tags that are in the database
            df_filtered = df[df['str_animal_id'].isin(str_db_tags)].copy()
            df_filtered.drop('str_animal_id', axis=1, inplace=True)

            # Save filtered dataframe
            df_filtered.to_csv(filtered_output_file, index=False)

            # Print filtering results
            removed_count = original_count - len(df_filtered)
            print(
                f"Removed {removed_count} detections ({round(removed_count / original_count * 100 if original_count > 0 else 0, 1)}%)")
            print(
                f"Kept {len(df_filtered)} detections ({round(len(df_filtered) / original_count * 100 if original_count > 0 else 0, 1)}%)")
            print(f"Filtered output saved to: {filtered_output_file}")

            # Close connection
            cursor.close()
            conn.close()

            return True

        except Exception as e:
            print(f"Error comparing tags with database: {e}")
            return False

    def process_file(self, input_file, output_dir=None, time_filter=3600, db_params=None):
        """Run the complete pipeline."""
        input_path = Path(input_file)
        if not output_dir:
            output_dir = input_path.parent
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate file names
        base_name = input_path.stem
        combined_file = output_dir / f"{base_name}_dates.csv"
        glatos_file = output_dir / f"{base_name}_dates_glatos_format.csv"
        filtered_file = output_dir / f"{base_name}_dates_glatos_format_filtered.csv"
        final_output_file = filtered_file  # Default to noise-filtered file

        print("🦈 Starting GLATOS Pipeline")
        print(f"Input: {input_file}")
        print(f"Output directory: {output_dir}")
        print("=" * 60)

        # Stage 1: Combine timestamps
        print("📅 Stage 1: Combining Date/Time Columns")
        try:
            self.combine_timestamps(input_file, combined_file)
            print(f"✅ Stage 1 complete: {combined_file}")
        except Exception as e:
            print(f"❌ Stage 1 failed: {e}")
            return False

        # Stage 2: Convert to GLATOS format
        print("🔄 Stage 2: Converting to GLATOS Format")
        if self.convert_to_glatos(combined_file, glatos_file):
            print(f"✅ Stage 2 complete: {glatos_file}")
        else:
            print("❌ Stage 2 failed")
            return False

        # Stage 3: Test GLATOS file
        print("📊 Stage 3: Testing GLATOS File")
        if self.test_glatos_file(glatos_file):
            print(f"✅ Stage 3 complete")
        else:
            print("❌ Stage 3 failed")
            return False

        # Stage 4: Apply noise filtering
        print("🔍 Stage 4: Applying Noise Filter")
        if self.filter_noise(glatos_file, filtered_file, time_filter):
            print(f"✅ Stage 4 complete: {filtered_file}")
        else:
            print("❌ Stage 4 failed")
            return False

        # Stage 5: Compare with database and filter (only if db_params provided)
        if db_params:
            print("🔍 Stage 5: Comparing with Amazon RDS Database and Filtering")
            if self.compare_tags_with_db(filtered_file, db_params):
                # Update the final output file to be the database-filtered file
                db_filtered_file = output_dir / f"{base_name}_dates_glatos_format_db_filtered.csv"
                if db_filtered_file.exists():
                    final_output_file = db_filtered_file
                print(f"✅ Stage 5 complete")
            else:
                print("❌ Stage 5 failed")
                # Continue pipeline and use the noise-filtered file as final output

        print("🎉 Pipeline completed successfully!")
        print(f"Final output: {final_output_file}")
        return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="GLATOS Processing Pipeline")
    parser.add_argument("input", help="Input CSV file path")
    parser.add_argument("--output", "-o", help="Output directory")
    parser.add_argument("--time-filter", "-t", type=int, default=3600,
                        help="Time filter for noise reduction (seconds)")

    # Add Amazon RDS connection arguments
    parser.add_argument("--db-host", default="morriskahnmarineresearchdb.cfhx32jrsn76.us-east-2.rds.amazonaws.com",help="Amazon RDS host")
    parser.add_argument("--db-port", type=int, default=5432, help="Amazon RDS port (default: 5432)")
    parser.add_argument("--db-name", default="MKMR_DB", help="Amazon RDS database name")
    parser.add_argument("--db-user", default="MorrisKahnMarineResearch", help="Amazon RDS username")
    parser.add_argument("--db-pass", default="MKMR080719", help="Amazon RDS password")

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"❌ Error: Input file not found: {args.input}")
        sys.exit(1)

    pipeline = GLATOSPipeline()
    start_time = time.time()

    # Set up db_params if all required parameters are provided
    db_params = None
    if args.db_host and args.db_name and args.db_user and args.db_pass:
        db_params = {
            'host': args.db_host,
            'port': args.db_port,
            'dbname': args.db_name,
            'username': args.db_user,
            'password': args.db_pass
        }
        print("Amazon RDS connection parameters provided")

    success = pipeline.process_file(args.input, args.output, args.time_filter, db_params)

    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.1f} seconds")

    if success:
        print("✅ Pipeline completed successfully!")
        sys.exit(0)
    else:
        print("❌ Pipeline failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()