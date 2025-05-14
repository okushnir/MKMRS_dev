#!/usr/bin/env python3
"""
Python script to run the GLATOS R package for aquatic telemetry data analysis.
GLATOS (Great Lakes Acoustic Telemetry Observation System) is an R package
for managing and analyzing acoustic telemetry data.
"""

import os
import sys
import subprocess
from pathlib import Path

# Try to import rpy2 for R integration
try:
    import rpy2.robjects as robjects
    from rpy2.robjects import r, pandas2ri
    from rpy2.robjects.packages import importr

    pandas2ri.activate()
    HAS_RPY2 = True
except ImportError as e:
    print(f"rpy2 is not available: {e}")
    print("Using alternative approach with subprocess.")
    HAS_RPY2 = False
except OSError as e:
    if "incompatible architecture" in str(e):
        print("Architecture mismatch detected between Python and R.")
        print("Python and R have different architectures (x86_64 vs ARM64).")
        print("Using subprocess approach instead.")
    else:
        print(f"Error loading rpy2: {e}")
    HAS_RPY2 = False


class GLATOSRunner:
    """
    A class to run GLATOS R package operations from Python.
    """

    def __init__(self):
        self.glatos = None
        self.setup_r_environment()

    def setup_r_environment(self):
        """Set up R environment and import GLATOS package."""
        if HAS_RPY2:
            try:
                # Check architecture compatibility
                import platform
                python_arch = platform.machine()

                # Try to get R architecture
                r_arch_result = subprocess.run(['R', '--slave', '-e', 'cat(R.version$arch)'],
                                               capture_output=True, text=True)
                r_arch = r_arch_result.stdout.strip() if r_arch_result.returncode == 0 else "unknown"

                print(f"Python architecture: {python_arch}")
                print(f"R architecture: {r_arch}")

                # Import required R packages
                base = importr('base')
                utils = importr('utils')

                # Check if GLATOS is installed, install if not
                r('''
                if (!require(glatos, quietly = TRUE)) {
                    if (!require(remotes, quietly = TRUE)) {
                        install.packages("remotes")
                    }
                    remotes::install_github("ocean-tracking-network/glatos")
                    library(glatos)
                }
                ''')

                # Import GLATOS
                self.glatos = importr('glatos')
                print("GLATOS package loaded successfully via rpy2")

            except Exception as e:
                print(f"Error setting up GLATOS with rpy2: {e}")
                if "incompatible architecture" in str(e):
                    print("\n*** ARCHITECTURE MISMATCH DETECTED ***")
                    print("Your Python and R installations have different architectures:")
                    print("- This typically happens on Apple Silicon Macs")
                    print("- Python may be running through Rosetta (x86_64) while R is native ARM64")
                    print("- Falling back to subprocess method...\n")
                self.glatos = None
        else:
            # Check if R and GLATOS are available via subprocess
            try:
                result = subprocess.run(['R', '--version'],
                                        capture_output=True, text=True)
                if result.returncode == 0:
                    print("R is available. Will use subprocess approach.")
                else:
                    print("R is not available in PATH")
            except FileNotFoundError:
                print("R is not installed or not in PATH")

    def run_r_script(self, r_code, suppress_output=False):
        """Execute R code using subprocess."""
        try:
            # Create temporary R script
            temp_script = "temp_glatos_script.R"
            with open(temp_script, 'w') as f:
                f.write(r_code)

            # Run R script
            result = subprocess.run(['Rscript', temp_script],
                                    capture_output=True, text=True)

            # Clean up
            os.remove(temp_script)

            if result.returncode == 0:
                if not suppress_output:
                    print("R script executed successfully")
                    if result.stdout:
                        print("Output:", result.stdout)
            else:
                print("Error running R script:")
                print(result.stderr)

            return result

        except Exception as e:
            print(f"Error running R script: {e}")
            return None

    def install_glatos_package(self):
        """Install GLATOS package using R subprocess if not already installed."""
        print("Checking if GLATOS is installed...")

        # Check if GLATOS is installed
        check_code = '''
        if (!require(glatos, quietly = TRUE)) {
            cat("not_installed")
        } else {
            cat("installed")
        }
        '''

        result = self.run_r_script(check_code, suppress_output=True)

        if result and "not_installed" in result.stdout:
            print("GLATOS not found. Installing GLATOS package...")

            install_code = '''
            # Install required packages
            if (!require(remotes, quietly = TRUE)) {
                install.packages("remotes", repos="https://cran.r-project.org/")
            }

            # Install GLATOS from GitHub
            remotes::install_github("ocean-tracking-network/glatos", dependencies = TRUE)

            # Verify installation
            if (require(glatos, quietly = TRUE)) {
                cat("GLATOS installed successfully!\\n")
            } else {
                cat("ERROR: Failed to install GLATOS\\n")
            }
            '''

            self.run_r_script(install_code)
        else:
            print("GLATOS is already installed.")

    def convert_to_glatos_format(self, input_file, output_file=None):
        """Convert a CSV file to GLATOS format."""
        if output_file is None:
            output_file = input_file.replace('.csv', '_glatos_format.csv')

        convert_code = f'''
        library(glatos)

        # Read the original file
        print("Reading original file...")
        raw_data <- read.csv("{input_file}")
        print(paste("Original file has", nrow(raw_data), "rows and", ncol(raw_data), "columns"))
        print("Original column names:")
        print(colnames(raw_data))
        print("\\nFirst few rows of original data:")
        print(head(raw_data))

        # Create a mapping for common column name variations
        print("\\nAnalyzing column names for potential matches...")
        cols <- colnames(raw_data)

        # Initialize the GLATOS format dataframe
        glatos_data <- raw_data

        # Try to identify and rename key columns
        # Animal/Fish/Tag ID (map 'id' to 'animal_id')
        if ("id" %in% cols) {{
            colnames(glatos_data)[colnames(glatos_data) == "id"] <- "animal_id"
            print("Mapped id -> animal_id")
        }}

        # Station/Receiver columns
        if ("receiver" %in% cols) {{
            colnames(glatos_data)[colnames(glatos_data) == "receiver"] <- "station"
            print("Mapped receiver -> station")
        }}

        # Transmitter/Tag code - use protocol as transmitter_id if available
        if ("protocol" %in% cols) {{
            # Extract the frequency part for transmitter_codespace
            glatos_data$transmitter_codespace <- raw_data$protocol
            glatos_data$transmitter_id <- as.character(glatos_data$animal_id)
            print("Added transmitter_codespace and transmitter_id based on protocol and animal_id")
        }}

        # Add required columns that GLATOS expects
        print("\\nAdding required GLATOS columns...")

        # Add glatos-specific columns with default values
        if (!"glatos_array" %in% colnames(glatos_data)) {{
            glatos_data$glatos_array <- "ARRAY1"  # Default array name
            print("Added default glatos_array")
        }}

        if (!"station_no" %in% colnames(glatos_data)) {{
            # Extract station number from station if it's numeric
            glatos_data$station_no <- as.character(glatos_data$station)
            print("Added station_no based on station")
        }}

        # Add other commonly expected columns
        glatos_data$sensor_value <- NA
        glatos_data$sensor_unit <- NA
        glatos_data$deploy_lat <- NA  # You'll need to add actual coordinates
        glatos_data$deploy_long <- NA
        glatos_data$receiver_sn <- paste0("SN", glatos_data$station)
        glatos_data$tag_type <- NA
        glatos_data$tag_model <- NA
        glatos_data$tag_serial_number <- NA
        glatos_data$common_name_e <- "shark"  # You should replace with actual species name
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

        print("Added additional GLATOS-required columns with default values")

        # Ensure proper data types
        print("\\nConverting data types...")

        # Convert detection_timestamp_utc to POSIXct if it's not already
        if (!inherits(glatos_data$detection_timestamp_utc, "POSIXct")) {{
            glatos_data$detection_timestamp_utc <- as.POSIXct(glatos_data$detection_timestamp_utc, 
                                                              format = "%Y-%m-%d %H:%M:%S", 
                                                              tz = "UTC")
            print("Converted detection_timestamp_utc to POSIXct")
        }}

        # Save the converted file
        write.csv(glatos_data, "{output_file}", row.names = FALSE)
        print(paste("Saved converted file:", "{output_file}"))

        # Show the new structure
        print("\\nNew column names:")
        print(colnames(glatos_data))
        print("\\nFirst few rows of converted data:")
        print(head(glatos_data))

        # Try to read it as GLATOS format
        print("\\nTesting if the converted file works with GLATOS...")
        tryCatch({{
            test_detections <- read_glatos_detections("{output_file}")
            print("SUCCESS! Converted file can be read by GLATOS")
            print(paste("Contains", nrow(test_detections), "detections"))

            # Show the structure of the successfully loaded data
            print("\\nStructure of loaded GLATOS data:")
            print(str(test_detections))
        }}, error = function(e) {{
            print(paste("Still having issues:", e$message))
            print("\\nTroubleshooting steps:")
            print("1. Check if the file was created successfully")
            print("2. Manual column mapping may be needed")
            print("3. Consider using read.csv() and manual conversion")

            print("\\nTo use this data with GLATOS, you can:")
            print("1. Read it as a regular CSV: data <- read.csv('file.csv')")
            print("2. Add the glatos_detections class: class(data) <- c('glatos_detections', 'data.frame')")
            print("3. Use GLATOS functions with this data")
        }})
        '''

        if HAS_RPY2 and self.glatos is not None:
            r(convert_code)
        else:
            self.run_r_script(convert_code)

    def test_converted_glatos_file(self, file_path):
        """Test using a converted GLATOS file with actual GLATOS functions."""
        # Get the directory of the input file for saving all outputs
        input_dir = Path(file_path).parent
        summary_file = input_dir / "glatos_test_summary.txt"
        detection_plot = input_dir / "detection_timeline.png"
        residence_index_file = input_dir / "residence_index_results.csv"

        test_code = f'''
        library(glatos)

        # Method 1: Read as CSV and manually add GLATOS class
        print("Testing converted GLATOS file...")
        detections <- read.csv("{file_path}")

        # Add the GLATOS class
        class(detections) <- c("glatos_detections", "data.frame")

        print(paste("Loaded", nrow(detections), "detections"))
        print("Data structure:")
        print(str(detections, max.level = 1))

        # Test GLATOS functions with your data
        print("\\n=== Testing GLATOS Functions ===")

        # 1. Basic summary
        print("\\n1. Basic data summary:")
        print(paste("Number of unique animals:", length(unique(detections$animal_id))))
        print(paste("Number of unique stations:", length(unique(detections$station))))
        print(paste("Date range:", min(detections$detection_timestamp_utc), "to", max(detections$detection_timestamp_utc)))

        # 2. Detection summary by animal
        print("\\n2. Detections by animal ID:")
        det_summary <- table(detections$animal_id)
        print(head(sort(det_summary, decreasing = TRUE), 10))

        # 3. Detection timeline
        print("\\n3. Creating detection timeline...")
        detections$date <- as.Date(detections$detection_timestamp_utc)
        daily_detections <- aggregate(animal_id ~ date, detections, length)
        names(daily_detections)[2] <- "count"
        print("Daily detection counts (first 10 days):")
        print(head(daily_detections, 10))

        # 4. Station activity
        print("\\n4. Station activity:")
        station_activity <- table(detections$station)
        print("Most active stations:")
        print(head(sort(station_activity, decreasing = TRUE), 5))

        # 5. Attempt residence index calculation
        print("\\n5. Attempting residence index calculation...")
        tryCatch({{
            ri <- residence_index(detections)
            print("Successfully calculated residence index!")
            print(head(ri))

            # Save results in the same directory as input file
            write.csv(ri, "{str(residence_index_file)}", row.names = FALSE)
            print("Saved residence index to: {residence_index_file.name}")
        }}, error = function(e) {{
            print(paste("Error calculating residence index:", e$message))
            print("This might be due to all detections being at one station")
        }})

        # 6. Create simple plots
        print("\\n6. Creating visualization...")
        tryCatch({{
            png("{str(detection_plot)}", width = 1000, height = 600)
            plot(daily_detections$date, daily_detections$count, 
                 type = "l", 
                 main = "Daily Detection Counts",
                 xlab = "Date", 
                 ylab = "Number of Detections")
            dev.off()
            print("Created timeline plot: {detection_plot.name}")
        }}, error = function(e) {{
            print(paste("Error creating plot:", e$message))
        }})

        # 7. Generate comprehensive summary report
        print("\\n7. Generating comprehensive summary report...")

        # Create summary report
        report <- c()
        report <- c(report, "GLATOS Data Analysis Summary")
        report <- c(report, paste("Generated on:", Sys.time()))
        report <- c(report, paste("File:", "{file_path}"))
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
        report <- c(report, paste("  * Daily detection counts created (saved as", "{detection_plot.name})"))
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

        # Save the report to the same directory as input file
        writeLines(report, "{str(summary_file)}")

        print("\\nSummary report saved to: {summary_file.name}")

        print("\\n=== Test Complete ===")
        print("Your converted file works with GLATOS functions!")
        print("\\nGenerated files:")
        print("  - {detection_plot.name}")
        print("  - {summary_file.name}")
        if (file.exists("{str(residence_index_file)}")) {{
            print("  - {residence_index_file.name}")
        }}
        '''

        if HAS_RPY2 and self.glatos is not None:
            r(test_code)
        else:
            self.run_r_script(test_code)

    def example_glatos_workflow(self, data_path=None):
        """Run an example GLATOS workflow for acoustic telemetry data analysis."""
        # First, install GLATOS if needed
        self.install_glatos_package()

        r_code = f'''
        library(glatos)

        print("Starting GLATOS example workflow...")

        # Load example detection data
        det_file <- system.file("extdata", "walleye_detections.csv", package = "glatos")
        detections <- read_glatos_detections(det_file)
        print(paste("Loaded", nrow(detections), "detections"))

        # Load example receiver locations
        rec_file <- system.file("extdata", "sample_receivers.csv", package = "glatos")
        receivers <- read_glatos_receivers(rec_file)
        print(paste("Loaded", nrow(receivers), "receiver locations"))

        # Basic data summary and structure
        print("Detection data structure:")
        print(str(detections))

        print("\\nGLATOS workflow completed!")
        '''

        if HAS_RPY2 and self.glatos is not None:
            try:
                r(r_code)
            except Exception as e:
                print(f"Error running with rpy2: {e}")
                self.run_r_script(r_code)
        else:
            self.run_r_script(r_code)


def main():
    """Main function to demonstrate GLATOS usage."""
    print("Python script for running GLATOS R package")
    print("=" * 50)

    # Initialize GLATOS runner
    runner = GLATOSRunner()

    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "example":
            print("Running example GLATOS workflow...")
            runner.example_glatos_workflow()

        elif command == "convert":
            if len(sys.argv) > 2:
                input_file = sys.argv[2]
                output_file = sys.argv[3] if len(sys.argv) > 3 else None
                print(f"Converting {input_file} to GLATOS format...")
                runner.convert_to_glatos_format(input_file, output_file)
            else:
                print("Please specify an input file to convert")
                return

        elif command == "test":
            if len(sys.argv) > 2:
                file_path = sys.argv[2]
                print(f"Testing converted GLATOS file {file_path}...")
                runner.test_converted_glatos_file(file_path)
            else:
                print("Please specify a converted GLATOS file to test")
                return

        else:
            print(f"Unknown command: {command}")
            print_usage()
            return
    else:
        print("No command specified. Running example workflow...")
        runner.example_glatos_workflow()


def print_usage():
    """Print usage instructions."""
    print("\nUsage:")
    print("python glatos_runner.py [command] [options]")
    print("\nCommands:")
    print("  example          - Run example GLATOS workflow")
    print("  convert [input] [output] - Convert CSV to GLATOS format")
    print("  test [file]      - Test converted GLATOS file with GLATOS functions")
    print("\nExamples:")
    print("  python glatos_runner.py example")
    print("  python glatos_runner.py convert my_data.csv glatos_data.csv")
    print("  python glatos_runner.py test my_data_glatos_format.csv")


if __name__ == "__main__":
    # Check dependencies
    print("Checking dependencies...")

    # Check if rpy2 is available
    if not HAS_RPY2:
        print("\nOptional dependency 'rpy2' not found.")
        print("Will use subprocess approach instead.\n")
    else:
        print("rpy2 found. Will attempt direct R integration.")

    # Check if R is available
    try:
        result = subprocess.run(['R', '--version'],
                                capture_output=True, text=True)
        if result.returncode != 0:
            raise FileNotFoundError
        print("R is installed and accessible via PATH")
    except FileNotFoundError:
        print("ERROR: R is not installed or not in PATH")
        print("Please install R to use this script.")
        sys.exit(1)

    # Architecture compatibility check on macOS
    if sys.platform == 'darwin':  # macOS
        import platform

        python_arch = platform.machine()

        # Get R architecture
        r_arch_result = subprocess.run(['R', '--slave', '-e', 'cat(R.version$arch)'],
                                       capture_output=True, text=True)
        r_arch = r_arch_result.stdout.strip() if r_arch_result.returncode == 0 else "unknown"

        print(f"\nArchitecture information:")
        print(f"Python architecture: {python_arch}")
        print(f"R architecture: {r_arch}")

        if python_arch != r_arch and HAS_RPY2:
            print(f"\nWARNING: Architecture mismatch detected!")
            print(f"Python ({python_arch}) and R ({r_arch}) have different architectures.")
            print("This will prevent rpy2 from working properly.")
            print("The script will use subprocess approach (less efficient but works)")
            print()

    # Run main function
    main()