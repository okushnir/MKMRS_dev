🦈 GLATOS Pipeline - Shark Tracking Data Processor

REQUIREMENTS:
- R must be installed on your computer
  Download from: https://cran.r-project.org/

COMMAND LINE USAGE:
  GLATOSPipeline.exe "path/to/your/data.csv"

OPTIONS:
  --output "path/to/output/folder"  # Specify output directory
  --time-filter 3600                # Time filter in seconds (default: 1 hour)

EXAMPLES:
  GLATOSPipeline.exe "C:\Data\sharks.csv"
  GLATOSPipeline.exe "C:\Data\sharks.csv" --output "C:\Results"
  GLATOSPipeline.exe "C:\Data\sharks.csv" --time-filter 1800

INPUT FORMAT:
Your CSV file should have columns: date, time, id, protocol, receiver

OUTPUT:
The pipeline creates filtered GLATOS-compatible data with comprehensive analysis reports.

For GUI version, double-click GLATOSPipeline_GUI.exe