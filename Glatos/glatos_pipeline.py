#!/usr/bin/env python3
"""
GLATOS Processing Pipeline
A complete pipeline for processing shark acoustic telemetry data.

Pipeline stages:
1. Combine date/time columns into detection_timestamp_utc
2. Convert to GLATOS format
3. Test the converted file
4. Apply noise filtering

Usage:
    python glatos_pipeline.py <input.csv> [options]
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import time


class GLATOSPipeline:
    def __init__(self, input_file, output_dir=None, time_filter=3600):
        self.input_file = Path(input_file)

        # Handle output directory correctly
        if output_dir:
            self.output_dir = Path(output_dir)
            # If output_dir is specified but looks like a file name, use its parent
            if self.output_dir.suffix:
                self.output_dir = self.output_dir.parent
        else:
            self.output_dir = self.input_file.parent

        self.time_filter = time_filter

        # Generate file names for each stage
        self.base_name = self.input_file.stem
        self.combined_file = self.output_dir / f"{self.base_name}_dates.csv"
        self.glatos_file = self.output_dir / f"{self.base_name}_dates_glatos_format.csv"
        self.filtered_file = self.output_dir / f"{self.base_name}_dates_glatos_format_filtered.csv"

        # Find script paths (assume they're in the same directory)
        script_dir = Path(__file__).parent
        self.csv_combiner = script_dir / "csv_timestamp_combiner.py"
        self.glatos_runner = script_dir / "glatos_runner.py"
        self.noise_filter = script_dir / "noise_filter.py"

    def run_command(self, cmd, description):
        """Run a command and handle errors."""
        print(f"\n{'=' * 60}")
        print(f"STAGE: {description}")
        print('=' * 60)

        print(f"Running: {' '.join(cmd)}")
        start_time = time.time()

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            elapsed = time.time() - start_time

            if result.returncode == 0:
                print(f"✅ SUCCESS ({elapsed:.1f}s)")
                if result.stdout:
                    print(result.stdout)
            else:
                print(f"❌ ERROR ({elapsed:.1f}s)")
                print(result.stderr)
                return False
        except FileNotFoundError as e:
            print(f"❌ ERROR: Script not found - {e}")
            return False
        except Exception as e:
            print(f"❌ ERROR: {e}")
            return False

        return True

    def check_file_exists(self, file_path, stage_name):
        """Check if a file exists and show its size."""
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"✅ {stage_name} output created: {file_path}")
            print(f"   File size: {size:,} bytes")
            return True
        else:
            print(f"❌ {stage_name} output not found: {file_path}")
            return False

    def run_pipeline(self):
        """Run the complete pipeline."""
        print(f"🦈 GLATOS PROCESSING PIPELINE")
        print(f"Input file: {self.input_file}")
        print(f"Output directory: {self.output_dir}")
        print(f"Time filter: {self.time_filter} seconds")

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Stage 1: Combine timestamp columns
        if not self.run_command([
            "python", str(self.csv_combiner),
            str(self.input_file),
            "--output", str(self.combined_file)
        ], "1. Combining Date/Time Columns"):
            return False

        if not self.check_file_exists(self.combined_file, "Stage 1"):
            return False

        # Stage 2: Convert to GLATOS format
        if not self.run_command([
            "python", str(self.glatos_runner),
            "convert",
            str(self.combined_file),
            str(self.glatos_file)
        ], "2. Converting to GLATOS Format"):
            return False

        if not self.check_file_exists(self.glatos_file, "Stage 2"):
            return False

        # Stage 3: Test the converted file
        if not self.run_command([
            "python", str(self.glatos_runner),
            "test",
            str(self.glatos_file)
        ], "3. Testing GLATOS Conversion"):
            return False

        # Stage 4: Apply noise filtering
        if not self.run_command([
            "python", str(self.noise_filter),
            str(self.glatos_file),
            str(self.filtered_file),
            str(self.time_filter)
        ], "4. Applying Noise Filter"):
            return False

        if not self.check_file_exists(self.filtered_file, "Stage 4"):
            return False

        # Final summary
        print(f"\n{'=' * 60}")
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print('=' * 60)
        print(f"Final output: {self.filtered_file}")

        if self.filtered_file.exists():
            size = self.filtered_file.stat().st_size
            print(f"Final file size: {size:,} bytes")

        print("\nGenerated files:")
        for stage, file_path in [
            ("1. Combined timestamps", self.combined_file),
            ("2. GLATOS format", self.glatos_file),
            ("3. Test results", self.output_dir / "glatos_test_summary.txt"),
            ("4. Filtered data", self.filtered_file)
        ]:
            if file_path.exists():
                print(f"  ✅ {stage}: {file_path.name}")
            else:
                print(f"  ❌ {stage}: Not created")

        return True


def main():
    parser = argparse.ArgumentParser(
        description="Complete GLATOS processing pipeline for acoustic telemetry data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python glatos_pipeline.py input.csv

  # Specify output directory
  python glatos_pipeline.py input.csv --output /path/to/output

  # Custom time filter (default: 3600 seconds)
  python glatos_pipeline.py input.csv --time-filter 1800
        """
    )

    parser.add_argument("input", help="Input CSV file with detection data")
    parser.add_argument("--output", "-o", help="Output directory (default: same as input)")
    parser.add_argument("--time-filter", "-t", type=int, default=3600,
                        help="Time filter for noise reduction in seconds (default: 3600)")

    args = parser.parse_args()

    # Check if input file exists
    if not Path(args.input).exists():
        print(f"❌ ERROR: Input file not found: {args.input}")
        sys.exit(1)

    # Run the pipeline
    pipeline = GLATOSPipeline(args.input, args.output, args.time_filter)

    start_time = time.time()
    success = pipeline.run_pipeline()
    total_time = time.time() - start_time

    print(f"\nTotal pipeline execution time: {total_time:.1f} seconds")

    if success:
        print("✅ Pipeline completed successfully!")
        sys.exit(0)
    else:
        print("❌ Pipeline failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()