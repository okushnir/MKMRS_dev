#!/usr/bin/env python3
"""
GUI Wrapper for GLATOS Pipeline
Simple file browser interface for the GLATOS pipeline.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import subprocess
import os
import sys
from pathlib import Path
import threading
import webbrowser


# Ensure common R installation paths are in PATH
def setup_environment():
    current_path = os.environ.get('PATH', '')
    r_paths = ['/usr/local/bin', '/usr/bin', '/opt/homebrew/bin']
    for r_path in r_paths:
        if r_path not in current_path and os.path.exists(r_path):
            os.environ['PATH'] = f"{r_path}:{current_path}"

# Call this before anything else
setup_environment()
class GLATOSPipelineGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("GLATOS Pipeline - Shark Tracking Data Processor")
        self.root.geometry("600x450")

        # Variables
        self.input_file = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.time_filter = tk.IntVar(value=3600)

        self.create_widgets()

        # Check R installation after GUI is created
        self.root.after(1000, self.check_r_status)

    def create_widgets(self):
        # Create menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Install R", command=self.show_r_installation_guide)
        help_menu.add_command(label="About GLATOS", command=self.show_about_glatos)
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self.show_about)

        # Title
        title_label = tk.Label(self.root, text="🦈 GLATOS Pipeline",
                               font=("Arial", 18, "bold"))
        title_label.pack(pady=10)

        # R Installation status
        r_frame = tk.Frame(self.root)
        r_frame.pack(pady=5, padx=20, fill="x")

        self.r_status_label = tk.Label(r_frame, text="Checking R installation...",
                                       font=("Arial", 10))
        self.r_status_label.pack(side="left")

        self.check_r_button = tk.Button(r_frame, text="Check R",
                                        command=self.handle_r_button_click,
                                        font=("Arial", 9))
        self.check_r_button.pack(side="right")

        # Input file selection
        input_frame = tk.Frame(self.root)
        input_frame.pack(pady=10, padx=20, fill="x")

        tk.Label(input_frame, text="Input CSV File:", font=("Arial", 12)).pack(anchor="w")

        file_frame = tk.Frame(input_frame)
        file_frame.pack(fill="x", pady=5)

        tk.Entry(file_frame, textvariable=self.input_file, width=50).pack(side="left", fill="x", expand=True)
        tk.Button(file_frame, text="Browse", command=self.browse_input_file).pack(side="right")

        # Output directory selection
        output_frame = tk.Frame(self.root)
        output_frame.pack(pady=10, padx=20, fill="x")

        tk.Label(output_frame, text="Output Directory (optional):", font=("Arial", 12)).pack(anchor="w")

        dir_frame = tk.Frame(output_frame)
        dir_frame.pack(fill="x", pady=5)

        tk.Entry(dir_frame, textvariable=self.output_dir, width=50).pack(side="left", fill="x", expand=True)
        tk.Button(dir_frame, text="Browse", command=self.browse_output_dir).pack(side="right")

        # Time filter setting
        filter_frame = tk.Frame(self.root)
        filter_frame.pack(pady=10, padx=20, fill="x")

        tk.Label(filter_frame, text="Time Filter (seconds):", font=("Arial", 12)).pack(anchor="w")
        tk.Scale(filter_frame, from_=600, to=7200, orient="horizontal",
                 variable=self.time_filter, resolution=300).pack(fill="x", pady=5)
        tk.Label(filter_frame, text="600s = 10min, 3600s = 1hour, 7200s = 2hours",
                 font=("Arial", 9), fg="gray").pack()

        # Process button
        self.process_button = tk.Button(self.root, text="🚀 Process Data",
                                        command=self.start_processing,
                                        font=("Arial", 14, "bold"),
                                        bg="#4CAF50", fg="white",
                                        padx=20, pady=10)
        self.process_button.pack(pady=20)

        # Status label
        self.status_label = tk.Label(self.root, text="Ready to process",
                                     font=("Arial", 12), fg="blue")
        self.status_label.pack()

        # Progress bar
        self.progress = ttk.Progressbar(self.root, mode='indeterminate')
        self.progress.pack(pady=5, padx=20, fill="x")

        # Progress percentage label
        self.progress_label = tk.Label(self.root, text="", font=("Arial", 10))
        self.progress_label.pack()

        # Log area
        log_frame = tk.Frame(self.root)
        log_frame.pack(pady=10, padx=20, fill="both", expand=True)

        tk.Label(log_frame, text="Processing Log:", font=("Arial", 12)).pack(anchor="w")

        self.log_text = tk.Text(log_frame, height=8, width=70)
        scrollbar = tk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)

        self.log_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def handle_r_button_click(self):
        """Handle R button click - either check R or show installation guide."""
        if self.check_r_button.cget("text") == "Install R":
            self.show_r_installation_guide()
        else:
            self.check_r_status()

    def check_r_status(self):
        """Check and display R installation status with multiple methods."""
        print("DEBUG: Starting R status check...")  # Debug print

        # Method 1: Try just 'R'
        try:
            result = subprocess.run(['R', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0]
                self.r_status_label.config(text=f"✅ {version_line}", fg="green")
                self.check_r_button.config(text="R OK", state="normal")
                print("DEBUG: R found with simple command")  # Debug print
                return True
        except FileNotFoundError:
            print("DEBUG: Simple 'R' command not found")  # Debug print
        except Exception as e:
            print(f"DEBUG: Error with simple R command: {e}")  # Debug print

        # Method 2: Try full paths
        r_paths = ['/usr/local/bin/R', '/usr/bin/R', '/opt/homebrew/bin/R']
        for r_path in r_paths:
            if os.path.exists(r_path):
                print(f"DEBUG: Trying {r_path}")  # Debug print
                try:
                    result = subprocess.run([r_path, '--version'], capture_output=True, text=True)
                    if result.returncode == 0:
                        version_line = result.stdout.split('\n')[0]
                        self.r_status_label.config(text=f"✅ {version_line} (at {r_path})", fg="green")
                        self.check_r_button.config(text="R OK", state="normal")
                        # Store the working R path for later use
                        self.r_executable = r_path
                        print(f"DEBUG: R found at {r_path}")  # Debug print
                        return True
                except Exception as e:
                    print(f"DEBUG: Error with {r_path}: {e}")  # Debug print

        # Method 3: Try with shell=True
        try:
            print("DEBUG: Trying with shell=True")  # Debug print
            result = subprocess.run('R --version', shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0]
                self.r_status_label.config(text=f"✅ {version_line} (via shell)", fg="green")
                self.check_r_button.config(text="R OK", state="normal")
                self.r_executable = 'R'  # Use simple 'R' with shell=True
                self.use_shell = True  # Flag to remember to use shell=True
                print("DEBUG: R found via shell")  # Debug print
                return True
        except Exception as e:
            print(f"DEBUG: Error with shell method: {e}")  # Debug print

        # Method 4: Check PATH and log current environment
        print(f"DEBUG: Current PATH: {os.environ.get('PATH', 'Not set')}")  # Debug print

        # If all fails
        self.r_status_label.config(text="❌ R not installed", fg="red")
        self.check_r_button.config(text="Install R", state="normal")
        print("DEBUG: R not found by any method")  # Debug print
        return False

    def browse_input_file(self):
        filename = filedialog.askopenfilename(
            title="Select Input CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.input_file.set(filename)

    def browse_output_dir(self):
        dirname = filedialog.askdirectory(title="Select Output Directory")
        if dirname:
            self.output_dir.set(dirname)

    def log_message(self, message):
        """Add message to log area."""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def check_r_installation(self):
        """Check if R is installed and accessible with multiple methods."""
        print("DEBUG: Starting R installation check...")  # Debug print

        # Method 1: Try just 'R'
        try:
            result = subprocess.run(['R', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                self.log_message("✅ R is installed and accessible")
                print("DEBUG: R found with simple command in check_r_installation")  # Debug print
                return True
        except FileNotFoundError:
            print("DEBUG: Simple 'R' command not found in check_r_installation")  # Debug print
        except Exception as e:
            print(f"DEBUG: Error with simple R command in check_r_installation: {e}")  # Debug print

        # Method 2: Try full paths
        r_paths = ['/usr/local/bin/R', '/usr/bin/R', '/opt/homebrew/bin/R']
        for r_path in r_paths:
            if os.path.exists(r_path):
                print(f"DEBUG: Trying {r_path} in check_r_installation")  # Debug print
                try:
                    result = subprocess.run([r_path, '--version'], capture_output=True, text=True)
                    if result.returncode == 0:
                        self.log_message(f"✅ R is installed and accessible at {r_path}")
                        # Store the working R path for later use
                        self.r_executable = r_path
                        print(f"DEBUG: R found at {r_path} in check_r_installation")  # Debug print
                        return True
                except Exception as e:
                    print(f"DEBUG: Error with {r_path} in check_r_installation: {e}")  # Debug print

        # Method 3: Try with shell=True
        try:
            print("DEBUG: Trying with shell=True in check_r_installation")  # Debug print
            result = subprocess.run('R --version', shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                self.log_message("✅ R is installed and accessible (via shell)")
                self.r_executable = 'R'
                self.use_shell = True
                print("DEBUG: R found via shell in check_r_installation")  # Debug print
                return True
        except Exception as e:
            print(f"DEBUG: Error with shell method in check_r_installation: {e}")  # Debug print

        # Log current environment for debugging
        self.log_message(f"DEBUG - Current PATH: {os.environ.get('PATH', 'Not set')}")

        self.log_message("❌ R is not installed or not in PATH")
        print("DEBUG: R not found by any method in check_r_installation")  # Debug print
        return False

    def show_r_installation_guide(self):
        """Show R installation guide dialog."""
        message = """R is required for GLATOS processing but was not found.

Please install R:

1. Download R from: https://cran.r-project.org/
   - For Windows: Choose "Download R for Windows" → "base" → "Download R X.X.X for Windows"
   - For macOS: Choose "Download R for (Mac) OS X" → appropriate version
   - For Linux: Follow the instructions for your distribution

2. After installation:
   - Windows: R should be automatically added to PATH
   - macOS/Linux: You may need to add R to your PATH environment variable

3. Restart this application after installing R

Would you like to open the R download page in your browser?"""

        result = messagebox.askyesno("R Installation Required", message)

        if result:
            webbrowser.open("https://cran.r-project.org/")

    def show_about_glatos(self):
        """Show information about GLATOS."""
        message = """GLATOS (Great Lakes Acoustic Telemetry Observation System)

GLATOS is an R package for processing and analyzing acoustic telemetry data from fish and other aquatic animals.

Key features:
• Detection data processing and filtering
• False detection identification
• Residence index calculations
• Data visualization and analysis
• Standardized data formats

Learn more at: https://github.com/ocean-tracking-network/glatos

This GUI provides a user-friendly interface for the GLATOS pipeline, helping researchers process their acoustic telemetry data efficiently."""

        messagebox.showinfo("About GLATOS", message)

    def show_about(self):
        """Show about dialog."""
        message = """GLATOS Pipeline GUI
Version 1.0

A user-friendly interface for processing acoustic telemetry data using the GLATOS R package.

Created for MKMRS shark tracking research.

Features:
• Automatic timestamp combining
• GLATOS format conversion
• Noise filtering
• Comprehensive analysis reports

Requirements:
• R (https://cran.r-project.org/)
• Input CSV with columns: date, time, id, protocol, receiver"""

        messagebox.showinfo("About", message)

    def start_processing(self):
        """Start processing in a separate thread."""
        if not self.input_file.get():
            messagebox.showerror("Error", "Please select an input file!")
            return

        # Check if R is installed before proceeding
        self.status_label.config(text="🔍 Checking R installation...", fg="blue")
        self.log_text.delete(1.0, tk.END)
        self.log_message("Checking R installation...")

        if not self.check_r_installation():
            self.status_label.config(text="❌ R not found - installation required", fg="red")
            self.show_r_installation_guide()
            return

        # Disable button and start progress bar
        self.process_button.config(state="disabled", text="⏳ Processing...", bg="#FF9800")
        self.status_label.config(text="🔄 Processing in progress...", fg="orange")
        self.progress.start()
        self.progress_label.config(text="Initializing pipeline...")

        # Start processing in background thread
        processing_thread = threading.Thread(target=self.run_pipeline)
        processing_thread.daemon = True
        processing_thread.start()

    def update_progress(self, message):
        """Update the progress label."""
        self.root.after(0, lambda: self.progress_label.config(text=message))

    def run_pipeline(self):
        """Run the GLATOS pipeline."""
        try:
            self.update_progress("🔍 Searching for GLATOS Pipeline executable...")

            # Find the GLATOS pipeline executable
            exe_name = "GLATOSPipeline"
            if sys.platform == "win32":
                exe_name += ".exe"

            # Look for executable in multiple locations
            possible_paths = [
                # Same directory as the GUI executable (if frozen)
                Path(sys.executable).parent / exe_name if getattr(sys, 'frozen', False) else None,
                # Same directory as this script
                Path(__file__).parent / exe_name if not getattr(sys, 'frozen', False) else None,
                # Current working directory
                Path.cwd() / exe_name,
                # Directory from which the GUI was launched
                Path(os.getcwd()) / exe_name,
            ]

            # Filter out None values and find existing executable
            exe_path = None
            for path in filter(None, possible_paths):
                if path.exists():
                    exe_path = path
                    break

            # If not found, let user browse for it
            if not exe_path:
                self.log_message(f"❌ Could not find {exe_name}")
                self.update_progress("❌ Executable not found - please locate it")

                # Use root.after to ensure file dialog opens in main thread
                file_selected = threading.Event()
                selected_path = [None]

                def open_file_dialog():
                    path = filedialog.askopenfilename(
                        title=f"Please locate {exe_name}",
                        filetypes=[("Executable files", "*.exe" if sys.platform == "win32" else "*")],
                        initialfile=exe_name
                    )
                    selected_path[0] = path
                    file_selected.set()

                self.root.after(0, open_file_dialog)
                file_selected.wait()

                if not selected_path[0]:
                    self.log_message("❌ Pipeline executable not selected")
                    self.update_progress("❌ Cancelled by user")
                    return

                exe_path = Path(selected_path[0])

            self.log_message(f"✅ Found executable: {exe_path}")
            self.update_progress("✅ Executable found")

            # Build command
            cmd = [str(exe_path), self.input_file.get()]

            if self.output_dir.get():
                cmd.extend(["--output", self.output_dir.get()])

            cmd.extend(["--time-filter", str(self.time_filter.get())])

            self.log_message(f"🚀 Running: {' '.join(cmd)}")
            self.update_progress("🚀 Starting pipeline...")

            # Execute the pipeline
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True,
                                       bufsize=1, universal_newlines=True)

            # Read output in real-time
            stage_count = 0
            total_stages = 4

            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    line = output.rstrip()
                    self.log_message(line)

                    # Update progress based on stage detection
                    if "STAGE:" in line:
                        stage_count += 1
                        progress_percent = int((stage_count / total_stages) * 100)
                        self.update_progress(f"Stage {stage_count}/{total_stages} ({progress_percent}%)")
                    elif "SUCCESS" in line:
                        self.update_progress(f"✅ Stage {stage_count} completed")
                    elif "ERROR" in line or "FAILED" in line:
                        self.update_progress(f"❌ Error in Stage {stage_count}")

            # Get any remaining stderr
            stderr_output = process.stderr.read()
            if stderr_output:
                self.log_message(f"ERROR OUTPUT:\n{stderr_output}")

            process.wait()

            if process.returncode == 0:
                self.log_message("\n🎉 Pipeline completed successfully!")
                self.update_progress("🎉 Pipeline completed successfully!")
                messagebox.showinfo("Success", "GLATOS Pipeline completed successfully!")
            else:
                self.log_message(f"\n❌ Pipeline failed with return code {process.returncode}")
                self.update_progress(f"❌ Pipeline failed (code: {process.returncode})")

                # Parse error output for specific issues
                all_output = self.log_text.get(1.0, tk.END)

                if "R is not installed" in all_output or "No such file or directory: 'R'" in all_output:
                    error_msg = ("R Installation Error\n\n"
                                 "R was not found on your system. GLATOS requires R to be installed.\n\n"
                                 "Please install R from: https://cran.r-project.org/\n\n"
                                 "After installation, restart this application.")
                    messagebox.showerror("R Not Found", error_msg)
                elif "Missing required columns" in all_output:
                    error_msg = ("Input File Format Error\n\n"
                                 "Your CSV file doesn't have the required columns.\n\n"
                                 "Required columns: date, time, id, protocol, receiver\n\n"
                                 "Please check your input file format.")
                    messagebox.showerror("Invalid Input Format", error_msg)
                elif "Permission denied" in all_output:
                    error_msg = ("File Permission Error\n\n"
                                 "Cannot access the input or output files.\n\n"
                                 "Please check:\n"
                                 "- File is not open in another program\n"
                                 "- You have read/write permissions\n"
                                 "- Output directory exists and is writable")
                    messagebox.showerror("Permission Error", error_msg)
                elif "GLATOS package" in all_output and "not installed" in all_output:
                    error_msg = ("GLATOS Package Error\n\n"
                                 "The GLATOS R package failed to install automatically.\n\n"
                                 "Please install it manually in R:\n"
                                 '1. Open R\n'
                                 '2. Run: install.packages("remotes")\n'
                                 '3. Run: remotes::install_github("ocean-tracking-network/glatos")\n'
                                 '\nThen try again.')
                    messagebox.showerror("GLATOS Package Error", error_msg)
                else:
                    # Generic error message
                    error_msg = f"Pipeline failed with return code {process.returncode}\n\n"
                    error_msg += "Common causes:\n"
                    error_msg += "1. R is not installed or not in PATH\n"
                    error_msg += "2. Input file format is incorrect\n"
                    error_msg += "3. Required R packages (GLATOS) not installed\n"
                    error_msg += "4. File permissions issue\n\n"
                    error_msg += "Check the log above for detailed error messages."

                    messagebox.showerror("Pipeline Failed", error_msg)

        except Exception as e:
            self.log_message(f"\n💥 Exception occurred: {str(e)}")
            self.update_progress(f"💥 Error: {str(e)}")
            messagebox.showerror("Error", f"Error running pipeline:\n{str(e)}\n\nCheck the log for details.")

        finally:
            # Re-enable button and stop progress bar
            self.root.after(0, self.finish_processing)

    def finish_processing(self):
        """Clean up after processing."""
        self.process_button.config(state="normal", text="🚀 Process Data", bg="#4CAF50")
        self.progress.stop()

        # Update status based on last log entry
        try:
            last_log = self.log_text.get("end-2l", "end-1l").strip()
            if "completed successfully" in last_log.lower():
                self.status_label.config(text="✅ Processing completed successfully!", fg="green")
            else:
                self.status_label.config(text="❌ Processing failed - check log for details", fg="red")
        except:
            self.status_label.config(text="⚠️ Processing finished", fg="gray")


def main():
    root = tk.Tk()
    app = GLATOSPipelineGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()