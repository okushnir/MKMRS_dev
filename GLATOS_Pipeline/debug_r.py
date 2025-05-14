#!/usr/bin/env python3
"""
Debug R Detection
This script helps debug why the GUI can't find R
"""

import os
import sys
import subprocess

print("=== R Detection Debug ===")
print()

# 1. Check current PATH
print("1. Current PATH environment:")
current_path = os.environ.get('PATH', '')
print(f"PATH = {current_path}")
print()

# 2. Check if R directories exist
print("2. Checking R installation directories:")
r_dirs = ['/usr/local/bin', '/usr/bin', '/opt/homebrew/bin']
for r_dir in r_dirs:
    exists = "EXISTS" if os.path.exists(r_dir) else "NOT FOUND"
    print(f"{r_dir}: {exists}")
print()

# 3. Check if R executable exists at specific paths
print("3. Checking R executables:")
r_paths = ['/usr/local/bin/R', '/usr/bin/R', '/opt/homebrew/bin/R']
for r_path in r_paths:
    exists = "EXISTS" if os.path.exists(r_path) else "NOT FOUND"
    print(f"{r_path}: {exists}")
print()

# 4. Try to run R with different methods
print("4. Testing R execution:")

# Method 1: Just 'R'
try:
    result = subprocess.run(['R', '--version'], capture_output=True, text=True)
    if result.returncode == 0:
        version_line = result.stdout.split('\n')[0]
        print(f"✅ Method 1 (R): SUCCESS - {version_line}")
    else:
        print(f"❌ Method 1 (R): FAILED - Return code {result.returncode}")
        print(f"   stdout: {result.stdout}")
        print(f"   stderr: {result.stderr}")
except FileNotFoundError as e:
    print(f"❌ Method 1 (R): FAILED - FileNotFoundError: {e}")
except Exception as e:
    print(f"❌ Method 1 (R): FAILED - Exception: {e}")

# Method 2: Try full paths
for r_path in r_paths:
    if os.path.exists(r_path):
        try:
            result = subprocess.run([r_path, '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0]
                print(f"✅ Method 2 ({r_path}): SUCCESS - {version_line}")
            else:
                print(f"❌ Method 2 ({r_path}): FAILED - Return code {result.returncode}")
        except Exception as e:
            print(f"❌ Method 2 ({r_path}): FAILED - Exception: {e}")

print()

# 5. Check which command
print("5. Using 'which' command:")
try:
    result = subprocess.run(['which', 'R'], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✅ which R: {result.stdout.strip()}")
    else:
        print("❌ which R: Not found")
except Exception as e:
    print(f"❌ which R: Error - {e}")

print()

# 6. Try with shell=True
print("6. Testing with shell=True:")
try:
    result = subprocess.run('R --version', shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        version_line = result.stdout.split('\n')[0]
        print(f"✅ Shell method: SUCCESS - {version_line}")
    else:
        print(f"❌ Shell method: FAILED - Return code {result.returncode}")
        print(f"   stdout: {result.stdout}")
        print(f"   stderr: {result.stderr}")
except Exception as e:
    print(f"❌ Shell method: FAILED - Exception: {e}")

print()
print("=== End Debug ===")