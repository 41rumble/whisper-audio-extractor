#!/usr/bin/env python3
"""
Setup script for the Whisper Audio Extractor application.
This script checks for dependencies and installs them if needed.
"""

import os
import sys
import subprocess
import pkg_resources

def check_ffmpeg():
    """Check if FFmpeg is installed."""
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print("✅ FFmpeg is installed")
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        print("❌ FFmpeg is not installed")
        return False

def install_ffmpeg():
    """Provide instructions for installing FFmpeg."""
    print("\nPlease install FFmpeg:")
    print("- Ubuntu/Debian: sudo apt-get install ffmpeg")
    print("- macOS: brew install ffmpeg")
    print("- Windows: Download from https://ffmpeg.org/download.html")
    print("\nAfter installing FFmpeg, run this script again.")

def check_and_install_dependencies():
    """Check and install Python dependencies."""
    required_packages = []
    with open('requirements.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                required_packages.append(line)
    
    installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    
    # Check if numpy is installed and its version
    numpy_installed = False
    numpy_version = None
    try:
        import numpy
        numpy_installed = True
        numpy_version = numpy.__version__
        major_version = int(numpy_version.split('.')[0])
        if major_version >= 2:
            print(f"⚠️ Incompatible NumPy version detected: {numpy_version}")
            print("   This application requires NumPy < 2.0")
            
            # Ask user if they want to downgrade
            response = input("Do you want to downgrade NumPy to a compatible version? (y/n): ")
            if response.lower() == 'y':
                print("Downgrading NumPy...")
                subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', '-y', 'numpy'])
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy<2.0.0'])
                print("✅ NumPy downgraded successfully")
            else:
                print("⚠️ The application may not work correctly with NumPy 2.x")
    except ImportError:
        pass
    
    # Install missing packages
    packages_to_install = []
    for package in required_packages:
        package_name = package.split('==')[0].split('<')[0].split('>')[0].strip()
        if package_name.lower() not in installed_packages and package_name.lower() != 'numpy':
            packages_to_install.append(package)
    
    if packages_to_install:
        print(f"Installing {len(packages_to_install)} missing packages...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + packages_to_install)
        print("✅ All Python dependencies installed successfully")
    else:
        print("✅ All Python dependencies are already installed")

def main():
    """Main function."""
    print("Setting up Whisper Audio Extractor...\n")
    
    # Check if FFmpeg is installed
    if not check_ffmpeg():
        install_ffmpeg()
        return
    
    # Check and install Python dependencies
    check_and_install_dependencies()
    
    print("\nSetup complete! You can now run the application with:")
    print("python app.py")
    print("\nIf port 52678 is already in use, you can specify a different port:")
    print("python app.py --port 8080")

if __name__ == "__main__":
    main()