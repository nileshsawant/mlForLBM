#!/bin/bash
# Course Downloader - Simple wrapper script
# Usage: ./download_course.sh <course_url>

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if URL is provided
if [ $# -eq 0 ]; then
    print_error "No URL provided!"
    echo "Usage: $0 <course_url> [output_directory]"
    echo "Example: $0 https://camlab.ethz.ch/teaching/ai-in-the-sciences-and-engineering-2024.html"
    exit 1
fi

COURSE_URL="$1"
OUTPUT_DIR="$2"

print_status "Course Material Downloader"
print_status "=========================="
print_status "Course URL: $COURSE_URL"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is required but not installed."
    exit 1
fi

# Check if required Python packages are available
python3 -c "import requests" 2>/dev/null || {
    print_warning "requests library not found. Installing..."
    python3 -m pip install requests
}

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DOWNLOADER_SCRIPT="$SCRIPT_DIR/course_downloader.py"

# Check if the downloader script exists
if [ ! -f "$DOWNLOADER_SCRIPT" ]; then
    print_error "course_downloader.py not found in $SCRIPT_DIR"
    exit 1
fi

# Prepare arguments
ARGS="$COURSE_URL"
if [ -n "$OUTPUT_DIR" ]; then
    ARGS="$ARGS -o $OUTPUT_DIR"
fi

print_status "Starting download..."

# Run the Python downloader
if python3 "$DOWNLOADER_SCRIPT" $ARGS --open; then
    print_success "Course materials downloaded successfully!"
    print_status "The materials have been organized and an index.html file has been created."
    print_status "You can now browse the course offline using the generated index page."
else
    print_error "Download failed!"
    exit 1
fi
