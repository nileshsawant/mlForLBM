# Course Material Downloader

A comprehensive script to automatically download course materials from educational websites, organize them, and create an offline browsing index.

## Features

🎯 **Automatic Detection & Download**
- Automatically finds and downloads PDF lecture slides
- Detects and clones associated GitHub repositories
- Downloads course webpage and metadata
- Extracts tutorial notebooks and datasets

📁 **Smart Organization**
- Creates organized directory structure
- Generates beautiful offline index.html for browsing
- Preserves original filenames and structure
- Creates summary statistics and metadata

🌐 **Offline Browsing**
- Beautiful responsive web interface
- Direct PDF viewing in browser
- Easy navigation between materials
- Links to notebooks and external resources

## Quick Start

### Method 1: Simple Bash Script (Recommended)
```bash
# Make the script executable (first time only)
chmod +x download_course.sh

# Download a course
./download_course.sh "https://camlab.ethz.ch/teaching/ai-in-the-sciences-and-engineering-2024.html"

# Or specify custom output directory
./download_course.sh "https://example.edu/course" "my_course_materials"
```

### Method 2: Direct Python Script
```bash
# Install dependencies
pip install requests

# Download course materials
python3 course_downloader.py "https://camlab.ethz.ch/teaching/ai-in-the-sciences-and-engineering-2024.html"

# With custom output directory
python3 course_downloader.py "https://example.edu/course" -o "my_course_materials"

# Auto-open in browser when done
python3 course_downloader.py "https://example.edu/course" --open
```

## Requirements

- Python 3.6 or higher
- `requests` library (`pip install requests`)
- `git` (optional, for repository cloning)
- Internet connection for downloading

## What Gets Downloaded

### 📚 Course Materials
- **PDF Slides**: All lecture slides and presentations
- **Notebooks**: Jupyter notebooks and tutorials
- **Datasets**: Training data and example files
- **Code**: Helper scripts and utilities
- **Metadata**: Course information and structure

### 🗂️ Generated Files
- **index.html**: Beautiful offline browsing interface
- **course_page.html**: Original course webpage
- **download_summary.json**: Download statistics and metadata
- **table_*.json**: Extracted course structure data

## Output Structure

```
course_materials/
├── index.html                 # Main offline browsing page
├── course_page.html           # Original course page
├── download_summary.json      # Download metadata
├── lecture_slides/            # PDF lecture materials
│   ├── Lecture 01.pdf
│   ├── Lecture 02.pdf
│   └── ...
├── repository_name/           # Cloned GitHub repository
│   ├── Tutorial 01.ipynb
│   ├── Tutorial 02.ipynb
│   ├── datasets/
│   └── ...
└── table_*.json              # Course structure data
```

## Supported Websites

The script is designed to work with educational websites that follow common patterns:

✅ **Optimized for:**
- ETH Zurich course pages
- University course websites
- Educational platforms with structured content

🔧 **Adaptable for:**
- Other academic institutions
- Online course platforms
- Documentation websites

## Command Line Options

### Python Script Options
```bash
python3 course_downloader.py [OPTIONS] <url>

Arguments:
  url                    Course website URL

Options:
  -o, --output DIR      Output directory (default: auto-generated)
  --open               Open index.html in browser after download
  -h, --help           Show help message
```

### Bash Script Options
```bash
./download_course.sh <course_url> [output_directory]
```

## Examples

### Download ETH Zurich AI Course
```bash
./download_course.sh "https://camlab.ethz.ch/teaching/ai-in-the-sciences-and-engineering-2024.html"
```

### Download to Specific Directory
```bash
./download_course.sh "https://example.edu/course" "ML_Course_2024"
```

### Download and Auto-Open
```bash
python3 course_downloader.py "https://example.edu/course" --open
```

## Troubleshooting

### Common Issues

**"requests not found"**
```bash
pip install requests
```

**"Permission denied"**
```bash
chmod +x download_course.sh
```

**"Git not found"**
- Install git: `brew install git` (macOS) or `sudo apt install git` (Ubuntu)
- Repository cloning will be skipped if git is not available

**"Download failed"**
- Check internet connection
- Verify the course URL is accessible
- Some websites may have rate limiting - try again later

### Debug Mode
Add debugging to see detailed information:
```bash
python3 course_downloader.py -v "https://example.edu/course"
```

## Advanced Usage

### Custom Website Adaptation
To adapt the script for other websites:

1. Modify the `find_json_tables()` method for different data sources
2. Update `extract_pdf_links()` for different HTML patterns
3. Adjust `find_github_repo()` for different repository patterns

### Batch Processing
Download multiple courses:
```bash
#!/bin/bash
courses=(
    "https://example.edu/course1"
    "https://example.edu/course2"
    "https://example.edu/course3"
)

for course in "${courses[@]}"; do
    ./download_course.sh "$course"
done
```

## Contributing

Feel free to improve the script:

1. **Add support for new websites**
2. **Improve error handling**
3. **Add new features**
4. **Fix bugs**

## License

This script is provided as-is for educational purposes. Please respect website terms of service and copyright when downloading materials.

---

**Created by:** Course Material Downloader Script  
**Last Updated:** September 2025  
**Version:** 1.0
