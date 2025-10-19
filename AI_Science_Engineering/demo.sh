#!/bin/bash
# Demo script to show how to use the course downloader

echo "==================================="
echo "Course Material Downloader - Demo"
echo "==================================="
echo

echo "This demo shows how to use the course downloader for future courses."
echo

echo "📚 BASIC USAGE:"
echo "./download_course.sh \"https://example.edu/course-url\""
echo

echo "📁 WITH CUSTOM DIRECTORY:"
echo "./download_course.sh \"https://example.edu/course-url\" \"my_course_folder\""
echo

echo "⚙️ PYTHON SCRIPT DIRECTLY:"
echo "python3 course_downloader.py \"https://example.edu/course-url\""
echo

echo "🌐 AUTO-OPEN IN BROWSER:"
echo "python3 course_downloader.py \"https://example.edu/course-url\" --open"
echo

echo "📋 WHAT THE SCRIPT DOES:"
echo "  ✅ Downloads all PDF lecture slides"
echo "  ✅ Clones associated GitHub repositories"
echo "  ✅ Downloads Jupyter notebooks and tutorials"
echo "  ✅ Creates beautiful offline index.html"
echo "  ✅ Organizes everything in folders"
echo "  ✅ Preserves original website structure"
echo

echo "💡 EXAMPLE - Download ETH AI Course:"
echo "./download_course.sh \"https://camlab.ethz.ch/teaching/ai-in-the-sciences-and-engineering-2024.html\""
echo

echo "🗂️ OUTPUT STRUCTURE:"
echo "course_materials/"
echo "├── index.html                 # Offline browsing page"
echo "├── lecture_slides/            # All PDF files"
echo "├── repository_name/           # GitHub repo with notebooks"
echo "├── course_page.html           # Original webpage"
echo "└── download_summary.json      # Download metadata"
echo

echo "📖 For more details, see README_Course_Downloader.md"
echo

read -p "Press Enter to continue..."
