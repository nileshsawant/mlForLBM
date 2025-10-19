#!/usr/bin/env python3
"""
Course Material Downloader
A script to automatically download course materials from educational websites.
Specifically designed for ETH Zurich course pages but can be adapted for others.

Usage: python course_downloader.py <course_url>
Example: python course_downloader.py https://camlab.ethz.ch/teaching/ai-in-the-sciences-and-engineering-2024.html
"""

import os
import sys
import json
import re
import requests
from urllib.parse import urljoin, urlparse
from pathlib import Path
import time
from typing import List, Dict, Optional
import argparse
from datetime import datetime
import subprocess

class CourseDownloader:
    def __init__(self, course_url: str, output_dir: str = None):
        self.course_url = course_url.rstrip('/')
        self.base_url = '/'.join(course_url.split('/')[:3])
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Create output directory
        if output_dir is None:
            parsed_url = urlparse(course_url)
            course_name = parsed_url.path.split('/')[-1].replace('.html', '').replace('-', '_')
            self.output_dir = Path(f"{course_name}_materials")
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(exist_ok=True)
        
        # Course information
        self.course_info = {}
        self.lectures = []
        self.tutorials = []
        self.github_repo = None
        
    def log(self, message: str):
        """Print timestamped log message"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        
    def download_file(self, url: str, filename: str, subdir: str = None) -> bool:
        """Download a file from URL to local path"""
        try:
            if subdir:
                local_dir = self.output_dir / subdir
                local_dir.mkdir(exist_ok=True)
                local_path = local_dir / filename
            else:
                local_path = self.output_dir / filename
                
            if local_path.exists():
                self.log(f"File already exists: {filename}")
                return True
                
            self.log(f"Downloading: {filename}")
            response = self.session.get(url, stream=True)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            self.log(f"Downloaded: {filename} ({len(response.content)} bytes)")
            return True
            
        except Exception as e:
            self.log(f"Error downloading {filename}: {e}")
            return False
            
    def extract_course_info(self, html_content: str):
        """Extract course information from HTML"""
        self.log("Extracting course information...")
        
        # Extract title
        title_match = re.search(r'<title>(.*?)</title>', html_content, re.IGNORECASE)
        if title_match:
            self.course_info['title'] = title_match.group(1).strip()
        
        # Extract institution info (ETH Zurich specific patterns)
        if 'ethz.ch' in self.course_url:
            self.course_info['institution'] = 'ETH Zurich'
            
        # Extract lecturer information
        lecturer_pattern = r'Prof\.\s*Dr\.\s*([^<]+)'
        lecturers = re.findall(lecturer_pattern, html_content)
        if lecturers:
            self.course_info['lecturers'] = lecturers
            
    def find_json_tables(self, html_content: str) -> List[str]:
        """Find JSON table URLs in HTML content"""
        json_urls = []
        
        # Look for table JSON URLs
        json_pattern = r'data-json-url="([^"]*\.json)"'
        matches = re.findall(json_pattern, html_content)
        
        for match in matches:
            if match.startswith('/'):
                full_url = self.base_url + match
            else:
                full_url = urljoin(self.course_url, match)
            json_urls.append(full_url)
            
        return json_urls
        
    def extract_pdf_links(self, json_content: Dict) -> List[Dict]:
        """Extract PDF links from JSON table data"""
        pdf_links = []
        
        try:
            if 'tbody' in json_content and 'rows' in json_content['tbody']:
                for row in json_content['tbody']['rows']:
                    for cell in row.get('cells', []):
                        content = cell.get('content', '')
                        
                        # Extract PDF URLs
                        pdf_matches = re.findall(r'href="([^"]*\.pdf)"', content)
                        for pdf_url in pdf_matches:
                            # Extract title from the same cell or adjacent cells
                            title_match = re.search(r'>([^<]+)</span>', content.replace('Download ', ''))
                            title = title_match.group(1) if title_match else "Unknown"
                            
                            # Clean up title
                            title = re.sub(r'\s+', ' ', title).strip()
                            if title.endswith('Slides (PDF'):
                                title = title.replace('Slides (PDF', '').strip()
                                
                            pdf_links.append({
                                'url': pdf_url,
                                'title': title,
                                'filename': self.get_filename_from_url(pdf_url)
                            })
                            
        except Exception as e:
            self.log(f"Error extracting PDF links: {e}")
            
        return pdf_links
        
    def get_filename_from_url(self, url: str) -> str:
        """Extract and clean filename from URL"""
        filename = os.path.basename(url)
        # URL decode common patterns
        filename = filename.replace('%20', ' ')
        filename = filename.replace('%E2%80%93', '–')
        filename = filename.replace('%2D', '-')
        return filename
        
    def find_github_repo(self, html_content: str) -> Optional[str]:
        """Find GitHub repository URL"""
        github_pattern = r'https://github\.com/[^"\'>\s]+'
        matches = re.findall(github_pattern, html_content)
        
        for match in matches:
            if 'github.com' in match and not match.endswith('.git'):
                return match
                
        return None
        
    def clone_github_repo(self, repo_url: str) -> bool:
        """Clone GitHub repository"""
        try:
            repo_name = repo_url.split('/')[-1]
            repo_path = self.output_dir / repo_name
            
            if repo_path.exists():
                self.log(f"Repository already exists: {repo_name}")
                return True
                
            self.log(f"Cloning repository: {repo_url}")
            subprocess.run(['git', 'clone', repo_url, str(repo_path)], 
                         check=True, capture_output=True)
            self.log(f"Repository cloned successfully: {repo_name}")
            return True
            
        except subprocess.CalledProcessError as e:
            self.log(f"Error cloning repository: {e}")
            return False
        except FileNotFoundError:
            self.log("Git not found. Please install git to clone repositories.")
            return False
            
    def create_index_html(self):
        """Create an index.html file for offline browsing"""
        self.log("Creating index.html...")
        
        # Get course title
        title = self.course_info.get('title', 'Course Materials')
        
        # Count materials
        pdf_count = len([f for f in self.output_dir.rglob('*.pdf')])
        notebook_count = len([f for f in self.output_dir.rglob('*.ipynb')])
        
        # Calculate total size
        total_size = sum(f.stat().st_size for f in self.output_dir.rglob('*') if f.is_file())
        size_mb = total_size / (1024 * 1024)
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #1f2937;
            border-bottom: 3px solid #3b82f6;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #374151;
            margin-top: 30px;
            border-left: 4px solid #3b82f6;
            padding-left: 15px;
        }}
        .course-info {{
            background: #eff6ff;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .stats {{
            display: flex;
            justify-content: space-around;
            background: #f3f4f6;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .stat-item {{
            text-align: center;
        }}
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
            color: #3b82f6;
        }}
        .stat-label {{
            color: #6b7280;
            font-size: 0.9em;
        }}
        .file-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .file-item {{
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 15px;
            transition: all 0.3s ease;
        }}
        .file-item:hover {{
            background: #e0f2fe;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        .file-link {{
            display: inline-block;
            background: #dc2626;
            color: white;
            padding: 8px 15px;
            text-decoration: none;
            border-radius: 5px;
            font-size: 0.9em;
            margin-top: 10px;
            transition: background 0.3s;
        }}
        .file-link:hover {{
            background: #b91c1c;
        }}
        .notebook-link {{
            background: #059669;
        }}
        .notebook-link:hover {{
            background: #047857;
        }}
        footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e5e7eb;
            text-align: center;
            color: #6b7280;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📚 {title}</h1>
        
        <div class="course-info">
            <h3>📖 Course Information</h3>
            <p><strong>Source URL:</strong> <a href="{self.course_url}" target="_blank">{self.course_url}</a></p>
            <p><strong>Downloaded:</strong> {datetime.now().strftime('%B %d, %Y at %H:%M')}</p>
            <p><strong>Generated by:</strong> Course Material Downloader Script</p>
        </div>

        <div class="stats">
            <div class="stat-item">
                <div class="stat-number">{pdf_count}</div>
                <div class="stat-label">PDF Files</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{notebook_count}</div>
                <div class="stat-label">Notebooks</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{size_mb:.1f}MB</div>
                <div class="stat-label">Total Size</div>
            </div>
        </div>
"""

        # Add PDF files section
        pdf_files = list(self.output_dir.rglob('*.pdf'))
        if pdf_files:
            html_content += """
        <h2>📄 PDF Files</h2>
        <div class="file-grid">
"""
            for pdf_file in sorted(pdf_files):
                rel_path = pdf_file.relative_to(self.output_dir)
                file_size = pdf_file.stat().st_size / (1024 * 1024)
                html_content += f"""
            <div class="file-item">
                <div class="file-title">{pdf_file.stem}</div>
                <div class="file-size">{file_size:.1f} MB</div>
                <a href="{rel_path}" class="file-link">📄 View PDF</a>
            </div>
"""
            html_content += "        </div>\n"

        # Add notebook files section
        notebook_files = list(self.output_dir.rglob('*.ipynb'))
        if notebook_files:
            html_content += """
        <h2>📓 Jupyter Notebooks</h2>
        <div class="file-grid">
"""
            for notebook_file in sorted(notebook_files):
                rel_path = notebook_file.relative_to(self.output_dir)
                html_content += f"""
            <div class="file-item">
                <div class="file-title">{notebook_file.stem}</div>
                <a href="{rel_path}" class="file-link notebook-link">📓 Open Notebook</a>
            </div>
"""
            html_content += "        </div>\n"

        # Add additional files section
        other_files = [f for f in self.output_dir.rglob('*') 
                      if f.is_file() and f.suffix not in ['.pdf', '.ipynb', '.html']]
        
        if other_files:
            html_content += """
        <h2>📁 Additional Files</h2>
        <ul>
"""
            for file in sorted(other_files)[:20]:  # Limit to first 20 files
                rel_path = file.relative_to(self.output_dir)
                html_content += f'            <li><a href="{rel_path}">{rel_path}</a></li>\n'
            html_content += "        </ul>\n"

        html_content += f"""
        <footer>
            <p>Course materials downloaded from: <a href="{self.course_url}" target="_blank">{self.course_url}</a></p>
            <p>Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}</p>
        </footer>
    </div>
</body>
</html>
"""

        # Write HTML file
        index_path = self.output_dir / 'index.html'
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        self.log(f"Index page created: {index_path}")
        
    def download_course(self):
        """Main method to download entire course"""
        self.log(f"Starting download of course: {self.course_url}")
        self.log(f"Output directory: {self.output_dir}")
        
        try:
            # Download main page
            self.log("Downloading main course page...")
            response = self.session.get(self.course_url)
            response.raise_for_status()
            html_content = response.text
            
            # Save main page
            with open(self.output_dir / 'course_page.html', 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            # Extract course information
            self.extract_course_info(html_content)
            
            # Find and download JSON tables
            json_urls = self.find_json_tables(html_content)
            self.log(f"Found {len(json_urls)} JSON table files")
            
            all_pdfs = []
            for i, json_url in enumerate(json_urls):
                try:
                    self.log(f"Processing JSON table {i+1}/{len(json_urls)}")
                    json_response = self.session.get(json_url)
                    json_response.raise_for_status()
                    json_data = json_response.json()
                    
                    # Save JSON file
                    json_filename = f"table_{i+1}.json"
                    with open(self.output_dir / json_filename, 'w') as f:
                        json.dump(json_data, f, indent=2)
                    
                    # Extract PDF links
                    pdfs = self.extract_pdf_links(json_data)
                    all_pdfs.extend(pdfs)
                    
                except Exception as e:
                    self.log(f"Error processing JSON {json_url}: {e}")
            
            # Remove duplicate PDFs
            unique_pdfs = []
            seen_urls = set()
            for pdf in all_pdfs:
                if pdf['url'] not in seen_urls:
                    unique_pdfs.append(pdf)
                    seen_urls.add(pdf['url'])
            
            self.log(f"Found {len(unique_pdfs)} unique PDF files")
            
            # Download PDFs
            if unique_pdfs:
                pdf_dir = self.output_dir / 'lecture_slides'
                pdf_dir.mkdir(exist_ok=True)
                
                for i, pdf in enumerate(unique_pdfs):
                    self.log(f"Downloading PDF {i+1}/{len(unique_pdfs)}: {pdf['filename']}")
                    success = self.download_file(pdf['url'], pdf['filename'], 'lecture_slides')
                    if success:
                        time.sleep(1)  # Be nice to the server
            
            # Find and clone GitHub repository
            github_url = self.find_github_repo(html_content)
            if github_url:
                self.log(f"Found GitHub repository: {github_url}")
                self.clone_github_repo(github_url)
            
            # Create index page
            self.create_index_html()
            
            # Create summary
            self.create_summary()
            
            self.log("Download completed successfully!")
            self.log(f"All materials saved to: {self.output_dir.absolute()}")
            
        except Exception as e:
            self.log(f"Error during download: {e}")
            raise
            
    def create_summary(self):
        """Create a summary file with download information"""
        summary = {
            'course_url': self.course_url,
            'download_date': datetime.now().isoformat(),
            'output_directory': str(self.output_dir.absolute()),
            'course_info': self.course_info,
            'statistics': {
                'pdf_files': len(list(self.output_dir.rglob('*.pdf'))),
                'notebook_files': len(list(self.output_dir.rglob('*.ipynb'))),
                'total_files': len(list(self.output_dir.rglob('*'))),
                'total_size_mb': sum(f.stat().st_size for f in self.output_dir.rglob('*') if f.is_file()) / (1024 * 1024)
            }
        }
        
        with open(self.output_dir / 'download_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Download course materials from educational websites')
    parser.add_argument('url', help='Course website URL')
    parser.add_argument('-o', '--output', help='Output directory (optional)')
    parser.add_argument('--open', action='store_true', help='Open index.html in browser after download')
    
    args = parser.parse_args()
    
    # Validate URL
    if not args.url.startswith(('http://', 'https://')):
        print("Error: Please provide a valid URL starting with http:// or https://")
        sys.exit(1)
    
    try:
        downloader = CourseDownloader(args.url, args.output)
        downloader.download_course()
        
        if args.open:
            index_path = downloader.output_dir / 'index.html'
            if index_path.exists():
                import webbrowser
                webbrowser.open(f'file://{index_path.absolute()}')
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
