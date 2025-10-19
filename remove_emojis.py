#!/usr/bin/env python3
"""
Remove all emojis from Python files in the current directory
"""

import os
import re
import glob

def remove_emojis(text):
    """Remove emojis from text"""
    # Define emoji pattern (covers most common emojis)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    
    return emoji_pattern.sub('', text)

def process_python_files():
    """Process all Python files in current directory"""
    
    # Find all Python files
    python_files = glob.glob("*.py")
    
    print(f"Found {len(python_files)} Python files")
    
    for py_file in python_files:
        print(f"Processing {py_file}...")
        
        try:
            # Read file
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Remove emojis
            clean_content = remove_emojis(content)
            
            # Write back if content changed
            if content != clean_content:
                with open(py_file, 'w', encoding='utf-8') as f:
                    f.write(clean_content)
                print(f"  Cleaned emojis from {py_file}")
            else:
                print(f"  No emojis found in {py_file}")
                
        except Exception as e:
            print(f"  Error processing {py_file}: {e}")

if __name__ == "__main__":
    process_python_files()
    print("Emoji removal completed!")