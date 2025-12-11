#!/usr/bin/env python3
"""
Script to patch wgp.py by removing deprecated Gradio arguments
"""
import re

def patch_wgp_py():
    file_path = "wgp.py"
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove show_reset_button arguments
    content = re.sub(r',\s*show_reset_button\s*=\s*[a-zA-Z0-9_.]+', '', content)
    content = re.sub(r'\s+show_reset_button\s*=\s*[a-zA-Z0-9_.]+', '', content)
    
    # Remove show_download_button arguments
    content = re.sub(r',\s*show_download_button\s*=\s*[a-zA-Z0-9_.]+', '', content)
    content = re.sub(r'\s+show_download_button\s*=\s*[a-zA-Z0-9_.]+', '', content)
    
    # Write the patched content back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Successfully patched wgp.py to remove deprecated Gradio arguments")

if __name__ == "__main__":
    patch_wgp_py()