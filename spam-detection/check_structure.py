#!/usr/bin/env python3
"""
Check file structure for Render deployment
"""

import os

def check_structure():
    print("=== File Structure Check for Render ===")
    print(f"Current directory: {os.getcwd()}")
    print(f"Files in current directory:")
    
    files = os.listdir('.')
    for file in sorted(files):
        if os.path.isfile(file):
            size = os.path.getsize(file)
            print(f"  📄 {file} ({size:,} bytes)")
        else:
            print(f"  📁 {file}/")
    
    print("\n=== Required Files Check ===")
    required_files = [
        'app.py',
        'requirements.txt',
        'train_model.py',
        'spam_sms_model.pkl',
        'spam_email_model.pkl'
    ]
    
    missing = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - MISSING")
            missing.append(file)
    
    if missing:
        print(f"\n❌ Missing files: {missing}")
        print("Make sure these files are in the root directory of your deployment.")
    else:
        print("\n✅ All required files found!")
    
    print("\n=== Data Directory Check ===")
    if os.path.exists('data'):
        data_files = os.listdir('data')
        print(f"✅ data/ directory found with {len(data_files)} files:")
        for file in data_files:
            print(f"  - {file}")
    else:
        print("❌ data/ directory not found")

if __name__ == "__main__":
    check_structure() 