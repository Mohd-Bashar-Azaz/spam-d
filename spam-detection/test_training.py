#!/usr/bin/env python3
"""
Test script to verify training process works correctly
"""

import os
import sys
import subprocess

def test_dependencies():
    """Test if all required dependencies are available"""
    print("=== Testing Dependencies ===")
    
    required_packages = [
        'pandas',
        'sklearn', 
        'joblib',
        'streamlit'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
                print(f"OK {package}: {sklearn.__version__}")
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                print(f"OK {package}: {version}")
        except ImportError as e:
            print(f"ERROR {package}: {e}")
            missing.append(package)
    
    if missing:
        print(f"\nERROR Missing packages: {missing}")
        return False
    else:
        print("\nOK All dependencies available!")
        return True

def test_data_files():
    """Test if data files exist"""
    print("\n=== Testing Data Files ===")
    
    data_files = [
        'data/spam_sms.csv',
        'data/spam_email.csv'
    ]
    
    missing = []
    for file_path in data_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"OK {file_path}: {size:,} bytes")
        else:
            print(f"ERROR {file_path}: Not found")
            missing.append(file_path)
    
    if missing:
        print(f"\nERROR Missing data files: {missing}")
        return False
    else:
        print("\nOK All data files found!")
        return True

def test_training_script():
    """Test if training script can be executed"""
    print("\n=== Testing Training Script ===")
    
    if not os.path.exists('train_model.py'):
        print("ERROR train_model.py not found")
        return False
    
    print("OK train_model.py found")
    
    # Try to import the training script
    try:
        import train_model
        print("OK train_model.py can be imported")
        return True
    except Exception as e:
        print(f"ERROR Error importing train_model.py: {e}")
        return False

def run_training_test():
    """Run a quick training test"""
    print("\n=== Running Training Test ===")
    
    try:
        # Run training script with timeout
        result = subprocess.run(
            [sys.executable, 'train_model.py'],
            capture_output=True,
            text=True,
            timeout=60,  # 1 minute timeout for test
            cwd=os.getcwd()
        )
        
        print(f"Return code: {result.returncode}")
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("Training test completed successfully!")
            return True
        else:
            print("Training test failed!")
            return False
            
    except subprocess.TimeoutExpired:
        print("Training test timed out!")
        return False
    except Exception as e:
        print(f"Error running training test: {e}")
        return False

def main():
    """Main test function"""
    print("Spam Detection Training Test Suite")
    print("=" * 50)
    
    # Test current directory
    print(f"Current directory: {os.getcwd()}")
    print(f"Available files: {os.listdir('.')}")
    
    # Run tests
    deps_ok = test_dependencies()
    data_ok = test_data_files()
    script_ok = test_training_script()
    
    if deps_ok and data_ok and script_ok:
        print("\nAll basic tests passed! Running training test...")
        training_ok = run_training_test()
        
        if training_ok:
            print("\nAll tests passed! Training should work correctly.")
        else:
            print("\nBasic tests passed but training test failed.")
    else:
        print("\nSome basic tests failed. Please fix issues before training.")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main() 