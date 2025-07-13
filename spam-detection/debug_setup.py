#!/usr/bin/env python3
"""
Debug setup script for Spam Detection App
Helps identify and fix common deployment issues
"""

import os
import sys
import subprocess
import importlib

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def check_python_version():
    """Check Python version"""
    print_header("Python Version Check")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ required!")
        return False
    else:
        print("✅ Python version is compatible")
        return True

def check_dependencies():
    """Check if all required dependencies are installed"""
    print_header("Dependency Check")
    
    required_packages = {
        'pandas': 'pandas',
        'sklearn': 'scikit-learn', 
        'joblib': 'joblib',
        'streamlit': 'streamlit',
        'numpy': 'numpy'
    }
    
    missing = []
    for package, pip_name in required_packages.items():
        try:
            if package == 'sklearn':
                import sklearn
                version = sklearn.__version__
            else:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'unknown')
            print(f"✅ {package}: {version}")
        except ImportError:
            print(f"❌ {package}: Not installed")
            missing.append(pip_name)
    
    if missing:
        print(f"\n❌ Missing packages: {missing}")
        print("Install with: pip install " + " ".join(missing))
        return False
    else:
        print("\n✅ All dependencies available!")
        return True

def check_file_structure():
    """Check if all required files exist"""
    print_header("File Structure Check")
    
    required_files = [
        'app.py',
        'train_model.py',
        'test_training.py',
        'requirements.txt',
        'data/spam_sms.csv',
        'data/spam_email.csv',
        'data/test_sms.csv',
        'data/test_email.csv'
    ]
    
    missing = []
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file_path}: {size:,} bytes")
        else:
            print(f"❌ {file_path}: Not found")
            missing.append(file_path)
    
    if missing:
        print(f"\n❌ Missing files: {missing}")
        return False
    else:
        print("\n✅ All required files found!")
        return True

def check_data_files():
    """Check data file contents"""
    print_header("Data File Check")
    
    try:
        import pandas as pd
        
        # Check SMS data
        sms_df = pd.read_csv('data/spam_sms.csv')
        print(f"✅ SMS data: {len(sms_df)} rows, {len(sms_df.columns)} columns")
        print(f"   Columns: {list(sms_df.columns)}")
        
        # Check Email data
        email_df = pd.read_csv('data/spam_email.csv')
        print(f"✅ Email data: {len(email_df)} rows, {len(email_df.columns)} columns")
        print(f"   Columns: {list(email_df.columns)}")
        
        return True
    except Exception as e:
        print(f"❌ Error reading data files: {e}")
        return False

def test_training_script():
    """Test if training script can be imported and run"""
    print_header("Training Script Test")
    
    try:
        # Try to import the training module
        import train_model
        print("✅ train_model.py can be imported")
        
        # Check if main function exists
        if hasattr(train_model, 'main'):
            print("✅ main() function found")
        else:
            print("❌ main() function not found")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Error importing train_model.py: {e}")
        return False

def test_streamlit_app():
    """Test if Streamlit app can be imported"""
    print_header("Streamlit App Test")
    
    try:
        # Try to import the app module
        import app
        print("✅ app.py can be imported")
        
        # Check if required functions exist
        required_functions = ['analyze_text', 'process_csv_file']
        for func in required_functions:
            if hasattr(app, func):
                print(f"✅ {func}() function found")
            else:
                print(f"❌ {func}() function not found")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Error importing app.py: {e}")
        return False

def run_quick_training_test():
    """Run a quick training test"""
    print_header("Quick Training Test")
    
    try:
        # Run training script with timeout
        result = subprocess.run(
            [sys.executable, 'train_model.py'],
            capture_output=True,
            text=True,
            timeout=30,  # 30 second timeout
            cwd=os.getcwd()
        )
        
        print(f"Return code: {result.returncode}")
        
        if result.stdout:
            print("STDOUT (first 500 chars):")
            print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
        
        if result.stderr:
            print("STDERR (first 500 chars):")
            print(result.stderr[:500] + "..." if len(result.stderr) > 500 else result.stderr)
        
        if result.returncode == 0:
            print("✅ Quick training test passed!")
            return True
        else:
            print("❌ Quick training test failed!")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Quick training test timed out!")
        return False
    except Exception as e:
        print(f"❌ Error running quick training test: {e}")
        return False

def generate_fix_commands():
    """Generate commands to fix common issues"""
    print_header("Fix Commands")
    
    print("If you have issues, try these commands:")
    print("\n1. Install dependencies:")
    print("   pip install -r requirements.txt")
    
    print("\n2. Upgrade pip:")
    print("   python -m pip install --upgrade pip")
    
    print("\n3. Install specific versions:")
    print("   pip install pandas>=1.3.0 scikit-learn>=1.0.0 joblib>=1.1.0 streamlit>=1.25.0")
    
    print("\n4. Test locally:")
    print("   streamlit run app.py")
    
    print("\n5. Check Python path:")
    print("   python -c \"import sys; print(sys.path)\"")

def main():
    """Main debug function"""
    print("🔧 Spam Detection App - Debug Setup")
    print("This script will help identify and fix deployment issues")
    
    # Run all checks
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("File Structure", check_file_structure),
        ("Data Files", check_data_files),
        ("Training Script", test_training_script),
        ("Streamlit App", test_streamlit_app),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Error in {name} check: {e}")
            results.append((name, False))
    
    # Summary
    print_header("Summary")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name}: {status}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All checks passed! Your setup should work correctly.")
        
        # Run quick training test
        if run_quick_training_test():
            print("🎉 Ready for deployment!")
        else:
            print("⚠️ Setup looks good but training test failed.")
    else:
        print("⚠️ Some checks failed. See issues above.")
        generate_fix_commands()
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main() 