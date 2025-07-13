import os
import sys
import traceback

# Check and install required dependencies
def check_and_install_dependencies():
    """Check if required packages are available and install if needed"""
    required_packages = ['pandas', 'sklearn', 'joblib']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {missing_packages}")
        print("Please install required packages using: pip install " + " ".join(missing_packages))
        return False
    return True

# Import required libraries with error handling
try:
    import pandas as pd
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import make_pipeline
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Please ensure all required packages are installed:")
    print("pip install pandas scikit-learn joblib")
    sys.exit(1)

def train_sms_model():
    """Train SMS spam detection model"""
    try:
        # Check if data file exists
        sms_data_path = 'data/spam_sms.csv'
        if not os.path.exists(sms_data_path):
            print(f"Error: SMS data file not found at {sms_data_path}")
            return False, "SMS data file not found"
        
        print("Loading SMS data...")
        sms_df = pd.read_csv(sms_data_path)
        print(f"SMS data loaded: {len(sms_df)} rows")
        
        # Try to handle both possible column names
        if 'label' in sms_df.columns and 'text' in sms_df.columns:
            sms_X = sms_df['text'].astype(str)
            sms_y = sms_df['label'].astype(str)
        elif len(sms_df.columns) >= 2:
            sms_X = sms_df.iloc[:,1].astype(str)
            sms_y = sms_df.iloc[:,0].astype(str)
        else:
            return False, "SMS data format not recognized"
        
        print("Training SMS model...")
        sms_X_train, sms_X_test, sms_y_train, sms_y_test = train_test_split(
            sms_X, sms_y, test_size=0.2, random_state=42
        )
        sms_model = make_pipeline(TfidfVectorizer(), MultinomialNB())
        sms_model.fit(sms_X_train, sms_y_train)
        
        # Save model
        joblib.dump(sms_model, 'spam_sms_model.pkl')
        sms_score = sms_model.score(sms_X_test, sms_y_test)
        print(f"SMS Spam Detection Model trained and saved. Test accuracy: {sms_score:.2f}")
        return True, f"SMS model trained successfully with accuracy: {sms_score:.2f}"
        
    except Exception as e:
        error_msg = f"Error training SMS model: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return False, error_msg

def train_email_model():
    """Train Email spam detection model"""
    try:
        # Check if data file exists
        email_data_path = 'data/spam_email.csv'
        if not os.path.exists(email_data_path):
            print(f"Error: Email data file not found at {email_data_path}")
            return False, "Email data file not found"
        
        print("Loading Email data...")
        email_df = pd.read_csv(email_data_path)
        print(f"Email data loaded: {len(email_df)} rows")
        
        # Try to handle both possible column names
        if 'Message' in email_df.columns and 'Category' in email_df.columns:
            email_X = email_df['Message'].astype(str)
            email_y = email_df['Category'].astype(str)
        elif len(email_df.columns) >= 2:
            email_X = email_df.iloc[:,1].astype(str)
            email_y = email_df.iloc[:,0].astype(str)
        else:
            return False, "Email data format not recognized"
        
        print("Training Email model...")
        email_X_train, email_X_test, email_y_train, email_y_test = train_test_split(
            email_X, email_y, test_size=0.2, random_state=42
        )
        email_model = make_pipeline(TfidfVectorizer(), MultinomialNB())
        email_model.fit(email_X_train, email_y_train)
        
        # Save model
        joblib.dump(email_model, 'spam_email_model.pkl')
        email_score = email_model.score(email_X_test, email_y_test)
        print(f"Email Spam Detection Model trained and saved. Test accuracy: {email_score:.2f}")
        return True, f"Email model trained successfully with accuracy: {email_score:.2f}"
        
    except Exception as e:
        error_msg = f"Error training Email model: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return False, error_msg

def main():
    """Main training function"""
    print("=== Spam Detection Model Training ===")
    
    # Check dependencies
    if not check_and_install_dependencies():
        print("Dependency check failed. Exiting.")
        return
    
    # Check current working directory and data availability
    print(f"Current working directory: {os.getcwd()}")
    print(f"Available files in current directory: {os.listdir('.')}")
    
    if os.path.exists('data'):
        print(f"Data directory found. Contents: {os.listdir('data')}")
    else:
        print("Data directory not found!")
    
    # Train SMS model
    print("\n--- Training SMS Model ---")
    sms_success, sms_message = train_sms_model()
    
    # Train Email model
    print("\n--- Training Email Model ---")
    email_success, email_message = train_email_model()
    
    # Summary
    print("\n=== Training Summary ===")
    if sms_success:
        print(f"SUCCESS SMS Model: {sms_message}")
    else:
        print(f"FAILED SMS Model: {sms_message}")
    
    if email_success:
        print(f"SUCCESS Email Model: {email_message}")
    else:
        print(f"FAILED Email Model: {email_message}")
    
    if sms_success and email_success:
        print("\nSUCCESS: Both models trained successfully!")
    else:
        print("\nWARNING: Some models failed to train. Check the error messages above.")

if __name__ == "__main__":
    main()




























