import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib
import os
import re
import io
import time

# Model Names 
SMS_MODEL_NAME = './spam_sms_model.pkl'
EMAIL_MODEL_NAME = './spam_email_model.pkl'

st.set_page_config(page_title='Spam Detection System', layout='wide')

# ===== Custom Header =====
col_title = st.columns([1])
with col_title[0]:
    st.markdown("""
    <h1 style='font-size: 60px; margin-bottom:0; color:red; text-align:center;'>🛡️ SPAM DETECTION SYSTEM</h1>
    <p style='font-size: 24px; color:red; margin-top:0; text-align:center;'>AI-powered SMS & Email Spam Classifier</p>
    """, unsafe_allow_html=True)

st.markdown("---")

# Check if models exist and load them
sms_model_exists = os.path.exists(SMS_MODEL_NAME)
email_model_exists = os.path.exists(EMAIL_MODEL_NAME)

if not sms_model_exists:
    st.error("❌q SMS model not found! Please ensure 'spam_sms_model.pkl' is in the app directory.")
    st.info("💡 **For developers:** Make sure to include the pre-trained model files in your deployment.")
    st.stop()
if not email_model_exists:
    st.error("❌ Email model not found! Please ensure 'spam_email_model.pkl' is in the app directory.")
    st.info("💡 **For developers:** Make sure to include the pre-trained model files in your deployment.")
    st.stop()

# Load the trained models
try:
    sms_model = joblib.load(SMS_MODEL_NAME)
    email_model = joblib.load(EMAIL_MODEL_NAME)
    st.success("✅ Pre-trained models loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.info("💡 **For developers:** Ensure model files are compatible with the current scikit-learn version.")
    st.stop()

# Function to extract email content
def extract_email_content(email_text):
    lines = email_text.split('\n')
    content_lines = []
    in_body = False
    for line in lines:
        if line.strip() == '':
            in_body = True
            continue
        if in_body:
            content_lines.append(line)
    if not content_lines:
        content_lines = lines
    return '\n'.join(content_lines)

# Function to analyze text for spam
def analyze_text(text, text_type):
    if not text.strip():
        return None, None
    if text_type == "Email":
        prediction = email_model.predict([text])[0]
        probability = email_model.predict_proba([text])[0]
    else:
        prediction = sms_model.predict([text])[0]
        probability = sms_model.predict_proba([text])[0]
    return prediction, probability

# Function to process CSV file
def process_csv_file(uploaded_file, file_type, selected_columns=None):
    try:
        # Try different CSV reading approaches
        df = None
        error_messages = []
        
        # Method 1: Standard CSV reading
        try:
            df = pd.read_csv(uploaded_file)
            st.success("CSV read successfully with standard method")
        except Exception as e:
            error_messages.append(f"Standard method failed: {str(e)}")
        
        # Method 2: Try with different encoding
        if df is None:
            try:
                uploaded_file.seek(0)  # Reset file pointer
                df = pd.read_csv(uploaded_file, encoding='utf-8')
                st.success("CSV read successfully with UTF-8 encoding")
            except Exception as e:
                error_messages.append(f"UTF-8 method failed: {str(e)}")
        
        # Method 3: Try with different separator
        if df is None:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=',', engine='python')
                st.success("CSV read successfully with Python engine")
            except Exception as e:
                error_messages.append(f"Python engine method failed: {str(e)}")
        
        # Method 4: Try reading as string first
        if df is None:
            try:
                uploaded_file.seek(0)
                content = uploaded_file.read().decode('utf-8')
                # Clean the content
                content = content.strip()
                if not content:
                    st.error("Uploaded file is empty")
                    return None
                
                # Try to parse manually
                lines = content.split('\n')
                if len(lines) < 2:
                    st.error("CSV file must have at least a header and one data row")
                    return None
                
                # Parse header
                header = lines[0].split(',')
                header = [col.strip().strip('"').strip("'") for col in header]
                
                # Parse data rows
                data = []
                for line in lines[1:]:
                    if line.strip():
                        row = line.split(',')
                        row = [col.strip().strip('"').strip("'") for col in row]
                        if len(row) == len(header):
                            data.append(row)
                
                df = pd.DataFrame(data, columns=header)
                st.success("CSV read successfully with manual parsing")
                
            except Exception as e:
                error_messages.append(f"Manual parsing failed: {str(e)}")
        
        if df is None:
            st.error("Failed to read CSV file with all methods")
            st.write("Error messages:", error_messages)
            return None
        
        # Display file info for debugging
        st.write(f"**File loaded successfully:**")
        st.write(f"- Rows: {len(df)}")
        st.write(f"- Columns: {list(df.columns)}")
        st.write(f"- First few rows:")
        st.dataframe(df.head())
        
        if file_type == "SMS":
            # For SMS, look for common column names
            text_column = None
            for col in ['text', 'message', 'sms', 'content', 'body']:
                if col in df.columns:
                    text_column = col
                    break
            
            if text_column is None:
                st.error("Could not find text column. Please ensure your CSV has a column named 'text', 'message', 'sms', 'content', or 'body'.")
                st.write("Available columns:", list(df.columns))
                return None
            
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, row in df.iterrows():
                text = str(row[text_column])
                prediction, probability = analyze_text(text, "SMS")
                
                results.append({
                    'original_text': text,
                    'prediction': prediction,
                    'spam_probability': f"{probability[1]*100:.2f}%" if probability is not None and len(probability) > 1 else "N/A",
                    'ham_probability': f"{probability[0]*100:.2f}%" if probability is not None and len(probability) > 0 else "N/A",
                    'confidence': f"{max(probability)*100:.2f}%" if probability is not None and len(probability) > 0 else "N/A"
                })
                
                progress_bar.progress((idx + 1) / len(df))
                status_text.text(f"Processing {idx + 1}/{len(df)} messages...")
            
            progress_bar.empty()
            status_text.empty()
            
            return pd.DataFrame(results)
            
        else:  # Email
            if not selected_columns:
                st.error("Please select the columns for email processing.")
                return None
            
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, row in df.iterrows():
                # Combine selected columns into email text
                email_parts = []
                for col in selected_columns:
                    if col in df.columns and pd.notna(row[col]):
                        email_parts.append(f"{col}: {row[col]}")
                
                email_text = "\n".join(email_parts)
                prediction, probability = analyze_text(email_text, "Email")
                
                result_row = {
                    'prediction': prediction,
                    'spam_probability': f"{probability[1]*100:.2f}%" if probability is not None and len(probability) > 1 else "N/A",
                    'ham_probability': f"{probability[0]*100:.2f}%" if probability is not None and len(probability) > 0 else "N/A",
                    'confidence': f"{max(probability)*100:.2f}%" if probability is not None and len(probability) > 0 else "N/A"
                }
                
                # Add original columns
                for col in selected_columns:
                    if col in df.columns:
                        result_row[f'original_{col}'] = row[col]
                
                results.append(result_row)
                
                progress_bar.progress((idx + 1) / len(df))
                status_text.text(f"Processing {idx + 1}/{len(df)} emails...")
            
            progress_bar.empty()
            status_text.empty()
            
            return pd.DataFrame(results)
            
    except Exception as e:
        st.error(f"Error processing CSV file: {e}")
        st.write("Full error details:", str(e))
        return None

# Create tabs for SMS, Email, and CSV Upload
tab1, tab2, tab3 = st.tabs(["📱 SMS Detection", "📧 Email Detection", "📁 CSV Batch Processing"])

with tab1:
    st.header("SMS Spam Detection")
    st.write("Enter an SMS message to check if it's spam or legitimate.")
    sms_text = st.text_area("SMS Message:", height=150, placeholder="Enter the SMS message you want to check...")
    if st.button("🔍 Check SMS", key="sms_button"):
        prediction, probability = analyze_text(sms_text, "SMS")
        if prediction is not None and probability is not None:
            st.subheader("📊 Analysis Results")
            if prediction.lower() == 'spam':
                st.error(f"🚨 **SPAM DETECTED**")
                st.write(f"Confidence: {probability[1]*100:.2f}%")
            else:
                st.success(f"✅ **NOT SPAM** (Legitimate)")
                st.write(f"Confidence: {probability[0]*100:.2f}%")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Legitimate Probability", f"{probability[0]*100:.2f}%")
            with col2:
                st.metric("Spam Probability", f"{probability[1]*100:.2f}%")
        else:
            st.warning("Please enter an SMS message to check.")
    
    # SMS Examples
    with st.expander("🧪 SMS Examples to Test"):
        st.subheader("📱 SMS Test Examples")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**🚨 Spam SMS Examples:**")
            st.code("""Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121""")
            st.code("""WINNER!! As a valued network customer you have been selected to receivea £900 prize reward!""")
            st.code("""URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot!""")
            st.code("""SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575.""")
        with col2:
            st.write("**✅ Legitimate SMS Examples:**")
            st.code("""Go until jurong point, crazy.. Available only in bugis n great world la e buffet""")
            st.code("""I'm gonna be home soon and i don't want to talk about this stuff anymore tonight""")
            st.code("""Ok lar... Joking wif u oni...""")
            st.code("""Hi! How's you and how did saturday go? I was just texting to see if you'd decided to do anything tomo.""")

with tab2:
    st.header("Email Spam Detection")
    st.write("Enter an email (including headers) to check if it's spam or legitimate.")
    email_text = st.text_area("Email Content:", height=200, 
                             placeholder="""From: sender@example.com
To: recipient@example.com
Subject: Email Subject
Date: Mon, 01 Jan 2024 12:00:00 +0000

Email body content goes here...""")
    if st.button("🔍 Check Email", key="email_button"):
        prediction, probability = analyze_text(email_text, "Email")
        if prediction is not None and probability is not None:
            st.subheader("📊 Analysis Results")
            if prediction.lower() == 'spam':
                st.error(f"🚨 **SPAM DETECTED**")
                st.write(f"Confidence: {probability[1]*100:.2f}%")
            else:
                st.success(f"✅ **NOT SPAM** (Legitimate)")
                st.write(f"Confidence: {probability[0]*100:.2f}%")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Legitimate Probability", f"{probability[0]*100:.2f}%")
            with col2:
                st.metric("Spam Probability", f"{probability[1]*100:.2f}%")
        else:
            st.warning("Please enter an email to check.")
    
    # Email Examples
    with st.expander("🧪 Email Examples to Test"):
        st.subheader("📧 Email Test Examples")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**🚨 Spam Email Examples:**")
            st.code("""From: lottery@win.com
To: user@example.com
Subject: CONGRATULATIONS! You've won $1,000,000!

Dear Winner,
You have been selected to receive $1,000,000! Click here to claim your prize now!""")
            st.code("""From: pharmacy@meds.com
To: user@example.com
Subject: URGENT: Your prescription is ready

Hi there,
Your prescription for Viagra is ready for pickup. 
Click here to order now: meds.com""")
            st.code("""From: bank@security.com
To: user@example.com
Subject: ACCOUNT SUSPENDED - Immediate Action Required

Dear Customer,
Your account has been suspended due to suspicious activity.
Click here to verify your identity: secure-bank.com""")
        with col2:
            st.write("**✅ Legitimate Email Examples:**")
            st.code("""From: boss@company.com
To: employee@example.com
Subject: Meeting Tomorrow

Hi,
Let's have a meeting tomorrow at 2 PM to discuss the project progress.
Best regards,
Boss""")
            st.code("""From: friend@gmail.com
To: user@example.com
Subject: Weekend Plans

Hey!
Are you free this weekend? I was thinking we could grab coffee.
Let me know what works for you.
Cheers!""")
            st.code("""From: support@amazon.com
To: user@example.com
Subject: Your Order #12345 has been shipped

Dear Customer,
Your recent order has been shipped and will arrive on Monday.
Track your package: amazon.com/track""")

with tab3:
    st.header("📁 CSV Batch Processing")
    st.write("Upload a CSV file to process multiple SMS messages or emails at once.")
    
    # Quick test examples section
    st.subheader("🚀 Quick Test Examples")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📱 Test SMS File:**")
        st.write("Contains 5 sample SMS messages (mix of spam and legitimate)")
        if st.button("📥 Load Test SMS File", key="load_test_sms"):
            try:
                test_sms_path = os.path.join('data', 'test_sms.csv')
                st.write(f"Looking for file at: {test_sms_path}")
                if os.path.exists(test_sms_path):
                    with open(test_sms_path, 'r', encoding='utf-8') as f:
                        test_sms_content = f.read()
                    st.session_state['uploaded_file'] = test_sms_content
                    st.session_state['file_type'] = "SMS"
                    st.session_state['file_name'] = "test_sms.csv"
                    st.success("✅ Test SMS file loaded!")
                    st.write("**Preview:**")
                    test_df = pd.read_csv(test_sms_path)
                    st.dataframe(test_df)
                    st.write(f"**File size:** {len(test_sms_content)} characters")
                    st.write(f"**Columns found:** {list(test_df.columns)}")
                else:
                    st.error(f"Test SMS file not found at: {test_sms_path}")
                    st.write("Available files in data folder:")
                    data_files = os.listdir('data') if os.path.exists('data') else []
                    for file in data_files:
                        st.write(f"- {file}")
            except Exception as e:
                st.error(f"Error loading test SMS file: {e}")
                st.write("Full error details:", str(e))
    
    with col2:
        st.write("**📧 Test Email File:**")
        st.write("Contains 5 sample emails (mix of spam and legitimate)")
        if st.button("📥 Load Test Email File", key="load_test_email"):
            try:
                test_email_path = os.path.join('data', 'test_email.csv')
                st.write(f"Looking for file at: {test_email_path}")
                if os.path.exists(test_email_path):
                    with open(test_email_path, 'r', encoding='utf-8') as f:
                        test_email_content = f.read()
                    st.session_state['uploaded_file'] = test_email_content
                    st.session_state['file_type'] = "Email"
                    st.session_state['file_name'] = "test_email.csv"
                    st.success("✅ Test Email file loaded!")
                    st.write("**Preview:**")
                    test_df = pd.read_csv(test_email_path)
                    st.dataframe(test_df)
                    st.write(f"**File size:** {len(test_email_content)} characters")
                    st.write(f"**Columns found:** {list(test_df.columns)}")
                else:
                    st.error(f"Test Email file not found at: {test_email_path}")
                    st.write("Available files in data folder:")
                    data_files = os.listdir('data') if os.path.exists('data') else []
                    for file in data_files:
                        st.write(f"- {file}")
            except Exception as e:
                st.error(f"Error loading test Email file: {e}")
                st.write("Full error details:", str(e))
    
    st.markdown("---")
    
    # File type selection with auto-selection
    default_index = 0  # Default to SMS
    
    # Check if we have a test file loaded and set the appropriate index
    if 'file_type' in st.session_state:
        if st.session_state['file_type'] == "SMS":
            default_index = 0
        elif st.session_state['file_type'] == "Email":
            default_index = 1
    
    file_type = st.selectbox("Select file type:", ["SMS", "Email"], index=default_index)
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    # Check if we have a test file loaded in session state
    if 'uploaded_file' in st.session_state and st.session_state.get('file_type') == file_type:
        st.success(f"✅ Test file loaded: {st.session_state.get('file_name', 'test_file.csv')}")
        if st.button("🔄 Clear Test File"):
            # Clear all test file related session state
            keys_to_clear = ['uploaded_file', 'file_type', 'file_name', 'auto_select_sms', 'auto_select_email']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    # Process the file (either uploaded or from session state)
    file_to_process = uploaded_file
    if 'uploaded_file' in st.session_state and st.session_state.get('file_type') == file_type:
        # Create a file-like object from session state content
        file_to_process = io.StringIO(st.session_state['uploaded_file'])
    
    if file_to_process is not None:
        try:
            # Preview the file
            df_preview = pd.read_csv(file_to_process)
            st.write("**File Preview:**")
            st.dataframe(df_preview.head())
            st.write(f"**Total rows:** {len(df_preview)}")
            st.write(f"**Columns:** {list(df_preview.columns)}")
            
            # Column selection for email
            selected_columns = None
            if file_type == "Email":
                st.write("**Select columns to include in email analysis:**")
                selected_columns = st.multiselect(
                    "Choose columns:",
                    df_preview.columns,
                    default=df_preview.columns[:3] if len(df_preview.columns) >= 3 else df_preview.columns
                )
            
            # Process button
            if st.button("🚀 Process CSV File"):
                if file_type == "Email" and not selected_columns:
                    st.error("Please select at least one column for email processing.")
                else:
                    with st.spinner("Processing CSV file..."):
                        # Reset file pointer for processing
                        if hasattr(file_to_process, 'seek'):
                            file_to_process.seek(0)
                        results_df = process_csv_file(file_to_process, file_type, selected_columns)
                        
                        if results_df is not None:
                            st.success("✅ Processing completed!")
                            
                            # Display results
                            st.write("**Results Preview:**")
                            st.dataframe(results_df.head())
                            
                            # Download button
                            csv_buffer = io.StringIO()
                            results_df.to_csv(csv_buffer, index=False)
                            csv_str = csv_buffer.getvalue()
                            
                            st.download_button(
                                label="📥 Download Results CSV",
                                data=csv_str,
                                file_name=f"spam_detection_results_{file_type.lower()}.csv",
                                mime="text/csv"
                            )
                            
                            # Summary statistics
                            st.write("**Summary Statistics:**")
                            if 'prediction' in results_df.columns:
                                spam_count = len(results_df[results_df['prediction'].str.lower() == 'spam'])
                                ham_count = len(results_df[results_df['prediction'].str.lower() == 'ham'])
                                total_count = len(results_df)
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Messages", total_count)
                                with col2:
                                    st.metric("Spam Detected", spam_count)
                                with col3:
                                    st.metric("Legitimate", ham_count)
                                
                                if total_count > 0:
                                    st.write(f"**Spam Rate:** {spam_count/total_count*100:.1f}%")
                                    
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")

# ===== Sidebar Polishing =====
with st.sidebar:
    st.header("☁️ Cloud9")
    st.markdown("🛡️ **Spam Detection System**")
    st.markdown("---")
    
    with st.expander("🚦 Quick Guide", expanded=True):
        st.markdown("""
        1. **Choose a feature tab** (SMS, Email, or CSV Batch)
        2. **Enter or upload your message(s)**
        3. **View prediction and confidence**
        4. **[Optional] Download results as CSV**
        5. **[Optional] Add new labeled data to improve the model**
        """)
    
    with st.expander("📋 About Project"):
        st.write("Cloud9 is an intelligent spam detection system that uses machine learning to identify and classify spam messages in both SMS and email communications. Built with state-of-the-art NLP techniques and TF-IDF vectorization.")
    
    with st.expander("🚀 Key Features"):
        st.write("**📱 SMS Detection:** Real-time analysis, TF-IDF, high accuracy, confidence scoring")
        st.write("**📧 Email Detection:** Multi-column, header/body, spam pattern recognition")
        st.write("**📁 Batch Processing:** CSV upload, bulk analysis, export, progress tracking")
        st.write("**🔧 Advanced:** Dual models, real-time training, test data integration, pro UI/UX")
    
    with st.expander("👥 Team Members"):
        st.write("**Group Name:** Cloud9")
        st.write("**Team Lead:** Mohd Bashar Azaz (AIIUL135)")
        st.write("**Members:** Mohd Hamza, Mohd Haris, Mohd Tariq Khan, Mohd Salman, Mohd Zeeshan, Mohd Kamran, Mohd Asadullah Siddiqui, Mohd Sulaiman Warsi, Mohd Armaan Khan")
    
    with st.expander("⚙️ Technical Stack"):
        st.write("**Frontend:** Streamlit\n**Backend:** Python\n**ML:** Scikit-learn (MultinomialNB, TF-IDF)\n**Data:** Pandas")
    
    with st.expander("📈 Model Details"):
        st.write(f"**SMS Model:** {SMS_MODEL_NAME}")
        st.write(f"**Email Model:** {EMAIL_MODEL_NAME}")
        st.write("**Algorithm:** Multinomial Naive Bayes\n**Features:** TF-IDF vectorization\n**Training Data:** SMS & Email datasets")
    
    with st.expander("🔧 How It Works"):
        st.write("1. Input message(s)\n2. System analyzes with model\n3. Returns spam/legitimate\n4. Shows probability\n5. Export as CSV")
    
    st.markdown("---")
    st.header("📤 Add Labeled Data & Retrain")
    st.write("Upload new labeled data to improve the model. The data will be appended to the training set and the model will be retrained.")
    retrain_file_type = st.selectbox("Select data type to add:", ["SMS", "Email"], key="retrain_file_type")
    retrain_file = st.file_uploader("Upload labeled CSV (same columns as training data)", type=["csv"], key="retrain_file")
    if retrain_file is not None:
        st.info("**Note:** Uploaded data must be labeled and match the training columns. Retraining may take a moment.")
        if st.button("✅ Append & Retrain Model", key="append_retrain_btn"):
            try:
                if retrain_file_type == "SMS":
                    train_path = os.path.join("data", "spam_sms.csv")
                    expected_cols = ["label", "text"]
                else:
                    train_path = os.path.join("data", "spam_email.csv")
                    expected_cols = ["Category", "Message"]
                new_df = pd.read_csv(retrain_file)
                if list(new_df.columns) != expected_cols:
                    st.error(f"Uploaded CSV must have columns: {expected_cols}")
                else:
                    orig_df = pd.read_csv(train_path)
                    combined_df = pd.concat([orig_df, new_df], ignore_index=True)
                    combined_df.to_csv(train_path, index=False)
                    st.success(f"✅ Data appended to {train_path}. Retraining model...")
                    with st.spinner("Retraining model. Please wait..."):
                        result = subprocess.run(['python', 'train_model.py'], capture_output=True, text=True, cwd=os.path.dirname(__file__) or '.')
                        if result.returncode == 0:
                            st.success('✅ Model retrained successfully!')
                            st.text(result.stdout)
                            st.info("Please reload the page to use the updated model.")
                        else:
                            st.error('❌ Retraining failed!')
                            st.text(result.stderr)
            except Exception as e:
                st.error(f"Error during append/retrain: {e}")
    st.markdown("---")
    st.write("**Version:** Cloud9-SpamShield-v1.0.0")
    st.write("**Last Updated:** July 2025")
    st.write("**License:** Cloud9 Development Team")
    st.markdown("<hr style='border:1px solid #eee;'>", unsafe_allow_html=True)
    st.markdown("<center><small>Made with ❤️ by Cloud9 Team</small></center>", unsafe_allow_html=True)

# ===== Footer =====
st.markdown("<hr style='border:1px solid #eee;'>", unsafe_allow_html=True)
st.markdown("<center><small>Cloud9 Spam Detection System &copy; 2025 | All rights reserved.</small></center>", unsafe_allow_html=True)
