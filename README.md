# 🛡️ Cloud9 Spam Detection System

An intelligent AI-powered spam detection system that classifies SMS messages and emails as spam or legitimate using machine learning techniques. Built with Streamlit for an interactive web interface and scikit-learn for robust ML models.

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![License](https://img.shields.io/badge/License-Cloud9-green.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

## � Quick Start

```bash
# 1. Clone and navigate
git clone <repository-url>
cd "spam detection by custom llm"

# 2. Setup environment
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# 3. Train models and run
cd spam-detection
python train_model.py
streamlit run app.py
```

**🌐 Open your browser to `http://localhost:8501` and start detecting spam!**

## �📋 Table of Contents
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Training](#-model-training)
- [API Reference](#-api-reference)
- [Dataset Information](#-dataset-information)
- [Technical Stack](#-technical-stack)
- [Team](#-team)
- [License](#-license)

## ✨ Features

### 🎯 Core Capabilities
- **🔍 Real-time Detection**: Instant spam classification for SMS and emails
- **� Confidence Scoring**: Probability percentages for each prediction
- **📁 Batch Processing**: Upload CSV files for bulk analysis
- **🔄 Model Retraining**: Add new data and retrain models on-the-fly
- **�📱 Mobile Responsive**: Works on desktop, tablet, and mobile devices

### 📱 SMS Spam Detection
- Real-time SMS message classification
- High accuracy using TF-IDF vectorization
- Confidence scoring for predictions
- Built-in test examples
- Support for multiple languages

### 📧 Email Spam Detection
- Multi-column email analysis (From, To, Subject, Body)
- Header and body content processing
- Advanced spam pattern recognition
- Email-specific feature extraction
- Handles various email formats

### 📁 Batch Processing
- CSV file upload for bulk analysis
- Progress tracking for large datasets
- Downloadable results in CSV format
- Support for both SMS and email datasets
- Error handling and validation

### 🔧 Advanced Features
- Dual model architecture (separate SMS and Email models)
- Real-time model training and retraining
- Test data integration
- Professional UI/UX with Streamlit
- Model performance metrics
- Export functionality
- Data visualization

## 📁 Project Structure

```
spam-detection/
├── app.py                    # Main Streamlit application
├── train_model.py           # Model training script
├── spam_sms_model.pkl       # Trained SMS model
├── spam_email_model.pkl     # Trained Email model
├── data/
│   ├── spam_sms.csv        # SMS training dataset
│   ├── spam_email.csv      # Email training dataset
│   ├── test_sms.csv        # SMS test examples
│   └── test_email.csv      # Email test examples
├── requirements.txt         # Python dependencies
└── README.md               # Project documentation
```

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd "spam detection by custom llm"
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv myenv

# Activate virtual environment
# On Windows:
myenv\Scripts\activate
# On macOS/Linux:
source myenv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Navigate to Project Directory
```bash
cd spam-detection
```

### Step 5: Train Initial Models (First Time Setup)
```bash
python train_model.py
```

This will create the required model files:
- `spam_sms_model.pkl`
- `spam_email_model.pkl`

## 🎯 Usage

### Running the Application
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Using the Web Interface

#### 1. SMS Detection Tab
- Enter an SMS message in the text area
- Click "🔍 Check SMS" to analyze
- View prediction results with confidence scores
- Try built-in examples for testing

#### 2. Email Detection Tab
- Enter complete email content (including headers)
- Click "🔍 Check Email" to analyze
- View detailed analysis results
- Test with provided email examples

#### 3. CSV Batch Processing Tab
- Select file type (SMS or Email)
- Upload CSV file or use test files
- For emails: select columns to analyze
- Click "🚀 Process CSV File"
- Download results as CSV

### CSV File Format

#### SMS CSV Format
```csv
text
"Your SMS message here"
"Another SMS message"
```

#### Email CSV Format
```csv
From,To,Subject,Body
sender@example.com,recipient@example.com,Subject Line,Email body content
```

## 🔄 Model Training

### Training New Models
```bash
python train_model.py
```

### Retraining with New Data
1. Use the sidebar in the web app
2. Upload labeled CSV data
3. Click "✅ Append & Retrain Model"
4. Wait for training completion

### Training Data Format

#### SMS Training Data (`data/spam_sms.csv`)
```csv
label,text
ham,"Normal message"
spam,"Spam message"
```

#### Email Training Data (`data/spam_email.csv`)
```csv
Category,Message
ham,"Legitimate email content"
spam,"Spam email content"
```

## 🔧 API Reference

### Core Functions

#### `analyze_text(text, text_type)`
Analyzes text for spam classification.
- **Parameters:**
  - `text` (str): Text to analyze
  - `text_type` (str): "SMS" or "Email"
- **Returns:** 
  - `prediction` (str): "spam" or "ham"
  - `probability` (array): Confidence scores

#### `process_csv_file(uploaded_file, file_type, selected_columns)`
Processes CSV files for batch analysis.
- **Parameters:**
  - `uploaded_file`: Uploaded CSV file
  - `file_type` (str): "SMS" or "Email"
  - `selected_columns` (list): Columns to analyze (for emails)
- **Returns:** DataFrame with results

## 📊 Dataset Information

### SMS Dataset
- **Size:** ~5,000+ messages
- **Labels:** ham (legitimate), spam
- **Features:** Text content
- **Source:** SMS Spam Collection Dataset

### Email Dataset
- **Size:** ~5,000+ emails
- **Labels:** ham (legitimate), spam
- **Features:** Email headers and body content
- **Source:** Email Spam Dataset

## 🛠️ Technical Stack

- **Frontend:** Streamlit
- **Backend:** Python 3.7+
- **Machine Learning:** 
  - Scikit-learn
  - Multinomial Naive Bayes
  - TF-IDF Vectorization
- **Data Processing:** Pandas, NumPy
- **Model Persistence:** Joblib

### Dependencies
```
numpy
pandas
scikit-learn
joblib
streamlit
```

## 🏆 Model Performance

### SMS Model
- **Algorithm:** Multinomial Naive Bayes with TF-IDF
- **Test Accuracy:** ~97%
- **Features:** Text vectorization with TF-IDF

### Email Model
- **Algorithm:** Multinomial Naive Bayes with TF-IDF
- **Test Accuracy:** ~96%
- **Features:** Combined email headers and body content

## 🚨 Troubleshooting

### Common Issues

#### Models Not Found Error
```bash
# Solution: Train the models first
python train_model.py
```

#### CSV Upload Issues
- Ensure CSV has proper headers
- Check file encoding (UTF-8 recommended)
- Verify column names match expected format

#### Streamlit Port Issues
```bash
# Run on different port
streamlit run app.py --server.port 8502
```

## 🔮 Future Enhancements

- [ ] Deep learning models (LSTM, BERT)
- [ ] Real-time email integration
- [ ] API endpoints for external integration
- [ ] Multi-language support
- [ ] Advanced visualization dashboard
- [ ] Mobile app development

## 👥 Team

**Group Name:** Cloud9

**Team Lead:** Mohd Bashar Azaz (AIIUL135)

**Team Members:**
- Mohd Hamza
- Mohd Haris
- Mohd Tariq Khan
- Mohd Salman
- Mohd Zeeshan
- Mohd Kamran
- Mohd Asadullah Siddiqui
- Mohd Sulaiman Warsi
- Mohd Armaan Khan

## 📄 License

© 2025 Cloud9 Development Team. All rights reserved.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📞 Support

For support and questions, please contact the Cloud9 development team.

---

**Made with ❤️ by Cloud9 Team**

*Cloud9 Spam Detection System - Protecting your communications with AI*

## 📚 Examples

### SMS Spam Detection Example

```python
from spam_detection import analyze_text

# Example SMS message
sms_message = "You have won a prize! Click here to claim it."

# Analyze the SMS message
prediction, probability = analyze_text(sms_message, "SMS")

print("Prediction:", prediction)
print("Confidence:", probability)
```

### Email Spam Detection Example

```python
from spam_detection import analyze_text

# Example email content
email_content = """
From: sender@example.com
To: recipient@example.com
Subject: Urgent: Your account has been compromised!

Dear recipient,

Your account has been compromised. Please click the link below to reset your password.

Best,
Sender
"""

# Analyze the email content
prediction, probability = analyze_text(email_content, "Email")

print("Prediction:", prediction)
print("Confidence:", probability)
```

### CSV Batch Processing Example

```python
from spam_detection import process_csv_file

# Example CSV file
csv_file = "example_sms.csv"

# Process the CSV file
results = process_csv_file(csv_file, "SMS", [])

print("Results:")
print(results)