# Spam Detection App - Deployment Guide

## 🚀 Streamlit Cloud Deployment

### Prerequisites

- All files must be in the same directory
- `requirements.txt` must be in the root directory
- Data files must be included in the deployment

### File Structure

```
spam-detection/
├── app.py                 # Main Streamlit application
├── train_model.py         # Training script
├── test_training.py       # Test script for debugging
├── requirements.txt       # Python dependencies
├── data/                  # Data directory
│   ├── spam_sms.csv      # SMS training data
│   ├── spam_email.csv    # Email training data
│   ├── test_sms.csv      # Test SMS data
│   └── test_email.csv    # Test email data
└── DEPLOYMENT_GUIDE.md   # This file
```

### Common Deployment Issues & Solutions

#### 1. **Missing Dependencies**

**Problem**: Import errors for pandas, sklearn, etc.
**Solution**:

- Ensure `requirements.txt` is in the root directory
- Check that all dependencies are listed with correct versions
- Deploy with the updated requirements.txt

#### 2. **Data Files Not Found**

**Problem**: Training fails because data files are missing
**Solution**:

- Ensure the `data/` directory is included in deployment
- Check file paths are relative to the app directory
- Use the "Test Training Environment" button to verify

#### 3. **Training Timeout**

**Problem**: Training takes too long and times out
**Solution**:

- Training is limited to 5 minutes in the app
- For large datasets, consider pre-training models locally
- Use smaller datasets for testing

#### 4. **Permission Issues**

**Problem**: Cannot write model files
**Solution**:

- Streamlit Cloud has read-only filesystem
- Models must be pre-trained and included in deployment
- Use the training feature only for development

### Deployment Steps

1. **Prepare Files**

   ```bash
   # Ensure all files are in the spam-detection directory
   ls -la spam-detection/
   ```

2. **Check Requirements**

   ```bash
   # Verify requirements.txt is correct
   cat spam-detection/requirements.txt
   ```

3. **Test Locally**

   ```bash
   # Test the app locally first
   cd spam-detection
   streamlit run app.py
   ```

4. **Deploy to Streamlit Cloud**
   - Upload the entire `spam-detection` directory
   - Ensure `app.py` is in the root of the uploaded files
   - Set the main file to `app.py`

### Troubleshooting

#### Use the Test Button

1. Click "🧪 Test Training Environment" in the app
2. Check the output for specific error messages
3. Address any issues found

#### Check File Paths

The app uses these relative paths:

- `data/spam_sms.csv`
- `data/spam_email.csv`
- `train_model.py`
- `test_training.py`

#### Common Error Messages

**"ModuleNotFoundError: No module named 'pandas'"**

- Solution: Check requirements.txt includes pandas

**"FileNotFoundError: data/spam_sms.csv"**

- Solution: Ensure data directory is included in deployment

**"PermissionError: [Errno 13] Permission denied"**

- Solution: Pre-train models locally and include .pkl files

### Pre-trained Models

For production deployment, it's recommended to:

1. Train models locally using `train_model.py`
2. Include the generated `.pkl` files in deployment
3. Disable the training button in production

### Performance Tips

1. **Use Pre-trained Models**: Include `spam_sms_model.pkl` and `spam_email_model.pkl`
2. **Limit Training**: Only use training for development/testing
3. **Optimize Data**: Use smaller datasets for faster training
4. **Cache Results**: Use Streamlit's caching for repeated predictions

### Support

If you encounter issues:

1. Check the test output using the "Test Training Environment" button
2. Verify all files are present and correctly named
3. Ensure requirements.txt is properly formatted
4. Check Streamlit Cloud logs for detailed error messages
