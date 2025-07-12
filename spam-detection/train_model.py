import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# --- SMS Model ---
sms_df = pd.read_csv('data/spam_sms.csv')
# Try to handle both possible column names
if 'label' in sms_df.columns and 'text' in sms_df.columns:
    sms_X = sms_df['text'].astype(str)
    sms_y = sms_df['label'].astype(str)
else:
    sms_X = sms_df.iloc[:,1].astype(str)
    sms_y = sms_df.iloc[:,0].astype(str)

sms_X_train, sms_X_test, sms_y_train, sms_y_test = train_test_split(sms_X, sms_y, test_size=0.2, random_state=42)
sms_model = make_pipeline(TfidfVectorizer(), MultinomialNB())
sms_model.fit(sms_X_train, sms_y_train)
joblib.dump(sms_model, 'spam_sms_model.pkl')
sms_score = sms_model.score(sms_X_test, sms_y_test)
print(f"SMS Spam Detection Model trained and saved as 'spam_sms_model.pkl'. Test accuracy: {sms_score:.2f}")

# --- Email Model ---
email_df = pd.read_csv('data/spam_email.csv')
if 'Message' in email_df.columns and 'Category' in email_df.columns:
    email_X = email_df['Message'].astype(str)
    email_y = email_df['Category'].astype(str)
else:
    email_X = email_df.iloc[:,1].astype(str)
    email_y = email_df.iloc[:,0].astype(str)

email_X_train, email_X_test, email_y_train, email_y_test = train_test_split(email_X, email_y, test_size=0.2, random_state=42)
email_model = make_pipeline(TfidfVectorizer(), MultinomialNB())
email_model.fit(email_X_train, email_y_train)
joblib.dump(email_model, 'spam_email_model.pkl')
email_score = email_model.score(email_X_test, email_y_test)
print(f"Email Spam Detection Model trained and saved as 'spam_email_model.pkl'. Test accuracy: {email_score:.2f}")




























