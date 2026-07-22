import pandas as pd
import numpy as np
import random
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 1. GENERATE 2,100 BALANCED USER STORIES DATASET
# -------------------------------------------------------------
roles = [
    "user", "administrator", "customer", "support manager", "guest", 
    "vendor", "system admin", "billing coordinator", "security auditor"
]

# HIGH PRIORITY ACTIONS (Security, Auth, Payments, Data Loss)
high_actions = [
    "process credit card payments securely via payment gateway",
    "reset forgotten password using multi-factor authentication SMS code",
    "revoke user permissions and restrict access to admin panel",
    "encrypt sensitive personal health and financial billing records",
    "process high-value online bank transfer transactions",
    "authenticate using OAuth2 single sign-on token",
    "delete user account and permanently erase associated PII data",
    "update credit card tokenization and security CVV verification",
    "audit security logs and flag suspicious IP login attempts",
    "authorize API keys for external service integrations"
]

# MEDIUM PRIORITY ACTIONS (Core Features, Functional Logic, Search, Filters)
medium_actions = [
    "filter search results by category, rating, and price range",
    "update profile avatar picture and personal account biography",
    "receive automated email notifications upon order dispatch",
    "export user transaction logs to downloadable CSV and PDF files",
    "add multiple items to shopping cart and calculate tax",
    "view past order history and download invoice receipts",
    "save favorite search queries to personal dashboard",
    "post comments and star ratings on product review pages",
    "schedule automated daily data backup reports",
    "apply promotional discount coupon codes during checkout"
]

# LOW PRIORITY ACTIONS (UI/UX, Styling, Formatting, Alignment)
low_actions = [
    "align footer copyright text in the horizontal center",
    "change submit button hover color to navy blue",
    "display company logo at top left corner of navbar",
    "show tooltip popups when hovering over dashboard icons",
    "adjust font size and line spacing on privacy policy terms",
    "toggle dark mode appearance theme on settings page",
    "reorder table columns on administrative grid view",
    "add subtle fade animation effect when opening modal windows",
    "update copyright year text in footer section",
    "customize background color scheme of side navigation drawer"
]

dataset = []

# Generate exactly 700 samples per class = 2,100 total balanced samples
random.seed(42)
for _ in range(700):
    role = random.choice(roles)
    dataset.append({"user_story": f"As a {role}, I want to {random.choice(high_actions)}.", "priority": "High"})
    dataset.append({"user_story": f"As a {role}, I want to {random.choice(medium_actions)}.", "priority": "Medium"})
    dataset.append({"user_story": f"As a {role}, I want to {random.choice(low_actions)}.", "priority": "Low"})

df = pd.DataFrame(dataset)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle data

print(f"✅ Dataset Successfully Generated: {len(df)} Total Requirements!")
print("\nClass Distribution:")
print(df['priority'].value_counts())

# -------------------------------------------------------------
# 2. TRAIN HIGH-ACCURACY CLASSIFIER
# -------------------------------------------------------------
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df['user_story'], df['priority'], test_size=0.2, random_state=42, stratify=df['priority']
)

# TF-IDF Feature Extraction
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
X_train = vectorizer.fit_transform(X_train_text).toarray()
X_test = vectorizer.transform(X_test_text).toarray()

# Model Initialization
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# -------------------------------------------------------------
# 3. EVALUATE MODEL ACCURACY
# -------------------------------------------------------------
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("\n==========================================")
print(f"🚀 Final Model Accuracy: {acc * 100:.2f}%")
print("==========================================\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

# -------------------------------------------------------------
# 4. EXPORT MODEL ARTIFACTS FOR STREAMLIT
# -------------------------------------------------------------
joblib.dump(model, 'priority_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("\n✅ Saved 'priority_model.pkl' & 'tfidf_vectorizer.pkl' ready for download!")
