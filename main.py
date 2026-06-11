import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ====================================
# 1. Dataset
# ====================================

messages = [
    # Spam
    "Win money now",
    "Claim your free prize",
    "You won a lottery",
    "Free iphone click now",
    "Congratulations you won cash",
    "Limited offer buy now",
    "Earn money quickly",
    "Claim your reward now",

    # Not Spam
    "Meeting starts at five",
    "Please call me later",
    "How are you doing",
    "Lets have lunch together",
    "Send me the report",
    "See you tomorrow",
    "Project submission is today",
    "Happy birthday friend"
]

labels = [
    # Spam = 1
    1,1,1,1,1,1,1,1,
    # Not Spam = 0
    0,0,0,0,0,0,0,0
]

# ====================================
# 2. Create DataFrame
# ====================================

df = pd.DataFrame({
    "message": messages,
    "label": labels
})

print(df.head())

# ====================================
# 3. Text Cleaning Function
# ====================================

def clean_text(text):
    # lowercase
    text = text.lower()
    # remove punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

df["cleaned"] = df["message"].apply(clean_text)

# ====================================
# 4. TF-IDF Vectorization
# ====================================

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df["cleaned"])
y = df["label"]

print("\nTF-IDF Shape:", X.shape)

# ====================================
# 5. Train Test Split
# ====================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ====================================
# 6. Logistic Regression Model
# ====================================

model = LogisticRegression()
model.fit(X_train, y_train)

# ====================================
# 7. Evaluation
# ====================================

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:")
print(round(accuracy * 100, 2), "%")

# ====================================
# 8. Prediction Function
# ====================================

def predict_message(text):
    text = clean_text(text)
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]

    if prediction == 1:
        result = "SPAM"
    else:
        result = "NOT SPAM"

    print("\nMessage:", text)
    print("Prediction:", result)

# ====================================
# 9. Test Predictions
# ====================================

predict_message("win free money now")
predict_message("please send me the report")
predict_message("claim your reward")
predict_message("see you tomorrow")
