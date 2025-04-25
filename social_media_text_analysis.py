# social_media_text_analysis.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# --------- PART 1: Sentiment Analysis ---------

def sentiment_analysis():
    print("Running Sentiment Analysis...")

    # Sample data
    data = {
        'text': [
            "I love this product!", 
            "This is the worst experience ever", 
            "Amazing quality and great support", 
            "I will never buy from here again",
            "Totally worth the money", 
            "Very disappointing"
        ],
        'sentiment': ['positive', 'negative', 'positive', 'negative', 'positive', 'negative']
    }

    df = pd.DataFrame(data)

    # Vectorization
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['text'])
    y = df['sentiment']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\nSentiment Analysis Results:")
    print(classification_report(y_test, predictions))
    print("Accuracy:", accuracy_score(y_test, predictions))


# --------- PART 2: Spam Email Detection ---------

def spam_email_detection():
    print("\nRunning Spam Email Detection...")

    # Sample dataset
    data = {
        'email': [
            "Congratulations! You've won a $1000 Walmart gift card. Click here to claim now.",
            "Hey, are we still meeting for lunch tomorrow?",
            "This is not spam. Let's catch up soon!",
            "Urgent! Your account has been compromised. Act now!",
            "Win a brand new iPhone. Participate now!",
            "Can you please send me the files by today?"
        ],
        'label': ['spam', 'ham', 'ham', 'spam', 'spam', 'ham']
    }

    df = pd.DataFrame(data)

    # Vectorization
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['email'])
    y = df['label']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Model
    model = MultinomialNB()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\nSpam Detection Results:")
    print(classification_report(y_test, predictions))
    print("Accuracy:", accuracy_score(y_test, predictions))


# --------- MAIN ---------
if __name__ == "__main__":
    sentiment_analysis()
    spam_email_detection()
