import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def load_data():
    mail_data = pd.read_csv("mail_data.csv")
    mail_data = mail_data.fillna('')
    return mail_data

def preprocess_data(mail_data):
    mail_data = mail_data.replace("spam", 0)
    mail_data = mail_data.replace("ham", 1)
    return mail_data

def train_model(X_train, Y_train):
    vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
    X_train_features = vectorizer.fit_transform(X_train)
    model = LogisticRegression()
    model.fit(X_train_features, Y_train)
    return vectorizer, model

def predict_email(vectorizer, model, email):
    email_features = vectorizer.transform([email])
    prediction = model.predict(email_features)
    return prediction

def main():
    st.title("Spam Detector")

    # Load data
    mail_data = load_data()

    # Preprocess data
    mail_data = preprocess_data(mail_data)

    # Split data
    X = mail_data['Message']
    Y = mail_data['Category']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=4)

    # Train model
    vectorizer, model = train_model(X_train, Y_train)

    # Evaluate model
    Y_predictions = model.predict(vectorizer.transform(X_test))
    accuracy = accuracy_score(Y_test, Y_predictions)

    # Predict new email
    new_email = st.text_area("Enter a new email:")
    if st.button("Predict"):
        prediction = predict_email(vectorizer, model, new_email)
        outcome = 'Spam' if prediction[0] == 0 else 'Ham'
        st.write(f"Prediction: {outcome}")
        #st.write(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    main()
