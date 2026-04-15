import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample training data
texts = [
    "Win money now",
    "Claim your prize",
    "Limited offer just for you",
    "Hello how are you",
    "Let's meet tomorrow",
    "Are you coming to class",
    "Free entry in a contest",
    "Call me when you can"
]

labels = [1, 1, 1, 0, 0, 0, 1, 0]  # 1 = Spam, 0 = Not Spam

# Train model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

model = MultinomialNB()
model.fit(X, labels)

# Streamlit UI
st.title("Simple Spam Detection App")

user_input = st.text_area("Enter a message")

if st.button("Predict"):
    if user_input.strip() != "":
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)

        if prediction[0] == 1:
            st.error("Spam Message")
        else:
            st.success("Not Spam")
    else:
        st.warning("Please enter some text")