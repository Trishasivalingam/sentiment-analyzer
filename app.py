import joblib
import pandas as pd
import streamlit as st
from utils import clean_text

st.set_page_config(page_title="Review Sentiment Analyzer", page_icon="💬")

@st.cache_resource
def load():
    return joblib.load("models/model.pkl"), joblib.load("models/tfidf.pkl")

model, tfidf = load()

def predict(texts):
    X = tfidf.transform([clean_text(t) for t in texts])
    return model.predict(X), model.predict_proba(X)

st.title("💬 Review Sentiment Analyzer")
st.write("Check whether customer reviews are positive, negative or neutral.")

tab1, tab2 = st.tabs(["Single review", "Upload CSV"])

with tab1:
    text = st.text_area("Type or paste a review", height=150)
    if st.button("Analyze") and text.strip():
        label, proba = predict([text])
        st.subheader(f"Sentiment: {label[0].upper()}")
        st.bar_chart(pd.Series(proba[0], index=model.classes_))

with tab2:
    file = st.file_uploader("Upload a CSV with reviews", type="csv")
    if file:
        data = pd.read_csv(file)
        col = st.selectbox("Select the review column", data.columns)
        if st.button("Analyze all"):
            labels, _ = predict(data[col].astype(str).tolist())
            data["sentiment"] = labels
            st.bar_chart(data["sentiment"].value_counts())
            st.dataframe(data)
            st.download_button("Download results",
                               data.to_csv(index=False), "results.csv")