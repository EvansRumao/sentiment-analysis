from transformers import pipeline
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------------
# Load Sentiment Model
# -----------------------------
@st.cache_resource
def load_model():
    pipe = pipeline(
        "text-classification",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    )
    return pipe

pipe = load_model()

# -----------------------------
# Sentiment Function
# -----------------------------
def analyze_sentiment(text):
    result = pipe(text)[0]

    return {
        "Label": result['label'],
        "Confidence": round(result['score'], 2)
    }

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Sentiment Analysis App")
st.write("Upload an Excel file and analyze review sentiments using HuggingFace Transformers.")

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Excel File",
    type=["xlsx", "xls"]
)

if uploaded_file:

    # Read Excel File
    df = pd.read_excel(uploaded_file)

    st.subheader("📄 Uploaded Dataset")
    st.dataframe(df.head())

    # Select Review Column
    review_column = st.selectbox(
        "Select Review Column",
        df.columns
    )

    # Number of rows
    num_reviews = st.slider(
        "Number of Reviews to Analyze",
        min_value=1,
        max_value=len(df),
        value=min(10, len(df))
    )

    # Analyze Button
    if st.button("Analyze Sentiment"):

        with st.spinner("Analyzing sentiments..."):

            # Subset Data
            df_subset = df.head(num_reviews).copy()

            sentiments = []
            confidences = []

            # Analyze
            for review in df_subset[review_column].astype(str):

                result = analyze_sentiment(review)

                sentiments.append(result["Label"])
                confidences.append(result["Confidence"])

            # Add Results
            df_subset["Sentiment"] = sentiments
            df_subset["Confidence"] = confidences

            st.success("Analysis Completed!")

            # Display Results
            st.subheader("📊 Sentiment Results")
            st.dataframe(df_subset)

            # -----------------------------
            # Sentiment Distribution
            # -----------------------------
            st.subheader("📈 Sentiment Distribution")

            sentiment_counts = df_subset["Sentiment"].value_counts()

            fig, ax = plt.subplots()
            ax.pie(
                sentiment_counts,
                labels=sentiment_counts.index,
                autopct='%1.1f%%'
            )

            st.pyplot(fig)

            # -----------------------------
            # Download Button
            # -----------------------------
            output_file = "sentiment_output.xlsx"

            df_subset.to_excel(output_file, index=False)

            with open(output_file, "rb") as file:
                st.download_button(
                    label="⬇ Download Results",
                    data=file,
                    file_name="sentiment_output.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )