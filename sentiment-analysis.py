from transformers import pipeline
import torch
import gradio as gr
import pandas as pd

pipe = pipeline("text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

# print(pipe("I love this movie!"))


def analyze_sentiment(text):
    result = pipe(text)[0]
    label = result['label']
    score = result['score']
    return f"Sentiment: {label}, Confidence: {score:.2f}"




def analyze_reviews(excel_file_path):
    """
    Reads Excel file and performs sentiment analysis
    on selected number of reviews
    """

    # Read Excel file
    df = pd.read_excel(excel_file_path)

    # Display columns in the file
    print("\nColumns in Excel File:")
    print(df.columns.tolist())

    # Enter review column name
    review_column = input("\nEnter the review column name: ")

    if review_column not in df.columns:
        raise ValueError(f"Column '{review_column}' not found in Excel file")

    # Ask user how many rows to analyze
    num_values = int(input("\nHow many reviews do you want to analyze? "))

    # Take only required rows
    df_subset = df.head(num_values).copy()

    # Perform sentiment analysis
    df_subset["Sentiment"] = df_subset[review_column].astype(str).apply(analyze_sentiment)

    return df_subset


# Main Program
if __name__ == "__main__":

    # Excel file path
    file_path = "twitter_training.xlsx"

    # Analyze reviews
    result_df = analyze_reviews(file_path)

    # Print result
    print("\nSentiment Analysis Result:\n")
    print(result_df)

    # Save output to new Excel file
    output_file = "sentiment_output.xlsx"
    result_df.to_excel(output_file, index=False)

    print(f"\nOutput saved to: {output_file}")


# iface = gr.Interface(fn=analyze_sentiment, 
#                      inputs=gr.Textbox(label="Input Text",lines=4), 
#                      outputs=gr.Textbox(label="Sentiment Analysis Result",lines=4), 
#                      title="Sentiment Analysis",
#                      description="Enter a sentence to analyze its sentiment."
#                      )
# iface.launch()


