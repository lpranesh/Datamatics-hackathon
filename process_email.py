from transformers import pipeline
from fpdf import FPDF
import os

# Hugging Face Pipelines
summarizer = pipeline("summarization", model="t5-small")
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def summarize_text(text, max_length=130, min_length=30):
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

def analyze_sentiment(text):
    sentiment = sentiment_analyzer(text)
    sentiment_score = sentiment[0]['score']
    sentiment_label = sentiment[0]['label']
    return sentiment_label, sentiment_score

def save_pdf(subject, summary, sentiment, sentiment_score, output_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Email Analysis Report", ln=True, align='C')

    pdf.ln(10)
    pdf.cell(0, 10, f"Subject: {subject}", ln=True)
    pdf.ln(10)
    pdf.multi_cell(0, 10, f"Summary: {summary}")
    pdf.ln(10)
    pdf.cell(0, 10, f"Sentiment: {sentiment}", ln=True)
    pdf.cell(0, 10, f"Sentiment Score: {sentiment_score:.2f}", ln=True)

    pdf.output(output_path)
    print(f"PDF saved to: {output_path}")


def process_email(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_folder, filename)
            with open(file_path, 'r') as file:
                content = file.read()
            
            # Extract subject from file name
            subject = os.path.splitext(filename)[0]

            # Summarize and analyze sentiment
            summary = summarize_text(content)
            sentiment, sentiment_score = analyze_sentiment(content)

            # Save results to PDF
            output_path = os.path.join(output_folder, f"{subject}_processed.pdf")
            save_pdf(subject, summary, sentiment, sentiment_score, output_path)


if __name__ == "__main__":
    input_folder = "C:\\trucap\project_email_analysis\input_emails"
    output_folder ="C:\\trucap\project_email_analysis\processed_emails"
    os.makedirs(output_folder, exist_ok=True)
    process_email(input_folder, output_folder)
