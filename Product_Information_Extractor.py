import streamlit as st
import json
import csv
import re
import io
import PyPDF2
import pandas as pd
from langchain_openai import ChatOpenAI

# Function to extract text from a PDF
def extract_text_from_pdf(uploaded_file):
    """Extract text from an uploaded PDF file while handling errors."""
    text = ""
    try:
        pdf_stream = io.BytesIO(uploaded_file.getvalue())  # Convert uploaded file to byte stream
        reader = PyPDF2.PdfReader(pdf_stream)

        for page in reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"

    except PyPDF2.errors.PdfReadError:
        st.error("Error: Unable to read the PDF file. It may be corrupted or not a valid format.")
        return None

    except Exception as e:
        st.error(f"Unexpected Error: {e}")
        return None

    return text

# Function to send extracted text to DeepSeek for JSON response
def send_to_llm(extracted_text):
    """Send extracted text to DeepSeek and get structured JSON response."""
    llm = ChatOpenAI(
        model="deepseek-r1-32b",
        api_key="-----",  # Replace with actual API key
        base_url="------",
    )

    prompt = f"""Extract details for each product mentioned in the given text and structure them into **valid JSON** format.

    Each product should be stored in a **list**, where each entry represents a unique product with:
    - 'product_name'
    - 'product_attributes' (dictionary with key features like type, operating system, RAM, storage, etc.)
    - 'features' (list)
    - 'specifications' (dictionary with display details, battery life, dimensions, charging methods, and rating)
    - 'review_summary' (A brief summary of overall product reviews, highlighting strengths and weaknesses, but **NOT including individual customer reviews**)

    Given pdf_text:
    {extracted_text}

    **Important Instructions:**
    1. Multiple products must be **stored in a list**, ensuring each product appears in a separate JSON entry.
    2. Ensure the response is a **valid JSON array** starting with '[' and ending with ']'.
    3. The 'review_summary' field must provide an overall sentiment analysis of the product's reviews, **excluding individual customer reviews**.
    4. **Do NOT include explanations, comments, or any extra text** outside of the JSON structure.
    """

    response = llm.invoke(prompt)
    response_text = response.content.strip()

    # Extract JSON from response text
    match = re.search(r"\[.*\]", response_text, re.DOTALL)
    if match:
        response_text = match.group(0)
    else:
        return "[]"

    try:
        json.loads(response_text)  # Validate JSON format
        print(response_text)  # Debugging output
        return response_text
    except json.JSONDecodeError:
        return "[]"

# Function to flatten nested JSON
def flatten_json(nested_json, prefix=""):
    """Flatten nested JSON into a single-level dictionary."""
    flattened_data = {}
    for key, value in nested_json.items():
        new_key = f"{prefix}{key}" if prefix else key

        if isinstance(value, dict):
            flattened_data.update(flatten_json(value, prefix=new_key + "_"))
        elif isinstance(value, list):
            if all(isinstance(i, dict) for i in value):  # Handle list of dictionaries separately
                for i, item in enumerate(value):
                    flattened_data.update(flatten_json(item, prefix=new_key + f"_{i}_"))
            else:
                flattened_data[new_key] = ", ".join(map(str, value))  # Convert list to comma-separated string
        else:
            flattened_data[new_key] = value

    return flattened_data

# Function to convert JSON text to CSV automatically
def json_to_csv(llm_response_json, csv_filename):
    """Convert JSON text to CSV ensuring multiple products are displayed row-wise."""
    try:
        data = json.loads(llm_response_json)  # Parse JSON text

        if isinstance(data, dict):
            data = [data]  # Convert dictionary into a list

        if not data or not isinstance(data, list):
            return None

        # Flatten each product entry for separate rows in CSV
        flattened_data = [flatten_json(item) for item in data]

        # Extract column names from the first product entry instead of dynamically from the dataset
        all_keys = sorted(flattened_data[0].keys()) if flattened_data else []

        # Write CSV file ensuring columns appear in expected order
        with open(csv_filename, mode="w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(flattened_data)

        return csv_filename

    except json.JSONDecodeError:
        return None

# Streamlit UI
st.title("Product Information Extractor (PDF2CSV) - Powered by DeepSeek")
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    st.write("Processing the uploaded file...")

    # Extract text from PDF
    extracted_text = extract_text_from_pdf(uploaded_file)

    if extracted_text:
        # Get JSON response from DeepSeek
        llm_response_json = send_to_llm(extracted_text)

        # Generate CSV
        csv_filename = "output.csv"
        csv_filepath = json_to_csv(llm_response_json, csv_filename)

        if csv_filepath:
            st.success("CSV file generated successfully!")

            # Read CSV using Pandas for better display
            df = pd.read_csv(csv_filename)

            # Display CSV data in table format
            st.dataframe(df)

            # Provide CSV download link
            with open(csv_filename, "rb") as file:
                st.download_button(label="Download CSV", data=file, file_name="output.csv", mime="text/csv")
        else:
            st.error("Failed to generate CSV. Please try again.")