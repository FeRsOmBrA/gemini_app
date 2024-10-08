# Gemini App for Key Information Extraction

This Streamlit application leverages the power of Google Cloud's Gemini Pro model to extract key information from text. Users can input text and receive a concise summary, along with the most important keywords.

## Features

* **Text Summarization:** Powered by Gemini Pro, the app generates a brief and professional summary of the input text.
* **Keyword Extraction:** Identifies and presents the most relevant keywords from the text.
* **User-Friendly Interface:** Simple and intuitive design makes it easy to use.

## How to Use

1. **Input Text:** Paste or type your text into the provided text area.
2. **Generate Response:** Click the "Generate Response" button.
3. **View Results:** The summarized text and extracted keywords will be displayed below.

## Technical Details

* **Model:** Google Cloud's Gemini Pro
* **Framework:** Streamlit
* **Language:** Python
* **Libraries:** `google-cloud-aiplatform`, `PyMuPDF`, `streamlit`

## Requirements

To run this application locally, you need to install the required libraries:

```bash
pip install google-cloud-aiplatform PyMuPDF streamlit
```

## Setup and Configuration

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/gemini_app.git
   ```
2. **Set up Google Cloud Credentials:** Follow the instructions in the `main.py` file to configure your Google Cloud credentials.
3. **Run the App:**
   ```bash
   streamlit run main.py
   ```

## Disclaimer

This application is a demonstration of the capabilities of Google Cloud's Gemini Pro model. While it strives to provide accurate and helpful information, it should not be considered a substitute for professional advice or judgment.
