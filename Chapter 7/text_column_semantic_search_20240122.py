# Run the app by opening a terminal then executing: `streamlit run text_column_semantic_search_20240122.py`

import streamlit as st
import pandas as pd
import openai
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from io import BytesIO
import time


# Function to convert DataFrame to Excel
def convert_df_to_excel(df):
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    return excel_buffer.getvalue()

# Function to calculate similarity
def calculate_similarity(api_key, persona, text_data, progress_bar):
    # Authenticate with OpenAI
    openai.api_key = api_key

    # Function to get embeddings with error handling
    def get_embeddings(text_batch):
        while True:
            try:
                response = openai.Embedding.create(
                    input=text_batch, 
                    engine="text-embedding-ada-002"
                )
                return [np.array(item['embedding']) for item in response['data']]
            except Exception as e:
                if 'rate limit' in str(e).lower():
                    st.warning("Rate limit reached, waiting for 60 seconds.")
                    time.sleep(60)
                else:
                    raise e

    # Get embedding for persona
    persona_embedding = get_embeddings([persona])[0]

    # Initialize list for similarity scores
    similarity_scores = []

    # Process in batches
    batch_size = 100  # Reduced batch size
    for i in range(0, len(text_data), batch_size):
        batch = text_data[i:i + batch_size]
        text_embeddings = get_embeddings(batch)
        for text_embedding in text_embeddings:
            similarity = cosine_similarity([persona_embedding], [text_embedding])[0][0]
            similarity_scores.append(similarity)

        # Update progress bar
        progress_bar.progress(min((i + batch_size) / len(text_data), 1.0))

    return similarity_scores

# Streamlit app layout
st.title("Semantic Search App")

# Sidebar inputs
api_key = st.sidebar.text_input("OpenAI API Key", type="password")
persona_description = st.sidebar.text_area("Persona Description")
similarity_threshold = st.sidebar.number_input("Similarity Score Threshold", min_value=0.0, max_value=1.0, value=0.7)
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx'])

if st.sidebar.button('Run') and uploaded_file:
    # Load data
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)

    # Check if 'text' column exists
    if 'text' in df.columns:
        df['text'] = df['text'].apply(lambda x: x[:500] if isinstance(x, str) else x)

        # Progress bar
        progress_bar = st.progress(0)

        # Calculate similarity
        try:
            scores = calculate_similarity(api_key, persona_description, df['text'].tolist(), progress_bar)
            df['Score'] = scores
            df = df.sort_values(by='Score', ascending=False)
            df_filtered = df[df['Score'] >= similarity_threshold]

            # Display dataframe
            st.write(df_filtered)

            # Export to Excel button
            excel_data = convert_df_to_excel(df_filtered)
            st.sidebar.download_button(
                label="Export to Excel",
                data=excel_data,
                file_name="semantic_search_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.error(f"An error occurred: {e}")

        # Reset progress bar
        progress_bar.empty()
    else:
        st.error("Uploaded file must have a 'text' column.")
