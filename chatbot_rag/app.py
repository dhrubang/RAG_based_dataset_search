from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
import requests
import json

app = Flask(__name__)

# Initialize model and index
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index('dataset_index.faiss')
df = pd.read_csv('metadata_with_embeddings.csv')

# Together.ai API configuration
API_KEY = "tgp_v1_k6Diuqimve112DYRKb2c2JPyLm8BsKELAlzVNbFLqNE"
API_URL = "https://api.together.xyz/v1/chat/completions"

TEMP_FILE = "temporary.txt"

def retrieve_datasets(query, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    results = df.iloc[indices[0]].copy()
    results['similarity_score'] = 1 - distances[0]
    return results[['dataset_name', 'header', 'about_dataset', 'dataset_url', 'category2', 'similarity_score']]

def eda_and_prompt(top_results):
    if 'desc_length' not in top_results.columns:
        top_results['desc_length'] = top_results['about_dataset'].apply(lambda x: len(str(x)))

    top_categories = top_results['category2'].value_counts()
    avg_similarity = top_results['similarity_score'].mean()
    avg_desc_len = top_results['desc_length'].mean()

    datasets = top_results[['dataset_name', 'header', 'about_dataset', 'category2', 'dataset_url']].head(5)

    dataset_blocks = []
    for i, row in datasets.iterrows():
        block = f"""
📁 Dataset Name: {row['dataset_name']}
📌 Header Summary: {row['header']}
🗂️ Category: {row['category2']}
🔗 Link: {row['dataset_url']}
📝 Description: {row['about_dataset']}
"""
        dataset_blocks.append(block)

    combined_dataset_info = '\n'.join(dataset_blocks)

    insights = f"""
📊 EDA Insights:
- Most common dataset categories: {', '.join(top_categories.index[:3])}
- Average similarity score across datasets: {avg_similarity:.3f}
- Average description length: {avg_desc_len:.0f} characters

🧠 Prompt to LLM:
Below are multiple datasets retrieved from a search. Each includes its name, description, category, and link.

{combined_dataset_info}

Please help with the following:
1. Return the dataset names, url to the dataset and information about the dataset
2. Summarize what each dataset is about in simpler terms.
3. Suggest possible EDA steps or interesting patterns that could be explored based on the description.
4. Recommend project ideas or machine learning use cases aligned with each dataset.
5. If possible, suggest or generate basic plots that could be used for initial EDA using only the description above.

Your responses should reflect the unique nature of each dataset and the category it's part of.
"""
    return insights

def generate_response_with_together(prompt):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "meta-llama/Llama-3-70b-chat-hf",
        "messages": [
            {"role": "system", "content": "You are a helpful data science assistant."},
            {"role": "user", "content": str(prompt)}
        ],
        "temperature": 0.7,
        "max_tokens": 1024
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    except requests.exceptions.RequestException as e:
        return f"Error generating response: {e}"

def summarize_previous_chat(content):
    prompt = f"""
    Summarize the following conversation in about 300 words, focusing on the key points:
    
    {content}
    
    Summary:
    """
    return generate_response_with_together(prompt)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_query = request.json['query']
    is_new_chat = request.json.get('new_chat', False)
    
    if is_new_chat:
        # Clear temporary file for new chat
        open(TEMP_FILE, 'w').close()
    
    if os.path.exists(TEMP_FILE) and os.path.getsize(TEMP_FILE) > 0 and not is_new_chat:
        # Subsequent queries
        with open(TEMP_FILE, 'r') as f:
            previous_content = f.read()
        
        summary = summarize_previous_chat(previous_content)
        enhanced_query = f"{previous_content}\n\nSummary of previous conversation:\n{summary}\n\nTake reference from the words earlier and only answer the question written: {user_query}"
        
        # Generate response
        response = generate_response_with_together(enhanced_query)
        
        # Append to temporary file
        with open(TEMP_FILE, 'a') as f:
            f.write(f"\nUser: {user_query}\nAssistant: {response}\n")
    else:
        # First query - dataset retrieval flow
        top_results = retrieve_datasets(user_query)
        llm_query = eda_and_prompt(top_results)
        response = generate_response_with_together(llm_query)
        
        # Store in temporary file
        with open(TEMP_FILE, 'w') as f:
            f.write(f"Initial dataset query: {user_query}\nAssistant: {response}\n")
    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)