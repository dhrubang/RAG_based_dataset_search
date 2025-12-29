# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import faiss
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama   # ← This is correct now
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

app = Flask(__name__)

# ==================== CONFIGURATION ====================
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "dataset_indexq.faiss"
METADATA_CSV = "metadata_with_embeddings1.csv"
OLLAMA_MODEL = "gemma3:1b"  # Change to your model e.g. "llama3", "phi3", "gemma3:1b"

# ==================== LOAD DATA ====================
df = pd.read_csv(METADATA_CSV)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
index = faiss.read_index(FAISS_INDEX_PATH)

# ==================== LLM (Local Ollama) ====================
llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.7)

# ==================== MEMORY (Simple list-based) ====================
# LangChain memory can be tricky; we'll use a simple list for history
chat_history = []

# ==================== RETRIEVAL + FORMATTING ====================
def retrieve_and_format(query: str, top_k=5):
    query_emb = np.array([embeddings.embed_query(query)])
    distances, indices = index.search(query_emb.astype('float32'), top_k)
    
    top_results = df.iloc[indices[0]].copy()
    
    dataset_blocks = []
    for _, row in top_results.iterrows():
        block = f"""
📁 **Dataset Name:** {row.get('dataset_name', 'Unknown')}
🔗 **URL:** {row.get('dataset_url', 'N/A')}
🗂️ **Category:** {row.get('category2', 'Unknown')}
📝 **Description:** {row.get('about_dataset', 'No description')}
"""
        dataset_blocks.append(block)
    
    context = "\n".join(dataset_blocks)
    return context

# ==================== PROMPTS ====================
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a friendly data science assistant helping users discover datasets.
Use the retrieved datasets below to answer enthusiastically.

Retrieved Datasets:
{context}

Provide:
- Dataset names and URLs
- Simple summaries
- EDA suggestions
- Project/ML ideas
- Basic plot ideas"""),
    ("placeholder", "{chat_history}"),
    ("human", "{input}")
])

prediction_prompt = ChatPromptTemplate.from_messages([
    ("system", """Based on the conversation, suggest 2-3 natural follow-up questions the user might ask next.
Keep them concise and numbered."""),
    ("placeholder", "{chat_history}"),
    ("human", "Latest query: {input}\nLatest response: {response}")
])

# ==================== CHAINS (Modern LCEL) ====================
rag_chain = rag_prompt | llm
prediction_chain = prediction_prompt | llm

# ==================== FLASK ROUTES ====================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    global chat_history
    user_query = request.json['query']
    is_new_chat = request.json.get('new_chat', False)

    if is_new_chat:
        chat_history = []
        return jsonify({'response': 'New chat started! Hello! What datasets are you looking for?'})

    # Retrieve context
    context = retrieve_and_format(user_query)

    # Add user message to history
    chat_history.append(HumanMessage(content=user_query))

    # Generate response
    response_ai = rag_chain.invoke({
        "context": context,
        "input": user_query,
        "chat_history": chat_history
    })
    response = response_ai.content

    # Add AI response to history
    chat_history.append(AIMessage(content=response))

    # Predict next queries
    pred_ai = prediction_chain.invoke({
        "input": user_query,
        "response": response,
        "chat_history": chat_history
    })
    predictions = pred_ai.content

    full_response = f"{response}\n\n**Suggested Next Queries:**\n{predictions}"

    return jsonify({'response': full_response})

if __name__ == '__main__':
    app.run(debug=True)