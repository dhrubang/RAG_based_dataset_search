# Dataset RAG Chatbot

## Description
Flask web application with RAG pipeline for dataset discovery and analysis. Uses FAISS for semantic search and LLM (Llama-3-70b-chat) for generating insights.

## Features
- Dataset search via semantic similarity
- Context-aware conversation with memory
- Automatic dataset analysis suggestions
- Conversation history management

## Setup
1. Install requirements: `pip install -r requirements.txt`
2. Place these files in root directory:
   - dataset_index.faiss (FAISS index)
   - metadata_with_embeddings.csv (dataset metadata)
3. Set Together.ai API key in app.py
4. Run: `python app.py`

## Usage
1. Enter dataset query (e.g. "climate change data")
2. Ask follow-up questions
3. Click "New Chat" to restart

## Files
- app.py - Main application
- temporary.txt - Conversation history
- templates/index.html - Chat interface
- static/style.css - Stylesheet

## Requirements
Python 3.8+, 4GB RAM