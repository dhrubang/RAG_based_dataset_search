# RAG-Based Dataset Discovery Chatbot 🤖📊

A Retrieval Augmented Generation (RAG) chatbot designed to facilitate intelligent dataset discovery through natural language queries.

## 📋 Project Overview

This project develops an advanced chatbot that leverages web scraping, exploratory data analysis, and RAG architecture to provide context-aware dataset recommendations. Built between May 2025 - June 2025, it combines modern NLP techniques with efficient vector search to help users discover relevant datasets effortlessly.

## ✨ Features

- **🔍 Dataset Scraping**: Extracted metadata from 2,400+ datasets using API calls, Selenium, and BeautifulSoup
- **📈 Exploratory Data Analysis (EDA)**: Applied Latent Dirichlet Allocation (LDA), fuzzy matching, and sentiment analysis to extract meaningful labels and categories
- **🧠 RAG Pipeline**: Built using FAISS for efficient vector search and LLaMA-3-70B for generating context-aware responses
- **💬 Interactive Chatbot**: Flask-based web interface for natural language dataset discovery

## 🛠️ Technologies Used

### Programming & Web Development
- **Python** - Core development language
- **Flask** - Web framework for chatbot interface

### Data Collection & Processing
- **Selenium** - Web automation and scraping
- **BeautifulSoup** - HTML parsing and data extraction
- **API requests** - Direct data source integration

### Machine Learning & NLP
- **FAISS** - Efficient similarity search and vector indexing
- **LLaMA-3-70B** - Large language model for response generation
- **Latent Dirichlet Allocation (LDA)** - Topic modeling
- **Fuzzy matching** - Text similarity and matching
- **Sentiment analysis** - Content categorization

### Supporting Libraries
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **scikit-learn** - Machine learning utilities
- **NLTK/Spacy** - Natural language processing tasks

## 💡 Example Usage

**User Query:**
```
"Show me datasets about renewable energy."
```

**Chatbot Response:**
```
Here are some datasets related to renewable energy:

🌞 Solar Energy Production (2023)
   • Metadata: Positive sentiment
   • Category: Renewable Energy
   • Source: [Dataset Link]

🌪️ Wind Turbine Efficiency
   • Metadata: Neutral sentiment  
   • Category: Renewable Energy
   • Source: [Dataset Link]
```


## 🔮 Future Improvements

- [ ] **Expand Data Sources**: Include Kaggle, UCI, and other major repositories
- [ ] **Multi-modal Support**: Add dataset previews and visualization capabilities
- [ ] **Scalability**: Optimize FAISS index for handling larger dataset collections
- [ ] **User Features**: Implement authentication and query logging
- [ ] **Advanced Analytics**: Add dataset quality scoring and recommendation refinement

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License 


---

⭐ **Star this repository if you find it helpful!**
