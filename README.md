# LangChain_and_LangGraph_RAG_Chatbot_with_ChromaDB

This project implements a Retrieval-Augmented Generation (RAG) chatbot using LangGraph for orchestration, ChromaDB for vector storage, and a Hugging Face's TinyLlama model for generation. The user interface is built with Streamlit, providing a persistent, interactive chat experience. 

## ✨ Features 
* RAG Architecture: Uses context retrieved from a local vector database (ChromaDB) to ground the language model's responses. 
* Persistent Chat History: Utilizes LangGraph's SqliteSaver to persist conversation history across sessions and threads. 
* Dynamic Thread Management: Allows users to start new chats and revisit old conversations via a Streamlit sidebar. 
* Local LLM Integration: Uses the lightweight TinyLlama-1.1B-Chat-v1.0 model via the Hugging Face pipeline for efficient local execution. 

## ⚙️ Setup and Installation 
Follow these steps to set up and run the project locally. 

### 1. Prerequisites 
Ensure you have Python 3.10 installed. It is highly recommended to use a virtual environment. 

Bash 

#Create a virtual environment 

python -m venv venv 

#Activate the environment (on Windows)

.\venv\Scripts\activate

#Activate the environment (on macOS/Linux) 

source venv/bin/activate 

## 2. Install Dependencies 
Due to conflicts with newer deep learning libraries, specific versions of numpy and torch are required. 

Bash 

Install required packages 

pip install torch 

pip install numpy==1.26.4  # Critical fix for TensorFlow/NumPy version conflicts 

pip install streamlit langgraph langchain-core langchain-huggingface transformers 

pip install langchain-community langchain-chroma sentence-transformers 

## 3. Project Structure 
Ensure your project directory contains the following two main files: 

/your-project-folder 

├── database.py       # Contains LangGraph graph definition, LLM setup, and ChromaDB logic. 

├── frontend.py       # Contains the Streamlit application interface and state management. 

├── chatbot.db        # Created automatically by SqliteSaver for chat history. 

└── chroma_db_rag/    # Created automatically by ChromaDB for vector data. 

## 🚀 How to Run the Application 
The application is launched using Streamlit: 

1. Make sure your virtual environment is active. 

2. Run the application from your project directory: 

Bash 

streamlit run frontend.py 

3. The application will automatically open in your web browser (usually at http://localhost:8501). 

Note: On the first run, the TinyLlama model and the Sentence Transformer embedding model will be downloaded, which may take several minutes depending on your 
connection speed. 

## 💡 Technical Overview 

database.py (Backend) 

* State: The ChatState tracks messages (for conversation history) and context (for RAG data). 

* Nodes: 

1. retrieve_node: Takes the latest user message, invokes the retriever (connected to ChromaDB), and saves the relevant context documents into the state. 

2. chat_node: Takes the accumulated messages and the retrieved context, formats them into a RAG prompt using ChatPromptTemplate, and invokes the TinyLlama model for the final response. 

* Persistence: Uses SqliteSaver to checkpoint the full graph state, enabling persistent conversation threads. 
frontend.py (Streamlit Interface) 

* Stability Fix: The dynamic creation of sidebar buttons for thread selection uses on_click callback functions and unique keys to maintain stable application state across Streamlit reruns. 

* RAG Invocation: The app initiates the RAG flow by streaming the output from the three-node graph (START → retrieve → chat → END). 

## Known Limitations 

* TinyLlama Constraints: The TinyLlama-1.1B model is small and may occasionally produce unhelpful, repetitive, or poorly formatted answers, especially for complex queries or when the context is short. 

* ChromaDB Initialization: The database is initialized with dummy data (Acme Corp information). For a real application, you would need a mechanism to load external documents (e.g., PDFs, TXT files) before runtime.
