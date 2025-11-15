from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
import sqlite3
import os
import torch
import chromadb
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# --- RAG Imports ---
from langchain_chroma import Chroma # NEW: Using dedicated chroma package
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFacePipeline # NEW: Consolidated imports


# Optional: Set a cache directory for Hugging Face models
os.environ['HF_HOME'] = 'D:/huggingface_cache'

# Check for CUDA availability and set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set to use {device}")

# 1. Load the Hugging Face model and tokenizer
model_id = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

# 2. Create a Hugging Face text-generation pipeline (Fixed for stability)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    device=device,
    return_full_text=False, # Only return the generated text
    do_sample=True,
    temperature=0.7,
    top_p=0.95
)

# 3. Wrap the Hugging Face pipeline with LangChain components
llm_pipeline = HuggingFacePipeline(pipeline=pipe)
llm = ChatHuggingFace(llm=llm_pipeline)

# --- CHROMA DB SETUP (for RAG) ---
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

CHROMA_PATH = "./chroma_db_rag"
vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

if vectorstore.get()['ids'] == []: # Check if collection is empty
    print("\n--- Initializing ChromaDB with dummy data ---")
    vectorstore.add_texts(
        texts=["The official name of the company is Acme Corp, and its founder is Jane Doe.",
               "All employees must complete the annual compliance training by December 31st."],
        metadatas=[{"source": "company_info"}, {"source": "hr_policy"}]
    )
    vectorstore.persist() 
    print(f"ChromaDB initialized with {vectorstore.get()['count']} documents.")

retriever = vectorstore.as_retriever()
# -----------------------------------

# --- LangGraph State Definition ---
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    context: str

# --- Graph Nodes ---

def retrieve_node(state: ChatState):
    """Retrieves relevant context from ChromaDB based on the last user message."""
    user_query = state['messages'][-1].content
    
    # CRITICAL FIX: Use the standard LCEL .invoke() method
    docs = retriever.invoke(user_query)
    
    context_text = "\n\n".join([doc.page_content for doc in docs])
    
    return {"context": context_text}

def chat_node(state: ChatState):
    """Invokes the chat model using the RAG chain structure."""
    messages = state['messages']
    context = state['context']
    
    # 1. Define the RAG Prompt Template
    rag_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", 
             "You are a helpful assistant. Answer the user's question ONLY using the provided context. If the context does not contain the answer, you MUST respond: 'I cannot find the answer based on the provided context.'\n\nCONTEXT:\n{context}"),
            *messages # Pass the entire conversation history
        ]
    )
    
    # 2. Define the RAG Chain
    rag_chain = rag_prompt | llm 
    
    # 3. Invoke the chain
    response_message = rag_chain.invoke(
        {"context": context, "messages": messages}
    )
    
    return {"messages": [response_message]}


# --- LangGraph Setup ---
conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("chat", chat_node)
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "chat")
graph.add_edge("chat", END)

chatbot = graph.compile(checkpointer=checkpointer)

# database.py (FIXED Code)
def retrieve_all_threads():
    all_threads = set()
    for item in checkpointer.list(None): 
        # item might be a tuple like (config, ...) or the config dict itself.
        # Check if item is a tuple, and if so, take the first element (index 0).
        checkpoint = item[0] if isinstance(item, tuple) and len(item) > 0 else item
        
        # Now access the dictionary keys
        try:
            thread_id = checkpoint['config']['configurable']['thread_id']
            all_threads.add(thread_id)
        except (KeyError, TypeError) as e:
            # Skip items that don't have the expected dictionary structure
            print(f"Skipping malformed checkpoint item: {e}") 
            continue
            
    return list(all_threads)

if __name__ == "__main__":
    print("\n--- Example RAG Chatbot Run with ChromaDB ---")
    
    # Query that relies on RAG
    inputs_rag = {"messages": [HumanMessage(content="Who is the founder of Acme Corp?")]}
    
    for s in chatbot.stream(inputs_rag, config={"configurable": {"thread_id": "rag_test_1"}}):
        if s.get("chat"):
            print(f"AI Response Chunk: {s['chat']['messages'][0].content}")