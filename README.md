🩺 MediBot: A RAG Medical Chatbot

MediBot is a high-speed, local-first medical chatbot. It uses Retrieval-Augmented Generation (RAG) to answer questions based *only* on the content of your private PDF documents.

This project is built with a modern stack, prioritizing privacy for data ingestion (local embeddings) and speed for inference (Groq's Llama 3.1 API). The interface is a clean, conversational web app built with Streamlit.

## ✨ Key Features

  * **Fast Conversational AI:** Powered by the **Groq** API (running Llama 3.1) for near-instant responses.
  * **Streamlit Web App:** A simple, clean, and interactive chat interface (`medibot.py`).
  * **Local-First & Private:**
      * **On-Prem Embeddings:** Uses `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`) to create vectors on your own machine. Your data never leaves your computer during ingestion.
      * **Local Vector Store:** Uses `FAISS` to save all vectors to your local disk in the `vectorstore/` folder.
  * **RAG Pipeline:** Built with **LangChain**, it uses a "stuff" chain (`create_retrieval_chain`) to find the most relevant document chunks and "stuff" them into the LLM's context.
  * **Easy Setup:** A `requirements.txt` file is included for simple one-command installation.

-----

## 🤖 How It Works: The RAG Pipeline

This project is a classic example of a Retrieval-Augmented Generation (RAG) pipeline, which happens in two phases.

### 1\. Ingestion Phase (`create_memory_for_llm.py`)

This script builds the chatbot's "memory." It's a one-time process you run locally.

1.  **Load:** The `DirectoryLoader` finds and loads all your PDF documents from the `data/` folder.
2.  **Split:** The `RecursiveCharacterTextSplitter` breaks the large documents into smaller, 800-character chunks with a 200-character overlap (to maintain context).
3.  **Embed:** The `HuggingFaceEmbeddings` model (running 100% on-prem) converts each text chunk into a numerical vector.
4.  **Store:** `FAISS` takes all these vectors and saves them into a highly-efficient, searchable database in the `vectorstore/db_faiss` folder.

### 2\. Retrieval Phase (`medibot.py` / `connect_memory_with_llm.py`)

This is what happens every time you ask a question.

1.  **Query:** You type a question (e.g., "what is Secondary dysmenorrhea?") into the Streamlit app.
2.  **Retrieve:** The app searches the local FAISS database to find the top 3 document chunks (`k=3`) that are most semantically similar to your question.
3.  **Augment:** These relevant chunks (the "context") are combined with your original question into a prompt using a template from LangChain Hub (`langchain-ai/retrieval-qa-chat`).
4.  **Generate:** This complete prompt is sent to the **Groq API**, which uses Llama 3.1 to generate a fast, accurate answer based *only* on the provided context.
5.  **Respond:** The final answer is displayed in the chat window.

-----

## 🚀 Getting Started

Follow these steps to run your own instance of MediBot.

### 1\. Prerequisites

  * Python 3.9+
  * A Groq API Key (you can get one for free at [groq.com](https://groq.com/))

### 2\. Setup & Installation

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/your-username/rag-medical-chatbot.git
    cd rag-medical-chatbot
    ```

2.  **Create and Activate a Virtual Environment:**

    ```bash
    # Windows
    python -m venv .venv
    .venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    A `requirements.txt` file is included. Run:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Your API Key:**

      * Rename the `.env.example` file to `.env` (or create a new `.env` file).
      * Open the `.env` file and add your Groq API key:
        ```
        GROQ_API_KEY="gsk_YourSecretKeyGoesHere"
        ```

5.  **Add Your Documents:**

      * Place all your medical PDF files inside the `data/` folder.

### 3\. Running the Application

You must run these scripts in order.

**Step 1: Create the Vector Database (One-Time Setup)**
First, you must "teach" the bot by letting it read and index your documents.

```bash
python create_memory_for_llm.py
```

*This will create the `vectorstore/db_faiss` folder. You only need to run this once, or again if you add new documents to the `data/` folder.*

**Step 2: (Optional) Test in Your Terminal**
You can test the RAG chain directly in your terminal using the `connect_memory_with_llm.py` script.

```bash
python connect_memory_with_llm.py
```

**Step 3: Run the Streamlit Web App**
This is the main command to start the chatbot.

```bash
streamlit run medibot.py
```

Your browser will automatically open to the chat interface\!

-----

## 📂 Project Structure

```
rag medical chatbot/
├── data/
│   └── (Your medical PDFs go here...)
├── vectorstore/
│   └── db_faiss/
│       ├── index.faiss   (The vector database)
│       └── index.pkl     (The document mappings)
├── .env                  (Stores your GROQ_API_KEY)
├── create_memory_for_llm.py  (Script 1: Run this first to create the DB)
├── connect_memory_with_llm.py(Script 2: Optional terminal-based test)
├── medibot.py              (Script 3: The main Streamlit application)
├── requirements.txt        (All Python dependencies)
└── README.md               (This file)
```
