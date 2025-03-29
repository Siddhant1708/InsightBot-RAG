# InsightBot-RAG

InsightBot-RAG is a Retrieval-Augmented Generation (RAG) chatbot that answers queries from uploaded PDFs using Llama 3.3-70B and FAISS for efficient document retrieval.

## Features
- Upload a PDF and ask questions based on its content.
- Uses FAISS for fast and efficient similarity search.
- Runs on Streamlit for an interactive user experience.
- Utilizes HuggingFace embeddings for document vectorization.
- Implements a conversational retrieval chain with memory for better interactions.

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/InsightBot-RAG.git
cd InsightBot-RAG
```

### Step 2: Create a Conda Environment & Install Dependencies
```bash
conda create --name insightbot-rag python=3.11 -y
conda activate insightbot-rag
pip install -r requirements.txt
```

### Step 3: Set Up Environment Variables
Create a `.env` file in the project directory and add your API key:
```
GROQ_API_KEY=your_api_key_here
```

### Step 4: Run the Application
```bash
streamlit run app.py
```

## Usage
1. Upload a PDF file.
2. Ask questions about the document.
3. Get AI-generated responses based on the document's content.



## Contributing
Feel free to fork this repository, create a branch, and submit a pull request with your improvements!

## License
MIT License

