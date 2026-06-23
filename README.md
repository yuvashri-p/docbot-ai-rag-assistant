# 📄 DocBot AI — Personal RAG-Based Document Assistant

DocBot is an AI-powered assistant that lets you upload any PDF and ask questions about it in plain English. It uses Retrieval-Augmented Generation (RAG) to find the most relevant content and Google's Gemini API to generate accurate, context-grounded answers.

## Features
- Upload any text-based PDF
- Ask questions in natural language
- See exactly which part of the document was used to answer
- Clean, custom-designed chat interface built in Streamlit

## Tech Stack
- **Frontend:** Streamlit
- **PDF Processing:** PyPDF
- **RAG Framework:** LangChain
- **Vector Database:** ChromaDB
- **AI Model:** Google Gemini API (gemini-2.5-flash)
- **Embeddings:** Google Generative AI Embeddings

## How it works
1. PDF text is extracted and split into chunks
2. Each chunk is converted into a vector embedding
3. Embeddings are stored in ChromaDB
4. User questions are matched against stored chunks using semantic search
5. Relevant chunks + question are sent to Gemini for a final answer

## Setup
\`\`\`bash
pip install -r requirements.txt
streamlit run app.py
\`\`\`

Add your Gemini API key in a `.env` file:
\`\`\`
GOOGLE_API_KEY=your_key_here
\`\`\`

Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

## Live Demo
🔗 [Try DocBot AI live](https://docbot-ai-rag-assistant-iyvea67yhowvtliuthywyx.streamlit.app/)