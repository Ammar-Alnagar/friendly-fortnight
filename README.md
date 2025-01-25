# Friendly Fortnight project:

**README.md**


```markdown
# Friendly Fortnight - Mawared HR Assistant 🤖

[![GitHub License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A specialized AI assistant for Mawared HR System using RAG architecture with Qdrant vector storage and Google's Gemini model.

## Features ✨
- Real-time conversational interface using Gradio
- Context-aware responses with retrieval-augmented generation
- Qdrant vector storage for HR knowledge base
- Conversation logging and analytics
- Support for multiple AI model providers
- Streamed responses for natural interaction

## Installation 🛠️

1. Clone the repository:
```bash
git clone https://github.com/yourusername/friendly-fortnight.git
cd friendly-fortnight
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration ⚙️

Create a `.env` file with your API credentials:
```ini
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
GEMINI=your_google_gemini_api_key
HF_TOKEN=your_huggingface_token
# Optional:
C_APIKEY=your_cerebras_api_key
OPENAPI_KEY=your_openai_api_key
```

## Usage 🚀
```bash
python app.py
```

The Gradio interface will launch in your default browser. Interact with the HR assistant by:
1. Typing questions in the input box
2. Viewing streamed responses in real-time
3. Using the "Clear Chat" button to reset conversation

## Logging 📝
All interactions are logged to Qdrant with:
- Question/answer pairs
- Timestamps
- Vector embeddings for analytics

## Tech Stack 🔧
| Component               | Technology                          |
|-------------------------|-------------------------------------|
| Language Model          | Google Gemini                       |
| Vector Store            | Qdrant                              |
| Embeddings              | sentence-transformers/all-MiniLM-L6 |
| UI Framework            | Gradio                              |
| NLP Pipeline            | LangChain                           |
| Environment Management  | python-dotenv                       |

## Contributing 🤝
Contributions are welcome! Please open an issue first to discuss proposed changes.

## License 📄
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

**requirements.txt**
```
gradio>=3.50.0
langchain-core>=0.1.0
langchain-community>=0.0.1
langchain-google-genai>=0.0.1
qdrant-client>=1.6.0
python-dotenv>=1.0.0
torch>=2.0.0
transformers>=4.30.0
huggingface_hub>=0.16.0
sentence-transformers>=2.2.0
langchain-huggingface>=0.0.1
```

Key features highlighted in the README:
1. Clear installation and configuration instructions
2. Visual hierarchy with emojis and badges
3. Tech stack table for quick overview
4. Streaming response visualization
5. Logging architecture explanation
6. Contribution guidelines
7. License information

