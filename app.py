import uuid
import subprocess 
import os
import torch
from dotenv import load_dotenv
from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from qdrant_client import QdrantClient, models
from langchain_openai import ChatOpenAI
import gradio as gr
import logging
from typing import List, Tuple, Generator
from dataclasses import dataclass
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_cerebras import ChatCerebras
from queue import Queue
from threading import Thread
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Message:
    role: str
    content: str
    timestamp: str

class ChatHistory:
    def __init__(self):
        self.messages: List[Message] = []
    
    def add_message(self, role: str, content: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.messages.append(Message(role=role, content=content, timestamp=timestamp))
    
    def get_formatted_history(self, max_messages: int = 10) -> str:
        recent_messages = self.messages[-max_messages:] if len(self.messages) > max_messages else self.messages
        formatted_history = "\n".join([
            f"{msg.role}: {msg.content}" for msg in recent_messages
        ])
        return formatted_history
    
    def clear(self):
        self.messages = []

# Load environment variables and setup
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
C_apikey = os.getenv("C_apikey")
OPENAPI_KEY = os.getenv("OPENAPI_KEY")
GEMINI = os.getenv("GEMINI")
if not HF_TOKEN:
    logger.error("HF_TOKEN is not set in the environment variables.")
    exit(1)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

try:
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        prefer_grpc=True
    )
except Exception as e:
    logger.error("Failed to connect to Qdrant.")
    exit(1)

# Create the main collection for Mawared HR
collection_name = "mawared"

try:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=384,
            distance=models.Distance.COSINE
        )
    )
except Exception as e:
    if "already exists" not in str(e):
        logger.error(f"Error creating collection: {e}")
        exit(1)

db = Qdrant(
    client=client,
    collection_name=collection_name,
    embeddings=embeddings,
)

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# Create a new collection for logs
logs_collection_name = "mawared_logs"

try:
    client.create_collection(
        collection_name=logs_collection_name,
        vectors_config=models.VectorParams(
            size=384,  # Same size as embeddings
            distance=models.Distance.COSINE
        )
    )
    logger.info(f"Created new Qdrant collection: {logs_collection_name}")
except Exception as e:
    if "already exists" not in str(e):
        logger.error(f"Error creating logs collection: {e}")
        exit(1)

def log_to_qdrant(question: str, answer: str):
    """Logs the question and answer to the Qdrant logs collection."""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            "question": question,
            "answer": answer,
            "timestamp": timestamp
        }
        
        # Convert the log entry to a vector (using embeddings)
        log_vector = embeddings.embed_documents([str(log_entry)])[0]
        
        # Generate a valid 64-bit unsigned integer ID
        valid_id = uuid.uuid4().int & (1 << 64) - 1  # Ensure it's a 64-bit unsigned integer
        
        # Insert the log into the Qdrant collection
        client.upsert(
            collection_name=logs_collection_name,
            points=[
                models.PointStruct(
                    id=valid_id,  # Use a valid 64-bit unsigned integer ID
                    vector=log_vector,
                    payload=log_entry
                )
            ]
        )
        logger.info(f"Logged question and answer to Qdrant collection: {logs_collection_name}")
    except Exception as e:
        logger.error(f"Failed to log to Qdrant: {e}")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-thinking-exp-01-21",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=GEMINI,
    stream=True,
)

template = """
You are a specialized AI assistant for the Mawared HR System, designed to deliver accurate and contextually relevant support based solely on the provided context and chat history.

---

Core Principles

1. Source of Truth: Rely exclusively on the information available in the retrieved context and chat history. Avoid fabricating details or using external knowledge.

2. Clarity and Precision: Provide clear, concise, and professional responses, ensuring they are easy to understand.

3. Actionable Guidance: Offer practical solutions, step-by-step workflows, and troubleshooting advice tailored to Mawared HR queries.

4. Structured Instructions: Use numbered or bullet-point lists for complex processes to ensure clarity.

5. Targeted Clarification: Ask specific, polite questions to gather missing details when a query lacks sufficient information.

6. Exclusive Focus: Limit your responses strictly to Mawared HR-related topics, avoiding unrelated discussions.

7. Professional Tone: Maintain a friendly, approachable, and professional demeanor in all communications.

---

Response Guidelines

1. Analyze the Query Thoughtfully
   Carefully review the user’s question and the chat history.
   Identify the user’s explicit intent and infer additional context where applicable.
   Note any gaps in the provided information.

2. Break Down Context Relevance
   Extract and interpret relevant details from the provided context or chat history.
   Match the user's needs to the most applicable information available.

3. Develop the Response Step-by-Step
   Frame a clear, logical structure to your response:
   - What is the user trying to achieve?
   - Which parts of the context directly address this?
   - What steps or details should be highlighted for clarity?
   Provide answers in a structured, easy-to-follow format, using numbered steps or bullet points.

4. Ask for Clarifications Strategically
   If details are insufficient, specify the missing information politely and clearly (e.g., “Could you confirm [specific detail] to proceed with [action/task]?”).

5. Ensure Directness and Professionalism
   Keep responses focused, avoiding unnecessary elaboration or irrelevant details.
   Uphold a professional and courteous tone throughout.

6. Double-Check for Exclusivity
   Verify that all guidance is strictly derived from the retrieved context or chat history.
   Avoid speculating or introducing external information about Mawared HR.

---

Handling Information Gaps

If the context is insufficient to answer the query:
- Clearly state that additional details are needed.
- Specify what information is required.
- Avoid fabricating or making assumptions to fill gaps.

---

Critical Constraints

- Strict Context Reliance: Base all responses solely on the provided context and chat history.
- Non-Mawared HR Queries: Politely decline to answer questions unrelated to Mawared HR.
- Answer Format: Always provide accurate answers in numbered steps without revealing your thought process or using code.

---

By adhering to these principles and guidelines, ensure every response is accurate, professional, and easy to follow.

Previous Conversation: {chat_history}
Retrieved Context: {context}
Current Question: {question}
Answer: {{answer}}
"""

prompt = ChatPromptTemplate.from_template(template)

def create_rag_chain(chat_history: str):
    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
            "chat_history": lambda x: chat_history
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

chat_history = ChatHistory()

def process_stream(stream_queue: Queue, history: List[List[str]]) -> Generator[List[List[str]], None, None]:
    """Process the streaming response and update the chat interface"""
    current_response = ""
    
    while True:
        chunk = stream_queue.get()
        if chunk is None:  # Signal that streaming is complete
            break
            
        current_response += chunk
        new_history = history.copy()
        new_history[-1][1] = current_response  # Update the assistant's message
        yield new_history


def ask_question_gradio(question: str, history: List[List[str]]) -> Generator[tuple, None, None]:
    try:
        if history is None:
            history = []
            
        chat_history.add_message("user", question)
        formatted_history = chat_history.get_formatted_history()
        rag_chain = create_rag_chain(formatted_history)
        
        # Update history with user message and empty assistant message
        history.append([question, ""])  # User message
        
        # Create a queue for streaming responses
        stream_queue = Queue()
        
        # Function to process the stream in a separate thread
        def stream_processor():
            try:
                for chunk in rag_chain.stream(question):
                    stream_queue.put(chunk)
                stream_queue.put(None)  # Signal completion
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                stream_queue.put(None)
        
        # Start streaming in a separate thread
        Thread(target=stream_processor).start()
        
        # Yield updates to the chat interface
        response = ""
        for updated_history in process_stream(stream_queue, history):
            response = updated_history[-1][1]
            yield "", updated_history
        
        # Add final response to chat history
        chat_history.add_message("assistant", response)
        
        # Log the question and answer to Qdrant
        logger.info("Attempting to log question and answer to Qdrant")
        log_to_qdrant(question, response)
        
    except Exception as e:
        logger.error(f"Error during question processing: {e}")
        if not history:
            history = []
        history.append([question, "An error occurred. Please try again later."])
        yield "", history

def clear_chat():
    chat_history.clear()
    return [], ""

# Gradio Interface
with gr.Blocks() as iface:
    gr.Image("Image.jpg", width=750, height=300, show_label=False, show_download_button=False)
    gr.Markdown("# Mawared HR Assistant 3.2.6")
    gr.Markdown('### Instructions')
    gr.Markdown("Ask a question about MawaredHR and get a detailed answer , Added new model and Logging")
    
    chatbot = gr.Chatbot(
        height=750,
        show_label=False,
        bubble_full_width=False,
    )
    
    with gr.Row():
        with gr.Column(scale=20):
            question_input = gr.Textbox(
                label="Ask a question:",
                placeholder="Type your question here...",
                show_label=False
            )
        with gr.Column(scale=4):
            with gr.Row():
                with gr.Column():
                    send_button = gr.Button("Send", variant="primary", size="sm")
                    clear_button = gr.Button("Clear Chat", size="sm")
    
    # Handle both submit events (Enter key and Send button)
    submit_events = [question_input.submit, send_button.click]
    for submit_event in submit_events:
        submit_event(
            ask_question_gradio,
            inputs=[question_input, chatbot],
            outputs=[question_input, chatbot]
        )
    
    clear_button.click(
        clear_chat,
        outputs=[chatbot, question_input]
    )

if __name__ == "__main__":
    iface.launch()