import os
import logging
import base64
import asyncio
from io import BytesIO
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Union, AsyncGenerator, TypedDict, Annotated, Any
from dataclasses import field, dataclass
import concurrent.futures
import inspect
from functools import wraps
import tempfile
import shutil

from PIL import Image
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.tools import tool, Tool
import time
from langchain_google_genai import ChatGoogleGenerativeAI
import langchain
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Add error handling for Qdrant imports
try:
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient, models
    from qdrant_client.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logging.warning("Qdrant packages not found. Vector storage features will be disabled.")

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Add error handling for retriever imports
try:
    from langchain_community.retrievers import BM25Retriever
    from langchain.retrievers import EnsembleRetriever
    RETRIEVER_AVAILABLE = True
except ImportError:
    RETRIEVER_AVAILABLE = False
    logging.warning("BM25Retriever or EnsembleRetriever not available. Hybrid search will be disabled.")

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langsmith import traceable

from langchain_community.document_loaders import (
    PDFPlumberLoader, Docx2txtLoader, BSHTMLLoader, TextLoader, UnstructuredURLLoader, JSONLoader
)

# Import the web search tool
from websearch_code import PerplexityWebSearchTool


load_dotenv()

# LangSmith configuration
LANGSMITH_TRACING="true"
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY=os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT="Vamshi-test"

QDRANT_VECTOR_PARAMS = VectorParams(size=1536, distance=Distance.COSINE)
CONTENT_PAYLOAD_KEY = "page_content"
METADATA_PAYLOAD_KEY = "metadata"

default_qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
default_qdrant_api_key = os.getenv("QDRANT_API_KEY")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@tool
def general_conversation_tool(query: str) -> str:
    """
    Use this tool ONLY for simple, non-substantive conversational turns like 'hello', 'thanks', or 'how are you?'.
    You MUST NOT use this tool for any question that seeks information, facts, definitions, or explanations (e.g., 'what is X?', 'explain Y').
    For informational queries, you MUST use another tool or answer from your own knowledge. This tool's purpose is strictly for handling conversational pleasantries.
    """
    return "give relevant response to the user query and encourage user to ask questions."


def async_error_handler(func):
    """
    Decorator for enhanced async error handling that works with both
    coroutines and async generators.
    """
    if inspect.isasyncgenfunction(func):
        # Wrapper for async generators
        @wraps(func)
        async def generator_wrapper(*args, **kwargs):
            try:
                async for item in func(*args, **kwargs):
                    yield item
            except Exception as e:
                logging.error(f"Error in async generator {func.__name__}: {e}")
                raise
        return generator_wrapper
    else:
        # Wrapper for regular async functions (coroutines)
        @wraps(func)
        async def coroutine_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logging.error(f"Error in coroutine {func.__name__}: {e}")
                raise
        return coroutine_wrapper

class VectorStoreManager:
    """Manages the Qdrant vector store operations."""
    def __init__(self, config):
        self.config = config
        self.qdrant_client = None
        self.vector_store = None
        self.embeddings = OpenAIEmbeddings(
            model=self.config.embedding_model,
            openai_api_key=self.config.openai_api_key
        )
        if QDRANT_AVAILABLE:
            self.qdrant_client = QdrantClient(
                url=self.config.qdrant_url, 
                api_key=self.config.qdrant_api_key,
                timeout=20.0
            )

    async def initialize_collection(self):
        """Initializes the Qdrant collection, creating it if it doesn't exist."""
        if not self.qdrant_client:
            logging.error("Qdrant client not available.")
            return

        target_collection = self.config.qdrant_collection_name
        if not target_collection:
            logging.error("Qdrant collection name is not set.")
            raise ValueError("Cannot initialize collection without a name.")
            
        try:
            collections = await asyncio.to_thread(self.qdrant_client.get_collections)
            collection_names = [col.name for col in collections.collections]

            if target_collection not in collection_names:
                logging.info(f"Creating Qdrant collection: {target_collection}")
                await asyncio.to_thread(
                    self.qdrant_client.create_collection,
                    collection_name=target_collection,
                    vectors_config=QDRANT_VECTOR_PARAMS
                )
            
            self.vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=target_collection,
                embedding=self.embeddings,
                content_payload_key=CONTENT_PAYLOAD_KEY,
                metadata_payload_key=METADATA_PAYLOAD_KEY
            )
            logging.info(f"Vector store initialized for collection: {target_collection}")

        except Exception as e:
            logging.error(f"Failed to initialize Qdrant collection: {e}")
            raise
            
    def get_retriever(self, k: int):
        """Gets a retriever from the initialized vector store."""
        if not self.vector_store:
            raise RuntimeError("Vector store is not initialized. Call initialize_collection first.")
        return self.vector_store.as_retriever(search_kwargs={"k": k})
        
    async def aadd_documents(self, documents: List[Document]):
        """Asynchronously adds documents to the vector store."""
        if not self.vector_store:
            raise RuntimeError("Vector store is not initialized. Call initialize_collection first.")
        await self.vector_store.aadd_documents(documents)
    

    @async_error_handler
    async def clear_collection_async(self):
        """
        Deletes and immediately recreates the collection to wipe all its data.
        """
        if not self.qdrant_client:
            logging.error("Qdrant client not available. Cannot clear collection.")
            return

        target_collection = self.config.qdrant_collection_name
        logging.warning(f"Clearing all documents from Qdrant collection: {target_collection}")
        
        try:
            await asyncio.to_thread(
                self.qdrant_client.delete_collection,
                collection_name=target_collection
            )
            logging.info(f"Successfully deleted collection: {target_collection}")
            
            await self.initialize_collection()
            logging.info(f"Successfully re-initialized collection: {target_collection}")

        except Exception as e:
            logging.error(f"Error during collection clearing for {target_collection}, attempting to re-initialize. Error: {e}")
            await self.initialize_collection()

class TutorState(TypedDict):
    """State for the AI tutor agent."""
    messages: Annotated[list, add_messages]

@dataclass
class RAGTutorConfig:
    """Configuration class for the AI Tutor."""
    openai_api_key: str = os.getenv("OPENAI_API_KEY")
    google_api_key: str = os.getenv("GOOGLE_API_KEY")
    llm_model: str = "gpt-4o-mini"  # Default to OpenAI's GPT-4o-mini
    streaming: bool = True
    temperature: float = 0.2
    max_tokens: int = 2000
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 5
    image_extensions: Tuple[str, ...] = field(default_factory=lambda: (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"))
    max_workers: int = 4
    qdrant_url: str = field(default_factory=lambda: default_qdrant_url)
    qdrant_api_key: Optional[str] = field(default_factory=lambda: default_qdrant_api_key)
    qdrant_collection_name: Optional[str] = None
    web_search_enabled: bool = False
    system_prompt: str = """You are an expert AI tutor specializing in homework assistance. Your mission is to help students understand and learn from their assignments.

**🎓 Teaching Philosophy:**
- **Guide, don't solve**: Provide step-by-step explanations rather than direct answers. This is your default behavior.
- **Encourage learning**: Ask follow-up questions to ensure understanding.
- **Build connections**: Relate homework topics to real-world applications.

**📚 Homework Analysis Approach:**
- **Document Analysis**: When homework documents are uploaded, identify key learning objectives and concepts.
- **Web Search**: If web search is enabled, you can use the 'websearch_tool' to find current, real-world information or to answer general knowledge questions that are not covered in the uploaded documents.
- **Combined Search**: When a knowledge base is available AND web search is enabled, you MUST consider using both the `knowledge_base_retriever` for document-specific information and the `websearch_tool` for broader context or recent information. Synthesize the findings from both sources to provide a comprehensive answer. Prioritize the knowledge base if the query seems directly related to it, but augment with web search for added depth.

**🖼️ Answering Questions About Images:**
- When the user uploads an image, the system analyzes it and stores a detailed description in the knowledge base.
- To answer a question about an image (e.g., "what text is in 'chart.png'?"), you **MUST** use the `knowledge_base_retriever` tool to find the description of that image.
- Your first step is to use the retrieved information to provide a direct, factual answer. For example, if the user asks "What does this image say?", you should use the tool and then respond with: "Based on the analysis of the image, the text reads: '...'"
- After providing the direct answer from the knowledge base, you can then move into your tutor role to provide further explanations.

**🔍 Response Strategy (for non-image queries):**
- **Concept Explanation**: Break down complex topics into digestible parts.
- **Solution Methodology**: Teach the "how" and "why" behind solutions.
- **Common Mistakes**: Highlight typical errors students make.
- **Study Tips**: Suggest effective learning strategies for the subject.
- for greeting and praise responses, use the `general_conversation_tool` to engage in simple conversation and encourage user to ask questions.

**❓ Interactive Learning:**
- Ask clarifying questions about specific difficulties
- Provide hints before full explanations
- Encourage students to attempt solutions first
- Offer additional practice suggestions
**🌐 Web Search using websearch_tool **: When providing information from a web search, for each source, you MUST always display its favicon. for explaining user in much more detail, you MUST display the video link to explain in more detailed manner. Format the citations at the end of your response. For each citation, include the favicon, the title of the page, and the URL.
** Rag Knowledge Base**: If a knowledge base is available, use the knowledge base tool to find relevant information from the uploaded documents. Always prioritize this tool for questions about specific documents.
** If RAG and Web Search are both enabled**: Use the knowledge base tool first, and then augment with web search results as needed.
** Remember, your goal is to empower students to learn and grow, not just complete assignments for them. Always prioritize understanding over rote answers, and always provide detailed explanations and justifications for your answers.
** Always give lengthy responses with detailed explanations, examples, and justifications for your answers in proper markdown format with proper headings, bullet points, and code blocks where necessary. if its websearch_tool response provide citations.

**🕒 Current Time**: {current_time}
"""

    @classmethod
    def from_env(cls) -> 'RAGTutorConfig':
        """Create configuration from environment variables."""
        return cls()

class AsyncRAGTutor:
    def __init__(self, storage_manager: Any, config: Optional[RAGTutorConfig] = None):
        self.config = config or RAGTutorConfig()
        
        unique_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
        self.config.qdrant_collection_name = f"rag_session_{unique_id}"
        logging.info(f"Initialized new tutor instance with collection: {self.config.qdrant_collection_name}")

        try:
            logging.info("Initializing response through OpenAI's API model ( GPT-4o-mini).")
            self.llm = ChatOpenAI(
                model=self.config.llm_model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                streaming=self.config.streaming,
                openai_api_key=self.config.openai_api_key,
            )
        except Exception as e:
            logging.error(f"Error initializing ChatOpenAI: {e}")
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-lite",
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                google_api_key=self.config.google_api_key,
                streaming=self.config.streaming,
                disable_streaming=False
        )
        
        self.storage_manager = storage_manager

        self.retriever_tool = Tool(
            name="knowledge_base_retriever",
            func=self.knowledge_base_retrieval_tool,
            coroutine=self.knowledge_base_retrieval_tool,
            description=(
                "Use this tool to answer questions about any uploaded documents, including text, PDFs, and images. "
                "This is your primary tool for retrieving information. If the user's query mentions a specific filename "
                "(e.g., 'homework.pdf', 'diagram.png'), you MUST use this tool. It is the only way to access the "
                "content of the files the user has provided."
            )
        )
        
        self.tools = [self.retriever_tool, general_conversation_tool]
        
        if self.config.web_search_enabled:
            if os.getenv("PPLX_API_KEY"):
                logging.info("Web search is enabled and PPLX_API_KEY is set.")
                # Updated to use Perplexity 
                websearch_tool = PerplexityWebSearchTool(
                    max_results=5, 
                    model="sonar", 
                    include_links=True
                ).get_tool()
                self.tools.append(websearch_tool)
            else:
                logging.warning("Web search is enabled in config, but PPLX_API_KEY is not set. Web search will be disabled.")
                
        self.tool_map = {tool.name: tool for tool in self.tools}
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        
        self.vectorstore_manager = VectorStoreManager(self.config) if QDRANT_AVAILABLE else None
        self.ensemble_retriever = None
        self.graph = None
        self.short_responses = ["ok", "okay", "thanks", "thank you", "great", "good", "cool","hello", "hi", "hey", "greetings", "yo", "sup", "good morning", "good afternoon", "good evening"]
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers)

        self.rephrase_prompt = PromptTemplate.from_template(
            """Given a chat history and a follow-up question, rephrase the follow-up question into a clear, standalone.

**Instructions:**
1.  **Handle Conversational Fillers First:** If the `Follow-up Question` is a simple, common conversational phrase (e.g., "okay", "great", "thanks", "sounds good", "I see", "okay great"), your most important task is to return it **UNCHANGED**. Do not try to add context or rephrase it. This rule overrides all others.
    - **Example:**
        - Chat History: "...all about FastAPI."
        - Follow-up Question: okay great
        - Standalone Question: okay great

2.  **Handle Uploaded Files:** If the question is NOT a conversational filler (per Rule #1) AND the `Chat History` contains a `System Note` listing recently uploaded files, you MUST rewrite the `Follow-up Question` to be specifically about those files, including the filename(s).
    - **Example for documents:**
        - System Note: The user has just uploaded 'homework_chapter_3.pdf'.
        - Follow-up Question: can you explain this?
        - Standalone Question: Can you explain the content of the document 'homework_chapter_3.pdf'?

3.  **General Rephrasing:** If the question is not a filler and there's no file context, use the chat history to create a clear, standalone question.
4.  If the original question is already a perfectly standalone question, return it as is.

 Chat History:
 {chat_history}
 
 Follow-up Question: {question}
 
 Standalone Question:"""
        )
        self.rephrase_chain = self.rephrase_prompt | self.llm | StrOutputParser()


    @async_error_handler
    async def clear_knowledge_base_async(self):
        """Public method to clear the knowledge base and reset the retriever."""
        logging.info("Clearing knowledge base and resetting retriever.")
        if self.vectorstore_manager:
            await self.vectorstore_manager.clear_collection_async()
        self.ensemble_retriever = None

    async def knowledge_base_retrieval_tool(self, query: str) -> str:
        """Use this tool to answer questions by retrieving relevant information from the knowledge base."""
        if not self.ensemble_retriever:
            return "No knowledge base has been configured. Please upload documents to create one."
        logging.info(f"Activating knowledge base tool for query: {query}")
        retrieved_docs = await self.ensemble_retriever.ainvoke(query)
        if not retrieved_docs:
            return "No relevant information was found in the knowledge base for this query. You can try rephrasing the question."
        return self.format_docs(retrieved_docs)

    def update_web_search_status(self, web_search_enabled: bool):
        """Dynamically enables or disables the web search tool without re-initializing."""
        self.config.web_search_enabled = web_search_enabled
        logging.info(f"Updating web search status to: {self.config.web_search_enabled}")

        # Check if the web search tool is currently in our list of tools
        websearch_tool_present = any(tool.name == 'perplexity_search' for tool in self.tools)

        if self.config.web_search_enabled and not websearch_tool_present:
            if os.getenv("PPLX_API_KEY"):
                logging.info("Enabling and adding web search tool.")
                # We instantiate the tool using the class from websearch_code.py
                websearch_tool = PerplexityWebSearchTool(
                    max_results=5, 
                    model="sonar", 
                    include_links=True
                ).get_tool()
                self.tools.append(websearch_tool)
            else:
                logging.warning("Cannot enable web search: PPLX_API_KEY is not set.")
                self.config.web_search_enabled = False # Ensure config reflects reality

        elif not self.config.web_search_enabled and websearch_tool_present:
            logging.info("Disabling and removing web search tool.")
            self.tools = [tool for tool in self.tools if tool.name != 'perplexity_search']

        # CRITICAL: Re-create the tool map and re-bind the tools to the LLM
        self.tool_map = {tool.name: tool for tool in self.tools}
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        logging.info(f"Tools updated. Current tools: {[tool.name for tool in self.tools]}")

    def _is_greeting_or_short_response(self, query: str) -> bool:
        """Checks if the query is a simple greeting or a short, common response."""
        normalized_query = query.strip().lower()
        return normalized_query in self.short_responses

    @staticmethod
    async def encode_image_async(image_path: str) -> str:
        """Convert image to base64 string asynchronously."""
        loop = asyncio.get_event_loop()
        def _encode_image():
            with Image.open(image_path) as img:
                buffer = BytesIO()
                img_format = img.format if img.format in ['JPEG', 'PNG'] else 'JPEG'
                img.save(buffer, format=img_format)
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
        return await loop.run_in_executor(None, _encode_image)

    def format_docs(self, docs: List[Document]) -> str:
        """Format documents for the prompt."""
        if not docs:
            return "No relevant documents found in the knowledge base."
        return "\n\n".join(f"Source: {doc.metadata.get('source', 'N/A')}\nContent: {doc.page_content}" for doc in docs)

    @traceable(name="initialize_vectorstore")
    async def initialize_vectorstore_async(self, documents: List[Document]):
        """Initialize the vector store with documents."""
        try:
            if not documents:
                logging.info("No documents to initialize vector store with.")
                return False

            if not QDRANT_AVAILABLE:
                logging.warning("Qdrant is not available. Vector store will not be initialized.")
                return False

            if self.vectorstore_manager is None:
                if self.config.qdrant_collection_name is None:
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
                    self.config.qdrant_collection_name = f"rag_session_{timestamp}"
                
                config = {
                    'url': self.config.qdrant_url,
                    'api_key': self.config.qdrant_api_key,
                    'collection_name': self.config.qdrant_collection_name
                }
                self.vectorstore_manager = VectorStoreManager(config)
                await self.vectorstore_manager.initialize_collection()
                logging.info(f"Vector store initialized for collection: {self.config.qdrant_collection_name}")
            
            await self.vectorstore_manager.aadd_documents(documents)
            
            self.retriever = self.vectorstore_manager.get_retriever(k=self.config.retrieval_k)
            
            if RETRIEVER_AVAILABLE:
                try:
                    bm25_retriever = BM25Retriever.from_documents(
                        documents, preprocess_func=lambda text: text.split()
                    )
                    bm25_retriever.k = self.config.retrieval_k
                    
                    self.ensemble_retriever = EnsembleRetriever(
                        retrievers=[self.retriever, bm25_retriever],
                        weights=[0.7, 0.3]
                    )
                    logging.info("Ensemble retriever configured with vector + BM25")
                except Exception as e:
                    logging.error(f"Error setting up hybrid retriever: {e}. Falling back to vector retriever.")
                    self.ensemble_retriever = self.retriever
            else:
                self.ensemble_retriever = self.retriever
            
            return True
        except Exception as e:
            logging.error(f"Error initializing vector store: {e}")
            return False

    @async_error_handler
    async def ingest_async(self, storage_keys: List[str]) -> bool:
        """Concurrently ingests documents from storage keys, now with image support."""
        if not storage_keys:
            logging.warning("No storage keys provided for ingestion.")
            return False

        logging.info(f"Starting concurrent ingestion for {len(storage_keys)} storage keys.")

        async def _process_single_key(key: str) -> List[Document]:
            """Fetches a file from storage and processes it as a document or image."""
            try:
                file_content = await self.storage_manager.get_file_content_bytes_async(key)
                if not file_content:
                    logging.error(f"Failed to get content for key: {key}")
                    return []

                filename = os.path.basename(key)
                
                if filename.lower().endswith(self.config.image_extensions):
                    logging.info(f"🖼️ Detected image file: {filename}. Analyzing with vision model.")
                    image_description = await self._process_image_from_bytes_async(file_content, filename)
                    if image_description:
                        return [Document(page_content=image_description, metadata={'source': filename, 'type': 'image'})]
                else:
                    return await self._process_document_from_bytes_async(file_content, filename)
            except Exception as e:
                logging.error(f"Exception while processing file for key {key}: {e}", exc_info=True)
            return []

        tasks = [_process_single_key(key) for key in storage_keys]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_processed_docs = []
        for res in results:
            if isinstance(res, list):
                all_processed_docs.extend(res)
            elif isinstance(res, Exception):
                logging.error(f"Error during concurrent ingestion task: {res}")

        if all_processed_docs:
            logging.info(f"Ingesting {len(all_processed_docs)} processed documents into vector store.")
            return await self.initialize_vectorstore_async(all_processed_docs)
        else:
            logging.warning("No documents were successfully processed for ingestion.")
            return False

    async def _process_image_from_bytes_async(self, image_bytes: bytes, filename: str) -> Optional[str]:
        """
        Processes an image from bytes, generating a description using a vision model.
        It first tries to use Google's Generative AI model and falls back to OpenAI's model on failure.
        """
        try:
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            prompt_text = "Describe this image for a search index. Be detailed about any objects, text, people, and the overall scene or context. This description will be used to find this image in a knowledge base."
            image_url = f"data:image/jpeg;base64,{base64_image}"

            try:
                logging.info(f"Attempting to generate description for '{filename}' with OpenAI's model.")
                vision_model_openai = ChatOpenAI(
                    model=self.config.llm_model,
                    max_tokens=1024,
                    openai_api_key=self.config.openai_api_key
                    )
                response = await vision_model_openai.ainvoke([
                    HumanMessage(content=[
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ])
                ])
                description = f"Image Content (from file: {filename}):\n{response.content}"
                logging.info(f"Successfully generated description for '{filename}' using OpenAI's model.")
                return description

            except Exception as e_openai:
                logging.error(f"Error using OpenAI's model for '{filename}': {e_openai}. Falling back to OpenAI's model.")

                # Fallback to OpenAI's model
                try:
                    logging.info(f"Attempting to generate description for '{filename}' with Google's model.")
                    vision_model = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash-lite",
                    max_tokens=self.config.max_tokens,
                    google_api_key=self.config.google_api_key,
                )
                    response = await vision_model.ainvoke([
                    HumanMessage(content=[
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ])
                ])
                    description = f"Image Content (from file: {filename}):\n{response.content}"
                    logging.info(f"Successfully generated description for '{filename}' using Google's model.")
                    return description

                except Exception as e_google:
                    logging.error(f"Error processing image '{filename}' with fallback Google model: {e_google}", exc_info=True)
                    return None

        except Exception as e_initial:
            # This catches errors in the initial setup (e.g., base64 encoding)
            logging.error(f"An initial error occurred while processing '{filename}': {e_initial}", exc_info=True)
            return None
            
    async def _process_document_from_bytes_async(self, file_bytes: bytes, filename: str) -> List[Document]:
        """Processes a standard document from bytes by writing to a temp file for loading."""
        temp_file_path = None
        try:
            file_extension = os.path.splitext(filename)[1].lower()
            with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
                temp_file.write(file_bytes)
                temp_file_path = temp_file.name
            
            loader = None
            if file_extension == '.pdf': loader = PDFPlumberLoader(temp_file_path)
            elif file_extension == '.docx': loader = Docx2txtLoader(temp_file_path)
            elif file_extension == '.json': loader = JSONLoader(temp_file_path, jq_schema='.[*]', text_content=False)
            elif file_extension in ['.html', '.htm', '.xhtml']: loader = BSHTMLLoader(temp_file_path)
            elif file_extension in ['.txt', '.md']: loader = TextLoader(temp_file_path, autodetect_encoding=True)
            
            if loader:
                logging.info(f"📄 Loading document: {filename}")
                docs = await asyncio.to_thread(loader.load)
                for doc in docs:
                    doc.metadata['source'] = filename 
                return docs
            else:
                logging.warning(f"No loader available for file extension {file_extension} of file {filename}")

        except Exception as e:
            logging.error(f"Error processing document {filename}: {e}", exc_info=True)
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        return []

    @async_error_handler
    async def _agent_executor_stream_async(self, query: str, formatted_time: str, image_path: Optional[str] = None, is_knowledge_base_ready: bool = False) -> AsyncGenerator[str, None]:
        """Private method to invoke the tool-enabled LLM with a finalized query."""
        system_prompt_text = self.config.system_prompt.format(current_time=formatted_time)
        prompt_notes = []
        if is_knowledge_base_ready:
            prompt_notes.append("- **Knowledge Base**: AVAILABLE. Prioritize the 'knowledge_base_retriever' tool for questions about uploaded documents.")
        else:
            prompt_notes.append("- **Knowledge Base**: NOT AVAILABLE. Do not use the 'knowledge_base_retriever' tool.")

        if self.config.web_search_enabled:
            prompt_notes.append("- **Web Search**: ENABLED. You can use the 'websearch_tool' tool for web searches.")
        else:
            prompt_notes.append("- **Web Search**: DISABLED.")
        
        if prompt_notes:
            system_prompt_text += "\n\n**Current Session Status:**\n" + "\n".join(prompt_notes)

        message_content = [{"type": "text", "text": query}]
        if image_path:
            logging.info(f"Encoding image {image_path} for direct agent analysis.")
            base64_image = await self.encode_image_async(image_path)
            message_content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            )

        messages = [SystemMessage(content=system_prompt_text), HumanMessage(content=message_content)]
        ai_response_with_tool = await self.llm_with_tools.ainvoke(messages)
        

        # **FIXED LOGIC STARTS HERE**
        if not ai_response_with_tool.tool_calls:
            # Scenario 1: The LLM decided to answer directly.
            # To ensure a consistent streaming experience, we will re-invoke the LLM
            # in streaming mode to deliver the final answer.
            logging.info("LLM provided a direct answer without tool usage. Invoking a new stream for the response.")
            final_chain = self.llm | StrOutputParser()
            # `messages` already contains the System and Human messages that led to this decision.
            async for chunk in final_chain.astream(messages):
                yield chunk
            return

        messages.append(ai_response_with_tool)
        # Scenario 2: The LLM decided to call one or more tools.
        for tool_call in ai_response_with_tool.tool_calls:
            tool_name = tool_call["name"]
            logging.info(f"LLM decided to call tool: {tool_name} with args {tool_call['args']}")
            if tool_name in self.tool_map:
                selected_tool = self.tool_map[tool_name]
                tool_output = await selected_tool.ainvoke(tool_call["args"])
            else:
                tool_output = f"Error: Tool '{tool_name}' not found."
            messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"]))

        # Now, invoke the model again with the tool results to get the final answer.
        final_chain = self.llm | StrOutputParser()
        async for chunk in final_chain.astream(messages):
            yield chunk
    async def setup_langgraph_async(self):
        """Set up the LangGraph workflow asynchronously."""
        async def agent_logic(state: TutorState):
            last_message = state['messages'][-1]
            history_messages = state['messages'][:-1]
            response = await self.process_query_async(last_message.content, history=history_messages)
            return {"messages": [AIMessage(content=response)]}
 
        workflow = StateGraph(TutorState)
        workflow.add_node("agent", agent_logic)
        workflow.add_edge(START, "agent")
        workflow.add_edge("agent", END)
        
        self.graph = workflow.compile()
        logging.info("LangGraph workflow created successfully.")
        return self.graph
        
    @async_error_handler
    async def _rephrase_query_with_history_async(self, query: str, history: List[Dict[str, Any]], uploaded_files: Optional[List[str]] = None) -> str:
        """Rephrase the query using chat history to make it standalone."""
        try:
            chat_history_str = ""
            if uploaded_files:
                files_str = "', '".join(uploaded_files)
                chat_history_str += f"System Note: The user has just uploaded the following file(s): '{files_str}'. The follow-up question likely refers to these files.\n\n"
            
            for msg in history[-5:]:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role and content:
                    chat_history_str += f"{role.title()}: {content}\n"
            
            rephrased = await self.rephrase_chain.ainvoke({
                "chat_history": chat_history_str,
                "question": query
            })
            logging.info(f"Query rephrased from '{query}' to '{rephrased}'")
            return rephrased
        except Exception as e:
            logging.error(f"Error rephrasing query: {e}")
            return query

    @async_error_handler
    async def run_agent_async(self, query: str, history: List[Dict[str, Any]], image_storage_key: Optional[str] = None, is_knowledge_base_ready: bool = False, uploaded_files: Optional[List[str]] = None) -> AsyncGenerator[str, None]:
        """Run the agent with a query and history, handling storage keys directly."""
        formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        rephrased_query = await self._rephrase_query_with_history_async(query, history, uploaded_files)
        temp_image_path = None
        if image_storage_key:
            try:
                image_bytes = await self.storage_manager.get_file_content_bytes_async(image_storage_key)
                if image_bytes:
                    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(image_storage_key)[1], delete=False) as temp_file:
                        temp_file.write(image_bytes)
                        temp_image_path = temp_file.name
                    logging.info(f"Loaded image from storage key '{image_storage_key}' for processing.")
                else:
                    logging.error(f"Failed to load image from storage key: {image_storage_key}")
            except Exception as e:
                logging.error(f"Error handling image storage key {image_storage_key}: {e}")

        try:
            logging.info("Processing rephrased query via the tool-enabled agent.")
            async for chunk in self._agent_executor_stream_async(
                query=rephrased_query,
                formatted_time=formatted_time,
                image_path=temp_image_path,
                is_knowledge_base_ready=is_knowledge_base_ready
            ):
                yield chunk
        finally:
            if temp_image_path and os.path.exists(temp_image_path):
                try:
                    os.unlink(temp_image_path)
                    logging.info(f"Cleaned up temporary image file: {temp_image_path}")
                except Exception as e:
                    logging.error(f"Error cleaning up temporary image file: {e}")

    def __del__(self):
        """Clean up the executor on deletion."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)