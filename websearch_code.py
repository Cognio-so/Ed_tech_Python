import os
import logging
import time
import asyncio
from typing import Annotated, TypedDict, List, Dict, Any, Optional, Literal
from dotenv import load_dotenv

# LangChain imports
from langchain_core.messages import BaseMessage
from langchain_core.tools import StructuredTool
from langchain_perplexity import ChatPerplexity
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# LangGraph imports
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class WebSearchState(TypedDict):
    """State for the web search graph."""
    messages: Annotated[List[BaseMessage], add_messages]
    search_results: Optional[List[Dict[str, Any]]]

class PerplexityWebSearchTool:
    """Reusable Perplexity web search tool for LangGraph."""
    
    def __init__(
        self, 
        max_results: int = 5,
        api_key: Optional[str] = None,
        model: str = "sonar",
        temperature: float = 0.7,
        include_links: bool = True,
    ):
        """
        Initialize the Perplexity web search tool.
        
        Args:
            max_results: Maximum number of search results to return
            api_key: Perplexity API key (defaults to PPLX_API_KEY env variable)
            model: Model to use (sonar recommended for search functionality)
            temperature: Temperature setting for the model
            include_links: Whether to include links in the response
        """
        self.max_results = max_results
        self.api_key = api_key
        self.include_links = include_links
        self.model = model
        
        # Set API key in environment if provided
        if api_key:
            os.environ["PPLX_API_KEY"] = api_key
        elif not os.getenv("PPLX_API_KEY"):
            raise ValueError("Perplexity API key is required. Set the PPLX_API_KEY environment variable.")
        
        try:
            # Initialize the search tool with Perplexity chat model
            self.chat_model = ChatPerplexity(
                model=model,
                temperature=temperature,
                pplx_api_key=os.getenv("PPLX_API_KEY"),
                streaming=False,
            )
            
            # Convert the chat model to a structured tool for searching
            self.search_tool = StructuredTool.from_function(
                func=self._search_func,
                name="perplexity_search",
                description="Search the web using Perplexity AI's API to find current and factual information",
                args_schema=self._get_args_schema(),
            )
            
            logger.info(f"PerplexityWebSearchTool initialized with max_results={max_results}, model={model}")
        except Exception as e:
            logger.error(f"Failed to initialize PerplexityWebSearchTool: {str(e)}")
            raise
    
    def _get_args_schema(self):
        """Create a dynamic args schema for the search tool."""
        from pydantic import BaseModel, Field
        
        class SearchSchema(BaseModel):
            query: str = Field(..., description="The search query to execute")
        
        return SearchSchema
    
    def _search_func(self, query: str) -> Dict[str, Any]:
        """
        Internal function to execute web search using Perplexity.
        
        Args:
            query: The search query
        
        Returns:
            Dictionary with search results
        """
        search_prompt = self._format_search_prompt(query)
        response = self.chat_model.invoke(search_prompt)
        
        return {
            "query": query,
            "results": response.content,
        }
    
    def _format_search_prompt(self, query: str) -> str:
        """
        Format the search prompt to instruct Perplexity to return search results.
        
        Args:
            query: The search query
            
        Returns:
            Formatted search prompt
        """
        link_instruction = "Include source URLs for each piece of information." if self.include_links else ""
        
        prompt = (
            f"Please provide comprehensive search results for: '{query}'\n\n"
            f"Return up to {self.max_results} relevant results. {link_instruction}\n"
            f"Format each result with main information, a brief summary, and the source URL (if available).\n"
            f"Make your response detailed and factual, with up-to-date information from the web."
        )
        
        return prompt
    
    # Async search method
    async def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute a web search using Perplexity asynchronously.
        
        Args:
            query: The search query
            
        Returns:
            List of search result objects. Returns an empty list on error.
        """
        try:
            # Log and time the search
            logger.info(f"Executing async web search for query: {query}")
            start_time = time.time()
            
            search_prompt = self._format_search_prompt(query)
            
            # Using 'ainvoke' for non-blocking I/O
            response = await self.chat_model.ainvoke(search_prompt)
            
            # Parse the response into search results format
            search_results = [{
                "content": response.content,
                "query": query,
            }]
            
            elapsed = time.time() - start_time
            logger.info(f"Search completed in {elapsed:.2f}s for: {query}")
            
            return search_results
        except Exception as e:
            logger.error(f"Error executing web search: {str(e)}")
            return []
    
    def get_tool(self) -> StructuredTool:
        """
        Get the underlying LangChain StructuredTool.
        
        Returns:
            StructuredTool instance for use in chains
        """
        return self.search_tool
    
    def create_tool_node(self) -> ToolNode:
        """
        Create a LangGraph ToolNode for the search tool.
        
        Returns:
            ToolNode instance for use in LangGraph
        """
        return ToolNode(tools=[self.search_tool])
    
    def bind_to_llm(self, llm):
        """
        Bind the tool to an LLM.
        
        Args:
            llm: Language model instance
            
        Returns:
            LLM with tools binding
        """
        return llm.bind_tools([self.search_tool])

def get_llm(model_name: str = "gpt-4o-mini", temperature: float = 0.5):
    """
    Get an LLM instance. Tries to initialize OpenAI's models first
    and falls back to Google's models on any error.
    
    Args:
        model_name: Name of the LLM to use.
        temperature: Temperature setting for the LLM.
        
    Returns:
        LLM instance.
    """
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OpenAI API key is required for the fallback LLM.")
                
        logger.info(f"Initializing OpenAI LLM: {model_name}")
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=openai_api_key
            )
    except Exception as e:
        logger.warning(f"Could not initialize OpenAI LLM ({e}). Falling back to gemini-2.5-flash-lite")
        try:
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not found.")

            logger.info(f"Initializing Google fallback LLM: gemini-2.5-flash-lite")
            return ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-lite",
                temperature=temperature,
                google_api_key=google_api_key,
            )
        except Exception as e_google:
            logger.error(f"Fatal: Could not initialize fallback Google LLM. Error: {e_google}")
            raise

def get_search_components(llm):
    """
    Get components needed for web search without creating graph nodes.
    
    Args:
        llm: Language model instance
        
    Returns:
        Dictionary with search tool and LLM with tools bound
    """
    # This now uses the Perplexity class 
    search_tool_instance = PerplexityWebSearchTool(max_results=5, model="sonar", include_links=True)
    llm_with_tools = search_tool_instance.bind_to_llm(llm)
    
    return {
        "search_tool": search_tool_instance.get_tool(),
        "llm_with_tools": llm_with_tools
    }