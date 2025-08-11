import os
import json
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# Get the API key from environment variables
pplx_api_key = os.getenv("PPLX_API_KEY")

# Initialize chat as None first
chat = None

try:
    from langchain_perplexity import ChatPerplexity
    
    # Check if the API key was loaded successfully
    if not pplx_api_key:
        raise ValueError("PPLX_API_KEY not found in environment variables. Please check your .env file.")

    # Instantiate the ChatPerplexity model
    chat = ChatPerplexity(
        temperature=0.7,
        model="sonar",  # Updated to a current model name
        pplx_api_key=pplx_api_key
    )
    logger.info("Perplexity chat initialized successfully")
except ImportError as e:
    logger.warning(f"Could not import langchain_perplexity: {e}")
except Exception as e:
    logger.warning(f"Failed to initialize Perplexity chat: {e}")

def get_query_from_user_input():
    """Prompts the user for details and builds a natural language query."""
    print("Please provide the following details to generate your query:")
    
    # Prompt the user for each piece of information
    topic = input("Enter the topic: ")
    grade_level = input("Enter the grade level (e.g., 10): ")
    subject = input("Enter the subject (e.g., History): ")
    content_type = input("Enter the preferred content type (e.g., articles, videos): ")
    language = input("Enter the language (e.g., English): ")
    comprehension = input("Enter the comprehension level (e.g., beginner, intermediate): ")
    num_results = input("Enter the number of results (e.g., 5): ")
    
    # Prompt for max_results and handle potential errors
    try:
        max_results = int(num_results)
    except ValueError:
        print("Invalid input for maximum results. Using a default value of 10.")
        max_results = 10    

    # Building the query from user inputs
    query = (
        f"Show me up to {max_results} {content_type} about '{topic}' "
        f"for a grade {grade_level} {subject} class. "
        f"The content should be in {language} with a {comprehension} comprehension level. "
        "Include links in the response with detailed lengthy response content." 
        "Include the source of the content in the response."
    )
    
    return query

def main():
    """Main function to run the script."""
    if not chat:
        print("Error: Perplexity chat is not initialized. Please check your API key and dependencies.")
        return

    # Generate the query directly from user prompts
    meaningful_query = get_query_from_user_input()
    
    if meaningful_query:
        print("\n--- Generated Query ---")
        print(meaningful_query)
        print("-----------------------\n")
        
        print("--- Model Response ---")
        try:
            # Stream the response from the model using the generated query
            full_response = ""
            for chunk in chat.stream(meaningful_query):
                print(chunk.content, end="", flush=True)
                full_response += chunk.content
            
            if not full_response.strip():
                print("\nReceived an empty response from the API.")

        except Exception as e:
            print(f"\nAn error occurred while communicating with the API: {e}")

if __name__ == "__main__":
    main()