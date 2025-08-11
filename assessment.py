import os
import asyncio  # ASYNC: Imported asyncio
# from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
# openai_api_key = os.getenv("OPENAI_API_KEY")
google_api_key= os.getenv("GOOGLE_API_KEY")
SYSTEM_PROMPT = """
You are an expert AI assistant specialized in creating educational materials. Your task is to generate a set of test questions based on the user-provided schema.

Please adhere to the following specifications:
- **Role:** Act as an experienced teacher designing a test for your students.
- **Tone:** The tone should be professional, clear, and appropriate for the specified grade level.
- **Accuracy:** All questions must be factually accurate and directly relevant to the provided topic.

**Test Generation Schema:**
- **Test Title:** {test_title}
- **Grade Level:** {grade_level}
- **Subject:** {subject}
- **Topic:** {topic}
- **Assessment Type:** {assessment_type}
- **Test Duration:** {test_duration}
- **Number of Questions:** {number_of_questions}
- **Difficulty Level:** {difficulty_level}
- **User-Specific Instructions:** {user_prompt}

**Crucial Instructions:**
- **Priority of Instructions:** In the event of a conflict between the `User-Specific Instructions` and the rules below, you **must** prioritize these Crucial Instructions to ensure the output format and integrity are maintained.
- **Strictly Adhere to Assessment Type:** You must *only* generate questions that match the specified `{assessment_type}`. For instance, if 'MCQ' is requested, all questions must be Multiple Choice Questions. Do not mix formats.
- **Separate Questions and Answers:** The entire output must be structured into two distinct parts: the questions first, followed by a clearly marked answer key.

**Output Formatting Rules:**
1.  **Generate Questions:**
    - First, generate the exact number of questions requested.
    - For 'MCQ' type, provide four options labeled A, B, C, and D.
    - For 'True or False' type, provide a clear statement.
    - For 'Short Answer' type, ask a clear question.
    - **Do not** provide the answer immediately after a question.

2.  **Generate the Answer Key:**
    - After listing all the questions, add a separator and a heading for the answers, formatted exactly like this:
---
**Solutions**
    - Below this heading, list each question number and its corresponding correct answer.
    - Example for 'MCQ': `1. C`
    - Example for 'True or False': `1. True`
    - Example for 'Short Answer': `1. The Treaty of Paris.`

- **Final Output:** The final output should contain *only* the generated questions and the separate answer key section. Do not include any other text, introductory phrases, or explanations.
"""

def create_question_generation_chain(google_api_key: str, model_name: str = "gemini-2.5-flash"):
    """
    Creates the LangChain model using LangChain Expression Language (LCEL).
    This function remains synchronous as it's for setup, not I/O.
    """
    if not google_api_key:
        raise ValueError("google_api_key is not provided. Please provide a valid key.")
    prompt_template = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
    # model = ChatOpenAI(model=model_name, temperature=0.7, openai_api_key=openai_api_key)
    model = ChatGoogleGenerativeAI(model=model_name, temperature=0.7, google_api_key=google_api_key)
    output_parser = StrOutputParser()
    chain = prompt_template | model | output_parser
    return chain

# ASYNC: Converted the generation function to be asynchronous
async def generate_test_questions_async(chain, schema: dict):
    """
    Invokes the provided chain asynchronously to generate test questions.

    Args:
        chain: The LangChain runnable sequence.
        schema (dict): A dictionary containing the test specifications.

    Returns:
        The generated text content.
    """
    # ASYNC: Using 'ainvoke' for a non-blocking API call
    return await chain.ainvoke(schema)

def get_user_input_from_terminal():
    """
    Prompts the user to enter the test specifications in the terminal.
    This remains synchronous as input() is a blocking operation.
    """
    print("--- Create Your Custom Test (CLI Mode) ---")
    schema = {
        "test_title": input("Enter the title of the test: "),
        "grade_level": input("Enter the grade or class (e.g., '10th Grade'): "),
        "subject": input("Enter the subject (e.g., 'History'): "),
        "topic": input("Enter the specific topic (e.g., 'The Indian Revolution'): "),
        "assessment_type": input("Enter assessment type (MCQ, True or False, Short Answer): "),
        "test_duration": input("Enter the test duration (e.g., '45 minutes'): "),
        "number_of_questions": int(input("Enter the number of questions: ")),
        "difficulty_level": input("Enter the difficulty level (Easy, Medium, Hard): "),
        "user_prompt": input("Enter optional instructions (or press Enter to skip): ")
    }
    if not schema["user_prompt"]:
        schema["user_prompt"] = "None."
    return schema

# ASYNC: Converted the main CLI function to be asynchronous
async def main_cli_async():
    """
    Main async function to run the question generation model from the command line.
    """
    load_dotenv()
    try:
        google_api_key = os.environ.get("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it in a .env file.")

        question_chain = create_question_generation_chain(google_api_key) 
        user_schema = get_user_input_from_terminal()
        
        print("\n" + "="*50)
        print("Generating questions based on your specifications. Please wait...")
        print("="*50 + "\n")
        
        # ASYNC: Awaiting the asynchronous generation function
        generated_content = await generate_test_questions_async(question_chain, user_schema)
        
        print("--- Generated Test Questions ---")
        print(generated_content)
        print("--- End of Test ---")
        
    except ValueError as ve:
        print(f"Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# ASYNC: Main execution block to run the async CLI
if __name__ == "__main__":
    # ASYNC: Using asyncio.run to start the main async function
    asyncio.run(main_cli_async())