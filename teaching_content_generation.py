import os
import logging
import asyncio
from dotenv import load_dotenv

# LangChain components
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Import from your websearch module (using the new Perplexity search)
from websearch_code import PerplexityWebSearchTool

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#langsmith
LANGSMITH_TRACING="true"
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY=os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT="Vamshi-test"


# Load environment variables from .env file
load_dotenv()

# --- Prompt Engineering ---
PROMPT_TEMPLATE = """
You are an expert AI instructional designer and a world-class {subject} teacher. Your primary task is to generate exceptionally detailed, comprehensive, and ready-to-use teaching content based on the user's precise specifications. Your output must be so thorough that a substitute teacher could use it effectively with no prior preparation. The content you generate must be the complete, final product, not a summary or a set of instructions for a teacher to follow.

**Content Goal:** Generate a "{content_type}".

**Content Configuration:**
- **Subject:** {subject}
- **Lesson Topic:** {lesson_topic}
- **Grade Level:** {grade}
- **Learning Objective:** {learning_objective}
- **Emotional Considerations:** {emotional_consideration}
- **Instructional Depth:** {instructional_depth}
- **Content Version:** {content_version}

**Core Directives:**
- **Absolute Completeness & Verbatim Content:** The generated output MUST be a complete, stand-alone resource. This means you will write out the **full, unabridged text** for all parts of the content. For example, do not just write "Teacher explains photosynthesis"; instead, you must write the **exact, word-for-word script** of that explanation: "Photosynthesis is the process plants use to convert light energy into chemical energy...". A teacher should need no other materials and should have to do no additional writing.
- **Deep Elaboration & Full Detail:** You must provide rich, fully-written descriptions and detailed, verbatim instructions. All examples, concepts, and activities must be fully elaborated with maximum clarity and completeness. Brevity is not acceptable. You are to generate the entire content, not just an outline or a procedural guide.
- **Integrate All Parameters:** Every configuration setting provided above must be clearly and thoughtfully integrated into the final output. The {grade} level should dictate the language and complexity of the complete content. The {emotional_consideration} must shape the tone and the fully-written examples. The {instructional_depth} and {content_version} must define the level of detail in the final text.

**Additional AI Options:**
{additional_ai_options_instructions}
 - **Adaptive Difficulty:** The generated content, especially assessments or activities, should offer varying levels of challenge. For example, include both foundational and advanced questions. If creating a worksheet, you might have a 'Getting Started' section and a 'Challenge Problems' section. This allows the teacher to tailor the experience to individual student needs.
 - **Include Assessment:** You must create and include a distinct assessment component (e.g., a quiz, a set of discussion questions, a rubric for a project) that directly measures the specified learning objective. The assessment should be integrated into the content.
 - **Multimedia Suggestions:** You must include a section with suggestions for relevant multimedia resources. This should include at least one recommended YouTube video (with a full URL) and a description of a relevant image or diagram (with a URL if possible). These suggestions should directly support the lesson topic."

**Web Search Context:**
{web_context}

---

**Output Structure and Citation Mandate:**
You MUST structure your output according to the requested "{content_type}". Adherence to this structure is mandatory, and every section must contain **complete, fully written content**. You MUST use the information from the 'Web Search Context' to ensure your content is factually accurate and up-to-date. At the end of your generation, you MUST include a section that lists all the source URLs provided in the context.

- If the content type is a **"lesson plan"**, it must include all of the following sections, in this order: Title, Estimated Duration, Learning Objectives, Materials, a **highly detailed, verbatim Step-by-Step Procedure with complete content for every step** (this means writing out the full teacher script, all explanations, all questions to ask students, and the complete text of examples or stories), a fully developed Assessment/Check for Understanding, Differentiation strategies with ready-to-use alternative explanations or tasks, and a **"References"** section listing all source URLs from the web search.
- If the content type is a **"presentation"**, it must be structured as a series of detailed slides. Each slide needs a clear title, the **complete and full text content** for the slide body (not just bullet points), and extensive, **verbatim speaker notes** that a presenter could read word-for-word. It **must end with a "Bibliography" slide** listing all source URLs from the web search.
- If the content type is a **"worksheet"** or **"quiz"**, it must be a complete and ready-to-distribute document. This includes clear, detailed instructions for the student, a variety of fully-formed question types, any and all **reading passages, data sets, or background information** needed to answer the questions included directly in the document, a comprehensive answer key with full explanations for each answer, and a **"Sources" section** at the end listing all source URLs from the web search.

---

**Your Task:**
Please generate the requested "{content_type}" now. You MUST strictly adhere to all configurations and structural requirements detailed above. Based on the generated content, you MUST determine and specify an appropriate duration (e.g., 45 minutes, 1 hour). The generated content must be **exceptionally detailed, containing the complete and unabridged text and materials, making it directly usable by a teacher with absolutely no further writing or content creation required.**
"""

def _get_choice_from_user(options: list[str], prompt_text: str, default: str | None = None) -> str:
    """
    A robust helper function to get a choice from a list of options from the user.
    It allows for selection by number, exact name, or unique prefix (all case-insensitive).
    """
    while True:
        default_info = f" (default: {default})" if default else ""
        choice_str = input(f"   {prompt_text}{default_info}: ").lower().strip()

        if not choice_str and default:
            return default

        # 1. Check for numeric choice
        if choice_str.isdigit() and 1 <= int(choice_str) <= len(options):
            return options[int(choice_str) - 1]

        # 2. Check for exact match (case-insensitive)
        lower_options = [opt.lower() for opt in options]
        if choice_str in lower_options:
            return options[lower_options.index(choice_str)]

        # 3. Check for unique prefix match (case-insensitive)
        matches = [opt for opt in options if opt.lower().startswith(choice_str)]
        if len(matches) == 1:
            print(f"   --> Interpreted '{choice_str}' as '{matches[0]}'.")
            return matches[0]

        # 4. If no valid choice, print error and loop
        print("   Invalid choice. Please enter the full name, a unique starting part of the name, or the corresponding number.")


def get_user_input() -> dict:
    """
    Interactively collects content generation parameters from the user in the terminal.
    """
    print("--- Teaching Content Generation Model ---")

    # --- Content Type ---
    print("\n1. Choose Content Type:")
    content_types = ["lesson plan", "worksheet", "presentation", "quiz"]
    for i, ct in enumerate(content_types, 1):
        print(f"   {i}. {ct.title()}")
    content_type = _get_choice_from_user(content_types, f"Enter name or number (1-{len(content_types)})")

    # --- Content Configuration ---
    print("\n2. Configure Content:")
    subject = input("   - Subject (e.g., Physics, History): ")
    lesson_topic = input("   - Lesson Topic (e.g., Newton's Laws of Motion): ")
    grade = input("   - Grade Level (e.g., 10th Grade): ")
    learning_objective = input("   - Learning Objective (optional, press Enter to skip): ") or "Not specified"

    # --- Advanced Settings ---
    print("\n3. Advanced Settings:")
    emotional_consideration = input("   - Emotional Considerations (comma-separated, e.g., anxiety, low confidence, Enter to skip): ") or "None"

    depths = ["Basic", "Standard", "Advanced"]
    print("   - Instructional Depth:")
    for i, d in enumerate(depths, 1):
        print(f"     {i}. {d.title()}")
    instructional_depth = _get_choice_from_user(depths, f"Enter name or number (1-{len(depths)})", default="Standard")

    versions = ["Simplified", "Standard", "Enriched"]
    print("   - Content Version:")
    for i, v in enumerate(versions, 1):
        print(f"     {i}. {v.title()}")
    content_version = _get_choice_from_user(versions, f"Enter name or number (1-{len(versions)})", default="Standard")

    # --- Additional AI Options ---
    print("\n4. Additional AI Options (optional):")
    ai_options = ["adaptive difficulty", "include assessment", "multimedia suggestion"]
    print(f"   Choose any of the following, separated by commas: {', '.join([opt.title() for opt in ai_options])}")
    selected_options_str = input("   Enter your choices (or press Enter to skip): ").lower().strip()
    
    selected_ai_options = []
    if selected_options_str:
        user_choices = [choice.strip() for choice in selected_options_str.split(',')]
        
        for choice in user_choices:
            # Use a case-insensitive check for exact matches
            lower_ai_options = [o.lower() for o in ai_options]
            if choice in lower_ai_options:
                original_option = ai_options[lower_ai_options.index(choice)]
                if original_option not in selected_ai_options:
                    selected_ai_options.append(original_option)
                    print(f"   --> Added '{original_option.title()}'.")
            else:
                # Use a case-insensitive check for prefixes
                matches = [opt for opt in ai_options if opt.lower().startswith(choice)]
                if len(matches) == 1:
                    if matches[0] not in selected_ai_options:
                        selected_ai_options.append(matches[0])
                        print(f"   --> Interpreted '{choice}' as '{matches[0].title()}'.")
                else:
                    print(f"   --> Invalid or ambiguous choice: '{choice}'. It will be ignored.")

    # --- Web Search is now always enabled ---
    web_search_enabled = True

    return {
        "content_type": content_type,
        "subject": subject,
        "lesson_topic": lesson_topic,
        "grade": grade,
        "duration": "To be determined by AI",
        "learning_objective": learning_objective,
        "emotional_consideration": emotional_consideration,
        "instructional_depth": instructional_depth,
        "content_version": content_version,
        "web_search_enabled": web_search_enabled,
        "additional_ai_options": selected_ai_options
    }

async def run_generation_pipeline_async(config: dict):
    """
    Constructs and runs the LCEL pipeline for content generation asynchronously.
    Returns the generated content instead of just printing it.
    """
    logger.info("Initializing Model and Tools for content generation")

    # --- LLM Initialization Block ---
    try:
        logger.info("Initializing local OpenAI LLM: gpt-4o")
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.5,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    except Exception as e:
        logger.error(f"Fatal: Could not initialize the OpenAI LLM. Error: {e}")
        raise Exception(f"Failed to initialize OpenAI LLM: {e}")

    # --- Process Additional AI Options ---
    additional_options = config.get("additional_ai_options", [])
    options_instructions = []
    if not additional_options:
        options_instructions.append("- No additional AI options were selected.")

    if "adaptive difficulty" in additional_options:
        options_instructions.append(
            "- **Adaptive Difficulty:** The generated content, especially assessments or activities, should offer varying levels of challenge. For example, include both foundational and advanced questions. If creating a worksheet, you might have a 'Getting Started' section and a 'Challenge Problems' section. This allows the teacher to tailor the experience to individual student needs."
        )

    if "include assessment" in additional_options:
        options_instructions.append(
            "- **Include Assessment:** You must create and include a distinct assessment component (e.g., a quiz, a set of discussion questions, a rubric for a project) that directly measures the specified learning objective. The assessment should be integrated into the content."
        )

    if "multimedia suggestion" in additional_options:
        options_instructions.append(
            "- **Multimedia Suggestions:** You must include a section with suggestions for relevant multimedia resources. This should include at least one recommended YouTube video (with a full URL) and a description of a relevant image or diagram (with a URL if possible). These suggestions should directly support the lesson topic."
        )
    
    config['additional_ai_options_instructions'] = "\n".join(options_instructions)


    web_context = "No web search was performed for this generation."

    if config.get("web_search_enabled", True):
        logger.info("Web search is enabled. Fetching latest content...")
        try:
            # Updated to use Perplexity
            search_tool = PerplexityWebSearchTool(max_results=5, model="sonar")
            search_query = (
                f"Teaching resources and ideas for a {config['grade']} {config['subject']} "
                f"{config['content_type']} on '{config['lesson_topic']}'"
            )
            if 'multimedia suggestion' in config.get('additional_ai_options', []):
                search_query += f" including youtube videos and images"
                
            if config.get('learning_objective', '') != 'Not specified':
                search_query += f" with the learning objective: '{config['learning_objective']}'"

            results = await search_tool.search(query=search_query)

            if results:
                web_context = "Web search has been performed. Use the following latest information and source URLs to enrich your content:\n\n"
                for result in results:
                    web_context += result["content"] + "\n\n"
                logger.info("Web search completed and context created with results.")
            else:
                web_context = "Web search was enabled but returned no relevant results. Proceed with general knowledge."
                logger.warning(f"Web search for query '{search_query}' returned no results.")
        except Exception as e:
            logger.error(f"An error occurred during the web search process: {e}")
            web_context = f"Web search was enabled but failed with an error: {e}."

    config['web_context'] = web_context

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    output_parser = StrOutputParser()
    
    # This is the LangChain Expression Language (LCEL) chain
    chain = prompt | llm | output_parser

    logger.info("Generating content with AI...")

    try:
        response = await chain.ainvoke(config)
        logger.info("Content generated successfully.")
        return response
    except Exception as e:
        logger.error(f"Error during content generation: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("PPLX_API_KEY"):
        print("FATAL ERROR: Make sure you have created a .env file with your OPENAI_API_KEY and PPLX_API_KEY.")
    else:
        user_config = get_user_input()
        # To see the output in the terminal, we need to print the returned response
        response = asyncio.run(run_generation_pipeline_async(user_config))
        print("\n--- Generated Content ---")
        print(response)