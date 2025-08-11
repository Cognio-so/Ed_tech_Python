import os
import uuid
import logging
from typing import List, Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from fastapi.concurrency import run_in_threadpool

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
load_dotenv()

# --- Import functionalities from your scripts ---

# Chatbot imports
from AI_tutor import AsyncRAGTutor, RAGTutorConfig

# Assessment generation imports
from assessment import create_question_generation_chain, generate_test_questions_async

# Teaching content generation imports
from teaching_content_generation import run_generation_pipeline_async as generate_teaching_content

# Media toolkit imports
from media_toolkit.slides_generation import SlideSpeakGenerator
from media_toolkit.image_generation_model import ImageGenerator
from media_toolkit.comics_generation import create_comical_story_prompt, generate_comic_image

# Import the Perplexity chat instance from your module
try:
    from media_toolkit.websearch_schema_based import chat as pplx_chat
except Exception as e:
    pplx_chat = None
    logger.warning(f"Perplexity chat not initialized: {e}")

# --- FastAPI App Initialization ---
logger.info("Starting AI Education Platform API...")
app = FastAPI(
    title="AI Education Platform API",
    description="An AI-powered tools for tutoring, assessment creation, and teaching content generation.",
    version="1.0.0"
)

# --- Add CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://ed-tech-dusky-two.vercel.app"],  # Add your frontend URLs
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# --- In-memory Storage for Simplicity ---
# In a production environment, this should be replaced with a proper file storage solution like S3 or a persistent disk.
class SimpleInMemoryStorage:
    """A simple in-memory storage manager to mock file storage for the AI Tutor."""
    def __init__(self):
        self._storage: Dict[str, bytes] = {}
        self.temp_dir = "temp_uploads"
        os.makedirs(self.temp_dir, exist_ok=True)

    async def save_file_async(self, file: UploadFile) -> str:
        """Saves an uploaded file to a temporary local path and returns the path."""
        # Ensure filename is secure and unique to prevent conflicts
        safe_filename = f"{uuid.uuid4()}_{file.filename}"
        file_path = os.path.join(self.temp_dir, safe_filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        logger.info(f"File '{file.filename}' saved to temporary path: {file_path}")
        return file_path

    async def get_file_content_bytes_async(self, storage_key: str) -> Optional[bytes]:
        """Reads file content from the temporary local path."""
        if os.path.exists(storage_key):
            with open(storage_key, "rb") as f:
                return f.read()
        logger.error(f"File not found at storage key: {storage_key}")
        return None

# --- Global Objects and Initializations ---
logger.info("Initializing global components...")

try:
    # Initialize Storage Manager
    storage_manager = SimpleInMemoryStorage()
    logger.info("✅ In-memory storage manager initialized successfully.")

    # Initialize Tutor Sessions Dictionary
    tutor_sessions: Dict[str, AsyncRAGTutor] = {}
    logger.info("✅ AI Tutor session manager is ready.")

    # Initialize Assessment Generation Chain
    google_api_key = os.getenv("GOOGLE_API_KEY")
    logger.info(f"GOOGLE_API_KEY: {google_api_key}")
    if not google_api_key:
        logger.warning("GOOGLE_API_KEY not found. Assessment endpoint may fail.")
        assessment_chain = None
    else:
        assessment_chain = create_question_generation_chain(google_api_key)
        logger.info("✅ Assessment generation chain created successfully.")
    
    # Check SlideSpeak API Key
    slidespeak_api_key = os.getenv("SLIDESPEAK_API_KEY")
    if not slidespeak_api_key:
        logger.warning("SLIDESPEAK_API_KEY not found. Presentation endpoint may fail.")
    else:
        logger.info("✅ SlideSpeak API key is configured.")
    
    # Note: Teaching content generation is initialized on-the-fly, so we confirm its readiness here.
    logger.info("✅ Teaching content generation module is ready.")
    logger.info("All components initialized. The API is ready to accept requests.")

except Exception as e:
    logger.error(f"❌ CRITICAL ERROR during initialization: {e}", exc_info=True)
    # Depending on the desired behavior, you might want to exit the application
    # raise SystemExit(f"Failed to initialize critical components: {e}")

# ==============================================================================
# HEALTH CHECK ENDPOINT
# ==============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint to verify the API is running."""
    return {
        "status": "healthy",
        "message": "AI Education Platform API is running",
        "version": "1.0.0"
    }

# ==============================================================================
# 1. CHATBOT ENDPOINT
# ==============================================================================

class ChatbotRequest(BaseModel):
    session_id: str = Field(..., description="A unique identifier for the chat session. This maintains the context and knowledge base for the user.")
    query: Optional[str] = Field(None, description="The user's text query to the chatbot.")
    history: List[Dict[str, Any]] = Field([], description="A list of previous messages in the chat history.")
    web_search_enabled: bool = Field(False, description="Enable or disable web search functionality for the tutor.")

@app.post("/chatbot_endpoint")
async def chatbot_endpoint(
    files: Optional[List[UploadFile]] = File(None, description="Optional list of files (documents, images) to upload to the knowledge base."),
    request: ChatbotRequest = Body(...)
):
    """
    Handles interactions with the AI tutor. It supports file uploads for
    building a knowledge base and conversational queries with streaming responses.
    A `session_id` must be provided to maintain conversation state.
    """
    session_id = request.session_id
    
    # Get or create a tutor instance for the session
    if session_id not in tutor_sessions:
        # This log confirms a new user/session is starting
        logger.info(f"Creating new AI Tutor session: {session_id}")
        tutor_config = RAGTutorConfig.from_env()
        tutor_sessions[session_id] = AsyncRAGTutor(storage_manager=storage_manager, config=tutor_config)
    
    tutor = tutor_sessions[session_id]

    # Dynamically update web search status for the tutor
    tutor.update_web_search_status(request.web_search_enabled)

    # --- File Ingestion Logic ---
    if files:
        storage_keys = []
        filenames = []
        for file in files:
            if not file.filename:
                continue
            storage_key = await storage_manager.save_file_async(file)
            storage_keys.append(storage_key)
            filenames.append(file.filename)
        
        if storage_keys:
            logger.info(f"Ingesting {len(storage_keys)} files for session {session_id}...")
            # Initialize the vectorstore and ingest documents
            await tutor.ingest_async(storage_keys)
            
            # Since this is a file upload, we return a confirmation message, not a stream.
            return {
                "message": "Files uploaded and processed successfully.",
                "uploaded_files": filenames
            }

    # --- Query Processing Logic ---
    if not request.query:
        raise HTTPException(status_code=400, detail="A 'query' is required when no files are uploaded.")

    # Generate the response as a stream
    is_kb_ready = tutor.ensemble_retriever is not None
    response_generator = tutor.run_agent_async(
        query=request.query,
        history=request.history,
        is_knowledge_base_ready=is_kb_ready
    )
    
    return StreamingResponse(response_generator, media_type="text/plain")


# ==============================================================================
# 2. ASSESSMENT ENDPOINT
# ==============================================================================

class AssessmentSchema(BaseModel):
    test_title: str = Field(..., description="The title of the test.", example="The American Revolution")
    grade_level: str = Field(..., description="The target grade or class for the test.", example="8th Grade")
    subject: str = Field(..., description="The subject of the test.", example="History")
    topic: str = Field(..., description="The specific topic the test will cover.", example="Key Battles of the Revolutionary War")
    assessment_type: str = Field(..., description="The type of questions to generate or 'Mixed' for multiple types.", example="MCQ")
    question_types: Optional[List[str]] = Field(None, description="List of question types when using mixed assessments.", example=["mcq", "true_false"])
    question_distribution: Optional[Dict[str, int]] = Field(None, description="Distribution of questions by type.", example={"mcq": 6, "true_false": 2, "short_answer": 2})
    test_duration: str = Field(..., description="The estimated duration for completing the test.", example="30 minutes")
    number_of_questions: int = Field(..., description="The exact number of questions to generate.", example=10)
    difficulty_level: str = Field(..., description="The difficulty level of the questions.", example="Medium", pattern="^(Easy|Medium|Hard)$")
    learning_objectives: Optional[str] = Field("", description="Learning objectives for the assessment.")
    anxiety_triggers: Optional[str] = Field("", description="Anxiety considerations to account for.")
    user_prompt: Optional[str] = Field("None.", description="Optional specific instructions for the AI.", example="Focus on the strategic importance of each battle.")

@app.post("/assessment_endpoint", response_model=Dict[str, Any])
async def assessment_endpoint(schema: AssessmentSchema):
    """
    Generates a set of test questions based on the provided schema.
    The response will contain the formatted questions and a separate answer key.
    Supports both single question type and mixed question type assessments.
    """
    if not assessment_chain:
        logger.error("Assessment endpoint called, but chain is not initialized (likely missing or invalid GOOGLE_API_KEY).")
        raise HTTPException(status_code=500, detail="Assessment generation is not configured on the server.")
    
    try:
        # Convert the schema to dict for processing
        schema_dict = schema.model_dump()
        
        # Validate and process mixed question types
        if schema.assessment_type == "Mixed" and schema.question_types and schema.question_distribution:
            # Validate that the distribution sums to the total number of questions
            total_distributed = sum(schema.question_distribution.values())
            if total_distributed != schema.number_of_questions:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Question distribution ({total_distributed}) does not match total questions ({schema.number_of_questions})"
                )
            
            logger.info(f"Generating mixed assessment for topic: {schema.topic}")
            logger.info(f"Question distribution: {schema.question_distribution}")
        else:
            logger.info(f"Generating {schema.assessment_type} assessment for topic: {schema.topic}")
        
        generated_content = await generate_test_questions_async(assessment_chain, schema_dict)
        return {"assessment": generated_content}
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error in assessment generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# 3. TEACHING CONTENT ENDPOINT
# ==============================================================================

class TeachingContentSchema(BaseModel):
    content_type: str = Field(
        ...,
        description="The type of teaching material to create.",
        example="lesson plan",
        pattern="^(lesson plan|worksheet|presentation|quiz)$"
    )
    subject: str = Field(..., description="The subject of the content.", example="Biology")
    lesson_topic: str = Field(..., description="The specific topic for the lesson.", example="Cellular Respiration")
    grade: str = Field(..., description="The target grade level for the content.", example="10th Grade")
    learning_objective: Optional[str] = Field(
        "Not specified",
        description="The specific learning objective for this content."
    )
    emotional_consideration: Optional[str] = Field(
        "None",
        description="Emotional factors to consider for students (e.g., anxiety)."
    )
    # Accept both old (low/high) and new (basic/advanced) forms, case-insensitive
    instructional_depth: str = Field(
        "standard",
        description="The level of detail and complexity.",
        pattern="(?i)^(low|standard|high|basic|advanced)$"
    )
    # Accept both old (low/high) and new (simplified/enriched), case-insensitive
    content_version: str = Field(
        "standard",
        description="The version of the content.",
        pattern="(?i)^(low|standard|high|simplified|enriched)$"
    )
    web_search_enabled: bool = Field(False, description="Enable web search to fetch up-to-date information.")
    # New: Forward additional AI options used by the generator module
    additional_ai_options: Optional[List[str]] = Field(
        default=None,
        description="List of AI options: 'adaptive difficulty', 'include assessment', 'multimedia suggestion'"
    )

@app.post("/teaching_content_endpoint", response_model=Dict[str, Any])
async def teaching_content_endpoint(schema: TeachingContentSchema):
    """
    Generates detailed teaching content based on the provided specifications.
    This can create lesson plans, worksheets, presentations, or quizzes,
    optionally enhanced with real-time web search results.
    """
    try:
        # Use model_dump() for Pydantic v2+ to avoid deprecation warnings
        config = schema.model_dump()
        logger.info(f"Generating teaching content: {config['content_type']} on {config['lesson_topic']}")
        
        generated_content = await generate_teaching_content(config)
        
        # Add a check to ensure the generated content is valid before returning
        if not generated_content or not isinstance(generated_content, str):
            logger.error(f"Content generation returned an empty or invalid result. Got: {generated_content}")
            raise HTTPException(status_code=500, detail="AI content generation returned an empty or invalid result.")
        
        logger.info(f"Successfully generated content of length {len(generated_content)}.")
        return {"generated_content": generated_content}

    except Exception as e:
        logger.error(f"Error in teaching content endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

# ==============================================================================
# 4. PRESENTATION ENDPOINT
# ==============================================================================

class PresentationSchema(BaseModel):
    plain_text: str = Field(..., description="The main topic or content of the presentation.", example="Introduction to Machine Learning")
    custom_user_instructions: str = Field("", description="Specific instructions for the AI.", example="Focus on practical applications")
    length: int = Field(..., description="The desired number of slides.", example=10, ge=1, le=50)
    language: str = Field("ENGLISH", description="The language of the presentation.", example="ENGLISH", pattern="^(ENGLISH|ARABIC)$")
    fetch_images: bool = Field(True, description="Whether to include stock images in the presentation.")
    verbosity: str = Field("standard", description="The desired text verbosity.", example="standard", pattern="^(concise|standard|text-heavy)$")

@app.post("/presentation_endpoint", response_model=Dict[str, Any])
async def presentation_endpoint(schema: PresentationSchema):
    """
    Generates a SlideSpeak presentation based on the provided specifications.
    Returns the complete task result including the presentation URL.
    """
    if not slidespeak_api_key:
        logger.error("Presentation endpoint called, but SLIDESPEAK_API_KEY is not configured.")
        raise HTTPException(status_code=500, detail="Presentation generation is not configured on the server.")
    
    try:
        # Instantiate the generator. It will automatically use the API key from the environment.
        try:
            generator = SlideSpeakGenerator()
        except ValueError as e:
            logger.error(f"Failed to initialize SlideSpeakGenerator: {e}")
            raise HTTPException(status_code=500, detail=str(e))

        logger.info(f"Generating presentation for topic: {schema.plain_text}")
        
        # The generate_presentation method in SlideSpeakGenerator is synchronous (uses requests and time.sleep).
        # To avoid blocking the server's event loop, we run it in a separate thread pool.
        result = await run_in_threadpool(
            generator.generate_presentation,
            plain_text=schema.plain_text,
            custom_user_instructions=schema.custom_user_instructions,
            length=schema.length,
            language=schema.language,
            fetch_images=schema.fetch_images,
            verbosity=schema.verbosity
        )
        
        # Check if the result contains an error key from the generator class
        if "error" in result:
            logger.error(f"SlideSpeak API error: {result['error']}")
            raise HTTPException(status_code=500, detail=f"Presentation generation failed: {result['error']}")
        
        # Check if the task was successful
        if result.get("task_status") == "SUCCESS":
            logger.info(f"Presentation generated successfully. URL: {result.get('task_result', {}).get('url')}")
            return {"presentation": result}
        elif result.get("task_status") == "FAILURE":
            error_msg = result.get("task_result", {}).get("error", "Unknown error occurred")
            logger.error(f"Presentation generation failed: {error_msg}")
            raise HTTPException(status_code=500, detail=f"Presentation generation failed: {error_msg}")
        else:
            logger.error(f"Unexpected task status: {result.get('task_status')}")
            raise HTTPException(status_code=500, detail="Presentation generation returned unexpected status")
        
    except HTTPException:
        # Re-raise HTTP exceptions to be handled by FastAPI
        raise
    except Exception as e:
        logger.error(f"Error in presentation endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

# ==============================
# 5. IMAGE GENERATION ENDPOINT
# ==============================
class ImageGenSchema(BaseModel):
    topic: str = Field(..., description="Topic for the image")
    grade_level: str = Field(..., description="Grade level")
    preferred_visual_type: str = Field(..., description="Visual type, e.g., image/chart/diagram")
    subject: str = Field(..., description="Subject")
    instructions: str = Field(..., description="Detailed instructions")
    difficulty_flag: str = Field("false", description="true/false flag")

@app.post("/image_generation_endpoint", response_model=Dict[str, Any])
async def image_generation_endpoint(schema: ImageGenSchema):
    try:
        generator = ImageGenerator()
        schema_dict = schema.model_dump()
        image_url = generator.generate_image_from_schema(schema_dict)
        if not image_url:
            raise HTTPException(status_code=500, detail="Image generation failed.")
        return {"image_url": image_url}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in image generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ==============================
# 6. WEB SEARCH ENDPOINT
# ==============================
class WebSearchSchema(BaseModel):
    topic: str = Field(..., description="Search topic")
    grade_level: str = Field(..., description="Grade level (e.g., 10)")
    subject: str = Field(..., description="Subject (e.g., History)")
    content_type: str = Field(..., description="Preferred content type (e.g., articles, videos)")
    language: str = Field("English", description="Language")
    comprehension: str = Field("intermediate", description="Comprehension level")
    max_results: int = Field(5, description="Maximum number of results")

@app.post("/web_search_endpoint", response_model=Dict[str, Any])
async def web_search_endpoint(schema: WebSearchSchema):
    if not pplx_chat:
        raise HTTPException(status_code=500, detail="Perplexity client not configured. Check PPLX_API_KEY.")

    try:
        data = schema.model_dump()
        query = (
            f"Show me up to {data['max_results']} {data['content_type']} about '{data['topic']}' "
            f"for a grade {data['grade_level']} {data['subject']} class. "
            f"The content should be in {data['language']} with a {data['comprehension']} comprehension level. "
            "Include links in the response with detailed lengthy response content."
            "Include the source of the content in the response."
        )
        full_response = ""
        for chunk in pplx_chat.stream(query):
            full_response += chunk.content or ""

        if not full_response.strip():
            raise HTTPException(status_code=500, detail="Web search returned empty response.")

        return {
            "query": query,
            "content": full_response
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in web search endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ==============================
# 7. COMICS STREAMING ENDPOINT
# ==============================
class ComicsSchema(BaseModel):
    instructions: str = Field(..., description="Educational story/topic, e.g., Water cycle")
    grade_level: str = Field(..., description="Grade level string, e.g., '5' or 'Grade 5'")
    num_panels: int = Field(..., description="Number of panels to generate", ge=1, le=20)

def _parse_panel_prompts(story_text: str):
    lines = story_text.strip().split("\n")
    panels = []
    for line in lines:
        if "Panel_Prompt:" in line:
            try:
                prompt = line.split("Panel_Prompt:")[1].strip()
                if prompt:
                    panels.append(prompt)
            except Exception:
                continue
    return panels

@app.post("/comics_stream_endpoint")
async def comics_stream_endpoint(schema: ComicsSchema):
    async def event_stream():
        import json
        async def send(obj: dict):
            # SSE event
            yield f"data: {json.dumps(obj)}\n\n"

        try:
            # 1) Generate story/panel prompts
            story_prompts = await run_in_threadpool(
                create_comical_story_prompt,
                schema.instructions,
                schema.grade_level,
                schema.num_panels
            )
            if not story_prompts:
                async for chunk in send({"type": "error", "message": "Failed to generate story prompts."}):
                    yield chunk
                return

            # Send the full story text first
            async for chunk in send({"type": "story_prompts", "content": story_prompts}):
                yield chunk

            # 2) Parse and send each panel prompt, then image URL per panel
            panel_prompts = _parse_panel_prompts(story_prompts)
            if not panel_prompts:
                async for chunk in send({"type": "error", "message": "No panel prompts parsed."}):
                    yield chunk
                return

            for i, prompt in enumerate(panel_prompts[:schema.num_panels]):
                panel_index = i + 1
                # Emit the panel prompt
                async for chunk in send({"type": "panel_prompt", "index": panel_index, "prompt": prompt}):
                    yield chunk

                # Generate panel image synchronously via threadpool to avoid blocking
                image_url = await run_in_threadpool(generate_comic_image, prompt, panel_index)
                async for chunk in send({
                    "type": "panel_image",
                    "index": panel_index,
                    "url": image_url or ""
                }):
                    yield chunk

            # Done
            async for chunk in send({"type": "done"}):
                yield chunk

        except Exception as e:
            logger.error(f"Error in comics stream: {e}", exc_info=True)
            async for chunk in send({"type": "error", "message": str(e)}):
                yield chunk

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Content-Type": "text/event-stream",
        "X-Accel-Buffering": "no",  # for some proxies
    }
    return StreamingResponse(event_stream(), headers=headers, media_type="text/event-stream")

# --- Uvicorn Server Runner ---
if __name__ == "__main__":
    logger.info("Starting Uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)