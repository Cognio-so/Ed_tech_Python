import os
import uuid
import logging
from typing import List, Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Body, WebSocket, WebSocketDisconnect
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

# Voice functionality imports
from AI_voice_functionality import (
    transcribe_audio, 
    get_comprehensive_answer_stream, 
    AI_STUDY_BUDDY_PROMPT
)

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

    # Initialize other components
    slide_generator = SlideSpeakGenerator()
    image_generator = ImageGenerator()
    
    # Initialize assessment chain
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if google_api_key:
        assessment_chain = create_question_generation_chain(google_api_key)
        logger.info("✅ Assessment chain initialized successfully.")
    else:
        assessment_chain = None
        logger.warning("⚠️ Google API key not found. Assessment functionality will be limited.")
    
    logger.info("✅ All global components initialized successfully.")
except Exception as e:
    logger.error(f"❌ Error initializing global components: {e}", exc_info=True)
    raise

# ==============================
# 1. HEALTH CHECK ENDPOINT
# ==============================
@app.get("/health")
async def health_check():
    """Health check endpoint to verify the API is running."""
    return {
        "status": "healthy",
        "message": "AI Education Platform API is running",
        "timestamp": "2024-01-01T00:00:00Z"
    }

# ==============================
# 2. VOICE FUNCTIONALITY ENDPOINTS
# ==============================

class VoiceTranscriptionSchema(BaseModel):
    """Schema for voice transcription requests."""
    pass

class VoiceChatSchema(BaseModel):
    """Schema for voice chat requests."""
    text: str = Field(..., description="User's text input")
    web_search_enabled: bool = Field(False, description="Whether to enable web search")

@app.post("/voice_transcription_endpoint")
async def voice_transcription_endpoint(audio_file: UploadFile = File(...)):
    """Transcribe audio to text using OpenAI's Whisper model."""
    try:
        logger.info(f"Processing voice transcription for file: {audio_file.filename}")
        
        temp_audio_path = await storage_manager.save_file_async(audio_file)
        import io
        audio_bytes = await storage_manager.get_file_content_bytes_async(temp_audio_path)
        audio_io = io.BytesIO(audio_bytes)
        audio_io.name = audio_file.filename
        
        transcription = await run_in_threadpool(transcribe_audio, audio_io)
        
        try:
            os.remove(temp_audio_path)
        except:
            pass
        
        if transcription:
            return {
                "success": True,
                "transcription": transcription,
                "message": "Audio transcribed successfully"
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to transcribe audio")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in voice transcription endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Add this new endpoint for real-time voice interaction
@app.post("/voice_realtime_endpoint")
async def voice_realtime_endpoint():
    """
    Real-time voice interaction endpoint that uses the actual voice functionality.
    This streams audio in real-time like ChatGPT.
    """
    async def event_stream():
        import json
        import asyncio
        from AI_voice_functionality import (
            transcribe_audio, 
            get_comprehensive_answer_stream,
            text_to_speech,
            AI_STUDY_BUDDY_PROMPT
        )
        
        async def send(obj: dict):
            yield f"data: {json.dumps(obj)}\n\n"

        try:
            # This simulates the real-time voice interaction
            # In a real implementation, you'd stream audio chunks here
            async for part in send({"type": "voice_ready", "message": "Voice system ready"}):
                yield part

        except Exception as e:
            logger.error(f"Error in real-time voice stream: {e}", exc_info=True)
            async for part in send({"type": "error", "message": str(e)}):
                yield part

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Content-Type": "text/event-stream",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_stream(), headers=headers, media_type="text/event-stream")

# Update the voice chat endpoint to use real-time processing
@app.post("/voice_chat_endpoint")
async def voice_chat_endpoint(schema: VoiceChatSchema):
    """
    Real-time voice chat using the actual voice functionality.
    """
    async def event_stream():
        import json
        from AI_voice_functionality import get_comprehensive_answer_stream
        
        async def send(obj: dict):
            yield f"data: {json.dumps(obj)}\n\n"

        try:
            full_response = ""
            
            # Use the real voice functionality for streaming responses
            for chunk in get_comprehensive_answer_stream(schema.text):
                if not chunk:
                    continue
                full_response += chunk
                # Stream text in real-time
                async for part in send({"type": "text_chunk", "content": chunk}):
                    yield part

            async for part in send({"type": "done"}):
                yield part

        except Exception as e:
            logger.error(f"Error in real-time voice chat: {e}", exc_info=True)
            async for part in send({"type": "error", "message": str(e)}):
                yield part

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Content-Type": "text/event-stream",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_stream(), headers=headers, media_type="text/event-stream")

# Add WebSocket support for real-time voice with proper user control
from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import numpy as np
import io
import wave
import base64
import tempfile
import os

@app.websocket("/ws/voice")
async def websocket_voice_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Initialize voice processing components
    from AI_voice_functionality import (
        transcribe_audio, 
        get_comprehensive_answer_stream,
        text_to_speech,
        AI_STUDY_BUDDY_PROMPT
    )
    
    # Audio buffer with proper user control
    audio_buffer = []
    is_processing = False
    is_listening = False
    user_activated = False  # Only process when user explicitly activates
    
    try:
        while True:
            # Receive real-time audio chunks
            data = await websocket.receive_json()
            
            if data["type"] == "start_listening":
                is_listening = True
                user_activated = True  # User has activated voice mode
                audio_buffer = []  # Clear buffer
                await websocket.send_json({
                    "type": "listening_started",
                    "message": "Ready to listen"
                })
                
            elif data["type"] == "audio_chunk":
                if not is_listening or is_processing or not user_activated:
                    continue
                    
                # Convert audio data to proper format
                audio_chunk = np.array(data["data"], dtype=np.float32)
                
                # Only buffer audio when user is speaking AND voice is activated
                if data.get("is_speaking", False):
                    audio_buffer.extend(audio_chunk)
                
            elif data["type"] == "process_audio":
                # Only process if user has activated voice mode
                if not user_activated or is_processing or len(audio_buffer) == 0:
                    continue
                    
                is_processing = True
                is_listening = False
                
                # Send processing status
                await websocket.send_json({
                    "type": "processing",
                    "message": "Processing speech..."
                })
                
                try:
                    # Convert buffer to audio file format for OpenAI Whisper
                    audio_data = np.array(audio_buffer, dtype=np.float32)
                    audio_buffer = []  # Clear buffer
                    
                    # Only process if we have enough audio (at least 1 second)
                    if len(audio_data) < 16000:  # Less than 1 second
                        logger.info("Audio too short, ignoring")
                        await websocket.send_json({
                            "type": "error",
                            "message": "Audio too short to process"
                        })
                        continue
                    
                    # Create a proper WAV file that OpenAI can recognize
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                        # Create WAV file with proper headers
                        with wave.open(temp_file.name, 'wb') as wf:
                            wf.setnchannels(1)  # Mono
                            wf.setsampwidth(2)  # 16-bit
                            wf.setframerate(16000)  # 16kHz
                            # Convert float32 to int16 properly
                            audio_int16 = (audio_data * 32767).astype(np.int16)
                            wf.writeframes(audio_int16.tobytes())
                    
                    logger.info(f"Processing audio: {len(audio_data)} samples, saved to {temp_file.name}")
                    
                    # Open the file for transcription
                    with open(temp_file.name, 'rb') as audio_file:
                        # Transcribe the audio using OpenAI Whisper
                        transcription = await asyncio.get_event_loop().run_in_executor(
                            None, transcribe_audio, audio_file
                        )
                    
                    # Clean up temporary file
                    try:
                        os.unlink(temp_file.name)
                    except:
                        pass
                    
                    if transcription and transcription.strip():
                        logger.info(f"Transcription: {transcription}")
                        
                        # Send transcription back
                        await websocket.send_json({
                            "type": "transcription",
                            "text": transcription
                        })
                        
                        # Generate AI response
                        response_stream = get_comprehensive_answer_stream(transcription)
                        full_response = ""
                        
                        for chunk in response_stream:
                            if chunk:
                                full_response += chunk
                                await websocket.send_json({
                                    "type": "response_chunk",
                                    "content": chunk
                                })
                        
                        # Generate TTS audio using OpenAI TTS
                        try:
                            logger.info(f"Generating TTS for response: {full_response[:100]}...")
                            audio_data = await asyncio.get_event_loop().run_in_executor(
                                None, text_to_speech, full_response
                            )
                            
                            if audio_data:
                                logger.info(f"TTS generated successfully, audio size: {len(audio_data)} bytes")
                                # Encode audio as base64 for sending over WebSocket
                                audio_b64 = base64.b64encode(audio_data).decode('utf-8')
                                audio_url = f"data:audio/mp3;base64,{audio_b64}"
                                
                                # Send completion with audio
                                await websocket.send_json({
                                    "type": "response_complete",
                                    "full_response": full_response,
                                    "audio_url": audio_url,
                                    "audio_format": "mp3"
                                })
                                logger.info("Sent response with TTS audio")
                            else:
                                logger.warning("TTS generation returned None")
                                # Send completion without audio
                                await websocket.send_json({
                                    "type": "response_complete",
                                    "full_response": full_response
                                })
                                
                        except Exception as e:
                            logger.error(f"TTS error: {e}", exc_info=True)
                            # Send completion without audio
                            await websocket.send_json({
                                "type": "response_complete",
                                "full_response": full_response
                            })
                    else:
                        logger.warning("No transcription generated")
                        await websocket.send_json({
                            "type": "error",
                            "message": "Could not transcribe audio"
                        })
                        
                except Exception as e:
                    logger.error(f"Voice processing error: {e}", exc_info=True)
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Processing error: {str(e)}"
                    })
                
                finally:
                    is_processing = False
                    is_listening = True
                    user_activated = False  # Reset user activation
                        
            elif data["type"] == "stop":
                user_activated = False
                break
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        await websocket.send_json({
            "type": "error",
            "message": f"WebSocket error: {str(e)}"
        })

# ==============================================================================
# 3. CHATBOT ENDPOINT (JSON-only, SSE text streaming)
# ==============================================================================

class ChatbotRequest(BaseModel):
    session_id: str = Field(..., description="A unique identifier for the chat session. This maintains the context and knowledge base for the user.")
    query: Optional[str] = Field(None, description="The user's text query to the chatbot.")
    history: List[Dict[str, Any]] = Field([], description="A list of previous messages in the chat history.")
    web_search_enabled: bool = Field(False, description="Enable or disable web search functionality for the tutor.")

@app.post("/chatbot_endpoint")
async def chatbot_endpoint(request: ChatbotRequest):
    """
    Handles interactions with the AI tutor with JSON-only requests.
    Streaming text responses, no audio files.
    """
    session_id = request.session_id
    
    # Get or create a tutor instance for the session
    if session_id not in tutor_sessions:
        logger.info(f"Creating new AI Tutor session: {session_id}")
        tutor_config = RAGTutorConfig.from_env()
        tutor_sessions[session_id] = AsyncRAGTutor(storage_manager=storage_manager, config=tutor_config)
    
    tutor = tutor_sessions[session_id]

    # Dynamically update web search status for the tutor
    tutor.update_web_search_status(request.web_search_enabled)

    # --- Query Processing Logic ---
    if not request.query:
        raise HTTPException(status_code=400, detail="A 'query' is required.")

    is_kb_ready = tutor.ensemble_retriever is not None
    response_generator = tutor.run_agent_async(
        query=request.query,
        history=request.history,
        is_knowledge_base_ready=is_kb_ready
    )

    async def event_stream():
        import json
        async def send(obj: dict):
            yield f"data: {json.dumps(obj)}\n\n"
        try:
            async for chunk in response_generator:
                if not chunk:
                    continue
                async for part in send({"type": "text_chunk", "content": chunk}):
                    yield part
            async for part in send({"type": "done"}):
                yield part
        except Exception as e:
            logger.error(f"Error in chatbot stream: {e}", exc_info=True)
            async for part in send({"type": "error", "message": str(e)}):
                yield part

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Content-Type": "text/event-stream",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_stream(), headers=headers, media_type="text/event-stream")

# ==============================
# 4. ASSESSMENT ENDPOINT
# ==============================

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
    language: Optional[str] = Field("English", description="The language to generate the assessment in (e.g., English, Arabic)")

@app.post("/assessment_endpoint", response_model=Dict[str, Any])
async def assessment_endpoint(schema: AssessmentSchema):
    """
    Generates a set of test questions based on the provided schema.
    The response will contain the formatted questions and a separate answer key.
    Supports both single question type and mixed question type assessments.
    """
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

# ==============================
# 5. TEACHING CONTENT ENDPOINT
# ==============================

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
        description="List of AI options: 'adaptive difficulty', 'include assessment', 'multimedia suggestion', 'generate slides'"
    )
    # Add language parameter
    language: str = Field(
        "English",
        description="The language for the content (e.g., English, Arabic)."
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

# ==============================
# 6. PRESENTATION ENDPOINT
# ==============================

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
# 7. IMAGE GENERATION ENDPOINT
# ==============================
class ImageGenSchema(BaseModel):
    topic: str = Field(..., description="Topic for the image")
    grade_level: str = Field(..., description="Grade level")
    preferred_visual_type: str = Field(..., description="Visual type, e.g., image/chart/diagram")
    subject: str = Field(..., description="Subject")
    instructions: str = Field(..., description="Detailed instructions")
    difficulty_flag: str = Field("false", description="true/false flag")
    language: str = Field("English", description="Language for labels (e.g., English, Arabic)")

@app.post("/image_generation_endpoint", response_model=Dict[str, Any])
async def image_generation_endpoint(schema: ImageGenSchema):
    try:
        generator = ImageGenerator()
        schema_dict = schema.model_dump()
        image_b64 = generator.generate_image_from_schema(schema_dict)
        if not image_b64:
            raise HTTPException(status_code=500, detail="Image generation failed.")
        data_url = f"data:image/png;base64,{image_b64}"
        return {"image_url": data_url}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in image generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ==============================
# 8. WEB SEARCH ENDPOINT
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
            "Include links in the response with detailed lengthy response content. "
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
# 9. COMICS STREAMING ENDPOINT
# ==============================
class ComicsSchema(BaseModel):
    instructions: str = Field(..., description="Educational story/topic, e.g., Water cycle")
    grade_level: str = Field(..., description="Grade level string, e.g., '5' or 'Grade 5'")
    num_panels: int = Field(..., description="Number of panels to generate", ge=1, le=20)
    language: str = Field("English", description="Language for comic text (e.g., English, Arabic)")

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
                schema.num_panels,
                schema.language  # Pass language parameter
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

# Add this new endpoint for TTS generation
class VoiceResponseSchema(BaseModel):
    """Schema for voice response (TTS) requests."""
    text: str = Field(..., description="Text to convert to speech")
    voice: str = Field("alloy", description="Voice to use for TTS")

@app.post("/voice_response_endpoint")
async def voice_response_endpoint(schema: VoiceResponseSchema):
    """Generate speech from text using OpenAI's TTS API."""
    try:
        logger.info(f"Generating speech for text: '{schema.text[:100]}...'")
        
        # Import the TTS function
        from AI_voice_functionality import text_to_speech
        
        # Generate speech
        audio_data = await run_in_threadpool(text_to_speech, schema.text)
        
        if audio_data:
            # Convert to hex string for JSON response
            audio_hex = audio_data.hex()
            return {
                "success": True,
                "audio_data": audio_hex,
                "message": "Speech generated successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to generate speech")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in voice response endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# --- Uvicorn Server Runner ---
if __name__ == "__main__":
    logger.info("Starting Uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)