import os
import asyncio
import tempfile
import shutil
from typing import List, Optional, Any, Dict
from datetime import datetime, timedelta
import streamlit as st
import json
import nest_asyncio
import logging

# Apply nest_asyncio patch first. This is critical.
nest_asyncio.apply()

# Import the AI tutor and its config
from AI_tutor import AsyncRAGTutor, RAGTutorConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- ROBUST ASYNC EXECUTION FOR STREAMLIT ---

def get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """
    Gets the current event loop or creates a new one if none exists.
    Crucially, it stores the loop in the Streamlit session state to ensure
    it persists across reruns.
    """
    if 'event_loop' not in st.session_state:
        logger.info("Creating and storing new event loop in session state.")
        st.session_state.event_loop = asyncio.new_event_loop()
    return st.session_state.event_loop

def run_async(coro):
    """
    Runs a coroutine in a stable, session-persistent event loop within Streamlit.
    This prevents the "attached to a different loop" error.
    """
    loop = get_or_create_event_loop()
    try:
        return loop.run_until_complete(coro)
    except Exception as e:
        logger.error(f"Error running async operation in Streamlit: {e}", exc_info=True)
        st.error(f"An internal asynchronous error occurred: {e}")
        return None


# --- LocalStorageManager Class ---
class LocalStorageManager:
    """Manages files locally, mimicking the R2 storage interface."""
    def __init__(self, base_path="tutor_session_data"):
        self.base_path = base_path
        self.user_docs_path = os.path.join(base_path, "user_docs")
        self.metadata_path = os.path.join(base_path, "metadata.json")
        self._initialize_storage()

    def _initialize_storage(self):
        try:
            os.makedirs(self.user_docs_path, exist_ok=True)
            if not os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'w') as f:
                    json.dump({}, f)
        except Exception as e:
            logger.error(f"Error initializing storage: {e}")

    def _load_metadata(self):
        try:
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def _save_metadata(self, metadata):
        try:
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")

    def upload_file_sync(self, file_data: bytes, filename: str, is_user_doc: bool, schedule_deletion_hours: int = 24) -> tuple[bool, str]:
        folder = self.user_docs_path if is_user_doc else os.path.dirname(self.user_docs_path)
        key = os.path.join("user_docs" if is_user_doc else "", filename).replace("\\", "/")
        local_path = os.path.join(self.base_path, key)

        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, 'wb') as f:
                f.write(file_data)
            logger.info(f"Successfully saved file to local path: {local_path}")

            metadata = self._load_metadata()
            expiration_time = datetime.now() + timedelta(hours=schedule_deletion_hours)
            metadata[key] = {"expiration_time": expiration_time.isoformat()}
            self._save_metadata(metadata)
            logger.info(f"Scheduled deletion for '{key}' at {expiration_time.isoformat()}")

            return True, key
        except Exception as e:
            logger.error(f"Error saving file '{filename}' locally: {e}")
            return False, str(e)

    async def upload_file_async(self, file_data: bytes, filename: str, is_user_doc: bool, schedule_deletion_hours: int = 24) -> tuple[bool, str]:
        return self.upload_file_sync(file_data, filename, is_user_doc, schedule_deletion_hours)

    def get_file_content_bytes_sync(self, key: str) -> Optional[bytes]:
        local_path = os.path.join(self.base_path, key)
        if os.path.exists(local_path):
            try:
                with open(local_path, 'rb') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Error reading local file '{key}': {e}")
        return None

    async def get_file_content_bytes_async(self, key: str) -> Optional[bytes]:
        return self.get_file_content_bytes_sync(key)

    def cleanup_expired_files(self):
        logger.info("Running cleanup for expired local files.")
        try:
            metadata = self._load_metadata()
            now = datetime.now()
            keys_to_delete = [
                key for key, data in metadata.items()
                if (exp_time_str := data.get("expiration_time")) and now > datetime.fromisoformat(exp_time_str)
            ]

            if not keys_to_delete:
                logger.info("No expired files to clean up.")
                return

            deleted_count = 0
            for key in keys_to_delete:
                local_path = os.path.join(self.base_path, key)
                if os.path.exists(local_path):
                    try:
                        os.remove(local_path)
                        deleted_count += 1
                        logger.info(f"Deleted expired file: {local_path}")
                    except OSError as e:
                        logger.error(f"Error deleting expired file '{local_path}': {e}")
                del metadata[key]
            self._save_metadata(metadata)
            logger.info(f"Cleanup complete. Deleted {deleted_count} files.")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def clear_all_data(self):
        logger.warning(f"Clearing all local storage at: {self.base_path}")
        try:
            if os.path.exists(self.base_path):
                shutil.rmtree(self.base_path)
            self._initialize_storage()
            logger.info("Local storage cleared and re-initialized.")
        except Exception as e:
            logger.error(f"Error clearing local storage: {e}")


# --- Streamlit App ---

# Page configuration
st.set_page_config(
    page_title="AI Tutor Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    body { color: black; }
    .chat-message { padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex; flex-direction: column; color: black; }
    .chat-message.user { background-color: #e3f2fd; }
    .chat-message.assistant { background-color: #f5f5f5; }
    .chat-message .avatar { width: 40px; height: 40px; border-radius: 50%; margin-bottom: 0.5rem; align-self: flex-start; }
    .uploaded-files { background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 0.5rem; padding: 0.5rem; margin: 0.5rem 0; color: black; }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "web_search_enabled" not in st.session_state:
    st.session_state.web_search_enabled = False
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "last_uploaded_files" not in st.session_state:
    st.session_state.last_uploaded_files = []
if "knowledge_base_ready" not in st.session_state:
    st.session_state.knowledge_base_ready = False

# Initialize LocalStorageManager once per session
if "storage_manager" not in st.session_state:
    st.session_state.storage_manager = LocalStorageManager()
    run_async(asyncio.sleep(0.01)) # Allow loop to initialize
    st.session_state.storage_manager.cleanup_expired_files()
    logger.info("LocalStorageManager initialized in session state.")

def initialize_tutor(force_reinit=False):
    """Initializes or reinitializes the AI tutor in session state."""
    if force_reinit or "tutor" not in st.session_state or st.session_state.tutor is None:
        try:
            tutor_config = RAGTutorConfig(web_search_enabled=st.session_state.web_search_enabled)
            st.session_state.tutor = AsyncRAGTutor(
                storage_manager=st.session_state.storage_manager,
                config=tutor_config
            )
            logger.info(f"AI Tutor initialized. Web Search: {st.session_state.web_search_enabled}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize AI Tutor: {e}", exc_info=True)
            st.error(f"Failed to initialize AI Tutor: {str(e)}")
            st.session_state.tutor = None
            return False
    return True

def display_chat_message(role: str, content: str, files: Optional[List[str]] = None):
    """Displays a single chat message with consistent styling."""
    avatar = "🧑‍💼" if role == "user" else "🎓"
    name = "You" if role == "user" else "AI Tutor"
    
    # Define alignment for user vs assistant
    align_class = "user" if role == "user" else "assistant"
    
    with st.container():
        st.markdown(f"""
        <div class="chat-message {align_class}">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-size: 1.5rem; margin-right: 0.5rem;">{avatar}</span>
                <strong>{name}</strong>
            </div>
            <div>{content}</div>
        </div>
        """, unsafe_allow_html=True)
        if files:
            with st.container():
                st.markdown(f'<div class="uploaded-files" style="margin-left: {"20%" if role == "user" else "0"};">', unsafe_allow_html=True)
                st.write("📎 **Referenced files:**")
                for filename in files:
                    st.write(f"• {filename}")
                st.markdown('</div>', unsafe_allow_html=True)

def handle_uploads_and_ingest(uploaded_files):
    """Handles file uploading and ingestion into the knowledge base."""
    if not initialize_tutor():
        st.error("AI Tutor initialization failed. Cannot process files.")
        return

    try:
        with st.spinner("✨ Clearing old knowledge base..."):
            run_async(st.session_state.tutor.clear_knowledge_base_async())

        storage_manager = st.session_state.storage_manager
        successful_keys = []
        with st.spinner("📤 Uploading files..."):
            for uploaded_file in uploaded_files:
                file_bytes = uploaded_file.getvalue()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_filename = "".join(c for c in uploaded_file.name if c.isalnum() or c in '._-').strip()
                unique_filename = f"{timestamp}_{safe_filename}"
                success, key_or_error = storage_manager.upload_file_sync(
                    file_data=file_bytes, filename=unique_filename, is_user_doc=True
                )
                if success:
                    successful_keys.append(key_or_error)
                else:
                    st.error(f"Error uploading {uploaded_file.name}: {key_or_error}")

        if successful_keys:
            with st.spinner("🧠 Processing documents into knowledge base..."):
                success = run_async(st.session_state.tutor.ingest_async(successful_keys))
                if success:
                    st.session_state.knowledge_base_ready = True
                    st.session_state.files_for_context = [f.name for f in uploaded_files]
                    st.success("✅ Knowledge base created! Files will be deleted in 24 hours. You can now ask questions about them.")
                else:
                    st.error("❌ Failed to create knowledge base from the uploaded files.")
    except Exception as e:
        logger.error(f"Error during upload/ingest process: {e}", exc_info=True)
        st.error(f"An error occurred while processing files: {str(e)}")

async def get_response_stream_async(query: str, history: List[Dict[str, Any]], image_storage_key: Optional[str] = None, uploaded_files: Optional[List[str]] = None):
    """Async generator to get streaming response from the AI tutor."""
    try:
        if not st.session_state.tutor:
            yield "Error: AI Tutor not initialized"
            return

        async for chunk in st.session_state.tutor.run_agent_async(
            query=query,
            history=history,
            image_storage_key=image_storage_key,
            is_knowledge_base_ready=st.session_state.knowledge_base_ready,
            uploaded_files=uploaded_files
        ):
            yield chunk
    except Exception as e:
        logger.error(f"Error getting response stream: {e}", exc_info=True)
        yield f"An error occurred while generating the response: {e}"

def main():
    """Main Streamlit application function."""
    st.title("🎓 AI Tutor Chatbot")
    st.markdown("Upload your documents. They are stored locally and auto-deleted in 24 hours.")

    if not initialize_tutor():
        st.error("Tutor initialization failed. Please refresh the page.")
        return

    with st.sidebar:
        st.markdown("### 📋 Chat Features")
        st.markdown("""
        - **📁 Local Storage**: Files are stored on the machine running this app.
        - **⏱️ 24h Lifecycle**: Uploads are auto-deleted after 24 hours.
        - **📄 Document & 🖼️ Image Upload**: Supports various formats.
        - **Isolated Sessions**: Each upload creates a fresh knowledge base.
        """)

        web_search_enabled = st.toggle(
            "🌐 Enable Web Search",
            value=st.session_state.web_search_enabled,
            help="Allow the tutor to search the web. Requires a PPLX_API_KEY."
        )
        

        if web_search_enabled != st.session_state.web_search_enabled:
            st.session_state.web_search_enabled = web_search_enabled
            
            # Check if the tutor object exists before trying to update it
            if "tutor" in st.session_state and st.session_state.tutor is not None:
                # Call the new method to dynamically update the tool
                st.session_state.tutor.update_web_search_status(web_search_enabled)
                status = "enabled" if web_search_enabled else "disabled"
                st.toast(f"✅ Web search {status}!", icon="🌐")
            else:
                # Fallback to re-initialization ONLY if the tutor object doesn't exist
                initialize_tutor(force_reinit=True)
            
            # Use st.rerun() to immediately reflect the change in the UI
            st.rerun()

        st.success("✅ Knowledge base ready!") if st.session_state.knowledge_base_ready else st.info("ℹ️ Upload files for RAG")

        if st.button("🗑️ Clear Chat, KB & Files", use_container_width=True):
            if "storage_manager" in st.session_state:
                st.session_state.storage_manager.clear_all_data()
            st.session_state.messages = []
            st.session_state.knowledge_base_ready = False
            st.session_state.uploader_key += 1
            if 'files_for_context' in st.session_state:
                del st.session_state['files_for_context']
            initialize_tutor(force_reinit=True)
            st.rerun()

    # Display chat history
    for message in st.session_state.messages:
        display_chat_message(message["role"], message["content"], message.get("files"))

    st.markdown("---")

    uploaded_files = st.file_uploader(
        "📎 Upload documents or images for a new session",
        accept_multiple_files=True,
        type=['pdf', 'docx', 'txt', 'md', 'json', 'html', 'jpg', 'jpeg', 'png', 'webp'],
        key=f"file_uploader_{st.session_state.uploader_key}"
    )

    if uploaded_files and uploaded_files != st.session_state.last_uploaded_files:
        st.session_state.last_uploaded_files = uploaded_files
        handle_uploads_and_ingest(uploaded_files)
        st.session_state.uploader_key += 1
        st.rerun()

    if user_input := st.chat_input("💬 Ask me anything..."):
        files_for_context = st.session_state.get("files_for_context")
        chat_history_for_rephrasing = st.session_state.messages[:]
        
        st.session_state.messages.append({"role": "user", "content": user_input, "files": files_for_context})
        display_chat_message("user", user_input, files_for_context)

        full_response = ""
        try:
            # Display assistant response in a streaming fashion
            with st.chat_message("assistant", avatar="🎓"):
                response_placeholder = st.empty()
                
                async def stream_response():
                    """Defines and runs the streaming coroutine."""
                    nonlocal full_response
                    # Add a blinking cursor effect
                    cursor = "▌"
                    
                    streamer = get_response_stream_async(
                        user_input,
                        chat_history_for_rephrasing,
                        uploaded_files=files_for_context
                    )
                    async for chunk in streamer:
                        full_response += chunk
                        response_placeholder.markdown(full_response + cursor)
                    # Remove cursor after streaming is complete
                    response_placeholder.markdown(full_response)

                run_async(stream_response())

            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            st.error(error_msg)
            logger.error(f"Error in chat response stream: {e}", exc_info=True)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

        # Clean up context for the next turn
        if 'files_for_context' in st.session_state:
            del st.session_state['files_for_context']

if __name__ == "__main__":
    main()