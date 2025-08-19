import torch
import pyaudio
import os
from openai import OpenAI
import sounddevice as sd
import numpy as np
import time
import io
import wave
from typing import Optional, Generator, List, Dict, Any
from dotenv import load_dotenv
import threading
import queue

# LangChain and Web Search Imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from websearch_code import PerplexityWebSearchTool

load_dotenv()

# --- Configuration ---
# OpenAI API Key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Common instruction prompt for the AI's persona
AI_STUDY_BUDDY_PROMPT = """You are a friendly and encouraging AI study buddy for school students. Your primary goal is to help them learn and feel supported. Your responses must be tailored to their emotional state and the context of the conversation.
Core Instructions:
1. Adopt a Persona: Always maintain a positive, encouraging, and helpful persona. Your language should be clear, easy to understand for a student audience, and avoid being overly technical or robotic.
2. Analyze and Adapt: Before responding, analyze the student's query and the outcome of any task. Your tone must dynamically change based on the following emotional layers:
    - Friendly Tone (Default for Explanations):
        * When: The student asks a question, requests an explanation, or you are providing general information.
        * How: Be warm, approachable, and encouraging. Use phrases like, "That's a great question!", "Let's break it down," "Think of it like this," or "I'm happy to help with that!"
    - Reassuring Tone (On Failure or Error):
        * When: The student's answer is incorrect, you cannot fulfill a request, or an error occurs.
        * How: Be gentle, supportive, and focus on the learning opportunity. Never be discouraging. Use phrases like, "No worries, that's a common mistake!", "That was a good try! We're very close," "It seems I had a little trouble with that request, let's try it another way," or "Don't worry if it's not perfect yet, learning is a process."
    - Excited Tone (On Success):
        * When: The student answers a question correctly, solves a problem, or completes a task successfully.
        * How: Celebrate their achievement with genuine enthusiasm! This helps build their confidence. Use phrases like, "Yes, that's exactly right! Great job!", "You nailed it! Fantastic work!", or "Awesome! You've successfully figured it out!"
    - Calm Tone (During Stress Detection):
        * When: The student's message contains keywords indicating stress, anxiety, or frustration (e.g., "I can't do this," "help," "I'm so confused," "this is too hard," "panic").
        * How: Shift to a calm, patient, and steady tone. Reassure them that it's okay to feel this way and that you're there to help them through it. Use phrases like, "It's okay, let's take a deep breath," "We can work through this together, one step at a time," "I understand this can be challenging, but don't give up," or "Let's try a simpler approach."""

# Silero VAD model
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False)
(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK = 512
TTS_SAMPLE_RATE = 24000  # OpenAI TTS uses 24kHz

# --- Global Threading Objects for Streaming and Interruption ---
audio_queue = queue.Queue()
interrupt_playback_event = threading.Event()
stop_player_thread_event = threading.Event()

# --- LangChain and Web Search Tool Setup ---
LANGCHAIN_SETUP_SUCCESS = False
llm_with_tools = None
tool_map = {}
llm = None

try:
    if os.getenv("PPLX_API_KEY"):
        # Initialize the web search tool from websearch_code.py
        websearch_tool_instance = PerplexityWebSearchTool(
            max_results=3,  # Fewer results are better for voice
            model="sonar",
            include_links=False  # Links are not useful for a voice assistant
        )
        websearch_tool = websearch_tool_instance.get_tool()
        websearch_tool.description = (
            "Use this tool to find current information, facts, or data from the internet. "
            "This is useful for questions about recent events, specific data, definitions, or topics "
            "not covered in your general knowledge."
        )

        # Initialize the LangChain LLM
        llm = ChatOpenAI(
            model="gpt-4.1",
            temperature=0.7,
            streaming=True
        )

        # Bind the tool to the LLM
        tools = [websearch_tool]
        tool_map = {tool.name: tool for tool in tools}
        llm_with_tools = llm.bind_tools(tools)
        
        LANGCHAIN_SETUP_SUCCESS = True
        print("LangChain and web search tool initialized successfully.")
    else:
        print("Perplexity API key not found. Web search functionality will be disabled.")
except Exception as e:
    print(f"Failed to initialize LangChain or WebSearch tool: {e}")


# --- Voice Activity Detection ---
def is_speech(chunk, vad_iterator):
    """
    Detects if a given audio chunk contains speech.
    """
    speech_dict = vad_iterator(chunk, return_seconds=True)
    return speech_dict is not None and 'start' in speech_dict

# --- Speech-to-Text ---
def transcribe_audio(audio_file: io.BytesIO) -> Optional[str]:
    """
    Transcribes the given audio file using GPT-4o-transcribe.
    """
    print("Transcribing audio...")
    try:
        if not hasattr(audio_file, "name"):
            audio_file.name = "speech.wav"
        transcription = client.audio.transcriptions.create(
            model="gpt-4o-transcribe", # Using whisper-1 as it is the latest stable version
            file=audio_file
        )
        return transcription.text
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

# --- Comprehensive Text Answer (Streaming) ---
def get_comprehensive_answer_stream(prompt: str) -> Generator[str, None, None]:
    """
    Gets a comprehensive answer from an LLM, streaming the response.
    Dynamically decides whether to use a web search tool based on the prompt.
    """
    print("Generating comprehensive answer stream...")
    if not LANGCHAIN_SETUP_SUCCESS or not llm_with_tools:
        print("Web search tool not available. Falling back to simple response.")
        try:
            # Fallback to the original non-tool-using method
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": AI_STUDY_BUDDY_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                stream=True
            )
            for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception as e:
            print(f"Error generating simple answer stream: {e}")
            yield "I'm sorry, I ran into an issue generating a response."
        return

    try:
        # Agentic Workflow with Web Search
        # 1. Prepare initial messages
        messages: List[Union[SystemMessage, HumanMessage, ToolMessage]] = [
            SystemMessage(content=AI_STUDY_BUDDY_PROMPT),
            HumanMessage(content=prompt)
        ]

        # 2. First invocation to decide on tool use
        # We use a non-streaming client for this initial check to get the full tool_calls object
        llm_non_streaming = ChatOpenAI(model="gpt-4.1", temperature=0.7)
        llm_with_tools_non_streaming = llm_non_streaming.bind_tools(list(tool_map.values()))
        
        ai_response = llm_with_tools_non_streaming.invoke(messages)
        messages.append(ai_response)

        # 3. Check for tool calls and execute them
        if ai_response.tool_calls:
            print(f"LLM decided to use tools: {[tc['name'] for tc in ai_response.tool_calls]}")
            for tool_call in ai_response.tool_calls:
                tool_name = tool_call["name"]
                if tool_name in tool_map:
                    selected_tool = tool_map[tool_name]
                    # Invoke the synchronous tool
                    tool_output = selected_tool.invoke(tool_call["args"])
                    print(f"Tool '{tool_name}' output received.")
                    messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"]))
                else:
                    error_msg = f"Error: Tool '{tool_name}' not found."
                    print(error_msg)
                    messages.append(ToolMessage(content=error_msg, tool_call_id=tool_call["id"]))

        # 4. Final invocation to generate a human-readable response (streamed)
        # We use the main streaming 'llm' instance here with the full message history
        final_stream = llm.stream(messages)
        for chunk in final_stream:
            if chunk.content:
                yield chunk.content

    except Exception as e:
        print(f"Error in LangChain agent workflow: {e}")
        yield "I'm sorry, I had a problem processing that request with my tools."


# --- Text-to-Speech ---
def text_to_speech(text: str) -> Optional[bytes]:
    """
    Converts text to speech using GPT-4o-mini-tts.
    """
    print(f"Generating speech for chunk: '{text}'")
    try:
        response = client.audio.speech.create(
            model="gpt-4o-mini-tts", 
            voice="alloy",
            input=text,
            response_format="wav"
        )
        return response.content
    except Exception as e:
        print(f"Error during text-to-speech: {e}")
        return None

# --- Streaming Audio Player Thread ---
def audio_player_thread():
    """
    A dedicated thread that plays audio chunks from a queue.
    This avoids the overhead of starting a new stream for each small chunk.
    """
    try:
        stream = sd.OutputStream(samplerate=TTS_SAMPLE_RATE, channels=1, dtype='int16')
        stream.start()
        print("Audio player thread started.")
    except Exception as e:
        print(f"Failed to open audio output stream: {e}")
        return

    while not stop_player_thread_event.is_set():
        try:
            audio_wav_bytes = audio_queue.get(timeout=0.1)
            
            if interrupt_playback_event.is_set():
                continue # Discard chunk if interrupted

            try:
                bio = io.BytesIO(audio_wav_bytes)
                with wave.open(bio, 'rb') as wf:
                    # Ensure audio properties match the stream
                    if wf.getframerate() != TTS_SAMPLE_RATE or wf.getnchannels() != 1:
                        print(f"Warning: Audio chunk has mismatched properties. Skipping.")
                        continue
                    
                    data = wf.readframes(1024)
                    while data and not interrupt_playback_event.is_set():
                        stream.write(np.frombuffer(data, dtype='int16'))
                        data = wf.readframes(1024)
            except Exception as e:
                print(f"Error playing audio chunk: {e}")

        except queue.Empty:
            continue
    
    stream.stop()
    stream.close()
    print("Audio player thread stopped.")

# --- Interrupt and Cleanup Functions ---
def interrupt_and_clear_audio():
    """
    Stops current playback, clears the audio queue, and sets the interrupt event.
    """
    print("Interrupting playback...")
    interrupt_playback_event.set()
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            continue

# --- LLM and TTS Processing Thread ---
def process_text_and_generate_speech(text_prompt: str):
    """
    Handles the entire process of getting a streaming response from the LLM,
    chunking it, converting to speech, and queueing it for playback.
    """
    interrupt_playback_event.clear()
    
    text_buffer = ""
    sentence_delimiters = {'.', '?', '!', '\n', ';', ':'}
    
    answer_stream = get_comprehensive_answer_stream(text_prompt)

    for text_chunk in answer_stream:
        if interrupt_playback_event.is_set():
            print("Speech processing was interrupted.")
            break
        
        text_buffer += text_chunk
        
        # Split buffer into sentences to create natural-sounding audio chunks
        while any(delim in text_buffer for delim in sentence_delimiters):
            split_pos = -1
            for delim in sentence_delimiters:
                pos = text_buffer.find(delim)
                if pos != -1 and (split_pos == -1 or pos < split_pos):
                    split_pos = pos
            
            if split_pos != -1:
                sentence = text_buffer[:split_pos + 1].strip()
                text_buffer = text_buffer[split_pos + 1:]
                
                if sentence and not interrupt_playback_event.is_set():
                    speech_audio = text_to_speech(sentence)
                    if speech_audio:
                        audio_queue.put(speech_audio)

    # Process any remaining text in the buffer
    if text_buffer.strip() and not interrupt_playback_event.is_set():
        speech_audio = text_to_speech(text_buffer.strip())
        if speech_audio:
            audio_queue.put(speech_audio)

# --- Main Application Logic ---
if __name__ == "__main__":
    p = pyaudio.PyAudio()
    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=SAMPLE_RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
    except Exception as e:
        print(f"Audio input error: {e}")
        p.terminate()
        exit(1)

    print("Application started. Listening for your voice.")
    
    # Start the persistent audio player thread
    player_thread = threading.Thread(target=audio_player_thread)
    player_thread.start()

    vad_iterator = VADIterator(model)
    processing_thread = None
    
    recorded_frames = []
    is_recording = False
    silence_start_time = None
    SILENCE_THRESHOLD_SECONDS = 2

    try:
        while True:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
            except IOError as e:
                print(f"Audio read error, skipping chunk: {e}")
                continue

            audio_chunk_torch = torch.from_numpy(np.frombuffer(data, dtype=np.int16).copy()).float() / 32768.0

            if is_speech(audio_chunk_torch, vad_iterator):
                if not is_recording:
                    print("Speech detected, starting to record...")
                    interrupt_and_clear_audio()
                    if processing_thread and processing_thread.is_alive():
                        processing_thread.join() # Wait for it to acknowledge the interrupt
                    is_recording = True
                
                recorded_frames.append(data)
                silence_start_time = None
            elif is_recording:
                if silence_start_time is None:
                    silence_start_time = time.time()
                
                if time.time() - silence_start_time > SILENCE_THRESHOLD_SECONDS:
                    print("Silence detected, processing speech.")
                    
                    audio_data = b''.join(recorded_frames)
                    recorded_frames = []
                    is_recording = False

                    audio_file = io.BytesIO()
                    with wave.open(audio_file, 'wb') as wf:
                        wf.setnchannels(CHANNELS)
                        wf.setsampwidth(p.get_sample_size(FORMAT))
                        wf.setframerate(SAMPLE_RATE)
                        wf.writeframes(audio_data)
                    audio_file.seek(0)
                    
                    transcribed_text = transcribe_audio(audio_file)

                    if transcribed_text:
                        print(f"You said: {transcribed_text}")
                        # Start the non-blocking processing thread
                        processing_thread = threading.Thread(target=process_text_and_generate_speech, args=(transcribed_text,))
                        processing_thread.start()
                    else:
                        prompt_text = "I couldn't quite catch that. Could you please say it again?"
                        speech_audio = text_to_speech(prompt_text)
                        if speech_audio:
                            audio_queue.put(speech_audio)
                else:
                    # Still in the silence period, keep recording
                    recorded_frames.append(data)

    except KeyboardInterrupt:
        print("\nExiting application.")
    finally:
        interrupt_and_clear_audio()
        if processing_thread and processing_thread.is_alive():
            processing_thread.join()
        
        stop_player_thread_event.set()
        player_thread.join()
        
        if 'stream' in locals() and stream.is_active():
            stream.stop_stream()
            stream.close()
        p.terminate()