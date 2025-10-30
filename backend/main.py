import asyncio
import json
import logging
import numpy as np
import torch
import torchaudio
import librosa
import soundfile as sf
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import SeamlessM4Tv2ForSpeechToSpeech
from transformers.models.seamless_m4t.processing_seamless_m4t import SeamlessM4TProcessor
import io
import wave
import tempfile
import os
import hashlib
import time
from collections import deque
import threading
from queue import Queue, Empty

# Import pydub for WebM processing (same as Gradio)
try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None

# Try to import SimulEval streaming agents if available
try:
    from seamless_communication.streaming.agents.seamless_streaming_s2st import (
        SeamlessStreamingS2STAgent,
        SeamlessStreamingS2STVADAgent
    )
    from seamless_communication.streaming.agents.seamless_s2st import (
        SeamlessS2STAgent
    )
    from simuleval.data.segments import SpeechSegment, EmptySegment
    from simuleval.agents.actions import ReadAction, WriteAction
    STREAMING_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("SeamlessStreaming agents are available")
except ImportError as e:
    STREAMING_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"SeamlessStreaming agents not available: {e}")
    logger.warning("Using fallback chunked processing instead of true streaming")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SeamlessStreaming Translation API")

# Serve static frontend files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Language code mapping
LANGUAGE_MAPPING = {
    "en": "eng",  # English
    "bn": "ben",  # Bangla
}

class StreamingTranslator:
    """Handles continuous streaming translation with buffering and chunking"""
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.chunk_duration = 4.0  # Process chunks of 4 seconds for better context
        self.overlap_duration = 1.0  # 1 second overlap between chunks for continuity
        self.sample_rate = 16000
        self.buffer = deque(maxlen=int(self.sample_rate * 15))  # 15 second circular buffer for more context
        logger.info(f"Using device: {self.device}")
        
    async def initialize(self):
        """Initialize the model asynchronously"""
        try:
            logger.info("Loading SeamlessM4T model for streaming...")
            self.model = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained(
                "facebook/seamless-m4t-v2-large",
                dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            
            self.processor = SeamlessM4TProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
            logger.info("Streaming model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load streaming model: {e}")
            raise e
    
    def add_audio_chunk(self, audio_data: np.ndarray):
        """Add audio chunk to the streaming buffer"""
        self.buffer.extend(audio_data)
    
    def get_processing_chunk(self) -> Optional[np.ndarray]:
        """Get the next chunk for processing with overlap"""
        if len(self.buffer) < int(self.sample_rate * self.chunk_duration):
            return None
        
        chunk_samples = int(self.sample_rate * self.chunk_duration)
        # Take the most recent chunk for better real-time processing
        chunk = np.array(list(self.buffer)[-chunk_samples:])
        
        # Don't remove samples in streaming mode - keep accumulating for context
        # This allows better translation continuity
        
        return chunk
    
    async def process_streaming_chunk(self, chunk: np.ndarray, src_lang: str, tgt_lang: str) -> Optional[bytes]:
        """Process a single streaming chunk"""
        try:
            # Check for minimum audio energy (avoid processing silence)
            audio_energy = np.sqrt(np.mean(chunk ** 2))
            if audio_energy < 0.01:  # Threshold for silence detection
                logger.debug("Skipping chunk with low audio energy (likely silence)")
                return None
            
            # Convert to tensor
            waveform = torch.from_numpy(chunk).float()
            
            # Normalize audio to prevent clipping
            if torch.max(torch.abs(waveform)) > 1.0:
                waveform = waveform / torch.max(torch.abs(waveform))
            
            # Process with model
            inputs = self.processor(
                audio=waveform,
                sampling_rate=self.sample_rate,
                src_lang=src_lang,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate with quality-focused settings
            with torch.no_grad():
                # Use consistent seed for better translation consistency
                torch.manual_seed(42)
                
                logger.info(f"Processing {chunk.shape[0]/self.sample_rate:.2f}s chunk: {src_lang} → {tgt_lang}")
                start_time = time.time()
                
                audio_array = self.model.generate(
                    **inputs, 
                    tgt_lang=tgt_lang,
                    do_sample=True,
                    temperature=0.1,  # Lower temperature for more deterministic, better quality
                    num_beams=2,  # Use beam search for better quality
                    max_length=400,  # Allow longer output for complete phrases
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    repetition_penalty=1.1,  # Prevent repetition
                    length_penalty=1.0  # Neutral length penalty
                )
                
                generation_time = time.time() - start_time
                logger.info(f"Translation completed in {generation_time:.2f}s")
            
            # Convert to PCM bytes
            if isinstance(audio_array, torch.Tensor):
                return self._tensor_to_pcm_bytes(audio_array.squeeze())
            elif isinstance(audio_array, (list, tuple)) and len(audio_array) > 0:
                audio_tensor = audio_array[0] if hasattr(audio_array[0], 'shape') else audio_array
                return self._tensor_to_pcm_bytes(audio_tensor.squeeze())
            else:
                logger.error(f"Unexpected output type from streaming model: {type(audio_array)}")
                return None
                
        except Exception as e:
            logger.error(f"Streaming translation error: {e}")
            return None
    
    def _tensor_to_pcm_bytes(self, tensor: torch.Tensor) -> bytes:
        """Convert tensor to raw PCM bytes"""
        try:
            audio_np = tensor.detach().cpu().numpy()
            
            # Normalize audio
            if audio_np.max() > 1.0 or audio_np.min() < -1.0:
                audio_np = audio_np / max(abs(audio_np.max()), abs(audio_np.min()))
            
            # Convert to 16-bit PCM
            audio_16bit = (audio_np * 32767).astype(np.int16)
            return audio_16bit.tobytes()
                
        except Exception as e:
            logger.error(f"Error converting streaming tensor to PCM: {e}")
            return None

# Global streaming translator instance
streaming_translator = StreamingTranslator()

class StreamingSession:
    """Manages a streaming translation session"""
    def __init__(self, websocket: WebSocket, src_lang: str, tgt_lang: str, sample_rate: int = 16000):
        self.websocket = websocket
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.sample_rate = sample_rate
        self.running = False
        self.last_output_time = 0
        self.min_output_interval = 1.0  # Minimum 1s between outputs for better quality
        # Use streaming buffer management without creating new translator
        self.buffer = deque(maxlen=int(sample_rate * 15))  # 15 second circular buffer
        self.chunk_duration = 5.0  # Process chunks of 5 seconds for better context
        self.overlap_duration = 1.5  # 1.5 second overlap between chunks for continuity
    
    async def initialize(self):
        """Initialize the streaming session - no model loading needed"""
        # No model loading - reuse global streaming_translator
        pass
    
    def add_audio_chunk(self, audio_data: np.ndarray):
        """Add audio chunk to the streaming buffer"""
        self.buffer.extend(audio_data)
    
    def get_processing_chunk(self) -> Optional[np.ndarray]:
        """Get the next chunk for processing with overlap"""
        if len(self.buffer) < int(self.sample_rate * self.chunk_duration):
            return None
        
        chunk_samples = int(self.sample_rate * self.chunk_duration)
        # Take the most recent chunk for better real-time processing
        chunk = np.array(list(self.buffer)[-chunk_samples:])
        
        # Don't remove samples in streaming mode for better continuity
        # Keep accumulating context but limit buffer size via maxlen
        
        return chunk

    async def process_audio_chunk(self, audio_data: bytes):
        """Process an incoming audio chunk"""
        try:
            # Convert audio bytes to numpy array
            if len(audio_data) % 2 != 0:
                audio_data = audio_data + b'\x00'
            
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Log audio characteristics for debugging
            audio_rms = np.sqrt(np.mean(audio_np ** 2))
            audio_max = np.max(np.abs(audio_np))
            logger.debug(f"Audio chunk: {len(audio_np)} samples, RMS: {audio_rms:.4f}, Max: {audio_max:.4f}")
            
            # Add to streaming buffer
            self.add_audio_chunk(audio_np)
            
            # Check if we can process a chunk
            current_time = time.time()
            if current_time - self.last_output_time >= self.min_output_interval:
                chunk = self.get_processing_chunk()
                if chunk is not None:
                    # Process the chunk using global streaming_translator
                    result = await streaming_translator.process_streaming_chunk(chunk, self.src_lang, self.tgt_lang)
                    if result:
                        await self.websocket.send_bytes(result)
                        self.last_output_time = current_time
                        logger.info(f"Sent streaming translation: {len(result)} bytes")
            
        except Exception as e:
            logger.error(f"Error processing streaming audio chunk: {e}")

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.streaming_sessions: Dict[str, StreamingSession] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected")
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.streaming_sessions:
            del self.streaming_sessions[client_id]
        logger.info(f"Client {client_id} disconnected")
    
    async def create_streaming_session(self, websocket: WebSocket, client_id: str, src_lang: str, tgt_lang: str):
        """Create a new streaming session"""
        session = StreamingSession(websocket, src_lang, tgt_lang)
        await session.initialize()
        self.streaming_sessions[client_id] = session
        return session

manager = ConnectionManager()

@app.on_event("startup")
async def startup_event():
    """Initialize the streaming model on startup"""
    await streaming_translator.initialize()

@app.get("/")
async def root():
    """Serve the frontend"""
    return FileResponse("frontend/index.html")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": streaming_translator.model is not None}



@app.websocket("/ws/stream")
async def websocket_streaming_translate(websocket: WebSocket):
    """WebSocket endpoint for continuous streaming speech translation"""
    await websocket.accept()
    
    try:
        # Wait for configuration message
        config_msg = await websocket.receive_text()
        config = json.loads(config_msg)
        
        if config.get("type") != "start_stream":
            await websocket.send_text('{"type":"error","msg":"expected start_stream message"}')
            await websocket.close()
            return
        
        # Parse streaming config
        src_lang = LANGUAGE_MAPPING.get(config.get('src_lang', 'en'), 'eng')
        tgt_lang = LANGUAGE_MAPPING.get(config.get('tgt_lang', 'bn'), 'ben')
        sample_rate = int(config.get('sample_rate', 16000))
        
        logger.info(f"Started streaming session: {src_lang} -> {tgt_lang} at {sample_rate}Hz")
        
        # Create streaming session
        client_id = f"stream_{int(time.time())}"
        session = await manager.create_streaming_session(websocket, client_id, src_lang, tgt_lang)
        session.running = True
        
        # Send ready confirmation
        await websocket.send_text('{"type":"stream_ready"}')
        
        # Process continuous audio chunks
        while session.running:
            message = await websocket.receive()
            
            if "bytes" in message:
                # Process streaming audio chunk
                audio_chunk = message["bytes"]
                
                if len(audio_chunk) > 0:
                    await session.process_audio_chunk(audio_chunk)
                    
            elif "text" in message:
                # Handle control messages
                try:
                    ctrl = json.loads(message["text"])
                    if ctrl.get("type") == "stop_stream":
                        session.running = False
                        break
                except Exception:
                    pass
        
        # Clean up
        manager.disconnect(client_id)
        await websocket.send_text('{"type":"stream_end","reason":"completed"}')
        
    except WebSocketDisconnect:
        logger.info("Streaming client disconnected")
    except Exception as e:
        logger.error(f"Streaming WebSocket error: {e}")
        try:
            await websocket.send_text(f'{{"type":"error","msg":"{str(e)}"}}')
        except:
            pass



if __name__ == "__main__":
    import os
    
    # Print startup info
    print("🚀 SeamlessStreaming Translation Service")
    print(f"🐍 Python: {os.sys.version}")
    print(f"⚡ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"📂 Model cache: {os.getenv('TRANSFORMERS_CACHE', 'default')}")
    print("=" * 50)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=7860,
        log_level="info",
        access_log=True
    )