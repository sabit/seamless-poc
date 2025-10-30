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
        self.chunk_duration = 2.0  # Process chunks of 2 seconds
        self.overlap_duration = 0.5  # 0.5 second overlap between chunks
        self.sample_rate = 16000
        self.buffer = deque(maxlen=int(self.sample_rate * 10))  # 10 second circular buffer
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
        chunk = np.array(list(self.buffer)[-chunk_samples:])
        
        # Remove processed samples (keeping overlap)
        overlap_samples = int(self.sample_rate * self.overlap_duration)
        samples_to_remove = chunk_samples - overlap_samples
        for _ in range(min(samples_to_remove, len(self.buffer))):
            self.buffer.popleft()
            
        return chunk
    
    async def process_streaming_chunk(self, chunk: np.ndarray, src_lang: str, tgt_lang: str) -> Optional[bytes]:
        """Process a single streaming chunk"""
        try:
            # Convert to tensor
            waveform = torch.from_numpy(chunk).float()
            
            # Process with model
            inputs = self.processor(
                audio=waveform,
                sampling_rate=self.sample_rate,
                src_lang=src_lang,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate with streaming-optimized settings
            with torch.no_grad():
                # Use faster generation settings for streaming
                torch.manual_seed(int(time.time() * 1000000) % 2**32)
                
                audio_array = self.model.generate(
                    **inputs, 
                    tgt_lang=tgt_lang,
                    do_sample=True,
                    temperature=0.6,  # Lower temperature for faster generation
                    num_beams=1,  # Always use sampling for speed
                    max_length=200,  # Limit output length for faster response
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )
            
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

class SeamlessTranslator:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
    async def initialize(self):
        """Initialize the model asynchronously"""
        try:
            logger.info("Loading SeamlessM4T model...")
            self.model = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained(
                "facebook/seamless-m4t-v2-large",
                dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            
            self.processor = SeamlessM4TProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e
    
    def audio_bytes_to_tensor(self, audio_bytes: bytes, sample_rate: int = 16000):
        """Convert audio bytes to tensor for model input - handles WAV and WebM formats"""
        try:
            logger.info(f"Processing {len(audio_bytes)} bytes, format signature: {audio_bytes[:12]}")
            
            # Detect audio format by checking file signature
            if audio_bytes.startswith(b'RIFF') and b'WAVE' in audio_bytes[:20]:
                # WAV format detected
                return self._process_wav_audio(audio_bytes, sample_rate)
            elif audio_bytes.startswith(b'\x1a\x45\xdf\xa3') or b'webm' in audio_bytes[:50].lower():
                # WebM format detected  
                return self._process_webm_audio(audio_bytes, sample_rate)
            else:
                # Try as raw PCM data
                return self._process_raw_pcm(audio_bytes, sample_rate)
                
        except Exception as e:
            logger.error(f"Error processing audio bytes: {e}")
            logger.error(f"Audio bytes length: {len(audio_bytes)}, first 20 bytes: {audio_bytes[:20]}")
            return None
    
    def _process_wav_audio(self, audio_bytes: bytes, sample_rate: int):
        """Process WAV audio format"""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_file.flush()
                
                # Load with librosa
                waveform, sr = librosa.load(temp_file.name, sr=sample_rate, mono=True)
                os.unlink(temp_file.name)
                
                logger.info(f"Processed WAV: {len(waveform)} samples at {sr}Hz")
                return torch.from_numpy(waveform).float()
                
        except Exception as e:
            logger.error(f"WAV processing failed: {e}")
            return None
    
    def _process_webm_audio(self, audio_bytes: bytes, sample_rate: int):
        """Process WebM audio format using PyDub (same as Gradio)"""
        try:
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_file.flush()
                
                try:
                    # Use PyDub's AudioSegment.from_file() - same as Gradio
                    if AudioSegment is None:
                        raise ImportError("pydub is not installed")
                    
                    audio = AudioSegment.from_file(temp_file.name, format="webm")
                    
                    # Convert to numpy array
                    samples = audio.get_array_of_samples()
                    waveform = np.array(samples).astype(np.float32)
                    
                    # Normalize to [-1, 1] range
                    if audio.sample_width == 2:  # 16-bit
                        waveform = waveform / 32768.0
                    elif audio.sample_width == 4:  # 32-bit
                        waveform = waveform / 2147483648.0
                    
                    # Convert to mono if stereo
                    if audio.channels == 2:
                        waveform = waveform.reshape(-1, 2).mean(axis=1)
                    
                    # Resample if needed
                    if audio.frame_rate != sample_rate:
                        waveform = librosa.resample(waveform, orig_sr=audio.frame_rate, target_sr=sample_rate)
                    
                    os.unlink(temp_file.name)
                    
                    if len(waveform) > 0:
                        logger.info(f"Processed WebM via pydub: {len(waveform)} samples at {sample_rate}Hz")
                        return torch.from_numpy(waveform).float()
                    else:
                        logger.warning("Empty waveform from WebM processing")
                        return None
                        
                except Exception as e:
                    logger.error(f"PyDub WebM processing failed: {e}")
                    os.unlink(temp_file.name)
                    return None
                            
        except Exception as e:
            logger.error(f"WebM processing completely failed: {e}")
            return None
    
    def _process_raw_pcm(self, audio_bytes: bytes, sample_rate: int):
        """Process raw PCM audio data"""
        try:
            # Handle potential padding issues
            if len(audio_bytes) % 2 != 0:
                audio_bytes = audio_bytes + b'\x00'
            
            # Convert bytes to numpy array (16-bit signed PCM)
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
            waveform = audio_np.astype(np.float32) / 32768.0
            
            logger.info(f"Processed raw PCM: {len(waveform)} samples")
            return torch.from_numpy(waveform).float()
            
        except Exception as e:
            logger.error(f"Raw PCM processing failed: {e}")
            return None
    
    def tensor_to_pcm_bytes(self, tensor: torch.Tensor, sample_rate: int = 16000):
        """Convert tensor to raw PCM bytes (like echo server)"""
        try:
            # Ensure tensor is on CPU and convert to numpy
            audio_np = tensor.detach().cpu().numpy()
            
            # Normalize audio to prevent clipping
            if audio_np.max() > 1.0 or audio_np.min() < -1.0:
                audio_np = audio_np / max(abs(audio_np.max()), abs(audio_np.min()))
            
            # Convert to 16-bit PCM (same as echo server output)
            audio_16bit = (audio_np * 32767).astype(np.int16)
            
            return audio_16bit.tobytes()
                
        except Exception as e:
            logger.error(f"Error converting tensor to PCM bytes: {e}")
            return None

    def tensor_to_audio_bytes(self, tensor: torch.Tensor, sample_rate: int = 16000):
        """Convert tensor to audio bytes"""
        try:
            # Ensure tensor is on CPU and convert to numpy
            audio_np = tensor.detach().cpu().numpy()
            
            # Normalize audio to prevent clipping
            if audio_np.max() > 1.0 or audio_np.min() < -1.0:
                audio_np = audio_np / max(abs(audio_np.max()), abs(audio_np.min()))
            
            # Convert to 16-bit PCM
            audio_16bit = (audio_np * 32767).astype(np.int16)
            
            # Create wav bytes
            with io.BytesIO() as wav_buffer:
                with wave.open(wav_buffer, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(audio_16bit.tobytes())
                
                return wav_buffer.getvalue()
                
        except Exception as e:
            logger.error(f"Error converting tensor to audio bytes: {e}")
            return None
    
    async def translate_speech_tensor(self, waveform: torch.Tensor, src_lang: str, tgt_lang: str, sample_rate: int = 16000):
        """Translate speech from tensor input - optimized for real-time processing"""
        try:
            logger.info(f"Translating tensor: {waveform.shape} samples, {src_lang} -> {tgt_lang}")
            
            # Add input validation
            if len(waveform) < sample_rate * 0.1:  # Less than 0.1 seconds
                logger.warning("Audio too short for reliable translation")
                return None
                
            # Process audio with the model
            inputs = self.processor(
                audio=waveform,
                sampling_rate=sample_rate,
                src_lang=src_lang,
                return_tensors="pt"
            ).to(self.device)
            
            # Debug: Check what we're sending to the model
            logger.info(f"Input keys: {list(inputs.keys())}")
            logger.info(f"Input shapes: {[(k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in inputs.items()]}")
            
            # Generate speech translation with some randomization to reduce identical outputs
            with torch.no_grad():
                # Set a different random seed for each translation to ensure variation
                torch.manual_seed(int(time.time() * 1000000) % 2**32)
                
                # Add some randomness to reduce identical outputs for similar inputs
                audio_array = self.model.generate(
                    **inputs, 
                    tgt_lang=tgt_lang,
                    do_sample=True,  # Enable sampling for more variation
                    temperature=0.8,  # Add some randomness (increased from 0.7)
                    num_beams=1,  # Use sampling instead of beam search
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )
            
            logger.info(f"Generated audio type: {type(audio_array)}, shape: {audio_array.shape if hasattr(audio_array, 'shape') else 'no shape'}")
            
            # The model returns audio tensor directly (according to docs)
            if isinstance(audio_array, torch.Tensor):
                # Convert tensor to PCM bytes for browser playback
                return self.tensor_to_pcm_bytes(audio_array.squeeze())
            elif isinstance(audio_array, (list, tuple)) and len(audio_array) > 0:
                # Handle case where it returns list/tuple with tensor
                audio_tensor = audio_array[0] if hasattr(audio_array[0], 'shape') else audio_array
                logger.info(f"Using first element: {type(audio_tensor)}, shape: {audio_tensor.shape}")
                return self.tensor_to_pcm_bytes(audio_tensor.squeeze())
            else:
                logger.error(f"Unexpected output type from model: {type(audio_array)}")
                return None
            
        except Exception as e:
            logger.error(f"Translation error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    async def translate_speech(self, audio_bytes: bytes, src_lang: str, tgt_lang: str, sample_rate: int = 16000):
        """Translate speech from source to target language"""
        try:
            # Convert audio bytes to tensor
            waveform = self.audio_bytes_to_tensor(audio_bytes, sample_rate)
            if waveform is None:
                return None
            
            return await self.translate_speech_tensor(waveform, src_lang, tgt_lang, sample_rate)
            
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return None

# Global translator instances
translator = SeamlessTranslator()
streaming_translator = StreamingTranslator()

class StreamingSession:
    """Manages a streaming translation session"""
    def __init__(self, websocket: WebSocket, src_lang: str, tgt_lang: str, sample_rate: int = 16000):
        self.websocket = websocket
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.sample_rate = sample_rate
        self.translator = StreamingTranslator()
        self.running = False
        self.last_output_time = 0
        self.min_output_interval = 0.5  # Minimum 500ms between outputs
    
    async def initialize(self):
        """Initialize the streaming session"""
        await self.translator.initialize()
    
    async def process_audio_chunk(self, audio_data: bytes):
        """Process an incoming audio chunk"""
        try:
            # Convert audio bytes to numpy array
            if len(audio_data) % 2 != 0:
                audio_data = audio_data + b'\x00'
            
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Add to streaming buffer
            self.translator.add_audio_chunk(audio_np)
            
            # Check if we can process a chunk
            current_time = time.time()
            if current_time - self.last_output_time >= self.min_output_interval:
                chunk = self.translator.get_processing_chunk()
                if chunk is not None:
                    # Process the chunk
                    result = await self.translator.process_streaming_chunk(chunk, self.src_lang, self.tgt_lang)
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
    """Initialize the models on startup"""
    await translator.initialize()
    await streaming_translator.initialize()

@app.get("/")
async def root():
    """Serve the frontend"""
    return FileResponse("frontend/index.html")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": translator.model is not None}

@app.websocket("/ws/translate")
async def websocket_translate(websocket: WebSocket):
    """WebSocket endpoint for complete recording speech translation"""
    await websocket.accept()
    
    try:
        # 1) Wait for start control message
        start_msg = await websocket.receive_text()
        config = json.loads(start_msg)
        
        if config.get("type") != "start":
            await websocket.send_text('{"type":"error","msg":"expected start message"}')
            await websocket.close()
            return
        
        # Parse session config
        src_lang = LANGUAGE_MAPPING.get(config.get('src_lang', 'en'), 'eng')
        tgt_lang = LANGUAGE_MAPPING.get(config.get('tgt_lang', 'bn'), 'ben')
        sample_rate = int(config.get('sample_rate', 16000))
        
        logger.info(f"Started session: {src_lang} -> {tgt_lang} at {sample_rate}Hz")
        
        # Send ready confirmation
        await websocket.send_text('{"type":"ready"}')
        
        # Process complete audio recordings (not chunks)
        while True:
            message = await websocket.receive()
            
            if "bytes" in message:
                # Handle complete audio recording from browser
                audio_bytes = message["bytes"]
                logger.info(f"Received complete recording: {len(audio_bytes)} bytes, format signature: {audio_bytes[:12]}")
                
                # Only process if we have substantial audio data (> 1KB to filter out noise)
                if len(audio_bytes) > 1000:
                    # Generate audio fingerprint for debugging
                    audio_hash = hashlib.md5(audio_bytes).hexdigest()[:8]
                    timestamp = int(time.time() * 1000)
                    
                    logger.info(f"Processing audio hash: {audio_hash}, timestamp: {timestamp}")
                    
                    # Convert audio bytes to tensor using our robust format detection
                    waveform = translator.audio_bytes_to_tensor(audio_bytes, sample_rate)
                    
                    if waveform is not None and len(waveform) > sample_rate * 0.5:  # At least 0.5 seconds
                        # Log audio characteristics for debugging
                        audio_stats = {
                            'duration': len(waveform) / sample_rate,
                            'rms': float(np.sqrt(np.mean(waveform.cpu().numpy()**2))),
                            'max_amplitude': float(torch.max(torch.abs(waveform)))
                        }
                        logger.info(f"Audio stats for {audio_hash}: {audio_stats}")
                        
                        # Translate speech
                        try:
                            logger.info(f"Processing complete recording: {len(waveform)} samples ({len(waveform)/sample_rate:.2f} seconds)")
                            translated_audio = await translator.translate_speech_tensor(waveform, src_lang, tgt_lang, sample_rate)
                            
                            if translated_audio is not None:
                                # Generate output fingerprint for debugging
                                output_hash = hashlib.md5(translated_audio).hexdigest()[:8]
                                logger.info(f"Generated translation {audio_hash} -> {output_hash}: {len(translated_audio)} bytes")
                                
                                # Send back as PCM bytes 
                                await websocket.send_bytes(translated_audio)
                                logger.info(f"Sent translated audio: {len(translated_audio)} bytes")
                            else:
                                logger.warning("Translation returned None")
                                
                        except Exception as e:
                            logger.error(f"Translation error: {e}")
                    else:
                        logger.info("Skipping short/invalid audio recording")
                else:
                    logger.info("Skipping small audio chunk (likely noise)")
                    
            elif "text" in message:
                # Handle control messages
                try:
                    ctrl = json.loads(message["text"])
                    if ctrl.get("type") == "stop":
                        break
                except Exception:
                    pass
        
        # Send end message
        await websocket.send_text('{"type":"end","reason":"completed"}')
        
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_text(f'{{"type":"error","msg":"{str(e)}"}}')
        except:
            pass

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

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time translation"""
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            # Receive data from client
            data = await websocket.receive_bytes()
            
            # Parse the message (expecting JSON header + audio data)
            try:
                # First 4 bytes indicate header length
                header_length = int.from_bytes(data[:4], byteorder='little')
                header_json = data[4:4+header_length].decode('utf-8')
                audio_data = data[4+header_length:]
                
                header = json.loads(header_json)
                src_lang = header.get('src_lang', 'eng')
                tgt_lang = header.get('tgt_lang', 'ben')
                
                logger.info(f"Received audio chunk: {len(audio_data)} bytes, {src_lang} -> {tgt_lang}")
                
                # Translate the audio
                translated_audio = await translator.translate_speech(
                    audio_data, src_lang, tgt_lang
                )
                
                if translated_audio:
                    # Send translated audio back to client
                    response_header = {
                        'type': 'audio',
                        'src_lang': src_lang,
                        'tgt_lang': tgt_lang,
                        'size': len(translated_audio)
                    }
                    
                    header_bytes = json.dumps(response_header).encode('utf-8')
                    header_length_bytes = len(header_bytes).to_bytes(4, byteorder='little')
                    
                    response_data = header_length_bytes + header_bytes + translated_audio
                    await websocket.send_bytes(response_data)
                    
                    logger.info(f"Sent translated audio: {len(translated_audio)} bytes")
                else:
                    # Send error message
                    error_msg = {"type": "error", "message": "Translation failed"}
                    await websocket.send_text(json.dumps(error_msg))
                    
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error", 
                    "message": "Invalid message format"
                }))
            except Exception as e:
                logger.error(f"Processing error: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error", 
                    "message": f"Processing error: {str(e)}"
                }))
                
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(client_id)

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