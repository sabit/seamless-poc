"""
Proper SeamlessStreaming Implementation
Based on the official Facebook Research SeamlessStreaming architecture
"""

import asyncio
import json
import logging
import numpy as np
import torch
import torchaudio
from typing import Dict, Any, Optional, List, Generator
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import time
from collections import deque
import threading
from queue import Queue, Empty

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SeamlessStreaming Translation API")
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Language code mapping
LANGUAGE_MAPPING = {
    "en": "eng",
    "bn": "ben",
}

# Try to import official SeamlessStreaming components
try:
    from seamless_communication.streaming.agents.seamless_streaming_s2st import SeamlessStreamingS2STAgent
    from seamless_communication.streaming.agents.seamless_streaming_s2t import SeamlessStreamingS2TAgent
    from simuleval.data.segments import SpeechSegment, EmptySegment, Segment
    from simuleval.agents.actions import ReadAction, WriteAction
    from simuleval.online.agent_pipeline import AgentPipeline
    from simuleval.agents import build_agent
    import fairseq2  # Required for seamless models
    OFFICIAL_STREAMING = True
    logger.info("✅ Official SeamlessStreaming agents available")
except ImportError as e:
    OFFICIAL_STREAMING = False
    logger.error(f"❌ Official SeamlessStreaming not available: {e}")
    logger.warning("Please install seamless_communication properly for official streaming support")
    
    # Fallback imports
    try:
        from transformers import SeamlessM4TForSpeechToSpeech, SeamlessM4TProcessor
        FALLBACK_AVAILABLE = True
        logger.info("📦 Fallback to transformers SeamlessM4T")
    except ImportError:
        FALLBACK_AVAILABLE = False
        logger.error("❌ No translation models available")

class OfficialStreamingTranslator:
    """Uses the official SeamlessStreaming agents for proper streaming translation"""
    
    def __init__(self):
        self.agent = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sample_rate = 16000
        self.initialized = False
        
    async def initialize(self, task="s2st", src_lang="eng", tgt_lang="ben"):
        """Initialize the official streaming agent"""
        try:
            logger.info(f"Initializing official SeamlessStreaming agent for {task}: {src_lang} → {tgt_lang}")
            
            # Configure agent arguments
            agent_args = {
                "device": self.device,
                "dtype": "fp16" if self.device == "cuda" else "fp32",
                "task": task,
                "tgt_lang": tgt_lang,
                "src_lang": src_lang,
                # Unity model for speech encoder and T2U
                "unity_model_name": "seamless_streaming_unity",
                # Monotonic decoder for streaming text generation
                "monotonic_decoder_model_name": "seamless_streaming_monotonic_decoder",
                # VAD settings
                "vad": True,
                "vad_chunk_size": 480,  # 30ms chunks at 16kHz
                # Latency control
                "min_starting_wait": 1000,  # Wait for 1 second of audio
                "max_len_a": 0.0,
                "max_len_b": 100,
                # Quality settings
                "beam_size": 3,
                "no_repeat_ngram_size": 3,
            }
            
            # Select appropriate agent class
            if task == "s2st":
                agent_class = SeamlessStreamingS2STAgent
            elif task == "s2t":
                agent_class = SeamlessStreamingS2TAgent
            else:
                raise ValueError(f"Unsupported task: {task}")
            
            # Build the agent
            self.agent = build_agent(agent_class, agent_args)
            self.initialized = True
            
            logger.info("✅ Official streaming agent initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize official streaming agent: {e}")
            self.initialized = False
            raise e
    
    def process_audio_segment(self, audio_data: np.ndarray, finished: bool = False) -> List[bytes]:
        """Process audio segment and return any generated output"""
        if not self.initialized or self.agent is None:
            logger.error("Agent not initialized")
            return []
        
        try:
            # Create speech segment
            if len(audio_data) > 0:
                segment = SpeechSegment(
                    content=audio_data,
                    sample_rate=self.sample_rate,
                    finished=finished
                )
            else:
                segment = EmptySegment(finished=finished)
            
            # Process with agent
            action = self.agent.policy(segment)
            
            results = []
            if isinstance(action, WriteAction):
                # Got output from the agent
                if hasattr(action, 'content') and action.content is not None:
                    if isinstance(action.content, torch.Tensor):
                        # Convert audio tensor to bytes
                        audio_np = action.content.detach().cpu().numpy()
                        if audio_np.ndim > 1:
                            audio_np = audio_np.squeeze()
                        
                        # Normalize and convert to PCM
                        if len(audio_np) > 0:
                            audio_np = audio_np / max(abs(audio_np.max()), abs(audio_np.min()), 1e-8)
                            audio_16bit = (audio_np * 32767).astype(np.int16)
                            results.append(audio_16bit.tobytes())
                    elif isinstance(action.content, str):
                        # Text output (for S2T task)
                        results.append(action.content.encode('utf-8'))
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing audio segment: {e}")
            return []

class FallbackStreamingTranslator:
    """Fallback implementation using transformers SeamlessM4T"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sample_rate = 16000
        self.buffer = deque(maxlen=int(self.sample_rate * 10))  # 10 second buffer
        self.min_chunk_duration = 3.0  # Process every 3 seconds
        self.last_process_time = 0
        
    async def initialize(self, src_lang="eng", tgt_lang="ben"):
        """Initialize the fallback model"""
        try:
            logger.info("Loading fallback SeamlessM4T model...")
            self.model = SeamlessM4TForSpeechToSpeech.from_pretrained(
                "facebook/seamless-m4t-large",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            
            self.processor = SeamlessM4TProcessor.from_pretrained("facebook/seamless-m4t-large")
            logger.info("✅ Fallback model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load fallback model: {e}")
            raise e
    
    def process_audio_segment(self, audio_data: np.ndarray, finished: bool = False) -> List[bytes]:
        """Process audio with fallback chunked approach"""
        current_time = time.time()
        
        # Add to buffer
        if len(audio_data) > 0:
            self.buffer.extend(audio_data)
        
        # Check if we should process
        should_process = (
            finished or
            (current_time - self.last_process_time >= self.min_chunk_duration and 
             len(self.buffer) >= int(self.sample_rate * 2))  # At least 2 seconds
        )
        
        if not should_process:
            return []
        
        try:
            # Get audio chunk
            chunk = np.array(list(self.buffer))
            if len(chunk) == 0:
                return []
            
            # Simple quality check
            audio_energy = np.sqrt(np.mean(chunk ** 2))
            if audio_energy < 0.01:
                self.buffer.clear()
                return []
            
            # Process with model
            waveform = torch.from_numpy(chunk).float().unsqueeze(0)
            
            inputs = self.processor(
                audio=waveform,
                sampling_rate=self.sample_rate,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                audio_array = self.model.generate(
                    **inputs,
                    tgt_lang="ben",  # Hardcoded for now
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=256
                )
            
            # Convert output
            if isinstance(audio_array, (list, tuple)) and len(audio_array) > 0:
                audio_tensor = audio_array[0]
                audio_np = audio_tensor.squeeze().detach().cpu().numpy()
                
                if len(audio_np) > 0:
                    # Normalize and convert
                    audio_np = audio_np / max(abs(audio_np.max()), abs(audio_np.min()), 1e-8)
                    audio_16bit = (audio_np * 32767).astype(np.int16)
                    
                    # Clear buffer and update time
                    self.buffer.clear()
                    self.last_process_time = current_time
                    
                    return [audio_16bit.tobytes()]
            
            return []
            
        except Exception as e:
            logger.error(f"Fallback processing error: {e}")
            return []

class StreamingSession:
    """Manages a streaming translation session"""
    
    def __init__(self, websocket: WebSocket, src_lang: str, tgt_lang: str):
        self.websocket = websocket
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.translator = None
        self.running = False
        self.sample_rate = 16000
        
    async def initialize(self):
        """Initialize the session with appropriate translator"""
        if OFFICIAL_STREAMING:
            self.translator = OfficialStreamingTranslator()
            await self.translator.initialize(
                task="s2st",
                src_lang=self.src_lang,
                tgt_lang=self.tgt_lang
            )
            logger.info("🎯 Using official SeamlessStreaming")
        elif FALLBACK_AVAILABLE:
            self.translator = FallbackStreamingTranslator()
            await self.translator.initialize(self.src_lang, self.tgt_lang)
            logger.info("🔄 Using fallback implementation")
        else:
            raise RuntimeError("No translation backend available")
    
    async def process_audio_chunk(self, audio_data: bytes):
        """Process incoming audio chunk"""
        try:
            # Convert bytes to numpy array
            if len(audio_data) % 2 != 0:
                audio_data = audio_data + b'\x00'
            
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            logger.info(f"Processing audio chunk: {len(audio_np)} samples")
            
            # Process with translator
            results = self.translator.process_audio_segment(audio_np)
            
            # Send any results
            for result in results:
                if result and len(result) > 0:
                    await self.websocket.send_bytes(result)
                    logger.info(f"Sent translation: {len(result)} bytes")
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
    
    async def finalize(self):
        """Finalize the session"""
        if self.translator:
            try:
                # Process any remaining audio
                results = self.translator.process_audio_segment(np.array([]), finished=True)
                for result in results:
                    if result and len(result) > 0:
                        await self.websocket.send_bytes(result)
            except Exception as e:
                logger.error(f"Error finalizing session: {e}")

class ConnectionManager:
    def __init__(self):
        self.sessions: Dict[str, StreamingSession] = {}
    
    async def create_session(self, websocket: WebSocket, client_id: str, src_lang: str, tgt_lang: str):
        session = StreamingSession(websocket, src_lang, tgt_lang)
        await session.initialize()
        self.sessions[client_id] = session
        return session
    
    def remove_session(self, client_id: str):
        if client_id in self.sessions:
            del self.sessions[client_id]

manager = ConnectionManager()

@app.get("/")
async def root():
    return FileResponse("frontend/index.html")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "official_streaming": OFFICIAL_STREAMING,
        "fallback_available": FALLBACK_AVAILABLE
    }

@app.websocket("/ws/stream")
async def websocket_streaming_translate(websocket: WebSocket):
    """WebSocket endpoint for proper streaming translation"""
    await websocket.accept()
    session = None
    client_id = None
    
    try:
        # Wait for configuration
        config_msg = await websocket.receive_text()
        config = json.loads(config_msg)
        
        if config.get("type") != "start_stream":
            await websocket.send_text('{"type":"error","msg":"Expected start_stream message"}')
            return
        
        # Parse config
        src_lang = LANGUAGE_MAPPING.get(config.get('src_lang', 'en'), 'eng')
        tgt_lang = LANGUAGE_MAPPING.get(config.get('tgt_lang', 'bn'), 'ben')
        
        logger.info(f"🎤 Starting streaming session: {src_lang} → {tgt_lang}")
        
        # Create session
        client_id = f"stream_{int(time.time())}"
        session = await manager.create_session(websocket, client_id, src_lang, tgt_lang)
        session.running = True
        
        # Send ready confirmation
        await websocket.send_text('{"type":"stream_ready"}')
        
        # Process audio stream
        while session.running:
            message = await websocket.receive()
            
            if "bytes" in message:
                audio_chunk = message["bytes"]
                if len(audio_chunk) > 0:
                    await session.process_audio_chunk(audio_chunk)
                    
            elif "text" in message:
                try:
                    ctrl = json.loads(message["text"])
                    if ctrl.get("type") == "stop_stream":
                        logger.info("🛑 Stopping stream")
                        session.running = False
                        if session:
                            await session.finalize()
                        break
                except Exception:
                    pass
        
        # Cleanup
        if client_id:
            manager.remove_session(client_id)
        await websocket.send_text('{"type":"stream_end"}')
        
    except WebSocketDisconnect:
        logger.info("📡 Client disconnected")
        if client_id:
            manager.remove_session(client_id)
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
        if client_id:
            manager.remove_session(client_id)
        try:
            await websocket.send_text(f'{{"type":"error","msg":"{str(e)}"}}')
        except:
            pass

if __name__ == "__main__":
    import os
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='SeamlessStreaming Translation Service')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=7860, help='Port to bind to (default: 7860)')
    parser.add_argument('--ssl-keyfile', help='Path to SSL private key file')
    parser.add_argument('--ssl-certfile', help='Path to SSL certificate file')
    parser.add_argument('--ssl-ca-certs', help='Path to SSL CA certificates file (optional)')
    parser.add_argument('--ssl-version', choices=['TLSv1', 'TLSv1_1', 'TLSv1_2'], 
                       default='TLSv1_2', help='SSL version (default: TLSv1_2)')
    parser.add_argument('--no-ssl-verify', action='store_true', 
                       help='Disable SSL certificate verification')
    args = parser.parse_args()
    
    print("🌊 SeamlessStreaming Translation Service")
    print(f"🔧 Official streaming: {OFFICIAL_STREAMING}")
    print(f"🔄 Fallback available: {FALLBACK_AVAILABLE}")
    print(f"⚡ CUDA: {torch.cuda.is_available()}")
    
    # SSL configuration
    ssl_config = None
    if args.ssl_keyfile and args.ssl_certfile:
        import ssl
        
        ssl_context = ssl.SSLContext(getattr(ssl, f"PROTOCOL_{args.ssl_version}"))
        ssl_context.load_cert_chain(args.ssl_certfile, args.ssl_keyfile)
        
        if args.ssl_ca_certs:
            ssl_context.load_verify_locations(args.ssl_ca_certs)
        
        if args.no_ssl_verify:
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
        else:
            ssl_context.verify_mode = ssl.CERT_REQUIRED
        
        ssl_config = ssl_context
        print(f"🔒 SSL enabled: {args.ssl_certfile}")
        print(f"🔑 SSL key: {args.ssl_keyfile}")
        if args.ssl_ca_certs:
            print(f"📜 SSL CA: {args.ssl_ca_certs}")
        protocol = "https"
    else:
        print("⚠️  SSL not configured - using HTTP")
        protocol = "http"
    
    print(f"🌐 Server URL: {protocol}://{args.host}:{args.port}")
    print("=" * 50)
    
    if not OFFICIAL_STREAMING and not FALLBACK_AVAILABLE:
        print("❌ No translation backend available!")
        exit(1)
    
    # Run server with SSL if configured
    uvicorn.run(
        "streaming_server:app",
        host=args.host,
        port=args.port,
        log_level="info",
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        ssl_ca_certs=args.ssl_ca_certs,
        ssl_version=getattr(__import__('ssl'), f"PROTOCOL_{args.ssl_version}") if ssl_config else None
    )