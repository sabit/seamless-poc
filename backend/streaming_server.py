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

# Determine the correct path for frontend files based on where the script is run from
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
frontend_path = os.path.join(project_root, "frontend")

# If running from project root, frontend is in current directory
if not os.path.exists(frontend_path):
    frontend_path = "frontend"

app.mount("/static", StaticFiles(directory=frontend_path), name="static")

# Language code mapping
LANGUAGE_MAPPING = {
    "en": "eng",
    "bn": "ben",
}

# Try to import official SeamlessStreaming components
try:
    # Core seamless streaming agents
    from seamless_communication.streaming.agents.seamless_streaming_s2st import SeamlessStreamingS2STAgent
    from seamless_communication.streaming.agents.seamless_streaming_s2t import SeamlessStreamingS2TAgent
    
    # Core SimulEval components
    from simuleval.data.segments import SpeechSegment, EmptySegment
    from simuleval.agents.actions import ReadAction, WriteAction
    
    # Try to find the correct build_agent function
    try:
        from simuleval.agents import build_agent
    except ImportError:
        try:
            from simuleval.agents.agent import build_agent
        except ImportError:
            # If build_agent is not available, we'll create a simple wrapper
            def build_agent(agent_class, args):
                return agent_class(args)
            build_agent = build_agent
    
    # Optional components (not critical)
    try:
        from simuleval.online.agent_pipeline import AgentPipeline
    except ImportError:
        try:
            from simuleval.agents.pipeline import AgentPipeline
        except ImportError:
            AgentPipeline = None
    
    import fairseq2  # Required for seamless models
    
    OFFICIAL_STREAMING = True
    logger.info("✅ Official SeamlessStreaming agents available")
    logger.info("✅ SeamlessStreamingS2STAgent imported successfully")
    logger.info("✅ SimulEval components imported successfully")
    logger.info("✅ fairseq2 available")
    if AgentPipeline:
        logger.info("✅ AgentPipeline available")
    else:
        logger.info("ℹ️  AgentPipeline not available (not critical)")
    logger.info("🎯 Using official streaming implementation")
    
except ImportError as e:
    OFFICIAL_STREAMING = False
    logger.error(f"❌ Official SeamlessStreaming not available: {e}")
    logger.info("ℹ️  Falling back to improved chunked implementation")
    
    # Fallback imports - try both v2 and v1 models
    try:
        # Try v2 model first (better quality)
        try:
            from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import SeamlessM4Tv2ForSpeechToSpeech as SeamlessM4TForSpeechToSpeech
            from transformers.models.seamless_m4t.processing_seamless_m4t import SeamlessM4TProcessor
            FALLBACK_MODEL = "facebook/seamless-m4t-v2-large"
            logger.info("📦 Using SeamlessM4T v2 fallback model")
        except ImportError:
            # Fallback to v1 model
            from transformers import SeamlessM4TForSpeechToSpeech, SeamlessM4TProcessor
            FALLBACK_MODEL = "facebook/seamless-m4t-large"
            logger.info("📦 Using SeamlessM4T v1 fallback model")
        
        FALLBACK_AVAILABLE = True
        logger.info("✅ Fallback implementation ready")
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
            
            # Build the agent - try build_agent first, fallback to direct instantiation
            try:
                self.agent = build_agent(agent_class, agent_args)
                logger.info("✅ Agent built using build_agent function")
            except Exception as build_error:
                logger.warning(f"⚠️ build_agent failed: {build_error}")
                logger.info("🔄 Trying direct agent instantiation...")
                try:
                    # Convert args dict to object-like structure that agents expect
                    class Args:
                        def __init__(self, **kwargs):
                            for key, value in kwargs.items():
                                setattr(self, key, value)
                    
                    args_obj = Args(**agent_args)
                    self.agent = agent_class(args_obj)
                    logger.info("✅ Agent created using direct instantiation")
                except Exception as direct_error:
                    logger.error(f"❌ Direct instantiation also failed: {direct_error}")
                    raise direct_error
            
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
            logger.info(f"Loading fallback model: {FALLBACK_MODEL}")
            self.model = SeamlessM4TForSpeechToSpeech.from_pretrained(
                FALLBACK_MODEL,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            
            self.processor = SeamlessM4TProcessor.from_pretrained(FALLBACK_MODEL)
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
    # Use the same frontend path logic
    index_path = os.path.join(frontend_path, "index.html")
    if not os.path.exists(index_path):
        index_path = "frontend/index.html"
    return FileResponse(index_path)

@app.get("/health")
async def health_check():
    # Check fallback availability at runtime
    try:
        fallback_available = FALLBACK_AVAILABLE
    except NameError:
        try:
            from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import SeamlessM4Tv2ForSpeechToSpeech
            fallback_available = True
        except ImportError:
            try:
                from transformers import SeamlessM4TForSpeechToSpeech
                fallback_available = True
            except ImportError:
                fallback_available = False
    
    return {
        "status": "healthy",
        "official_streaming": OFFICIAL_STREAMING,
        "fallback_available": fallback_available
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
    
    # Check if fallback is available (it should be since we imported it earlier)
    try:
        fallback_status = FALLBACK_AVAILABLE
    except NameError:
        # FALLBACK_AVAILABLE not defined, check if we can import fallback components
        try:
            from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import SeamlessM4Tv2ForSpeechToSpeech
            fallback_status = True
        except ImportError:
            try:
                from transformers import SeamlessM4TForSpeechToSpeech
                fallback_status = True
            except ImportError:
                fallback_status = False
    
    print(f"🔄 Fallback available: {fallback_status}")
    print(f"⚡ CUDA: {torch.cuda.is_available()}")
    
    # SSL configuration
    ssl_config = None
    if args.ssl_keyfile and args.ssl_certfile:
        import ssl
        
        # Use modern SSL context creation with better compatibility
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        
        # Set minimum TLS version based on user preference
        if args.ssl_version == 'TLSv1_2':
            ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
            ssl_context.maximum_version = ssl.TLSVersion.TLSv1_3  # Allow TLS 1.3
        elif args.ssl_version == 'TLSv1_1':
            ssl_context.minimum_version = ssl.TLSVersion.TLSv1_1
        else:  # TLSv1
            ssl_context.minimum_version = ssl.TLSVersion.TLSv1
        
        # Configure cipher suites for better compatibility
        ssl_context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')
        
        # Load certificate and key with error handling
        try:
            ssl_context.load_cert_chain(args.ssl_certfile, args.ssl_keyfile)
            print(f"✅ SSL certificate and key loaded successfully")
        except ssl.SSLError as ssl_err:
            print(f"❌ SSL certificate/key error: {ssl_err}")
            print(f"💡 Try regenerating certificates with: chmod +x scripts/fix_ssl_cert.sh && ./scripts/fix_ssl_cert.sh")
            exit(1)
        except Exception as cert_err:
            print(f"❌ Certificate loading error: {cert_err}")
            exit(1)
        
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
    
    if not OFFICIAL_STREAMING and not fallback_status:
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
        ssl_ca_certs=args.ssl_ca_certs
    )