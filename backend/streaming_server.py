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

# Check CUDA availability and enforce GPU usage
if not torch.cuda.is_available():
    logger.error("❌ CUDA not available! SeamlessStreaming requires GPU")
    logger.error("💡 This model does not run on CPU - GPU is mandatory")
    raise RuntimeError("CUDA required for SeamlessStreaming - model does not run on CPU")

logger.info(f"✅ CUDA available: {torch.cuda.is_available()}")
logger.info(f"🎮 GPU device: {torch.cuda.get_device_name()}")
logger.info(f"📦 PyTorch version: {torch.__version__}")

# Language code mapping for SeamlessStreaming
LANG_MAPPING = {
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
    logger.error(f"❌ Official SeamlessStreaming not available: {e}")
    logger.error("💡 Please install seamless-communication package:")
    logger.error("   pip install git+https://github.com/facebookresearch/seamless_communication.git")


class OfficialStreamingTranslator:
    """Official SeamlessStreaming implementation"""
    
    def __init__(self):
        self.agent = None
        self.initialized = False
        self.device = torch.device("cuda")
        
    def _create_official_args(self, task="s2st", src_lang="eng", tgt_lang="ben"):
        """Create arguments object for SeamlessStreaming agent"""
        
        # Import the Args class from SimulEval
        try:
            from argparse import Namespace
            args = Namespace()
        except ImportError:
            # Fallback to simple object
            class Args:
                pass
            args = Args()
        
        # Model configuration - using unity and monotonic multipath models  
        args.model_name = "seamless_streaming_unity"
        args.vocoder_name = "vocoder_pretssel" 
        args.unity_model_name = "seamless_streaming_unity"
        
        # Device and precision configuration
        args.device = torch.device("cuda")
        args.dtype = torch.float16
        
        # Language configuration
        args.source_lang = src_lang
        args.target_lang = tgt_lang
        args.task = task
        
        # Streaming configuration parameters
        args.min_unit_chunk_size = 50  # Minimum number of units to accumulate
        args.d_factor = 1.0  # Duration factor for timing
        args.shift_size = 160  # Audio frame shift size
        args.segment_size = 2000  # Audio segment size
        args.denormalize = False  # Whether to denormalize the output
        
        # Text generation parameters
        args.max_len_a = 1.2  # Length penalty coefficient a
        args.max_len_b = 100   # Length penalty coefficient b  
        args.beam_size = 5     # Beam search size
        args.len_penalty = 1.0 # Length penalty
        
        # Buffer and timing configuration
        args.buffer_size = 1000
        args.sample_rate = 16000
        
        logger.info(f"🔧 Creating args for SeamlessStreaming agent:")
        logger.info(f"   📋 Task: {args.task}")
        logger.info(f"   🔧 Unity model: {args.unity_model_name}")
        logger.info(f"   💾 Device: {args.device} (type: {args.device.type})")
        logger.info(f"   📊 Dtype: {args.dtype}")
        logger.info(f"   🔢 Min unit chunk size: {args.min_unit_chunk_size}")
        logger.info(f"   ⏱️  Duration factor: {args.d_factor}")
        logger.info(f"   🔄 Shift size: {args.shift_size}")
        logger.info(f"   📐 Segment size: {args.segment_size}")
        logger.info(f"   📏 Max len a: {args.max_len_a}")
        logger.info(f"   📏 Max len b: {args.max_len_b}")
        logger.info(f"   🔍 Beam size: {args.beam_size}")
        logger.info(f"   ⚖️  Len penalty: {args.len_penalty}")
        logger.info(f"   🗣️  Languages: {args.source_lang} → {args.target_lang}")
        
        return args
        
    async def initialize(self, task="s2st", src_lang="eng", tgt_lang="ben"):
        """Initialize the official streaming agent"""
        try:
            logger.info(f"Initializing official SeamlessStreaming agent for {task}: {src_lang} → {tgt_lang}")
            
            # Configure agent arguments with all required parameters
            args = self._create_official_args(task, src_lang, tgt_lang)
            
            # Initialize the streaming agent with proper configuration
            logger.info("🔧 Creating SeamlessStreamingS2STAgent...")
            self.agent = SeamlessStreamingS2STAgent(args)
            
            logger.info("✅ Official SeamlessStreaming agent initialized successfully!")
            logger.info("🎯 Agent is ready for streaming translation")
            
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize official streaming agent: {e}")
            logger.error(f"💡 Error type: {type(e).__name__}")
            import traceback
            logger.error(f"📋 Traceback: {traceback.format_exc()}")
            return False
    
    async def translate_stream(self, audio_chunk: bytes) -> Optional[str]:
        """Process audio chunk and return translation if available"""
        if not self.initialized or not self.agent:
            logger.warning("⚠️  Agent not initialized")
            return None
            
        try:
            # Convert audio bytes to numpy array
            audio_np = np.frombuffer(audio_chunk, dtype=np.float32)
            
            # Create speech segment for the agent
            segment = SpeechSegment(
                content=audio_np,
                sample_rate=16000,
                finished=False
            )
            
            # Process with the streaming agent
            action = self.agent.policy(segment)
            
            # Check if we have translation output
            if hasattr(action, 'content') and action.content:
                logger.info(f"🎯 Translation: {action.content}")
                return action.content
                
            return None
            
        except Exception as e:
            logger.error(f"❌ Translation error: {e}")
            import traceback
            logger.error(f"📋 Traceback: {traceback.format_exc()}")
            return None


class StreamingSession:
    """Manages a streaming translation session using official implementation only"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.official_translator = OfficialStreamingTranslator()
        self.audio_buffer = deque(maxlen=100)
        self.is_active = False
        self.src_lang = "eng"
        self.tgt_lang = "ben"
        self.task = "s2st"
        
    async def initialize(self, src_lang: str = "eng", tgt_lang: str = "ben", task: str = "s2st"):
        """Initialize the streaming session"""
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.task = task
        
        logger.info(f"🎬 Initializing session {self.session_id}")
        logger.info(f"🗣️  Task: {task}, Languages: {src_lang} → {tgt_lang}")
        
        # Initialize official translator
        success = await self.official_translator.initialize(task, src_lang, tgt_lang)
        
        if success:
            self.is_active = True
            logger.info(f"✅ Session {self.session_id} ready with official SeamlessStreaming")
            return True
        else:
            logger.error(f"❌ Session {self.session_id} failed to initialize")
            return False
    
    async def process_audio(self, audio_chunk: bytes) -> Optional[str]:
        """Process audio chunk and return translation"""
        if not self.is_active:
            return None
            
        try:
            # Process with official translator
            result = await self.official_translator.translate_stream(audio_chunk)
            
            if result:
                logger.info(f"📤 Session {self.session_id} translation: {result}")
                return result
                
            return None
            
        except Exception as e:
            logger.error(f"❌ Session {self.session_id} processing error: {e}")
            return None
    
    def close(self):
        """Close the streaming session"""
        self.is_active = False
        logger.info(f"🔚 Session {self.session_id} closed")


# Session manager
sessions: Dict[str, StreamingSession] = {}

# FastAPI app setup
app = FastAPI(title="SeamlessStreaming API", version="1.0.0")

# Mount frontend static files
app.mount("/static", StaticFiles(directory="../frontend"), name="static")

@app.get("/")
async def read_root():
    """Serve the frontend HTML"""
    return FileResponse("../frontend/index.html")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "cuda_available": torch.cuda.is_available()}

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for streaming translation"""
    await websocket.accept()
    session_id = f"session_{int(time.time())}"
    session = StreamingSession(session_id)
    sessions[session_id] = session
    
    logger.info(f"🔌 New WebSocket connection: {session_id}")
    
    try:
        # Wait for initialization message
        init_data = await websocket.receive_json()
        src_lang = LANG_MAPPING.get(init_data.get("srcLang", "en"), "eng")
        tgt_lang = LANG_MAPPING.get(init_data.get("tgtLang", "bn"), "ben")
        task = init_data.get("task", "s2st")
        
        # Initialize session
        success = await session.initialize(src_lang, tgt_lang, task)
        
        if success:
            await websocket.send_json({
                "type": "init_success",
                "message": "SeamlessStreaming ready"
            })
            
            # Main processing loop
            while True:
                try:
                    data = await websocket.receive()
                    
                    if "bytes" in data:
                        # Process audio data
                        audio_data = data["bytes"]
                        result = await session.process_audio(audio_data)
                        
                        if result:
                            await websocket.send_json({
                                "type": "translation",
                                "text": result,
                                "session_id": session_id
                            })
                    
                    elif "text" in data:
                        # Handle JSON messages
                        json_data = json.loads(data["text"])
                        if json_data.get("type") == "end_session":
                            logger.info(f"🔚 End session requested for {session_id}")
                            break
                            
                except WebSocketDisconnect:
                    logger.info(f"🔌 WebSocket disconnected: {session_id}")
                    break
                except Exception as e:
                    logger.error(f"❌ WebSocket error: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Processing error: {str(e)}"
                    })
        else:
            await websocket.send_json({
                "type": "init_error", 
                "message": "Failed to initialize SeamlessStreaming"
            })
            
    except Exception as e:
        logger.error(f"❌ WebSocket setup error: {e}")
        await websocket.send_json({
            "type": "error",
            "message": f"Connection error: {str(e)}"
        })
    finally:
        # Cleanup
        session.close()
        if session_id in sessions:
            del sessions[session_id]


# SSL Configuration
def create_ssl_context():
    """Create SSL context for secure connections"""
    import ssl
    
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.maximum_version = ssl.TLSVersion.TLSv1_3
    
    # Security settings
    context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    
    try:
        # Try to load SSL certificate and key
        context.load_cert_chain("../ssl/cert.pem", "../ssl/key.pem")
        logger.info("✅ SSL certificate loaded successfully")
        return context
    except FileNotFoundError:
        logger.warning("⚠️  SSL certificate not found, generating new one...")
        
        # Generate SSL certificate
        import subprocess
        import os
        
        # Create SSL directory
        os.makedirs("../ssl", exist_ok=True)
        
        # Generate private key and certificate
        try:
            subprocess.run([
                "openssl", "req", "-x509", "-newkey", "rsa:4096", 
                "-keyout", "../ssl/key.pem", "-out", "../ssl/cert.pem",
                "-days", "365", "-nodes", "-subj", "/CN=localhost"
            ], check=True, capture_output=True)
            
            # Load the generated certificate
            context.load_cert_chain("../ssl/cert.pem", "../ssl/key.pem")
            logger.info("✅ SSL certificate generated and loaded")
            return context
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to generate SSL certificate: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ SSL setup error: {e}")
            return None


if __name__ == "__main__":
    import ssl
    
    logger.info("🚀 Starting SeamlessStreaming Server...")
    logger.info("🎯 Official SeamlessStreaming implementation")
    logger.info("💪 CUDA-only mode (no CPU fallback)")
    
    # SSL configuration
    ssl_context = create_ssl_context()
    
    if ssl_context:
        logger.info("🔒 Starting server with SSL support")
        logger.info("🌐 HTTPS: https://localhost:8000")
        logger.info("🔌 WSS: wss://localhost:8000/ws/stream")
        
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8000,
            ssl_keyfile="../ssl/key.pem",
            ssl_certfile="../ssl/cert.pem",
            ssl_version=ssl.PROTOCOL_TLS,
            ssl_cert_reqs=ssl.CERT_NONE,
            log_level="info"
        )
    else:
        logger.info("🌐 Starting server without SSL")
        logger.info("🌐 HTTP: http://localhost:8000")
        logger.info("🔌 WS: ws://localhost:8000/ws/stream")
        
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8000,
            log_level="info"
        )