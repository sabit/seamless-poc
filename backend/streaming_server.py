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

# Configure logging - FOCUSED MODE
logging.basicConfig(level=logging.WARNING)  # Reduce general logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Keep our logger at INFO level

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
    
    def __init__(self, source_lang="eng", target_lang="ben", task="s2st", auto_init=False):
        self.agent = None
        self.agent_states = None
        self.initialized = False
        self.device = torch.device("cuda")
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.task = task
        self.args = None  # Store args for test access
        
        # Audio accumulation for streaming
        self.audio_accumulator = []
        self.total_samples = 0
        self.last_chunk_time = None
        
        # Auto-initialize if requested (useful for testing)
        if auto_init:
            import asyncio
            asyncio.run(self.initialize())
        
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
        args.vocoder_name = "vocoder_v2"  # Standard vocoder for SeamlessStreaming
        args.vocoder_speaker_id = -1  # Default speaker ID for vocoder
        args.unity_model_name = "seamless_streaming_unity"
        args.monotonic_decoder_model_name = "seamless_streaming_monotonic_decoder"
        
        # Device and precision configuration
        args.device = torch.device("cuda")
        args.dtype = torch.float16
        args.fp16 = True  # Enable fp16 precision
        
        # Language configuration
        args.source_lang = src_lang
        args.target_lang = tgt_lang
        args.src_lang = src_lang  # Alternative naming for compatibility
        args.tgt_lang = tgt_lang  # Alternative naming for compatibility
        args.task = task
        
        # Text decoder specific language parameters
        args.lang_pairs = f"{src_lang}-{tgt_lang}"  # Language pair specification
        args.target_language = tgt_lang  # Additional target language parameter
        
        # Streaming configuration parameters (optimized from research)
        args.min_unit_chunk_size = 25  # REDUCED: Less accumulation required (was 50)
        args.source_segment_size = 320  # Critical: segment size from research
        args.d_factor = 1.0  # Duration factor for timing
        args.shift_size = 160  # Audio frame shift size
        args.segment_size = 2000  # Audio segment size
        args.window_size = 2000  # Feature extraction window size
        args.feature_dim = 80  # Feature dimension (e.g., mel-spectrogram features)
        args.min_starting_wait_w2vbert = 192  # FIXED: From research (was 1000)
        args.max_consecutive_write = 10  # Maximum consecutive writes for text decoder
        args.min_starting_wait = 192  # FIXED: Match w2vbert parameter (was 1000)
        args.no_early_stop = False  # Disable early stopping for streaming
        args.decision_threshold = 0.5  # FIXED: From research (was 0.7)
        args.decision_method = "threshold"  # Decision method for text decoder
        args.block_ngrams = False  # Block repeated n-grams in text generation
        args.p_choose_start_layer = 0  # Layer to start choosing from in decoder
        args.denormalize = False  # Whether to denormalize the output
        
        # Text generation parameters
        args.max_len_a = 1.2  # Length penalty coefficient a
        args.max_len_b = 100   # Length penalty coefficient b  
        args.beam_size = 5     # Beam search size
        args.len_penalty = 1.0 # Length penalty
        
        # Buffer and timing configuration
        args.buffer_size = 1000
        args.sample_rate = 16000
        
        # FOCUSED: Only log critical parameters
        logger.info(f"� SeamlessStreaming config: {src_lang}→{tgt_lang}, min_wait={args.min_starting_wait_w2vbert}, threshold={args.decision_threshold}, chunk_size={args.min_unit_chunk_size}")
        
        return args
        
    async def initialize(self, task=None, src_lang=None, tgt_lang=None):
        """Initialize the official streaming agent"""
        try:
            # Use instance variables if parameters not provided
            task = task or self.task
            src_lang = src_lang or self.source_lang
            tgt_lang = tgt_lang or self.target_lang
            
            logger.info(f"🔧 Initializing SeamlessStreaming: {src_lang}→{tgt_lang}")
            
            # Configure agent arguments with all required parameters
            args = self._create_official_args(task, src_lang, tgt_lang)
            self.args = args  # Store for test access
            
            # Initialize the streaming agent with proper configuration
            self.agent = SeamlessStreamingS2STAgent(args)
            
            logger.info("✅ Agent initialized and ready")
            
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize official streaming agent: {e}")
            logger.error(f"💡 Error type: {type(e).__name__}")
            import traceback
            logger.error(f"📋 Traceback: {traceback.format_exc()}")
            return False
    
    def reset_accumulator(self):
        """Reset audio accumulator for new session"""
        self.audio_accumulator = []
        self.total_samples = 0
        self.last_chunk_time = None
    
    async def translate_stream(self, audio_chunk: bytes) -> Optional[bytes]:
        """Process audio chunk and return translated audio if available"""
        if not self.initialized or not self.agent:
            return None
            
        try:
            # Convert audio bytes to int16 array (frontend sends int16 PCM)
            audio_np = np.frombuffer(audio_chunk, dtype=np.int16)
            
            # Convert to float32 and normalize
            audio_float = audio_np.astype(np.float32) / 32768.0
            
            # Initialize agent states if this is the first chunk
            if not hasattr(self, 'agent_states') or self.agent_states is None:
                self.agent_states = self.agent.build_states()
                logger.info(f"🏗️ First chunk - built states: {type(self.agent_states)}")
                
                # Initialize audio accumulator for streaming
                self.audio_accumulator = []
                self.total_samples = 0
            
            # Accumulate audio samples for streaming processing
            self.audio_accumulator.extend(audio_float)
            self.total_samples += len(audio_float)
            current_time = time.time()
            
            # Determine if segment should be marked as finished
            time_based_finish = (self.last_chunk_time and 
                               (current_time - self.last_chunk_time) > 1.0)
            sample_based_finish = self.total_samples >= 16384  # Finish at 1 second of audio
            force_finish = self.total_samples >= 24576  # 1.5 seconds of audio
            segment_finished = time_based_finish or sample_based_finish or force_finish
            
            # Create speech segment with accumulated audio
            segment = SpeechSegment(
                content=np.array(self.audio_accumulator, dtype=np.float32),
                sample_rate=16000,
                finished=segment_finished
            )
            
            self.last_chunk_time = current_time
            
            # Process with the streaming agent
            try:
                # Update states with current segment
                if isinstance(self.agent_states, list):
                    # Pipeline agents - update each state
                    for state in enumerate(self.agent_states):
                        if hasattr(state, 'source'):
                            state.source = [segment]
                            state.source_finished = segment_finished
                        if hasattr(state, 'tgt_lang'):
                            state.tgt_lang = self.target_lang
                        if hasattr(state, 'target_lang'):
                            state.target_lang = self.target_lang
                        if hasattr(state, 'finished'):
                            state.finished = segment_finished
                    action = self.agent.policy(self.agent_states)
                else:
                    # Single agent
                    self.agent_states.source = [segment]
                    self.agent_states.source_finished = segment_finished
                    if hasattr(self.agent_states, 'tgt_lang'):
                        self.agent_states.tgt_lang = self.target_lang
                    if hasattr(self.agent_states, 'target_lang'):
                        self.agent_states.target_lang = self.target_lang
                    if hasattr(self.agent_states, 'finished'):
                        self.agent_states.finished = segment_finished
                    action = self.agent.policy(self.agent_states)
                
                # FOCUS: Only log critical translation results
                if action is None:
                    if segment_finished:
                        logger.warning(f"❌ PROBLEM: Segment FINISHED ({self.total_samples} samples) but agent returned None!")
                        logger.warning(f"   Parameters: min_wait={getattr(self.args, 'min_starting_wait_w2vbert', 'unknown')}, threshold={getattr(self.args, 'decision_threshold', 'unknown')}")
                    return None
                
                # FOCUS: Log successful translation only
                logger.info(f"✅ Got action: {action.__class__.__name__}")
                
                # Check if action is a ReadAction (needs more input)
                if hasattr(action, '__class__') and 'Read' in action.__class__.__name__:
                    return None
                
                # Check for WriteAction with content
                if hasattr(action, 'content') and action.content is not None:
                    # For s2st, action.content should be audio samples
                    if isinstance(action.content, (list, np.ndarray)):
                        # Convert audio samples to int16 PCM bytes
                        if isinstance(action.content, list):
                            audio_samples = np.array(action.content, dtype=np.float32)
                        else:
                            audio_samples = action.content.astype(np.float32)
                        
                        # Normalize and convert to int16
                        audio_samples = np.clip(audio_samples, -1.0, 1.0)
                        audio_int16 = (audio_samples * 32767).astype(np.int16)
                        
                        logger.info(f"🎉 SUCCESS! Generated {len(audio_int16)} audio samples")
                        
                        # Clear accumulator after successful translation
                        self.audio_accumulator = []
                        self.total_samples = 0
                        
                        return audio_int16.tobytes()
                    else:
                        logger.warning(f"⚠️ Unexpected content type: {type(action.content)}")
                
                # Check for text content (in case of s2t mode)
                if hasattr(action, 'content') and isinstance(action.content, str):
                    logger.info(f"📝 Got text: {action.content}")
                    return None
                
                return None
                
            except Exception as policy_error:
                logger.error(f"❌ Agent policy error: {policy_error}")
                import traceback
                logger.error(f"📋 Policy traceback: {traceback.format_exc()}")
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
        
        logger.info(f"🎬 New session: {src_lang}→{tgt_lang}")
        
        # Initialize official translator
        success = await self.official_translator.initialize(task, src_lang, tgt_lang)
        
        if success:
            self.official_translator.reset_accumulator()
            self.is_active = True
            logger.info(f"✅ Session ready")
            return True
        else:
            logger.error(f"❌ Session failed to initialize")
            return False
    
    async def process_audio(self, audio_chunk: bytes) -> Optional[bytes]:
        """Process audio chunk and return translated audio"""
        if not self.is_active:
            return None
            
        try:
            # Process with official translator
            result = await self.official_translator.translate_stream(audio_chunk)
            return result
            
        except Exception as e:
            logger.error(f"❌ Processing error: {e}")
            return None
    
    def close(self):
        """Close the streaming session"""
        self.is_active = False


# Session manager
sessions: Dict[str, StreamingSession] = {}

# FastAPI app setup
app = FastAPI(title="SeamlessStreaming API", version="1.0.0")

# Mount frontend static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def read_root():
    """Serve the frontend HTML"""
    return FileResponse("frontend/index.html")

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
                "type": "stream_ready",
                "message": "SeamlessStreaming ready"
            })
            
            # Main processing loop with timeout
            while True:
                try:
                    # Add timeout to prevent hanging on receive
                    data = await asyncio.wait_for(websocket.receive(), timeout=60.0)
                    
                    if "bytes" in data:
                        # Process audio data
                        audio_data = data["bytes"]
                        
                        try:
                            result = await session.process_audio(audio_data)
                            
                            if result:
                                # Send translated audio as binary data
                                await websocket.send_bytes(result)
                        except Exception as process_error:
                            logger.error(f"❌ Audio processing error: {process_error}")
                            # Continue processing instead of breaking the connection
                            await websocket.send_json({
                                "type": "processing_error",
                                "message": "Audio processing failed, but connection remains active"
                            })
                    
                    elif "text" in data:
                        # Handle JSON messages
                        json_data = json.loads(data["text"])
                        if json_data.get("type") == "end_session":
                            logger.info(f"🔚 End session requested for {session_id}")
                            break
                        elif json_data.get("type") == "ping":
                            # Respond to heartbeat ping with pong
                            try:
                                await websocket.send_json({"type": "pong"})
                            except Exception as ping_error:
                                logger.error(f"❌ Failed to send pong: {ping_error}")
                        elif json_data.get("type") == "stop_stream":
                            # Don't break, just acknowledge
                            try:
                                await websocket.send_json({"type": "stream_stopped"})
                            except Exception as stop_error:
                                logger.error(f"❌ Failed to send stream_stopped: {stop_error}")
                            
                except WebSocketDisconnect:
                    logger.info(f"🔌 WebSocket disconnected: {session_id}")
                    break
                except asyncio.TimeoutError:
                    logger.warning(f"⏰ WebSocket timeout for {session_id}")
                    try:
                        await websocket.send_json({
                            "type": "timeout_warning",
                            "message": "Connection timeout, please check your network"
                        })
                    except:
                        pass  # Connection might be dead
                except Exception as e:
                    logger.error(f"❌ WebSocket error: {e}")
                    import traceback
                    logger.error(f"📋 Error traceback: {traceback.format_exc()}")
                    try:
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Processing error: {str(e)}"
                        })
                    except:
                        # If we can't send the error message, connection is likely broken
                        logger.error(f"💥 Failed to send error message to {session_id}, connection broken")
                        break
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
        logger.info("🌐 HTTPS: https://localhost:7860")
        logger.info("🔌 WSS: wss://localhost:7860/ws/stream")
        
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=7860,
            ssl_keyfile="../ssl/key.pem",
            ssl_certfile="../ssl/cert.pem",
            ssl_version=ssl.PROTOCOL_TLS,
            ssl_cert_reqs=ssl.CERT_NONE,
            log_level="info"
        )
    else:
        logger.info("🌐 Starting server without SSL")
        logger.info("🌐 HTTP: http://localhost:7860")
        logger.info("🔌 WS: ws://localhost:7860/ws/stream")
        
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=7860,
            log_level="info"
        )