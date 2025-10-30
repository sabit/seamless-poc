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
    exit(1)

class OfficialStreamingTranslator:
    """Official SeamlessStreaming implementation using Facebook's agents"""
    
    def __init__(self, source_lang: str = "eng", target_lang: str = "ben"):
        logger.info("🔧 Initializing official SeamlessStreaming translator...")
        
        self.source_lang = source_lang
        self.target_lang = target_lang
        
        # Enforce CUDA requirement
        if not torch.cuda.is_available():
            raise RuntimeError("❌ CUDA is required for SeamlessStreaming. This model does not run on CPU.")
        
        self.device = torch.device("cuda")
        self.dtype = torch.float16
        
        logger.info(f"Using device: {self.device} with dtype: {self.dtype}")
        logger.info(f"Translation: {source_lang} → {target_lang}")
        
        try:
            # Create Args object based on official source code patterns
            self.args = self._create_official_args()
            
            # Initialize official SeamlessStreaming agent
            logger.info("🚀 Creating SeamlessStreamingS2STAgent...")
            self.agent = SeamlessStreamingS2STAgent(self.args)
            logger.info("✅ Official SeamlessStreaming agent initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize official agent: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            raise
    
    def _create_official_args(self):
        """Create Args object based on official SeamlessStreaming source code"""
        from argparse import Namespace
        import torch
        
        # Create args matching the official implementation patterns
        args = Namespace()
        
        # === Core UnitYPipelineMixin parameters ===
        args.task = "s2st"  # Speech-to-Speech Translation
        args.unity_model_name = "seamless_streaming_unity"
        args.monotonic_decoder_model_name = "seamless_streaming_monotonic_decoder"
        args.sample_rate = 16000.0
        args.dtype = "fp16"
        args.device = "cuda"
        args.fp16 = True
        
        # === Individual Agent Parameters (from add_args methods) ===
        
        # OnlineFeatureExtractorAgent parameters
        args.upstream_idx = 0
        args.feature_dim = 1024
        args.frame_num = 1
        
        # OfflineWav2VecBertEncoderAgent parameters  
        args.encoder_chunk_size = 480
        
        # UnitYMMATextDecoderAgent parameters
        args.min_starting_wait_w2vbert = 16
        args.min_starting_wait_mma = 4
        
        # NARUnitYUnitDecoderAgent parameters (CRITICAL - these were missing!)
        args.min_unit_chunk_size = 50  # Required parameter
        args.d_factor = 1.0  # Required parameter
        
        # VocoderAgent parameters
        args.vocoder_name = "vocoder_v2"
        
        # === SimulEval Framework Parameters ===
        args.source_segment_size = 480
        args.target_segment_size = None
        args.waitk_lagging = 3
        args.quality_metrics = "BLEU"
        args.latency_metrics = "StartOffset EndOffset"
        
        # === Device and Type Conversion ===
        args.device = torch.device(args.device)
        if args.dtype == "fp16" and args.device.type != "cpu":
            args.dtype = torch.float16
        else:
            args.dtype = torch.float32
        
        # === Additional Framework Parameters ===
        args.output = None
        args.config = None
        args.no_gpu = False
        
        # === Language Configuration ===
        args.source_lang = self.source_lang
        args.target_lang = self.target_lang
        
        logger.info("📝 Created official Args object with all required agent parameters")
        logger.info(f"   - Task: {args.task}")
        logger.info(f"   - Unity model: {args.unity_model_name}")
        logger.info(f"   - Device: {args.device}")
        logger.info(f"   - Min unit chunk size: {args.min_unit_chunk_size}")
        logger.info(f"   - Duration factor: {args.d_factor}")
        
        return args
        
    async def initialize(self, task="s2st", src_lang="eng", tgt_lang="ben"):
        """Initialize the official streaming agent"""
        try:
            logger.info(f"Initializing official SeamlessStreaming agent for {task}: {src_lang} → {tgt_lang}")
            
            # Configure agent arguments with all required parameters
            agent_args = {
                # Device and model settings
                "device": self.device,  # Always CUDA
                "dtype": "fp16",
                "fp16": True,
                "task": task,
                "tgt_lang": tgt_lang,
                "src_lang": src_lang,
                
                # Model names
                "unity_model_name": "seamless_streaming_unity",
                "monotonic_decoder_model_name": "seamless_streaming_monotonic_decoder",
                
                # VAD settings
                "vad": True,
                "vad_chunk_size": 480,  # 30ms chunks at 16kHz
                "vad_threshold": 0.5,
                
                # Latency and quality control - fix the NoneType comparison issue
                "min_starting_wait": 1000,  # Wait for 1 second of audio
                "max_len_a": 1.2,  # Set to a valid float instead of 0.0
                "max_len_b": 100,
                "beam_size": 3,
                "no_repeat_ngram_size": 3,
                
                # SimulEval specific parameters
                "quality_metrics": [],  # Empty list instead of None
                "latency_metrics": [],  # Empty list instead of None
                "computation_aware": False,
                "start_index": 0,
                "end_index": 999999,  # Use large number instead of None
                
                # Additional SimulEval parameters to prevent None comparisons
                "scores": [],
                "instances": [],
                "prediction": "",
                "reference": "",
                
                # Additional required parameters
                "output": "/tmp",  # Set to a valid path instead of None
                "log_level": "INFO",
                "port": 12321,  # Set to a valid port instead of None
                "host": "localhost",  # Set to a valid host instead of None
                
                # Model loading settings
                "gated_model_dir": "/tmp",  # Set to valid path instead of None
                "model_name": "seamless_streaming_unity",  # Set explicitly
                
                # Generation settings
                "temperature": 1.0,
                "length_penalty": 1.0,
                "max_new_tokens": 256,
                
                # Audio settings
                "sample_rate": self.sample_rate,
                "chunk_size": 4096,
                
                # Additional streaming parameters
                "waitk": 3,  # Wait-k policy
                "test_segment_size": 1000,  # Use default value instead of None
                "source_segment_size": 1000,  # Use default value instead of None
                
                # Additional parameters that might be expected
                "eval_latency": True,
                "eval_quality": False,
                "continue_finished": True,
                "reset_model": False,
                
                # More specific agent parameters
                "source": "/tmp/source.wav",
                "target": "/tmp/target.txt",
                "config": None,
                "system_dir": "/tmp",
                "user_dir": "/tmp",
            }
            
            # Select appropriate agent class
            if task == "s2st":
                agent_class = SeamlessStreamingS2STAgent
            elif task == "s2t":
                agent_class = SeamlessStreamingS2TAgent
            else:
                raise ValueError(f"Unsupported task: {task}")
            
            # Convert args dict to object-like structure for build_agent
            class Args:
                def __init__(self, **kwargs):
                    # Set default values for commonly expected attributes
                    defaults = {
                        'device': 'cuda',
                        'dtype': 'fp16',
                        'fp16': True,
                        'task': 's2st',
                        'tgt_lang': 'ben',
                        'src_lang': 'eng',
                        'unity_model_name': 'seamless_streaming_unity',
                        'monotonic_decoder_model_name': 'seamless_streaming_monotonic_decoder',
                        'vad': True,
                        'vad_chunk_size': 480,
                        'vad_threshold': 0.5,
                        'min_starting_wait': 1000,
                        'max_len_a': 1.2,  # Valid float instead of 0.0
                        'max_len_b': 100,
                        'beam_size': 3,
                        'no_repeat_ngram_size': 3,
                        'quality_metrics': [],
                        'latency_metrics': [],
                        'computation_aware': False,
                        'start_index': 0,
                        'end_index': 999999,  # Use large number instead of None
                        'output': '/tmp',  # Valid path instead of None
                        'log_level': 'INFO',
                        'port': 12321,  # Valid port instead of None
                        'host': 'localhost',  # Valid host instead of None
                        'model_name': 'seamless_streaming_unity',
                        'gated_model_dir': '/tmp',
                        'sample_rate': 16000,
                        'chunk_size': 4096,
                        'temperature': 1.0,
                        'length_penalty': 1.0,
                        'max_new_tokens': 256,
                        'waitk': 3,
                        'test_segment_size': 1000,  # Use default value instead of None
                        'source_segment_size': 1000,  # Use default value instead of None
                        'eval_latency': True,
                        'eval_quality': False,
                        'continue_finished': True,
                        'reset_model': False,
                        'scores': [],
                        'instances': [],
                        'prediction': '',
                        'reference': ''
                    }
                    
                    # Set defaults first
                    for key, value in defaults.items():
                        setattr(self, key, value)
                    
                    # Override with provided kwargs
                    for key, value in kwargs.items():
                        setattr(self, key, value)
                
                def __getattr__(self, name):
                    # Return None for any missing attributes instead of raising AttributeError
                    return None
            
            args_obj = Args(**agent_args)
            
            # Try direct agent instantiation with minimal args
            try:
                # Create a minimal args object with only essential parameters
                class MinimalArgs:
                    def __init__(self):
                        self.device = self.device
                        self.dtype = "fp16"
                        self.fp16 = True
                        self.task = task
                        self.tgt_lang = tgt_lang
                        self.src_lang = src_lang
                        self.unity_model_name = "seamless_streaming_unity"
                        self.monotonic_decoder_model_name = "seamless_streaming_monotonic_decoder"
                        self.vad = True
                        self.vad_chunk_size = 480
                        self.vad_threshold = 0.5
                        self.min_starting_wait = 1000
                        self.max_len_a = 1.2
                        self.max_len_b = 100
                        self.beam_size = 3
                        # Set all other attributes to safe defaults
                        for key, value in agent_args.items():
                            if not hasattr(self, key):
                                setattr(self, key, value)
                    
                    def __getattr__(self, name):
                        # Return safe defaults for missing attributes
                        return getattr(self, name, 0)  # Return 0 instead of None
                
                minimal_args = MinimalArgs()
                self.agent = agent_class(minimal_args)
                logger.info("✅ Agent created with minimal args")
            except Exception as minimal_error:
                logger.error(f"❌ Minimal instantiation failed: {minimal_error}")
                # Try the original approach as last resort
                try:
                    self.agent = build_agent(agent_class, args_obj)
                    logger.info("✅ Agent built using build_agent as fallback")
                except Exception as final_error:
                    logger.error(f"❌ All agent creation methods failed: {final_error}")
                    raise final_error
            
            self.initialized = True
            
            logger.info("✅ Official streaming agent initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize official streaming agent: {e}")
            logger.error(f"Error details: {str(e)}")
            logger.info("🔄 Will fall back to improved chunked implementation")
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
        """Initialize the session with official SeamlessStreaming"""
        self.translator = OfficialStreamingTranslator()
        await self.translator.initialize(
            task="s2st",
            src_lang=self.src_lang,
            tgt_lang=self.tgt_lang
        )
        logger.info("🎯 Using official SeamlessStreaming")
    
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
    return {
        "status": "healthy",
        "official_streaming": True,
        "cuda_available": torch.cuda.is_available()
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
    
    # Check CUDA availability before starting
    if not torch.cuda.is_available():
        print("❌ CUDA is not available!")
        print("💡 SeamlessStreaming models require CUDA/GPU to run properly.")
        print("💡 Please ensure you have:")
        print("   - NVIDIA GPU with CUDA support")
        print("   - PyTorch with CUDA installed")
        print("   - Sufficient GPU memory (8GB+ recommended)")
        exit(1)
    
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
    print("🎯 Official SeamlessStreaming Implementation")
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