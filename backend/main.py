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
    from argparse import Namespace
    import soundfile as sf
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
    """Handles continuous streaming translation with proper SeamlessStreaming architecture"""
    def __init__(self):
        self.agent = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sample_rate = 16000
        # Official SeamlessStreaming parameters from research
        self.source_segment_size = 320  # Official parameter from evaluation
        self.min_starting_wait_w2vbert = 192  # Official parameter
        self.decision_threshold = 0.5  # Official parameter
        self.min_unit_chunk_size = 50  # Official parameter
        self.window_size_samples = 1536  # Standard window size
        self.chunk_size_samples = 512  # VAD chunk size
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"SeamlessStreaming parameters: source_segment_size={self.source_segment_size}, min_starting_wait={self.min_starting_wait_w2vbert}")
        
    async def initialize(self):
        """Initialize the SeamlessStreaming agent with proper configuration"""
        try:
            if not STREAMING_AVAILABLE:
                logger.error("SeamlessStreaming agents not available - falling back to basic model")
                # Fallback to basic model
                self.model = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained(
                    "facebook/seamless-m4t-v2-large",
                    dtype=torch.float16 if self.device == "cuda" else torch.float32
                ).to(self.device)
                self.processor = SeamlessM4TProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
                return
                
            logger.info("Initializing SeamlessStreamingS2STAgent with proper configuration...")
            
            # Create proper args namespace with official parameters
            args = Namespace()
            args.task = "s2st"  # Speech-to-speech translation
            args.device = self.device
            args.dtype = "fp16" if self.device == "cuda" and torch.cuda.is_available() else "fp32"
            args.unity_model_name = "seamless_streaming_unity"
            args.monotonic_decoder_model_name = "seamless_streaming_monotonic_decoder"
            args.sample_rate = float(self.sample_rate)
            
            # Official evaluation parameters from research
            args.source_segment_size = self.source_segment_size
            args.min_starting_wait_w2vbert = self.min_starting_wait_w2vbert
            args.decision_threshold = self.decision_threshold
            args.no_early_stop = True
            args.max_len_a = 0
            args.max_len_b = 100
            args.min_unit_chunk_size = self.min_unit_chunk_size
            
            # Add additional required parameters that might be needed
            args.fp16 = args.dtype == "fp16"
            args.tgt_lang = "ben"  # Default target language
            
            logger.info(f"Agent configuration: device={args.device}, dtype={args.dtype}")
            logger.info(f"Streaming parameters: source_segment_size={args.source_segment_size}, min_wait={args.min_starting_wait_w2vbert}")
            
            # Initialize the SeamlessStreaming agent
            try:
                self.agent = SeamlessStreamingS2STAgent.from_args(args)
                logger.info("SeamlessStreaming agent initialized successfully!")
                
                # Test the agent with a small dummy segment to verify it's working
                logger.info("Testing agent with dummy audio...")
                test_audio = np.random.normal(0, 0.1, self.source_segment_size).astype(np.float32)  # Small noise signal
                test_segment = self.create_speech_segment(test_audio, "ben", finished=False)
                
                logger.info(f"Test segment: {len(test_audio)} samples, tgt_lang=ben")
                self.agent.push(test_segment)
                test_output = self.agent.pop()
                logger.info(f"Agent test result: {type(test_output).__name__ if test_output else 'None'}")
                
                if test_output is None:
                    logger.warning("⚠️  Agent returned None on test - this is the original issue we're fixing")
                    logger.info("The agent requires more sophisticated handling than simple push/pop")
                else:
                    logger.info("✅ Agent test successful - basic functionality working")
                
            except Exception as agent_error:
                logger.error(f"Failed to initialize SeamlessStreaming agent: {agent_error}")
                logger.error(f"Agent error type: {type(agent_error).__name__}")
                import traceback
                logger.error(f"Agent traceback: {traceback.format_exc()}")
                raise agent_error
            
        except Exception as e:
            logger.error(f"Failed to load SeamlessStreaming agent: {e}")
            logger.info("Falling back to basic SeamlessM4T model...")
            # Fallback to basic model
            self.model = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained(
                "facebook/seamless-m4t-v2-large",
                dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            self.processor = SeamlessM4TProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
            self.agent = None
    
    def reset_agent_state(self):
        """Reset the agent state to handle cases where policy returns None"""
        if self.agent is not None:
            try:
                # Reset the agent to clear any problematic internal states
                self.agent.reset()
                logger.info("Agent state reset successfully")
            except Exception as e:
                logger.warning(f"Could not reset agent state: {e}")
    
    def handle_agent_none_response(self, audio_segment: np.ndarray, tgt_lang: str) -> Optional[bytes]:
        """Handle cases where agent.policy() returns None by implementing proper state management"""
        if self.agent is None:
            return None
        
        try:
            # Reset agent state first
            self.reset_agent_state()
            
            # Try processing with explicit state management
            segment = self.create_speech_segment(audio_segment, tgt_lang, finished=False)
            
            # Push segment and attempt multiple pops in case processing needs time
            self.agent.push(segment)
            
            # Try popping multiple times as the agent might need several iterations
            max_attempts = 3
            for attempt in range(max_attempts):
                output = self.agent.pop()
                if output is not None and not isinstance(output, EmptySegment):
                    logger.info(f"Agent produced output on attempt {attempt + 1}")
                    return self._process_agent_output(output)
                logger.debug(f"Attempt {attempt + 1}: agent returned None/EmptySegment")
            
            logger.warning("Agent returned None after all attempts - this was the original issue")
            return None
            
        except Exception as e:
            logger.error(f"Error in agent None response handler: {e}")
            return None
    
    def _process_agent_output(self, output_segment) -> Optional[bytes]:
        """Process the output segment from the agent"""
        try:
            if hasattr(output_segment, 'content') and output_segment.content is not None:
                content = output_segment.content
                
                if isinstance(content, (list, tuple)):
                    if len(content) > 0 and isinstance(content[0], (int, float)):
                        audio_array = np.array(content, dtype=np.float32)
                    else:
                        audio_array = np.concatenate([np.array(c, dtype=np.float32) for c in content if c is not None])
                elif isinstance(content, np.ndarray):
                    audio_array = content.astype(np.float32)
                else:
                    logger.warning(f"Unexpected output content type: {type(content)}")
                    return None
                
                if len(audio_array) > 0:
                    return self._array_to_pcm_bytes(audio_array)
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing agent output: {e}")
            return None
    
    async def test_streaming_fixes(self) -> Dict[str, Any]:
        """Test the streaming fixes to verify the agent.policy() None issue is resolved"""
        test_results = {
            "agent_available": self.agent is not None,
            "basic_push_pop": False,
            "enhanced_state_mgmt": False,
            "proper_actions": False,
            "fallback_model": hasattr(self, 'model') and self.model is not None
        }
        
        if self.agent is None:
            logger.warning("Agent not available for testing")
            return test_results
        
        # Test data - synthetic speech-like signal
        test_duration = 2.0  # 2 seconds
        test_samples = int(self.sample_rate * test_duration)
        # Create a signal with speech-like characteristics
        t = np.linspace(0, test_duration, test_samples)
        test_audio = (np.sin(2 * np.pi * 440 * t) * 0.1 +  # 440Hz tone
                     np.random.normal(0, 0.02, test_samples) +  # Background noise
                     np.sin(2 * np.pi * 880 * t) * 0.05)  # Harmonic
        test_audio = test_audio.astype(np.float32)
        
        logger.info(f"🧪 Testing SeamlessStreaming fixes with {test_duration}s synthetic audio...")
        
        # Test 1: Basic push/pop
        try:
            result = self.process_with_agent(test_audio, "ben")
            test_results["basic_push_pop"] = result is not None
            logger.info(f"✅ Basic push/pop test: {'PASS' if result else 'FAIL (None returned)'}")
        except Exception as e:
            logger.error(f"❌ Basic push/pop test failed: {e}")
        
        # Test 2: Enhanced state management
        try:
            result = self.handle_agent_none_response(test_audio, "ben")
            test_results["enhanced_state_mgmt"] = result is not None
            logger.info(f"✅ Enhanced state management test: {'PASS' if result else 'FAIL (None returned)'}")
        except Exception as e:
            logger.error(f"❌ Enhanced state management test failed: {e}")
        
        # Test 3: Proper action patterns
        try:
            result = self.process_with_proper_actions(test_audio, "ben")
            test_results["proper_actions"] = result is not None
            logger.info(f"✅ Proper actions test: {'PASS' if result else 'FAIL (None returned)'}")
        except Exception as e:
            logger.error(f"❌ Proper actions test failed: {e}")
        
        # Summary
        passed_tests = sum([v for k, v in test_results.items() if k != "agent_available" and k != "fallback_model"])
        total_tests = len([k for k in test_results.keys() if k != "agent_available" and k != "fallback_model"])
        
        logger.info(f"🧪 Test Summary: {passed_tests}/{total_tests} approaches working")
        
        if passed_tests > 0:
            logger.info("✅ SUCCESS: At least one approach resolves the agent.policy() None issue!")
        else:
            logger.warning("❌ All approaches still return None - issue persists")
        
        return test_results
    
    def create_speech_segment(self, audio_data: np.ndarray, tgt_lang: str = "ben", finished: bool = False) -> SpeechSegment:
        """Create a proper SpeechSegment for the SeamlessStreaming agent"""
        return SpeechSegment(
            content=audio_data.tolist(),
            finished=finished,
            sample_rate=self.sample_rate,
            tgt_lang=tgt_lang
        )
    
    def process_with_proper_actions(self, audio_segment: np.ndarray, tgt_lang: str = "ben") -> Optional[bytes]:
        """Process audio using proper ReadAction/WriteAction patterns from SimulEval"""
        if self.agent is None:
            return None
        
        try:
            # Create segment with proper windowing based on research
            # Use source_segment_size (320 samples) for proper segmentation
            segment_size = min(self.source_segment_size, len(audio_segment))
            
            results = []
            
            # Process audio in segments matching the official parameters
            for i in range(0, len(audio_segment), segment_size):
                chunk = audio_segment[i:i + segment_size]
                if len(chunk) < segment_size // 2:  # Skip very small end chunks
                    continue
                
                # Pad chunk to segment_size if needed
                if len(chunk) < segment_size:
                    chunk = np.pad(chunk, (0, segment_size - len(chunk)), 'constant')
                
                segment = self.create_speech_segment(chunk, tgt_lang, finished=(i + segment_size >= len(audio_segment)))
                
                logger.debug(f"Processing segment {i//segment_size + 1}: {len(chunk)} samples")
                
                # Push segment and handle the response properly
                self.agent.push(segment)
                
                # The agent might need multiple iterations to produce output
                # This follows the ReadAction/WriteAction pattern where agent decides when to output
                max_pops = 5  # Allow multiple pops as agent processes through pipeline
                for pop_attempt in range(max_pops):
                    output = self.agent.pop()
                    
                    if output is not None and not isinstance(output, EmptySegment):
                        logger.debug(f"Got output on pop attempt {pop_attempt + 1}")
                        processed_output = self._process_agent_output(output)
                        if processed_output:
                            results.append(processed_output)
                        break
                    elif pop_attempt == max_pops - 1:
                        logger.debug(f"No output after {max_pops} pop attempts for segment {i//segment_size + 1}")
            
            # Combine results if we got multiple segments
            if results:
                # For audio, we concatenate the byte arrays
                combined_audio = b''.join(results)
                logger.info(f"Combined {len(results)} segment outputs into {len(combined_audio)} bytes")
                return combined_audio
            else:
                logger.debug("No output from any segment")
                return None
                
        except Exception as e:
            logger.error(f"Error in proper action processing: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def process_with_agent(self, audio_segment: np.ndarray, tgt_lang: str = "ben") -> Optional[bytes]:
        """Process audio using the SeamlessStreaming agent with proper UnitYAgentPipeline methods"""
        if self.agent is None:
            logger.warning("SeamlessStreaming agent not available")
            return None
            
        try:
            # Create speech segment with proper format
            segment = self.create_speech_segment(audio_segment, tgt_lang, finished=False)
            
            logger.info(f"Processing audio segment: {len(audio_segment)} samples ({len(audio_segment)/self.sample_rate:.2f}s)")
            
            # For UnitYAgentPipeline, we need to use push() and pop() methods correctly
            # The agent processes segments incrementally through the pipeline
            
            # Push the segment to start processing
            self.agent.push(segment)
            
            # Pop the result - this may return None if processing isn't complete
            output_segment = self.agent.pop()
            
            if output_segment is not None and not isinstance(output_segment, EmptySegment):
                logger.info(f"Agent produced output: {type(output_segment).__name__}")
                
                if hasattr(output_segment, 'content') and output_segment.content is not None:
                    # The output should be audio content for S2ST
                    content = output_segment.content
                    
                    # Convert various content types to audio array
                    if isinstance(content, (list, tuple)):
                        if len(content) > 0 and isinstance(content[0], (int, float)):
                            # Direct audio samples
                            audio_array = np.array(content, dtype=np.float32)
                        elif len(content) > 0 and hasattr(content[0], '__iter__'):
                            # Nested structure - flatten
                            audio_array = np.concatenate([np.array(c, dtype=np.float32) for c in content if c is not None])
                        else:
                            logger.warning(f"Unexpected content structure: {type(content[0]) if content else 'empty'}")
                            return None
                    elif isinstance(content, np.ndarray):
                        audio_array = content.astype(np.float32)
                    else:
                        logger.warning(f"Unexpected content type: {type(content)}")
                        return None
                    
                    if len(audio_array) > 0:
                        return self._array_to_pcm_bytes(audio_array)
                    else:
                        logger.debug("Empty audio array from agent")
                        return None
                else:
                    logger.debug("Output segment has no content")
                    return None
            else:
                logger.debug("Agent returned None or EmptySegment")
                return None
                
        except Exception as e:
            logger.error(f"Error processing with SeamlessStreaming agent: {e}")
            logger.error(f"Exception details: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    async def process_streaming_chunk(self, chunk: np.ndarray, src_lang: str, tgt_lang: str) -> Optional[bytes]:
        """Process a single streaming chunk using SeamlessStreaming agent or fallback"""
        try:
            # Basic audio quality checks
            audio_energy = np.sqrt(np.mean(chunk ** 2))
            logger.info(f"Audio energy: {audio_energy:.4f}")
            if audio_energy < 0.01:
                logger.debug("Skipping low energy audio")
                return None
            
            # Normalize audio
            chunk_normalized = chunk / max(abs(chunk.max()), abs(chunk.min())) if chunk.max() != 0 else chunk
            
            # Try SeamlessStreaming agent with multiple approaches
            if self.agent is not None:
                logger.info("Processing with SeamlessStreaming agent...")
                
                # Try approach 1: Direct push/pop
                result = self.process_with_agent(chunk_normalized, tgt_lang)
                if result is not None:
                    logger.info("Direct push/pop approach succeeded")
                    return result
                
                # Try approach 2: Enhanced state management  
                logger.info("Direct approach returned None - applying enhanced state management...")
                result = self.handle_agent_none_response(chunk_normalized, tgt_lang)
                if result is not None:
                    logger.info("Enhanced state management resolved the None response!")
                    return result
                
                # Try approach 3: Proper action patterns with segmentation
                logger.info("State management also returned None - trying proper action patterns...")
                result = self.process_with_proper_actions(chunk_normalized, tgt_lang)
                if result is not None:
                    logger.info("Proper action patterns resolved the None response!")
                    return result
                
                logger.warning("All SeamlessStreaming approaches returned None, using fallback...")
            
            # Fallback to basic model if agent fails or unavailable
            if hasattr(self, 'model') and self.model is not None:
                logger.info("Using fallback SeamlessM4T model...")
                return await self.process_with_fallback_model(chunk_normalized, src_lang, tgt_lang)
            else:
                logger.error("No model available for processing")
                return None
                
        except Exception as e:
            logger.error(f"Streaming translation error: {e}")
            return None
    
    async def process_with_fallback_model(self, chunk: np.ndarray, src_lang: str, tgt_lang: str) -> Optional[bytes]:
        """Fallback processing using the basic SeamlessM4T model"""
        try:
            waveform = torch.from_numpy(chunk).float()
            
            inputs = self.processor(
                audio=waveform,
                sampling_rate=self.sample_rate,
                src_lang=src_lang,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                audio_array = self.model.generate(
                    **inputs, 
                    tgt_lang=tgt_lang,
                    do_sample=False,
                    num_beams=3,
                    max_new_tokens=256,
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )
            
            if isinstance(audio_array, torch.Tensor):
                return self._tensor_to_pcm_bytes(audio_array.squeeze())
            elif isinstance(audio_array, (list, tuple)) and len(audio_array) > 0:
                audio_tensor = audio_array[0] if hasattr(audio_array[0], 'shape') else audio_array
                return self._tensor_to_pcm_bytes(audio_tensor.squeeze())
            
            return None
            
        except Exception as e:
            logger.error(f"Fallback model error: {e}")
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
    
    def _array_to_pcm_bytes(self, audio_array: np.ndarray) -> bytes:
        """Convert numpy array to raw PCM bytes"""
        try:
            # Ensure it's a numpy array
            if not isinstance(audio_array, np.ndarray):
                audio_array = np.array(audio_array, dtype=np.float32)
            
            # Normalize audio
            if audio_array.max() > 1.0 or audio_array.min() < -1.0:
                audio_array = audio_array / max(abs(audio_array.max()), abs(audio_array.min()))
            
            # Convert to 16-bit PCM
            audio_16bit = (audio_array * 32767).astype(np.int16)
            return audio_16bit.tobytes()
                
        except Exception as e:
            logger.error(f"Error converting array to PCM: {e}")
            return None

# Global streaming translator instance
streaming_translator = StreamingTranslator()

class StreamingSession:
    """Manages a streaming translation session with proper segment handling"""
    def __init__(self, websocket: WebSocket, src_lang: str, tgt_lang: str, sample_rate: int = 16000):
        self.websocket = websocket
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.sample_rate = sample_rate
        self.running = False
        self.last_output_time = 0
        
        # Use official SeamlessStreaming parameters from research
        self.source_segment_size = 320  # Official parameter
        self.min_starting_wait_samples = 192  # min_starting_wait_w2vbert from research
        self.window_size_samples = 1536  # Standard window size from research
        self.chunk_size_samples = 512  # VAD chunk size
        
        # Buffer management with proper accumulation pattern
        # Research shows agents need sufficient accumulation before processing
        buffer_duration = 10.0  # 10 second buffer
        self.buffer = deque(maxlen=int(self.sample_rate * buffer_duration))
        
        # Audio accumulation tracking - critical for agent processing
        self.accumulated_samples = 0
        self.min_processing_samples = max(
            self.min_starting_wait_samples, 
            self.window_size_samples
        )  # Use the larger of the two requirements
        
        self.min_output_interval = 1.5  # Reduced interval for better responsiveness
        self.last_chunk_hash = None
        
        logger.info(f"StreamingSession initialized with research parameters:")
        logger.info(f"  source_segment_size: {self.source_segment_size}")
        logger.info(f"  min_starting_wait_samples: {self.min_starting_wait_samples}")
        logger.info(f"  window_size_samples: {self.window_size_samples}")
        logger.info(f"  min_processing_samples: {self.min_processing_samples}")
    
    async def initialize(self):
        """Initialize the streaming session"""
        pass
    
    def add_audio_chunk(self, audio_data: np.ndarray):
        """Add audio chunk to the streaming buffer with accumulation tracking"""
        self.buffer.extend(audio_data)
        self.accumulated_samples += len(audio_data)
        logger.debug(f"Added {len(audio_data)} samples, buffer: {len(self.buffer)}, accumulated: {self.accumulated_samples}")
    
    def get_processing_chunk(self) -> Optional[np.ndarray]:
        """Get processing chunk using SeamlessStreaming accumulation patterns"""
        # Ensure we meet the minimum starting wait requirement from research
        if self.accumulated_samples < self.min_starting_wait_samples:
            logger.debug(f"Accumulation insufficient: {self.accumulated_samples} < {self.min_starting_wait_samples}")
            return None
        
        # Check buffer has enough samples for processing
        if len(self.buffer) < self.min_processing_samples:
            logger.debug(f"Buffer too small: {len(self.buffer)} < {self.min_processing_samples}")
            return None
        
        # Use window_size_samples for processing as per research
        processing_samples = min(self.window_size_samples, len(self.buffer))
        
        # Take from buffer maintaining proper overlap for continuous processing
        chunk = np.array(list(self.buffer)[-processing_samples:])
        
        # Remove processed samples but maintain overlap as per streaming requirements
        # Use source_segment_size for removal to maintain proper pipeline flow
        samples_to_remove = min(self.source_segment_size, len(self.buffer) - processing_samples // 2)
        
        for _ in range(samples_to_remove):
            if len(self.buffer) > processing_samples // 2:  # Ensure minimum buffer
                self.buffer.popleft()
        
        logger.debug(f"Processing chunk: {len(chunk)} samples, buffer remaining: {len(self.buffer)}")
        return chunk
    
    def should_process_now(self) -> bool:
        """Determine if we should process now based on research parameters"""
        current_time = time.time()
        
        # Time-based check
        time_ready = current_time - self.last_output_time >= self.min_output_interval
        
        # Accumulation-based check (critical for agent processing)
        accumulation_ready = self.accumulated_samples >= self.min_starting_wait_samples
        
        # Buffer-based check
        buffer_ready = len(self.buffer) >= self.min_processing_samples
        
        ready = time_ready and accumulation_ready and buffer_ready
        
        if not ready:
            logger.debug(f"Processing conditions: time={time_ready}, accumulation={accumulation_ready}, buffer={buffer_ready}")
        
        return ready

    async def process_audio_chunk(self, audio_data: bytes):
        """Process an incoming audio chunk with proper SeamlessStreaming handling"""
        try:
            # Convert audio bytes to numpy array
            if len(audio_data) % 2 != 0:
                audio_data = audio_data + b'\x00'
            
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Log audio characteristics
            audio_rms = np.sqrt(np.mean(audio_np ** 2))
            logger.debug(f"Audio chunk: {len(audio_np)} samples, RMS: {audio_rms:.4f}")
            
            # Add to streaming buffer
            self.add_audio_chunk(audio_np)
            
            # Check if we should process
            current_time = time.time()
            buffer_duration = len(self.buffer) / self.sample_rate
            
            # Use improved processing conditions based on research
            if self.should_process_now():
                chunk = self.get_processing_chunk()
                if chunk is not None:
                    logger.info(f"Processing chunk: {len(chunk)} samples ({len(chunk)/self.sample_rate:.2f}s)")
                    logger.info(f"Accumulated: {self.accumulated_samples}, Buffer: {len(self.buffer)}")
                    
                    # Create hash to avoid duplicate processing
                    chunk_hash = hashlib.md5(chunk.tobytes()).hexdigest()
                    
                    if chunk_hash != self.last_chunk_hash:
                        # Process with SeamlessStreaming using enhanced approach
                        result = await streaming_translator.process_streaming_chunk(chunk, self.src_lang, self.tgt_lang)
                        if result:
                            await self.websocket.send_bytes(result)
                            self.last_output_time = current_time
                            self.last_chunk_hash = chunk_hash
                            # Reset accumulation counter after successful processing
                            self.accumulated_samples = max(0, self.accumulated_samples - self.source_segment_size)
                            logger.info(f"✅ Translation sent: {len(result)} bytes, accumulation reset to {self.accumulated_samples}")
                        else:
                            logger.warning("❌ SeamlessStreaming returned no output - this indicates the original issue")
                    else:
                        logger.debug("Skipping duplicate chunk")
                else:
                    logger.debug("Processing conditions met but no chunk available")
            else:
                # More detailed logging about why we're not processing
                buffer_duration = len(self.buffer) / self.sample_rate
                time_since_last = current_time - self.last_output_time
                logger.debug(f"Waiting - buffer: {buffer_duration:.2f}s, time: {time_since_last:.2f}s, accumulated: {self.accumulated_samples}")
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")

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

@app.get("/test-streaming-fixes")
async def test_streaming_fixes():
    """Test endpoint to verify the SeamlessStreaming fixes"""
    try:
        results = await streaming_translator.test_streaming_fixes()
        return {
            "status": "success",
            "test_results": results,
            "summary": {
                "issue_resolved": any(v for k, v in results.items() if k not in ["agent_available", "fallback_model"]),
                "agent_available": results.get("agent_available", False),
                "fallback_available": results.get("fallback_model", False)
            }
        }
    except Exception as e:
        logger.error(f"Test endpoint error: {e}")
        return {"status": "error", "message": str(e)}



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
                logger.info(f"Received audio chunk: {len(audio_chunk)} bytes")
                
                if len(audio_chunk) > 0:
                    await session.process_audio_chunk(audio_chunk)
                else:
                    logger.warning("Received empty audio chunk")
                    
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