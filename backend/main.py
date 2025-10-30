import asyncio
import json
import logging
import numpy as np
import torch
import torchaudio
import librosa
import soundfile as sf
from typing import Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
from transformers import SeamlessM4Tv2Model, SeamlessM4TProcessor
import io
import wave
import tempfile
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SeamlessStreaming Translation API")

# Serve static frontend files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

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
            self.model = SeamlessM4Tv2Model.from_pretrained(
                "facebook/seamless-m4t-v2-large",
                dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            
            self.processor = SeamlessM4TProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e
    
    def audio_bytes_to_tensor(self, audio_bytes: bytes, sample_rate: int = 16000):
        """Convert audio bytes to tensor for model input using librosa"""
        try:
            # Create a temporary wav file from bytes
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_file.flush()
                
                # Load audio using librosa (more reliable than torchaudio)
                waveform, sr = librosa.load(temp_file.name, sr=None, mono=False)
                
                # Clean up temp file
                os.unlink(temp_file.name)
                
                # Convert to mono if stereo
                if len(waveform.shape) > 1:
                    waveform = librosa.to_mono(waveform)
                
                # Resample if necessary
                if sr != sample_rate:
                    waveform = librosa.resample(waveform, orig_sr=sr, target_sr=sample_rate)
                
                # Convert numpy array to torch tensor
                return torch.from_numpy(waveform).float()
                
        except Exception as e:
            logger.error(f"Error processing audio bytes: {e}")
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
    
    async def translate_speech(self, audio_bytes: bytes, src_lang: str, tgt_lang: str):
        """Translate speech from source to target language"""
        try:
            # Convert audio bytes to tensor
            waveform = self.audio_bytes_to_tensor(audio_bytes)
            if waveform is None:
                return None
            
            # Process audio with the model
            inputs = self.processor(
                audios=waveform,
                src_lang=src_lang,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate speech translation
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    tgt_lang=tgt_lang,
                    generate_speech=True
                )
            
            # Convert output to audio bytes
            if hasattr(outputs, 'waveform') and outputs.waveform is not None:
                return self.tensor_to_audio_bytes(outputs.waveform.squeeze())
            elif 'waveform' in outputs:
                return self.tensor_to_audio_bytes(outputs['waveform'].squeeze())
            
            return None
            
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return None

# Global translator instance
translator = SeamlessTranslator()

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected")
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected")

manager = ConnectionManager()

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    await translator.initialize()

@app.get("/")
async def root():
    """Serve the frontend"""
    return FileResponse("frontend/index.html")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": translator.model is not None}

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