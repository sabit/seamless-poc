#!/usr/bin/env python3
"""
WebSocket Audio Test Script

This script sends WAV file data to the SeamlessStreaming WebSocket server
to test audio translation without browser microphone issues.

Usage:
    python test_websocket_audio.py [wav_file] [--source_lang eng] [--target_lang ben] [--chunk_size 1024]
"""

import asyncio
import websockets
import json
import wave
import argparse
import time
import os
from pathlib import Path
import numpy as np
import ssl


class WebSocketAudioTester:
    def __init__(self, server_url="ws://localhost:8000/ws/stream"):
        self.server_url = server_url
        self.websocket = None
        
    async def connect(self):
        """Connect to the WebSocket server"""
        try:
            print(f"🔗 Connecting to {self.server_url}...")
            
            # Handle SSL for HTTPS/WSS connections
            ssl_context = None
            if self.server_url.startswith('wss://'):
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                print("🔓 Using unverified SSL context for self-signed certificates")
            
            # Connect with SSL context if needed
            self.websocket = await websockets.connect(self.server_url, ssl=ssl_context)
            print("✅ Connected to WebSocket server")
            return True
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            return False
    
    async def send_start_message(self, source_lang="eng", target_lang="ben"):
        """Send streaming start message"""
        start_message = {
            "type": "start_stream",
            "srcLang": source_lang,
            "tgtLang": target_lang,
            "task": "s2st",
            "sample_rate": 16000
        }
        
        print(f"📤 Sending start message: {start_message}")
        await self.websocket.send(json.dumps(start_message))
        
        # Wait for stream_ready response
        response = await self.websocket.recv()
        print(f"📥 Server response: {response}")
        
        try:
            msg = json.loads(response)
            if msg.get("type") == "stream_ready":
                print("🟢 Streaming session ready!")
                return True
            else:
                print(f"⚠️ Unexpected response: {msg}")
                return False
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON response: {response}")
            return False
    
    def load_wav_file(self, wav_path):
        """Load WAV file and return audio data"""
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"WAV file not found: {wav_path}")
        
        print(f"📁 Loading WAV file: {wav_path}")
        
        with wave.open(wav_path, 'rb') as wav_file:
            # Get WAV file properties
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frames = wav_file.getnframes()
            duration = frames / sample_rate
            
            print(f"📊 WAV Properties:")
            print(f"   Sample Rate: {sample_rate} Hz")
            print(f"   Channels: {channels}")
            print(f"   Sample Width: {sample_width} bytes")
            print(f"   Duration: {duration:.2f} seconds")
            print(f"   Total Frames: {frames}")
            
            # Read all audio data
            audio_data = wav_file.readframes(frames)
            
            # Convert to int16 array for processing
            if sample_width == 2:  # 16-bit
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
            elif sample_width == 4:  # 32-bit
                audio_array = np.frombuffer(audio_data, dtype=np.int32)
                # Convert to 16-bit
                audio_array = (audio_array / 65536).astype(np.int16)
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")
            
            # Convert to mono if stereo
            if channels == 2:
                print("🔄 Converting stereo to mono")
                audio_array = audio_array.reshape(-1, 2).mean(axis=1).astype(np.int16)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                print(f"🔄 Resampling from {sample_rate}Hz to 16kHz")
                # Simple downsampling/upsampling
                target_length = int(len(audio_array) * 16000 / sample_rate)
                indices = np.linspace(0, len(audio_array) - 1, target_length)
                audio_array = np.interp(indices, np.arange(len(audio_array)), audio_array).astype(np.int16)
            
            print(f"✅ Processed audio: {len(audio_array)} samples, {len(audio_array)/16000:.2f} seconds")
            return audio_array.tobytes()
    
    async def send_audio_chunks(self, audio_data, chunk_size=1024, delay=0.064):
        """Send audio data in chunks to simulate real-time streaming"""
        total_chunks = len(audio_data) // (chunk_size * 2)  # 2 bytes per int16 sample
        
        print(f"🎵 Sending {len(audio_data)} bytes of audio in {total_chunks} chunks")
        print(f"📦 Chunk size: {chunk_size} samples ({chunk_size * 2} bytes)")
        print(f"⏱️ Delay between chunks: {delay:.3f} seconds")
        
        responses_received = 0
        
        # Send audio in chunks
        for i in range(0, len(audio_data), chunk_size * 2):
            chunk = audio_data[i:i + chunk_size * 2]
            chunk_num = i // (chunk_size * 2) + 1
            
            print(f"📤 Sending chunk {chunk_num}/{total_chunks} ({len(chunk)} bytes)")
            
            try:
                await self.websocket.send(chunk)
                
                # Check for any responses (non-blocking)
                try:
                    response = await asyncio.wait_for(self.websocket.recv(), timeout=0.001)
                    responses_received += 1
                    
                    if isinstance(response, bytes):
                        print(f"🎵 Received audio response {responses_received}: {len(response)} bytes")
                        # Save received audio for inspection
                        with open(f"received_audio_{responses_received}.raw", "wb") as f:
                            f.write(response)
                        print(f"💾 Saved to received_audio_{responses_received}.raw")
                    else:
                        print(f"📥 Text response {responses_received}: {response}")
                        
                except asyncio.TimeoutError:
                    # No response yet, continue
                    pass
                
                # Wait before sending next chunk (simulate real-time)
                await asyncio.sleep(delay)
                
            except Exception as e:
                print(f"❌ Error sending chunk {chunk_num}: {e}")
                break
        
        print(f"✅ Finished sending audio. Received {responses_received} responses.")
        
        # Wait for any remaining responses
        print("⏳ Waiting for final responses...")
        try:
            while True:
                response = await asyncio.wait_for(self.websocket.recv(), timeout=2.0)
                responses_received += 1
                
                if isinstance(response, bytes):
                    print(f"🎵 Final audio response {responses_received}: {len(response)} bytes")
                    with open(f"received_audio_{responses_received}.raw", "wb") as f:
                        f.write(response)
                    print(f"💾 Saved to received_audio_{responses_received}.raw")
                else:
                    print(f"📥 Final text response {responses_received}: {response}")
                    
        except asyncio.TimeoutError:
            print("⏰ No more responses received")
        
        print(f"🏁 Total responses received: {responses_received}")
    
    async def send_stop_message(self):
        """Send streaming stop message"""
        stop_message = {"type": "stop_stream"}
        print(f"🛑 Sending stop message: {stop_message}")
        await self.websocket.send(json.dumps(stop_message))
        
        # Wait for acknowledgment
        try:
            response = await asyncio.wait_for(self.websocket.recv(), timeout=2.0)
            print(f"📥 Stop response: {response}")
        except asyncio.TimeoutError:
            print("⏰ No stop acknowledgment received")
    
    async def close(self):
        """Close WebSocket connection"""
        if self.websocket:
            await self.websocket.close()
            print("🔌 WebSocket connection closed")


async def main():
    parser = argparse.ArgumentParser(description="Test WebSocket audio streaming with WAV file")
    parser.add_argument("wav_file", nargs="?", default="samples/test_audio.wav", 
                       help="Path to WAV file (default: samples/test_audio.wav)")
    parser.add_argument("--source_lang", default="eng", 
                       help="Source language code (default: eng)")
    parser.add_argument("--target_lang", default="ben", 
                       help="Target language code (default: ben)")
    parser.add_argument("--chunk_size", type=int, default=1024, 
                       help="Audio chunk size in samples (default: 1024)")
    parser.add_argument("--delay", type=float, default=0.064, 
                       help="Delay between chunks in seconds (default: 0.064)")
    parser.add_argument("--server", default="ws://localhost:8000/ws/stream", 
                       help="WebSocket server URL")
    
    args = parser.parse_args()
    
    # Check if WAV file exists
    if not os.path.exists(args.wav_file):
        print(f"❌ WAV file not found: {args.wav_file}")
        print("\n💡 Create a samples directory and add a WAV file, or specify a different path")
        print("   Example: python test_websocket_audio.py path/to/your/audio.wav")
        return 1
    
    tester = WebSocketAudioTester(args.server)
    
    try:
        # Connect to server
        if not await tester.connect():
            return 1
        
        # Send start message
        if not await tester.send_start_message(args.source_lang, args.target_lang):
            return 1
        
        # Load and send audio file
        audio_data = tester.load_wav_file(args.wav_file)
        await tester.send_audio_chunks(audio_data, args.chunk_size, args.delay)
        
        # Send stop message
        await tester.send_stop_message()
        
        print("\n🎉 Test completed successfully!")
        print("📁 Check for received_audio_*.raw files containing server responses")
        
        return 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        await tester.close()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)