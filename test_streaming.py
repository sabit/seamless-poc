#!/usr/bin/env python3
"""
Test script for SeamlessM4T v2 Streaming Translation System
This script demonstrates both complete recording and streaming modes.
"""

import asyncio
import websockets
import json
import numpy as np
import time
import wave
import io

class StreamingTestClient:
    def __init__(self):
        self.ws_url_complete = "ws://localhost:7860/ws/translate"
        self.ws_url_streaming = "ws://localhost:7860/ws/stream" 
        
    async def test_complete_recording_mode(self):
        """Test the complete recording translation mode"""
        print("🎤 Testing Complete Recording Mode...")
        
        try:
            async with websockets.connect(self.ws_url_complete) as websocket:
                # Send start message
                start_msg = {
                    "type": "start",
                    "src_lang": "eng",
                    "tgt_lang": "ben",
                    "sample_rate": 16000,
                    "format": "pcm_s16le"
                }
                
                await websocket.send(json.dumps(start_msg))
                print("📤 Sent start message")
                
                # Wait for ready confirmation
                response = await websocket.recv()
                msg = json.loads(response)
                
                if msg.get("type") == "ready":
                    print("✅ Server ready for complete recording mode")
                    
                    # Generate test audio (simulate 3 seconds of speech)
                    sample_rate = 16000
                    duration = 3.0
                    t = np.linspace(0, duration, int(sample_rate * duration))
                    
                    # Create a simple tone (simulate speech)
                    frequency = 440  # A4 note
                    audio_data = 0.3 * np.sin(2 * np.pi * frequency * t)
                    
                    # Convert to 16-bit PCM
                    audio_int16 = (audio_data * 32767).astype(np.int16)
                    
                    # Create WAV format
                    wav_buffer = io.BytesIO()
                    with wave.open(wav_buffer, 'wb') as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2) 
                        wav_file.setframerate(sample_rate)
                        wav_file.writeframes(audio_int16.tobytes())
                    
                    wav_data = wav_buffer.getvalue()
                    print(f"📊 Sending test audio: {len(wav_data)} bytes")
                    
                    # Send audio data
                    await websocket.send(wav_data)
                    
                    # Wait for translation
                    print("⏳ Waiting for translation...")
                    response = await websocket.recv()
                    
                    if isinstance(response, bytes):
                        print(f"🎵 Received translated audio: {len(response)} bytes")
                        print("✅ Complete recording mode test successful!")
                    else:
                        print(f"📄 Received message: {response}")
                else:
                    print(f"❌ Unexpected response: {msg}")
                    
        except Exception as e:
            print(f"❌ Complete recording test failed: {e}")
    
    async def test_streaming_mode(self):
        """Test the continuous streaming translation mode"""
        print("\n🌊 Testing Continuous Streaming Mode...")
        
        try:
            async with websockets.connect(self.ws_url_streaming) as websocket:
                # Send start streaming message
                start_msg = {
                    "type": "start_stream",
                    "src_lang": "eng", 
                    "tgt_lang": "ben",
                    "sample_rate": 16000,
                    "chunk_size": 1024
                }
                
                await websocket.send(json.dumps(start_msg))
                print("📤 Sent start streaming message")
                
                # Wait for ready confirmation
                response = await websocket.recv()
                msg = json.loads(response)
                
                if msg.get("type") == "stream_ready":
                    print("✅ Server ready for streaming mode")
                    
                    # Simulate streaming audio chunks
                    sample_rate = 16000
                    chunk_size = 1024  # 64ms at 16kHz
                    num_chunks = 50     # ~3.2 seconds total
                    
                    print(f"📊 Sending {num_chunks} audio chunks ({chunk_size} samples each)")
                    
                    for i in range(num_chunks):
                        # Generate chunk (varying frequency for more interesting audio)
                        t = np.linspace(i * chunk_size / sample_rate, 
                                      (i + 1) * chunk_size / sample_rate, 
                                      chunk_size)
                        frequency = 440 + (i * 10)  # Gradually increasing pitch
                        audio_chunk = 0.2 * np.sin(2 * np.pi * frequency * t)
                        
                        # Convert to int16
                        chunk_int16 = (audio_chunk * 32767).astype(np.int16)
                        
                        # Send chunk
                        await websocket.send(chunk_int16.tobytes())
                        
                        if (i + 1) % 10 == 0:
                            print(f"📤 Sent {i + 1} chunks...")
                        
                        # Small delay to simulate real-time streaming
                        await asyncio.sleep(0.064)  # 64ms = chunk duration
                        
                        # Check for any responses
                        try:
                            response = await asyncio.wait_for(websocket.recv(), timeout=0.001)
                            if isinstance(response, bytes):
                                print(f"🎵 Received streaming translation: {len(response)} bytes")
                        except asyncio.TimeoutError:
                            pass  # No response yet, continue streaming
                    
                    print("📤 Finished sending audio chunks")
                    
                    # Send stop streaming message
                    stop_msg = {"type": "stop_stream"}
                    await websocket.send(json.dumps(stop_msg))
                    print("⏹️ Sent stop streaming message")
                    
                    # Wait for final responses
                    try:
                        while True:
                            response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                            if isinstance(response, bytes):
                                print(f"🎵 Final streaming response: {len(response)} bytes")
                            else:
                                msg = json.loads(response)
                                if msg.get("type") == "stream_end":
                                    print("✅ Streaming session ended successfully")
                                    break
                                else:
                                    print(f"📄 Server message: {msg}")
                    except asyncio.TimeoutError:
                        print("⏰ No more responses - streaming test complete")
                    
                    print("✅ Streaming mode test successful!")
                else:
                    print(f"❌ Unexpected response: {msg}")
                    
        except Exception as e:
            print(f"❌ Streaming test failed: {e}")
    
    async def run_tests(self):
        """Run both test modes"""
        print("🚀 Starting SeamlessM4T v2 Streaming Tests")
        print("=" * 50)
        
        # Test complete recording mode
        await self.test_complete_recording_mode()
        
        # Wait a bit between tests
        await asyncio.sleep(2)
        
        # Test streaming mode
        await self.test_streaming_mode()
        
        print("\n" + "=" * 50)
        print("🎉 All tests completed!")

async def main():
    """Main test function"""
    client = StreamingTestClient()
    await client.run_tests()

if __name__ == "__main__":
    print("SeamlessM4T v2 Streaming Translation Test Client")
    print("Make sure the server is running on localhost:7860")
    print("Press Ctrl+C to stop\n")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")