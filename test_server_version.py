#!/usr/bin/env python3
"""
Quick test to check server version and pipeline method
"""

import asyncio
import websockets
import json
import ssl

async def test_server_info():
    # Create SSL context for self-signed certificates
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    uri = "wss://34.81.26.139:7860/ws/stream"
    
    try:
        print(f"🔗 Connecting to {uri}...")
        async with websockets.connect(uri, ssl=ssl_context) as websocket:
            print("✅ Connected!")
            
            # Send a diagnostic message
            diagnostic_msg = {
                "type": "diagnostic",
                "request": "server_info"
            }
            
            await websocket.send(json.dumps(diagnostic_msg))
            print("📤 Sent diagnostic request")
            
            # Wait for response
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                print(f"📥 Response: {response}")
            except asyncio.TimeoutError:
                print("⏰ No diagnostic response - server may not support diagnostics")
                
                # Try sending a regular start message to see if agent.pop() method is being used
                start_msg = {
                    "type": "start_stream",
                    "srcLang": "eng", 
                    "tgtLang": "ben",
                    "task": "s2st",
                    "sample_rate": 16000
                }
                
                await websocket.send(json.dumps(start_msg))
                print("📤 Sent start stream message")
                
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                print(f"📥 Start response: {response}")
                
                # Send a tiny amount of audio to see logging
                tiny_audio = b'\x00' * 100  # 100 bytes of silence
                await websocket.send(tiny_audio)
                print("📤 Sent tiny audio chunk")
                
                # Check for any immediate response
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    print(f"📥 Audio response: {response}")
                except asyncio.TimeoutError:
                    print("⏰ No immediate audio response")
                
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_server_info())