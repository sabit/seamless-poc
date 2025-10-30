# Proper SeamlessStreaming Installation Guide

This guide shows how to install the official Facebook Research SeamlessStreaming implementation for proper streaming translation.

## Why Our Previous Implementation Had Issues

Our previous implementation used the Hugging Face Transformers version of SeamlessM4T, which is designed for batch processing, not streaming. The official SeamlessStreaming implementation uses:

1. **Specialized Streaming Models**: `seamless_streaming_unity` and `seamless_streaming_monotonic_decoder`
2. **SimulEval Framework**: For incremental, real-time processing
3. **Agent Pipeline**: Separate components for feature extraction, encoding, decoding, and vocoding
4. **Monotonic Attention**: Enables streaming without waiting for complete input

## Installation Steps

### 1. Install Prerequisites

```bash
# First, ensure you have fairseq2 (required for seamless models)
# Note: fairseq2 only has pre-built packages for Linux x86-64 and Apple Silicon Mac
pip install fairseq2

# Install libsndfile if not available (Linux)
# Ubuntu/Debian: sudo apt-get install libsndfile1
# CentOS/RHEL: sudo yum install libsndfile

# Install ffmpeg (required by Whisper dependency)
# Ubuntu/Debian: sudo apt-get install ffmpeg
# Windows: Download from https://ffmpeg.org/
# macOS: brew install ffmpeg
```

### 2. Install SeamlessStreaming

```bash
# Clone and install the official repository
git clone https://github.com/facebookresearch/seamless_communication.git
cd seamless_communication
pip install .

# Or install specific dependencies for streaming
pip install simuleval  # For streaming evaluation framework
pip install torch torchaudio  # PyTorch for model execution
pip install transformers  # For tokenizers and processors
```

### 3. Install Additional Dependencies

```bash
# For audio processing
pip install librosa soundfile
# For web server
pip install fastapi uvicorn websockets
# For audio format support
pip install pydub
```

## Model Architecture Comparison

### Our Previous Approach (Incorrect for Streaming)
```python
# Uses batch processing model - processes complete audio chunks
model = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained("facebook/seamless-m4t-v2-large")
processor = SeamlessM4TProcessor.from_pretrained("facebook/seamless-m4t-v2-large")

# Fixed chunk processing with overlap - NOT true streaming
chunk = audio_buffer[-4_seconds:]
result = model.generate(**inputs)  # Processes entire chunk at once
```

### Proper SeamlessStreaming Approach
```python
# Uses streaming-specific models with incremental processing
from seamless_communication.streaming.agents.seamless_streaming_s2st import SeamlessStreamingS2STAgent
from simuleval.data.segments import SpeechSegment

# Agent pipeline processes audio incrementally
agent = SeamlessStreamingS2STAgent(args)
segment = SpeechSegment(content=audio_chunk, sample_rate=16000, finished=False)
action = agent.policy(segment)  # Incremental processing
```

## Key Differences

| Aspect | Our Previous | Official Streaming |
|--------|-------------|-------------------|
| **Model** | `SeamlessM4Tv2ForSpeechToSpeech` | `seamless_streaming_unity` + `seamless_streaming_monotonic_decoder` |
| **Processing** | Fixed 4s chunks with overlap | Incremental segment processing |
| **Attention** | Standard attention | Monotonic attention for streaming |
| **Latency** | High (waits for chunk completion) | Low (outputs as soon as possible) |
| **Quality** | Good for complete audio | Optimized for partial input |
| **VAD** | Manual energy thresholds | Integrated Silero VAD |

## Usage Examples

### Official SeamlessStreaming (Recommended)
```python
# Initialize streaming agent
agent_args = {
    "device": "cuda",
    "task": "s2st",
    "tgt_lang": "ben",
    "src_lang": "eng",
    "unity_model_name": "seamless_streaming_unity",
    "monotonic_decoder_model_name": "seamless_streaming_monotonic_decoder",
    "vad": True,
    "min_starting_wait": 1000,
}

agent = SeamlessStreamingS2STAgent(agent_args)

# Process audio incrementally
for audio_chunk in audio_stream:
    segment = SpeechSegment(content=audio_chunk, sample_rate=16000)
    action = agent.policy(segment)
    if isinstance(action, WriteAction):
        # Got translation output
        play_audio(action.content)
```

### CLI Usage
```bash
# Streaming evaluation (includes inference)
streaming_evaluate \
  --task s2st \
  --data-file test.tsv \
  --audio-root-dir /path/to/audio \
  --output /path/to/output \
  --tgt-lang ben \
  --no-scoring  # Skip evaluation, just do inference
```

## Testing the Implementation

1. **Run the new streaming server**:
   ```bash
   cd backend
   python streaming_server.py
   ```

2. **Check the health endpoint**:
   ```bash
   curl http://localhost:7860/health
   # Should show: {"official_streaming": true, "fallback_available": true}
   ```

3. **Test with frontend**: The frontend should now use proper streaming with much lower latency and better quality.

## Troubleshooting

### Common Issues

1. **"fairseq2 not available"**:
   - Only works on Linux x86-64 and Apple Silicon Mac
   - Windows users need WSL or Docker

2. **"libsndfile not found"**:
   - Install system audio libraries
   - Ubuntu: `sudo apt-get install libsndfile1`

3. **Model loading errors**:
   - Ensure sufficient GPU memory (>8GB recommended)
   - Models are downloaded automatically on first use

4. **Import errors**:
   - Make sure seamless_communication is properly installed
   - Check that fairseq2 installation completed successfully

### Performance Notes

- **GPU Memory**: Streaming models require ~4-6GB VRAM
- **Latency**: Official streaming should achieve <1s latency
- **Quality**: Should eliminate repetitive translations
- **Stability**: Much more robust than chunked processing

## Next Steps

If the official installation works, you should see:
- ✅ Lower latency (near real-time)
- ✅ Better translation quality
- ✅ No repetitive output
- ✅ Proper voice activity detection
- ✅ Smoother audio streaming

The new `streaming_server.py` will automatically detect and use the official implementation if available, falling back to the improved chunked approach if not.