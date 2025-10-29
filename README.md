# SeamlessStreaming Real-time Speech Translation

This project implements a real-time speech-to-speech translation system using Meta's SeamlessM4T model, deployed in Docker on a GCP Compute Engine VM.

## 🚀 Quick Start

### Prerequisites
- GCP Compute Engine VM with T4 GPU (4vCPU, 16GB RAM)
- CUDA 12.4 (pre-installed on VM)
- Docker installed with GPU support (nvidia-docker2)
- Port 7860 open in firewall

### Deployment Options

#### Option 1: Native VM (Recommended - Faster)

1. **Clone/Upload the project** to your VM:
   ```bash
   # If using git
   git clone <your-repo-url>
   cd seamless-poc
   
   # Or upload the files directly to your VM
   ```

2. **Set up native environment**:
   ```bash
   chmod +x scripts/setup-native.sh
   ./scripts/setup-native.sh
   ```

3. **Run the service**:
   ```bash
   chmod +x scripts/run-native.sh
   ./scripts/run-native.sh
   ```

4. **Development mode** (auto-restart on changes):
   ```bash
   chmod +x scripts/dev-mode.sh
   ./scripts/dev-mode.sh
   ```

#### Option 2: Docker (Alternative)

1. **Test GPU setup**:
   ```bash
   chmod +x scripts/test-gpu.sh
   ./scripts/test-gpu.sh
   ```

2. **Build and run**:
   ```bash
   chmod +x scripts/build-and-run.sh
   ./scripts/build-and-run.sh
   ```

#### Access the Service
```
http://YOUR_VM_EXTERNAL_IP:7860
```

### Usage

1. **Open the web interface** in your browser
2. **Select languages**: Choose English ↔ Bangla direction
3. **Click "Start Recording"** and speak
4. **Listen** to the real-time translation

## 📁 Project Structure

```
seamless-poc/
├── backend/
│   └── main.py              # FastAPI WebSocket server
├── frontend/
│   └── index.html          # Web client interface
├── scripts/
│   ├── build-and-run.sh    # Linux deployment script
│   ├── build-and-run.bat   # Windows deployment script
│   └── stop.sh             # Cleanup script
├── Dockerfile              # Container configuration
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🔧 Technical Details

### Backend (FastAPI)
- **WebSocket streaming** for real-time audio processing
- **SeamlessM4T model** for speech-to-speech translation
- **GPU acceleration** using NVIDIA PyTorch runtime
- **Audio processing** with librosa and torchaudio
- **Chunked streaming** for low latency (< 2 seconds)

### Frontend (HTML/JavaScript)
- **WebRTC MediaRecorder** for microphone capture
- **WebSocket client** for real-time communication
- **Web Audio API** for immediate audio playback
- **Responsive UI** with language swap functionality

### Model Configuration
- **Base Image**: NVIDIA PyTorch 24.01 (CUDA 12.4 compatible)
- **Model**: `facebook/seamless-m4t-large`
- **Languages**: English (eng) ↔ Bangla (ben)
- **Audio format**: 16kHz mono WAV
- **Streaming**: 1-second audio chunks
- **GPU**: Optimized for NVIDIA T4 with CUDA 12.4

## 🛠️ Management Commands

### Container Management
```bash
# View logs
docker logs -f seamless-translator-app

# Stop service
docker stop seamless-translator-app

# Start service
docker start seamless-translator-app

# Remove container
docker rm seamless-translator-app

# Full cleanup
./scripts/stop.sh
```

### Volume Management
```bash
# Make volume script executable
chmod +x scripts/manage-volume.sh

# Create model cache volume
./scripts/manage-volume.sh create

# Inspect volume contents
./scripts/manage-volume.sh inspect

# Backup model cache
./scripts/manage-volume.sh backup

# Restore from backup
./scripts/manage-volume.sh restore backup-file.tar

# Check volume size
./scripts/manage-volume.sh size

# Clean volume (removes cached models)
./scripts/manage-volume.sh clean
```

## 🔍 Troubleshooting

### Model Import Issues
```bash
# Test model availability before building
python3 test_model.py

# Check transformers version
python3 -c "import transformers; print(transformers.__version__)"

# Verify model exists on Hugging Face
curl -I https://huggingface.co/facebook/seamless-m4t-large
```

### Container won't start
```bash
# Check CUDA compatibility and GPU access
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi

# Check container logs
docker logs seamless-translator-app

# Verify port is available
netstat -tlnp | grep :7860

# Test model loading in container
docker run --rm --gpus all seamless-translator python3 test_model.py
```

### Audio issues
- **Check browser permissions**: Allow microphone access
- **Test different browsers**: Chrome/Firefox work best
- **Check audio format**: WebRTC should encode as WebM/Opus

### Performance optimization
- **GPU memory**: Model requires ~4-6GB GPU memory
- **CPU usage**: 4vCPU should handle real-time processing
- **Network**: Ensure stable connection for WebSocket

## 📊 System Requirements

### Minimum
- **GPU**: NVIDIA T4 or equivalent
- **RAM**: 16GB system memory
- **CPU**: 4vCPU
- **Storage**: 10GB free space
- **Network**: Stable internet for model download

### Recommended
- **GPU**: T4 with 16GB VRAM
- **RAM**: 32GB system memory
- **CPU**: 8vCPU
- **Storage**: SSD with 20GB free space

## 🌐 Language Codes

- **English**: `eng`
- **Bangla**: `ben`

## 🔒 Security Notes

- Service runs on port 7860 (configured in firewall)
- No authentication implemented (POC only)
- Use HTTPS in production environments
- Consider VPN access for production use

## 📈 Performance Metrics

- **Target latency**: < 2 seconds end-to-end
- **Audio quality**: 16kHz mono
- **Throughput**: Real-time streaming
- **Memory usage**: ~8-12GB during operation

## 🐛 Known Issues

1. **Cold start**: First translation may take 10-15 seconds
2. **Memory usage**: Large model requires significant RAM
3. **Browser compatibility**: Best with Chrome/Firefox
4. **Network sensitivity**: Requires stable connection

## 🔄 Updates and Maintenance

```bash
# Rebuild with latest changes (preserves model cache)
docker stop seamless-translator-app
docker rm seamless-translator-app
./scripts/build-and-run.sh

# Force model re-download (if needed)
./scripts/manage-volume.sh clean
./scripts/build-and-run.sh

# Backup before major changes
./scripts/manage-volume.sh backup
```

## 📞 Support

For issues or questions:
1. Check container logs: `docker logs -f seamless-translator-app`
2. Verify GPU availability: `nvidia-smi`
3. Test network connectivity: `curl http://localhost:7860/health`