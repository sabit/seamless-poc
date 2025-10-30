# SeamlessStreaming Reference Documentation

This document provides comprehensive reference links and documentation for the SeamlessStreaming components used in this project.

## 📚 Main Repository & Documentation

### **SeamlessM4T & SeamlessStreaming**
- **Main Repository**: https://github.com/facebookresearch/seamless_communication
- **SeamlessStreaming Paper**: https://arxiv.org/abs/2312.05187 - "SeamlessStreaming: Streaming Speech Translation with Seamless" (2023)
- **SeamlessM4T Paper**: https://arxiv.org/abs/2308.11596 - "SeamlessM4T—Massively Multilingual & Multimodal Machine Translation" (2023)

## 🔧 Component Documentation

### **1. SeamlessStreaming Architecture**
- **Streaming Module**: https://github.com/facebookresearch/seamless_communication/tree/main/streaming
- **Agent Classes**: https://github.com/facebookresearch/seamless_communication/tree/main/src/seamless_communication/streaming/agents
- **Unity Pipeline**: https://github.com/facebookresearch/seamless_communication/blob/main/src/seamless_communication/streaming/agents/unity_pipeline.py
- **S2ST Agent**: https://github.com/facebookresearch/seamless_communication/blob/main/src/seamless_communication/streaming/agents/seamless_streaming_s2st.py

### **2. SimulEval Framework (Streaming Evaluation)**
- **SimulEval Repository**: https://github.com/facebookresearch/SimulEval
- **Documentation**: https://simuleval.readthedocs.io/
- **Agent API**: https://simuleval.readthedocs.io/en/latest/agent.html
- **Data Segments**: https://simuleval.readthedocs.io/en/latest/data.html
- **Actions**: https://simuleval.readthedocs.io/en/latest/actions.html

### **3. Model Components**

#### **Unity Model**
- **Purpose**: Main speech-to-speech translation model
- **Architecture**: Transformer-based multimodal translation
- **Model Card**: https://github.com/facebookresearch/seamless_communication/tree/main/models

#### **Monotonic Decoder**
- **Purpose**: Handles streaming decoding with monotonic attention
- **Paper**: "Monotonic Multihead Attention" for streaming applications
- **Implementation**: Part of the Unity pipeline for real-time processing

#### **Vocoder (PreTSSEL)**
- **Purpose**: Converts speech units back to audio waveform
- **Architecture**: Pre-trained Speech Synthesis Enhancement Layer
- **Quality**: High-fidelity speech reconstruction

### **4. fairseq2 (Model Backend)**
- **Repository**: https://github.com/facebookresearch/fairseq2
- **Documentation**: https://github.com/facebookresearch/fairseq2/tree/main/docs
- **Model Loading**: https://github.com/facebookresearch/fairseq2/blob/main/docs/model_loading.md
- **Device Management**: https://github.com/facebookresearch/fairseq2/blob/main/docs/device.md

## 🛠️ Configuration & Examples

### **CLI Examples**
- **Streaming Examples**: https://github.com/facebookresearch/seamless_communication/tree/main/streaming/examples
- **Command Line Usage**: https://github.com/facebookresearch/seamless_communication/blob/main/streaming/README.md

### **Agent Configuration**
- **S2ST Configuration**: https://github.com/facebookresearch/seamless_communication/blob/main/src/seamless_communication/streaming/agents/seamless_streaming_s2st.py
- **Pipeline Setup**: https://github.com/facebookresearch/seamless_communication/blob/main/src/seamless_communication/streaming/agents/unity_pipeline.py
- **Argument Parsing**: Agent classes use `add_args()` methods for parameter configuration

### **Model Parameters Reference**

#### **Core Model Parameters**
- `model_name`: "seamless_streaming_unity"
- `unity_model_name`: "seamless_streaming_unity"
- `monotonic_decoder_model_name`: "seamless_streaming_monotonic_decoder"
- `vocoder_name`: "vocoder_pretssel"

#### **Device & Precision**
- `device`: torch.device("cuda") - GPU required
- `dtype`: torch.float16 - Half precision for efficiency
- `fp16`: True - Enable FP16 mode

#### **Streaming Parameters**
- `min_unit_chunk_size`: 50 - Minimum units to accumulate
- `d_factor`: 1.0 - Duration factor for timing
- `shift_size`: 160 - Audio frame shift size  
- `segment_size`: 2000 - Audio segment size
- `denormalize`: False - Output normalization

#### **Text Generation**
- `max_len_a`: 1.2 - Length penalty coefficient a
- `max_len_b`: 100 - Length penalty coefficient b
- `beam_size`: 5 - Beam search width
- `len_penalty`: 1.0 - Length penalty weight

#### **Language Configuration**
- `source_lang`: Language code (e.g., "eng")
- `target_lang`: Language code (e.g., "ben") 
- `task`: "s2st" for speech-to-speech translation

## 📖 Research Papers

### **Primary Papers**
1. **SeamlessStreaming (2023)**
   - Title: "SeamlessStreaming: Streaming Speech Translation with Seamless"
   - Link: https://arxiv.org/abs/2312.05187
   - Focus: Real-time streaming translation architecture

2. **SeamlessM4T (2023)**
   - Title: "SeamlessM4T—Massively Multilingual & Multimodal Machine Translation"
   - Link: https://arxiv.org/abs/2308.11596
   - Focus: Multilingual multimodal translation foundation

### **Related Research**
- **Monotonic Attention**: Papers on streaming attention mechanisms
- **Speech-to-Speech Translation**: End-to-end S2ST architectures
- **Real-time Translation**: Low-latency translation systems

## 🔍 Debugging & Development

### **Parameter Discovery**
- **Method**: Each agent class has `add_args()` method defining required parameters
- **Source Code**: Check agent source files for complete parameter lists
- **Error-driven**: AttributeErrors reveal missing parameters systematically

### **Model Downloads**
- **Automatic**: Models download automatically on first use
- **Cache Location**: `~/.cache/huggingface/` or similar
- **Size**: Unity models ~3-4GB each

### **Common Issues**
- **CUDA Requirement**: These models do not run on CPU
- **Memory**: Requires significant GPU memory (8GB+ recommended)
- **Parameters**: Extensive parameter requirements for all pipeline agents

## 🏗️ Architecture Overview

```
SeamlessStreamingS2STAgent
├── Unity Pipeline
│   ├── Speech Encoder
│   ├── Monotonic Decoder  
│   └── Speech Decoder
├── Vocoder (PreTSSEL)
└── SimulEval Framework
    ├── SpeechSegment handling
    ├── Action management
    └── Streaming coordination
```

## 📝 Installation

### **Requirements**
```bash
pip install git+https://github.com/facebookresearch/seamless_communication.git
pip install fairseq2
pip install simuleval
```

### **Hardware**
- **GPU**: CUDA-capable GPU required
- **Memory**: 8GB+ GPU memory recommended
- **Storage**: 10GB+ for model cache

---

*This reference document is maintained alongside the SeamlessStreaming implementation in this repository.*