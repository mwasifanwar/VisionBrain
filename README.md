<h1>VisionBrain: Real-Time Multimodal AI Perception System</h1>

<p><strong>VisionBrain</strong> is an advanced real-time AI system that integrates computer vision, natural language processing, and speech synthesis to create an intelligent visual perception engine. The system sees through webcam input, understands visual scenes contextually, and describes them through natural speech—effectively creating an artificial visual cortex with language capabilities.</p>

<h2>Overview</h2>
<p>This project implements a sophisticated multimodal AI pipeline that combines state-of-the-art object detection with vision-language understanding. Unlike traditional computer vision systems that simply detect objects, VisionBrain generates contextual descriptions and communicates them through natural speech, enabling human-like visual comprehension and interaction.</p>

<img width="1040" height="576" alt="image" src="https://github.com/user-attachments/assets/fc4199fc-905d-41a7-af04-8fca33ef202d" />


<p><strong>Key Innovation:</strong> Real-time fusion of YOLO-based object detection with BLIP vision-language modeling, creating a closed-loop perception system that not only sees but understands and communicates visual information.</p>

<h2>System Architecture</h2>
<p>The system follows a modular pipeline architecture with three core components:</p>

<pre><code>Webcam Input → Object Detection → Scene Understanding → Speech Synthesis → Audio Output
     ↓              ↓                  ↓                   ↓
   OpenCV         YOLOv5              BLIP                gTTS
  (Capture)    (Detection)       (Captioning)         (Synthesis)
</code></pre>


<img width="621" height="701" alt="image" src="https://github.com/user-attachments/assets/eedd2a10-dcb4-410a-9d57-58604d495f9e" />


<p><strong>Data Flow:</strong> Raw video frames are processed through parallel detection and understanding pathways, with temporal coordination ensuring smooth real-time performance and natural speech pacing.</p>

<h2>Technical Stack</h2>
<ul>
  <li><strong>Computer Vision:</strong> OpenCV, YOLOv5, PIL</li>
  <li><strong>Vision-Language Model:</strong> Salesforce BLIP (Base)</li>
  <li><strong>Speech Synthesis:</strong> gTTS (Google Text-to-Speech), Pygame</li>
  <li><strong>Deep Learning Framework:</strong> PyTorch, Transformers</li>
  <li><strong>Core Dependencies:</strong> NumPy, Accelerate</li>
</ul>

<h2>Mathematical Foundation</h2>
<p>The system integrates multiple AI models through a coordinated inference pipeline:</p>

<p><strong>Object Detection (YOLOv5):</strong> Implements anchor-based detection with CIoU loss:</p>
<p>$L_{YOLO} = \lambda_{coord} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} [(x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2] + \lambda_{coord} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} [(\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2] + \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} (C_i - \hat{C}_i)^2 + \lambda_{noobj} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{noobj} (C_i - \hat{C}_i)^2 + \sum_{i=0}^{S^2} \mathbb{1}_{i}^{obj} \sum_{c \in classes} (p_i(c) - \hat{p}_i(c))^2$</p>

<p><strong>Vision-Language Understanding (BLIP):</strong> Uses cross-modal attention between visual features and text tokens:</p>
<p>$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$</p>
<p>where $Q = \text{Linear}(V_{visual})$, $K, V = \text{Linear}(E_{text})$</p>

<h2>Features</h2>
<ul>
  <li><strong>Real-time Multimodal Processing:</strong> Simultaneous visual analysis and language generation at 15-30 FPS</li>
  <li><strong>Contextual Scene Understanding:</strong> Goes beyond object detection to generate natural language descriptions</li>
  <li><strong>Intelligent Speech Synthesis:</strong> Natural voice output with configurable timing and language options</li>
  <li><strong>Interactive Control System:</strong> Multiple operation modes with keyboard controls</li>
  <li><strong>Modular Architecture:</strong> Easily extensible components for research and development</li>
  <li><strong>Zero-Shot Capabilities:</strong> No training required—works out-of-the-box with pre-trained models</li>
</ul>

<img width="855" height="666" alt="image" src="https://github.com/user-attachments/assets/0da243d0-bd79-4362-b635-d86f6fa76f29" />


<h2>Installation</h2>
<p><strong>Prerequisites:</strong> Python 3.8+, 4GB RAM minimum, webcam, internet connection for model download</p>

<pre><code># Clone repository
git clone https://github.com/mwasifanwar/VisionBrain.git
cd VisionBrain

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download models automatically on first run
# Models will be cached for subsequent executions
</code></pre>

<h2>Usage / Running the Project</h2>
<p>Execute the main application with default settings:</p>

<pre><code>python main.py</code></pre>

<p><strong>Interactive Controls:</strong></p>
<ul>
  <li><code>Q</code> - Quit application</li>
  <li><code>S</code> - Force immediate scene description</li>
  <li><code>Space</code> - Toggle auto-description mode (15-second intervals)</li>
</ul>

<p><strong>Advanced Usage:</strong> Modify core parameters in the source code for research purposes:</p>
<pre><code># In main.py - Adjust analysis interval
analysis_interval = 10  # seconds between auto-descriptions

# In core/vision.py - Change detection confidence threshold
confident_objects = objects[objects['confidence'] > 0.6]  # from 0.5
</code></pre>

<h2>Configuration / Parameters</h2>
<p><strong>Core System Parameters:</strong></p>
<ul>
  <li><code>analysis_interval</code>: 15 seconds (auto-description frequency)</li>
  <li><code>detection_confidence</code>: 0.5 (YOLO object detection threshold)</li>
  <li><code>speech_cooldown</code>: 10 seconds (minimum time between speech outputs)</li>
  <li><code>max_caption_length</code>: 50 tokens (BLIP generation limit)</li>
  <li><code>num_beams</code>: 5 (beam search for text generation)</li>
</ul>

<p><strong>Performance Optimization:</strong> Frame resolution (640×480), batch processing disabled for real-time operation, model precision (FP32).</p>

<h2>Folder Structure</h2>
<pre><code>VisionBrain/
├── main.py                 # Application entry point and control logic
├── core/                   # Core AI components
│   ├── vision.py           # Computer vision and understanding module
│   └── speech.py           # Text-to-speech engine with threading
├── requirements.txt        # Python dependencies specification
└── README.md              # Project documentation

# Model Cache (auto-created)
~/.cache/
  └── torch/
      ├── hub/             # YOLOv5 weights
      └── transformers/    # BLIP model weights
</code></pre>

<h2>Results / Experiments / Evaluation</h2>
<p><strong>Performance Metrics:</strong></p>
<ul>
  <li><strong>Inference Speed:</strong> 100-200ms per frame (5-10 FPS) on CPU, 30+ FPS on GPU</li>
  <li><strong>Object Detection Accuracy:</strong> mAP@0.5: 56.8% (YOLOv5s on COCO dataset)</li>
  <li><strong>Caption Quality:</strong> BLIP achieves CIDEr score of 117.4 on COCO Captions</li>
  <li><strong>Memory Usage:</strong> ~1.5GB RAM during operation</li>
</ul>

<img width="887" height="625" alt="image" src="https://github.com/user-attachments/assets/b0d7511d-2bc6-4d8a-bd3c-be51f244cf75" />


<p><strong>Qualitative Evaluation:</strong> The system demonstrates robust scene understanding across diverse environments including indoor spaces, outdoor scenes, and complex multi-object arrangements. Generated descriptions show contextual awareness beyond simple object listing.</p>

<h2>References / Citations</h2>
<ol>
  <li>J. Redmon et al., "You Only Look Once: Unified, Real-Time Object Detection," <em>arXiv:1506.02640</em>, 2016.</li>
  <li>J. Li et al., "BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation," <em>International Conference on Machine Learning</em>, 2022.</li>
  <li>A. Kirillov et al., "Segment Anything," <em>arXiv:2304.02643</em>, 2023.</li>
  <li>A. Vaswani et al., "Attention Is All You Need," <em>Advances in Neural Information Processing Systems</em>, 2017.</li>
</ol>

<h2>Acknowledgements</h2>
<p>This project builds upon several open-source contributions:</p>
<ul>
  <li><strong>YOLOv5:</strong> Ultralytics for efficient object detection implementation</li>
  <li><strong>BLIP Model:</strong> Salesforce Research for vision-language pre-training</li>
  <li><strong>Hugging Face Transformers:</strong> Model hosting and inference optimization</li>
  <li><strong>Google Text-to-Speech:</strong> Natural voice synthesis API</li>
  <li><strong>OpenCV:</strong> Real-time computer vision infrastructure</li>
</ul>

<p><em>This project demonstrates the practical integration of multiple AI modalities to create interactive, intelligent systems that bridge the gap between visual perception and natural language communication.</em></p>

<br>

<h2 align="center">✨ Author</h2>

<p align="center">
  <b>M Wasif Anwar</b><br>
  <i>AI/ML Engineer | Effixly AI</i>
</p>

<p align="center">
  <a href="https://www.linkedin.com/in/mwasifanwar" target="_blank">
    <img src="https://img.shields.io/badge/LinkedIn-blue?style=for-the-badge&logo=linkedin" alt="LinkedIn">
  </a>
  <a href="mailto:wasifsdk@gmail.com">
    <img src="https://img.shields.io/badge/Email-grey?style=for-the-badge&logo=gmail" alt="Email">
  </a>
  <a href="https://mwasif.dev" target="_blank">
    <img src="https://img.shields.io/badge/Website-black?style=for-the-badge&logo=google-chrome" alt="Website">
  </a>
  <a href="https://github.com/mwasifanwar" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
  </a>
</p>

<br>

---

<div align="center">

### ⭐ Don't forget to star this repository if you find it helpful!

</div>
