# Generative AI Service

A modular FastAPI backend service providing asynchronous endpoints for AI-powered content generation including 3D models, audio synthesis, video generation, chat, and image creation. Designed for high-performance, extendability, and ease of integration.

---

## Features

- Generate 3D geometry meshes from text prompts with OpenAI's Shap-E pipeline.
- Synthesize high-quality audio from text prompts with selectable voice presets.
- Create video sequences derived from image inputs and parameters.
- Chat endpoint for AI dialogue generation.
- Image generation from textual descriptions.
- Fully asynchronous design using FastAPI and Python concurrency primitives.
- Simple dependency and environment setup with Docker (optional).
- Streaming responses for efficient media delivery.

---

## Getting Started

### Prerequisites

- Python 3.10+
- `pip` package manager
- Optional but recommended: Docker and Docker Compose for containerized setup
- GPU recommended for 3D and audio model performance

### Installation

```
git clone https://github.com/yourusername/generative-ai-service.git
cd generative-ai-service
python -m venv venv
source venv/bin/activate # Linux/macOS

.\venv\Scripts\activate # Windows
pip install -r requirements.txt
```