# CyberGuard – AI-Powered Cyberbullying Detection Application

**A Context-Aware Cyberbullying Detection System Using Local-Global Split-Memory Transformer (LGS-MT)**

## Overview

CyberGuard is an AI-powered chat application designed to identify and moderate potentially harmful communication by analyzing messages along with their conversational context.

Unlike traditional approaches that classify messages independently, CyberGuard incorporates conversation history, contextual signals, semantic memory, and sentiment analysis to distinguish cyberbullying from friendly banter, sarcasm, and ordinary conversations.

The application implements a **Local-Global Split-Memory Transformer (LGS-MT)** architecture that combines a lightweight local MuRIL model with a global MuRIL model and memory-based contextual retrieval.

The system also provides user authentication, private messaging, friend management, moderation tracking, and an administrative dashboard.

## Key Features

- **Context-Aware Detection:** Analyzes messages using recent conversation history rather than treating each message independently.
- **Local-Global Model Architecture:** Combines a lightweight 4-layer MuRIL student model with a 12-layer global MuRIL model.
- **Semantic Memory:** Uses ChromaDB for conversation-specific and generalized contextual information retrieval.
- **Contextual Signal Analysis:** Considers offensive expressions, threat-related language, banter, reciprocity, and sentiment patterns.
- **Adaptive Model Routing:** Uses the local model for initial predictions and falls back to the global model when additional contextual analysis is required.
- **Real-Time Chat Application:** Supports private messaging and friend management.
- **Moderation System:** Tracks potentially harmful messages and maintains moderation records.
- **Admin Dashboard:** Provides administrative access to moderation-related information.

## LGS-MT Architecture

The proposed Local-Global Split-Memory Transformer architecture consists of the following components:

**1. Local Model**
- Uses a distilled 4-layer MuRIL student model.
- Performs lightweight message classification.
- Supports context-aware detection.

**2. Global Model**
- Uses a 12-layer MuRIL model.
- Performs deeper contextual analysis when the local model is uncertain.
- Incorporates generalized semantic memory for contextual retrieval.

**3. Memory Components**
- Local Memory maintains conversation-specific contextual information.
- Global Memory stores generalized patterns and relevant semantic information.
- ChromaDB provides vector-based storage and retrieval.

**4. Score Fusion**
- Combines model predictions with contextual signals and memory-based evidence.
- Produces the final cyberbullying risk score and classification.

## Technology Stack

| Component | Technologies |
|---|---|
| Frontend | React.js, Vite, JavaScript |
| Backend | Python, FastAPI |
| Deep Learning | PyTorch, Hugging Face Transformers |
| NLP Model | MuRIL |
| Vector Database | ChromaDB |
| Database | SQLite |
| API Communication | Axios |
| Testing | Pytest |

## Project Structure

```text
CyberGuard/
│
├── backend/
│   ├── test/
│   ├── tests/
│   ├── __init__.py
│   ├── auth.py
│   ├── backfill_chroma.py
│   ├── chat.py
│   ├── create_admin.py
│   ├── database.py
│   ├── lgs_pipeline.py
│   ├── main.py
│   ├── memory.py
│   ├── pipeline.py
│   ├── requirements.txt
│   └── users.db
│
├── frontend/
│   ├── public/
│   ├── src/
│   ├── .gitignore
│   ├── eslint.config.js
│   ├── index.html
│   ├── package-lock.json
│   ├── package.json
│   └── vite.config.js
│
├── data/
├── model/
├── chroma_db/
├── .gitignore
└── readme.txt
```

**Note:** `data/`, `model/`, and `chroma_db/` contain project data, trained model resources, and vector database storage respectively. Their contents may need to be obtained or generated separately.

## Installation and Setup

### Prerequisites

- Python 3.10 or later
- Node.js and npm
- Git

### 1. Clone the Repository

```bash
git clone <repository-url>
cd CyberGuard
```

### 2. Backend Setup

Create and activate a virtual environment:

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r backend/requirements.txt
```

Start the backend server from the project root:

```bash
python -m uvicorn backend.main:app --reload --port 8000
```

The backend API will be available at:

`http://localhost:8000`

API documentation:

`http://localhost:8000/docs`

### 3. Frontend Setup

Open a new terminal and navigate to the frontend directory:

```bash
cd frontend
npm install
npm run dev
```

The frontend development server will be available at:

`http://localhost:5173`

## Model Requirements

CyberGuard requires the trained model checkpoints to perform inference.

The expected model directories are:

- `model/global_model/muril_cyberbullying/`
- `model/local_model/muril_student_4layer/`

The corresponding model weights, configuration files, and tokenizer files must be available in these directories.

**Note:** Model checkpoints and datasets are not included in the repository. They must be obtained separately and placed in the expected locations before running the detection pipeline.

## Testing

The backend contains test modules for validating the application's components and detection pipeline.

Run the tests from the project root using:

```bash
pytest
```

The application can also be validated through its API endpoints and frontend interface while the backend server is running.

## Future Enhancements

- Improve detection across regional languages, Hinglish, Kannada-English, and other code-mixed conversations.
- Develop adaptive learning mechanisms to handle evolving slang and communication patterns.
- Optimize inference latency and scalability for larger chat environments.
- Explore multimodal cyberbullying detection using text, images, and other communication signals.
- Improve contextual memory management and model performance.

## Project Scope

The current implementation focuses on **text-based conversational cyberbullying detection** using contextual analysis, semantic memory, and transformer-based models. Multimodal detection is considered a potential future extension.

## Contributors

Developed as an academic capstone project.

## Disclaimer

CyberGuard is an academic project developed for research and educational purposes. Its predictions are intended to assist moderation and should not be treated as definitive judgments of user intent.
