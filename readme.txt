# CyberGuard – AI-Powered Cyberbullying Detection Application

**A Context-Aware Cyberbullying Detection System Using Local-Global Split-Memory Transformer (LGS-MT)**

## Overview

CyberGuard is an AI-powered chat application designed to identify and moderate potentially harmful communication by analyzing messages along with their conversational context.

Unlike traditional approaches that classify messages independently, CyberGuard incorporates conversation history, contextual signals, semantic memory, and sentiment analysis to distinguish cyberbullying from friendly banter, sarcasm, and ordinary conversations.

The application implements a **Local-Global Split-Memory Transformer (LGS-MT)** architecture that combines a lightweight local MuRIL model with a global MuRIL model and memory-based contextual retrieval.

The system also provides user authentication, private messaging, friend management, moderation tracking, and an administrative dashboard.

## Key Features

- **Context-Aware Cyberbullying Detection:** Analyzes messages in the context of previous interactions rather than treating each message independently.
- **LGS-MT Architecture:** Combines local and global model processing through confidence-based routing.
- **Local and Global Memory:** Uses ChromaDB to retrieve conversation-specific information and generalized harmful communication patterns.
- **Friendly Banter Detection:** Identifies playful interactions using laughter, slang, reciprocity, and contextual signals.
- **Sentiment Trajectory Analysis:** Examines changes in conversational sentiment to identify potentially escalating negative interactions.
- **Multilingual and Code-Mixed Signals:** Incorporates MuRIL and language-specific offensive expression patterns, including regional-language expressions.
- **User Authentication:** Supports account registration, login, and protected application routes.
- **Private Chat:** Enables users to connect through friend requests and exchange messages.
- **Moderation System:** Tracks flagged messages and maintains user strike counts.
- **Admin Dashboard:** Provides administrative access to user information, conversations, and moderation statistics.

## System Architecture

### Local-Global Split-Memory Transformer (LGS-MT)

The central component of CyberGuard is the LGS-MT architecture, which combines conversation-specific processing with generalized cyberbullying detection.

The architecture consists of the following components:

**1. Local Model – 4-Layer MuRIL Student**

- Processes messages using a lightweight distilled MuRIL model.
- Analyzes conversational context.
- Provides an initial classification score and confidence estimate.

**2. Global Model – 12-Layer MuRIL**

- Acts as a second-stage classifier.
- Processes messages when the local model is uncertain or contextual signals indicate potential risk.
- Provides an additional classification score.

**3. Local Memory**

- Stores conversation-specific message representations.
- Uses semantic similarity retrieval to retrieve relevant messages from the same conversation.
- Maintains conversational context for further analysis.

**4. Global Memory**

- Stores generalized cyberbullying patterns.
- Retrieves semantically similar patterns using ChromaDB.
- Provides additional evidence during global evaluation.

**5. Contextual Signal Extraction**

The pipeline extracts several contextual signals:

- Offensive language detection
- Banter and laughter detection
- Reciprocity analysis
- Sentiment trajectory
- Threat language detection
- Defensive language detection

**6. Score Fusion and Final Decision**

The model scores and contextual signals are combined through a score-fusion mechanism to generate the final moderation score and classification decision.

### Processing Workflow

1. A user sends a message through the chat interface.
2. The backend retrieves recent conversation history.
3. The local MuRIL model evaluates the message.
4. Contextual signals and semantic memory are evaluated.
5. The routing mechanism determines whether global evaluation is required.
6. If necessary, the global MuRIL model and global memory provide additional evidence.
7. The score-fusion mechanism calculates the final moderation score.
8. The message is classified and the moderation result is returned to the application.

## Technology Stack

| Component | Technologies |
|---|---|
| Frontend | React.js, Vite, JavaScript, CSS |
| Backend | Python, FastAPI |
| Deep Learning | PyTorch, Hugging Face Transformers |
| NLP Models | MuRIL |
| Vector Database | ChromaDB |
| Database | SQLite |
| Semantic Processing | MuRIL embeddings |
| Sentiment Analysis | VADER |
| Numerical Processing | NumPy, SciPy |
| API Communication | Axios |
| Testing | Pytest, FastAPI TestClient |

## Project Structure

```text
CyberGuard/
│
├── backend/
│   ├── tests/
│   │   ├── test_api.py
│   │   ├── test_chat_db.py
│   │   ├── test_lgs_routing.py
│   │   ├── test_score_fusion.py
│   │   ├── test_unit_signals.py
│   │   └── test_vector_db.py
│   │
│   ├── test/
│   │   └── Individual pipeline and memory tests
│   │
│   ├── auth.py
│   ├── chat.py
│   ├── create_admin.py
│   ├── database.py
│   ├── lgs_pipeline.py
│   ├── main.py
│   ├── memory.py
│   ├── pipeline.py
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   │   ├── Admin.jsx
│   │   │   ├── Chat.jsx
│   │   │   ├── LgsTest.jsx
│   │   │   ├── Login.jsx
│   │   │   └── Signup.jsx
│   │   ├── api.js
│   │   ├── App.jsx
│   │   └── main.jsx
│   │
│   ├── package.json
│   └── vite.config.js
│
├── model/                 # Excluded model checkpoints
│   ├── global_model/
│   │   └── muril_cyberbullying/
│   └── local_model/
│       └── muril_student_4layer/
│
├── chroma_db/             # Generated vector database
│
├── .gitignore
└── README.md
```

## Installation and Setup

### Prerequisites

Ensure that the following are installed:

- Python 3.10 or later
- Node.js and npm
- Git

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd CyberGuard
```

### 2. Backend Setup

Create and activate a Python virtual environment.

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**

```bash
python -m venv venv
source venv/bin/activate
```

Install the backend dependencies:

```bash
pip install -r backend/requirements.txt
```

### 3. Configure the Model Checkpoints

The trained model weights are not included in this repository because of their size.

Obtain the trained model checkpoints separately and place them in the following directories:

```text
model/
├── global_model/
│   └── muril_cyberbullying/
│       ├── config.json
│       ├── model.safetensors
│       └── tokenizer files
│
└── local_model/
    └── muril_student_4layer/
        ├── config.json
        ├── model.safetensors
        └── tokenizer files
```

Both models must be compatible with the Hugging Face Transformers loading interface used by the backend.

**Note:** The application requires both trained checkpoints to initialize the detection pipeline. Merely installing the Python dependencies is not sufficient.

### 4. Start the Backend

From the project root directory, run:

```bash
python -m uvicorn backend.main:app --reload --port 8000
```

The backend will be available at:

```text
http://localhost:8000
```

FastAPI interactive API documentation:

```text
http://localhost:8000/docs
```

The application initializes its SQLite authentication and chat databases automatically. ChromaDB persistent storage is also initialized when its memory components are loaded.

### 5. Frontend Setup

Open another terminal and navigate to the frontend directory:

```bash
cd frontend
```

Install the dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

The frontend will be available at:

```text
http://localhost:5173
```

The frontend API client is configured to communicate with the backend at `http://localhost:8000`.

## Running Tests

The project includes tests for different components of the application.

Run the backend test suite from the project root:

```bash
pytest backend/tests -v
```

The tests cover:

- API endpoint functionality
- Authentication and access control
- Chat database operations
- Local-global model routing
- Score fusion and moderation decisions
- Contextual signal extraction
- ChromaDB operations and semantic memory retrieval

For live conversation scenario testing, start the backend and run:

```bash
python -m backend.tests.live_scenarios
```

## Data and Model Requirements

Large datasets, trained model checkpoints, and generated database files are excluded from this repository to keep the source repository lightweight.

| Resource | Repository Status |
|---|---|
| Training/evaluation datasets | Not included |
| Global MuRIL checkpoint | Not included |
| Local MuRIL student checkpoint | Not included |
| ChromaDB persistent storage | Generated locally |
| SQLite databases | Generated locally |
| Python cache files | Excluded |

The original training dataset is not required to run the application when the trained model checkpoints are available.

The dataset and training resources are required separately if the models need to be retrained or the research experiments reproduced.

## Future Enhancements

- Improve detection performance for regional languages, Hinglish, and code-mixed conversations.
- Introduce adaptive learning to accommodate evolving slang and communication patterns.
- Improve inference efficiency and scalability.
- Extend detection capabilities to additional modalities.
- Expand evaluation using larger and more diverse conversational datasets.
- Explore improved privacy-preserving memory management.

## Project Scope

CyberGuard is an academic capstone and research-oriented prototype focused on contextual cyberbullying detection in chat applications.

The current repository implements text-based conversational analysis. Although multimodal detection is part of the broader project direction, image, audio, and video detection should not be considered implemented functionality in the current version.

## Contributors

Developed as part of an academic capstone project.

**Project:** AI-Powered Multimodal Cyberbullying Detection in Chat Application

**Core Architecture:** Local-Global Split-Memory Transformer (LGS-MT)

---

**Note:** This project is intended for research and educational purposes. Automated moderation predictions should not be treated as definitive judgments of user intent or as a replacement for human review.
