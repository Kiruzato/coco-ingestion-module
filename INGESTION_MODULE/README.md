# CoCo Standalone Ingestion Module

This module processes documents and creates vector store packages for deployment to Raspberry Pi or other runtime environments.

## Why Standalone?

The Raspberry Pi runs in **runtime mode only** - it loads and uses pre-built vector stores but does NOT perform document ingestion. This design:

- Reduces RPi resource usage (no heavy embedding computation)
- Ensures consistent vector stores across deployments
- Speeds up RPi setup (no ingestion step)
- Allows document updates without RPi access

## Setup

```bash
cd INGESTION_MODULE

# Create virtual environment (Python 3.11 recommended)
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Create .env with your OpenAI API key
echo OPENAI_API_KEY=sk-your-key-here > .env
```

## Usage

### Basic Usage

```bash
cd INGESTION_MODULE

# Ingest from default folder (documents_to_ingest/)
python ingest.py

# Ingest from custom folder
python ingest.py ./my_documents

# Custom settings
python ingest.py --chunk-size 1000 --chunk-overlap 100 --no-zip
```

### Admin CLI

```bash
cd INGESTION_MODULE

# List ingested documents
python modules/admin.py list

# Show document info
python modules/admin.py info abc123

# Delete a document
python modules/admin.py delete abc123

# Rebuild vector store
python modules/admin.py rebuild
```

### Output

The module creates a timestamped package folder inside the documents folder:

```
my_documents/
├── file1.pdf
├── file2.docx
└── rag_package_2026-02-15_14-30/
    ├── vector_store/
    │   ├── index.faiss
    │   └── index.pkl
    ├── document_registry.json
    └── rag_package_2026-02-15_14-30.zip  (ready for upload)
```

## Deploying to Raspberry Pi

### Option 1: Admin UI Upload (Recommended)

1. Open Admin UI: `http://<rpi-ip>:8000/admin`
2. Click **RAG Package** in sidebar
3. Upload the `.zip` file
4. Vector store reloads automatically

### Option 2: Manual Copy

```bash
scp -r rag_package_2026-02-15_14-30/vector_store/ pi@<rpi-ip>:/home/pi/coco/WEB_APP/modules/
sudo systemctl restart coco
```

### Option 3: Git (for version control)

```bash
cp -r rag_package_2026-02-15_14-30/vector_store/ WEB_APP/modules/
git add WEB_APP/modules/vector_store/
git commit -m "Update vector store"
git push
# On RPi: git pull && sudo systemctl restart coco
```

## Requirements

- Python 3.11+ (recommended)
- OpenAI API key (for embeddings)
- Dependencies from `requirements.txt`

## Environment Variables

Create `.env` in `INGESTION_MODULE/`:

```
OPENAI_API_KEY=sk-your-key-here
```

## Supported Document Formats

- `.txt` - Plain text
- `.pdf` - PDF documents (layout-aware parsing via unstructured)
- `.docx` - Microsoft Word documents
