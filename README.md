# 🧠 CoCo Ingestion Module (Windows)

The **CoCo Ingestion Module** prepares documents for the CoCo AI kiosk by processing files into a Retrieval-Augmented Generation (RAG) knowledge package.

This repository contains the **Windows-based ingestion pipeline** responsible for document parsing, chunking, embedding, and packaging of knowledge data.
The web application is maintained in a separate repository.

---

# ⚙️ Installation Guide (Windows)

## 📌 Requirements

* Windows 10 or later
* Internet connection
* OpenAI API Key

---

## 🪟 Step 1: Download the Ingestion Module

1. Create a new folder on your Desktop.
2. Copy the folder path.

Example:

```
C:\Users\YourName\Desktop\coco-ingestion
```

3. Open Terminal (PowerShell or Command Prompt).
4. Navigate to the folder path:

```powershell
cd C:\Users\YourName\Desktop
```

5. Clone the repository:

```powershell
git clone --depth 1 --branch main https://github.com/Kiruzato/coco-ingestion-module.git coco-ingestion
```

6. Wait for the download to complete.

---

## ⚙️ Step 2: Setup the Environment

Run the setup script:

```powershell
.\INGESTION_MODULE\setup.ps1
```

Wait for the setup process to complete successfully.

---

# ✅ Installation Complete

The CoCo Ingestion Module is now ready to use.

---

# 📘 Usage Guide

## ⚠️ Preparation of Documents

Place all documents you want to ingest inside a dedicated folder.

IMPORTANT:

* Do NOT use your backup folder as the ingestion folder.
* Always create a separate disposable folder for ingestion.
* The ingestion process may modify or reorganize files.
* Keep backup copies of your documents in another location.

Example:

```
Documents (Backup)
   ├── original_files

Documents_for_Ingestion (Disposable)
   ├── file1.pdf
   ├── file2.docx
```

---

## 🚀 Steps to Run the Ingestion Module

1. Navigate to the folder where the ingestion module was downloaded.

2. Run the application:

```
INGESTION_MODULE\CoCo Ingestion Module.bat
```

The Ingestion Module application window will appear.

3. Enter your OpenAI API Key and save.

4. Select the folder containing the documents to ingest.

5. Start the ingestion process and wait for completion.

6. After processing, a RAG package (.zip file) will be generated.

---

## 📦 Output

The ingestion process produces a RAG package (.zip file) containing:

* processed text chunks
* embeddings
* metadata
* vector index files

This package is used by the CoCo Web App knowledge base.

---

## 🔗 Next Step

Upload the generated RAG package to the CoCo Web App.

For detailed instructions, refer to the Instruction Manual.

---

# 📂 Repository Scope

This repository includes:

* document preprocessing pipeline
* text chunking and normalization
* embedding generation
* vector database packaging
* Windows setup scripts

This repository does NOT include:

* Web App
* Kiosk interface
* Voice interaction components
