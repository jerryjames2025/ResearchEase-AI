# ResearchEase AI — Version 1

ResearchEase AI is a local transformer-powered research paper analyst built
for students and researchers.

## Version 1 features

- Upload a text-based research-paper PDF
- Extract page-aware text using PyMuPDF
- Analyze long papers in manageable chunks
- Generate a structured paper analysis with Ollama
- Choose Beginner, Intermediate, Advanced, or "Explain like I am 10"
- Download the generated analysis as PDF or TXT
- Run locally without a paid LLM API

## Project structure

```text
ResearchEase-AI/
├── app.py
├── config.py
├── requirements.txt
├── README.md
└── services/
    ├── __init__.py
    ├── pdf_parser.py
    ├── prompts.py
    ├── ollama_client.py
    └── report_builder.py
```

## 1. Create and activate a virtual environment

### Windows PowerShell

```powershell
cd ResearchEase-AI
python -m venv venv
venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
venv\Scripts\Activate.ps1
```

## 2. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Confirm Ollama is installed

```powershell
ollama --version
```

Download the recommended lightweight model:

```powershell
ollama pull qwen2.5:1.5b
```

Test it:

```powershell
ollama run qwen2.5:1.5b
```

Exit the interactive model using:

```text
/bye
```

Ollama normally runs its local API at:

```text
http://localhost:11434
```

## 4. Run the application

```powershell
streamlit run app.py
```

Then open the local URL displayed in the terminal, normally:

```text
http://localhost:8501
```

## Current limitations

- Scanned PDFs are not supported yet.
- PDF tables, charts, and images are not interpreted.
- The model analyzes up to the configured number of chunks.
- Answers may still contain model errors, so important academic conclusions
  should be checked against the original paper.
- Chat-with-paper RAG will be introduced in Version 2.

## Planned Version 2

- Text chunk embeddings
- FAISS vector database
- Ask questions about the paper
- Paper-only grounded answers
- Relevant page references
- Chat history
