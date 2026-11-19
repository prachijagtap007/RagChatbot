# Python RAG chatbot backend

This backend parses local documents, creates structure-aware text chunks, embeds
them with Azure OpenAI, and uses Azure AI Search hybrid retrieval plus semantic
ranking to answer questions with source citations. The existing `index.html`
chat UI is served by FastAPI at `/`, on the same origin as `/api/chat`.

## Pipeline folders

- `app/ingestion/` - PDF, DOCX, Markdown, and plain-text parsing; directory ingestion.
- `app/chunking/` - paragraph- and sentence-aware chunks with configurable overlap.
- `app/embeddings/` - Azure OpenAI document and query vector creation.
- `app/search/` - vector index definition, batch upload, and hybrid semantic search.
- `app/formatting/` - readable source context and citation metadata.
- `app/services/` - RAG answer orchestration.
- `app/api/` - FastAPI chat endpoint and response validation.
- `app/config/` - environment-backed configuration and validation.

## Configure and run (Python 3.10+, Windows PowerShell)

From the project root:

```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `backend\.env` with your Azure OpenAI endpoint, key, API version, deployed
chat and embedding model names, Azure AI Search endpoint/key, and matching
embedding dimensions. Set `DOCUMENTS_PATH` to the folder containing your source
files (default placeholder: `./data/documents`, relative to the backend working
directory; a placeholder folder is included). The configured embedding dimensions
must match the embedding deployment and the Azure AI Search vector field.

Place PDF, DOCX, TXT, or Markdown documents in that folder, then index them:

```powershell
py -m app.ingest
```

Start the API and UI:

```powershell
py -m uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000` (serve the UI through FastAPI so its same-origin
`/api/chat` endpoint is available). The chat UI sends `POST /api/chat` with
`{"message": "..."}` and receives `{"answer": "...", "sources": [...]}`. `/health`
checks that the API process is running. Azure settings are validated when the
first chat request is made; missing configuration returns a clear HTTP 503.

## Pipeline behavior

1. Parsing retains Markdown/Word headings and PDF page numbers. Scanned PDFs
   need OCR before ingestion; image OCR is not included.
2. Chunking groups paragraphs within each section, prefers sentence boundaries,
   and repeats a configurable character-based tail to preserve context between chunks.
3. Embeddings are generated in batches. The index stores searchable text,
   document metadata, and the vector for each chunk.
4. Search combines lexical search (`search_text`) with vector search, then asks
   the configured Azure AI Search semantic ranker to rerank results.
5. The chat model receives numbered passages only and is instructed to cite
   them. Responses include document titles and available section/page metadata.

The index is created automatically when ingestion or the first chat request
runs. Existing indexes are not modified; recreate/update an existing index
manually if its vector dimensions or schema need to change.

## Tests

From the `backend` directory, after installing requirements:

```powershell
py -m unittest discover -s tests
```
