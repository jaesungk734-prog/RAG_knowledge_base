import os 
import shutil
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from loguru import logger
from pathlib import Path


router = APIRouter()

@router.post("/ingest")
async def ingest_document(request: Request, file: UploadFile = File(...)):
    #run the pipeline after acceping a PDF upload

    ALLOWED_EXTENSIONS = (".pdf", ".md", ".markdown")

    if not file.filename.endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code = 400,
            detail = "Unsupported file type. Try .pdf or .md file"
        )

    logger.info(f"Received file for ingestion: {file.filename}")


    #save upload to a temp file
    try: 
        original_suffix = Path(file.filename).suffix.lower()
        with tempfile.NamedTemporaryFile(
            delete = False,
            suffix = original_suffix
        ) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path  = tmp.name
        logger.info(f"Saved upload to temp file: {tmp_path}")
    
        #pull from app state
        embedder = request.app.state.embedder
        store = request.app.state.store
        chunker = request.app.state.chunker

        #run pipeline
        from ingestion.pdf_loader import load_pdf
        from ingestion.note_loader import load_markdown
        
        if file.filename.lower().endswith(".pdf"):
            documents = load_pdf(tmp_path)
        else:
            documents = load_markdown(tmp_path)

        if not documents:
            raise HTTPException(
                status_code = 422,
                detail = "Could not extract text from this PDF",
            )
        
        chunks = chunker.chunk_documents(documents)
        logger.info(f"Created {len(chunks)} chunks from {file.filename}")

        texts = [chunk.text for chunk in chunks]
        embeddings = embedder.embed_batch(texts)

        store.store_chunks(chunks,embeddings)

        logger.info(f"Successfully ingested {file.filename}: {len(chunks)} chunks stored")

        return {
            "status": "sucess",
            "filename": file.filename,
            "pages": len(documents),
            "chunks_stored": len(chunks),
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingestion failed for {file.filename} : {e}")
        raise HTTPException(
            status_code = 500,
            detail = f"Ingestion failed: {str(e)}"
        )
    
    # clean up the temp file
    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
            logger.info(f"Cleaned up temp file: {tmp_path}")