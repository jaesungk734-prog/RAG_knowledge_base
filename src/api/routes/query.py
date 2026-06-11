import json
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator
from loguru import logger
from typing import AsyncIterator

router = APIRouter()

class QueryRequest(BaseModel):
    #pydantic model auto parses JSON from API

    question: str
    top_k: int = 5

    @field_validator("question")
    @classmethod
    def question_must_not_be_empty(cls,v:str)->str:
        if not v or not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()


async def token_stream(question:str, retriever,generator) ->AsyncIterator[str]:
    # retrieves chunks from Qdrant and SSE streams GPT tokens
    context, chunks = retriever.retrieve_and_format(question)

    if not chunks:
        yield f"data: I couldn't find relevant information about that in your knowledge base. \n\n"
        yield "data: [DONE]\n\n"
        return 

    #stream
    for token in generator.generate(question,context):
        yield f"data: {token}\n\n"
    
    import os
    sources = [
        {
            "filename": os.path.basename(chunk.get("source_path", "unknown")),
            "page_number": chunk.get("page_number"),
            "score": chunk.get("score"),
        }
        for chunk in chunks
    ]

    yield f"data: [SOURCES]{json.dumps(sources)}"
    yield f"data: [DONE]\n\n"


@router.post("/query")
async def query_knowledge_base(request: Request, body: QueryRequest):
    #get a question an dretrieve context from Qdrant, stream answer from it
    
    logger.info(f"Query receieved: '{body.question}'")

    retriever = request.app.state.retriever
    generator = request.app.state.generator

    retriever.top_k = body.top_k

    try:
        return StreamingResponse(
            token_stream(body.question,retriever,generator),
            media_type="text/event-stream",
            headers = {
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                }
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(
            status_code = 500,
            detail = str(e)
        )

