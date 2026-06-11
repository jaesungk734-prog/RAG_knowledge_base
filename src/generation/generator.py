import os
from openai import OpenAI
from dotenv import load_dotenv
from loguru import logger
from typing import Iterator 

load_dotenv()


SYSTEM_PROMPT = """You are a helpful personal knowledge base assistant.

Your job is to answer questions using ONLY the context provided below.
The context comes from the user's own documents — PDFs, notes, and bookmarks.

Rules you must follow:
1. Only use information from the provided context to answer.
2. When you use information, cite the source file and page number like this:
   (source: filename.pdf, page 3)
3. If the context doesn't contain enough information to answer the question,
   say: "I couldn't find relevant information about that in your knowledge base."
4. Do not make up information or use knowledge outside the provided context.
5. Be concise and direct. Prefer bullet points for lists of facts.
6. If multiple sources support an answer, cite all of them.
"""

class Generator:
    #uses streaming
    
    def __init__(self,model : str = "gpt-4o"):
        self.model = model

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. "
                "Make sure it's set in your .env file."
            )
        
        self.client = OpenAI(api_key = api_key)
        logger.info(f"Generator initialized with model: {self.model}")

    
    def _build_user_prompt(self,question: str, context: str) -> str:
        #assemble prompt here
        return f"""Here is the relevant context from your knowledge base:

{context}

---

Question: {question}

Answer using only the context above. Cite sources where relevant."""
    

    def generate(self, question: str, context: str) -> Iterator[str]:
        #generate an answer with the given context

        user_prompt = self._build_user_prompt(question,context)

        logger.info(f"Generating answer for: '{question}'")

        stream = self.client.chat.completions.create(
            model = self.model,
            messages =  [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            stream = True,
            temperature = 0.2,
            max_tokens = 1000
        )

        for chunk in stream: 
            token = chunk.choices[0].delta.content
            if token is not None:
                yield token
    

    def generate_full(self,question:str, context: str) ->str:
        #puts everything together
        return "".join(self.generate(question,context))
    