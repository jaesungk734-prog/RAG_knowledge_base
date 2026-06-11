from pathlib import Path
from dataclasses import dataclass, field
import pypdf
from loguru import logger

@dataclass
#represents a document before chunking. 
class Document:
    content: str
    source_path: str
    page_number: int
    doc_type: str = "pdf"
    metadata: dict = field(default_factory=dict)

# loads a PDf and return one Document object per page
def load_pdf(file_path: str | Path) -> list[Document]:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix}")
    
    documents = []

    #rb is read binary
    with open(path,"rb") as f:
        reader = pypdf.PdfReader(f)

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
        
            #in case the pdf pages are images
            if not text or not text.strip():
                logger.warning(f"Page {page_num+1} of {path.name} has no extractable text")
                continue
            
            text = "".join(text.split())

            documents.append(Document(
                content = text,
                source_path = str(path.absolute()),
                page_number = page_num+1,
                doc_type = "pdf",
                metadata = {
                    "filename": path.name,
                    "total_pages": len(reader.pages)
                }
            ))
    logger.info(f"Loaded {len(documents)} pages from {path.name}")
    return documents