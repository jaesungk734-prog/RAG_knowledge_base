from pathlib import Path
from loguru import logger
from .pdf_loader import Document

def _extract_title(content: str, fallback: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswtith("# "):
            return stripped[2:].strip()
        return fallback
    

def load_markdown(file_path: str | Path) -> list[Document]:
    path = Path(file_path)

    if not path.exists():
        return FileNotFoundError(f"Markdown file not found: {path}")
    if path.suffix.lower() not in (".md", ".markdown"):
        return ValueError(f"Expected a .md file, got: {path.suffix}")
    
    content = path.read_text(encoding = "utf-8")
    if not content.strip():
        logger.warning(f"{path.name} is empty - skipping")
        return []
    
    title = _extract_title(content, fallback = path.stem)

    document = Document(
        content = content,
        source_path = str(path.absolute()),
        page_number = 1,
        doc_type= "markdown",
        metadata= {
            "filename": path.name,
            "title": title
        }
    )

    logger.info(f"Loaded {path.name} ({len(content)} chars, title: '{title}')")
    return [document]

def load_markdown_directory(dir_path: str | Path) ->list[Document]:
    dir_path = Path(dir_path)

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    if not dir_path.is_dir():
        raise ValueError(f"Not a directory: {dir_path}")
    
    all_documents = []
    md_files = sorted(dir_path.glob("*.md"))

    if not md_files:
        logger.warning(f"No .md files found in {dir_path}")
        return []

    for md_file in md_files:
        try:
            doc = load_markdown(md_file)
            all_documents.extend(doc)
        except Exception as e:
            logger.error(f"Failed to load {md_file.name}: {e}")
            continue
    
    logger.info(f"Loaded {len(all_documents)} markdown documents from {dir_path}")
    return all_documents
