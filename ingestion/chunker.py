import re
import logging
logger = logging.getLogger(__name__)

def chunk_text(text: str):
    """
    Chunk by markdown headers, KEEPING the headers.
    Ideal for small, structured policy documents.
    """
    # Split but keep headers
    parts = re.split(r"(\n#{1,3} .+)", text)

    chunks = []
    current = ""

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if part.startswith("#"):
            # save previous chunk
            if len(current.split()) >= 40:
                chunks.append(current.strip())
            current = part
        else:
            current += "\n" + part

    # add last chunk
    if len(current.split()) >= 40:
        chunks.append(current.strip())

    logger.info(f"Markdown chunks created: {len(chunks)}")
    return chunks
