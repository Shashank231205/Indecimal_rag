from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def load_documents(data_dir, allowed_exts):
    docs = []
    for file in Path(data_dir).glob("*"):
        if file.suffix.replace(".", "") not in allowed_exts:
            continue
        text = file.read_text(encoding="utf-8").strip()
        if not text:
            logger.warning(f"Empty file skipped: {file.name}")
            continue
        docs.append({"source": file.name, "text": text})
        logger.debug(f"Loaded document: {file.name}")
    if not docs:
        raise RuntimeError("No valid documents loaded.")
    return docs
