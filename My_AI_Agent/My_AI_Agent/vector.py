import os
import re
import uuid
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# OCR deps
try:
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image
except Exception:
    pytesseract = None  
POPPLER_BIN = r"C:\poppler-24.02.0\Library\bin"  

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    UnstructuredFileLoader,  
)

DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_langchain_db")
EMB = OllamaEmbeddings(model="mxbai-embed-large")


def _safe_collection_name(file_path: str, suffix_len: int = 6) -> str:
    base = os.path.splitext(os.path.basename(file_path))[0].strip()
    base = re.sub(r"[^a-zA-Z0-9._-]+", "_", base)
    base = re.sub(r"_+", "_", base).strip("._-")
    if not base:
        base = "collection"
    name = f"{base}-{uuid.uuid4().hex[:suffix_len]}"
    name = name[:512]
    if not name[0].isalnum():
        name = "c" + name
    if not name[-1].isalnum():
        name = name + "c"
    return name


def _pick_loader(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".txt", ".csv", ".md", ".markdown"):
        return TextLoader(path, encoding="utf-8", autodetect_encoding=True)
    if ext == ".pdf":
        return PyPDFLoader(path)
    if ext == ".docx":
        return UnstructuredFileLoader(path)
    if ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"):
        return None
    return TextLoader(path, encoding="utf-8", autodetect_encoding=True)


def _ocr_pdf_to_text(path: str) -> str:
    if pytesseract is None:
        raise RuntimeError("OCR requires pytesseract/pdf2image/pillow and Tesseract installed.")
    pages = convert_from_path(path, dpi=200, poppler_path=POPPLER_BIN)
    return "\n\n".join(pytesseract.image_to_string(img) for img in pages).strip()


def _ocr_image_to_text(path: str) -> str:
    if pytesseract is None:
        raise RuntimeError("OCR requires pytesseract/pillow and Tesseract installed.")
    img = Image.open(path).convert("L")              # grayscale
    w, h = img.size
    img = img.resize((w*2, h*2))                     # upscale
    # simple binarization
    img = img.point(lambda x: 255 if x > 180 else 0) # threshold
    return pytesseract.image_to_string(
        img, config="--oem 3 --psm 6"
    ).strip()



def _doc(text: str, source: str, title: Optional[str] = None) -> Document:
    return Document(page_content=text, metadata={"source": source, "title": title or source})


def _load_generic(path: str) -> List[Document]:
    """Load any supported file; if no text, OCR images/scanned PDFs."""
    loader = _pick_loader(path)
    docs: List[Document] = []

    if loader:
        docs = loader.load()
        for d in docs:
            d.metadata.setdefault("source", os.path.basename(path))
            d.metadata.setdefault("title", d.metadata.get("source"))

    non_empty = [d for d in docs if (d.page_content or "").strip()]
    if non_empty:
        return non_empty

    # OCR fallback
    ext = os.path.splitext(path)[1].lower()
    text = ""
    if ext == ".pdf":
        text = _ocr_pdf_to_text(path)
    elif ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"):
        text = _ocr_image_to_text(path)

    if text.strip():
        return [_doc(text, source=f"OCR:{os.path.basename(path)}")]

    return []


def _chunk(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    return splitter.split_documents(docs)


def index_file(file_path: str, collection_name: str | None = None) -> str:
    docs = _load_generic(file_path)
    docs = [d for d in docs if (d.page_content or "").strip()]
    if not docs:
        raise ValueError(
            "No readable text found in the uploaded file. "
            "If it's a scanned PDF/image-only document, ensure OCR is installed and configured."
        )

    chunks = _chunk(docs)
    chunks = [c for c in chunks if (c.page_content or "").strip()]
    if not chunks:
        raise ValueError("File loaded, but produced no chunks. Try a different file or adjust chunking.")

    collection = collection_name or _safe_collection_name(file_path)

    Chroma.from_documents(
        documents=chunks,
        embedding=EMB,
        collection_name=collection,
        persist_directory=DB_DIR,
    )
    return collection


def get_retriever(collection_name: str, k: int = 5):
    vs = Chroma(
        collection_name=collection_name,
        persist_directory=DB_DIR,
        embedding_function=EMB,
    )
    return vs.as_retriever(search_kwargs={"k": k})
