import os
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from .vector import index_file, get_retriever 
def ask_ai(request):
    # Ensure upload dir inside MEDIA_ROOT
    media_root = getattr(settings, "MEDIA_ROOT", os.path.join(os.path.dirname(__file__), "media"))
    uploads_dir = os.path.join(media_root, "uploads")
    os.makedirs(uploads_dir, exist_ok=True)

    # Configure fast/normal based on request
    fast_flag = False
    if request.method == "POST":
        # allow 'fast' in either form
        fast_flag = (request.POST.get("fast") or "0") in {"1", "true", "True"}

    # LLM (ensure: `ollama serve` and model pulled)
    model_name = getattr(settings, "OLLAMA_FAST_MODEL", getattr(settings, "OLLAMA_MODEL", "llama3.2")) if fast_flag else getattr(settings, "OLLAMA_MODEL", "llama3.2")
    num_predict = int(getattr(settings, "FAST_NUM_PREDICT", 160)) if fast_flag else int(getattr(settings, "OLLAMA_NUM_PREDICT", 256))
    num_ctx = int(getattr(settings, "FAST_NUM_CTX", 1024)) if fast_flag else int(getattr(settings, "OLLAMA_NUM_CTX", 2048))
    llm = OllamaLLM(
        model=model_name,
        temperature=getattr(settings, "OLLAMA_TEMPERATURE", 0.2),
        num_predict=num_predict,
        num_ctx=num_ctx,
        base_url=getattr(settings, "OLLAMA_HOST", None) or None,
    )

    # Generic prompt (no schema assumptions)
    template = (
        "You are an assistant that answers using only the provided context.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "If the answer is not in the context, say you don't know from the provided documents."
    )
    chain = ChatPromptTemplate.from_template(template) | llm

    collection_name = request.session.get("collection_name")
    answer, sources, error = "", [], ""

    if request.method == "POST":
        action = request.POST.get("action")
        # Fallbacks so Enter key submits work even if the submit button isn't included
        if not action:
            if request.FILES.get("doc"):
                action = "upload"
            elif (request.POST.get("question") or "").strip():
                action = "ask"

        # Upload & index
        if action == "upload" and request.FILES.get("doc"):
            f = request.FILES["doc"]
            fs = FileSystemStorage(location=uploads_dir)
            saved_name = fs.save(f.name, f)
            abs_path = fs.path(saved_name)

            try:
                collection_name = index_file(abs_path, fast=fast_flag)  # build a new collection
                request.session["collection_name"] = collection_name
            except Exception as e:
                error = f"Could not index file: {e}"

        # Ask
        elif action == "ask":
            question = (request.POST.get("question") or "").strip()
            if not collection_name:
                error = "Please upload a document first."
            elif question:
                try:
                    rk = int(getattr(settings, "FAST_RETRIEVAL_K", 2)) if fast_flag else int(getattr(settings, "RETRIEVAL_K", 3))
                    retriever = get_retriever(collection_name, k=rk)
                    docs = retriever.invoke(question)

                    # Build generic context (trim for speed)
                    max_total = int(getattr(settings, "FAST_MAX_CONTEXT_CHARS", 2500)) if fast_flag else int(getattr(settings, "MAX_CONTEXT_CHARS", 4000))
                    per_doc = int(getattr(settings, "FAST_MAX_CHARS_PER_DOC", 800)) if fast_flag else int(getattr(settings, "MAX_CHARS_PER_DOC", 1200))
                    parts = []
                    total = 0
                    for d in docs:
                        src = d.metadata.get('source', '')
                        content = (d.page_content or '')[:per_doc]
                        chunk = f"Source: {src}\nContent: {content}"
                        if total + len(chunk) > max_total:
                            # take only what fits
                            remain = max_total - total
                            if remain > 0:
                                parts.append(chunk[:remain])
                                total += remain
                            break
                        parts.append(chunk)
                        total += len(chunk)
                    context = "\n\n".join(parts)

                    result = chain.invoke({"context": context, "question": question})
                    answer = getattr(result, "content", str(result))
                    sources = sorted({d.metadata.get("source", "source") for d in docs})
                except Exception as e:
                    error = f"Query failed: {e}"

    return render(request, "chatbot.html", {
        "answer": answer,
        "sources": sources,
        "current_collection": request.session.get("collection_name"),
        "error": error,
    })
