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

    # LLM (ensure: `ollama pull llama3.2` and `ollama serve`)
    llm = OllamaLLM(model="llama3.2")

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

        # Upload & index
        if action == "upload" and request.FILES.get("doc"):
            f = request.FILES["doc"]
            fs = FileSystemStorage(location=uploads_dir)
            saved_name = fs.save(f.name, f)
            abs_path = fs.path(saved_name)

            try:
                collection_name = index_file(abs_path)  # build a new collection
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
                    retriever = get_retriever(collection_name, k=5)
                    docs = retriever.invoke(question)

                    # Build generic context
                    context = "\n\n".join(
                        f"Source: {d.metadata.get('source','')}\n"
                        f"Content: {d.page_content}"
                        for d in docs
                    )

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
