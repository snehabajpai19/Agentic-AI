from django.shortcuts import render
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from .vector import retriever

def ask_ai(request):
    llm = OllamaLLM(model="llama3.2")
    template = (
    "You are an expert in answering questions.\n\n"
    "Here are some relevant reviews: {reviews}\n\n"
    "Here is the question to answer: {question}"
    )
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm
    answer = ""
    if request.method == "POST":
        question = request.POST.get("question")
        reviews = retriever.invoke(question)
        reviews_text = "\n\n".join([
            f"Title: {doc.metadata.get('title', '')}\nRating: {doc.metadata.get('rating', '')}\nDate: {doc.metadata.get('date', '')}\nReview: {doc.page_content}"
            for doc in reviews
        ])
        result = chain.invoke({"reviews": reviews_text, "question": question})
        answer = getattr(result, "content", str(result))
    return render(request, "chatbot.html", {"answer": answer})




