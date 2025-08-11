from langchain_ollama import OllamaLLM #This imports the Ollama LLM class from LangChain’s Ollama integration module
from langchain_core.prompts import ChatPromptTemplate #This imports the ChatPromptTemplate class from LangChain’s core prompts module
from vector import retriever  # This imports the retriever object from the vector module, which is used to retrieve relevant documents based on queries.

llm = OllamaLLM(model="llama3.2")# This initializes the Ollama LLM with the specified model, in this case, "llama3.2".
# The OllamaLLM class is used to interact with the Ollama model for generating responses.Make sure you have Ollama installed and running.

template = (
    "You are an expert in answering questions.\n\n"
    "Here are some relevant reviews: {reviews}\n\n"
    "Here is the question to answer: {question}"
)# This defines a template for the chat prompt, which includes placeholders for reviews and the question to be answered.
# The template provides context for the LLM to generate a relevant response based on the provided reviews

prompt = ChatPromptTemplate.from_template(template)# This creates a chat prompt template from the defined template string.

chain = prompt | llm  # This creates a chain that combines the chat prompt template with the Ollama LLM.
# The chain will use the prompt to format the input before passing it to the LLM for generating a response.
while True: # This starts an infinite loop to continuously prompt the user for input.
    print("\n\n------------------------------------------------")
    question = input("Enter your question (q to quit): ")
    print("\n\n------------------------------------------------")
    if question.lower() == 'q':
        break
    reviews = retriever.invoke(question)
    # Convert Document objects to readable text
    reviews_text = "\n\n".join([f"Title: {doc.metadata.get('title', '')}\nRating: {doc.metadata.get('rating', '')}\nDate: {doc.metadata.get('date', '')}\nReview: {doc.page_content}" for doc in reviews])
    result = chain.invoke({"reviews": reviews_text, "question": question})
    print(result)  #result is a ChatMessage object so result.content is the text of the message


#Now, retriever goes to vector.py, from where it will use similarty search algorithm and search the vector store for the most relevant reviews based on the user's question . It will pass these reviews to the chain, which will then format the input and pass it to the LLM for generating a response.