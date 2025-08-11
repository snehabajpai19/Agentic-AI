#1.Importing necessary libraries and modules
from langchain_ollama import OllamaEmbeddings  
from langchain_chroma import Chroma  
from langchain_core.documents import Document
import os
import pandas as pd  # This imports the pandas library for reading the CSV file
#2. Reading the CSV file and initializing the Ollama embeddings
df=pd.read_csv("D:\\Local AI Agent\\My_AI_Agent\\realistic_restaurant_reviews.csv")  
embeddings= OllamaEmbeddings(model="mxbai-embed-large")  # This initializes the Ollama embeddings with the specified model, "mxbai-embed-large".
#3. Setting up the Chroma vector store
db_location= "./chroma_langchain_db"  # This sets the location for the Chroma database.
add_documents= not os.path.exists(db_location)  # This checks if the Chroma database already exists.If it does not exist, it will add documents to the database.If it exists, it will skip this step.
if add_documents:
    # This block will only execute if the Chroma database does not already exist.
    documents=[]
    ids=[]
    for i, row in df.iterrows():  # This iterates over each row in the DataFrame.
        documents.append(
            Document(
                page_content=row["Title"]+" "+row["Review"], # This combines the "Title" and "Review" columns to create the content of the document.
                metadata={"rating": row["Rating"], "date":row["Date"]},# This creates metadata for the document, including the rating and date from the DataFrame.
                id=str(i)
                ))  
        ids.append(str(i))  
        # This appends the index of the row as a string to the ids list.

vector_store= Chroma( # This initializes the Chroma vector store with the specified parameters.
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings,

)

if add_documents:  # This checks if documents need to be added to the vector store.
    vector_store.add_documents(documents=documents, ids=ids)  # This adds the documents to the vector store with their corresponding IDs.

#Now we need to make vector_store be usable by our Ollama LLM in main.py
retriever= vector_store.as_retriever(
     # This converts the vector store into a retriever that can be used to retrieve relevant documents based on queries.
     search_kwargs={"k": 5}
     )# This sets the number of documents to retrieve for each query to 5-> Five reviews .
# The retriever will return the top 5 documents that are most relevant to the query based on the embeddings and pass them to the LLM for generating responses.
#Now use retriever in main.py
