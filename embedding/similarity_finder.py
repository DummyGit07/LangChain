from langchain_ollama import OllamaEmbeddings
import numpy as np

llm = OllamaEmbeddings(model="llama3.2")

query1 = input("Enter1: ")
query2 = input("Enter2: ")

response1 = llm.embed_query(query1)
response2 = llm.embed_query(query2)

# range(0,1) higher the value higher the similarity(semantic similarity)
similarity = np.dot(response1, response2)
print(similarity)
