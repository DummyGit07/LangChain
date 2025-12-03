# additional library require,       pip install angchain-ollama

from langchain_ollama import OllamaEmbeddings

llm = OllamaEmbeddings(model="llama3.2")

question = input("Enter: ")
response = llm.embed_query(question)
print(response)
