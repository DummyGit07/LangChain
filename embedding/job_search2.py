from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

llm = OllamaEmbeddings(model = 'llama3.2')

document = TextLoader("job_listings.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap=10)
chunks = text_splitter.split_documents(document)

db = Chroma.from_documents(chunks, llm)
retriver = db.as_retriever() # as we have already tell the database which llm we are using

query = input("Enter The query: ")

docs = retriver.invoke(query)
for doc in docs:
    print(doc.page_content)
