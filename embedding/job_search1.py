from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

llm = OllamaEmbeddings(model = 'llama3.2')

document = TextLoader("job_listings.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap=10)
chunks = text_splitter.split_documents(document)

db = Chroma.from_documents(chunks, llm)

query = input("Enter The query: ")
embedding_vector = llm.embed_query(query)

docs = db.similarity_search_by_vector(embedding_vector)
for doc in docs:
    print(doc.page_content)
