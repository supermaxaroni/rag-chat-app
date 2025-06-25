import os
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from functools import partial

# Use partial to set encoding for TextLoader
CustomTextLoader = partial(TextLoader, encoding="utf-8")
loader = DirectoryLoader("documentsmax", glob="**/*.txt", loader_cls=CustomTextLoader)
docs = loader.load()
print(f"Loaded {len(docs)} documents from 'documentsmax'.")

if not docs:
    raise ValueError("No documents found in 'documentsmax'. Please add .txt files to this folder.")

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"Split into {len(chunks)} chunks.")

if not chunks:
    raise ValueError("No text chunks created. Check that your .txt files in 'documentsmax' are not empty.")

# 3. Embed and store in vector DB
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = FAISS.from_documents(chunks, embedding)

# 4. Set up the local LLaMA model
llm = Ollama(model="llama3")

# 5. Create retrieval-based QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True
)

# 6. Ask a question
query = input("🧠 Ask your question: ")
result = qa.invoke(query)

print("\n💬 Answer:")
print(result['result'])

print("\n📄 Sources:")
for doc in result['source_documents']:
    print("-", doc.metadata['source'])
