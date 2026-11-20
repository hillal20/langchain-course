import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"

load_dotenv()

if __name__ == "__main__":

    # 1. Initialize the Pinecone Client (for admin tasks like create/delete)
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = os.environ["INDEX_NAME"]

    # 2. Check and delete the index if it exists
    if pc.has_index(index_name):
        print(f"===> Index : '{index_name}' found, deleting...")
        pc.delete_index(index_name)

    # 3. Create the index using the Pinecone client
    print(f"====> Creating index : '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=1536,  # OpenAIEmbeddings default (text-embedding-3-small)
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print(f"====> Index: '{index_name}' created successfully!")

    # 4. Initialize the LangChain VectorStore for the newly created index
    index = pc.Index(index_name)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    
    print("+++++++++  PineconeVectorStore is ready to use.+++++++++++++++++++++++++")

    # 5. Load the document
    # Using a raw string (r"...") or forward slashes (/) to fix Windows path issues
    file_path = r"C:/Users/hilal/projects/udemy-courses/langchain-course/5_ingestion_embedding_victors/5_medium_blog_text_1.txt"
    print(f"Loading document from: {file_path}")
    loader = TextLoader(file_path, encoding="UTF-8")
    docs = loader.load()

    # 6. Split the document into smaller chunks
    # This is necessary because embedding models have token limits.
    print("Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,# 1000 characters 
        chunk_overlap=0
    )
    splits = text_splitter.split_documents(docs)
    print(f"Split the document into {len(splits)} chunks.")

    # 7. Add the chunks to Pinecone
    # The vector_store will automatically call the OpenAI embeddings API 
    # for each chunk and save the vectors to Pinecone.
    print("Adding documents to Pinecone (this may take a moment)...")
    vector_store.add_documents(documents=splits)
    
    print("✅ Documents successfully ingested into Pinecone!")