from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

loader = TextLoader("data/faq_knowledge_base.txt")
raw_docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
)
chunks = splitter.split_documents(raw_docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

if __name__ == "__main__":
    results = retriever.invoke("can I cancel my order after it ships?")
    for i, doc in enumerate(results):
        print(f"--- chunk {i} ---")
        print(doc.page_content)
        print()     