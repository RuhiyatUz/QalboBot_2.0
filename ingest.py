# ingest.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://172.16.213.1:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

def create_vector_db():
    # Название Вашего файла с знаниями
    pdf_file = "knowledge_base.pdf"
    
    if not os.path.exists(pdf_file):
        print(f"❌ ОШИБКА: Файл {pdf_file} не найден в папке.")
        return

    print(f"📖 Читаю {pdf_file}...")
    loader = PyPDFLoader(pdf_file)
    data = loader.load()

    # Разбиваем текст на куски, чтобы GPT было легче их 'переварить'
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)

    print(f"🧩 Текст разбит на {len(chunks)} фрагментов. Создаю векторы...")
    
    # Локальные эмбеддинги через Ollama (автономно, без внешнего API)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    
    # Создаем базу знаний и сохраняем её локально
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local("faiss_index")
    
    print("✅ ГОТОВО! Папка 'faiss_index' создана. Теперь бот сможет использовать эти знания.")

if __name__ == "__main__":
    create_vector_db()