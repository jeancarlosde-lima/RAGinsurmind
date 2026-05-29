import os
import glob
import time
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "insurmind_agro"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def get_embeddings():
    print(f"Carregando modelo de embeddings local: {EMBEDDING_MODEL}...")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

def scan_pdfs(root_dir: str = ".") -> list[str]:
    pdf_files = glob.glob(os.path.join(root_dir, "*.pdf"))
    pdf_files = [os.path.abspath(p) for p in pdf_files]
    return sorted(pdf_files)

def load_and_split_pdfs(pdf_paths: list[str]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    all_docs = []
    total = len(pdf_paths)
    print(f"Iniciando processamento de {total} PDFs...")
    for idx, pdf_path in enumerate(pdf_paths):
        nome_arquivo = Path(pdf_path).name
        print(f"PDF - Processando ({idx + 1}/{total}): {nome_arquivo}")
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            chunks = splitter.split_documents(pages)
            all_docs.extend(chunks)
        except Exception as e:
            print(f"[AVISO] Erro ao processar '{nome_arquivo}': {e}")
    print("PDFs processados com sucesso!")
    return all_docs

def build_vectorstore(docs, embeddings):
    total = len(docs)
    print(f"Indexando {total} chunks localmente com all-MiniLM-L6-v2...")
    start_time = time.time()
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    elapsed = time.time() - start_time
    print(f"Indexados {total} chunks com sucesso no ChromaDB local em {elapsed:.2f}s!")
    return vectorstore

def main():
    embeddings = get_embeddings()
    pdf_paths = scan_pdfs(".")
    if not pdf_paths:
        print("[ERRO] Nenhum PDF encontrado!")
        return
    docs = load_and_split_pdfs(pdf_paths)
    if not docs:
        print("[ERRO] Nenhum conteudo extraido!")
        return
    build_vectorstore(docs, embeddings)

if __name__ == "__main__":
    main()
