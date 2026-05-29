"""
Chatbot RAG — Insurmind Agro (Google Gemini)
=============================================
Stack: Streamlit + LangChain + ChromaDB + Google Gemini 2.5 Flash

v6 — versão limpa e direta:
  - Embedding simples sem prefixo (paraphrase-multilingual-MiniLM-L12-v2)
  - Query RAG via invoke() direto, sem LCEL aninhado
  - LLM com transport=rest + timeout + max_retries corretos
  - st.secrets com try/except robusto
  - Sem classes wrapper desnecessárias
"""

import os
import glob
import time
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
CHROMA_DIR      = "./chroma_db"
COLLECTION_NAME = "insurmind_agro_v6"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GEMINI_MODEL    = "gemini-2.0-flash"
CHUNK_SIZE      = 1000
CHUNK_OVERLAP   = 200
TOP_K_DOCS      = 8

# ---------------------------------------------------------------------------
# Embeddings — simples, sem prefixo
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_embeddings():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# ---------------------------------------------------------------------------
# PDFs
# ---------------------------------------------------------------------------

def scan_pdfs(root_dir: str = ".") -> list[str]:
    return sorted(os.path.abspath(p) for p in glob.glob(os.path.join(root_dir, "*.pdf")))


def load_and_split_pdfs(pdf_paths: list[str]):
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    all_docs = []
    bar = st.progress(0, text="Iniciando processamento...")
    total = len(pdf_paths)

    for idx, path in enumerate(pdf_paths):
        nome = Path(path).name
        bar.progress(idx / total, text=f"📄 ({idx+1}/{total}): {nome}")
        try:
            pages = PyPDFLoader(path).load()
            all_docs.extend(splitter.split_documents(pages))
        except Exception as e:
            st.warning(f"⚠️ Erro em '{nome}': {e}")

    bar.progress(1.0, text="✅ Concluído!")
    time.sleep(0.5)
    bar.empty()
    return all_docs

# ---------------------------------------------------------------------------
# Vector Store
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_vectorstore(_embeddings):
    from langchain_chroma import Chroma
    import shutil

    try:
        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=_embeddings,
            persist_directory=CHROMA_DIR,
        )
    except Exception:
        try:
            shutil.rmtree(CHROMA_DIR)
        except Exception:
            pass
        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=_embeddings,
            persist_directory=CHROMA_DIR,
        )

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def get_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = None
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except (KeyError, FileNotFoundError):
        api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        return None

    try:
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=api_key,
            temperature=0,
            transport="rest",
            timeout=60,
            max_retries=1,
        )
    except Exception as e:
        st.sidebar.warning(f"⚠️ Erro Gemini: {e}")
        return None

# ---------------------------------------------------------------------------
# Query RAG — direto e transparente
# ---------------------------------------------------------------------------

PROMPT = """Você é um assistente de seguros agrícolas da Insurmind.

Os TRECHOS abaixo são extraídos diretamente dos documentos oficiais da apólice.
Sua tarefa é responder a PERGUNTA usando APENAS o conteúdo dos TRECHOS.

INSTRUÇÕES:
- Leia todos os trechos com atenção antes de responder.
- Se a resposta estiver nos trechos, responda de forma clara e direta.
- Só diga "Não encontrei essa informação" se os trechos realmente não contiverem nada relevante.
- NUNCA use conhecimento externo. NUNCA invente.
- Responda em português do Brasil.

TRECHOS DOS DOCUMENTOS:
{context}

PERGUNTA: {question}

RESPOSTA:"""


def query_rag(question: str, vectorstore, llm) -> dict:
    from langchain_core.messages import HumanMessage

    docs = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_DOCS},
    ).invoke(question)

    context = "\n\n---\n\n".join(doc.page_content for doc in docs)
    prompt = PROMPT.format(context=context, question=question)
    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "result": response.content,
        "source_documents": docs,
    }

# ---------------------------------------------------------------------------
# Session State
# ---------------------------------------------------------------------------

def init_session_state():
    for key, val in {
        "messages": [],
        "vectorstore": None,
        "llm": None,
        "llm_available": False,
        "docs_loaded": False,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = val

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_rag(force_reload: bool = False):
    embeddings = get_embeddings()
    vectorstore = get_vectorstore(embeddings)

    if force_reload:
        with st.spinner("🔄 Limpando índice..."):
            try:
                vectorstore.reset_collection()
            except Exception:
                pass

    try:
        count = vectorstore._collection.count()
        needs_index = count == 0 or force_reload
    except Exception:
        needs_index = True
        count = 0

    if not needs_index:
        st.sidebar.success(f"✅ {count} chunks indexados")

    if needs_index:
        pdf_paths = scan_pdfs(".")
        if not pdf_paths:
            st.error("❌ Nenhum PDF encontrado na pasta raiz.")
            st.stop()

        st.sidebar.info(f"📄 {len(pdf_paths)} PDF(s) encontrado(s)")
        docs = load_and_split_pdfs(pdf_paths)

        if not docs:
            st.error("❌ Nenhum conteúdo extraído dos PDFs.")
            st.stop()

        with st.spinner("💾 Indexando no ChromaDB..."):
            BATCH = 100
            lotes = [docs[i:i+BATCH] for i in range(0, len(docs), BATCH)]
            bar = st.progress(0, text="Indexando...")
            for idx, lote in enumerate(lotes):
                bar.progress(idx / len(lotes), text=f"Lote {idx+1}/{len(lotes)}...")
                vectorstore.add_documents(lote)
            bar.progress(1.0, text="✅ Pronto!")
            time.sleep(0.5)
            bar.empty()
            st.success(f"✅ {len(docs)} chunks indexados!")

    st.session_state.vectorstore = vectorstore
    st.session_state.docs_loaded = True

    with st.spinner(f"✨ Conectando ao Gemini..."):
        llm = get_llm()

    if llm is None:
        st.session_state.llm_available = False
        st.sidebar.error(
            "⚠️ **Gemini indisponível**\n\n"
            "Verifique `GOOGLE_API_KEY` no `.env` ou Settings → Secrets"
        )
    else:
        st.session_state.llm = llm
        st.session_state.llm_available = True
        st.sidebar.success(f"✨ {GEMINI_MODEL} conectado!")

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Insurmind RAG — Seguros Agrícolas",
        page_icon="🌾",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_session_state()

    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/wheat.png", width=64)
        st.title("Insurmind Agro")
        st.caption("Chatbot RAG — Seguros Agrícolas")
        st.divider()
        st.subheader("📊 Status do Sistema")

        if not st.session_state.docs_loaded:
            setup_rag(force_reload=False)

        st.divider()
        if st.button("🔄 Recarregar PDFs", use_container_width=True):
            get_embeddings.clear()
            get_vectorstore.clear()
            st.session_state.docs_loaded = False
            st.session_state.messages = []
            setup_rag(force_reload=True)
            st.rerun()

        st.divider()
        st.subheader("📄 Documentos Indexados")
        pdfs = scan_pdfs(".")
        if pdfs:
            for pdf in pdfs:
                st.markdown(f"• {Path(pdf).stem.replace('_', ' ')}")
        else:
            st.caption("Nenhum PDF encontrado.")

        st.divider()
        if st.button("🗑️ Limpar Conversa", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.caption(f"🤖 `{GEMINI_MODEL}`")
        st.caption("Streamlit · LangChain · ChromaDB · Gemini")

    # Chat
    st.title("🌾 Assistente de Seguros Agrícolas")
    st.markdown(
        "Faça perguntas sobre as **Condições Gerais** dos seguros agrícolas indexados. "
        "As respostas são geradas com base exclusivamente nos documentos da sua apólice."
    )

    if not st.session_state.llm_available:
        st.warning("⚠️ **Google Gemini indisponível.** Verifique a `GOOGLE_API_KEY`.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 Fontes utilizadas", expanded=False):
                    for src in msg["sources"]:
                        st.caption(f"📄 **{src['arquivo']}** — Página {src['pagina']}")

    if prompt := st.chat_input(
        "Faça sua pergunta sobre os seguros agrícolas...",
        disabled=not st.session_state.docs_loaded,
    ):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if not st.session_state.llm_available:
                resp = "⚠️ Gemini indisponível. Verifique a chave de API."
                st.markdown(resp)
                st.session_state.messages.append({"role": "assistant", "content": resp})
            else:
                with st.spinner("🤔 Consultando os documentos..."):
                    try:
                        resultado = query_rag(
                            question=prompt,
                            vectorstore=st.session_state.vectorstore,
                            llm=st.session_state.llm,
                        )
                        resposta = resultado["result"]
                        source_docs = resultado["source_documents"]

                        fontes, vistas = [], set()
                        for doc in source_docs:
                            arq = Path(doc.metadata.get("source", "?")).name
                            pag = doc.metadata.get("page", "?")
                            chave = f"{arq}_{pag}"
                            if chave not in vistas:
                                vistas.add(chave)
                                p = (pag + 1) if isinstance(pag, int) else pag
                                fontes.append({"arquivo": arq, "pagina": p})

                        st.markdown(resposta)
                        if fontes:
                            with st.expander("📚 Fontes utilizadas", expanded=False):
                                for src in fontes:
                                    st.caption(f"📄 **{src['arquivo']}** — Página {src['pagina']}")

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": resposta,
                            "sources": fontes,
                        })

                    except Exception as e:
                        erro = f"❌ Erro: {str(e)}"
                        st.error(erro)
                        st.session_state.messages.append({"role": "assistant", "content": erro})


if __name__ == "__main__":
    main()
