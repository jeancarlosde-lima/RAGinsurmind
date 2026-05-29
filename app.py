"""
Chatbot RAG — Insurmind Agro (Google Gemini)
=============================================
Stack: Streamlit + LangChain + ChromaDB + Google Gemini 2.5 Flash

Correções v4:
  1. gemini-2.5-flash + transport=rest + timeout=90 + max_retries=1
  2. st.secrets com try/except correto
  3. temperature=0 para ancorar respostas nas fontes
  4. [NOVO] Prefixo E5 correto: "passage: " nos docs, "query: " nas perguntas
     → sem isso o ChromaDB não encontra os chunks certos (causa do "não encontrei")
  5. [NOVO] Prompt reforçado com instrução explícita anti-alucinação
  6. [NOVO] EmbeddingsFunctionWithPrefix: aplica prefixo automaticamente no add_documents
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
COLLECTION_NAME = "insurmind_agro_v4"          # v4 — reindexação necessária por mudança de embedding
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
GEMINI_MODEL    = "gemini-2.5-flash"
CHUNK_SIZE      = 1000
CHUNK_OVERLAP   = 200
TOP_K_DOCS      = 8


# ---------------------------------------------------------------------------
# Embeddings com prefixo E5 correto
# ---------------------------------------------------------------------------

class E5Embeddings:
    """
    Wrapper sobre HuggingFaceEmbeddings que aplica automaticamente os prefixos
    obrigatórios do modelo intfloat/multilingual-e5-*:
      - Documentos (indexação): "passage: <texto>"
      - Queries (busca):        "query: <texto>"
    Sem esses prefixos o espaço vetorial fica desalinhado e a recuperação falha.
    """
    def __init__(self):
        from langchain_community.embeddings import HuggingFaceEmbeddings
        self._model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"passage: {t}" for t in texts]
        return self._model.embed_documents(prefixed)

    def embed_query(self, text: str) -> list[float]:
        return self._model.embed_query(f"query: {text}")


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return E5Embeddings()


# ---------------------------------------------------------------------------
# PDFs
# ---------------------------------------------------------------------------

def scan_pdfs(root_dir: str = ".") -> list[str]:
    pdf_files = glob.glob(os.path.join(root_dir, "*.pdf"))
    return sorted(os.path.abspath(p) for p in pdf_files)


def load_and_split_pdfs(pdf_paths: list[str]):
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    all_docs = []
    total = len(pdf_paths)
    progress_bar = st.progress(0, text="Iniciando processamento dos PDFs...")

    for idx, pdf_path in enumerate(pdf_paths):
        nome = Path(pdf_path).name
        progress_bar.progress(
            idx / total,
            text=f"📄 Processando ({idx + 1}/{total}): {nome}",
        )
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            chunks = splitter.split_documents(pages)
            all_docs.extend(chunks)
        except Exception as e:
            st.warning(f"⚠️ Erro ao processar '{nome}': {e}")

    progress_bar.progress(1.0, text="✅ PDFs processados!")
    time.sleep(0.5)
    progress_bar.empty()
    return all_docs


# ---------------------------------------------------------------------------
# Vector Store
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_vectorstore_instance(_embeddings):
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
            if os.path.exists(CHROMA_DIR):
                shutil.rmtree(CHROMA_DIR)
        except Exception:
            pass
        fallback_dir = f"{CHROMA_DIR}_fallback"
        try:
            return Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=_embeddings,
                persist_directory=fallback_dir,
            )
        except Exception:
            return Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=_embeddings,
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
        pass
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None

    try:
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=api_key,
            temperature=0,
            transport="rest",        # CRÍTICO: gRPC é bloqueado no Streamlit Cloud
            timeout=90,
            max_retries=1,
            convert_system_message_to_human=True,
        )
    except Exception as e:
        st.sidebar.warning(f"⚠️ Erro ao conectar ao Gemini: {e}")
        return None


# ---------------------------------------------------------------------------
# Pipeline RAG
# ---------------------------------------------------------------------------

def build_rag_chain(vectorstore, llm):
    from langchain_core.runnables import RunnableParallel
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    # Prompt reforçado: proíbe explicitamente uso de conhecimento externo
    prompt_template = """Você é um assistente de seguros agrícolas da Insurmind.

REGRAS ABSOLUTAS — leia antes de responder:
1. USE EXCLUSIVAMENTE o texto dos TRECHOS DO DOCUMENTO abaixo.
2. NUNCA use conhecimento próprio, definições gerais ou fontes externas.
3. Se a resposta não estiver nos trechos, responda EXATAMENTE: "Não encontrei essa informação nos documentos disponíveis."
4. Não complemente, não infira, não expanda além do que está escrito.
5. Responda em português do Brasil, de forma clara e direta.

TRECHOS DO DOCUMENTO:
{context}

PERGUNTA DO USUÁRIO: {question}

RESPOSTA (baseada SOMENTE nos trechos acima):"""

    prompt = ChatPromptTemplate.from_template(prompt_template)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_DOCS},
    )

    def format_docs(docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    # O E5Embeddings.embed_query já adiciona "query: " automaticamente,
    # então NÃO precisamos mais do prepare_query manual aqui.
    setup = RunnableParallel(
        {
            "context": (lambda x: x["query"]) | retriever | format_docs,
            "question": (lambda x: x["query"]),
            "source_documents": (lambda x: x["query"]) | retriever,
        }
    )

    chain = setup | RunnableParallel(
        {
            "result": prompt | llm | StrOutputParser(),
            "source_documents": lambda x: x["source_documents"],
        }
    )

    return chain


# ---------------------------------------------------------------------------
# Estado da sessão
# ---------------------------------------------------------------------------

def init_session_state():
    defaults = {
        "messages": [],
        "vectorstore": None,
        "rag_chain": None,
        "llm_available": False,
        "docs_loaded": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ---------------------------------------------------------------------------
# Setup RAG
# ---------------------------------------------------------------------------

def setup_rag(force_reload: bool = False):
    embeddings = get_embeddings()
    vectorstore = get_vectorstore_instance(embeddings)

    if force_reload:
        with st.spinner("🔄 Limpando índice e reindexando PDFs..."):
            try:
                vectorstore.reset_collection()
            except Exception:
                pass
            st.session_state.vectorstore = None

    try:
        count = vectorstore._collection.count()
        if count == 0 and not force_reload:
            force_reload = True
    except Exception:
        force_reload = True
        count = 0

    if not force_reload:
        st.sidebar.success(f"✅ ChromaDB conectado ({count} chunks)")

    if force_reload:
        pdf_paths = scan_pdfs(".")
        if not pdf_paths:
            st.error("❌ Nenhum PDF encontrado. Adicione arquivos `.pdf` na pasta raiz.")
            st.stop()

        st.sidebar.info(f"📄 {len(pdf_paths)} PDF(s) encontrado(s)")
        docs = load_and_split_pdfs(pdf_paths)

        if not docs:
            st.error("❌ Nenhum conteúdo extraído dos PDFs.")
            st.stop()

        with st.spinner("💾 Indexando no ChromaDB..."):
            BATCH_SIZE = 100
            lotes = [docs[i:i + BATCH_SIZE] for i in range(0, len(docs), BATCH_SIZE)]
            barra = st.progress(0, text="Iniciando indexação...")
            for idx, lote in enumerate(lotes):
                barra.progress(
                    idx / len(lotes),
                    text=f"Lote {idx + 1}/{len(lotes)} — {len(lote)} chunks...",
                )
                vectorstore.add_documents(lote)
            barra.progress(1.0, text="✅ Indexação concluída!")
            time.sleep(0.5)
            barra.empty()
            st.success(f"✅ {len(docs)} chunks indexados com sucesso!")

    st.session_state.vectorstore = vectorstore
    st.session_state.docs_loaded = True

    with st.spinner(f"✨ Conectando ao Gemini ({GEMINI_MODEL})..."):
        llm = get_llm()

    if llm is None:
        st.session_state.llm_available = False
        st.sidebar.error(
            "⚠️ **Google Gemini indisponível!**\n\n"
            "Verifique:\n"
            "• `GOOGLE_API_KEY` no `.env` (local) ou **Settings → Secrets** (Cloud)\n"
            "• Cota disponível em aistudio.google.com"
        )
    else:
        st.session_state.llm_available = True
        st.session_state.rag_chain = build_rag_chain(vectorstore, llm)
        st.sidebar.success(f"✨ Gemini ({GEMINI_MODEL}) conectado!")


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

    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/wheat.png", width=64)
        st.title("Insurmind Agro")
        st.caption("Chatbot RAG — Seguros Agrícolas")
        st.divider()
        st.subheader("📊 Status do Sistema")

        if not st.session_state.docs_loaded:
            setup_rag(force_reload=False)

        st.divider()
        if st.button("🔄 Recarregar PDFs", use_container_width=True,
                     help="Use se novos PDFs foram adicionados."):
            get_embeddings.clear()
            get_vectorstore_instance.clear()
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
        st.caption(f"🤖 Modelo: `{GEMINI_MODEL}`")
        st.caption("Stack: Streamlit · LangChain · ChromaDB · Gemini")

    # Chat
    st.title("🌾 Assistente de Seguros Agrícolas")
    st.markdown(
        "Faça perguntas sobre as **Condições Gerais** dos seguros agrícolas indexados. "
        "As respostas são geradas com base exclusivamente nos documentos da sua apólice."
    )

    if not st.session_state.llm_available:
        st.warning(
            "⚠️ **Google Gemini indisponível.** Verifique `GOOGLE_API_KEY` "
            "no `.env` (local) ou em **Settings → Secrets** (Streamlit Cloud)."
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("📚 Fontes utilizadas", expanded=False):
                    for src in message["sources"]:
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
                resposta = "⚠️ Google Gemini indisponível. Verifique a chave de API."
                st.markdown(resposta)
                st.session_state.messages.append({"role": "assistant", "content": resposta})
            else:
                with st.spinner("🤔 Consultando os documentos..."):
                    try:
                        resultado = st.session_state.rag_chain.invoke({"query": prompt})
                        resposta = resultado.get("result", "Não foi possível gerar uma resposta.")
                        source_docs = resultado.get("source_documents", [])

                        fontes = []
                        fontes_vistas = set()
                        for doc in source_docs:
                            meta = doc.metadata
                            arquivo = Path(meta.get("source", "Desconhecido")).name
                            pagina = meta.get("page", "?")
                            chave = f"{arquivo}_{pagina}"
                            if chave not in fontes_vistas:
                                fontes_vistas.add(chave)
                                p = (pagina + 1) if isinstance(pagina, int) else pagina
                                fontes.append({"arquivo": arquivo, "pagina": p})

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
                        erro_msg = f"❌ Erro ao processar sua pergunta: {str(e)}"
                        st.error(erro_msg)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": erro_msg}
                        )


if __name__ == "__main__":
    main()
