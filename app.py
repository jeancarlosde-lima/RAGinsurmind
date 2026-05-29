"""
Chatbot RAG — Insurmind Agro (Google Gemini)
=============================================
Stack: Streamlit + LangChain + ChromaDB + Google Gemini 2.5 Flash

Correções aplicadas (v3):
  1. Modelo corrigido para "gemini-2.5-flash" (gemini-3.1-pro não existe)
  2. transport="rest" adicionado — resolve retry loop no Streamlit Cloud (gRPC bloqueado)
  3. st.secrets com try/except correto — st.secrets.get() lança KeyError, não retorna None
  4. temperature=0 para RAG — ancora respostas nas fontes, elimina alucinações
  5. max_retries=1 — evita loop infinito de tentativas
  6. timeout=90 — suficiente para Gemini 2.5 Flash (thinking model tem latência maior)
"""

import os
import glob
import time
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Constantes de configuração
# ---------------------------------------------------------------------------
CHROMA_DIR       = "./chroma_db"
COLLECTION_NAME  = "insurmind_agro_v2"
EMBEDDING_MODEL  = "intfloat/multilingual-e5-small"
GEMINI_MODEL     = "gemini-2.5-flash"          # ← modelo estável e disponível via Gemini API
CHUNK_SIZE       = 1000
CHUNK_OVERLAP    = 200
TOP_K_DOCS       = 8


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_embeddings():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


# ---------------------------------------------------------------------------
# Utilitários de PDF
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
# Vector Store (Singleton via cache)
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
        # Fallback 1: limpar e recriar
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
            # Fallback 2: in-memory
            return Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=_embeddings,
            )


# ---------------------------------------------------------------------------
# LLM — Google Gemini
# ---------------------------------------------------------------------------

def get_llm():
    """
    Inicializa ChatGoogleGenerativeAI com configurações corretas para
    Streamlit Cloud:
      - transport="rest"  → força HTTP/1.1 (gRPC é bloqueado pelo ambiente)
      - timeout=90        → Gemini 2.5 Flash tem thinking, latência maior
      - max_retries=1     → evita loop infinito de retries que trava o app
      - temperature=0     → respostas ancoradas no contexto RAG
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    # Lê API key com tratamento correto — st.secrets.get() pode lançar KeyError
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
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=api_key,
            temperature=0,                       # 0 = ancora nas fontes, sem alucinação
            transport="rest",                    # CRÍTICO: resolve retry loop no Cloud
            timeout=90,                          # 90s para thinking model
            max_retries=1,                       # 1 retry máximo — falha rápida e clara
            convert_system_message_to_human=True,
        )
        return llm
    except Exception as e:
        st.sidebar.warning(f"⚠️ Erro ao conectar ao Gemini: {e}")
        return None


# ---------------------------------------------------------------------------
# Pipeline RAG (LCEL)
# ---------------------------------------------------------------------------

def build_rag_chain(vectorstore, llm):
    from langchain_core.runnables import RunnablePassthrough, RunnableParallel
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    prompt_template = """Você é um assistente especialista em seguros agrícolas da Insurmind.
Use APENAS as informações do contexto abaixo para responder à pergunta do usuário.
Se a resposta não estiver no contexto, diga exatamente: "Não encontrei essa informação nos documentos disponíveis."
Responda sempre em português do Brasil, de forma clara e objetiva.
Não invente informações. Não use conhecimento externo aos documentos.

Contexto:
{context}

Pergunta: {question}

Resposta:"""

    prompt = ChatPromptTemplate.from_template(prompt_template)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_DOCS},
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def prepare_query(q):
        # Prefixo obrigatório para embeddings E5
        if "e5" in EMBEDDING_MODEL:
            return f"query: {q}"
        return q

    setup_and_retrieval = RunnableParallel(
        {
            "context": (lambda x: prepare_query(x["query"])) | retriever | format_docs,
            "question": (lambda x: x["query"]),
            "source_documents": (lambda x: prepare_query(x["query"])) | retriever,
        }
    )

    chain = setup_and_retrieval | RunnableParallel(
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
        "reload_requested": False,
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
        with st.spinner("🔄 Limpando índice e escaneando PDFs..."):
            try:
                vectorstore.reset_collection()
            except Exception:
                pass
            st.session_state.vectorstore = None

    # Verifica se a coleção está vazia
    try:
        count = vectorstore._collection.count()
        if count == 0 and not force_reload:
            force_reload = True
    except Exception:
        force_reload = True

    if not force_reload:
        st.sidebar.success(f"✅ ChromaDB conectado ({count} chunks)")

    # Processa PDFs se necessário
    if force_reload:
        pdf_paths = scan_pdfs(".")

        if not pdf_paths:
            st.error(
                "❌ Nenhum arquivo PDF encontrado na pasta raiz.\n\n"
                "Adicione arquivos `.pdf` e reinicie a aplicação."
            )
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
            barra.progress(1.0, text="Indexação concluída!")
            time.sleep(0.5)
            barra.empty()
            st.success(f"✅ {len(docs)} chunks indexados!")

    st.session_state.vectorstore = vectorstore
    st.session_state.docs_loaded = True

    # Inicializa LLM
    with st.spinner(f"✨ Conectando ao Gemini ({GEMINI_MODEL})..."):
        llm = get_llm()

    if llm is None:
        st.session_state.llm_available = False
        st.sidebar.error(
            "⚠️ **Google Gemini indisponível!**\n\n"
            "Verifique:\n"
            "• `GOOGLE_API_KEY` em `.env` (local) ou Secrets (Streamlit Cloud)\n"
            "• Cota da API no Google AI Studio"
        )
    else:
        st.session_state.llm_available = True
        st.session_state.rag_chain = build_rag_chain(vectorstore, llm)
        st.sidebar.success(f"✨ Gemini ({GEMINI_MODEL}) conectado!")


# ---------------------------------------------------------------------------
# UI Principal
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
        if st.button(
            "🔄 Recarregar PDFs",
            help="Use se novos PDFs foram adicionados.",
            use_container_width=True,
        ):
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

    # Chat principal
    st.title("🌾 Assistente de Seguros Agrícolas")
    st.markdown(
        "Faça perguntas sobre as **Condições Gerais** dos seguros agrícolas indexados. "
        "As respostas são geradas com base exclusivamente nos documentos da sua apólice."
    )

    if not st.session_state.llm_available:
        st.warning(
            "⚠️ **Google Gemini indisponível.** Verifique a variável `GOOGLE_API_KEY` "
            "no `.env` (local) ou em **Settings → Secrets** (Streamlit Cloud)."
        )

    # Histórico
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("📚 Fontes utilizadas", expanded=False):
                    for src in message["sources"]:
                        st.caption(f"📄 **{src['arquivo']}** — Página {src['pagina']}")

    # Input
    if prompt := st.chat_input(
        "Faça sua pergunta sobre os seguros agrícolas...",
        disabled=not st.session_state.docs_loaded,
    ):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if not st.session_state.llm_available:
                resposta = (
                    "⚠️ O Google Gemini não está disponível. "
                    "Verifique a chave de API e a conexão com a internet."
                )
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
                                pagina_display = (pagina + 1) if isinstance(pagina, int) else pagina
                                fontes.append({"arquivo": arquivo, "pagina": pagina_display})

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
