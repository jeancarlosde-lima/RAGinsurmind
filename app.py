"""
Chatbot RAG — Insurmind Agro (Google Gemini)
=============================================
Autor: Engenheiro de Software / IA
Stack: Streamlit + LangChain + ChromaDB + Google Gemini 1.5 Pro

Descrição:
  - Escaneia automaticamente a pasta raiz em busca de PDFs.
  - Cria/reutiliza um banco vetorial ChromaDB persistente em ./chroma_db.
  - Embeddings e LLM via API Google Gemini (sem downloads locais pesados).
  - Responde perguntas com base nos documentos indexados via pipeline RAG.
  - Acessível na rede local via 0.0.0.0:8501.
"""

import os
import glob
import time
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# Carrega variáveis do arquivo .env (GOOGLE_API_KEY, etc.)
load_dotenv()

# ---------------------------------------------------------------------------
# Constantes de configuração
# ---------------------------------------------------------------------------
CHROMA_DIR = "./chroma_db"                          # Diretório de persistência do ChromaDB
COLLECTION_NAME = "insurmind_agro"                  # Nome da coleção no ChromaDB
EMBEDDING_MODEL = "models/text-embedding-004"        # Modelo de embeddings Google Gemini (estável e disponível)
GEMINI_MODEL = "gemini-3.5-flash"                   # Modelo LLM Google Gemini (generoso no free tier)
CHUNK_SIZE = 1000                                   # Tamanho dos chunks de texto
CHUNK_OVERLAP = 200                                 # Sobreposição entre chunks
TOP_K_DOCS = 4                                      # Número de chunks recuperados por consulta


# ---------------------------------------------------------------------------
# Funções auxiliares de carregamento e indexação
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_embeddings():
    """
    Inicializa e retorna o modelo de embeddings Google Gemini (nuvem).
    Nenhum arquivo de modelo é baixado localmente — tudo via API.
    O cache do Streamlit garante que o objeto seja criado apenas uma vez.
    Lê a API key via st.secrets (Streamlit Cloud) ou os.getenv() (local).
    """
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    # Lê a chave de API: st.secrets tem prioridade (Streamlit Cloud), depois .env local
    api_key = st.secrets.get("GOOGLE_API_KEY", None) or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error(
            "❌ **GOOGLE_API_KEY não encontrada.**\n\n"
            "No Streamlit Cloud: adicione em Settings → Secrets.\n"
            "Localmente: verifique o arquivo `.env` na raiz do projeto."
        )
        st.stop()

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
    )
    return embeddings


def scan_pdfs(root_dir: str = ".") -> list[str]:
    """
    Escaneia recursivamente a pasta raiz em busca de todos os arquivos .pdf.

    Args:
        root_dir: Diretório raiz para escanear (padrão: diretório atual).

    Returns:
        Lista de caminhos absolutos para os PDFs encontrados.
    """
    # Busca arquivos PDF diretamente na raiz (sem subpastas para evitar indexar exemplos)
    pdf_files = glob.glob(os.path.join(root_dir, "*.pdf"))
    pdf_files = [os.path.abspath(p) for p in pdf_files]
    return sorted(pdf_files)


def load_and_split_pdfs(pdf_paths: list[str]):
    """
    Carrega e divide os PDFs em chunks de texto usando LangChain.

    Args:
        pdf_paths: Lista de caminhos para os arquivos PDF.

    Returns:
        Lista de documentos (chunks) processados.
    """
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
        nome_arquivo = Path(pdf_path).name
        progress_bar.progress(
            (idx) / total,
            text=f"📄 Processando ({idx + 1}/{total}): {nome_arquivo}",
        )
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            chunks = splitter.split_documents(pages)
            all_docs.extend(chunks)
        except Exception as e:
            st.warning(f"⚠️ Erro ao processar '{nome_arquivo}': {e}")

    progress_bar.progress(1.0, text="✅ PDFs processados com sucesso!")
    time.sleep(0.5)
    progress_bar.empty()

    return all_docs


def build_vectorstore(docs, embeddings):
    """
    Cria e persiste o banco vetorial ChromaDB a partir dos documentos indexados.

    Processa em lotes de 100 (limite máximo de lote da API do Gemini) para enviar
    poucas requisições e respeitar com segurança o limite de 100 req/min do
    plano gratuito da API do Gemini, mantendo o processo rápido.

    Args:
        docs: Lista de documentos (chunks) para indexar.
        embeddings: Modelo de embeddings a ser utilizado.

    Returns:
        Instância do vectorstore ChromaDB.
    """
    from langchain_chroma import Chroma

    # Limite máximo de itens por requisição da API de embeddings do Gemini
    BATCH_SIZE = 100
    PAUSA_ENTRE_LOTES = 2  # segundos (pequena folga de segurança)

    total = len(docs)
    lotes = [docs[i:i + BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]
    num_lotes = len(lotes)

    st.info(
        f"Indexando **{total} chunks** em **{num_lotes} lote(s)** de até {BATCH_SIZE}. "
        f"(Otimizado para o plano gratuito do Gemini)"
    )

    vectorstore = None
    barra = st.progress(0, text="Iniciando indexacao no ChromaDB...")

    for idx, lote in enumerate(lotes):
        barra.progress(
            idx / num_lotes,
            text=f"Lote {idx + 1}/{num_lotes} - indexando {len(lote)} chunks..."
        )

        if vectorstore is None:
            # Primeiro lote: cria a colecao do zero
            vectorstore = Chroma.from_documents(
                documents=lote,
                embedding=embeddings,
                collection_name=COLLECTION_NAME,
                persist_directory=CHROMA_DIR,
            )
        else:
            # Lotes seguintes: adiciona a colecao existente
            vectorstore.add_documents(lote)

        # Aguarda entre lotes para respeitar o rate limit (exceto no ultimo)
        if idx < num_lotes - 1:
            for seg in range(PAUSA_ENTRE_LOTES, 0, -1):
                barra.progress(
                    (idx + 0.5) / num_lotes,
                    text=f"Processando... próximo lote em {seg}s "
                         f"(lote {idx + 1}/{num_lotes} concluído)"
                )
                time.sleep(1)

    barra.progress(1.0, text="Indexacao concluida!")
    time.sleep(0.5)
    barra.empty()

    st.success(f"Indexados {total} chunks com sucesso no ChromaDB!")
    return vectorstore


def load_vectorstore(embeddings):
    """
    Carrega um banco vetorial ChromaDB já existente do disco.

    Args:
        embeddings: Modelo de embeddings (deve ser o mesmo usado na criação).

    Returns:
        Instância do vectorstore ChromaDB ou None em caso de falha.
    """
    from langchain_chroma import Chroma

    try:
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
        )
        return vectorstore
    except Exception as e:
        st.warning(f"⚠️ Não foi possível carregar ChromaDB existente: {e}")
        return None


def get_llm():
    """
    Inicializa e retorna o modelo de linguagem Google Gemini.
    Lê a API key via st.secrets (Streamlit Cloud) ou os.getenv() (local).

    Returns:
        Instância do ChatGoogleGenerativeAI ou None em caso de falha.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    # Lê a chave de API: st.secrets tem prioridade (Streamlit Cloud), depois .env local
    api_key = st.secrets.get("GOOGLE_API_KEY", None) or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None

    try:
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=0.2,
            google_api_key=api_key,
            # Converte saída de HumanMessage para string simples
            convert_system_message_to_human=True,
        )
        # Teste rápido de conectividade com a API
        llm.invoke("Responda apenas: ok")
        return llm
    except Exception as e:
        st.sidebar.warning(f"⚠️ Erro ao conectar ao Gemini: {e}")
        return None


def build_rag_chain(vectorstore, llm):
    """
    Constrói o pipeline RAG (Retrieval-Augmented Generation) usando LCEL
    para compatibilidade com LangChain 1.x (sem depender de langchain.chains).

    Args:
        vectorstore: Banco vetorial ChromaDB para recuperação de contexto.
        llm: Modelo de linguagem para geração de respostas.

    Returns:
        Chain RAG configurada com retriever + LLM.
    """
    from langchain_core.runnables import RunnablePassthrough, RunnableParallel
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    # Prompt otimizado para português e domínio de seguros agrícolas
    prompt_template = """Você é um assistente especialista em seguros agrícolas da Insurmind.
Use APENAS as informações do contexto abaixo para responder à pergunta do usuário.
Se a resposta não estiver no contexto, diga "Não encontrei essa informação nos documentos disponíveis."
Responda sempre em português do Brasil, de forma clara e objetiva.

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

    # O pipeline RAG com retorno de documentos fontes usando RunnableParallel.
    # Como st.session_state.rag_chain.invoke({"query": prompt}) envia a chave "query",
    # mapeamos a extração desse valor para o retriever e prompt.
    setup_and_retrieval = RunnableParallel(
        {
            "context": (lambda x: x["query"]) | retriever | format_docs,
            "question": (lambda x: x["query"]),
            "source_documents": (lambda x: x["query"]) | retriever
        }
    )

    # Chain final estruturado para retornar {"result": ..., "source_documents": ...}
    chain = setup_and_retrieval | RunnableParallel(
        {
            "result": prompt | llm | StrOutputParser(),
            "source_documents": lambda x: x["source_documents"]
        }
    )

    return chain


# ---------------------------------------------------------------------------
# Inicialização do estado da sessão
# ---------------------------------------------------------------------------

def init_session_state():
    """Inicializa as variáveis de estado da sessão do Streamlit."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "llm_available" not in st.session_state:
        st.session_state.llm_available = False
    if "docs_loaded" not in st.session_state:
        st.session_state.docs_loaded = False
    if "reload_requested" not in st.session_state:
        st.session_state.reload_requested = False


# ---------------------------------------------------------------------------
# Função principal de configuração do RAG
# ---------------------------------------------------------------------------

def setup_rag(force_reload: bool = False):
    """
    Orquestra a configuração completa do pipeline RAG.
    Reutiliza o ChromaDB existente se disponível (a menos que force_reload=True).

    Args:
        force_reload: Se True, apaga o índice existente e reprocessa os PDFs.
    """
    embeddings = get_embeddings()
    vectorstore = None

    # --- Verifica se existe ChromaDB persistido e não é recarregamento forçado ---
    chroma_exists = os.path.isdir(CHROMA_DIR) and any(Path(CHROMA_DIR).iterdir())

    if chroma_exists and not force_reload:
        with st.spinner("📂 Carregando índice ChromaDB existente..."):
            vectorstore = load_vectorstore(embeddings)

        if vectorstore:
            try:
                count = vectorstore._collection.count()
                if count > 0:
                    st.sidebar.success(f"✅ ChromaDB carregado ({count} chunks)")
                else:
                    st.sidebar.warning("⚠️ Índice vazio. Reprocessando PDFs...")
                    force_reload = True
            except Exception as e:
                st.sidebar.warning(f"⚠️ Erro ao ler índice existente: {e}. Reprocessando...")
                force_reload = True
        else:
            force_reload = True

    # --- Processa PDFs se necessário ---
    if vectorstore is None or force_reload:
        if force_reload:
            # 1. Limpa a coleção do ChromaDB para evitar duplicados, se houver referências ativas
            for vs in [st.session_state.vectorstore, vectorstore]:
                if vs is not None:
                    try:
                        vs.delete_collection()
                    except Exception:
                        pass
            
            st.session_state.vectorstore = None
            vectorstore = None

            # 2. Tenta remover o diretório fisicamente, mas de forma segura contra locks de arquivo
            if os.path.isdir(CHROMA_DIR):
                try:
                    import shutil
                    shutil.rmtree(CHROMA_DIR)
                    st.info("🗑️ Diretório do índice antigo removido.")
                except Exception as e:
                    st.warning("⚠️ O banco vetorial antigo está em uso. Os dados antigos serão limpos e reindexados.")

        pdf_paths = scan_pdfs(".")

        if not pdf_paths:
            st.error(
                "❌ Nenhum arquivo PDF encontrado na pasta raiz do projeto.\n\n"
                "Adicione arquivos `.pdf` na pasta do projeto e reinicie a aplicação."
            )
            st.stop()

        st.sidebar.info(f"📄 {len(pdf_paths)} PDF(s) encontrado(s)")

        docs = load_and_split_pdfs(pdf_paths)

        if not docs:
            st.error("❌ Nenhum conteúdo extraído dos PDFs. Verifique se os arquivos não estão corrompidos.")
            st.stop()

        vectorstore = build_vectorstore(docs, embeddings)

    st.session_state.vectorstore = vectorstore
    st.session_state.docs_loaded = True

    # --- Inicializa o LLM Google Gemini ---
    with st.spinner(f"✨ Conectando ao Google Gemini ({GEMINI_MODEL})..."):
        llm = get_llm()

    if llm is None:
        st.session_state.llm_available = False
        st.sidebar.error(
            "⚠️ **Google Gemini indisponível!**\n\n"
            "Verifique:\n"
            "• Arquivo `.env` com `GOOGLE_API_KEY` válida\n"
            "• Conexão com a internet ativa"
        )
    else:
        st.session_state.llm_available = True
        st.session_state.rag_chain = build_rag_chain(vectorstore, llm)
        st.sidebar.success(f"✨ Gemini ({GEMINI_MODEL}) conectado!")


# ---------------------------------------------------------------------------
# Interface Principal — Streamlit UI
# ---------------------------------------------------------------------------

def main():
    # --- Configuração da página ---
    st.set_page_config(
        page_title="Insurmind RAG — Seguros Agrícolas",
        page_icon="🌾",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()

    # -----------------------------------------------------------------------
    # Barra Lateral (Sidebar)
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.image(
            "https://img.icons8.com/fluency/96/wheat.png",
            width=64,
        )
        st.title("Insurmind Agro")
        st.caption("Chatbot RAG — Seguros Agrícolas")
        st.divider()

        # --- Status dos componentes ---
        st.subheader("📊 Status do Sistema")

        # Inicialização automática na primeira execução
        if not st.session_state.docs_loaded:
            setup_rag(force_reload=False)

        # Botão discreto para recarregar PDFs
        st.divider()
        if st.button(
            "🔄 Recarregar PDFs",
            help="Use este botão se novos PDFs foram adicionados à pasta raiz.",
            use_container_width=True,
        ):
            # Limpa o cache de embeddings e reinicia
            get_embeddings.clear()
            st.session_state.docs_loaded = False
            st.session_state.messages = []
            setup_rag(force_reload=True)
            st.rerun()

        st.divider()

        # --- Lista os PDFs disponíveis ---
        st.subheader("📄 Documentos Indexados")
        pdfs = scan_pdfs(".")
        if pdfs:
            for pdf in pdfs:
                nome = Path(pdf).stem.replace("_", " ")
                st.markdown(f"• {nome}")
        else:
            st.caption("Nenhum PDF encontrado.")

        st.divider()

        # --- Botão para limpar histórico ---
        if st.button("🗑️ Limpar Conversa", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.caption("💡 Acesso em rede: `http://<seu-ip>:8501`")
        st.caption("Stack: Streamlit · LangChain · ChromaDB · Gemini")

    # -----------------------------------------------------------------------
    # Área principal — Chat
    # -----------------------------------------------------------------------
    st.title("🌾 Assistente de Seguros Agrícolas")
    st.markdown(
        "Faça perguntas sobre as **Condições Gerais** dos seguros agrícolas indexados. "
        "As respostas são geradas com base exclusivamente nos documentos da sua apólice."
    )

    # Aviso se o LLM não estiver disponível
    if not st.session_state.llm_available:
        st.warning(
            "⚠️ **Google Gemini indisponível.** Verifique se o arquivo `.env` "
            "contém a variável `GOOGLE_API_KEY` com uma chave válida e se há "
            "conexão com a internet."
        )

    # --- Exibe histórico de mensagens ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Exibe fontes se disponíveis
            if message.get("sources"):
                with st.expander("📚 Fontes utilizadas", expanded=False):
                    for src in message["sources"]:
                        st.caption(
                            f"📄 **{src['arquivo']}** — Página {src['pagina']}"
                        )

    # --- Input do usuário ---
    if prompt := st.chat_input(
        "Faça sua pergunta sobre os seguros agrícolas...",
        disabled=not st.session_state.docs_loaded,
    ):
        # Adiciona mensagem do usuário ao histórico
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Gera resposta do assistente
        with st.chat_message("assistant"):
            if not st.session_state.llm_available:
                resposta = (
                    "⚠️ O Google Gemini não está disponível no momento. "
                    "Verifique a chave de API no arquivo `.env` e a conexão com a internet."
                )
                st.markdown(resposta)
                st.session_state.messages.append(
                    {"role": "assistant", "content": resposta}
                )
            else:
                with st.spinner("🤔 Consultando os documentos..."):
                    try:
                        resultado = st.session_state.rag_chain.invoke({"query": prompt})
                        resposta = resultado.get("result", "Não foi possível gerar uma resposta.")
                        source_docs = resultado.get("source_documents", [])

                        # Processa as fontes para exibição
                        fontes = []
                        fontes_vistas = set()
                        for doc in source_docs:
                            meta = doc.metadata
                            arquivo = Path(meta.get("source", "Desconhecido")).name
                            pagina = meta.get("page", "?")
                            chave = f"{arquivo}_{pagina}"
                            if chave not in fontes_vistas:
                                fontes_vistas.add(chave)
                                fontes.append({"arquivo": arquivo, "pagina": pagina + 1})

                        st.markdown(resposta)

                        if fontes:
                            with st.expander("📚 Fontes utilizadas", expanded=False):
                                for src in fontes:
                                    st.caption(
                                        f"📄 **{src['arquivo']}** — Página {src['pagina']}"
                                    )

                        # Salva no histórico
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


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
