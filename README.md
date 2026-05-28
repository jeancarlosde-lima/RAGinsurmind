# 🌾 Insurmind RAG — Chatbot de Seguros Agrícolas

Chatbot **100% local** com pipeline RAG para consulta de Condições Gerais de apólices de seguros agrícolas.

> **Stack:** Python · Streamlit · LangChain · ChromaDB · Ollama (llama3) · HuggingFace Embeddings

---

## ⚙️ Pré-requisitos

| Requisito | Versão mínima | Link |
|---|---|---|
| Python | 3.10+ | [python.org](https://python.org) |
| Ollama | Qualquer | [ollama.com](https://ollama.com) |

---

## 🚀 Instalação e Execução

### 1. Clone/acesse a pasta do projeto

```bash
cd RAG-Insurmind
```

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

> 💡 **Recomendado:** use um ambiente virtual:
> ```bash
> python -m venv .venv
> .venv\Scripts\activate   # Windows
> pip install -r requirements.txt
> ```

### 3. Instale e inicie o Ollama

```bash
# Instale o Ollama em: https://ollama.com
ollama pull llama3       # Baixa o modelo (~4.7 GB)
ollama serve             # Inicia o servidor (manter aberto)
```

### 4. Execute o chatbot

```bash
streamlit run app.py
```

O sistema irá automaticamente:
- Escanear os PDFs na pasta raiz
- Criar o índice vetorial em `./chroma_db/`
- Abrir o chat no navegador em `http://localhost:8501`

---

## 🌐 Compartilhar com a Equipe (Rede Local)

O `config.toml` já configura o servidor para `0.0.0.0:8501`, permitindo acesso por qualquer máquina na mesma rede Wi-Fi/Ethernet.

### Como descobrir o IP da sua máquina:

**Windows:**
```powershell
# Opção 1 — PowerShell
(Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.InterfaceAlias -notlike "*Loopback*" }).IPAddress

# Opção 2 — Prompt de comando (cmd)
ipconfig
# Procure por "Endereço IPv4" na seção da sua rede (Wi-Fi ou Ethernet)
```

**macOS / Linux:**
```bash
hostname -I       # Linux
ipconfig getifaddr en0   # macOS (Wi-Fi)
```

### Compartilhe o link:

```
http://<SEU-IP>:8501
```

**Exemplo:** `http://192.168.1.105:8501`

> ⚠️ **Atenção:** Todos na mesma rede poderão acessar o chatbot. Não exponha para redes públicas sem autenticação.

---

## 📂 Estrutura do Projeto

```
RAG-Insurmind/
├── app.py                        # Aplicação principal
├── requirements.txt              # Dependências Python
├── README.md                     # Este arquivo
├── .streamlit/
│   └── config.toml               # Configuração do servidor Streamlit
├── chroma_db/                    # Banco vetorial persistente (gerado automaticamente)
├── CG_Agro_Custeio.pdf
├── CG_Agro_Custeio_RN.pdf
├── CG_Agro_Granizo.pdf
├── CG_Agro_ProdutividadeMulti.pdf
└── CG_Agro_ProdutividadeRN.pdf
```

---

## 🔄 Adicionando Novos PDFs

1. Coloque o novo arquivo `.pdf` na pasta raiz do projeto
2. No chatbot, clique em **"🔄 Recarregar PDFs"** na barra lateral
3. O sistema reindexará todos os documentos automaticamente

---

## 🧠 Como funciona o RAG

```
PDF(s) → PyPDFLoader → Chunks (1000 chars / overlap 200)
       → HuggingFace Embeddings → ChromaDB (persistente)

Pergunta → Embedding → ChromaDB (Top-4 chunks)
         → Prompt + Contexto → Ollama (llama3)
         → Resposta fundamentada nos documentos
```

---

## 🛠️ Solução de Problemas

| Problema | Solução |
|---|---|
| `Ollama não disponível` | Execute `ollama serve` no terminal |
| `Modelo não encontrado` | Execute `ollama pull llama3` |
| `Nenhum PDF encontrado` | Adicione `.pdf` na pasta raiz |
| `Erro ao carregar ChromaDB` | Clique em "Recarregar PDFs" na sidebar |
| Respostas lentas | Normal na primeira execução (carregamento do modelo de embeddings) |
