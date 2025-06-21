import streamlit as st
from rag_handler import process_query
from document_loader import handle_document_upload
from pathlib import Path
from document_loader import handle_document_upload

try:# if on streamlit cloud
    MODEL="gpt-4.1-nano"
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except st.errors.StreamlitSecretNotFoundError:# if running locally
    from dotenv import load_dotenv
    load_dotenv()

st.set_page_config(
    page_title="RAG-Powered Educational Chatbot",
    page_icon="🤖",
    layout="wide"
)

Path("./temp").mkdir(exist_ok=True)

def main():
    st.title("Education Assistant Chatbot")
    
    # Configuração inicial de estado
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Elementos da sidebar
    with st.sidebar:
        uploaded_files = st.file_uploader(
            "Load documents (PDF, TXT)",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            key="doc_uploader"  # Chave única para uploader
        )
        
        use_web_search = st.toggle(
            "Use web search",
            value=True,
            key="web_search_toggle"  # Chave única para toggle
        )
        
        if st.button("Clear History", key="clear_chat"):
            st.session_state.messages = []
            st.rerun()

    # Processamento de documentos
    if uploaded_files:
        with st.spinner("Processing documents..."):
            handle_document_upload(uploaded_files)
            st.session_state.documents_processed = True
    else:
        st.session_state.documents_processed = False

    # Área de mensagens do chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input principal do chat com chave fixa
    if prompt := st.chat_input(
        "How can I help?", 
        key="main_chat_input"  # Chave fixa e única
    ):
        # Validação de entrada
        if not prompt.strip():
            st.warning("Por favor, digite uma mensagem válida")
            st.stop()
            
        # Verificação de pré-requisitos
        if not use_web_search and not st.session_state.get("documents_processed"):
            st.error("🔍 Ative a busca na web ou carregue documentos primeiro")
            st.stop()

        # Processamento da resposta
        with st.spinner("Analyzing..."):
            try:
                # Adiciona mensagem do usuário
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Obtém resposta
                response = process_query(prompt, use_web_search)
                
                # Adiciona e exibe resposta
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Rerun para atualização imediata
                st.rerun()
                
            except Exception as e:
                st.error(f"Erro no processamento: {str(e)}")
                st.stop()

if __name__ == "__main__":
    main()