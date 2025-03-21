import streamlit as st
from rag_handler import process_query
from document_loader import handle_document_upload
from pathlib import Path
from document_loader import handle_document_upload

MODEL="gpt-4o-mini"
OPENAI_API_KEY=st.secrets("OPENAI_API_KEY")

st.set_page_config(
    page_title="Chatbot RAG Avançado",
    page_icon="🤖",
    layout="wide"
)

Path("./temp").mkdir(exist_ok=True)

def main():
    st.title("Chatbot RAG Inteligente")
    
    # Configuração inicial de estado
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Elementos da sidebar
    with st.sidebar:
        uploaded_files = st.file_uploader(
            "Carregue documentos (PDF, TXT)",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            key="doc_uploader"  # Chave única para uploader
        )
        
        use_web_search = st.toggle(
            "Usar busca na web",
            value=True,
            key="web_search_toggle"  # Chave única para toggle
        )
        
        if st.button("Limpar Histórico", key="clear_chat"):
            st.session_state.messages = []
            st.rerun()

    # Processamento de documentos
    if uploaded_files:
        with st.spinner("Processando documentos..."):
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
        "Como posso ajudar?", 
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
        with st.spinner("Analisando..."):
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