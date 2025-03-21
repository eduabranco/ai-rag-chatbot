from langchain.llms import OpenAI
import os
import web_search
from pathlib import Path
import streamlit as st

def get_retriever():
    from langchain_community.vectorstores import FAISS
    from langchain.embeddings import HuggingFaceEmbeddings
    
    # Verificar se o vetorstore existe
    if not Path("vector_store/index.faiss").exists():
        raise FileNotFoundError("Execute o processamento de documentos primeiro")
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    vectorstore = FAISS.load_local(
        "vector_store",
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    return vectorstore.as_retriever(search_kwargs={"k": 3})

def process_query(query, use_web_search=False):
    llm = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat"
    )
    
    context = ""
    
    if use_web_search:
        context = web_search.web_search(query)
    elif st.session_state.get("documents_processed", False):
        retriever = get_retriever()
        docs = retriever.get_relevant_documents(query)
        context = "\n".join([d.page_content for d in docs])
    else:
        return "Por favor, carregue documentos ou ative a busca na web"
    
    return generate_response(llm, query, context)

def generate_response(llm, query, context):
    from crewai_manager import create_crew
    try:
        crew = create_crew(query, context)
        return crew.kickoff()
    except Exception as e:
        return f"Erro na geração de resposta: {str(e)}"