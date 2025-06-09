import re
from pathlib import Path
import os
import hashlib
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass  # Fall back to standard sqlite3 if unavailable


def sanitize_filename(filename):
    # Separa nome e extensão
    name, ext = os.path.splitext(filename)
    
    # Sanitiza apenas o nome do arquivo (mantém extensão)
    clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    clean_name = re.sub(r'_+', '_', clean_name).strip('_')
    
    # Mantém a extensão original
    clean_ext = ext.split('.')[-1][:5]  # Limita extensão a 5 caracteres
    return f"{clean_name[:50]}.{clean_ext}"  # Limita nome a 50 caracteres

def handle_document_upload(uploaded_files):
    temp_dir = Path("./temp").resolve()
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    documents = []
    error_messages = []

    for file in uploaded_files:
        temp_path = None
        try:
            # Geração de nome seguro mantendo extensão
            original_name = file.name
            safe_name = sanitize_filename(original_name)
            unique_hash = hashlib.md5(file.getvalue()).hexdigest()[:6]
            final_name = f"{unique_hash}_{safe_name}"
            temp_path = temp_dir / final_name

            # Salvamento do arquivo
            with open(temp_path, "wb") as f:
                file_bytes = file.getvalue()
                if len(file_bytes) == 0:
                    raise ValueError("Arquivo vazio")
                f.write(file_bytes)

            # Verificação de tipo pelo conteúdo
            if file.type == "application/pdf":
                loader = PyPDFLoader(str(temp_path))
            elif file.type == "text/plain":
                loader = TextLoader(str(temp_path))
            else:
                raise ValueError(f"Tipo não suportado: {file.type}")

            # Processamento do documento
            loaded_docs = loader.load()
            if not loaded_docs:
                raise ValueError("Nenhum conteúdo extraído")
                
            documents.extend(loaded_docs)
            st.success(f"Arquivo processado: {original_name}")

        except Exception as e:
            error_messages.append(f"Erro em {original_name}: {str(e)}")
            continue

        finally:
            # Limpeza segura
            if temp_path and temp_path.exists():
                try:
                    os.remove(temp_path)
                except Exception as e:
                    error_messages.append(f"Erro na limpeza de {original_name}: {str(e)}")

    # Exibe todos os erros de uma vez
    if error_messages:
        st.error("\n\n".join(error_messages))

    if not documents:
        raise ValueError("Nenhum documento válido foi processado. Verifique:\n"
                        "1. Formatos suportados (PDF/TXT)\n"
                        "2. Arquivos não corrompidos\n"
                        "3. Conteúdo legível")

    # Processamento dos documentos válidos
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    splits = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local("vector_store")