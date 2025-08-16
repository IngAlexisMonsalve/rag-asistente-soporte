

import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_huggingface import ChatHuggingFace
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# --- CONSTANTES ---
CHROMA_PATH = "vector_db"
DOCS_PATH = "docs/"
EMBEDDING_MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1"
LLM_REPO_ID = "moonshotai/Kimi-K2-Instruct"

# Cargar variables de entorno
load_dotenv()

# --- VERIFICACIÓN DEL API TOKEN ---
if not os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
    st.error("¡ERROR DE CONFIGURACIÓN! No se encontró el Hugging Face API Token.")
    st.info("Asegúrate de que tu archivo .env existe y contiene tu token, o que has configurado el secreto 'HUGGINGFACEHUB_API_TOKEN' en Streamlit Cloud.")
    st.stop()

# --- FUNCIONES DE CARGA Y PREPARACIÓN (CACHEADAS) ---

@st.cache_resource
def prepare_database():
    """
    Prepara la base de datos vectorial. Si no existe, la crea a partir de los documentos.
    Si ya existe, simplemente la carga.
    """
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    if not os.path.exists(CHROMA_PATH):
        st.info(f"Iniciando primera configuración: Creando base de datos desde '{DOCS_PATH}'...")
        
        # 1. Cargar documentos
        all_documents = []
        for filename in os.listdir(DOCS_PATH):
            filepath = os.path.join(DOCS_PATH, filename)
            if filename.endswith((".pdf", ".txt", ".md")):
                try:
                    if filename.endswith(".pdf"):
                        loader = PyPDFLoader(filepath)
                    else:
                        loader = TextLoader(filepath, encoding="utf-8")
                    documents = loader.load()
                    all_documents.extend(documents)
                except Exception as e:
                    st.warning(f"No se pudo cargar el archivo {filename}: {e}")
            else:
                st.info(f"Archivo omitido (tipo no soportado): {filename}")

        if not all_documents:
            st.error("No se encontraron documentos para procesar. Asegúrate de que la carpeta 'docs' contiene archivos .md, .txt o .pdf.")
            st.stop()

        # 2. Dividir texto en fragmentos
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs_chunks = text_splitter.split_documents(all_documents)

        # 3. Crear y persistir la base de datos
        db = Chroma.from_documents(docs_chunks, embedding_model, persist_directory=CHROMA_PATH)
        st.success(f"¡Base de datos creada con {len(docs_chunks)} fragmentos! Aplicación lista.")

    else:
        st.info("Cargando base de datos vectorial existente...")
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model)
        st.success("Base de datos cargada.")

    return db.as_retriever(search_kwargs={"k": 5})

@st.cache_resource
def load_chat_model():
    """Carga el modelo de lenguaje para el chat."""
    llm = HuggingFaceEndpoint(
        repo_id=LLM_REPO_ID,
        temperature=0.1,
        max_new_tokens=512
    )
    return ChatHuggingFace(llm=llm)

# --- INTERFAZ DE STREAMLIT ---

st.title("Asistente de Soporte Técnico")
st.write("Respondo preguntas sobre tus documentos usando un modelo conversacional.")

try:
    retriever = prepare_database()
    chat_model = load_chat_model()

    user_query = st.text_input("Escribe tu pregunta aquí:")

    if user_query:
        with st.spinner("Buscando documentos y generando respuesta..."):
            # 1. Recuperar documentos
            retrieved_docs = retriever.get_relevant_documents(user_query)

            if retrieved_docs:
                # 2. Crear el contexto y la lista de mensajes
                context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                
                messages = [
                    SystemMessage(content=f"Eres un asistente experto. Usa el siguiente contexto para responder la pregunta del usuario. **IMPORTANTE: Tu respuesta DEBE ser siempre en español.** Contexto: {context}"),
                    HumanMessage(content=user_query)
                ]

                # 3. Llamar al modelo de chat
                response = chat_model.invoke(messages)

                # 4. Mostrar respuesta y fuentes
                st.subheader("Respuesta:")
                st.write(response.content)

                st.subheader("Fuentes:")
                with st.expander("Ver documentos usados para la respuesta"):
                    for doc in retrieved_docs:
                        st.write(f"- **Fuente:** {doc.metadata.get('source', 'Desconocida')}")
                        st.divider()
            else:
                st.subheader("Respuesta:")
                st.write("No pude encontrar documentos relevantes para tu pregunta.")

except Exception as e:
    st.subheader("Ha ocurrido un error")
    st.error(f"Detalles: {e}")
