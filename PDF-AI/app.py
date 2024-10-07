import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain_community.llms.huggingface_hub import HuggingFaceHub
import requests
import os



response = requests.get(
    'https://api-inference.huggingface.co/models/google/flan-t5-large',
    headers={'Authorization': f'Bearer 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'},
    verify=False
)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="google/flan-t5-large")
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id="google/flan-t5-large", huggingfacehub_api_token='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')
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    if st.session_state.conversation:
        prompt = f"Resuma o seguinte documento ou responda à pergunta: {user_question}"
        response = st.session_state.conversation({'question': prompt})
        
    chat_history = response.get('chat_history', [])

    if isinstance(chat_history, list):
        st.session_state.chat_history = chat_history
    
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    else:
        st.error("Por favor, processe um PDF primeiro.")



def main():
    load_dotenv()
    st.set_page_config(page_title="Leitor de PDFs", page_icon="📕")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
        st.session_state.chat_history = []

    st.header("Importe seus PDFs 📕")
    user_question = st.text_input("Faça uma pergunta sobre seu PDF:")
    if user_question:
        handle_userinput(user_question)

    st.write(user_template.replace("{{MSG}}", "Olá, IA!"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Olá, como posso ajudar?"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Seus Documentos")
        pdf_docs = st.file_uploader(
            "Upload your PDF's here and click on Process", accept_multiple_files=True)
        
        if st.button("Processar"):
            if pdf_docs:  # Verifica se PDFs foram carregados
                with st.spinner("Processando"):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.success("PDF processado com sucesso! Você pode agora fazer perguntas.")
            else:
                st.error("Por favor, carregue pelo menos um PDF.")
        else:
            st.write("Por favor, carregue um PDF para começar.")

load_dotenv()
token = os.getenv('HUGGINGFACEHUB_API_TOKEN')
print(f"Loaded token: {token}")

if __name__ == '__main__':
    main()