import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS  # Facebook AI Similarity Search
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

OPENAI_API_KEY = " YOUR API KEY" #I have used my secret key here. Please use yours as required.

#Upload PDF Files
st.header("My Chatbot")

with st.sidebar:
    st.title("Your files")
    file = st.file_uploader("Upload a PDF file and ask questions", type="pdf")


#Extract text

if file is not None:
    pdfReader = PdfReader(file)
    text = ""

    for page in pdfReader.pages:
        text += page.extract_text()
        #st.write(text)

#Break into chunks
    textSplit = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )

    chunks = textSplit.split_text(text)
    #st.write(chunks)

    #Generate embedings

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    #Creating Vector store(FAISS)

    vectorStore = FAISS.from_texts(chunks,embeddings)

    #Get user question

    input = st.text_input("Type your Question here")

    #Do similarity search

    if input:
        match = vectorStore.similarity_search(input)
        #st.write(match)

        llm = ChatOpenAI(
        openai_api_key = OPENAI_API_KEY,
        temperature = 0,
        max_tokens = 1000,
        model_name = "gpt-3.5-turbo"
        )
         #Output results
        chain = load_qa_chain(llm, chain_type="stuff")
        resp = chain.run(input_documents = match, question = input)
        st.write(resp)



#References : https://www.udemy.com/course/generative-ai-for-beginners-b/learn/lecture/40913810#overview
