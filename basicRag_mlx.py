import streamlit as st


from langchain_community.llms.mlx_pipeline import MLXPipeline


from langchain.schema import (
    HumanMessage
)
from langchain_community.chat_models.mlx import ChatMLX
from langchain_community.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain import hub
from langchain_core.runnables import RunnablePassthrough, RunnablePick
from langchain_core.output_parsers import StrOutputParser

st.title('Basic RAG Application')



pdf_file = st.file_uploader('upload your pdfs here',type='pdf')

if pdf_file is not None:

    with st.spinner("Processing PDF..."):
        with open("temp.pdf", "wb") as f:
                f.write(pdf_file.getbuffer())
        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        all_splits = text_splitter.split_documents(documents)

        model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
        gpt4all_kwargs = {'allow_download': 'True'}
        embeddings = GPT4AllEmbeddings(
            model_name=model_name,
            gpt4all_kwargs=gpt4all_kwargs
        )
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)
    st.success("PDF processed and embeddings created!")


    query = st.text_input("Enter Prompt")
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    if query:
        with st.spinner('Getting Answers....'):
            docs = vectorstore.similarity_search(query)
            llm = MLXPipeline.from_model_id(
            "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
            pipeline_kwargs={"max_tokens": 100, "temp": 0.1},)

            rag_prompt_llama = hub.pull("rlm/rag-prompt-llama")
            chain = (
                RunnablePassthrough.assign(context=RunnablePick("context") | format_docs)
                | rag_prompt_llama
                | llm
                | StrOutputParser()
            )
            response =chain.invoke({"context": docs, "question": query})

        st.write(response.split('[/INST]')[0])

      
