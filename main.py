import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import os 
import requests

huggingface_api_key = st.secrets["HUGGINGFACEHUB_API_KEY"]
os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_key

def get_pdf_text(pdf_docs): # Understood & No modification reqr
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text): # Understood & No modification reqr
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device':'cpu'}
    # Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
    encode_kwargs = {'normalize_embeddings': False}
    # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
    embeddings = HuggingFaceEmbeddings(
        model_name=model_id,     # Provide the pre-trained model's path
        model_kwargs=model_kwargs, # Pass the model configuration options
        encode_kwargs=encode_kwargs # Pass the encoding options
    )
    # hf_token = "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"
    # api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
    # headers = {"Authorization": f"Bearer {hf_token}"}
    # embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    # Pass the directory path where the embedding model is stored on your system
    # embedding_model_name = "../model/instructor-base"

    # Initialize an instance of HuggingFaceInstructEmbeddings
    # instructor_embeddings = HuggingFaceInstructEmbeddings(
    #     model_name=embedding_model_name,
    #     model_kwargs={"device": "cpu"}  # Specify the device to be used for inference (GPU - "cuda" or CPU - "cpu")
    # )
    # embeddings = instructor_embeddings
    # response = requests.post(api_url, headers=headers, json={"inputs": texts, "options":{"wait_for_model":True}})
    # return response.json()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    # # Pass the directory path where the model is stored on your system
    # model_name = "../model/flan-t5-large/"

    # # Initialize a tokenizer for the specified model
    # tokenizer = AutoTokenizer.from_pretrained(model_name)

    # # Initialize a model for sequence-to-sequence tasks using the specified pretrained model
    # model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # # Create a pipeline for text-to-text generation using a specified model and tokenizer
    # pipe = pipeline(
    #     "text2text-generation",  # Specify the task as text-to-text generation
    #     model=model,              # Use the previously initialized model
    #     tokenizer=tokenizer,      # Use the previously initialized tokenizer
    #     max_length=512,           # Set the maximum length for generated text to 512 tokens
    #     temperature=0,            # Set the temperature parameter for controlling randomness (0 means deterministic)
    #     top_p=0.95,  
    #     # do_sample = True,              # Set the top_p parameter for controlling the nucleus sampling (higher values make output more focused)
    #     repetition_penalty=1.15   # Set the repetition_penalty to control the likelihood of repeated words or phrases
    # )

    # # Create a Hugging Face pipeline for local  language model (LLM) using the 'pipe' pipeline
    # llm = HuggingFacePipeline(pipeline=pipe)


    
    # conversation_chain = RetrievalQA.from_chain_type(llm=local_llm, chain_type="stuff", retriever=vectorstore.as_retriever(), return_source_documents=True, memory=memory)


    # llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature":0.5, "max_length":512})
    llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-v0.1", model_kwargs={"temperature":0.5, "max_length":512})

    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory
    )
    # conversation_chain = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     chain_type="stuff",
    #     retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    #     memory=memory
    # )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    # load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                st.write("Processing your documents...")
                st.write("It will take a while based on your cpu and number of documents...")
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                # vectorstore = get_vectorstore(text_chunks)
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("Documents processed successfully!")


if __name__ == '__main__':
    main()