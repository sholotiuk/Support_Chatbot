import os
import pickle
import openai
import streamlit as st

from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.callbacks import get_openai_callback
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Set the title of the Streamlit application
st.title('🤗💬 Amazon Support Chatbot')

# Get the OpenAI API key from the user
openai.api_key = st.sidebar.text_input('OpenAI API Key', type='password')

def generate_response(query):
    # Load the text document
    loader = TextLoader("data/amazon_support.txt")
    text_document = loader.load()
    text = str(text_document)  # Convert Document to string
 
    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
 
    # Load or generate embeddings
    store_name = "support_policies"
    if os.path.exists(f"data/{store_name}.pkl"):
        with open(f"data/{store_name}.pkl", "rb") as f:
            VectorStore = pickle.load(f)
    else:
        embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        with open(f"data/{store_name}.pkl", "wb") as f:
            pickle.dump(VectorStore, f)


    # Find the most similar documents to the query
    docs = VectorStore.similarity_search(query=query, k=3)
 
    # Generate a response using the OpenAI language model
    llm = OpenAI(openai_api_key=openai.api_key)
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=query)
        print(cb)
    st.info(response)

# Create a form for the user to enter their query
with st.form('my_form'):
    query = st.text_area("Hi! I'm a support chatbot for Amazon. I can answer questions about Returns Policies. How can I help you?")
    submitted = st.form_submit_button('Ask')
    if not openai.api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    if submitted and openai.api_key.startswith('sk-'):
        generate_response(query)