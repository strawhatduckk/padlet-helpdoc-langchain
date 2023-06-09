import os
# Import OpenAI as main LLM service
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
# UI
import streamlit as st 
# PDF document loaders
from langchain.document_loaders import PyPDFLoader
# Vector store 
from langchain.vectorstores import Chroma
# Vector store stuff
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

# Set APIkey for OpenAI Service
os.environ['OPENAI_API_KEY'] = 'api-key'

# Create instance of OpenAI LLM
llm = OpenAI(temperature=0.1, verbose=True)
embeddings = OpenAIEmbeddings()

# Create and load PDF Loader
loader = PyPDFLoader('annualreport.pdf')
# Split pages from pdf 
pages = loader.load_and_split()
# Load documents into ChromaDB
store = Chroma.from_documents(pages, embeddings, collection_name='annualreport')

# Create vectorstore info object
vectorstore_info = VectorStoreInfo(
    name="annual_report",
    description="a banking annual report as a pdf",
    vectorstore=store
)
# Convert the document store into a langchain toolkit
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

# Add the toolkit to an end-to-end LC
agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)

st.title('🏗 Padlet Helpdocs Proompt Engineer')
# Create a text input box for the user
prompt = st.text_input('Input your prompt here')

# If the user hits enter
if prompt:
    # Then pass the prompt to the LLM
    response = agent_executor.run(prompt)
    # ...and write it out to the screen
    st.write(response)

    # With a streamlit expander  
    with st.expander('Document Similarity Search'):
        # Find the relevant pages
        search = store.similarity_search_with_score(prompt) 
        # Write out the first 
        st.write(search[0][0].page_content) 