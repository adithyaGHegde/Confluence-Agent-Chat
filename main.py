import streamlit as st
import os
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain_community.chat_models import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import ConfluenceLoader
from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever, TFIDFRetriever
from langchain_core.prompts import PromptTemplate
import os
import streamlit as st
from langchain.agents import AgentExecutor, create_react_agent

import requests
from requests.auth import HTTPBasicAuth
import json

st.set_page_config(page_title="Chat with Confluence space")    

persist_directory = "./db/chroma/"
if not os.path.exists("Data"):
    os.makedirs("Data")

default_url = st.secrets["CONFLUENCE_URL"]
default_api = st.secrets["ATLASSIAN_API_KEY"]
default_email = st.secrets["DEFAULT_EMAIL"]

def load_pages_data(url,email,api_key):
    url =  f"{url}//wiki/api/v2/pages"

    auth = HTTPBasicAuth(email, api_key)

    headers = {
        "Accept": "application/json"
    }

    response = requests.request(
        "GET",
        url,
        headers=headers,
        auth=auth
    )

    data = json.loads(response.text)

    page_data = {}
    for page in data["results"]:
        page_data[page["id"]] = page["title"]
    
    return page_data

def load_tools_each_page(page_data,url,email,api_key):

    tools = []

    for page_id, page_title in page_data.items():
        loader = ConfluenceLoader(
            url=f"{url}/wiki", username=email, api_key=api_key
        )
        page = loader.load(page_ids=[page_id], include_attachments=True, limit=50)

        documents = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        ).split_documents(page)

        HF_token = st.secrets["HF_TOKEN"]

        embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key = HF_token,model_name = "sentence-transformers/all-MiniLM-L6-v2"
        )
        vector = Chroma.from_documents(documents, embeddings,persist_directory=persist_directory)
        retriever = vector.as_retriever()

        retrieverTool = create_retriever_tool(
            retriever,
            page_title,
            f"Contains information about {page_title}.",
        )
        tools.append(retrieverTool)

    return tools

def main():
    with st.sidebar:
        st.header("Database Selection")
        database_choice = st.selectbox(
            "Choose your database source:",
            ["Use Our Existing Confluence", "Use your own Confluence"],
            index=0
        )
        if database_choice == "Use your own Confluence":
            confluence_url = st.text_input("Confluence URL")
            email_or_name = st.text_input("Email")
            API_KEY = st.text_input("Atlassian API Key",type="password")
            
            tools = []
            if confluence_url and email_or_name and API_KEY:
                pages_data = load_pages_data(confluence_url,email_or_name,API_KEY)
                tools = load_tools_each_page(pages_data,confluence_url,email_or_name,API_KEY)
        else:
            confluence_url = default_url
            email_or_name = default_email
            API_KEY = default_api
            pages_data = load_pages_data(confluence_url,email_or_name,API_KEY)
            tools = load_tools_each_page(pages_data,confluence_url,email_or_name,API_KEY)
        
        DEPLOYMENT_NAME = st.secrets['DEPLOYMENT_NAME']
        AZURE_OPENAI_API_KEY = st.secrets['AZURE_OPENAI_API_KEY']
        AZURE_OPENAI_ENDPOINT = st.secrets['AZURE_OPENAI_ENDPOINT']

        llm = AzureChatOpenAI(
        openai_api_version="2023-07-01-preview",
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        deployment_name=DEPLOYMENT_NAME,
        openai_api_key=AZURE_OPENAI_API_KEY,
        model_name = 'gpt-4-32k',
        openai_api_type="azure",
        )
        prompt = PromptTemplate(input_variables=['agent_scratchpad', 'input', 'tool_names', 'tools'], template='Answer the following user queries related  as best you can.You have access to the following tools:\n\n{tools}\n\nUse the following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction: the action to take, should be one of [{tool_names}]\nAction Input: the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question\n\nBegin!\n\nQuestion: {input}\nThought:{agent_scratchpad}')
        
        if tools and len(tools)>0:
            agent = create_react_agent(llm, tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
        else:
            agent = create_react_agent(llm, tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
            st.write("\nPlease provide/recheck your Confluence details, press enter after entering each block")


    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["I'm Confluence ChatBot Assistant. I help you with your queries"]
    if 'past' not in st.session_state:
        st.session_state['past'] = ['How can you help me?']
    
    input_container = st.container()
    colored_header(label='', description='', color_name='blue-30')
    response_container = st.container()  
    
    def get_text():
        input_text = st.text_input("You: ", "", key="input")
        return input_text
    
    with input_container:
        user_input = get_text()
    
    def generate_response(user_input):
        if agent_executor:    
            response = agent_executor.invoke({"input": user_input})
            return response["output"]

    with response_container:
        if user_input:
            response = generate_response(user_input)
            st.session_state.past.append(user_input)
            st.session_state.generated.append(response)
            
        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))

if __name__ == '__main__':
    main()