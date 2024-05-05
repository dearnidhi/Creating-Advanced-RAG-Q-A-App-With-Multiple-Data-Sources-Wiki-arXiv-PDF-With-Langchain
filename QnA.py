from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)


from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = WebBaseLoader("https://docs.smith.langchain.com/")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)

# Provide the OpenAI API key as an argument to OpenAIEmbeddings
vectordb = FAISS.from_documents(documents, OpenAIEmbeddings(openai_api_key=""))
retriever = vectordb.as_retriever()

from langchain.tools.retriever import create_retriever_tool
retriever_tool = create_retriever_tool(retriever, "langsmith_search",
                                     "Search for information about LangSmith.")

from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

tools = [wiki, arxiv, retriever_tool]

from dotenv import load_dotenv
load_dotenv()

# No need to import ChatOpenAI again, it was already imported earlier
# from langchain_openai import ChatOpenAI
from langchain_openai import ChatOpenAI

# Create LLM agent
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

from langchain import hub
prompt = hub.pull("hwchase17/openai-functions-agent")

from langchain.agents import create_openai_tools_agent
agent = create_openai_tools_agent(llm, tools, prompt)

from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Invoke the agent executor with user queries
response = agent_executor.invoke({"input": "Tell me about Langsmith"})
print(response)

response = agent_executor.invoke({"input": "What's the paper 1605.08386 about?"})
print(response)


# Streamlit UI
import streamlit as st
st.title("RAG Q&A App")

query = st.text_input("Ask a question:", "What is Langchain?")

if st.button("Submit"):
    response = agent_executor.invoke({"input": query})
    st.write("Response:", response)