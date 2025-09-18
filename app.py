import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.tools import TavilySearchResults
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory

# 1. Load API keys
load_dotenv(dotenv_path=".env")
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# 2. LLM model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 3. PromptTemplate + LLMChain (basic QA)
prompt = PromptTemplate.from_template("Answer clearly: {question}")
qa_chain = LLMChain(llm=llm, prompt=prompt)

# 4. Load sample documents for RAG
loader = TextLoader("sample.txt")  # make sure you have this file
documents = loader.load()

# 5. Embed and store in FAISS vector DB
embedding = OpenAIEmbeddings()
vectordb = FAISS.from_documents(documents, embedding)
retriever = vectordb.as_retriever()

# 6. RetrievalQA chain (RAG)
rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 7. External Tool - Web Search
search_tool = Tool(
    name="Tavily Search",
    func=TavilySearchResults(max_results=3).run,
    description="Search the internet for additional information"
)

# 8. Wrap RAG and QA as tools
tools = [
    Tool(name="Simple QA", func=qa_chain.run, description="Answer with basic LLMChain"),
    Tool(name="RAG Search", func=rag_chain.run, description="Answer using document retrieval"),
    search_tool
]

# 9. Memory (stores chat history)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 10. Agent that chooses the best tool
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory
)

# 11. Run it
query = "What is langgraph in langchain?"
print("\nUser Question:", query)
answer = agent.run(query)
print("\nFinal Answer:", answer)
    