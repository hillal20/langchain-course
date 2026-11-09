import os

os.environ['OLLAMA_HOST'] = 'http://127.0.0.1:11434'

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama



load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from tavily import TavilyClient
from langchain_tavily import TavilySearch

tavily = TavilyClient()

@tool
def search(query: str) -> str:
    """ 
    This is a tool that search in the internet 
    Args:
        query: The query to search for 
    Returns:
        The search result for the query
    """

    print("==>searching for  query: ", query)
    return  tavily.search(query=query)

llm = ChatOpenAI(model="gpt-5", temperature=0)

# commenting the custom search tool 
# tools = [search]

# using out the box tavily search tool TavilyRearch 
tools = [TavilySearch()]


agent = create_agent(model=llm, tools=tools)

def main():
    print("Main functions is saying Hello from langchain-course!")
    # the messages are meant to be a list but it is ok to pass one single message , langchain will go the convresion
    result = agent.invoke({"messages": HumanMessage(content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list thier details ")})
    print(result)
if __name__ == "__main__":
    main()
