import os

from pydantic import BaseModel, Field

os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch
from tavily import TavilyClient

tavily = TavilyClient()


class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL for of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field(description="The agent answer to the query")
    sources: list[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )


# @tool
# def search(query: str) -> str:
#     """
#     This is a tool that search in the internet
#     Args:
#         query: The query to search for
#     Returns:
#         The search result for the query
#     """

#     print("==>searching for  query: ", query)
#     return  tavily.search(query=query)

llm = ChatOpenAI(model="gpt-5", temperature=0)

# commenting the custom search tool
# tools = [search]

# using out the box tavily search tool TavilyRearch
tools = [TavilySearch()]


agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Main functions is saying Hello from langchain-course!")
    # the messages are meant to be a list but it is ok to pass one single message , langchain will go the convresion
    result = agent.invoke(
        {
            "messages": HumanMessage(
                content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list thier details "
            )
        }
    )
    print(result)


if __name__ == "__main__":
    main()
