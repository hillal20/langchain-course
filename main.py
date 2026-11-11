import os

os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

load_dotenv()


def main():
    print("Hello from langchain-course!")

    information = """
     Elon Musk is a South African-born billionaire entrepreneur and business magnate who has become one of the most influential and controversial figures of the modern era. He is the CEO and chief architect of Tesla, the electric vehicle and clean energy company that revolutionized the automotive industry, and the founder and CEO of SpaceX, which has made significant strides in reusable rocket technology and space exploration. Musk also owns the social media platform X (formerly Twitter), founded the artificial intelligence company xAI, and co-founded Neuralink and The Boring Company. Known for his ambitious vision—including making humanity a multi-planetary species and accelerating the world's transition to sustainable energy—Musk is equally known for his provocative social media presence, bold predictions, and polarizing public persona.
    """
    summary_template = f"""
    given the informtation {information} about a person I want you to create:
    1. A short summary
    2. two iteresting facts about them
     """
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(temperature=0, model="gpt-5")
    # llm = ChatOllama(temperature=0, model="gemma3:270m")
    # the chain is going to get the ouput from the left component and pass it ans input for the left component
    chain = summary_prompt_template | llm
    respose = chain.invoke(input={"informtaion": information})
    print("result ===> ", respose.content)


if __name__ == "__main__":
    main()
