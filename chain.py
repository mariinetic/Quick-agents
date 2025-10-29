from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")

def main():
    information = """
    William Henry Gates III (born October 28, 1955, in Seattle, Washington) is an American businessman, philanthropist, and author, best known as the co-founder of Microsoft Corporation, one of the largest and most influential technology companies in the world.
    Gates began programming as a teenager and, together with his friend Paul Allen, founded Microsoft in 1975. The company became famous for developing the MS-DOS operating system and later Windows, which went on to become the most widely used operating system for personal computers.

    Throughout his career, Gates became one of the richest men on the planet, often topping the Forbes list of billionaires. He stepped down as Microsoft’s CEO in 2000 and has since dedicated himself primarily to philanthropy.
    Alongside his then-wife, Melinda, he created the Bill & Melinda Gates Foundation, one of the largest charitable foundations in the world, focused on fighting disease, reducing global poverty, and expanding access to education and technology.
    """

    summary_template = """
    Given the information {information} about a person, I want you to create:
    1. A short summary
    2. Two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template
    )

    llm = ChatOpenAI(
        model="gpt-4o",
        base_url=BASE_URL,
        api_key= OPENAI_API_KEY,
        temperature=0,
        max_tokens=4096
    )

    chain = summary_prompt_template | llm

    response = chain.invoke({"information": information})
    print(response.content)

if __name__ == "__main__":
    main()
