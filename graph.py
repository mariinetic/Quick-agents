import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")

llm = ChatOpenAI(
        model="gpt-4o",
        base_url= BASE_URL,
        api_key= OPENAI_API_KEY,
        temperature=0,
        max_tokens=4096
    )
class State(dict):
    text: str
    translator: str

def translator(state):
    text = state["text"]
    response = llm.invoke(f"translate this text to portugues: {text}")
    return {"translator": response.content}

workflow = StateGraph(State)
workflow.add_node("translator", translator)
workflow.add_edge("translator", END)
workflow.set_entry_point("translator")

app = workflow.compile()

response = app.invoke({"text": "I like to share what i learn."})
print(response)
