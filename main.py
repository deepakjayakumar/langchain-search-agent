from dotenv import load_dotenv
load_dotenv()

from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

tools = [TavilySearch()]
llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
react_prompt = hub.pull("hwchase17/react")
react_prompt.template += "\nVERY IMPORTANT: You must follow the format strictly. Do not include any text outside of the Thought/Action/Action Input/Observation blocks."
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt
)


agent_executor = AgentExecutor(agent=agent,tools=tools,verbose=True )
chain = agent_executor

def main():
    result = chain.invoke(
        input={
            "input":"search for 3 job posting for an ai engineer using langchain in chennai or bengaluru and list their details"
        }
    )
    print(result)


if __name__ == "__main__":
    main()
