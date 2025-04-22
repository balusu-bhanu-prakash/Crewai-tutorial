from crewai import Crew, Task, Agent
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import asyncio
import streamlit as st
from crewai_tools import SerperDevTool

print(os.getenv("GEMINI_API_KEY"))


def create_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-8b",
        temperature=0.1,
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )


url = st.sidebar.text_input(
    "Your reposirtory url",
    "https://github.com/mertakdut/Spring-Boot-Sample-Project.git",
)

try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

llm = create_llm()

tool = SerperDevTool()

repo_digester = Agent(
    role="Software Developer",
    goal=f"Understand the core functionalities of the project at {url}",
    backstory="An experienced software developer wtih experience in Java",
    llm=llm,
)

pomxml_contents = Task(
    description=f"Go to the {url+'/blob/master/pom.xml'} file in this project and fetch its dependencies with versions",
    agent=repo_digester,  # Assigning the agent
    llm=llm,
    expected_output="A list of all the dependencies with the versions used for this project",
)

crew = Crew(agents=[repo_digester], tasks=[pomxml_contents], verbose=1)

print(crew.kickoff())
