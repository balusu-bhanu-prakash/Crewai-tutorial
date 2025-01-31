from crewai import Crew, Task, Agent
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import asyncio
import streamlit as st

# Streamlit UI setup
st.title("❤️ AI Love Letter Generator")
st.sidebar.header("Love Configuration")

# Input fields
proposer_name = st.sidebar.text_input("Your Name", "John")
recipient_name = st.sidebar.text_input("Their Name", "Jane")
proposer_backstory = st.sidebar.text_area(
    "Your Personality", "Nervous but romantic introvert"
)
recipient_backstory = st.sidebar.text_area(
    "Their Personality", "Values small genuine gestures over grand romantic displays"
)
temperature = st.sidebar.slider("Creativity Level", 0.0, 1.0, 0.5)


# LLM initialization
def create_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-8b",
        temperature=temperature,
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )


try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

llm = create_llm()

# Agent definitions
proposer = Agent(
    role=f"Romantic Writer ({proposer_name})",
    goal=f"Write heartfelt letters FROM {proposer_name} TO {recipient_name}",
    backstory=proposer_backstory,
    llm=llm,
    constraints=[
        f"ALWAYS write from {proposer_name}'s perspective",
        "Never mention writing a letter itself",
        "Focus on specific shared memories",
    ],
)

analyst = Agent(
    role="Relationship Analyst",
    goal=f"Analyze text compatibility FOR {recipient_name}'s personality",
    backstory=f"Expert in {recipient_name}'s preferences: {recipient_backstory}",
    llm=llm,
    constraints=[
        "Never write any part of a letter",
        "Only provide bullet-point analysis",
        "Focus on emotional authenticity",
    ],
)

# Task pipeline
draft_task = Task(
    description=f"Write initial letter from {proposer_name} to {recipient_name}",
    agent=proposer,
    expected_output=f"300-character letter FROM {proposer_name} TO {recipient_name}",
    output_file="draft.txt",
)

analysis_task = Task(
    description=f"Analyze draft for alignment with {recipient_name}'s personality",
    agent=analyst,
    expected_output="3-5 bullet points of constructive feedback",
    context=[draft_task],
    output_file="analysis.txt",
)

final_letter_task = Task(
    description=f"Incorporate feedback to refine the letter, ensuring it aligns closely with the context and feels natural. Avoid over-polishing; focus on making it meaningful and relevant.",
    agent=proposer,
    expected_output=f"A 400-character letter FROM {proposer_name} TO {recipient_name} that is contextually rich, authentic, and matches the tone of the situation.",
    context=[analysis_task],
    output_file="final.txt",
)

# Crew setup
love_crew = Crew(
    agents=[proposer, analyst],
    tasks=[draft_task, analysis_task, final_letter_task],
    verbose=1,
)

# Generate button
if st.button("✨ Create Love Letter"):
    with st.spinner("Crafting your perfect message..."):
        try:
            # Execute workflow
            love_crew.kickoff()

            # Get final output
            final_letter = final_letter_task.output.raw_output

            # Validate sender orientation
            if f"From {proposer_name}" not in final_letter:
                final_letter = f"{final_letter}"

            # Display formatted letter
            st.subheader(f"From {proposer_name} to {recipient_name}")
            st.markdown(
                f"""
            <div style="
                background: #fff5f8;
                padding: 25px;
                border-radius: 15px;
                font-family: 'Georgia', serif;
                line-height: 1.8;
                color: #4a4a4a;
                white-space: pre-wrap;
            ">
            💌 {final_letter}
            </div>
            """,
                unsafe_allow_html=True,
            )

        except Exception as e:
            st.error(f"Couldn't create letter: {str(e)}")
