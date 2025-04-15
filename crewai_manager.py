from crewai import Agent, Task, Crew

def create_crew(query, context):
    researcher = Agent(
        role="Senior Researcher",
        goal="Deeply analyze the provided context and identify relevant information",
        backstory="Expert in critical analysis and synthesis of complex information",
        verbose=True
    )

    formulator = Agent(
        role="Answer Specialist",
        goal="Generate clear and contextualized answers based on the researcher's analysis",
        backstory="Technical writer with expertise in clear and effective communication",
        verbose=True
    )

    research_task = Task(
        description=f"Analyze the following context and extract relevant insights for: {query}\nContext: {context}",
        agent=researcher,
        expected_output="List of relevant insights with accurate citations"
    )

    formulation_task = Task(
        description="Transform insights into a well-structured and natural response",
        agent=formulator,
        expected_output="Final response formatted in markdown with clear sections"
    )

    return Crew(
        agents=[researcher, formulator],
        tasks=[research_task, formulation_task],
        verbose=True
    )