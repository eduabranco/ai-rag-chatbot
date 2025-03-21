from crewai import Agent, Task, Crew

def create_crew(query, context):
    researcher = Agent(
        role="Pesquisador Sênior",
        goal="Analisar profundamente o contexto fornecido e identificar informações relevantes",
        backstory="Especialista em análise crítica e síntese de informações complexas",
        verbose=True  # Alterado para booleano
    )

    formulator = Agent(
        role="Especialista em Respostas",
        goal="Gerar respostas claras e contextualizadas baseadas nas análises do pesquisador",
        backstory="Redator técnico com expertise em comunicação clara e eficaz",
        verbose=True  # Alterado para booleano
    )

    research_task = Task(
        description=f"Analisar o seguinte contexto e extrair insights relevantes para: {query}\nContexto: {context}",
        agent=researcher,
        expected_output="Lista de insights relevantes com citações precisas"
    )

    formulation_task = Task(
        description="Transformar os insights em uma resposta bem estruturada e natural",
        agent=formulator,
        expected_output="Resposta final formatada em markdown com seções claras"
    )

    return Crew(
        agents=[researcher, formulator],
        tasks=[research_task, formulation_task],
        verbose=True  # Alterado para booleano
    )