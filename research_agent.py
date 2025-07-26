from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama

from helper_agents.papers_finder import FinderAgent
from helper_agents.summarizer import SummarizerAgent


def create_orchestrator_agent(llm):
    """
    Creates the master orchestrator agent that can delegate tasks to the
    researcher and summarizer helper_agents.
    """
    # 1. Instantiate the sub-helper_agents (which will become tools)
    research_agent_executor = FinderAgent.create_finder_agent(llm)
    summarizer_agent_executor = SummarizerAgent.create_summarizer_agent(llm)

    # 2. Create tools for the orchestrator to use
    tools = [
        Tool(
            name="research_paper_finder",
            # MODIFICATION: Wrap invoke to return only the output string
            func=lambda query: research_agent_executor.invoke({"input": query})["output"],
            description="""
            Use this tool to find research papers on a specific academic topic.
            The input should be a clear, concise research topic (e.g., 'quantum computing advancements').
            This tool will return a list of relevant papers with their titles and URLs.
            """
        ),
        Tool(
            name="paper_summarizer",
            # MODIFICATION: Wrap invoke to return only the output string
            func=lambda query: summarizer_agent_executor.invoke({"input": query})["output"],
            description="""
            Use this tool to summarize a specific research paper.
            The input must be a string containing the paper's title and its URL,
            like this: 'Summarize the paper titled "Paper Title" from the URL: http://example.com/paper.pdf'
            """
        )
    ]

    # 3. Create the prompt for the orchestrator
    prompt_template = """
    You are an expert orchestrator agent. Your job is to understand a user's request and delegate tasks to specialized helper_agents.
    You have two helper_agents at your disposal: a 'research_paper_finder' and a 'paper_summarizer'.

    - If the user wants to find papers, use the `research_paper_finder`.
    - If the user wants to summarize a paper, use the `paper_summarizer`.
    - If the user asks for a multi-step task (e.g., find and then summarize), you must perform the steps sequentially.
      First, call the `research_paper_finder`, then use its output to call the `paper_summarizer`.

    You have access to the following tools:
    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do. This is your reasoning step.
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought: {agent_scratchpad}
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)

    # 4. Create the orchestrator agent
    orchestrator_agent = create_react_agent(llm, tools, prompt)

    # 5. Create the agent executor
    agent_executor = AgentExecutor(
        agent=orchestrator_agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    return agent_executor

if __name__ == '__main__':
    # Initialize the LLM
    llm = ChatOllama(model="gemma3:12b", temperature=0)

    # Create the orchestrator
    orchestrator = create_orchestrator_agent(llm)

    # Define a complex, multi-step task for the orchestrator
    task = """
    First, find research papers about 'multi-agent systems with orchestrators'.
    Then, take the first paper you find and summarize it.
    """

    print(f"--- Starting Orchestrator with task: --- \n{task}")

    # Run the orchestrator
    response = orchestrator.invoke({"input": task})

    print("\n--- Orchestrator Final Answer ---")
    print(response["output"])