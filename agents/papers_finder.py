from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain_community.tools import ArxivQueryRun, DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama  # Updated import


def create_research_agent(llm):
    """
    Creates a research agent equipped with Arxiv and DuckDuckGo search tools.

    Args:
        llm: An instance of a LangChain language model.

    Returns:
        An AgentExecutor ready to process research queries.
    """
    # --- 2. Define the Tools ---
    # The agent will have access to two tools:
    # - Arxiv: For searching academic papers specifically on arxiv.org.
    # - DuckDuckGoSearch: For general web searches to find papers on other sites
    #   or to get a broader context.
    arxiv_tool = ArxivQueryRun()
    ddg_search_tool = DuckDuckGoSearchRun()

    tools = [
        Tool(
            name="arxiv_search",
            func=arxiv_tool.run,
            description="Use this tool to search for research papers on arXiv.org. The input should be a search query."
        ),
        Tool(
            name="web_search",
            func=ddg_search_tool.run,
            description="Use this tool for general web searches to find research papers or articles not available on arXiv."
        ),
    ]

    # --- 3. Create the Agent's Prompt ---
    # This prompt acts as the agent's "brain," instructing it on its role,
    # how to use its tools, and how to format its final answer.
    # The ReAct (Reason+Act) framework is used here, which is excellent for
    # tool-using agents.
    prompt_template = """
    You are a diligent and expert research assistant. Your goal is to find relevant research papers on a given topic.
    You have access to two tools: an arXiv search tool and a general web search tool.

    To answer the user's request, you must follow these steps:
    1.  Start by using the `arxiv_search` tool to see if you can find papers on the primary repository.
    2.  If the arXiv search is not sufficient or you need broader context, use the `web_search` tool.
    3.  Analyze the results from your tools.
    4.  When you have found enough information and are confident in your findings, provide a final answer.
    5.  Your final answer should be a formatted list of the papers you found, including their titles and links.

    Here is the user's request:
    {input}

    Begin!

    {agent_scratchpad}
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)

    # --- 4. Create the Agent ---
    # We use the create_react_agent function, which creates an agent that
    # uses the ReAct framework to decide which tool to use based on the
    # user's input and its previous actions.
    agent = create_react_agent(llm, tools, prompt)

    # --- 5. Create the Agent Executor ---
    # The AgentExecutor is the runtime for the agent. It's what actually
    # calls the agent, executes the chosen tools, and passes the results
    # back to the agent for the next reasoning step.
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,  # Set to True to see the agent's thought process
        handle_parsing_errors=True  # Helps with robustness
    )

    return agent_executor


if __name__ == '__main__':
    # --- Updated LLM Initialization ---
    # Ensure Ollama is running in the background with the Llama 3.1 8B Instruct model.
    # To pull the model, run: `ollama pull llama3.1:8b-instruct` in your terminal.
    # The model name "llama3.1:8b-instruct" is used here.
    llm = ChatOllama(model="gemma3:12b", temperature=0)

    # Create the research agent
    research_agent = create_research_agent(llm)

    # --- 6. Run the Agent ---
    # Now, you can ask the agent to find papers on any topic.
    topic = "multi-agent systems with orchestrators"
    response = research_agent.invoke({"input": topic})

    print("\n--- Final Answer ---")
    print(response["output"])
