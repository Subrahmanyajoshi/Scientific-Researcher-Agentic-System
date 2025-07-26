from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader


class SummarizerAgent:

    @staticmethod
    def get_webpage_content(url: str) -> str:
        """
        Fetches the text content of a webpage given its URL.
        It cleans up whitespace and truncates the content to a manageable size for the LLM.
        """
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            content = " ".join([doc.page_content for doc in docs])
            # Clean up excessive whitespace and limit length to avoid overwhelming the context window
            cleaned_content = " ".join(content.split())
            max_chars = 15000
            return cleaned_content[:max_chars]
        except Exception as e:
            return f"Error fetching content from {url}: {e}"

    @staticmethod
    def create_summarizer_agent(llm):
        """
        Creates a summarization agent that can read a webpage and summarize it.
        """
        web_fetch_tool = Tool(
            name="web_content_fetcher",
            func=SummarizerAgent.get_webpage_content,
            description="Use this tool to fetch the text content of a webpage given its URL. The input must be a single URL."
        )

        tools = [web_fetch_tool]

        prompt_template = """
        You are an expert academic summarizer. Your goal is to provide a concise summary of a research paper.
    
        You have access to the following tool:
        {tools}
    
        The user will provide the paper's title and a URL. Follow these steps:
        1. Use the `web_content_fetcher` tool with the provided URL to get the paper's text content.
        2. Read the content and create a summary that includes the main objective, methodology, and key findings of the paper.
        3. Present the summary clearly.
    
        Use the following format:
    
        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: A summary of the paper.
    
        Begin!
    
        Question: {input}
        Thought: {agent_scratchpad}
        """

        prompt = ChatPromptTemplate.from_template(prompt_template)
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True
        )
        return agent_executor