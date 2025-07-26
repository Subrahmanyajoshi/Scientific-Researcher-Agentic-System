# Scientific Researcher Agentic System

This repository implements a multi-agent system for an intelligent researcher tool. The system leverages a local Large Language Model (LLM) via Ollama to find and summarize academic papers in a sophisticated, multi-step workflow.

## 🏛️ Architecture

The system is built on a modular, hierarchical agent architecture, which makes it robust and easy to extend.

1.  **Researcher Agent**: This is the master agent and the main entry point of the application. It understands complex, multi-step user requests and intelligently delegates tasks to the appropriate specialized agent.

2.  **Finder Agent**: A specialized agent responsible for finding academic papers. It uses two tools:
    *   **ArXiv**: To search for papers on the `arxiv.org` repository.
    *   **DuckDuckGo Search**: For general web searches to find papers on other sites or to gather broader context.

3.  **Summarizer Agent**: A specialized agent designed to summarize a given paper. It uses a tool to:
    *   **Fetch Web Content**: Reads the text content from a paper's URL.
    *   **Summarize**: Generates a concise summary covering the paper's objectives, methodology, and key findings.

### Workflow Example
When given a task like "Find papers on multi-agent systems and summarize the first one," the workflow is as follows:
1. The **Orchestrator** receives the task.
2. It first calls the **Researcher Agent** with the topic "multi-agent systems."
3. The **Researcher Agent** returns a list of relevant papers.
4. The **Orchestrator** extracts the title and URL of the first paper from the list.
5. It then calls the **Summarizer Agent** with this information.
6. The **Summarizer Agent** fetches the paper's content and returns a summary.
7. Finally, the **Orchestrator** presents the summary as the final answer.

---