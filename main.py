import os
from dotenv import load_dotenv
from fastmcp import FastMCP
from typing import Literal
from pydantic import Field
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults

# All workflows
from workflows.workflow_source_config import workflow

# All schemas
from schema.States.workflow_source_config_state import SourceConfigListStruction

# Load Environment Variables
load_dotenv()

mcp = FastMCP("velox_mcp_server")

# Demo Tool, Will be removed before deployment
@mcp.tool(
        name='demo_tool', 
        description='This is a demo tool for project inital deployment.'
)
def demo_tool() -> str:
    return "This is demo tool"


@mcp.tool(
    name='rank_sources_for_query_and_return_context',
    description='This tool helps you find the most relevant information for any query. It looks at multiple sources, figures out which ones matter most, and decides how many top documents to pull from each. Then, it fetches these documents from a vector database and organizes them into a clear, structured context. This context can be used by other models or applications to provide accurate and well-informed answers. By focusing on relevance and source summaries, the tool ensures you get the right information quickly and efficiently.'
)
def rank_sources_for_query_and_return_context(
    query: str,
    userID: str,
    notebookID: str
) -> str:
    """
    - It looks at the user's query and figures out which sources are most relevant, then creates a list of (source_id, top_k) pairs.

    - Each pair tells the system which sources to check and how many documents to pull from each one, helping it gather and organize the most useful information efficiently.

    - After that a function will fetch number of documents for every source based on user query.

    - Lastly return required context.  
    """

    # Define initial state for our MCP workflow
    initial_state = {
        "query": query,
        "indexID": userID,
        "notebookID": notebookID,
        "listOfSummaries": [],
        "sourceConfig": SourceConfigListStruction(items=[]),
        "context": ""
    }

    # Invoke our workflow asynchronously
    response = workflow.invoke(
        initial_state
    )

    # return response
    return response.get("context", "")


@mcp.tool(
        name="search_tool",
        description="Search the internet for different types of content: news, or general results."
)
def search_tool(
    query: str = Field(
        ...,
        description="The text that the user wants to search for."
    ),
    type: Literal["news", "search"] = Field(
        "search",
        description=(
            "The type of content to search. Must be one of: "
            "'news' or 'search' "
            "Defaults to 'search'."
        )
    )
) -> str:
    """
    Input:
        - Query: User actual query to search. Must be a string.
        - Type: Type of the query must be one of 'news' or 'search'.

    Output:
        - String that contains context based on user's query from both search engine [Google Serper & DuckDuckGo].

    Fetch context from both engines ensuring if one will fail to respond then another will hendle.
    Search user query using google serper tool and duckduckgo which fetches top information from the google and duckduckgo search engine. 
    """

    # Final Context String
    final_context = ""

    # Google Serper Search Engine
    engine = GoogleSerperAPIWrapper(type=type)
    
    # Fetch context from google serper
    try: 
        results = engine.run(query)
        final_context += results
        final_context += "\n\n\n"
    except:
        final_context += "Failed to fetch context with google serper.\n\n"
    
    wrapper = DuckDuckGoSearchAPIWrapper(max_results=5, source=type)
    ddg_engine = DuckDuckGoSearchResults(api_wrapper=wrapper, source=type)

    # Fetch context from duckduckgo
    try:
        results = ddg_engine.invoke(query)
        final_context += results
    except:
        final_context += "Failed to fetch context with duckduckgo."

    return final_context


if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 3000))
    )