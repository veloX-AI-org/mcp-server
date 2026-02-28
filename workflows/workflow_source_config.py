import os
from dotenv import load_dotenv
import requests

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from pinecone import Pinecone
from typing import List
from langgraph.graph import StateGraph, START, END

# Import STATE
from schema.States.workflow_source_config_state import RetrievalState, SourceConfigListStruction, SubState

load_dotenv()

# Verified
def post_request(state: RetrievalState) -> RetrievalState:
    """
    The first graph node after start node.

    Function: Takes secrete token and other user information to get summary of all the sources user's have.
    """

    # End point
    url = "https://veloxai.onrender.com/getAllDocSummary"

    # Prepare all the input to post
    header_inputs = {
        "userID": state.indexID,
        "notebookID": state.notebookID,
        "SECRETE_TOKEN": os.getenv("SECRETE_TOKEN")
    }

    # Post request
    response = requests.post(
        url=url,
        json=header_inputs
    )

    # Handle if response failed
    if not response:
        state.listOfSummaries =  [SubState(source_type="Unknown", source_id="Failed get ID.", source_name="Failed to get summary of every docs.", source_summary="Failed to get summary")]

        return state
    
    allDocsSummary = response.json().get("message")
    
    # Validate Summaries
    valid_docs_summary: List[SubState] = [SubState.model_validate(item) for item in allDocsSummary]
    
    state.listOfSummaries = valid_docs_summary

    return state

# Verified
def get_weighted(state: RetrievalState):
    """
    Get weights of every source
    """

    template = """
        You are a document routing assistant.

        Input:
        - A user query.
        - A list of sources in the format:
        `{{ source_type: Literal["URL", "Document"], source_id: str, source_name: str, source_summary: str }}`

        Task:
        1. Analyze the user query.
        2. Determine which sources are relevant based on their summaries.
        3. For each relevant source, assign a `top_k` value between 2 and 10:
        * Use a higher value (7-10) if the source is highly relevant.
        * Use a lower value (2-4 or 2-5) if the source is moderately relevant.
        4. Return ONLY a JSON array of objects

        Rules:
        * Do not include irrelevant sources.
        * Do not return explanations.
        * Output must be valid list of format instruction for every summary.

        Summary of each document:
        {listOfSummaries}

        User Query:
        {query}
    """

    model = ChatOpenAI(
        model='gpt-5-nano'
    )
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["query", "listOfSummaries"]
    )

    structuredOutput = model.with_structured_output(SourceConfigListStruction)

    chain = prompt | structuredOutput

    response = chain.invoke({
        "query": state.query,
        "listOfSummaries": state.listOfSummaries
    })
    
    return {
        "sourceConfig": response
    }

# Verified
def retrieve_docs_by_source(state: RetrievalState) -> RetrievalState:
    """
    After ranking fetch all top_k documents for each source 
    """
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(state.indexID)
    
    final_context = ""

    model = OpenAIEmbeddings(
        model="text-embedding-3-small"  
    )

    query_embedding = model.embed_query(state.query)
    
    for configs in state.sourceConfig.items:
        results = index.query(
            vector=query_embedding,
            top_k=configs.top_k,
            filter = {'source_urlID': configs.source_id} if configs.source_type == "URL" else {'source_key': configs.source_id},
            include_metadata=True
        )

        final_context += '\n'.join([docs['metadata']['text'] for docs in results.matches])
    
    state.context = final_context
    
    return state

graph = StateGraph(RetrievalState)

graph.add_node("postRequest", post_request)
graph.add_node("getWeights", get_weighted)
graph.add_node("getDocuments", retrieve_docs_by_source)

graph.add_edge(START, "postRequest")
graph.add_edge("postRequest", "getWeights")
graph.add_edge("getWeights", "getDocuments")
graph.add_edge("getDocuments", END)

workflow = graph.compile()