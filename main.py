import os
from dotenv import load_dotenv
from fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("velox MCP Server")

@mcp.tool(name='demo_tool', description='This is a demo tool for project inital deployment.')
def demo_tool():
    return "This is demo tool"

# for deployment
# uv sync --frozen

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=os.getenv("PORT"))