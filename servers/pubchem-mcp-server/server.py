import arxiv
import json
import os
from typing import List
from mcp.server.fastmcp import FastMCP
from python_version.mcp_server import handle_tool_call
mcp = FastMCP("pubchem", host="0.0.0.0",port=50001)
@mcp.tool()
def get_pubchem_data(query: str, format: str = 'JSON', include_3d: bool = False) -> str:
    """
    Get PubChem data for a given query.
    """
    return handle_tool_call("get_pubchem_data", {"query": query, "format": format, "include_3d": include_3d})

@mcp.tool()
def download_structure(cid: str, format: str = 'sdf') -> str:
    """
    Download a structure from PubChem.
    """
    return handle_tool_call("download_structure", {"cid": cid, "format": format})

if __name__ == "__main__":
    mcp.run(transport="sse")