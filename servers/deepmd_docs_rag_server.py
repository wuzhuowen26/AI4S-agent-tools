#%%

# server.py
from fastmcp import FastMCP, Client
from typing import List
import requests
import json
import os
from dotenv import load_dotenv

#%%

load_dotenv()
FASTGPT_AUTH_TOKEN = os.getenv("FASTGPT_AUTH_TOKEN")
# FASTGPT_AUTH_TOKEN = "fastgpt-xxxxxxxxxxxxxxxxxxxx"  # <--- substitute your token

#%%



backend = Client("https://api.fastgpt.in/api/mcp/app/wrSvRtDydWEb7faAP6uPooSX/mcp")
# backend = Client("https://mcp.tryfastgpt.ai/wrSvRtDydWEb7faAP6uPooSX/sse")


# Create a proxy server that bridges to the FastGPT backend
mcp = FastMCP.as_proxy(
    backend, 
    name="FastGPT Proxy Server",

)

#%%
# @mcp.tool()
# def chat(question: str):
#     """
#     Use the DeePMD-docs knowledge base to answer the question.
#     already implemented in the FastGPT backend.
#     The proxy server is used to bridge the FastGPT backend and the futurechat.
#     """
#     return "<deepmd-docs-chat-response>"

@mcp.tool()
def upload_single_file_to_deepmd_knowledge_base(file_url: str):
    """get the file from the specified URL and upload it DeePMD-docs knowledge database.
    These files will be used to train the DeePMD-docs knowledge base.
    And future chat will use these documents to answer questions.
    """

    API_URL = "https://api.fastgpt.in/api/core/dataset/collection/create/localFile"
    DATASET_ID = "683a517f292645ebd7a2cb7c"      #  knowledge base: deepmd-from-chat

    # 1. download the file from the specified URL
    file_name = file_url.split("/")[-1].replace(" ", "_")
    file_content = requests.get(file_url).content
    
    # 2. upload the file to the specified dataset
    response = requests.post(
        url=API_URL,
        headers={"Authorization": f"Bearer {FASTGPT_AUTH_TOKEN}"},
        files={
            'file': (file_name, file_content),
            'data': (None, json.dumps({"datasetId": DATASET_ID, "parentId": None, "trainingType": "chunk", "chunkSize": 512}))
        }
    )
    
    print(f"Status: {response.status_code}, Response: {response.text}")
    return response

#%%

# TEST_URL = "https://raw.githubusercontent.com/deepmodeling/deepmd-kit/r3.0/README.md"
# upload(TEST_URL)

#%%

if __name__ == "__main__":
    # mcp.run(transport='sse')
    mcp.run(transport='streamable-http', host="127.0.0.1", port=50001, path="/mcp")
    # mcp.run(transport='stdio')

#%%
