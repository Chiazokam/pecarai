import os
import boto3
from pathlib import Path
from typing import Any
from collections import OrderedDict
from strands import Agent, tool
from strands.agent.conversation_manager.null_conversation_manager import NullConversationManager
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from model.load import load_model
from mcp_client.client import get_streamable_http_mcp_client
from strands.models import BedrockModel


app = BedrockAgentCoreApp()
log = app.logger

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
MODEL_LITE = os.environ.get("MODEL_LITE", "")                    # Fast, lightweight
MODEL_MEDIUM = os.environ.get("MODEL_MEDIUM", "")     # Deep reasoning
MODEL_PRO = os.environ.get("MODEL_PRO", "")  
KB_ID = os.environ.get("KB_ID", "")  

model = BedrockModel(
    model_id=MODEL_MEDIUM,
    region_name=AWS_REGION,
    temperature=0.0,
)

_bedrock_runtime = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)

@tool
def classification_intent_agent(query: str) -> str:
    """
        Search the knowledge base for information about the user in question, and return the most relevant information.

        Args:
            query: The question or topic to search for
        Returns:
            Relevant information retrieved from the knowledge base
    """


    resp = _bedrock_runtime.retrieve(
        knowledgeBaseId=KB_ID,
        retrievalQuery={"text": query},
    )
    results = resp.get("retrievalResults", [])
    if not results:
        return f"No information found for: {query}"

    chunks = [r["content"]["text"] for r in results]
    return "\n---\n".join(chunks)


@app.entrypoint
async def invoke(payload, context):
    log.info("Invoking Agent.....")

    user_message = payload.get("prompt", "Hello!")

    system_prompt=Path(__file__).with_name("agent_prompt.txt").read_text(encoding="utf-8")

    agent = Agent(
        model=NOVA_LITE_MODEL,
        system_prompt=system_prompt,
        tools=[classification_intent_agent],
    )

    response = agent(user_message)
    return response


if __name__ == "__main__":
    app.run()
