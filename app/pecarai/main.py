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
from dotenv import load_dotenv
from agent_prompt import get_agent_prompt
from memory.session import get_memory_session_manager


app = BedrockAgentCoreApp()
log = app.logger

load_dotenv(Path(__file__).resolve().with_name(".env"), override=False)

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
MODEL_LITE = os.environ.get("MODEL_LITE", "")                    # Fast, lightweight
MODEL_MEDIUM = os.environ.get("MODEL_MEDIUM", "")     # Deep reasoning
MODEL_PRO = os.environ.get("MODEL_PRO", "")  
KB_ID = os.environ.get("KB_ID", "")  
log.info(
    "Agent configuration: AWS_REGION=%s, MODEL_LITE=%s, MODEL_MEDIUM=%s, MODEL_PRO=%s, KB_ID=%s",
    AWS_REGION, MODEL_LITE, MODEL_MEDIUM, MODEL_PRO, KB_ID,
)

missing_env_vars = [
    name for name, value in (
        ("MODEL_LITE", MODEL_LITE),
        ("MODEL_MEDIUM", MODEL_MEDIUM),
        ("KB_ID", KB_ID),
    ) if not value.strip()
]
if missing_env_vars:
    message = f"Missing required environment variables: {', '.join(missing_env_vars)}"
    log.error(message)
    raise ValueError(message)

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
    session_id = getattr(context, 'session_id', 'default-session')
    user_id = getattr(context, 'user_id', 'default-user')
    log.info("Invoking Agent.....")

    user_message = payload.get("prompt", "Hello!")

    system_prompt = get_agent_prompt("Chiazokam")

    agent = Agent(
        model=MODEL_LITE,
        session_manager=get_memory_session_manager(session_id, user_id),
        system_prompt=system_prompt,
        tools=[classification_intent_agent],
    )

    response = agent(user_message)
    return response


if __name__ == "__main__":
    app.run()
