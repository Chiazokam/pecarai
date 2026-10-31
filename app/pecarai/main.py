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
NOVA_LITE_MODEL = os.environ.get("NOVA_LITE_MODEL", "amazon.nova-lite-v1:0")                    # Fast, lightweight
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")     # Deep reasoning
NOVA_PRO_MODEL = os.environ.get("NOVA_PRO_MODEL", "amazon.nova-pro-v1:0")  
KB_ID = os.environ.get("KB_ID", "")  

model = BedrockModel(
    model_id=CLAUDE_MODEL,
    region_name=AWS_REGION,
    temperature=0.0,
)

_bedrock_runtime = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)

@tool
def classification_intent_agent(query: str) -> str:
    # """Classify each call to this agent using Nova Lite. Classify to either of two types chat or feedback"""
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


    # session_id = getattr(context, 'session_id', 'default-session')
    # agent = get_or_create_agent(session_id)

    # prompt = _extract_prompt(payload)


    # async for event in agent.stream_async(
    #     prompt, tools=[classification_intent_agent]
    # ):
    #     if not isinstance(event, dict) or "event" not in event:
    #         continue
    #     cbs = event["event"].get("contentBlockStart")
    #     if cbs is not None and not cbs.get("start"):
    #         continue
    #     yield event
    print(payload, "<<<<<<")

    user_message = payload.get("prompt", "Hello!")

    # logger.info("KB: %s | User: %s", KB_ID, user_message[:80])

    # system_prompt = """You are a classification agent. Your ONLY job:
    #     Classify the intent of the user based on the user prompt to either of these:
    #     - CHAT: When the prompt is asking about the user or wants to know more about the user, projects the user 
    #         has worked on, places the user has worked.
    #     - FEEDBACK: When the user needs to receive an update on what is going on in his/her portfolio, like page visits, 
    #         or update on questions being asked on the chat
    #     Your response should be a json object with the following fields:
    #     {"intent": <CHAT|FEEDBACK>,
    #     "language": <the language the prompt was sent in>}
    #     Do NOT add any other commentary."""

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
