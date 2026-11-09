import os
import logging
from hashlib import sha256
from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig
from bedrock_agentcore.memory.integrations.strands.session_manager import AgentCoreMemorySessionManager

log = logging.getLogger(__name__)


def get_memory_session_manager(session_id: str) -> AgentCoreMemorySessionManager:
    memory_id = os.getenv("MEMORY_ID", "").strip()
    if not memory_id:
        message = "MEMORY_ID is required to persist conversation memory"
        log.error(message)
        raise ValueError(message)

    if not isinstance(session_id, str) or not session_id.strip():
        message = "session_id is required and must be a non-empty string"
        log.error(message)
        raise ValueError(message)

    # Runtime session IDs can exceed Memory's 100-character limit. Each conversation
    # gets its own actor as well, so visitor facts cannot cross conversation boundaries.
    memory_session_id = f"conversation-{sha256(session_id.encode('utf-8')).hexdigest()}"

    return AgentCoreMemorySessionManager(
        AgentCoreMemoryConfig(
            memory_id=memory_id,
            session_id=memory_session_id,
            actor_id=memory_session_id,
            batch_size=1,
        ),
        region_name=os.getenv("AWS_REGION") or "us-east-1",
    )
