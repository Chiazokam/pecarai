import asyncio
import importlib
import os
import unittest
from unittest.mock import Mock, patch

from bedrock_agentcore.memory.integrations.strands import session_manager as sdk_memory
from bedrock_agentcore.runtime import RequestContext
from strands import Agent
from strands.models.model import Model

from agent_prompt import get_agent_prompt
from memory.session import get_memory_session_manager


class MemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (
            patch.dict(os.environ, {
                "MODEL_LITE": "test-model",
                "MODEL_MEDIUM": "test-model",
                "KB_ID": "test-kb",
            }),
            patch("dotenv.load_dotenv"),
            patch("boto3.client"),
            patch("strands.models.BedrockModel"),
        ):
            cls.main = importlib.import_module("main")

    def setUp(self):
        self.env = patch.dict(os.environ, {
            "MEMORY_ID": "testMemory-1234567890",
            "AWS_REGION": "us-east-1",
        })
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_environment_is_read_at_call_time(self):
        with patch("memory.session.AgentCoreMemorySessionManager") as factory:
            os.environ["MEMORY_ID"] = "anotherMemory-1234567890"
            os.environ["AWS_REGION"] = "eu-west-1"
            get_memory_session_manager("conversation-1")
            config = factory.call_args.args[0]
            self.assertEqual(config.memory_id, "anotherMemory-1234567890")
            self.assertEqual(factory.call_args.kwargs["region_name"], "eu-west-1")
            self.assertIsNone(config.retrieval_config)
            self.assertEqual(config.batch_size, 1)

    def test_missing_memory_is_an_explicit_error(self):
        for value in ("", "   "):
            with self.subTest(value=value), patch.dict(os.environ, {"MEMORY_ID": value}):
                with self.assertLogs("memory.session", level="ERROR"):
                    with self.assertRaisesRegex(ValueError, "MEMORY_ID"):
                        get_memory_session_manager("conversation-1")

    def test_memory_ids_are_stable_valid_and_isolated(self):
        with patch("memory.session.AgentCoreMemorySessionManager") as factory:
            configs = []
            for session_id in ("conversation-1", "conversation-1", "conversation-2", "x" * 256):
                get_memory_session_manager(session_id)
                configs.append(factory.call_args.args[0])
            self.assertEqual(configs[0].session_id, configs[1].session_id)
            self.assertNotEqual(configs[0].actor_id, configs[2].actor_id)
            for config in configs:
                self.assertLessEqual(len(config.session_id), 100)
                self.assertRegex(config.session_id, r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")
                self.assertEqual(config.session_id, config.actor_id)
            os.environ["AWS_REGION"] = ""
            get_memory_session_manager("conversation-1")
            self.assertEqual(factory.call_args.kwargs["region_name"], "us-east-1")

    def test_context_and_payload_session_handling(self):
        with (
            patch.object(self.main, "get_memory_session_manager") as factory,
            patch.object(self.main, "Agent") as agent_factory,
        ):
            manager = factory.return_value.__enter__.return_value
            agent_factory.return_value.return_value = "response"
            for payload, context in (
                ({"prompt": "Hi"}, RequestContext(session_id="conversation-1")),
                ({"session_id": "conversation-1"}, RequestContext()),
                ({"session_id": "conversation-1"}, None),
                ({"session_id": "conversation-1"}, RequestContext(session_id="conversation-1")),
            ):
                with self.subTest(payload=payload, context=context):
                    self.assertEqual(asyncio.run(self.main.invoke(payload, context)), "response")
                    factory.assert_called_with("conversation-1")
                    self.assertIs(agent_factory.call_args.kwargs["session_manager"], manager)
                    agent_factory.return_value.assert_called_with(payload.get("prompt", "Hello!"))
            self.assertEqual(factory.return_value.__exit__.call_count, 4)

    def test_invalid_invocations_do_not_access_memory(self):
        with patch.object(self.main, "get_memory_session_manager") as factory:
            for payload, context in (
                (None, None),
                ([], None),
                ({}, RequestContext()),
                ({"session_id": ""}, None),
                ({"session_id": "  "}, None),
                ({"session_id": 12}, None),
                ({"prompt": [], "session_id": "conversation-1"}, None),
                ({"prompt": None, "session_id": "conversation-1"}, None),
                ({"session_id": "conversation-2"}, RequestContext(session_id="conversation-1")),
            ):
                with self.subTest(payload=payload), self.assertLogs(self.main.log, level="ERROR"):
                    with self.assertRaises(ValueError):
                        asyncio.run(self.main.invoke(payload, context))
            factory.assert_not_called()

    def test_manager_is_closed_when_agent_fails(self):
        with (
            patch.object(self.main, "get_memory_session_manager") as factory,
            patch.object(self.main, "Agent") as agent_factory,
        ):
            agent_factory.return_value.side_effect = RuntimeError("model failed")
            with self.assertRaisesRegex(RuntimeError, "model failed"):
                asyncio.run(self.main.invoke({"session_id": "conversation-1"}, None))
            self.assertIs(factory.return_value.__exit__.call_args.args[0], RuntimeError)

    def test_prompt_distinguishes_visitor_from_portfolio_owner(self):
        prompt = get_agent_prompt("Portfolio Owner")
        self.assertIn("visitor-provided details, including their name", prompt)
        self.assertIn("Do not search the portfolio knowledge base for the visitor's name", prompt)
        self.assertIn("not\n   necessarily Portfolio Owner", prompt)

    def test_real_sdk_restores_history_across_fresh_agents(self):
        events = []
        service = Mock()
        client = Mock()
        boto_session = Mock(region_name="us-east-1")
        boto_session.client.return_value = service

        def create_event(**kwargs):
            event = dict(kwargs, eventId=f"event-{len(events)}")
            events.append(event)
            return {"event": event}

        def create_conversation_event(
            memory_id, actor_id, session_id, messages, event_timestamp, metadata
        ):
            response = create_event(
                memoryId=memory_id,
                actorId=actor_id,
                sessionId=session_id,
                eventTimestamp=event_timestamp,
                metadata=metadata,
                payload=[
                    {"conversational": {"role": role.upper(), "content": {"text": text}}}
                    for text, role in messages
                ],
            )
            return response["event"]

        def list_events(memory_id, actor_id, session_id, event_metadata=None, max_results=100):
            matching = [
                event for event in reversed(events)
                if event["memoryId"] == memory_id
                and event["actorId"] == actor_id
                and event["sessionId"] == session_id
                and all(
                    (event.get("metadata") or {}).get(item["left"]["metadataKey"])
                    == item["right"]["metadataValue"]
                    for item in event_metadata or []
                )
            ]
            return matching[:max_results]

        service.create_event.side_effect = create_event
        client.create_event.side_effect = create_conversation_event
        client.list_events.side_effect = list_events
        model = Mock(spec=Model)
        model.stateful = False
        model.get_config.return_value = {"model_id": "test-model"}
        histories = []

        async def stream(messages, tool_specs=None, system_prompt=None, **kwargs):
            histories.append([
                block["text"]
                for message in messages
                for block in message["content"]
                if "text" in block
            ])
            yield {"messageStart": {"role": "assistant"}}
            yield {"contentBlockDelta": {"delta": {"text": "Acknowledged."}}}
            yield {"contentBlockStop": {"contentBlockIndex": 0}}
            yield {"messageStop": {"stopReason": "end_turn"}}

        model.stream.side_effect = stream
        with (
            patch.object(sdk_memory, "MemoryClient", return_value=client),
            patch.object(sdk_memory.boto3, "Session", return_value=boto_session),
            patch.object(self.main, "Agent", side_effect=lambda **kwargs: Agent(
                **dict(kwargs, model=model, callback_handler=None)
            )),
        ):
            for prompt, session_id in (
                ("My name is Jane.", "conversation-1"),
                ("What is my name?", "conversation-1"),
                ("What is my name?", "conversation-2"),
            ):
                asyncio.run(self.main.invoke({"prompt": prompt, "session_id": session_id}, None))

        self.assertIn("My name is Jane.", histories[1])
        self.assertIn("Acknowledged.", histories[1])
        self.assertEqual(histories[2], ["What is my name?"])
        client.retrieve_memories.assert_not_called()
        self.assertTrue(events)


if __name__ == "__main__":
    unittest.main()
