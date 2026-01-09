from typing import Any
import pytest
import httpx
import json
import numpy as np
from uuid import uuid4
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, TextPart

import sys
from pathlib import Path

# Add src directory to path to import agent module
src_path = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(src_path))

from agent import Agent
from model import EvalRequest, InitPayload, ObservationPayload, ActionPayload, AckPayload


# A2A validation helpers - adapted from https://github.com/a2aproject/a2a-inspector/blob/main/backend/validators.py

def validate_agent_card(card_data: dict[str, Any]) -> list[str]:
    """Validate the structure and fields of an agent card."""
    errors: list[str] = []

    # Use a frozenset for efficient checking and to indicate immutability.
    required_fields = frozenset(
        [
            'name',
            'description',
            'url',
            'version',
            'capabilities',
            'defaultInputModes',
            'defaultOutputModes',
            'skills',
        ]
    )

    # Check for the presence of all required fields
    for field in required_fields:
        if field not in card_data:
            errors.append(f"Required field is missing: '{field}'.")

    # Check if 'url' is an absolute URL (basic check)
    if 'url' in card_data and not (
        card_data['url'].startswith('http://')
        or card_data['url'].startswith('https://')
    ):
        errors.append(
            "Field 'url' must be an absolute URL starting with http:// or https://."
        )

    # Check if capabilities is a dictionary
    if 'capabilities' in card_data and not isinstance(
        card_data['capabilities'], dict
    ):
        errors.append("Field 'capabilities' must be an object.")

    # Check if defaultInputModes and defaultOutputModes are arrays of strings
    for field in ['defaultInputModes', 'defaultOutputModes']:
        if field in card_data:
            if not isinstance(card_data[field], list):
                errors.append(f"Field '{field}' must be an array of strings.")
            elif not all(isinstance(item, str) for item in card_data[field]):
                errors.append(f"All items in '{field}' must be strings.")

    # Check skills array
    if 'skills' in card_data:
        if not isinstance(card_data['skills'], list):
            errors.append(
                "Field 'skills' must be an array of AgentSkill objects."
            )
        elif not card_data['skills']:
            errors.append(
                "Field 'skills' array is empty. Agent must have at least one skill if it performs actions."
            )

    return errors


def _validate_task(data: dict[str, Any]) -> list[str]:
    errors = []
    if 'id' not in data:
        errors.append("Task object missing required field: 'id'.")
    if 'status' not in data or 'state' not in data.get('status', {}):
        errors.append("Task object missing required field: 'status.state'.")
    return errors


def _validate_status_update(data: dict[str, Any]) -> list[str]:
    errors = []
    if 'status' not in data or 'state' not in data.get('status', {}):
        errors.append(
            "StatusUpdate object missing required field: 'status.state'."
        )
    return errors


def _validate_artifact_update(data: dict[str, Any]) -> list[str]:
    errors = []
    if 'artifact' not in data:
        errors.append(
            "ArtifactUpdate object missing required field: 'artifact'."
        )
    elif (
        'parts' not in data.get('artifact', {})
        or not isinstance(data.get('artifact', {}).get('parts'), list)
        or not data.get('artifact', {}).get('parts')
    ):
        errors.append("Artifact object must have a non-empty 'parts' array.")
    return errors


def _validate_message(data: dict[str, Any]) -> list[str]:
    errors = []
    if (
        'parts' not in data
        or not isinstance(data.get('parts'), list)
        or not data.get('parts')
    ):
        errors.append("Message object must have a non-empty 'parts' array.")
    if 'role' not in data or data.get('role') != 'agent':
        errors.append("Message from agent must have 'role' set to 'agent'.")
    return errors


def validate_event(data: dict[str, Any]) -> list[str]:
    """Validate an incoming event from the agent based on its kind."""
    if 'kind' not in data:
        return ["Response from agent is missing required 'kind' field."]

    kind = data.get('kind')
    validators = {
        'task': _validate_task,
        'status-update': _validate_status_update,
        'artifact-update': _validate_artifact_update,
        'message': _validate_message,
    }

    validator = validators.get(str(kind))
    if validator:
        return validator(data)

    return [f"Unknown message kind received: '{kind}'."]


# A2A messaging helpers
async def send_text_message(text: str, url: str, context_id: str | None = None, streaming: bool = False):
    async with httpx.AsyncClient(timeout=10) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=url)
        agent_card = await resolver.get_agent_card()
        config = ClientConfig(httpx_client=httpx_client, streaming=streaming)
        factory = ClientFactory(config)
        client = factory.create(agent_card)

        msg = Message(
            kind="message",
            role=Role.user,
            parts=[Part(TextPart(text=text))],
            message_id=uuid4().hex,
            context_id=context_id,
        )

        events = [event async for event in client.send_message(msg)]

    return events


# A2A conformance tests
def test_agent_card(agent_url):
    """Validate agent card structure and required fields."""
    response = httpx.get(f"{agent_url}/.well-known/agent-card.json")
    assert response.status_code == 200, "Agent card endpoint must return 200"

    card_data = response.json()
    errors = validate_agent_card(card_data)

    assert not errors, f"Agent card validation failed:\n" + "\n".join(errors)

@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [True, False])
async def test_message(agent_url, streaming):
    """Test that agent returns valid A2A message format."""
    events = await send_text_message("Hello", agent_url, streaming=streaming)

    all_errors = []
    for event in events:
        match event:
            case Message() as msg:
                errors = validate_event(msg.model_dump())
                all_errors.extend(errors)

            case (task, update):
                errors = validate_event(task.model_dump())
                all_errors.extend(errors)
                if update:
                    errors = validate_event(update.model_dump())
                    all_errors.extend(errors)

            case _:
                pytest.fail(f"Unexpected event type: {type(event)}")

    assert events, "Agent should respond with at least one event"
    assert not all_errors, f"Message validation failed:\n" + "\n".join(all_errors)


# Add your custom tests here
class TestAgentValidation:
    """Test Agent validation methods."""
    
    @pytest.fixture
    def test_agent(self):
        """Create an Agent instance for testing."""
        return Agent()
    
    def test_validate_request_valid_simple(self, test_agent):
        """Test validation with valid simple difficulty request."""
        request = EvalRequest(
            participants={"agent": "http://localhost:9019"},
            config={"difficulty": "simple"}
        )
        is_valid, msg = test_agent.validate_request(request)
        assert is_valid is True
        assert msg == "ok"
    
    def test_validate_request_valid_hard(self, test_agent):
        """Test validation with valid hard difficulty request."""
        request = EvalRequest(
            participants={"agent": "http://localhost:9019"},
            config={"difficulty": "hard"}
        )
        is_valid, msg = test_agent.validate_request(request)
        assert is_valid is True
        assert msg == "ok"
    
    def test_validate_request_missing_role(self, test_agent):
        """Test validation with missing required role."""
        request = EvalRequest(
            participants={},
            config={"difficulty": "simple"}
        )
        is_valid, msg = test_agent.validate_request(request)
        assert is_valid is False
        assert "Missing roles" in msg
        assert "agent" in msg
    
    def test_validate_request_missing_difficulty(self, test_agent):
        """Test validation with missing difficulty config."""
        request = EvalRequest(
            participants={"agent": "http://localhost:9019"},
            config={}
        )
        is_valid, msg = test_agent.validate_request(request)
        assert is_valid is False
        assert "Missing config keys" in msg
    
    def test_validate_request_invalid_difficulty(self, test_agent):
        """Test validation with invalid difficulty value."""
        request = EvalRequest(
            participants={"agent": "http://localhost:9019"},
            config={"difficulty": "medium"}
        )
        is_valid, msg = test_agent.validate_request(request)
        assert is_valid is False
        assert "Invalid difficulty" in msg
        assert "simple" in msg or "hard" in msg
    
    def test_validate_request_video_eval_with_rule_file(self, test_agent):
        """Test validation when video eval is properly configured."""
        request = EvalRequest(
            participants={"agent": "http://localhost:9019"},
            config={
                "difficulty": "simple",
                "task_names": [
                    "look at the sky",
                    "build a house"
                ],
                "num_tasks": 2,
                "max_steps": 2000,
                "enable_video_eval": True,
            }
        )
        is_valid, msg = test_agent.validate_request(request)
        assert is_valid is True
        assert msg == "ok"


class TestAgentResponseParsing:
    """Test Agent response parsing methods."""
    
    @pytest.fixture
    def test_agent(self):
        """Create an Agent instance for testing."""
        return Agent()
    
    def test_parse_agent_response_valid_action(self, test_agent):
        """Test parsing a valid action response."""
        response = json.dumps({
            "type": "action",
            "buttons": [1, 0, 1, 0, 0],
            "camera": [0.5, -0.3]
        })
        action = test_agent._parse_agent_response(response)
        
        assert "buttons" in action
        assert "camera" in action
        assert isinstance(action["buttons"], np.ndarray)
        assert isinstance(action["camera"], np.ndarray)
        assert action["buttons"].dtype == np.int32
        assert action["camera"].dtype == np.float32
        np.testing.assert_array_equal(action["buttons"], np.array([1, 0, 1, 0, 0], dtype=np.int32))
        np.testing.assert_array_almost_equal(action["camera"], np.array([0.5, -0.3], dtype=np.float32))
    
    def test_parse_agent_response_no_buttons(self, test_agent):
        """Test parsing response with missing buttons."""
        response = json.dumps({
            "type": "action",
            "camera": [0.1, 0.2]
        })
        action = test_agent._parse_agent_response(response)
        
        assert "buttons" in action
        assert "camera" in action
        assert len(action["buttons"]) == 1
        assert action["buttons"][0] == 0
    
    def test_parse_agent_response_no_camera(self, test_agent):
        """Test parsing response with missing camera."""
        response = json.dumps({
            "type": "action",
            "buttons": [1, 1, 0]
        })
        action = test_agent._parse_agent_response(response)
        
        assert "buttons" in action
        assert "camera" in action
        assert len(action["camera"]) == 1
        assert action["camera"][0] == 60
    
    def test_parse_agent_response_empty_string(self, test_agent):
        """Test parsing empty response."""
        action = test_agent._parse_agent_response("")
        
        assert "buttons" in action
        assert "camera" in action
        assert len(action["buttons"]) == 1
        assert len(action["camera"]) == 1
        assert action["buttons"][0] == 0
        assert action["camera"][0] == 60
    
    def test_parse_agent_response_invalid_json(self, test_agent):
        """Test parsing invalid JSON."""
        action = test_agent._parse_agent_response("{invalid json}")
        
        assert "buttons" in action
        assert "camera" in action
        assert len(action["buttons"]) == 1
        assert len(action["camera"]) == 1
    
    def test_parse_agent_response_dict_without_type(self, test_agent):
        """Test parsing dict response without type field."""
        response = json.dumps({
            "buttons": [0, 1],
            "camera": [0.0, 0.0]
        })
        action = test_agent._parse_agent_response(response)
        
        assert "buttons" in action
        assert "camera" in action
        np.testing.assert_array_equal(action["buttons"], np.array([0, 1], dtype=np.int32))


class MockPurpleAgent:
    """Mock purple agent for testing communication."""
    
    def __init__(self):
        self.init_received = False
        self.observations_received = []
        self.action_sequence = []
        self.current_action_index = 0
    
    def set_action_sequence(self, actions: list[dict]):
        """Set sequence of actions to return."""
        self.action_sequence = actions
        self.current_action_index = 0
    
    async def handle_message(self, message: str) -> str:
        """Handle incoming message and return response."""
        try:
            payload = json.loads(message)
            msg_type = payload.get("type")
            
            if msg_type == "init":
                self.init_received = True
                init_payload = InitPayload.model_validate(payload)
                # Return acknowledgment
                ack = AckPayload(
                    success=True,
                    message=f"Initialization success with task: {init_payload.text}",
                )
                return ack.model_dump_json()
            
            elif msg_type == "obs":
                obs_payload = ObservationPayload.model_validate(payload)
                self.observations_received.append({
                    "step": obs_payload.step,
                    "obs_length": len(obs_payload.obs)
                })
                
                # Return next action in sequence
                if self.current_action_index < len(self.action_sequence):
                    action_data = self.action_sequence[self.current_action_index]
                    self.current_action_index += 1
                else:
                    # Default action
                    action_data = {"buttons": [0, 0, 0, 0, 0], "camera": [0.0, 0.0]}
                
                action = ActionPayload(**action_data)
                return action.model_dump_json()
            
            else:
                return json.dumps({"error": f"Unknown message type: {msg_type}"})
                
        except Exception as e:
            return json.dumps({"error": str(e)})


class TestAgentPurpleAgentCommunication:
    """Test Agent communication with purple agent."""
    
    @pytest.fixture
    def test_agent(self):
        """Create an Agent instance for testing."""
        return Agent()
    
    @pytest.fixture
    def mock_purple_agent(self):
        """Create a mock purple agent."""
        return MockPurpleAgent()
    
    @pytest.mark.asyncio
    async def test_init_message_format(self, test_agent, mock_purple_agent):
        """Test that init message is properly formatted."""
        task_text = "build a house"
        init_payload = InitPayload(text=task_text)
        
        # Mock purple agent receives it
        response = await mock_purple_agent.handle_message(init_payload.model_dump_json())
        
        assert mock_purple_agent.init_received is True
        
        # Parse response
        ack = AckPayload.model_validate_json(response)
        assert ack.type == "ack"
        assert task_text in ack.message
    
    @pytest.mark.asyncio
    async def test_observation_message_format(self, test_agent, mock_purple_agent):
        """Test that observation message is properly formatted."""
        obs_payload = ObservationPayload(
            step=5,
            obs="base64_encoded_image_data_here"
        )
        
        response = await mock_purple_agent.handle_message(obs_payload.model_dump_json())
        
        assert len(mock_purple_agent.observations_received) == 1
        assert mock_purple_agent.observations_received[0]["step"] == 5
        
        # Parse action response
        action = ActionPayload.model_validate_json(response)
        assert action.type == "action"
        assert "buttons" in action.model_dump()
        assert "camera" in action.model_dump()
    
    @pytest.mark.asyncio
    async def test_action_sequence(self, test_agent, mock_purple_agent):
        """Test sequence of observations and actions."""
        # Set up action sequence
        action_sequence = [
            {"buttons": [1, 0, 0, 0, 0], "camera": [0.1, 0.0]},
            {"buttons": [0, 1, 0, 0, 0], "camera": [0.0, 0.1]},
            {"buttons": [0, 0, 1, 0, 0], "camera": [-0.1, 0.0]},
        ]
        mock_purple_agent.set_action_sequence(action_sequence)
        
        # Send init
        init_msg = InitPayload(text="test task").model_dump_json()
        await mock_purple_agent.handle_message(init_msg)
        
        # Send observations and receive actions
        for i, expected_action in enumerate(action_sequence):
            obs_msg = ObservationPayload(
                step=i,
                obs=f"fake_base64_image_{i}"
            ).model_dump_json()
            
            response = await mock_purple_agent.handle_message(obs_msg)
            action = ActionPayload.model_validate_json(response)
            
            assert action.buttons == expected_action["buttons"]
            assert action.camera == expected_action["camera"]
        
        assert len(mock_purple_agent.observations_received) == 3
    
    @pytest.mark.asyncio
    async def test_parse_action_payload(self, test_agent):
        """Test parsing ActionPayload in agent."""
        action_payload = ActionPayload(
            buttons=[1, 0, 1, 0, 0, 0, 0, 0],
            camera=[0.5, -0.3]
        )
        
        # Test agent's parse method
        action = test_agent._parse_agent_response(action_payload.model_dump_json())
        
        assert "buttons" in action
        assert "camera" in action
        np.testing.assert_array_equal(action["buttons"], [1, 0, 1, 0, 0, 0, 0, 0])
        np.testing.assert_array_almost_equal(action["camera"], [0.5, -0.3])
    
    @pytest.mark.asyncio
    async def test_full_communication_flow(self, test_agent, mock_purple_agent):
        """Test full communication flow: init -> obs -> action cycle."""
        # Setup mock
        mock_purple_agent.set_action_sequence([
            {"buttons": [1, 0, 0, 0, 0], "camera": [0.0, 0.0]},
            {"buttons": [0, 1, 0, 0, 0], "camera": [0.1, 0.0]},
        ])
        
        # Mock the messenger to use our mock purple agent
        async def mock_talk_to_agent(message, url, new_conversation):
            return await mock_purple_agent.handle_message(message)
        
        with patch.object(test_agent.messenger, 'talk_to_agent', side_effect=mock_talk_to_agent):
            # Step 1: Send init
            init_payload = InitPayload(text="build a ladder")
            init_response = await test_agent.messenger.talk_to_agent(
                message=init_payload.model_dump_json(),
                url="http://mock:8000",
                new_conversation=True
            )
            
            ack = AckPayload.model_validate_json(init_response)
            assert ack.type == "ack"
            assert mock_purple_agent.init_received
            
            # Step 2: Send observations and get actions
            for step in range(2):
                obs_payload = ObservationPayload(
                    step=step,
                    obs=f"fake_image_{step}"
                )
                action_response = await test_agent.messenger.talk_to_agent(
                    message=obs_payload.model_dump_json(),
                    url="http://mock:8000",
                    new_conversation=False
                )
                
                # Parse action
                action = test_agent._parse_agent_response(action_response)
                assert "buttons" in action
                assert "camera" in action
                assert isinstance(action["buttons"], np.ndarray)
                assert isinstance(action["camera"], np.ndarray)
            
            assert len(mock_purple_agent.observations_received) == 2
    
    @pytest.mark.asyncio
    async def test_purple_agent_validation_error(self, test_agent, mock_purple_agent):
        """Test handling of invalid messages."""
        # Send invalid message
        invalid_msg = json.dumps({"type": "invalid", "data": "test"})
        response = await mock_purple_agent.handle_message(invalid_msg)
        
        # Should return error
        data = json.loads(response)
        assert "error" in data
    
    @pytest.mark.asyncio
    async def test_pydantic_payload_validation(self, test_agent):
        """Test that Pydantic validates payloads correctly."""
        # Valid payload
        valid_init = InitPayload(text="test task")
        assert valid_init.type == "init"
        assert valid_init.text == "test task"
        
        # Valid observation
        valid_obs = ObservationPayload(step=10, obs="base64data")
        assert valid_obs.step == 10
        assert valid_obs.type == "obs"
        
        # Valid action
        valid_action = ActionPayload(
            buttons=[1, 0, 1],
            camera=[0.5, 0.3]
        )
        assert valid_action.type == "action"
        assert len(valid_action.buttons) == 3
        
        # Invalid step (negative)
        with pytest.raises(Exception):  # Pydantic validation error
            ObservationPayload(step=-1, obs="data")
