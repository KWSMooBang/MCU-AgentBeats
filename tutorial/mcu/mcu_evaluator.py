"""
MCU Evaluator - Green agent that evaluates MCU benchmark on purple agents.

This agent:
1. Sets up MCU simulation environment
2. Send task prompts to the purple agent
3. Parses the purple agent responses
4. Steps through the simulation environment and collects metrics
"""

import os
import argparse
import uvicorn
import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

import re
import json
import numpy as np
import base64
import cv2 
import torch
from pathlib import Path

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    DataPart,
    Part,
    TaskState,
    TextPart,
)
from a2a.utils import new_agent_text_message

from agentbeats.green_executor import GreenAgent, GreenExecutor
from agentbeats.models import EvalRequest, EvalResult
from agentbeats.tool_provider import ToolProvider

from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import CommandsCallback, RecordCallback, SpeedTestCallback, SummonMobsCallback, MaskActionsCallback, RewardsCallback, JudgeResetCallback, FastResetCallback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcu_evaluator")


def create_agent_card(name: str, url: str) -> AgentCard:
    """Create the agent card for the MCU evaluator agent."""
    skill = AgentSkill(
        id="mcu_evaluation",
        name="MCU Evaluation",
        description="Evaluates MCU benchmark tasks",
        tags=["benchmark", "mcu", "evaluation"],
        examples=[],
    )
    return AgentCard(
        name=name,
        description="MCU Evaluator Agent",
        url=url,
        version="1.0.0",
        default_input_modes=["application/json"],
        default_output_modes=["application/json"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )
    

def extract_info(yaml_content, filename):
    lines = yaml_content.splitlines()
    commands = []
    text = ''

    for line in lines:
        if line.startswith('-'):
            command = line.strip('- ').strip()
            commands.append(command)
        elif line.startswith('text:'):
            text = line.strip('text: ').strip()

    task_name = filename[:-5].replace('_', ' ')
    return task_name, commands, text

    
def get_tasks(difficulty: str, task_names: list[str]|None=None, num_tasks: int|None=None) -> list[str]:
    # Resolve task configs directory relative to this file's project root
    repo_root = Path(__file__).resolve().parents[2]
    directory = repo_root / 'MCU_benchmark' / 'task_configs' / difficulty
    if not directory.exists():
        raise FileNotFoundError(f"Task configs directory not found: {directory}")
    all_task_files = [p.name for p in directory.iterdir() if p.suffix == '.yaml']
    
    if task_names:
        task_files = [f"{name}.yaml" for name in task_names if f"{name}.yaml" in all_task_files]
        if len(task_files) == 0:
            print("No matching task names found. Using all tasks.")
            task_files = all_task_files
    else: 
        task_files = all_task_files
        
    if num_tasks:
        task_files = task_files[:num_tasks]
    
    tasks = []
    for filename in task_files:
        file_path = directory / filename
        with open(file_path, 'r', encoding='utf-8') as file:
            yaml_content = file.read()
        task_name, commands, text = extract_info(yaml_content, filename)
        tasks.append((task_name, commands, text))
    
    return tasks


class MCUEvaluator(GreenAgent):
    def __init__(self):
        self._required_roles = ["agent"]
        self._required_config_keys = ["difficulty"]
        self._tool_provider = ToolProvider()
        
    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self._required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"
        missing_config_keys = set(self._required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"
        return True, "ok"
    
    async def run_eval(self, req: EvalRequest, updater: TaskUpdater) -> None:
        logger.info(f"   Starting MCU evaluation: {req}")
        start_time = time.time()
        
        difficulty = req.config["difficulty"]
        if difficulty not in ['simple', 'hard']:
            print("Invalid difficulty level. Please choose 'simple' or 'hard'.")
            exit()
        task_names = req.config.get("task_names", None)
        num_tasks = req.config.get("num_tasks", None)
        max_steps = req.config.get("max_steps", 900)
        
        # Get the purple agent URL
        agent_url = str(req.participants["agent"])
        
        # Get task
        tasks = get_tasks(difficulty, task_names, num_tasks)
        logger.info(f"Running {len(tasks)} {difficulty} tasks")
        
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Starting MCU evaluation with {len(tasks)} {difficulty} tasks.")
        )
        
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        record_dir = f"./output/{difficulty}/{date_str}"
        os.makedirs(record_dir, exist_ok=True)
        metrics: dict[str, Any] = {"tasks": {}}

        try:
            for (task_name, commands, text) in tasks:
                logger.info(f"Running task '{task_name}'...")
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"Running task '{task_name}'...")
                )
                
                try:
                    reward = await self._run_single_task(
                        agent_url=agent_url,
                        commands=commands,
                        text=text,
                        max_steps=max_steps,
                        record_path=os.path.join(record_dir, task_name)
                    )
                    metrics["tasks"][task_name] = reward
                    logger.info(f"Task '{task_name}' completed with reward: {reward}")
                except Exception as e:
                    logger.error(f"Task '{task_name}' failed: {e}")
                    metrics["tasks"][task_name] = None
            
            time_used = time.time() - start_time
            total_reward = sum(metrics["tasks"].values())
            num_completed = len(metrics["tasks"]) 
            pass_rate = (total_reward / num_completed * 100) if num_completed > 0 else 0.0
            
            result = {
                "difficulty": difficulty,
                "score": total_reward,
                "pass_rate": pass_rate,
                "task_metrics": metrics["tasks"],
                "time_used": time_used,
            }
            
            task_result_str = "\n".join(
                f"Task '{task_name}': {'✓' if reward == 1.0 else '✗'} ({reward})"
                for task_name, reward in metrics["tasks"].items()
            )
            
            summary = f"""MCU Evaluation Result
Difficulty: {difficulty}
Tasks: {num_completed}
Pass Rate: {pass_rate:.2f}% ({int(total_reward)}/{num_completed})
Time Used: {time_used:.2f} seconds

Task Results:
{task_result_str}"""

            await updater.add_artifact(
                parts=[
                    Part(root=TextPart(text=summary)),
                    Part(root=DataPart(data=result))
                ],
                name="Result"
            )
            
        finally:
            self._tool_provider.reset()
        
        
    async def _run_single_task(
        self,
        agent_url: str,
        commands: list[str],
        text: str,
        max_steps: int,
        record_path: str
    ):
        """Run a single MCU task and return the reward."""
        env = MinecraftSim(
            obs_size=(128, 128),
            callbacks=[
                CommandsCallback(commands),
                JudgeResetCallback(max_steps),
                RecordCallback(record_path=record_path, fps=30, frame_type="pov"),
            ]
        )
        
        def encode_image(image: np.ndarray, fmt: str = '.jpeg') -> str:
            success, buffer = cv2.imencode(fmt, image)
            if not success:
                raise ValueError("Image encoding failed")

            enc_image = base64.b64encode(buffer).decode("utf-8")
            return enc_image

        reward = 0.0
        terminated = False
        
        payload = {
            'type': 'init',
            'text': text,
        }
        response = await self._tool_provider.talk_to_agent(
            message=json.dumps(payload),
            url=agent_url,
            new_conversation=True,
        )
        logger.info(f"Sent init message to agent, received response: {response}")
        
        try:
            obs, info = env.reset()

            for step in range(max_steps):
                obs_img = obs['image']
            
                payload = {
                    'type': 'obs',
                    'step': step,
                    'obs': encode_image(obs_img)
                }
                response = await self._tool_provider.talk_to_agent(
                    message=json.dumps(payload),
                    url=agent_url,
                    new_conversation=False,
                )
                
                action = self._parse_agent_response(response)
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated:
                    break
        finally:
            env.close()

        return float(reward)


    def _parse_agent_response(self, response: str):
        """
        Parse the purple agent's response string and convert it into an action dict acceptable by `env.step(action)`.
        """

        if not response:
            logger.warning("Empty response from agent; returning no-op action.")
            return {"buttons": np.zeros(1, dtype=np.int32), "camera": np.zeros(1, dtype=np.float32)}

        try:
            data = json.loads(response)
        except Exception:
            logger.warning("No JSON found in agent response; returning no-op action.")
            return {"buttons": np.zeros(1, dtype=np.int32), "camera": np.zeros(1, dtype=np.float32)}
                
                
        if isinstance(data, dict) and data.get("type") == "action":
            payload = data
        else:
            payload = data

        buttons = payload.get("buttons")
        camera = payload.get("camera")

        try:
            buttons = np.array(buttons)
            camera = np.array(camera)

        except Exception as e:
            logger.warning(f"Failed to convert action arrays: {e}; returning no-op action.")
            return {"buttons": np.zeros(1, dtype=np.int32), "camera": np.zeros(1, dtype=np.float32)}

        return {"buttons": buttons, "camera": camera}


    def _run_video_eval(self):
        pass
    
async def main():
    parser = argparse.ArgumentParser(description="Run the A2A MCU Evaluate")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9019, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL to provide in the agent card")
    args = parser.parse_args()

    logger.info("Starting MCU Evaluator agent...")
    agent_url = args.card_url or f"http://{args.host}:{args.port}/"

    agent = MCUEvaluator()
    executor = GreenExecutor(agent)
    agent_card = create_agent_card("MCUEvaluator", agent_url)
    
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )
    
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    uvicorn_config = uvicorn.Config(server.build(), host=args.host, port=args.port)
    uvicorn_server = uvicorn.Server(uvicorn_config)
    await uvicorn_server.serve()

if __name__ == '__main__':
    asyncio.run(main())
