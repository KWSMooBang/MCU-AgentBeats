import os
import json
import numpy as np
import base64
import cv2
import torch

from pathlib import Path
from datetime import datetime
from typing import Any
from pydantic import BaseModel, HttpUrl, ValidationError
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger

from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import CommandsCallback, RecordCallback, SpeedTestCallback, SummonMobsCallback, MaskActionsCallback, RewardsCallback, JudgeResetCallback, FastResetCallback


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
    repo_root = Path(__file__).resolve().parents[1]
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


class EvalRequest(BaseModel):
    """Request format sent by the AgentBeats platform to green agents."""
    participants: dict[str, HttpUrl] # role -> agent URL
    config: dict[str, Any]


class Agent:
    # Fill in: list of required participant roles, e.g. ["pro_debater", "con_debater"]
    required_roles: list[str] = ["agent"]
    # Fill in: list of required config keys, e.g. ["topic", "num_rounds"]
    required_config_keys: list[str] = ["difficulty"]

    def __init__(self):
        self.messenger = Messenger()
        # Initialize other state here

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self.required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"

        missing_config_keys = set(self.required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"

        # Add additional request validation here

        return True, "ok"

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Implement your agent logic here.

        Args:
            message: The incoming message
            updater: Report progress (update_status) and results (add_artifact)

        Use self.messenger.talk_to_agent(message, url) to call other agents.
        """
        input_text = get_message_text(message)

        try:
            request: EvalRequest = EvalRequest.model_validate_json(input_text)
            ok, msg = self.validate_request(request)
            if not ok:
                await updater.reject(new_agent_text_message(msg))
                return
        except ValidationError as e:
            await updater.reject(new_agent_text_message(f"Invalid request: {e}"))
            return

        # Replace example code below with your agent logic
        # Use request.participants to get participant agent URLs by role
        # Use request.config for assessment parameters
        
        # Get the purple agent URL
        agent_url = str(request.participants["agent"])
        
        # Get config parameters
        difficulty = request.config["difficulty"]
        if difficulty not in ['simple', 'hard']:
            print("Invalid difficulty level. Please choose 'simple' or 'hard'.")
            exit()
        task_names = request.config.get("task_names", None)
        num_tasks = request.config.get("num_tasks", None)
        max_steps = request.config.get("max_steps", 1000)
        
        # Get tasks
        tasks = get_tasks(difficulty, task_names, num_tasks)
        await updater.update_status(
            TaskState.working, 
            new_agent_text_message(f"Starting MCU evaluation with {len(tasks)} {difficulty} tasks")
        )
        
        date = datetime.now().strftime("%Y%m%d_%H%M%S")
        record_path = f".output/{difficulty}/{date}"
        os.makedirs(record_path, exist_ok=True)
        metrics: dict[str, Any] = {"tasks": {}}
        
        try: 
            for task_name, commands, text in tasks:
                await updater.update_status(
                    TaskState.working, 
                    new_agent_text_message(f"Running task: {task_name}")
                )
                
                try:
                    reward = await self._run_single_task(
                        agent_url,
                        commands,
                        text,
                        max_steps,
                        record_path=os.path.join(record_path, task_name)
                    )
                    metrics["tasks"][task_name] = reward
                except Exception as e:
                    metrics["tasks"][task_name] = None
        
            total_reward = sum(metrics["tasks"].values())
            num_completed = len(metrics["tasks"]) 
            pass_rate = (total_reward / num_completed * 100) if num_completed > 0 else 0.0
            
            result = {
                "difficulty": difficulty,
                "score": total_reward,
                "pass_rate": pass_rate,
                "task_metrics": metrics["tasks"],
            }
            
            task_result_str = "\n".join(
                f"Task '{task_name}': {'✓' if reward == 1.0 else '✗'} ({reward})"
                for task_name, reward in metrics["tasks"].items()
            )
            
            summary = f"""MCU Evaluation Result
    Difficulty: {difficulty}
    Tasks: {num_completed}
    Pass Rate: {pass_rate:.2f}% ({int(total_reward)}/{num_completed})

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
            self.messenger.reset()
    
    async def _run_single_task(
        self,
        agent_url: str,
        commands: list[str],
        text: str,
        max_steps: int,
        record_path: str
    ):
        """ 
        Run a single MCU task and reutrn the reward.
        """
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
        response = await self.messenger.talk_to_agent(
            message=json.dumps(payload),
            url=agent_url,
            new_conversation=True,
        )
        
        try:
            obs, info = env.reset()

            for step in range(max_steps):
                obs_img = obs['image']
            
                payload = {
                    'type': 'obs',
                    'step': step,
                    'obs': encode_image(obs_img)
                }
                response = await self.messenger.talk_to_agent(
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
    
    async def _run_video_eval(self):
        pass
    
    def _parse_agent_response(self, response: str):
        """
        Parse the purple agent's response string and convert it into an action dict acceptable by `env.step(action)`.
        """

        if not response:
            return {"buttons": np.zeros(1, dtype=np.int32), "camera": np.zeros(1, dtype=np.float32)}

        try:
            data = json.loads(response)
        except Exception:
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
            return {"buttons": np.zeros(1, dtype=np.int32), "camera": np.zeros(1, dtype=np.float32)}

        return {"buttons": buttons, "camera": camera}