import os
import json
import numpy as np
import base64
import cv2

from pathlib import Path
from datetime import datetime
from typing import Any
from pydantic import ValidationError

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message

from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import CommandsCallback, RecordCallback, JudgeResetCallback

from messenger import Messenger
from model import EvalRequest
from util import get_tasks, assess_video, process_video


class Agent:
    # Fill in: list of required participant roles, e.g. ["pro_debater", "con_debater"]
    required_roles: list[str] = ["agent"]
    # Fill in: list of required config keys, e.g. ["topic", "num_rounds"]
    required_config_keys: list[str] = ["difficulty"]

    def __init__(self):
        self.messenger = Messenger()
        # Initialize other state here
        self.root_dir = Path(__file__).resolve().parents[1]

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self.required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"

        missing_config_keys = set(self.required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"

        # Validate difficulty
        difficulty = request.config.get("difficulty")
        if difficulty not in ['simple', 'hard']:
            return False, f"Invalid difficulty: {difficulty}. Must be 'simple' or 'hard'"

        # Validate video evaluation config if enabled
        if request.config.get("enable_video_eval", False):
            if "rule_file" not in request.config:
                return False, "rule_file is required when enable_video_eval is True"

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
        
        # Get the purple agent URL
        agent_url = str(request.participants["agent"])
        
        # Get config parameters
        difficulty = request.config["difficulty"]
        task_names = request.config.get("task_names", None)
        num_tasks = request.config.get("num_tasks", None)
        max_steps = request.config.get("max_steps", 1000)
        enable_video_eval = request.config.get("enable_video_eval", False)
        
        # Get tasks
        tasks = get_tasks(difficulty, task_names, num_tasks)
        await updater.update_status(
            TaskState.working, 
            new_agent_text_message(f"Starting MCU evaluation with {len(tasks)} {difficulty} tasks")
        )
        
        date = datetime.now().strftime("%Y%m%d_%H%M%S")
        record_dir = self.root_dir / "record" / difficulty / date
        os.makedirs(record_dir, exist_ok=True)
        metrics: dict[str, dict[str, Any]] = {}
        
        try: 
            for task, commands, text in tasks:
                await updater.update_status(
                    TaskState.working, 
                    new_agent_text_message(f"Running task: {task}")
                )
                
                metrics[task] = {}
                
                try:
                    reward = await self._run_single_task(
                        agent_url,
                        commands,
                        text,
                        max_steps,
                        record_path=os.path.join(record_dir, task)
                    )
                    
                    metrics[task]["reward"] = reward
                    
                    if enable_video_eval:
                        criteria_dir = self.root_dir / "MCU_benchmark" / "auto_eval" / "criteria"
                        rule_file_path = os.path.join(criteria_dir, f"{task.replace(' ', '_')}.txt")
                        video_path = os.path.join(record_dir, f"{task}.mp4")
                        
                        video_score = await self._run_video_eval(
                            task=task,
                            rule_file_path=rule_file_path,
                            video_path=video_path
                        )
                        metrics[task]["video_score"] = video_score
                    else:
                        metrics[task]["video_score"] = None
                            
                except Exception as e:
                    print(f"Error running task {task}: {e}")
                    metrics["tasks"][task] = None
                    if enable_video_eval:
                        metrics["video_scores"][task] = None
        
            # Calculate metrics
            task_rewards = [r for r in metrics["tasks"].values() if r is not None]
            total_reward = sum(task_rewards)
            num_completed = len(task_rewards)
            pass_rate = (total_reward / num_completed * 100) if num_completed > 0 else 0.0
            
            result = {
                "difficulty": difficulty,
                "score": total_reward,
                "pass_rate": pass_rate,
                "num_tasks": len(metrics["tasks"]),
                "num_completed": num_completed,
                "task_metrics": metrics["tasks"],
            }
            
            if enable_video_eval:
                result["video_scores"] = metrics["video_scores"]
            
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
    
    async def _run_video_eval(self, task: str, rule_file_path: str, video_path: str) -> dict[str, Any]:
        try:
            video_frames = process_video(video_path)
            result = assess_video(task, rule_file_path, video_frames, video_path)
            return result
        except Exception as e:
            print(f"Video evaluation failed for {task}: {e}")
            return {"error": str(e)}
    
    def _parse_agent_response(self, response: str):
        """
        Parse the purple agent's response string and convert it into an action dict acceptable by `env.step(action)`.
        """

        if not response:
            return {"buttons": np.zeros(1, dtype=np.int32), "camera": np.zeros(1, dtype=np.float32)}

        try:
            data = json.loads(response)
        except Exception as e:
            print(f"Failed to parse agent response as JSON: {e}")
            return {"buttons": np.zeros(1, dtype=np.int32), "camera": np.zeros(1, dtype=np.float32)}
                
                
        if isinstance(data, dict) and data.get("type") == "action":
            payload = data
        else:
            payload = data

        buttons = payload.get("buttons")
        camera = payload.get("camera")

        try:
            buttons = np.array(buttons, dtype=np.int32) if buttons is not None else np.zeros(1, dtype=np.int32)
            camera = np.array(camera, dtype=np.float32) if camera is not None else np.zeros(1, dtype=np.float32)
        except Exception as e:
            print(f"Failed to convert action arrays: {e}")
            return {"buttons": np.zeros(1, dtype=np.int32), "camera": np.zeros(1, dtype=np.float32)}

        return {"buttons": buttons, "camera": camera}