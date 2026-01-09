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
from minestudio.simulator.callbacks import RecordCallback, JudgeResetCallback, CommandsCallback, RewardsCallback

from messenger import Messenger
from model import EvalRequest, InitPayload, ObservationPayload, AckPayload, ActionPayload
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
        task_list = request.config.get("task_list", None)
        num_tasks = request.config.get("num_tasks", None)
        max_steps = request.config.get("max_steps", 1000)
        
        # Get tasks
        tasks = get_tasks(difficulty, task_list, num_tasks)
        await updater.update_status(
            TaskState.working, 
            new_agent_text_message(f"Starting MCU evaluation with {len(tasks)} {difficulty} tasks")
        )
        
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        record_dir = self.root_dir / "record" / difficulty / date_str
        os.makedirs(record_dir, exist_ok=True)
        metrics: dict = {}
        
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
                        record_path=os.path.join(record_dir, task.replace(' ', '_'))
                    )
                    metrics[task]["reward"] = reward
                    
                    criteria_dir = self.root_dir / "MCU_benchmark" /"auto_eval" / "criteria_files"
                    rule_file_path = os.path.join(criteria_dir, f"{task.replace(' ', '_')}.txt")
                    video_path = os.path.join(record_dir, f"{task.replace(' ', '_')}.mp4")
                        
                    video_score = await self._run_video_eval(
                        task=task,
                        rule_file_path=rule_file_path,
                        video_path=video_path
                    )
                    metrics[task]["video_score"] = video_score
                            
                except Exception as e:
                    print(f"Error running task {task}: {e}")
                    metrics[task]["reward"] = None
                    metrics[task]["video_score"] = None
        
            # Calculate metrics
            task_rewards = [m["reward"] for m in metrics.values() if m.get("reward") is not None]
            total_reward = sum(task_rewards)
            num_completed = len(task_rewards)
            pass_rate = (total_reward / num_completed * 100) if num_completed > 0 else 0.0
            
            result = {
                "difficulty": difficulty,
                "score": total_reward,
                "pass_rate": pass_rate,
                "num_tasks": len(metrics),
                "num_completed": num_completed,
                "task_metrics": {task: m["reward"] for task, m in metrics.items()},
                "video_scores": {task: m.get("video_score") for task, m in metrics.items()}
            }
            
            task_result_str = "\n".join(
                f"Task '{task_name}': {'✓' if m['reward'] == 1.0 else '✗'} ({m['reward']})"
                for task_name, m in metrics.items()
            )
            
            summary = f"""MCU Evaluation Result
    Difficulty: {difficulty}
    Tasks: {num_completed}
    Pass Rate: {pass_rate:.2f}% ({int(total_reward)}/{num_completed})

    Task Results:
    {task_result_str}"""
    
            # Save results to txt file
            result_file = os.path.join(record_dir, "result.txt")
            with open(result_file, 'w', encoding='utf-8') as f:
                f.write(summary)

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
                RecordCallback(record_path=record_path, fps=30, frame_type="pov"),
                JudgeResetCallback(max_steps),
                CommandsCallback(commands),
                RewardsCallback(),
            ]
        )
        
        def encode_image(image: np.ndarray, fmt: str = '.jpeg') -> str:
            success, buffer = cv2.imencode(fmt, image)
            if not success:
                raise ValueError("Image encoding failed")

            enc_image = base64.b64encode(buffer).decode("utf-8")
            return enc_image
        
        total_reward = 0.0
        terminated = False
        
        try:
            init_payload = InitPayload(text=text)
            res = await self.messenger.talk_to_agent(
                message=init_payload.model_dump_json(),
                url=agent_url,
                new_conversation=True,
            )
            action_payload = AckPayload.model_validate_json(res)
            assert action_payload.success, f"Agent initialization failed: {action_payload.message}"
            
            obs, info = env.reset()
            for step in range(max_steps):
                obs_img = obs['image']
            
                obs_payload = ObservationPayload(
                    step=step,
                    obs=encode_image(obs_img)
                )
                res = await self.messenger.talk_to_agent(
                    message=obs_payload.model_dump_json(),
                    url=agent_url,
                    new_conversation=False,
                )
                
                action = self._parse_agent_response(res)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if terminated:
                    break
        finally:
            env.close()

        return total_reward
    
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
            return {
                "buttons": np.array([0]),
                "camera": np.array([60]),
            }

        try:
            action_payload = ActionPayload.model_validate_json(response)
            buttons = np.array(action_payload.buttons, dtype=np.int32)
            camera = np.array(action_payload.camera, dtype=np.int32)
            return {"buttons": buttons, "camera": camera}
        except Exception as e:
            print(f"Failed to parse agent response as ActionPayload: {e}")
            
            try:
                data = json.loads(response)
                buttons = data.get("buttons")
                camera = data.get("camera")
                
                buttons = np.array(buttons, dtype=np.int32) if buttons is not None else np.array([0])
                camera = np.array(camera, dtype=np.int32) if camera is not None else np.array([60])
                return {"buttons": buttons, "camera": camera}
            except Exception as e2:
                print(f"Failed to parse agent response as JSON: {e2}")
                return {
                    "buttons": np.array([0]),
                    "camera": np.array([60]),
                }