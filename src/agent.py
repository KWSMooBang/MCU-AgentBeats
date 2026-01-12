import sys, os
import json
import numpy as np
import base64
import cv2
import asyncio

from pathlib import Path
from datetime import datetime
from typing import Any
from pydantic import ValidationError

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import RecordCallback, JudgeResetCallback, CommandsCallback, RewardsCallback, TaskCallback, FastResetCallback
from MCU_benchmark.utility.milestone_tracker import MilestoneTrackerCallback

from messenger import Messenger
from model import EvalRequest, InitPayload, ObservationPayload, AckPayload, ActionPayload
from util import get_tasks, assess_video, process_video


class Agent:
    # Fill in: list of required participant roles, e.g. ["pro_debater", "con_debater"]
    required_roles: list[str] = ["agent"]
    # Fill in: list of required config keys, e.g. ["topic", "num_rounds"]
    required_config_keys: list[str] = ["task_category"]
    
    # Valid task categories (based on task_configs/tasks folder names)
    valid_categories: set[str] = {
        "building", "combat", "crafting", "decoration", "ender_dragon",
        "mine_diamond_from_scratch", "explore", "find", "long_horizon", "mining_and_collecting",
        "motion", "overall", "tool_use", "trapping"
    }
    
    def __init__(self):
        self.messenger = Messenger()
        # Initialize other state here
        self.root_dir = Path(__file__).resolve().parents[1]
        # Load prompt template
        prompt_path = self.root_dir / "src" / "prompt.txt"
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.prompt_template = f.read()

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self.required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"

        missing_config_keys = set(self.required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"

        # Validate task_category
        task_category = request.config.get("task_category")
        if task_category and task_category not in self.valid_categories:
            return False, f"Invalid task_category: '{task_category}'. Valid categories: {sorted(self.valid_categories)}"

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
        task_category = request.config.get("task_category", "overall")
        max_steps = request.config.get("max_steps", None)
        
        # Get tasks
        tasks = get_tasks(task_category)
        num_tasks = len(tasks)
        
        await updater.update_status(
            TaskState.working, 
            new_agent_text_message(f"Starting MCU evaluation with {num_tasks} tasks from {task_category}")
        )
        
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.root_dir / "output" / date_str
        os.makedirs(output_dir, exist_ok=True)
        metrics: dict = {}
        
        try: 
            for task, commands, text, reward_cfg, milestone_reward_cfg in tasks:
                print(f"Running task: {task.replace('_', ' ')}")
                await updater.update_status(
                    TaskState.working, 
                    new_agent_text_message(f"Running task: {task.replace('_', ' ')}")
                )
                
                metrics[task] = {}
                # Determine max score for this task
                if task == "kill_ender_dragon":
                    max_score = 100.0
                elif task == "mine_diamond_from_scratch":
                    max_score = 50.0
                else:
                    max_score = 10.0
                
                metrics[task]["max_score"] = max_score
                
                try:        
                    sim_score = await self._run_single_task(
                        agent_url,
                        task, 
                        commands,
                        text,
                        reward_cfg,
                        milestone_reward_cfg,
                        max_steps,
                        record_path=os.path.join(output_dir, task)
                    )
                    metrics[task]["sim_score"] = sim_score
                    
                    criteria_dir = self.root_dir / "MCU_benchmark" /"auto_eval" / "criteria_files"
                    rule_file_path = os.path.join(criteria_dir, f"{task}.txt")
                    video_path = os.path.join(output_dir, task, "episode_1.mp4")
                    
                    # Initialize video evaluation results
                    video_eval_result = None
                    video_score = None
                    
                    # Validate video files
                    if not os.path.exists(video_path):
                        print(f"Warning: Video file not found for {task}")
                        metrics[task]["video_score"] = None
                        metrics[task]["video_details"] = {}
                    else:
                        video_eval_result = await self._run_video_eval(
                            task=task,
                            rule_file_path=rule_file_path,
                            video_path=video_path
                        )
                        video_score = video_eval_result['final_score']
                        metrics[task]["video_score"] = round(video_score, 2)
                        metrics[task]["video_details"] = video_eval_result['details']
                        
                    # Evaluate score combination logic
                    if video_score is None:
                        # If video evaluation failed, use only simulator score
                        metrics[task]["score"] = round(sim_score, 2)
                    else:
                        if reward_cfg or milestone_reward_cfg:
                            # Weighted combination: sim_score 70%, video_score 30%
                            video_scaled = (video_score / 10) * max_score
                            metrics[task]["score"] = round(sim_score * 0.7 + video_scaled * 0.3, 2)
                        else:
                            # If no reward config, use only video evaluation total score and scale it
                            metrics[task]["score"] = round((video_score / 10) * max_score, 2)
                        
                    print(f"Task {task} - score: {metrics[task]['score']}, sim_score: {sim_score}, video_score: {video_score if video_score is not None else 'N/A'}")
                    
                    if metrics[task].get("score") is not None:
                        result_msg = f"""Task {task} complete
- Score: {metrics[task]['score']:.2f} / {max_score}
- Simulator Score: {sim_score:.2f} / {max_score}
- Video Score: {video_score if video_score is not None else 'N/A'} / {max_score}"""
                        await updater.update_status(
                            TaskState.working,
                            new_agent_text_message(result_msg)
                        )
                except Exception as e:
                    print(f"Error running task {task}: {e}")
                    metrics[task]["sim_score"] = 0.0
                    metrics[task]["video_score"] = 0.0
                    metrics[task]["video_details"] = {}
                    metrics[task]["score"] = 0.0
                    metrics[task]["error"] = str(e)
                    metrics[task]["error_type"] = type(e).__name__
        
            # Calculate metrics
            score_list = [m["score"] for m in metrics.values() if m.get("score") is not None]
            total_score = sum(score_list)
            
            # Calculate total possible score
            total_max_score = sum(m.get("max_score", 10.0) for m in metrics.values())
            
            result = {
                "task_category": task_category,
                "num_tasks": num_tasks,
                "total_max_score": total_max_score,
                "total_score": total_score,
                "task_metrics": metrics
            }
            
            task_result_str = "\n".join(
                f"""    Task '{task_name}':
        - max_score: {m.get('max_score', 10.0)}
        - score: {m.get('score', 0.0):.2f} / {m.get('max_score', 10.0)}
        - sim_score: {m.get('sim_score', 0.0):.2f} / {m.get('max_score', 10.0)}
        - video_score: {(m.get('video_score', 0.0)):.2f} / 10.0"""
                for task_name, m in metrics.items() 
            )
            
            summary = f"""MCU Evaluation Result
Categories: {task_category}
Number of Tasks: {num_tasks}
Total Score: {total_score:.2f} / {total_max_score}

Task Results:
{task_result_str}"""
    
            # Save results to txt file
            result_file = os.path.join(output_dir, "result.txt")
            with open(result_file, 'w', encoding='utf-8') as f:
                f.write(summary)

            await updater.add_artifact(
                parts=[
                    Part(root=TextPart(text=summary)),
                    Part(root=DataPart(data=result))
                ],
                name="Result"
            )
            print("========== Evaluation completed. ==========")
            print(summary)
        
        finally:
            self.messenger.reset()
    
    async def _run_single_task(
        self,
        agent_url: str,
        task: str,
        commands: list[str],
        text: str,
        reward_cfg: list[dict],
        milestone_reward_cfg: list[dict],
        max_steps: int | None,
        record_path: str
    ) -> float:
        """ 
        Run a single MCU task and return the reward.
        """
        
        # Determine max_steps based on task and input
        if max_steps is None:
            # Default steps based on task type
            if task == "kill_ender_dragon":
                max_steps = 12000
            elif task == "mine_diamond_from_scratch":
                max_steps = 6000
            else:
                max_steps = 600
        elif max_steps <= 600:
            # Use the provided max_steps for all tasks
            max_steps = max_steps
        else:
            # max_steps > 600: apply only to long horizon tasks
            if task not in ["kill_ender_dragon", "mine_diamond_from_scratch"]:
                max_steps = 600
        
        task_dict = {
            'name': task.replace('_', ' '),
            'text': text
        }
        
        callbacks = [
            TaskCallback([task_dict]),  
            FastResetCallback(
                biomes=['plains', 'forest'],
                random_tp_range=1000,
            ),
            RecordCallback(
                record_path=record_path,
                fps=20,
                frame_type='pov',
            )
        ]
        
        if reward_cfg:
            callbacks.insert(1, RewardsCallback(reward_cfg=reward_cfg))
            
        if milestone_reward_cfg:
            callbacks.insert(1, MilestoneTrackerCallback(
                reward_cfg=milestone_reward_cfg,
                output_path=record_path,
                task_name=task
            ))
            
        if commands:
            callbacks.insert(0, CommandsCallback(commands))
            
        env = MinecraftSim(
            obs_size=(128, 128),
            callbacks=callbacks
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
            # Send prompt and task description separately
            init_payload = InitPayload(prompt=self.prompt_template, text=text)
            res = await asyncio.wait_for(
                self.messenger.talk_to_agent(
                    message=init_payload.model_dump_json(),
                    url=agent_url,
                    new_conversation=True,
                ),
                timeout=60.0
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
                
                try:
                    res = await asyncio.wait_for(
                        self.messenger.talk_to_agent(
                            message=obs_payload.model_dump_json(),
                            url=agent_url,
                            new_conversation=False,
                        ),
                        timeout=60.0
                    )
                    action = self._parse_agent_response(res)
                except asyncio.TimeoutError:
                    print(f"Agent timeout at step {step}, using noop action")
                    action = {"buttons": np.array([0]), "camera": np.array([60])}
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                
                if terminated:
                    break
        except Exception as e:
            print(f"Error: {e}")
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception as e:
                    print(f"Error closing environment: {e}")

        return total_reward
    
    async def _run_video_eval(self, task: str, rule_file_path: str, video_path: str) -> dict:
        """Run video evaluation and return detailed results.
        
        Returns:
            dict: {
                'final_score': float,
                'details': dict,
                'weights': dict,
                'excluded_criteria': list
            }
        """
        try:
            video_frames = process_video(video_path)
            result = assess_video(task, rule_file_path, video_frames, video_path)
            return result
        except Exception as e:
            print(f"Video evaluation failed for {task}: {e}")
            return {
                'final_score': 0.0,
                'details': {},
                'weights': {},
                'excluded_criteria': []
            }
    
    def _parse_agent_response(self, response: str) -> dict:
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