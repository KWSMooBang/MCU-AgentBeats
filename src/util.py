import cv2  
import base64
from openai import OpenAI
import os
import json
import yaml
import time

from datetime import datetime
from pathlib import Path

root_dir = Path(__file__).resolve().parents[1]

def extract_info(yaml_content: str, filename: str) -> tuple:
    data = yaml.safe_load(yaml_content)
    
    commands = data.get('custom_init_commands', [])
    text = data.get('text', '')
    reward_cfg = data.get('reward_cfg', [])
    milestone_reward_cfg = data.get('milestone_reward_cfg', [])
    
    task = filename[:-5]
    return task, commands, text, reward_cfg, milestone_reward_cfg

def get_tasks(task_category: list[str] = []) -> list[tuple]:
    # Resolve task configs directory relative to this file's project root
    tasks_dir = root_dir / 'MCU_benchmark' / 'task_configs' / 'tasks'
    if not tasks_dir.exists():
        raise FileNotFoundError(f"Task configs directory not found: {tasks_dir}")
    
    # If task_category is empty, use all categories except overall
    if not task_category:
        categories = [d.name for d in tasks_dir.iterdir() if d.is_dir() and d.name != 'overall']
    else:
        categories = task_category
    
    # Collect all task files from specified categories
    task_files = []
    for category in categories:
        category_dir = tasks_dir / category
        if not category_dir.exists():
            print(f"Warning: Category '{category}' not found, skipping.")
            continue
        
        for task_file in category_dir.glob('*.yaml'):
            task_files.append((category, task_file))
    
    # Parse each task file
    tasks = []
    for category, file_path in task_files:
        with open(file_path, 'r', encoding='utf-8') as file:
            yaml_content = file.read()
        task, commands, text, reward_cfg, milestone_reward_cfg = extract_info(yaml_content, file_path.name)
        tasks.append((task, commands, text, reward_cfg, milestone_reward_cfg))
    
    return tasks

def fetch(query: list[dict], model: str = 'gpt-4o') -> str:  # gpt4
    print(f'fetching {model} ...')
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    client = OpenAI(api_key=api_key)
    
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=query,
                temperature=0.5,
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"API call failed (attempt {attempt + 1}/{max_attempts}): {e}")
            if attempt < max_attempts - 1:
                time.sleep(2 ** attempt)  
            else:
                raise

def process_video(video_path: str) -> list[str]:
    """Extract frames from video and encode as base64.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        list[str]: List of base64-encoded frames
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    
    # Adaptive sampling rate
    if total_frames < 60 * 20:
        sample_rate = 20  # Every 20 frames (~1 sec at 20fps)
    else:
        sample_rate = 60
    
    base64Frames = []
    frame_count = 0
    
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        
        if frame_count % sample_rate == 0:
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        
        frame_count += 1
    
    video.release()
    print(f"{len(base64Frames)} frames sampled from {total_frames} total frames.")
    return base64Frames

def assess_video(task: str, rule_file: str, frames: list[str], video_path: str) -> float:
    prompt_dir = root_dir / 'MCU_benchmark' / 'auto_eval' / 'prompt'
    prompt_file = prompt_dir / 'single_rating_prompt.txt'
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    
    with open(prompt_file, 'r', encoding='utf-8') as file:  
        system_content = file.read()
    
    if not os.path.exists(rule_file):
        raise FileNotFoundError(f"Rule file not found: {rule_file}")
        
    with open(rule_file, 'r', encoding='utf-8') as file:  
        grading_rule = file.read()
        
    query = [
        {
        "role": "system",
        "content": system_content
        },
        {
        "role": "user", "content":  
        f'The task name is ' + task + ' '
        + f'You should follow the following grading criteria to score the performance of agents in videos' + grading_rule +'\n'
        + f'Here are the image frames of the video '
        }]

    query.append(
        {
            "role": "user", 
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frame}"
                    },
                } for frame in frames
            ]
        }
    )

    response = fetch(query)
    result = save_data_json(response, task, video_path, grading_rule)
    return result

def save_data_json(response: str, task: str, video_path: str, grading_rule: str = "") -> float:
    result = {}
    keys_to_extract = [  
        "Task Progress",  
        "Action Control",  
        "Error Recognition and Correction",  
        "Creative Attempts",  
        "Task Completion Efficiency",  
        "Material Selection and Usage"  
    ]  

    result['task'] = task
    result['video_path'] = video_path
    
    # Check which criteria are marked as "Not applicable" in the rule file
    not_applicable_keys = set()
    if grading_rule:
        for key in keys_to_extract:
            # Check if the criterion section contains "Not applicable"
            if f"**{key}:" in grading_rule:
                section_start = grading_rule.find(f"**{key}:")
                section_end = grading_rule.find("\n\n**", section_start)
                if section_end == -1:
                    section_end = len(grading_rule)
                section = grading_rule[section_start:section_end]
                if "Not applicable" in section or "not applicable" in section:
                    not_applicable_keys.add(key)
                    print(f"Excluding '{key}' from scoring (marked as not applicable)")
    
    for line in response.strip().split('\n'):
        for key in keys_to_extract:
            if line.startswith(f'- {key}: '):
                try:
                    value = line.split(': ', 1)[1].strip()
                    result[key] = float(value)
                    break
                except (ValueError, IndexError):
                    pass
            elif line.startswith(f'{key}: '):
                try:
                    value = line.split(': ', 1)[1].strip()
                    result[key] = float(value)
                    break
                except (ValueError, IndexError):
                    pass
    
    # Validate all scores are within 0-10 range
    for key in keys_to_extract:
        if key in result:
            result[key] = max(0.0, min(10.0, result[key]))
    
    # Weighted scoring - Task Progress is most important
    base_weights = {
        "Task Progress": 0.40,                    # 40% - Most important
        "Material Selection and Usage": 0.15,     # 15% - Material usage
        "Action Control": 0.15,                   # 15% - Action control
        "Task Completion Efficiency": 0.15,       # 15% - Efficiency
        "Error Recognition and Correction": 0.10, # 10% - Error correction
        "Creative Attempts": 0.05                 # 5%  - Bonus
    }
    
    # Remove not applicable criteria from weights
    weights = {k: v for k, v in base_weights.items() if k not in not_applicable_keys}
    
    # Renormalize weights to sum to 1.0
    weight_sum = sum(weights.values())
    if weight_sum > 0:
        weights = {k: v / weight_sum for k, v in weights.items()}
    
    weighted_score = 0.0
    total_weight = 0.0
    for key in keys_to_extract:
        if key in result and key in weights:
            weighted_score += result[key] * weights[key]
            total_weight += weights[key]
    
    result['final score'] = weighted_score / total_weight if total_weight > 0 else 0
    result['applicable_criteria'] = list(weights.keys())
    result['excluded_criteria'] = list(not_applicable_keys)
    result['origin response'] = response
    
    output_dir = os.path.dirname(video_path)
    result_path = os.path.join(output_dir, f"video_eval_result.json")
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=4)

    return result['final score']
