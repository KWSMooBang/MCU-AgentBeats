import cv2  
import base64
from openai import OpenAI
import os
import json
import yaml

from datetime import datetime
from pathlib import Path

root_dir = Path(__file__).resolve().parents[1]

def extract_info(yaml_content: str, filename: str) -> tuple:
    data = yaml.safe_load(yaml_content)
    
    commands = data.get('custom_init_commands', [])
    text = data.get('text', '')
    reward_cfg = data.get('reward_cfg', [])
    
    task = filename[:-5]
    return task, commands, text, reward_cfg

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
        task, commands, text, reward_cfg = extract_info(yaml_content, file_path.name)
        tasks.append((task, commands, text, reward_cfg, category))
    
    return tasks

def fetch(query: list[dict], model: str = 'gpt-4o') -> str:  # gpt4
    print(f'fetching {model} ...')
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.create(
        model=model,
        messages=query,
        temperature=0.7
    )
    res = completion.choices[0].message.content
    return res

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
    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        # frame = cv2.resize(orig_frame, (224, 224))
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()

    base64Frames1 = base64Frames[0::25]
    print(len(base64Frames1), "frames read.")
    if (len(base64Frames1) > 60):
        base64Frames1 = base64Frames[0::70]
        
    return base64Frames1

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
    result = save_data_json(response, task, video_path)
    return result

def save_data_json(response: str, task: str, video_path: str) -> float:
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

    for line in response.strip().split('\n'):  
        for key in keys_to_extract:  
            if line.startswith(f'- {key}: '):   
                value = (line.split(': ', 1)[1].strip()) 
                
                if value: 
                    result[key] = float(value)  
                    break  
    
    sum_score = 0
    num_criteria = 0
    for key in keys_to_extract:
        if key in result:
            num_criteria += 1
            sum_score += result[key]
    result['final score'] = sum_score / num_criteria if num_criteria > 0 else 0
    result['origin response'] = response
    
    output_dir = os.path.dirname(video_path)
    result_path = os.path.join(output_dir, f"video_eval_result.json")
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=4)

    return result['final score']
