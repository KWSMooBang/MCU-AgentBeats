import cv2  
import base64
import time
from openai import OpenAI
import os
import requests
import shutil
from PIL import Image
from io import BytesIO
import json
import datetime
import openai
import requests
import argparse

from pathlib import Path

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


def fetch(query, model='gpt-4o'): # gpt4
    print(f'fetching {model} ...')
    client = OpenAI(api_key='empty')
    completion = client.chat.completions.create(
        model=model,
        messages=query,
        temperature=0.7
    )
    res = completion.choices[0].message.content
    return res


def assess_video(task_name, rule_file, frames, video_path):
    repo_root = Path(__file__).resolve().parents[1]
    directory = repo_root / 'MCU_benchmark' / 'auto_eval' / 'prompt'
    with open(directory / 'single_rating_prompt.txt', 'r', encoding='utf-8') as file:  
        system_content = file.read()
    with open(rule_file, 'r', encoding='utf-8') as file:  
        grading_rule = file.read()
    query = [
        {
        "role": "system",
        "content": system_content
        },
        {
        "role": "user", "content":  
        f'The task name is ' + task_name + ' '
        + f'You should follow the following grading criteria to compare the performance of agents in videos A and B' + grading_rule +'\n'
        + f'Here are the image frames of the video A '
        }]

    query.append({"role": "user", "content": [{
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{frame}"
          },
        } for frame in frames
        ]})

    ans = fetch(query)
    print(ans)
    save_data_json(ans, task_name, video_path)
    answer = {"role": "assistant", "content": f'{ans}'}
    return ans


def save_data_json(ans, task_name, video_path):
    result_dict= {}
    keys_to_extract = [  
        "Task Progress",  
        "Action Control",  
        "Error Recognition and Correction",  
        "Creative Attempts",  
        "Task Completion Efficiency",  
        "Material Selection and Usage"  
    ]  

    for line in ans.strip().split('\n'):  
        for key in keys_to_extract:  
            if line.startswith(f'- {key}: '):   
                value = (line.split(': ', 1)[1].strip()) 
                
                if value: 
                    result_dict[key] = value  
                    break  
    result_dict['video_path'] = video_path
    result_dict['task_name'] = task_name
    
    out_file = os.path.join(out_dir, json_name)
    with open(out_file, 'w') as f:
        json.dump(result_dict, f, indent = 4)

    return result_dict  


def process_video(video_path):
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
    assert(len(base64Frames1)<150)
    return base64Frames1


def find_mp4_files(directory):
    mp4_files = []
    task_list = []
    video_name = []
    for root, dirs, files in os.walk(directory):
        files = files[:5]
        for file in files:
        
            if file.endswith('.mp4'):

                parent_dir = os.path.basename(root)
                full_path = os.path.join(root, file)
                video_name.append(file)
                mp4_files.append(full_path)
                task_list.append(parent_dir)
    return mp4_files, task_list, video_name