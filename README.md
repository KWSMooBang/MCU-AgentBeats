# MCU Green Agent - AgentBeats Competition

MCU (Minecraft Universe) Benchmark Green Agent implementation for the AgentBeats Competition.

## 🎯 Overview

This Green Agent evaluates Purple Agents using the MCU benchmark, providing the following capabilities:
- Agent performance evaluation in Minecraft environments
- Task execution across multiple difficulty levels
- Video-based performance analysis (optional)
- A2A protocol-based inter-agent communication

## 🚀 Quick Start

### 1. Set Environment Variables

Create a `.env` file or set environment variables:
```bash
export OPENAI_API_KEY="your-openai-api-key"  # Required for video evaluation
```

### 2. Run the Server

```bash
cd /workspace/Agent-AI/MCU-AgentBeats/green_agent

# Run with default settings
python src/server.py --host 0.0.0.0 --port 9009

# Specify agent card URL
python src/server.py --host 0.0.0.0 --port 9009 --card-url "http://your-host:9009/"
```

### 3. Send Evaluation Request

```python
import httpx
import json

# Create evaluation request
request = {
    "participants": {
        "agent": "http://purple-agent-url:8080"  # Purple agent URL
    },
    "config": {
        "difficulty": "simple",  # 'simple' or 'hard'
        "task_names": None,  # None for all tasks, or ["task1", "task2"]
        "num_tasks": None,  # Limit number of tasks (None for all)
        "max_steps": 1000,  # Max steps per task
        "enable_video_eval": False,  # Enable video evaluation
        "rule_file": None  # Rule file for video eval (required if enable_video_eval=True)
    }
}

# Send A2A message
async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:9009/messages",
        json={
            "kind": "message",
            "role": "user",
            "parts": [{"kind": "text", "text": json.dumps(request)}],
            "message_id": "test-123",
            "context_id": None
        }
    )
    print(response.json())
```

## 📋 Configuration Parameters

### Required Parameters
- **difficulty** (str): 'simple' or 'hard'
  - `simple`: Basic tasks (block placement, simple crafting, etc.)
  - `hard`: Complex tasks (construction, advanced crafting, etc.)

### Optional Parameters
- **task_names** (list[str] | None): Specific task names to run
  - `None`: Run all tasks
  - `["build_house", "craft_tools"]`: Run only specified tasks

- **num_tasks** (int | None): Limit the number of tasks to run
  - `None`: All tasks
  - `5`: Only first 5 tasks

- **max_steps** (int): Maximum steps per task (default: 1000)

- **enable_video_eval** (bool): Enable video-based evaluation (default: False)
  - Qualitative evaluation using GPT-4 Vision
  - Requires `OPENAI_API_KEY` environment variable

- **rule_file** (str | None): Video evaluation rule file path
  - Required when `enable_video_eval=True`

## 📊 Evaluation Results

After evaluation completes, returns results in the following format:

```json
{
  "difficulty": "simple",
  "score": 8.0,
  "pass_rate": 80.0,
  "num_tasks": 10,
  "num_completed": 10,
  "task_metrics": {
    "build_house": 1.0,
    "craft_tools": 1.0,
    "find_diamond": 0.0,
    ...
  },
  "video_scores": {  // Only included when enable_video_eval=True
    "build_house": {
      "Task Progress": "8/10",
      "Action Control": "Good",
      ...
    }
  }
}
```

### Metrics Explanation
- **score**: Number of successful tasks (each task is 0.0 or 1.0)
- **pass_rate**: Success rate (%)
- **task_metrics**: Success status for each task
- **video_scores**: Video-based qualitative evaluation (optional)

## 🏗️ Architecture

```
green_agent/
├── src/
│   ├── server.py       # A2A server entry point
│   ├── executor.py     # Task execution management
│   ├── agent.py        # Main agent logic
│   ├── messenger.py    # Communication with Purple agent
│   ├── util.py         # Utility functions
│   └── model.py        # Pydantic models
└── MCU_benchmark/      # Benchmark task definitions
    └── task_configs/
        ├── simple/     # Simple tasks
        └── hard/       # Hard tasks
```

## 🔧 Purple Agent Requirements

The Purple Agent must handle the following message formats:

### 1. Initialization Message
```json
{
  "type": "init",
  "text": "Build a house with wooden planks"
}
```

### 2. Observation Message
```json
{
  "type": "obs",
  "step": 42,
  "obs": "base64_encoded_image_data"
}
```

### 3. Response Format
The Purple Agent must return actions in the following format:
```json
{
  "type": "action",
  "buttons": [0, 1, 0, 0, ...],  // Button input array
  "camera": [0.0, 0.5]           // Camera movement
}
```

## 🐛 Troubleshooting

### OpenAI API Error
```
ValueError: OPENAI_API_KEY environment variable not set
```
**Solution**: Set environment variable
```bash
export OPENAI_API_KEY="sk-..."
```

### Purple Agent Communication Error
```
RuntimeError: Agent http://... responded with status 'failed'
```
**Checklist**:
1. Verify Purple agent is running
2. Verify URL is correct
3. Verify Purple agent returns correct response format

### Task Config Error
```
FileNotFoundError: Task configs directory not found
```
**Solution**: Verify MCU_benchmark directory is in the correct location

## 📝 Development Notes

### Key Improvements
- ✅ Integrated video evaluation results into metrics
- ✅ Retry logic with exponential backoff
- ✅ Detailed error logging and traceback
- ✅ Environment variable-based API key management
- ✅ Safe handling of None values

### Future Improvements
- [ ] Add unit tests
- [ ] Add integration tests
- [ ] Integrate Prometheus metrics
- [ ] Resource usage monitoring
- [ ] Support for parallel task execution

## 📚 References

- [AgentBeats Competition](https://rdi.berkeley.edu/agentx-agentbeats#custom-tracks)
- [A2A Protocol](https://a2a-protocol.org/)
- [MCU Benchmark](https://github.com/CraftJarvis/MCU)

## 🤝 Contributing

Please report bugs or suggest improvements by creating an issue.

## 📄 License

MIT License
