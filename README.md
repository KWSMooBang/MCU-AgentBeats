# MCU Green Agent - AgentBeats Competition

A comprehensive Minecraft-based benchmark evaluation agent for evaluating embodied AI agents' planning, reasoning, and multi-step execution capabilities.
Green Agent evaluates Purple Agent in complex, open-world environments using the A2A protocol.

## 🎯 Overview

This Green Agent provides a rigorous evaluation framework that goes beyond simple task completion metrics. It evaluates:

- **Multi-step Planning**: Complex tasks requiring sequential action planning (e.g., crafting requires gathering materials → using crafting table → combining items)
- **Spatial Reasoning**: Navigation, building, and exploration in 3D space
- **Tool Use**: Strategic use of items and environmental interactions
- **Resource Management**: Efficient use of materials and time under constraints
- **Adaptability**: Handling diverse task types across 10 distinct categories

### Key Features

✅ **Diverse Tasks** across 10 categories  
✅ **Nuanced Multi-dimensional Evaluation** (simulation + video-based assessment)  
✅ **Reproducible & Consistent** results with deterministic task initialization  
✅ **A2A Protocol Compatible** - works with any compliant agent  
✅ **Automated + Human-interpretable** scoring  
✅ **Resource Efficient** - configurable step limits and category selection

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker (for containerized deployment)
- OpenAI API key (for video evaluation)

### 1. Installation & Environment Setup
First, clone repository and install dependencies:
```bash
# Clone repository
git clone https://github.com/KWSMooBang/MCU-AgentBeats.git
cd MCU-AgentBeats

# Install dependencies with uv
uv sync
```

Then, Create a `.env` file in the project root:
```bash
# Required for video evaluation
OPENAI_API_KEY=your-openai-api-key-here
```

### 3. Run Green Agent Server

#### Running Locally
```bash
uv python src/server.py --host 0.0.0.0 --port 9009

# With custom agent card URL
python src/server.py --host 0.0.0.0 --port 9009 --card-url "http://localhost:9009"
```

#### Running with Docker
```bash
docker build -t mcu-green-agent .
docker run \ 
  -p 9009:9009 \ 
  --env-file .env \ 
  mcu-green-agent \
  --host 0.0.0.0 \
  --port 9009
```

**Server Health Check:**
```bash
curl http://localhost:9009/.well-known/agent-card.json
```

## 📊 Test Evaluation

### Step 1: Start Purple Agent Server (Test Agent)

The repository includes a sample Purple Agent using STEVE-1 model for testing:

```bash
# In a separate terminal, start the test Purple Agent
uv run python test_purple_agent.py --host 0.0.0.0 --port 9019

# When you run green agent server with Docker
uv run python test_purple_agent.py --card-url "http://host.docker.internal:9019"
```

### Step 2: Configure Test Scenario

Edit `test_scenario.toml` to configure your evaluation:

```toml
[green_agent]
endpoint = "http://127.0.0.1:9009"

[[participants]]
role = "agent"
endpoint = http://127.0.0.1:9019" or # "http://host.docker.internal:9019" (with Docker)

[config]
task_category = ["hunt"]  # Change to desired categories
max_steps = 900           # Optional: customize step limit
```

**Available categories**: `build`, `craft`, `combat`, `explore`, `mine`, `hunt`, `collect`, `use`, `find`, `misc`

### Step 3: Run Evaluation

```bash
# Run evaluation with scenario file
uv run python test_evaluation.py test_scenario.toml

# Save results to JSON file
uv run python test_evaluation.py test_scenario.toml output/results.json
```

**Expected Output:**
```
[Status: submitted]
Starting MCU evaluation with 2 tasks from hunt

[Status: working]
Running task: hunt pigs (category: hunt)

[Status: working]
Running task: hunt horses (category: hunt)

[Status: completed]
MCU Evaluation Result
Categories: hunt
Number of Tasks: 2
Total Score: 15.3

Task Results:
Task 'hunt_pigs': 8.1
Task 'hunt_horses': 7.2
```

## ⚙️ Configuration Parameters

### Required
- **participants.agent** (str): URL of the Purple Agent to evaluate
  - Example: `"http://localhost:8080"` or `"http://purple-agent:8080"`
  - Must be A2A protocol compatible

### Optional
- **task_category** (list[str]): Task categories to evaluate
  - Empty list `[]`: All categories except 'overall' (default)
  - `["craft"]`: Only craft tasks (11 tasks)
  - `["craft", "build"]`: Craft and build tasks (~23 tasks)
  - `["combat", "mine", "explore"]`: Combat, mining, exploration
  - Available categories: `build`, `craft`, `combat`, `explore`, `mine`, `hunt`, `collect`, `use`, `find`, `misc`

- **max_steps** (int): Maximum steps per task (default: 900)
  - Recommended: 500-1500 depending on task complexity
  - Lower values test efficiency, higher values allow complex strategies

## 📁 Project Structure

```
MCU-AgentBeats/
├── src/
│   ├── server.py           # A2A server entry point
│   ├── agent.py            # Main evaluation logic
│   ├── messenger.py        # Purple agent communication
│   ├── executor.py         # Task execution management
│   ├── model.py            # Pydantic data models
│   └── util.py             # Helper functions (task loading, video processing)
│
├── MCU_benchmark/
│   ├── task_configs/
│   │   ├── tasks/          # Task definitions by category
│   │   │   ├── build/      # Building tasks
│   │   │   ├── craft/      # Crafting tasks
│   │   │   ├── combat/     # Combat tasks
│   │   │   ├── explore/    # Exploration tasks
│   │   │   ├── mine/       # Mining tasks
│   │   │   ├── hunt/       # Hunting tasks
│   │   │   ├── collect/    # Collection tasks
│   │   │   ├── use/        # Item usage tasks
│   │   │   ├── find/       # Finding tasks
│   │   │   └── misc/       # Miscellaneous tasks
│   │   ├── simple/         # Legacy simple tasks
│   │   └── hard/           # Legacy hard tasks
│   │
│   └── auto_eval/
│       ├── prompt/         # Video evaluation prompts
│       └── criteria_files/ # Task-specific evaluation criteria
│
├── output/                 # Evaluation results and recordings
├── logs/                   # Server logs
└── pyproject.toml          # Project dependencies
```

## 🏆 Benchmark Design Quality

### Why Minecraft for Agent Evaluation?

Minecraft provides a rich, open-world environment that closely mirrors real-world agent challenges:

1. **Complex State Space**: Thousands of block types, items, and environmental states
2. **Long-horizon Tasks**: Require 100s-1000s of steps with intermediate goals
3. **Emergent Complexity**: Simple actions combine into complex behaviors
4. **Grounded Multimodal Input**: Visual observation (RGB images) with spatial reasoning
5. **Diverse Skill Requirements**: Navigation, crafting, combat, exploration, construction

### Task Categories & Progression

Our benchmark includes **10 distinct categories**, each testing different capabilities:

| Category | Skills Tested | Example Tasks | Difficulty |
|----------|---------------|---------------|------------|
| **Craft** | Sequential planning, recipe knowledge | Craft enchanting table, clock | ⭐⭐⭐ |
| **Build** | Spatial reasoning, multi-step execution | Build house, tower, maze | ⭐⭐⭐ |
| **Combat** | Real-time decision making, positioning | Combat zombies, skeletons | ⭐⭐ |
| **Mine** | Navigation, tool use, resource gathering | Mine diamond ore, obsidian | ⭐⭐ |
| **Explore** | Pathfinding, environment interaction | Explore chest, climb ladder | ⭐⭐ |
| **Hunt** | Entity tracking, strategic combat | Hunt pigs, horses | ⭐⭐ |
| **Collect** | Efficient resource gathering | Collect wood, wool, dirt | ⭐ |
| **Use** | Tool understanding, context-aware actions | Use bow, shield, trident | ⭐⭐ |
| **Find** | Visual search, navigation | Find diamond, village, bedrock | ⭐⭐⭐ |
| **Misc** | Creative problem solving, combinations | Trap entities, light surroundings | ⭐⭐⭐ |

### Task Difficulty Progression

- **Level 1 (⭐)**: Single-step or direct actions (collect, drop items)
- **Level 2 (⭐⭐)**: Multi-step with clear sub-goals (combat, mining)
- **Level 3 (⭐⭐⭐)**: Complex planning with multiple dependencies (crafting, building)

### Avoiding Trivial Solutions

Each task is designed to prevent simple heuristics:

- **Randomized Initialization**: Entity positions and initial states vary
- **Constrained Resources**: Limited materials prevent brute-force approaches
- **Time Limits**: 900 steps max per task encourages efficiency
- **Reward Shaping**: Tasks reward optimal strategies, not just completion

## 📊 Evaluation Methodology

### Multi-dimensional Scoring

We use **two complementary evaluation metrics** in original MCU benchmark:

#### 1. Simulation-based Scoring (Automated)
- **Real-time performance tracking** during task execution
- **Objective, deterministic** metrics
- **Efficient** - no human annotation required

#### 2. Video-based Evaluation (GPT-4 Vision)
Provides nuanced assessment across **6 dimensions** :

| Criterion | Description | Weight |
|-----------|-------------|--------|
| **Task Progress** | How far the agent progressed toward the goal | High |
| **Action Control** | Precision and appropriateness of actions | High |
| **Error Recognition** | Ability to detect and correct mistakes | Medium |
| **Creative Attempts** | Novel or intelligent problem-solving strategies | Medium |
| **Task Efficiency** | Speed and resource optimization | High |
| **Material Selection** | Correct choice and use of tools/items | Medium |

**Scoring Range**: 0-10 per criterion, averaged for final score

### Why Multi-dimensional Evaluation?

Single metrics (pass/fail) miss important nuances:
- An agent might complete a task inefficiently (wasted resources)
- An agent might fail but demonstrate good partial planning
- An agent might succeed by luck rather than reasoning

Our dual scoring captures both **outcomes and process quality**.

### Scoring Example

```
Task: Craft Oak Planks
Agent A: Completes in 50 steps, uses crafting table correctly
  → Simulation Score: 10.0 (success)
  → Video Score: 9.2 (efficient, correct tool use)

Agent B: Completes in 500 steps, many failed attempts
  → Simulation Score: 10.0 (success)
  → Video Score: 6.5 (inefficient, poor planning)

Agent C: 80% progress, correct approach but ran out of time
  → Simulation Score: 0.0 (incomplete)
  → Video Score: 7.8 (good understanding, time management issue)
```

## 📈 Evaluation Results

### Output Directory Structure

Results are saved in `output/{timestamp}/` directory with complete reproducibility:

```
output/20260111_143000/
├── result.txt                          # Human-readable summary
├── craft_oak_planks/
│   ├── episode_1.mp4                   # Task video recording (30 FPS)
│   └── video_eval_result.json          # GPT-4 Vision evaluation
├── build_a_house/
│   ├── episode_1.mp4
│   └── video_eval_result.json
├── combat_zombies/
│   ├── episode_1.mp4
│   └── video_eval_result.json
└── ...
```

### Result Format

**Summary (result.txt):**
```
MCU Evaluation Result
Categories: craft, build
Number of Tasks: 23
Total Score: 195.3

Task Results:
Task 'craft_oak_planks': 9.2
Task 'craft_ladder': 8.7
Task 'build_a_house': 8.1
...
```

**JSON Output:**
```json
{
  "task_category": ["craft", "build"],
  "num_tasks": 23,
  "total_score": 195.3,
  "task_metrics": {
    "craft_oak_planks": 9.2,
    "craft_ladder": 8.7,
    "build_a_house": 8.1,
    ...
  }
}
```

**Individual Task Evaluation (video_eval_result.json):**
```json
{
  "task": "craft_oak_planks",
  "video_path": "output/.../craft_oak_planks/episode_1.mp4",
  "Task Progress": 10.0,
  "Action Control": 9.0,
  "Error Recognition and Correction": 8.5,
  "Creative Attempts": 9.0,
  "Task Completion Efficiency": 9.5,
  "Material Selection and Usage": 9.2,
  "final score": 9.2,
  "origin response": "..."
}
```

### Metrics Explanation

- **total_score**: Sum of all video evaluation scores (0-10 per task)
- **task_metrics**: Individual task scores based on 6-dimensional GPT-4 Vision assessment
- **video_score**: Averaged score across all evaluation criteria
- **final score**: Overall benchmark performance (higher is better)

### Performance Benchmarks

| Agent Type | Avg Score | Interpretation |
|------------|-----------|----------------|
| 9.0 - 10.0 | Expert | Near-perfect execution, efficient strategies |
| 7.0 - 8.9 | Proficient | Completes tasks with minor inefficiencies |
| 5.0 - 6.9 | Competent | Basic task completion, needs optimization |
| 3.0 - 4.9 | Novice | Partial success, significant errors |
| 0.0 - 2.9 | Struggling | Minimal progress or task understanding |

## 🔄 Reproducibility

### Deterministic Task Initialization

Every task uses **predefined initialization commands** ensuring:
- ✅ Identical starting states across runs
- ✅ Same resources and environment layout
- ✅ Consistent entity spawning
- ✅ Reproducible random seeds

### Consistency Guarantees

```yaml
# Example: craft_oak_planks.yaml
custom_init_commands:
  - /give @s minecraft:oak_log 10      # Always 10 logs
  - /give @s minecraft:crafting_table   # Always 1 table
  - /give @s minecraft:apple 5          # Always 5 apples
reward_cfg:
  - event: craft_item
    objects: [oak_planks]
    reward: 10.0                        # Fixed reward
    max_reward_times: 1                 # Success threshold
```

### Cross-run Validation

We ensure reproducibility through:
1. **Fixed step limits** (900 steps per task)
2. **Deterministic environment** (same Minecraft version, mods)
3. **Standardized observation** (128x128 RGB images)
4. **Version-controlled task configs** (all tasks in YAML)

### Easy Integration

Any A2A-compatible agent can run the benchmark:
- **Standard protocol** - no custom modifications needed
- **Clear API contract** - well-defined message formats
- **Docker containerization** - consistent runtime environment
- **Comprehensive documentation** - setup takes <10 minutes

## 💡 Innovation & Impact

### Original Contributions

1. **First A2A-compatible Minecraft Benchmark**
   - Extends MCU benchmark with multi-agent evaluation capabilities
   - Enables distributed agent evaluation at scale

2. **Hybrid Evaluation Methodology**
   - Combines quantitative (simulation) + qualitative (video) metrics
   - Goes beyond binary pass/fail to nuanced performance assessment

3. **Task Reward Configuration System**
   - Flexible reward shaping per task
   - Supports incremental progress tracking
   - Configurable success criteria

4. **Category-based Organization**
   - 10 skill-specific categories vs. flat task list
   - Enables targeted capability assessment
   - Supports progressive difficulty testing

### Addressing Evaluation Gaps

| Existing Benchmarks | MCU AgentBeats Green Agent |
|---------------------|----------------------------|
| Text-only tasks | **Grounded visual perception** |
| Single-step actions | **Multi-step planning (100+ steps)** |
| Static environments | **Dynamic, interactive world** |
| Binary scoring | **Multi-dimensional evaluation** |
| Predefined solutions | **Open-ended problem solving** |

### Use Cases & Target Audience

**Research**: Benchmark embodied AI agents in complex environments  
**Development**: Test agent planning and reasoning capabilities  
**Competition**: Fair, standardized evaluation for agent competitions  
**Industry**: Assess AI agents for robotics and autonomous systems

### Complementary to Existing Benchmarks

- **vs. AlfWorld**: MCU adds visual grounding and spatial reasoning
- **vs. BabyAI**: MCU provides realistic, high-dimensional observations
- **vs. Habitat**: MCU adds tool use and crafting mechanics
- **vs. MineDojo**: MCU focuses on structured evaluation vs. open-ended exploration

## 🧪 Testing

### Test Individual Components

```bash
# Test task loading
python test_extract_info.py

# Test Purple Agent communication
python -m pytest test/ -v

# Run specific task
python MCU_benchmark/run_task.py --task craft_oak_planks
```

### Integration Test

```bash
# Start Purple Agent (example)
cd ../purple-agent && python server.py --port 8080

# Start Green Agent
python src/server.py --port 9009

# Send test request
python client.py --agent http://localhost:8080 --category craft
```

## 🔧 Purple Agent Requirements

Your Purple Agent must implement the following A2A message handlers:

### 1. Initialization
**Request:**
```json
{
  "text": "craft oak planks from oak logs"
}
```
**Response:**
```json
{
  "success": true,
  "message": "Ready"
}
```

### 2. Observation
**Request:**
```json
{
  "step": 42,
  "obs": "<base64_encoded_128x128_image>"
}
```
**Response:**
```json
{
  "buttons": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  "camera": [0, 60]
}
```

### Action Space
- **buttons**: Array of 23 integers (0 or 1)
  - Forward, Back, Left, Right, Jump, Sneak, Sprint, Attack, Use, etc.
- **camera**: Array of 2 integers [yaw, pitch]
  - Range: typically [-180, 180] for yaw, [-90, 90] for pitch

## 🐛 Troubleshooting

### Common Issues

**1. Docker credential error**
```
ERROR: failed to solve: error getting credentials - err: exec: "docker-credential-desktop.exe"
```
**Solution:**
```bash
echo '{"auths":{}}' > ~/.docker/config.json
chmod 444 ~/.docker/config.json
```

**2. OpenAI API error**
```
ValueError: OPENAI_API_KEY environment variable not set
```
**Solution:** Set the environment variable or disable video evaluation

**3. Task config not found**
```
FileNotFoundError: Task configs directory not found
```
**Solution:** Ensure you're running from the project root directory

**4. Purple Agent timeout**
```
httpx.ReadTimeout: timed out
```
**Solution:** Increase timeout or optimize Purple Agent response time

## 📝 Task Configuration

Each task YAML file contains:
```yaml
custom_init_commands:
  - /give @s minecraft:oak_log 10
  - /give @s minecraft:crafting_table
text: craft oak planks from oak logs
reward_cfg:
  - event: craft_item
    identity: craft_oak_planks
    objects: [oak_planks]
    reward: 10.0
    max_reward_times: 1
```

## 🔬 Development

### Adding New Tasks

1. Create YAML file in appropriate category folder
2. Define init commands, task text, and reward config
3. Test with `run_task.py`

### Modifying Evaluation Logic

Main files to modify:
- `src/agent.py`: Core evaluation loop
- `src/util.py`: Task loading and video processing
- `MCU_benchmark/auto_eval/`: Video evaluation prompts

## 📚 References

- [AgentBeats Competition](https://rdi.berkeley.edu/agentx-agentbeats#custom-tracks)
- [A2A Protocol](https://a2a-protocol.org/)
- [MCU Benchmark](https://github.com/CraftJarvis/MCU)
- [MineStudio Framework](https://github.com/CraftJarvis/MineStudio)

## 📄 License

MIT License

---

For questions or issues, please create an issue on GitHub or contact the maintainers.
