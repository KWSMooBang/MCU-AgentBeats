# Example evaluation request configurations

# Example 1: Simple evaluation with all tasks
SIMPLE_ALL_TASKS = {
    "participants": {
        "agent": "http://purple-agent:8080"
    },
    "config": {
        "difficulty": "simple",
        "task_names": None,
        "num_tasks": None,
        "max_steps": 1000,
        "enable_video_eval": False
    }
}

# Example 2: Hard evaluation with specific tasks
HARD_SPECIFIC_TASKS = {
    "participants": {
        "agent": "http://purple-agent:8080"
    },
    "config": {
        "difficulty": "hard",
        "task_names": ["build_house", "craft_diamond_pickaxe"],
        "max_steps": 2000,
        "enable_video_eval": False
    }
}

# Example 3: Simple evaluation with video analysis
SIMPLE_WITH_VIDEO = {
    "participants": {
        "agent": "http://purple-agent:8080"
    },
    "config": {
        "difficulty": "simple",
        "num_tasks": 5,
        "max_steps": 1000,
        "enable_video_eval": True,
        "rule_file": "/workspace/Agent-AI/MCU-AgentBeats/MCU_benchmark/auto_eval/rules/simple_rules.txt"
    }
}

# Example 4: Quick test with limited tasks
QUICK_TEST = {
    "participants": {
        "agent": "http://localhost:8080"
    },
    "config": {
        "difficulty": "simple",
        "num_tasks": 2,
        "max_steps": 500,
        "enable_video_eval": False
    }
}
