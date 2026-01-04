#!/usr/bin/env python3
"""
Example script to test the MCU Green Agent.
"""
import asyncio
import json
import httpx
from uuid import uuid4


async def send_eval_request(
    green_agent_url: str,
    purple_agent_url: str,
    difficulty: str = "simple",
    num_tasks: int = 2,
):
    """Send an evaluation request to the green agent."""
    
    request = {
        "participants": {
            "agent": purple_agent_url
        },
        "config": {
            "difficulty": difficulty,
            "num_tasks": num_tasks,
            "max_steps": 1000,
            "enable_video_eval": False
        }
    }
    
    message = {
        "kind": "message",
        "role": "user",
        "parts": [{"kind": "text", "text": json.dumps(request)}],
        "message_id": uuid4().hex,
        "context_id": None
    }
    
    print(f"🚀 Sending evaluation request to {green_agent_url}")
    print(f"📝 Config: difficulty={difficulty}, num_tasks={num_tasks}")
    print(f"🎯 Target agent: {purple_agent_url}")
    
    async with httpx.AsyncClient(timeout=600.0) as client:
        try:
            response = await client.post(
                f"{green_agent_url}/messages",
                json=message
            )
            response.raise_for_status()
            
            result = response.json()
            print("\n✅ Evaluation completed!")
            print(json.dumps(result, indent=2))
            
        except httpx.HTTPError as e:
            print(f"\n❌ HTTP Error: {e}")
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test MCU Green Agent")
    parser.add_argument(
        "--green-agent",
        default="http://localhost:9009",
        help="Green agent URL (default: http://localhost:9009)"
    )
    parser.add_argument(
        "--purple-agent",
        default="http://localhost:8080",
        help="Purple agent URL (default: http://localhost:8080)"
    )
    parser.add_argument(
        "--difficulty",
        choices=["simple", "hard"],
        default="simple",
        help="Task difficulty (default: simple)"
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=2,
        help="Number of tasks to run (default: 2)"
    )
    
    args = parser.parse_args()
    
    asyncio.run(send_eval_request(
        green_agent_url=args.green_agent,
        purple_agent_url=args.purple_agent,
        difficulty=args.difficulty,
        num_tasks=args.num_tasks
    ))
