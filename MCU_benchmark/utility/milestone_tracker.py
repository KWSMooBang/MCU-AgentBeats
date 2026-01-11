'''
Milestone tracking callback for long-term tasks.
Extends RewardsCallback to save milestone achievement data to JSON.

Date: 2026-01-11
'''

import time
import json
from pathlib import Path
from minestudio.simulator.callbacks import RewardsCallback


class MilestoneTrackerCallback(RewardsCallback):
    """
    Enhanced RewardsCallback that tracks milestone achievements with timestamps.

    Tracks:
    - Milestone achievement timestamps
    - Step numbers when achieved
    - Reward values
    - Cumulative progress

    Saves milestone data to JSON for easy evaluation and analysis.

    Args:
        reward_cfg: List of milestone reward configurations (same as RewardsCallback)
        output_path: Path to save milestone_tracking.json
        task_name: Name of the task (for JSON output)

    Example:
        callback = MilestoneTrackerCallback(
            reward_cfg=milestone_reward_cfg,
            output_path=Path('./output'),
            task_name='kill_ender_dragon'
        )
    """

    def __init__(self, reward_cfg, output_path, task_name='unknown'):
        super().__init__(reward_cfg)
        self.output_path = Path(output_path)
        self.task_name = task_name
        self.milestone_log = []  # List of achievement events
        self.start_time = None

    def after_reset(self, sim, obs, info):
        """Initialize tracking on episode reset."""
        self.start_time = time.time()
        self.milestone_log = []
        return super().after_reset(sim, obs, info)

    def after_step(self, sim, obs, reward, terminated, truncated, info):
        """Track milestone achievements during episode."""
        # Call parent to get reward and milestone updates
        obs, override_reward, terminated, truncated, info = super().after_step(
            sim, obs, reward, terminated, truncated, info
        )

        # Track new milestone achievements
        if override_reward > 0:
            # Check which milestones were newly achieved
            for cfg in self.reward_cfg:
                identity = cfg['identity']
                # If this milestone is now in memory but not yet logged
                if self.reward_memory.get(identity, 0) > 0 and \
                   identity not in [m['identity'] for m in self.milestone_log]:
                    self.milestone_log.append({
                        'identity': identity,
                        'step': self.current_step,
                        'timestamp': time.time() - self.start_time,
                        'reward': cfg['reward']
                    })

        return obs, override_reward, terminated, truncated, info

    def before_close(self, sim):
        """Save milestone data to JSON before closing."""
        self._save_milestone_json()

    def _save_milestone_json(self):
        """Save milestone tracking data to JSON file."""
        # Calculate summary statistics
        total_reward = sum(m['reward'] for m in self.milestone_log)
        milestones_achieved = len(self.milestone_log)
        total_milestones = len(self.reward_cfg)
        completion_rate = milestones_achieved / total_milestones if total_milestones > 0 else 0.0

        # Build milestone summary (all milestones with achievement status)
        milestone_summary = {}
        for cfg in self.reward_cfg:
            identity = cfg['identity']
            # Find if this milestone was achieved
            achieved_milestone = next(
                (m for m in self.milestone_log if m['identity'] == identity),
                None
            )

            if achieved_milestone:
                milestone_summary[identity] = {
                    'achieved': True,
                    'step': achieved_milestone['step'],
                    'timestamp': achieved_milestone['timestamp']
                }
            else:
                milestone_summary[identity] = {
                    'achieved': False,
                    'step': None,
                    'timestamp': None
                }

        # Construct output JSON
        output = {
            'task': self.task_name,
            'total_steps': self.current_step,
            'total_reward': total_reward,
            'milestones_achieved': milestones_achieved,
            'total_milestones': total_milestones,
            'completion_rate': completion_rate,
            'milestones': self.milestone_log,  # Sequential log
            'milestone_summary': milestone_summary  # Summary view
        }

        # Save to file
        output_file = self.output_path / 'milestone_tracking.json'
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"[green]Milestone data saved to: {output_file}[/green]")
