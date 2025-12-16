"""
Custom Gym Environment for Nightmare Comfort RL Training
"""
import gym
from gym import spaces
import numpy as np
import torch
from transformers import GPT2Tokenizer
from typing import Dict, Tuple, List, Any
import json
import random

class NightmareComfortEnv(gym.Env):
    """
    OpenAI Gym environment for training comfort response generation
    
    State: Nightmare text (tokenized)
    Action: Comfort text (generated tokens)
    Reward: Quality score from reward model
    """
    
    def __init__(
        self,
        nightmares_file: str,
        reward_model,
        tokenizer_name: str = 'gpt2-medium',
        max_length: int = 100,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        
        self.device = device
        self.max_length = max_length
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load nightmare dataset
        self.nightmares = self._load_nightmares(nightmares_file)
        self.current_nightmare = None
        self.current_nightmare_text = None
        
        # Reward model
        self.reward_model = reward_model
        
        # Define action and observation spaces
        # Observation: tokenized nightmare (fixed length)
        self.observation_space = spaces.Box(
            low=0,
            high=self.tokenizer.vocab_size,
            shape=(max_length,),
            dtype=np.int32
        )
        
        # Action: generated comfort text (we'll handle this differently in PPO)
        # For now, this is just a placeholder
        self.action_space = spaces.Discrete(self.tokenizer.vocab_size)
        
        # Episode tracking
        self.episode_length = 0
        self.max_episode_length = 1  # One response per nightmare
        
    def _load_nightmares(self, filepath: str) -> List[str]:
        """Load nightmare descriptions from file"""
        nightmares = []
        
        # Try JSON first
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    nightmares = [item if isinstance(item, str) else item.get('nightmare', '') for item in data]
                elif isinstance(data, dict):
                    nightmares = list(data.values())
        except json.JSONDecodeError:
            # Try text file
            with open(filepath, 'r', encoding='utf-8') as f:
                nightmares = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(nightmares)} nightmares from {filepath}")
        return nightmares
    
    def reset(self) -> np.ndarray:
        """
        Reset environment and return initial observation (nightmare)
        """
        # Select random nightmare
        self.current_nightmare_text = random.choice(self.nightmares)
        
        # Tokenize nightmare
        tokens = self.tokenizer.encode(
            self.current_nightmare_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='np'
        )
        
        self.current_nightmare = tokens.squeeze()
        self.episode_length = 0
        
        return self.current_nightmare
    
    def step(self, action: str) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment
        
        Args:
            action: Generated comfort text (string)
            
        Returns:
            observation: Next state (same nightmare for simplicity)
            reward: Reward from reward model
            done: Episode completion flag
            info: Additional information
        """
        # Calculate reward using reward model
        reward = self.reward_model.calculate_reward(
            self.current_nightmare_text,
            action
        )
        
        # Episode ends after one response
        self.episode_length += 1
        done = self.episode_length >= self.max_episode_length
        
        # Info dict for logging
        info = {
            'nightmare': self.current_nightmare_text,
            'comfort': action,
            'reward': reward,
            'comfort_length': len(action.split())
        }
        
        # Next observation (same nightmare)
        next_obs = self.current_nightmare
        
        return next_obs, reward, done, info
    
    def render(self, mode='human'):
        """Render the environment"""
        if mode == 'human':
            print(f"\nNightmare: {self.current_nightmare_text}")


class BatchedNightmareEnv:
    """
    Vectorized environment for parallel training
    Processes multiple nightmares simultaneously
    """
    def __init__(
        self,
        nightmares_file: str,
        reward_model,
        batch_size: int = 8,
        tokenizer_name: str = 'gpt2-medium',
        max_length: int = 100,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.batch_size = batch_size
        self.device = device
        self.max_length = max_length
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load nightmares
        with open(nightmares_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    self.nightmares = [item if isinstance(item, str) else item.get('nightmare', '') for item in data]
                else:
                    self.nightmares = list(data.values())
            except json.JSONDecodeError:
                f.seek(0)
                self.nightmares = [line.strip() for line in f if line.strip()]
        
        self.reward_model = reward_model
        self.current_nightmares = []
        self.current_nightmare_tokens = None
        
        print(f"Initialized batched environment with {len(self.nightmares)} nightmares")
    
    def reset(self) -> torch.Tensor:
        """Reset and return batch of nightmares"""
        # Sample batch of nightmares
        self.current_nightmares = random.choices(self.nightmares, k=self.batch_size)
        
        # Tokenize all nightmares
        tokens = self.tokenizer(
            self.current_nightmares,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        self.current_nightmare_tokens = tokens['input_ids'].to(self.device)
        return self.current_nightmare_tokens
    
    def step(self, actions: List[str]) -> Tuple[torch.Tensor, torch.Tensor, List[bool], List[Dict]]:
        """
        Batched step
        
        Args:
            actions: List of generated comfort texts
            
        Returns:
            observations, rewards, dones, infos
        """
        # Calculate rewards for all actions
        rewards = []
        infos = []
        
        for nightmare, comfort in zip(self.current_nightmares, actions):
            reward = self.reward_model.calculate_reward(nightmare, comfort)
            rewards.append(reward)
            
            infos.append({
                'nightmare': nightmare,
                'comfort': comfort,
                'reward': reward
            })
        
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = [True] * self.batch_size  # Each episode is one interaction
        
        return self.current_nightmare_tokens, rewards_tensor, dones, infos


# Wrapper for compatibility with stable-baselines3 or trl
class GymWrapper(gym.Env):
    """Wrapper to make environment compatible with standard RL libraries"""
    
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
    
    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        return self.env.step(action)
    
    def render(self, mode='human'):
        return self.env.render(mode)
    
    def close(self):
        pass


if __name__ == "__main__":
    # Test the environment
    from reward_model import HeuristicRewardModel
    
    print("Testing Nightmare Comfort Environment...")
    print("="*70)
    
    # Create reward model
    reward_model = HeuristicRewardModel()
    
    # Create a dummy nightmares file for testing
    test_nightmares = [
        "I'm falling from a tall building",
        "Everyone is staring at me and laughing",
        "I'm trapped in a small space with no way out",
        "I'm being chased through dark streets",
        "I have an important presentation and I'm not ready"
    ]
    
    with open('test_nightmares.txt', 'w') as f:
        for nightmare in test_nightmares:
            f.write(nightmare + '\n')
    
    # Initialize environment
    env = NightmareComfortEnv(
        nightmares_file='test_nightmares.txt',
        reward_model=reward_model,
        max_length=50
    )
    
    print("\n1. Testing Single Environment")
    print("-"*70)
    
    # Test episode
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Current nightmare: {env.current_nightmare_text}")
    
    # Simulate action (comfort response)
    test_comfort = "You notice your breath, steady and real. You realize you're safe in your bed. The fear transforms into calm awareness."
    
    next_obs, reward, done, info = env.step(test_comfort)
    print(f"\nGenerated comfort: {test_comfort}")
    print(f"Reward: {reward:.3f}")
    print(f"Done: {done}")
    print(f"Info: {info}")
    
    print("\n2. Testing Batched Environment")
    print("-"*70)
    
    batched_env = BatchedNightmareEnv(
        nightmares_file='test_nightmares.txt',
        reward_model=reward_model,
        batch_size=3
    )
    
    batch_obs = batched_env.reset()
    print(f"Batch observation shape: {batch_obs.shape}")
    print(f"Current nightmares:")
    for i, nightmare in enumerate(batched_env.current_nightmares):
        print(f"  {i+1}. {nightmare}")
    
    # Simulate batch actions
    batch_comforts = [
        "You're safe now. Take a deep breath.",
        "Notice the ground beneath you. You're stable and secure.",
        "The fear dissolves as you recognize this is just a dream."
    ]
    
    _, batch_rewards, batch_dones, batch_infos = batched_env.step(batch_comforts)
    print(f"\nBatch rewards: {batch_rewards}")
    print(f"Average reward: {batch_rewards.mean().item():.3f}")
    
    print("\n" + "="*70)
    print("Environment testing complete!")