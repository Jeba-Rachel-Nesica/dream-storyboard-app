"""
PPO Training for Nightmare Comfort Generation
Single-Agent RL with GPT-2
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
import numpy as np
from tqdm import tqdm
from typing import List, Dict
import json
import os

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Logging to W&B will be disabled.")

from reward_model import HeuristicRewardModel, ComfortRewardModel
from nightmare_comfort_env import BatchedNightmareEnv


class ComfortPPOTrainer:
    """
    PPO Trainer for Nightmare Comfort Generation
    Uses TRL (Transformer Reinforcement Learning) library
    """
    
    def __init__(
        self,
        model_path: str,
        nightmares_file: str,
        output_dir: str = 'checkpoints/ppo_comfort',
        use_neural_reward: bool = False,
        batch_size: int = 8,
        learning_rate: float = 1.41e-5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load tokenizer
        print(f"Loading tokenizer and model from {model_path}...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path, use_fast=True)

        # Give the model a real PAD token, left padding, and align config
        if (self.tokenizer.pad_token is None) or (self.tokenizer.pad_token_id == self.tokenizer.eos_token_id):
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokenizer.padding_side = "left"
        
        # Load model with value head for PPO
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path)

        # resize embeddings on the wrapped base model, not on the value-head wrapper
        base_model = getattr(self.model, "pretrained_model", None)
        if base_model is None:
            base_model = getattr(self.model, "transformer", None)  # GPT-2 fallback
        if base_model is not None:
            base_model.resize_token_embeddings(len(self.tokenizer))

        self.model = self.model.to(device)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.use_cache = False  # safer for training
        
        # Reference model (frozen copy for KL divergence)
        self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path)

        # same fix for the ref model
        ref_base = getattr(self.ref_model, "pretrained_model", None)
        if ref_base is None:
            ref_base = getattr(self.ref_model, "transformer", None)  # GPT-2 fallback
        if ref_base is not None:
            ref_base.resize_token_embeddings(len(self.tokenizer))

        self.ref_model = self.ref_model.to(device)
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # Initialize reward model
        print("Initializing reward model...")
        if use_neural_reward:
            self.reward_model = ComfortRewardModel()
            if torch.cuda.is_available():
                self.reward_model = self.reward_model.cuda()
        else:
            self.reward_model = HeuristicRewardModel()
        
        # Load nightmares dataset
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
        
        print(f"Loaded {len(self.nightmares)} nightmares")
        
        # PPO Configuration
        self.ppo_config = PPOConfig(
            model_name=model_path,
            learning_rate=learning_rate,
            batch_size=batch_size,
            mini_batch_size=batch_size,
            gradient_accumulation_steps=1,
            optimize_cuda_cache=True,
            early_stopping=False,
            target_kl=6.0,
            ppo_epochs=4,
            seed=42,
            steps=20000,
            init_kl_coef=0.2,
            adap_kl_ctrl=True,
            cliprange=0.1,
            cliprange_value=0.1,
        )
        
        # Initialize PPO trainer
        self.ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
        )
        
        # Generation settings
        self.generation_kwargs = {
            "min_length": -1,
            "do_sample": True,
            "top_p": 0.9,
            "temperature": 0.7,
            "top_k": 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "max_new_tokens": 64,
        }
    
    def prepare_batch(self, nightmares: List[str]) -> Dict:
        """Prepare batch of nightmares for generation"""
        # Add prompt format (same as your training)
        queries = [f"Nightmare: {nightmare}\nComfort:" for nightmare in nightmares]
        
        enc = self.tokenizer(
            queries,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        query_tensors = [input_ids[i].detach().clone() for i in range(input_ids.size(0))]
        mask_tensors  = [attention_mask[i].detach().clone() for i in range(attention_mask.size(0))]
        
        return query_tensors, mask_tensors, queries
    
    def generate_responses(self, query_tensors: List[torch.Tensor], mask_tensors: List[torch.Tensor]) -> List[str]:
        """Generate comfort responses for batch of nightmares"""
        responses = []
        response_tensors = []
        
        for query_tensor, mask_tensor in zip(query_tensors, mask_tensors):
            if query_tensor.dim() > 1:
                query_tensor = query_tensor.squeeze(0)
            if mask_tensor.dim() > 1:
                mask_tensor = mask_tensor.squeeze(0)

            # >>> ONLY CHANGE HERE: make mask 2D to match TRL's internal unsqueeze of input_ids
            generated = self.ppo_trainer.generate(
                query_tensor,
                attention_mask=mask_tensor.unsqueeze(0),
                **self.generation_kwargs
            )
            # <<< END CHANGE

            if generated.dim() > 1:
                generated = generated[0]

            prompt_len = query_tensor.size(0)
            response_only = generated[prompt_len:]

            response_text = self.tokenizer.decode(response_only, skip_special_tokens=True).strip()

            if response_text.startswith("Comfort:"):
                response_text = response_text[len("Comfort:"):].strip()
            
            responses.append(response_text)
            response_tensors.append(response_only)
        
        return responses, response_tensors
    
    def calculate_rewards(self, nightmares: List[str], comforts: List[str]) -> List[float]:
        """Calculate rewards for generated comfort responses"""
        rewards = []
        for nightmare, comfort in zip(nightmares, comforts):
            r = float(self.reward_model.calculate_reward(nightmare, comfort))
            r = max(min(r, 1.0), -1.0) * 0.1
            rewards.append(r)
        return rewards
    
    def train(
        self,
        num_epochs: int = 3,
        save_every: int = 500,
        log_wandb: bool = False
    ):
        """
        Main training loop
        """
        if log_wandb and WANDB_AVAILABLE:
            wandb.init(project="nightmare-comfort-ppo", config=self.ppo_config)
        elif log_wandb and not WANDB_AVAILABLE:
            print("Warning: W&B logging requested but wandb not installed. Skipping W&B logging.")
        
        print("\n" + "="*70)
        print("Starting PPO Training")
        print("="*70)
        
        total_steps = 0
        best_reward = -float('inf')
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Shuffle nightmares
            import random
            random.shuffle(self.nightmares)
            
            # Training loop
            epoch_rewards = []
            
            for i in tqdm(range(0, len(self.nightmares), self.ppo_config.batch_size)):
                batch_nightmares = self.nightmares[i:i + self.ppo_config.batch_size]
                if len(batch_nightmares) < self.ppo_config.batch_size:
                    continue
                
                query_tensors, mask_tensors, queries = self.prepare_batch(batch_nightmares)
                comfort_texts, response_tensors = self.generate_responses(query_tensors, mask_tensors)
                rewards = self.calculate_rewards(batch_nightmares, comfort_texts)
                rewards_tensor = [torch.tensor(r, dtype=torch.float32).to(self.device) for r in rewards]
                
                try:
                    stats = self.ppo_trainer.step(query_tensors, response_tensors, rewards_tensor)
                except Exception as e:
                    print(f"Error in PPO step: {e}")
                    print(f"Query tensor shapes: {[qt.shape for qt in query_tensors]}")
                    print(f"Response tensor shapes: {[rt.shape for rt in response_tensors]}")
                    print(f"Rewards tensor shapes: {[r.shape for r in rewards_tensor]}")
                    raise
                
                epoch_rewards.extend(rewards)
                total_steps += 1
                
                if total_steps % 50 == 0:
                    avg_reward = np.mean(epoch_rewards[-50:]) if len(epoch_rewards) >= 50 else np.mean(epoch_rewards)
                    print(f"\nStep {total_steps} | Avg Reward: {avg_reward:.4f}")
                    print(f"Sample Nightmare: {batch_nightmares[0]}")
                    print(f"Sample Comfort: {comfort_texts[0][:150]}...")
                    print(f"Sample Reward: {rewards[0]:.4f}")
                
                if log_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        "reward": np.mean(rewards),
                        "epoch": epoch,
                        "step": total_steps
                    })
                
                if total_steps % save_every == 0:
                    checkpoint_path = os.path.join(self.output_dir, f"checkpoint-{total_steps}")
                    self.save_model(checkpoint_path)
                    print(f"Saved checkpoint to {checkpoint_path}")
            
            avg_epoch_reward = np.mean(epoch_rewards) if epoch_rewards else 0.0
            print(f"\nEpoch {epoch + 1} Complete | Avg Reward: {avg_epoch_reward:.4f}")
            
            if avg_epoch_reward > best_reward:
                best_reward = avg_epoch_reward
                best_path = os.path.join(self.output_dir, "best_model")
                self.save_model(best_path)
                print(f"New best model saved! Reward: {best_reward:.4f}")
        
        final_path = os.path.join(self.output_dir, "final_model")
        self.save_model(final_path)
        print(f"\nTraining complete! Final model saved to {final_path}")
        
        if log_wandb and WANDB_AVAILABLE:
            wandb.finish()
    
    def save_model(self, path: str):
        """Save model and tokenizer"""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")
    
    def evaluate(self, test_nightmares: List[str]) -> Dict:
        """Evaluate model on test nightmares"""
        self.model.eval()
        
        all_rewards = []
        examples = []
        
        print("\n" + "="*70)
        print("EVALUATION")
        print("="*70)
        
        with torch.no_grad():
            for nightmare in test_nightmares:
                query_tensors, mask_tensors, _ = self.prepare_batch([nightmare])
                comfort_texts, _ = self.generate_responses(query_tensors, mask_tensors)
                comfort = comfort_texts[0]
                
                r = float(self.reward_model.calculate_reward(nightmare, comfort))
                r = max(min(r, 1.0), -1.0) * 0.1
                all_rewards.append(r)
                
                examples.append({
                    'nightmare': nightmare,
                    'comfort': comfort,
                    'reward': r
                })
                
                print(f"\nNightmare: {nightmare}")
                print(f"Comfort: {comfort}")
                print(f"Reward: {r:.4f}")
                print("-"*70)
        
        self.model.train()
        
        avg_reward = np.mean(all_rewards) if all_rewards else 0.0
        print(f"\nAverage Test Reward: {avg_reward:.4f}")
        
        return {
            'avg_reward': avg_reward,
            'examples': examples
        }


# Simplified training script
def train_simple_ppo(
    model_path: str = 'checkpoints/comfort_large',
    nightmares_file: str = 'data/nightmares_train.txt',
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 1.41e-5,
    output_dir: str = 'checkpoints/ppo_comfort'
):
    """
    Simplified PPO training function
    """
    trainer = ComfortPPOTrainer(
        model_path=model_path,
        nightmares_file=nightmares_file,
        output_dir=output_dir,
        use_neural_reward=False,  # Use heuristic reward for simplicity
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    trainer.train(
        num_epochs=num_epochs,
        save_every=500,
        log_wandb=False  # Set True if you want W&B logging
    )
    
    return trainer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PPO for Nightmare Comfort')
    parser.add_argument('--model_path', type=str, default='checkpoints/comfort_large',
                        help='Path to pretrained comfort model')
    parser.add_argument('--nightmares_file', type=str, default='data/final_train.jsonl',
                        help='Path to nightmares dataset')
    parser.add_argument('--output_dir', type=str, default='checkpoints/ppo_comfort',
                        help='Output directory for checkpoints')
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-6,
                        help='Learning rate')
    parser.add_argument('--use_neural_reward', action='store_true',
                        help='Use neural reward model instead of heuristic')
    parser.add_argument('--wandb', action='store_true',
                        help='Log to Weights & Biases')
    
    args = parser.parse_args()
    
    print("PPO Training Configuration:")
    print(f"  Model: {args.model_path}")
    print(f"  Dataset: {args.nightmares_file}")
    print(f"  Output: {args.output_dir}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Neural Reward: {args.use_neural_reward}")
    
    # Train
    trainer = ComfortPPOTrainer(
        model_path=args.model_path,
        nightmares_file=args.nightmares_file,
        output_dir=args.output_dir,
        use_neural_reward=args.use_neural_reward,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    trainer.train(
        num_epochs=args.num_epochs,
        save_every=500,
        log_wandb=args.wandb
    )
    
    # Evaluate on test set
    test_nightmares = [
        "I'm falling from a tall building",
        "Everyone is staring at me and laughing",
        "I'm trapped in a small space",
        "I'm being chased by something terrifying",
        "I forgot to study for an important exam"
    ]
    
    results = trainer.evaluate(test_nightmares)
    
    # Save results
    with open(os.path.join(args.output_dir, 'eval_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
