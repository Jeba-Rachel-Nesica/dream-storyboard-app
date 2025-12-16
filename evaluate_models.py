"""
Comparison and Evaluation Script
Compare Supervised Learning vs RL-trained models
"""
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from trl import AutoModelForCausalLMWithValueHead
import json
import numpy as np
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from reward_model import HeuristicRewardModel, ComfortRewardModel

class ModelEvaluator:
    """
    Evaluate and compare different models
    """
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.reward_model = HeuristicRewardModel()
        
    def load_model(self, model_path: str, is_ppo: bool = False):
        """Load a model for evaluation"""
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        
        if is_ppo:
            model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path)
        else:
            model = GPT2LMHeadModel.from_pretrained(model_path)
        
        model = model.to(self.device)
        model.eval()
        
        return model, tokenizer
    
    def generate_comfort(
        self,
        model,
        tokenizer,
        nightmare: str,
        max_length: int = 100,
        temperature: float = 0.8,
        is_ppo: bool = False
    ) -> str:
        """Generate comfort response for a nightmare"""
        # Format input
        prompt = f"Nightmare: {nightmare}\nComfort:"
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors='pt').to(self.device)
        
        # Generate
        with torch.no_grad():
            if is_ppo:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    top_p=0.9,
                    top_k=50
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    top_p=0.9,
                    top_k=50
                )
        
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract comfort part
        if "Comfort:" in response:
            comfort = response.split("Comfort:")[-1].strip()
        else:
            comfort = response.strip()
        
        return comfort
    
    def evaluate_model(
        self,
        model,
        tokenizer,
        test_nightmares: List[str],
        is_ppo: bool = False,
        model_name: str = "Model"
    ) -> Dict:
        """Evaluate a model on test set"""
        print(f"\n{'='*70}")
        print(f"Evaluating {model_name}")
        print('='*70)
        
        results = []
        rewards = []
        comfort_lengths = []
        
        for i, nightmare in enumerate(test_nightmares):
            # Generate comfort
            comfort = self.generate_comfort(
                model, tokenizer, nightmare, is_ppo=is_ppo
            )
            
            # Calculate reward
            reward = self.reward_model.calculate_reward(nightmare, comfort)
            
            # Store results
            results.append({
                'nightmare': nightmare,
                'comfort': comfort,
                'reward': reward,
                'length': len(comfort.split())
            })
            
            rewards.append(reward)
            comfort_lengths.append(len(comfort.split()))
            
            # Print sample
            print(f"\n{i+1}. Nightmare: {nightmare}")
            print(f"   Comfort: {comfort[:200]}...")
            print(f"   Reward: {reward:.4f} | Length: {len(comfort.split())} words")
            print('-'*70)
        
        # Calculate statistics
        stats = {
            'model_name': model_name,
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'avg_length': np.mean(comfort_lengths),
            'results': results
        }
        
        print(f"\nSummary for {model_name}:")
        print(f"  Average Reward: {stats['avg_reward']:.4f} ± {stats['std_reward']:.4f}")
        print(f"  Reward Range: [{stats['min_reward']:.4f}, {stats['max_reward']:.4f}]")
        print(f"  Average Length: {stats['avg_length']:.1f} words")
        
        return stats
    
    def compare_models(
        self,
        supervised_path: str,
        ppo_path: str,
        test_nightmares: List[str],
        output_dir: str = 'evaluation_results'
    ):
        """Compare supervised and RL models"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*70)
        print("MODEL COMPARISON: SUPERVISED vs PPO")
        print("="*70)
        
        # Load models
        print("\nLoading models...")
        supervised_model, supervised_tokenizer = self.load_model(supervised_path, is_ppo=False)
        ppo_model, ppo_tokenizer = self.load_model(ppo_path, is_ppo=True)
        
        # Evaluate supervised model
        supervised_stats = self.evaluate_model(
            supervised_model,
            supervised_tokenizer,
            test_nightmares,
            is_ppo=False,
            model_name="Supervised (Fine-tuned GPT-2)"
        )
        
        # Evaluate PPO model
        ppo_stats = self.evaluate_model(
            ppo_model,
            ppo_tokenizer,
            test_nightmares,
            is_ppo=True,
            model_name="PPO (RL-trained)"
        )
        
        # Calculate improvement
        improvement = ((ppo_stats['avg_reward'] - supervised_stats['avg_reward']) 
                      / supervised_stats['avg_reward'] * 100)
        
        print("\n" + "="*70)
        print("COMPARISON RESULTS")
        print("="*70)
        print(f"Supervised Model Avg Reward: {supervised_stats['avg_reward']:.4f}")
        print(f"PPO Model Avg Reward: {ppo_stats['avg_reward']:.4f}")
        print(f"Improvement: {improvement:+.2f}%")
        
        # Visualization
        self.plot_comparison(supervised_stats, ppo_stats, output_dir)
        
        # Save detailed results
        comparison_results = {
            'supervised': supervised_stats,
            'ppo': ppo_stats,
            'improvement_percentage': improvement
        }
        
        with open(f'{output_dir}/comparison_results.json', 'w') as f:
            json.dump(comparison_results, f, indent=2)
        
        print(f"\nResults saved to {output_dir}/")
        
        return comparison_results
    
    def plot_comparison(self, supervised_stats: Dict, ppo_stats: Dict, output_dir: str):
        """Create comparison visualizations"""
        # Set style
        sns.set_style("whitegrid")
        
        # 1. Reward comparison bar plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        models = ['Supervised', 'PPO']
        rewards = [supervised_stats['avg_reward'], ppo_stats['avg_reward']]
        errors = [supervised_stats['std_reward'], ppo_stats['std_reward']]
        
        axes[0].bar(models, rewards, yerr=errors, capsize=5, 
                   color=['skyblue', 'lightcoral'], alpha=0.7)
        axes[0].set_ylabel('Average Reward')
        axes[0].set_title('Average Reward Comparison')
        axes[0].set_ylim([0, max(rewards) * 1.2])
        
        # Add value labels on bars
        for i, (model, reward) in enumerate(zip(models, rewards)):
            axes[0].text(i, reward + 0.02, f'{reward:.3f}', 
                        ha='center', va='bottom', fontweight='bold')
        
        # 2. Reward distribution
        supervised_rewards = [r['reward'] for r in supervised_stats['results']]
        ppo_rewards = [r['reward'] for r in ppo_stats['results']]
        
        axes[1].hist(supervised_rewards, bins=10, alpha=0.5, 
                    label='Supervised', color='skyblue')
        axes[1].hist(ppo_rewards, bins=10, alpha=0.5, 
                    label='PPO', color='lightcoral')
        axes[1].set_xlabel('Reward')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Reward Distribution')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/comparison_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Individual nightmare comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(supervised_rewards))
        width = 0.35
        
        ax.bar(x - width/2, supervised_rewards, width, 
               label='Supervised', color='skyblue', alpha=0.7)
        ax.bar(x + width/2, ppo_rewards, width, 
               label='PPO', color='lightcoral', alpha=0.7)
        
        ax.set_xlabel('Nightmare Index')
        ax.set_ylabel('Reward')
        ax.set_title('Reward Comparison per Nightmare')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/per_nightmare_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {output_dir}/")


def run_evaluation():
    """Main evaluation function"""
    
    # Test nightmares
    test_nightmares = [
        "I'm falling from a tall building and can't control my descent",
        "Everyone at my workplace is staring at me and laughing",
        "I'm trapped in a small, dark space with walls closing in",
        "Something terrifying is chasing me through dark streets",
        "I have an important presentation tomorrow and haven't prepared",
        "My teeth are falling out one by one",
        "I'm lost in an unfamiliar city and can't find my way home",
        "I'm trying to run but my legs won't move",
        "I'm drowning and can't reach the surface",
        "I showed up to class completely naked"
    ]
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Compare models
    results = evaluator.compare_models(
        supervised_path='checkpoints/comfort_large',
        ppo_path='checkpoints/ppo_comfort/best_model',
        test_nightmares=test_nightmares,
        output_dir='evaluation_results'
    )
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate and compare models')
    parser.add_argument('--supervised_path', type=str, 
                       default='checkpoints/comfort_large',
                       help='Path to supervised model')
    parser.add_argument('--ppo_path', type=str, 
                       default='checkpoints/ppo_comfort/best_model',
                       help='Path to PPO model')
    parser.add_argument('--output_dir', type=str, 
                       default='evaluation_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Run evaluation
    run_evaluation()