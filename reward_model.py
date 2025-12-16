"""
Reward Model for Evaluating Nightmare Comfort Quality
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple
import numpy as np
from textblob import TextBlob


class ComfortRewardModel(nn.Module):
    """
    Multi-dimensional reward model for comfort responses
    Evaluates: empathy, coherence, emotional shift, and safety
    """
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Freeze BERT layers (optional - unfreeze for better performance)
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Reward heads for different aspects
        hidden_size = self.bert.config.hidden_size
        self.empathy_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.coherence_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.emotional_shift_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.safety_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Get BERT embeddings for texts"""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.bert(**inputs)
        
        # Use [CLS] token embedding
        return outputs.last_hidden_state[:, 0, :]
    
    def forward(self, nightmares: List[str], comforts: List[str]) -> dict:
        """
        Calculate multi-dimensional rewards
        
        Args:
            nightmares: List of nightmare descriptions
            comforts: List of comfort responses
            
        Returns:
            Dictionary with individual reward components and total reward
        """
        # Get embeddings
        nightmare_embeds = self.get_embeddings(nightmares)
        comfort_embeds = self.get_embeddings(comforts)
        
        # Concatenate for paired evaluation
        paired_embeds = torch.cat([nightmare_embeds, comfort_embeds], dim=1)
        
        # Calculate individual rewards
        empathy_scores = self.empathy_head(paired_embeds).squeeze(-1)
        coherence_scores = self.coherence_head(paired_embeds).squeeze(-1)
        emotional_shift_scores = self.emotional_shift_head(paired_embeds).squeeze(-1)
        safety_scores = self.safety_head(comfort_embeds).squeeze(-1)
        
        # Weighted combination for total reward
        total_reward = (
            0.3 * empathy_scores +
            0.25 * coherence_scores +
            0.3 * emotional_shift_scores +
            0.15 * safety_scores
        )
        
        return {
            'empathy': empathy_scores,
            'coherence': coherence_scores,
            'emotional_shift': emotional_shift_scores,
            'safety': safety_scores,
            'total': total_reward
        }
    
    def calculate_reward(self, nightmare: str, comfort: str) -> float:
        """Calculate single reward value"""
        self.eval()
        with torch.no_grad():
            rewards = self.forward([nightmare], [comfort])
        return rewards['total'].item()


class HeuristicRewardModel:
    """
    Simpler heuristic-based reward model (no training needed)
    Use this if you don't have time to train the neural reward model
    """
    def __init__(self):
        from textblob import TextBlob
        self.fear_keywords = [
            'afraid', 'scared', 'fear', 'terror', 'panic', 'anxiety',
            'worried', 'frightened', 'nervous', 'dread'
        ]
        self.comfort_keywords = [
            'safe', 'calm', 'peace', 'relax', 'breathe', 'comfort',
            'warm', 'gentle', 'okay', 'alright', 'here', 'present'
        ]
    
    def calculate_reward(self, nightmare: str, comfort: str) -> float:
        """
        Calculate reward based on multiple heuristics
        """
        reward = 0.0
        
        # 1. Sentiment shift (fear -> calm)
        nightmare_sentiment = TextBlob(nightmare).sentiment.polarity
        comfort_sentiment = TextBlob(comfort).sentiment.polarity
        sentiment_shift = comfort_sentiment - nightmare_sentiment
        reward += 0.3 * max(0, sentiment_shift)  # Reward positive shift
        
        # 2. Comfort keyword presence
        comfort_lower = comfort.lower()
        comfort_keyword_count = sum(
            1 for word in self.comfort_keywords if word in comfort_lower
        )
        reward += 0.2 * min(comfort_keyword_count / 3, 1.0)
        
        # 3. Length appropriateness (50-150 words ideal)
        word_count = len(comfort.split())
        if 50 <= word_count <= 150:
            reward += 0.2
        elif 30 <= word_count < 50 or 150 < word_count <= 200:
            reward += 0.1
        
        # 4. Avoid repetition (penalize if starts same as nightmare)
        if not comfort.lower().startswith(nightmare.lower()[:20]):
            reward += 0.15
        
        # 5. Coherence (nightmare keywords mentioned in comfort)
        nightmare_words = set(nightmare.lower().split())
        comfort_words = set(comfort_lower.split())
        overlap = len(nightmare_words & comfort_words) / max(len(nightmare_words), 1)
        reward += 0.15 * min(overlap, 0.5) * 2  # Some overlap is good
        
        return reward


# Training script for neural reward model (optional)
def train_reward_model(model, train_data, epochs=5, lr=1e-4):
    """
    Train the reward model on labeled data
    
    train_data format: List of (nightmare, comfort, label) tuples
    where label is 0-1 score for quality
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for nightmare, comfort, label in train_data:
            optimizer.zero_grad()
            
            rewards = model([nightmare], [comfort])
            predicted_reward = rewards['total']
            target = torch.tensor([label], dtype=torch.float32)
            
            if torch.cuda.is_available():
                target = target.cuda()
            
            loss = criterion(predicted_reward, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_data):.4f}")
    
    return model


if __name__ == "__main__":
    # Test the reward models
    print("Testing Heuristic Reward Model...")
    heuristic_model = HeuristicRewardModel()
    
    nightmare = "I'm falling from a tall building"
    comfort = "You notice your breath, steady and real. You realize you're safe in your bed."
    
    reward = heuristic_model.calculate_reward(nightmare, comfort)
    print(f"Nightmare: {nightmare}")
    print(f"Comfort: {comfort}")
    print(f"Reward: {reward:.3f}")
    
    print("\n" + "="*70 + "\n")
    print("Testing Neural Reward Model...")
    neural_model = ComfortRewardModel()
    if torch.cuda.is_available():
        neural_model = neural_model.cuda()
    
    reward_dict = neural_model([nightmare], [comfort])
    print(f"Empathy: {reward_dict['empathy'].item():.3f}")
    print(f"Coherence: {reward_dict['coherence'].item():.3f}")
    print(f"Emotional Shift: {reward_dict['emotional_shift'].item():.3f}")
    print(f"Safety: {reward_dict['safety'].item():.3f}")
    print(f"Total Reward: {reward_dict['total'].item():.3f}")