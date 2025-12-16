import re
from typing import List, Dict
import random

# Beat object: beat_id, scene_text, emotion_targets, positive_prompt, negative_prompt, camera_hint

SAFE_ENDINGS = [
    "The dreamer feels safe and calm as the scene fades.",
    "A sense of mastery and peace settles in.",
    "The dream ends with comfort and resolution."
]

EMOTIONS = ["calm", "mastery", "arousal"]


def extract_beats(script: str) -> List[Dict]:
    # Split into sentences, group into 4–8 beats
    sents = re.split(r'(?<=[.!?])\s+', script.strip())
    n_beats = min(max(len(sents) // 2, 4), 8)
    beats = []
    for i in range(n_beats-1):
        scene = sents[i*len(sents)//n_beats:(i+1)*len(sents)//n_beats]
        text = " ".join(scene).strip()
        beat = {
            "beat_id": f"beat_{i+1:02d}",
            "scene_text": text,
            "emotion_targets": {k: round(random.uniform(0.3, 0.8),2) for k in EMOTIONS},
            "positive_prompt": f"A calm, dreamlike scene: {text}",
            "negative_prompt": "violence, gore, sexual, self-harm, illegal, disturbing",
            "camera_hint": "soft focus, gentle lighting"
        }
        beats.append(beat)
    # Last beat: safe ending
    ending = random.choice(SAFE_ENDINGS)
    beat = {
        "beat_id": f"beat_{n_beats:02d}",
        "scene_text": ending,
        "emotion_targets": {k: 1.0 if k in ["calm", "mastery"] else 0.0 for k in EMOTIONS},
        "positive_prompt": f"A peaceful, resolved ending: {ending}",
        "negative_prompt": "violence, gore, sexual, self-harm, illegal, disturbing",
        "camera_hint": "soft fade out"
    }
    beats.append(beat)
    return beats
