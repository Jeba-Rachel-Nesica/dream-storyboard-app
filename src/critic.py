"""
Emotion Critic (Nonlinear Reward Version)
-----------------------------------------
Refines linear scoring into psychologically meaningful nonlinear reward shaping.
Encourages strong emotional shifts from fear → calm, while keeping coherence and grounding.
"""

import re
import math
from typing import Dict


try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    _HAS_VADER = True
except Exception:
    _HAS_VADER = False

class EmotionCritic:
    def __init__(self):
        self.comfort_keywords = ["safe","calm","soothe","steady","relax","peace","gentle",
                                 "comfort","warm","soft","support","trust","present","grounded",
                                 "breathe","okay","secure","home","kind","rest"]
        self.fear_keywords = ["scary","afraid","fear","panic","anxious","nervous","terror",
                              "dread","threat","danger","horrible","awful","worst","chase",
                              "fall","trapped","dark","scream","alone"]
        self.grounding_phrases = ["notice your","feel your","breathe","right now","in this moment",
                                  "you are safe","you can","you have","feet on the ground","hand on heart"]
        self.vader = SentimentIntensityAnalyzer() if _HAS_VADER else None

    def _valence(self, text: str) -> float:
        t = text.lower()
        if self.vader:
            return self.vader.polarity_scores(t)["compound"]  # [-1,1]
        words = max(len(t.split()),1)
        pos = sum(1 for kw in self.comfort_keywords if kw in t)
        neg = sum(1 for kw in self.fear_keywords if kw in t)
        return max(-1, min(1, (pos - neg)/words * 5.0))

    def _coh(self, a: str, b: str) -> float:
        A = set(re.findall(r"\w+", a.lower()))
        B = set(re.findall(r"\w+", b.lower()))
        return 0.0 if not A else min(1.0, len(A & B)/len(A))

    def score_response(self, response: str, ctx: Dict) -> float:
        role = ctx.get("role","").upper()
        return self._fear(response, ctx) if role=="FEAR" else (
               self._comfort(response, ctx) if role=="COMFORT" else 0.0)

    def _fear(self, r: str, ctx: Dict) -> float:
        wc = len(r.split())
        length = 1/(1+math.exp(-0.30*(wc-12)))             # ideal ~8–30
        val = self._valence(r)                              # negative is “good” for fear
        val_score = (-val + 1)/2                            # map val=-1→1, val=+1→0
        coh = self._coh(ctx.get("nightmare",""), r)
        raw = 0.4*length + 0.4*val_score + 0.2*coh
        return max(-1, min(1, math.tanh(2.0*(raw-0.5))))

    def _comfort(self, r: str, ctx: Dict) -> float:
        rc = r.lower().strip()
        wc = len(rc.split())
        length = 1/(1+math.exp(-0.12*(wc-30)))             # ideal ~20–80
        val = (self._valence(rc)+1)/2                      # [0,1]
        gcount = sum(1 for phr in self.grounding_phrases if phr in rc)
        grounding = math.tanh(0.9*(gcount/2.0))
        coh = self._coh(ctx.get("nightmare",""), rc)
        last_fear = ""
        for role,text in reversed(ctx.get("history",[])):
            if role.upper()=="FEAR":
                last_fear = text; break
        delta_score = (math.tanh(2.5*(self._valence(rc) - self._valence(last_fear))) * 0.5 + 0.5) if last_fear else 0.5
        raw = 0.25*length + 0.25*val + 0.20*grounding + 0.15*coh + 0.15*delta_score
        return max(-1, min(1, math.tanh(2.0*(raw-0.45))))