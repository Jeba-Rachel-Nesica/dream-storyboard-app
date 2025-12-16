import re
from profanity_check import predict_prob
import streamlit as st

UNSAFE_KEYWORDS = [
    "violence", "gore", "blood", "kill", "murder", "suicide", "self-harm", "sexual", "abuse", "illegal", "weapon", "drugs"
]


def is_safe_text(text: str) -> bool:
    # Check for unsafe keywords
    for word in UNSAFE_KEYWORDS:
        if re.search(rf"\\b{re.escape(word)}\\b", text, re.IGNORECASE):
            return False
    # Profanity check
    if predict_prob([text])[0] > 0.3:
        return False
    return True


def show_disclaimer():
    st.markdown(
        """
        **Disclaimer:** This is a research prototype for demonstration only. It is not medical advice. Unsafe, graphic, or illegal content is strictly prohibited and will be blocked.
        """
    )
