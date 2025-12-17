import re
import streamlit as st

# Profanity check with fallback
try:
    from profanity_check import predict_prob
    PROFANITY_CHECK_AVAILABLE = True
except ImportError:
    PROFANITY_CHECK_AVAILABLE = False
    try:
        from better_profanity import profanity
        profanity.load_censor_words()
        BETTER_PROFANITY_AVAILABLE = True
    except ImportError:
        BETTER_PROFANITY_AVAILABLE = False

UNSAFE_KEYWORDS = [
    "violence", "gore", "blood", "kill", "murder", "suicide", "self-harm", "sexual", "abuse", "illegal", "weapon", "drugs"
]


def is_safe_text(text: str) -> bool:
    # Check for unsafe keywords
    for word in UNSAFE_KEYWORDS:
        if re.search(rf"\\b{re.escape(word)}\\b", text, re.IGNORECASE):
            return False
    
    # Profanity check with fallback
    if PROFANITY_CHECK_AVAILABLE:
        try:
            if predict_prob([text])[0] > 0.3:
                return False
        except Exception:
            pass
    elif BETTER_PROFANITY_AVAILABLE:
        try:
            if profanity.contains_profanity(text):
                return False
        except Exception:
            pass
    
    return True


def show_disclaimer():
    st.markdown(
        """
        **Disclaimer:** This is a research prototype for demonstration only. It is not medical advice. Unsafe, graphic, or illegal content is strictly prohibited and will be blocked.
        """
    )
