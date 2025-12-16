import re
BLOCKLIST = re.compile(r"\b(suicide|kill myself|harm others|overdose|hopeless)\b", re.I)
def basic_screen(text: str) -> dict:
    return {"crisis_language": bool(BLOCKLIST.search(text or ""))}
