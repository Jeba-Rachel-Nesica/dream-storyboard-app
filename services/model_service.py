from typing import Dict
from services.safety import basic_screen

# TODO: plug your trained model here.
def _local_soft_rewrite(nightmare: str, controls: Dict[str, bool]) -> str:
    soften = controls.get("soften", True)
    helper = controls.get("helper", False)
    exitd  = controls.get("exit", True)
    base = nightmare.strip().replace("can't","can").replace("alone","supported")
    parts = [p.strip() for p in base.split(".") if p.strip()][:6] or [base]
    out = []
    for i, p in enumerate(parts):
        s = p
        if i == 0: s = "Notice a steady breath. " + s
        if soften: s += " The space feels a bit safer."
        if helper and i == 1: s += " A calm helper stands with you."
        if exitd and i == 2: s += " A safe exit appears and you step toward it."
        out.append(s)
    if exitd: out.append("You end in a safe, warm place; your body remembers the calm.")
    return ". ".join(out) + "."

def generate_rewrite(nightmare: str, controls: Dict[str, bool]) -> Dict:
    # try:
    #   return {"text": your_model(nightmare, controls), "flags": basic_screen(nightmare)}
    # except Exception:
    #   pass
    return {"text": _local_soft_rewrite(nightmare, controls), "flags": basic_screen(nightmare)}
