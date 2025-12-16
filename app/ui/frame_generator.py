import streamlit as st
from app.pipelines.generation import generate_candidates, auto_rank_candidates
from app.pipelines.identity import compute_face_similarity
from app.storage import state
import numpy as np

def frame_generator():
    st.header("4. Generate & Select Keyframes")
    beats = state.get_beats()
    identity = state.get_identity()
    if not (beats and identity):
        st.info("Complete previous steps first.")
        return
    N = 3
    chosen = []
    for i, beat in enumerate(beats):
        st.subheader(f"Beat {i+1}")
        candidates = generate_candidates(beat, identity, n=N)
        # Optional: auto-rank by face similarity
        ranked = auto_rank_candidates(candidates, identity)
        cols = st.columns(N)
        selected = 0
        for j, (img, score) in enumerate(ranked):
            with cols[j]:
                st.image(img, caption=f"Seed {score['seed']}\nSim: {score['sim']:.2f}", use_column_width=True)
                if st.button(f"Select", key=f"sel_{i}_{j}"):
                    selected = j
        chosen.append({
            'img': ranked[selected][0],
            'seed': ranked[selected][1]['seed'],
            'prompt': beat['positive_prompt'],
            'negative_prompt': beat['negative_prompt'],
        })
    state.save_keyframes(chosen)
    st.success("Keyframes selected. Proceed to stylization.")
