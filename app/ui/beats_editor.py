import streamlit as st
from app.pipelines.beats import extract_beats
from app.pipelines.safety import is_safe_text
from app.storage import state

def beats_editor():
    st.header("3. Extract & Edit Beats")
    script = state.get_script()
    if not script:
        st.info("Upload a rewritten script first.")
        return
    if st.button("Regenerate beats") or not state.get_beats():
        beats = extract_beats(script)
        state.save_beats(beats)
    beats = state.get_beats()
    if not beats:
        st.warning("No beats extracted.")
        return
    edited = []
    for i, beat in enumerate(beats):
        st.subheader(f"Beat {i+1}")
        beat['scene_text'] = st.text_input(f"Scene text", beat['scene_text'], key=f"scene_{i}")
        beat['positive_prompt'] = st.text_input(f"Positive prompt", beat['positive_prompt'], key=f"pos_{i}")
        beat['negative_prompt'] = st.text_input(f"Negative prompt", beat['negative_prompt'], key=f"neg_{i}")
        beat['camera_hint'] = st.text_input(f"Camera hint", beat['camera_hint'], key=f"cam_{i}")
        for k in ['calm', 'mastery', 'arousal']:
            beat['emotion_targets'][k] = st.slider(f"{k.title()} (0–1)", 0.0, 1.0, float(beat['emotion_targets'][k]), 0.01, key=f"emo_{k}_{i}")
        if not is_safe_text(beat['scene_text'] + ' ' + beat['positive_prompt'] + ' ' + beat['negative_prompt']):
            st.error("Unsafe content detected in this beat. Please edit.")
        edited.append(beat)
    state.save_beats(edited)
    st.success("Beats ready. You can edit them as needed.")
