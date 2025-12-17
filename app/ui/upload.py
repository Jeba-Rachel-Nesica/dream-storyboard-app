import streamlit as st
from app.pipelines.identity import detect_and_embed_faces
from app.storage import state

CONSENT_TEXT = "I have permission to use these photos."

def upload_photos():
    st.header("1. Upload Person Photos")
    uploaded = st.file_uploader(
        "Upload 3–10 clear photos of the person (jpg/png)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="photo_upload",
    )
    consent = st.checkbox(CONSENT_TEXT, key="consent")
    if not consent:
        st.warning("You must confirm you have permission to use these photos.")
        return
    if uploaded:
        if not (3 <= len(uploaded) <= 10):
            st.error("Please upload between 3 and 10 photos.")
            return
        with st.spinner("Analyzing faces..."):
            faces_ok, face_info = detect_and_embed_faces(uploaded)
        if not faces_ok:
            st.error(face_info)
            st.info("Tips: Use well-lit, front-facing photos. Only one face per image.")
            return
        state.save_identity(face_info)
        st.success(f"{len(face_info['embeddings'])} faces processed. Identity profile ready.")
        st.image(face_info['best_crop'], caption="Best reference face", width=200)
    else:
        st.info("Upload 3–10 clear, front-facing photos.")
