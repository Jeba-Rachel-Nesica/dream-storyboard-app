import streamlit as st
from pipelines.stylize import stylize_img2img
from pipelines.compose_video import compose_video
from storage import state
import os

def video_export():
    st.header("5. Stylize & Export Video")
    keyframes = state.get_keyframes()
    if not keyframes:
        st.info("Select keyframes first.")
        return
    style = st.text_input("Style preset", "abstract dreamy watercolor, soft gradients, calming, low detail")
    styled = []
    for i, kf in enumerate(keyframes):
        st.subheader(f"Beat {i+1}")
        img = stylize_img2img(kf['img'], style)
        st.image(img, caption="Styled Frame", use_column_width=True)
        styled.append(img)
    state.save_styled_frames(styled)
    st.success("All frames stylized.")
    if st.button("Compose & Preview Video"):
        video_path = compose_video(styled, state.get_beats())
        if video_path and os.path.exists(video_path):
            st.video(video_path)
            st.download_button("Download Video", open(video_path, "rb"), file_name="rehearsal_video.mp4")
        else:
            st.error("Video composition failed.")
    if st.button("Delete project outputs"):
        state.delete_outputs()
        st.success("All outputs deleted.")
