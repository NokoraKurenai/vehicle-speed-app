import streamlit as st
from speed_detector import process_video
import tempfile

st.title("Vehicle Speed Estimation")

uploaded_file = st.file_uploader(
    "Upload AVI/MP4",
    type=["avi", "mp4"]
)

if uploaded_file:

    st.video(uploaded_file)

    if st.button("Process"):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".avi"
        ) as tmp:

            tmp.write(uploaded_file.read())

            input_path = tmp.name

        output_path = process_video(
            input_path
        )

        st.video(output_path)

        with open(output_path, "rb") as f:

            st.download_button(
                "Download Result",
                f,
                file_name="result.mp4"
            )