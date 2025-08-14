import streamlit as st
import tempfile
import os
import json
from analysis import analyze_video

st.set_page_config(page_title="AthleteRise Cricket Analysis", layout="wide")

st.title("🏏 AI-Powered Cricket Cover Drive Analysis")
st.write("Upload a video of a cover drive, and this app will perform a real-time pose analysis, providing feedback on key biomechanical metrics.")

uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Temporary file handling
    input_tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.' + uploaded_file.name.split('.')[-1])
    input_tfile.write(uploaded_file.read())
    input_path = input_tfile.name
    input_tfile.close() 

    output_path = tempfile.mktemp(suffix=".mp4")

    with st.spinner('Analyzing video... This might take a moment. ⏳'):
        annotated_video_path, evaluation_report = analyze_video(input_path, output_path)

    st.success('Analysis complete! 🎉')
    
    col1, col2 = st.columns(2)

    with col1:
        # Video display
        st.header("Annotated Video")
        with open(annotated_video_path, 'rb') as video_file:
            video_bytes = video_file.read()
        st.video(video_bytes)
        st.download_button(
            label="Download Annotated Video",
            data=video_bytes,
            file_name="annotated_video.mp4",
            mime="video/mp4"
        )

    with col2:
        st.header("Final Shot Evaluation")
        if evaluation_report:
            # Display the Overall Grade
            grade = evaluation_report.get("Overall Skill Grade", "N/A")
            avg_score = evaluation_report.get("Average Score", 0)
            
            st.subheader(f"Overall Skill Grade: {grade}")
            st.progress(avg_score / 10)
            st.divider()

            # --- CORRECTED LOOP ---
            # Ensure you are looping over the "breakdown" key.
            for category, values in evaluation_report.get("breakdown", {}).items():
                st.metric(label=category, value=f"{values['score']}/10")
                st.write(f"**Feedback:** {values['feedback']}")
                st.divider()

            # Download button for the full report
            report_json = json.dumps(evaluation_report, indent=4)
            st.download_button(
                label="Download Full Report (JSON)",
                data=report_json,
                file_name="evaluation_report.json",
                mime="application/json"
            )
        else:
            st.warning("Could not generate a report for this video.")

    # Clean up temporary files
    os.remove(input_path)
    os.remove(output_path)