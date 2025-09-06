import streamlit as st
import cv2
import numpy as np
import os
import tempfile
from vr180_converter import convert_to_vr180

st.set_page_config(
    page_title="NeRF VR180 Converter",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 NeRF-Enhanced VR180 Converter")
st.markdown("Convert your videos to VR180 format using Neural Radiance Fields simulation")

# Sidebar
st.sidebar.header("Settings")
st.sidebar.markdown("### NeRF Features")
st.sidebar.checkbox("Enable NeRF depth simulation", value=True)
st.sidebar.checkbox("Use Instant-NGP processing", value=True)
st.sidebar.checkbox("Apply VR180 optimization", value=True)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Video")
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video to convert to VR180 format"
    )
    
    if uploaded_file is not None:
        st.video(uploaded_file)
        
        # Show file info
        st.info(f"**File:** {uploaded_file.name}")
        st.info(f"**Size:** {uploaded_file.size / (1024*1024):.2f} MB")

with col2:
    st.header("⚙️ Conversion Process")
    
    if uploaded_file is not None:
        if st.button("🚀 Convert to VR180", type="primary"):
            with st.spinner("Converting video with NeRF enhancement..."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Convert using NeRF VR180 converter
                    output_path = convert_to_vr180(tmp_path)
                    
                    if os.path.exists(output_path):
                        st.success("✅ Conversion completed successfully!")
                        
                        # Show output video
                        st.video(output_path)
                        
                        # Download button
                        with open(output_path, "rb") as file:
                            st.download_button(
                                label="📥 Download VR180 Video",
                                data=file.read(),
                                file_name=f"vr180_{uploaded_file.name}",
                                mime="video/mp4"
                            )
                        
                        # Clean up
                        os.unlink(tmp_path)
                        os.unlink(output_path)
                    else:
                        st.error("❌ Conversion failed - no output file created")
                        
                except Exception as e:
                    st.error(f"❌ Conversion failed: {str(e)}")
                    st.exception(e)
    else:
        st.info("👆 Upload a video file to start conversion")

# Features section
st.header("🧠 NeRF Technology Features")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🎯 Depth Simulation")
    st.markdown("""
    - Neural Radiance Fields training
    - Instant-NGP processing
    - Depth-aware stereo offset
    - Perspective correction
    """)

with col2:
    st.subheader("👁️ VR180 Enhancement")
    st.markdown("""
    - Side-by-side stereo rendering
    - Color grading optimization
    - VR metadata embedding
    - High-quality encoding
    """)

with col3:
    st.subheader("⚡ Performance")
    st.markdown("""
    - Real-time processing
    - Frame-by-frame enhancement
    - Adaptive stereo calculation
    - Professional output quality
    """)

# Footer
st.markdown("---")
st.markdown("**Built with:** NeRF, Instant-NGP, OpenCV, Streamlit")
st.markdown("**GitHub:** [Leoash14/vr180-converter](https://github.com/Leoash14/vr180-converter)")
