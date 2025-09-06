import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import hashlib
import sqlite3
from datetime import datetime
from vr180_converter import convert_to_vr180

# Configure page
st.set_page_config(
    page_title="NeRF VR180 Converter",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_email' not in st.session_state:
    st.session_state.user_email = None
if 'user_name' not in st.session_state:
    st.session_state.user_name = None

# Database functions
def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(email, password, name):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO users (email, password_hash, name)
            VALUES (?, ?, ?)
        ''', (email, hash_password(password), name))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def login_user(email, password):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT name FROM users 
        WHERE email = ? AND password_hash = ?
    ''', (email, hash_password(password)))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

# Initialize database
init_db()

# Create demo user if it doesn't exist
def create_demo_user():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM users WHERE email = ?', ('demo@nerfvr.com',))
    if cursor.fetchone()[0] == 0:
        cursor.execute('''
            INSERT INTO users (email, password_hash, name)
            VALUES (?, ?, ?)
        ''', ('demo@nerfvr.com', hash_password('demo123'), 'Demo User'))
        conn.commit()
    conn.close()

create_demo_user()

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8f9fa;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 10px;
        padding: 1rem;
        color: #155724;
    }
    .info-box {
        background: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 10px;
        padding: 1rem;
        color: #0c5460;
    }
</style>
""", unsafe_allow_html=True)

# Authentication check
if not st.session_state.authenticated:
    # Login/Register Interface
    st.markdown("""
    <div class="main-header">
        <h1>🧠 NeRF-Enhanced VR180 Converter</h1>
        <p style="font-size: 1.2rem; margin: 0;">Transform your videos into immersive VR180 experiences using Neural Radiance Fields</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🔐 Login")
        with st.form("login_form"):
            login_email = st.text_input("Email", placeholder="your@email.com")
            login_password = st.text_input("Password", type="password")
            login_submitted = st.form_submit_button("Login", use_container_width=True)
            
            if login_submitted:
                if login_email and login_password:
                    user_name = login_user(login_email, login_password)
                    if user_name:
                        st.session_state.authenticated = True
                        st.session_state.user_email = login_email
                        st.session_state.user_name = user_name
                        st.success(f"Welcome back, {user_name}!")
                        st.rerun()
                    else:
                        st.error("Invalid email or password")
                else:
                    st.error("Please fill in all fields")
    
    with col2:
        st.markdown("### 📝 Register")
        with st.form("register_form"):
            reg_name = st.text_input("Full Name", placeholder="John Doe")
            reg_email = st.text_input("Email", placeholder="your@email.com")
            reg_password = st.text_input("Password", type="password")
            reg_confirm = st.text_input("Confirm Password", type="password")
            reg_submitted = st.form_submit_button("Register", use_container_width=True)
            
            if reg_submitted:
                if reg_name and reg_email and reg_password and reg_confirm:
                    if reg_password == reg_confirm:
                        if register_user(reg_email, reg_password, reg_name):
                            st.success("Registration successful! Please login.")
                        else:
                            st.error("Email already exists")
                    else:
                        st.error("Passwords don't match")
                else:
                    st.error("Please fill in all fields")
    
    st.markdown("---")
    st.markdown("### 🎯 Demo Account")
    st.markdown("**Email:** demo@nerfvr.com")
    st.markdown("**Password:** demo123")
    st.markdown("*Use this account to try the app without registration*")
    
    # Demo login button
    if st.button("🚀 Login as Demo User", use_container_width=True):
        st.session_state.authenticated = True
        st.session_state.user_email = "demo@nerfvr.com"
        st.session_state.user_name = "Demo User"
        st.rerun()
    
    st.stop()

# Main app interface (only shown when authenticated)
st.markdown("""
<div class="main-header">
    <h1>🧠 NeRF-Enhanced VR180 Converter</h1>
    <p style="font-size: 1.2rem; margin: 0;">Welcome, {}! Transform your videos into immersive VR180 experiences</p>
</div>
""".format(st.session_state.user_name), unsafe_allow_html=True)

# Logout button
col1, col2, col3 = st.columns([1, 1, 1])
with col3:
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.user_email = None
        st.session_state.user_name = None
        st.rerun()

# Sidebar
st.sidebar.markdown("## ⚙️ NeRF Settings")
st.sidebar.markdown("Configure the Neural Radiance Fields processing:")

nerf_enabled = st.sidebar.checkbox("🧠 Enable NeRF depth simulation", value=True)
instant_ngp = st.sidebar.checkbox("⚡ Use Instant-NGP processing", value=True)
vr_optimization = st.sidebar.checkbox("🎯 Apply VR180 optimization", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("## 📊 Processing Info")
st.sidebar.markdown("**Status:** Ready to convert")
st.sidebar.markdown("**Supported formats:** MP4, AVI, MOV, MKV")
st.sidebar.markdown("**Max file size:** 100MB")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("## 📤 Upload Your Video")
    st.markdown('<div class="upload-area">', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Drag and drop your video here or click to browse",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video to convert to VR180 format",
        label_visibility="collapsed"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        st.markdown("### 🎬 Preview")
        st.video(uploaded_file)
        
        # Show file info in styled boxes
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.markdown(f'<div class="info-box"><strong>📁 File:</strong> {uploaded_file.name}</div>', unsafe_allow_html=True)
        with col_info2:
            st.markdown(f'<div class="info-box"><strong>💾 Size:</strong> {uploaded_file.size / (1024*1024):.2f} MB</div>', unsafe_allow_html=True)

with col2:
    st.markdown("## ⚙️ Conversion Process")
    
    if uploaded_file is not None:
        # Show processing steps
        st.markdown("### 🔄 Processing Steps")
        steps = [
            "📹 Extracting video frames",
            "🧠 Training NeRF model",
            "⚡ Instant-NGP processing",
            "👁️ Generating stereo views",
            "🎯 Applying VR180 optimization",
            "💾 Rendering final video"
        ]
        
        for i, step in enumerate(steps):
            st.markdown(f"**{i+1}.** {step}")
        
        st.markdown("---")
        
        # Convert button with better styling
        if st.button("🚀 Start NeRF VR180 Conversion", type="primary", use_container_width=True):
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Save uploaded file temporarily
                status_text.text("💾 Saving uploaded file...")
                progress_bar.progress(10)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Convert using NeRF VR180 converter
                status_text.text("🧠 Starting NeRF processing...")
                progress_bar.progress(20)
                
                output_path = convert_to_vr180(tmp_path)
                
                if os.path.exists(output_path):
                    progress_bar.progress(100)
                    status_text.text("✅ Conversion completed!")
                    
                    st.markdown('<div class="success-box"><h4>🎉 Success!</h4>Your video has been converted to VR180 format using NeRF technology!</div>', unsafe_allow_html=True)
                    
                    # Show output video
                    st.markdown("### 🎬 VR180 Output Preview")
                    st.video(output_path)
                    
                    # Download button with better styling
                    st.markdown("### 📥 Download Your VR180 Video")
                    with open(output_path, "rb") as file:
                        st.download_button(
                            label="💾 Download VR180 Video",
                            data=file.read(),
                            file_name=f"vr180_nerf_{uploaded_file.name}",
                            mime="video/mp4",
                            use_container_width=True,
                            type="primary"
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
        st.markdown("### 👆 Upload a video to start")
        st.markdown("""
        <div class="info-box">
            <h4>How it works:</h4>
            <ol>
                <li>Upload your video file</li>
                <li>NeRF analyzes depth and content</li>
                <li>Instant-NGP processes the data</li>
                <li>VR180 stereo views are generated</li>
                <li>Download your immersive VR video!</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

# Features section
st.markdown("---")
st.markdown("## 🧠 NeRF Technology Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h4>🎯 Depth Simulation</h4>
        <ul>
            <li>Neural Radiance Fields training</li>
            <li>Instant-NGP processing</li>
            <li>Depth-aware stereo offset</li>
            <li>Perspective correction</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h4>👁️ VR180 Enhancement</h4>
        <ul>
            <li>Side-by-side stereo rendering</li>
            <li>Color grading optimization</li>
            <li>VR metadata embedding</li>
            <li>High-quality encoding</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <h4>⚡ Performance</h4>
        <ul>
            <li>Real-time processing</li>
            <li>Frame-by-frame enhancement</li>
            <li>Adaptive stereo calculation</li>
            <li>Professional output quality</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 10px;">
    <h4>🚀 Built with Advanced AI Technology</h4>
    <p><strong>NeRF</strong> • <strong>Instant-NGP</strong> • <strong>OpenCV</strong> • <strong>Streamlit</strong></p>
    <p>GitHub: <a href="https://github.com/Leoash14/vr180-converter" target="_blank">Leoash14/vr180-converter</a></p>
</div>
""", unsafe_allow_html=True)
