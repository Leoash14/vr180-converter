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

# Custom CSS for attractive UI with animations
st.markdown("""
<style>
    /* Hide Streamlit default elements */
    .stApp > header { visibility: hidden; }
    .stApp > div:first-child { padding-top: 0; }
    
    /* Main background with animation */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
        animation: gradientShift 10s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        50% { background: linear-gradient(135deg, #764ba2 0%, #667eea 100%); }
        100% { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    }
    
    /* Main container with animation */
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        animation: slideInUp 0.8s ease-out;
    }
    
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Header with animation */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        animation: fadeInDown 0.6s ease-out;
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Upload area with animation */
    .upload-section {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        border: 2px dashed #667eea;
        transition: all 0.3s ease;
        animation: fadeIn 0.8s ease-out 0.2s both;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .upload-section:hover {
        border-color: #764ba2;
        background: linear-gradient(135deg, #f0f2f6 0%, #e3e6ea 100%);
        transform: scale(1.02);
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    
    /* File info cards with animation */
    .file-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        animation: slideInLeft 0.6s ease-out;
        transition: all 0.3s ease;
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .file-info:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
    }
    
    /* Success message with animation */
    .success-message {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 10px 25px rgba(40, 167, 69, 0.3);
        animation: bounceIn 0.8s ease-out;
    }
    
    @keyframes bounceIn {
        0% {
            opacity: 0;
            transform: scale(0.3);
        }
        50% {
            opacity: 1;
            transform: scale(1.05);
        }
        70% {
            transform: scale(0.9);
        }
        100% {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    /* Login container with animation */
    .login-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        animation: slideInUp 0.8s ease-out;
    }
    
    /* Remove white blocks from forms */
    .stForm {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    
    .stForm > div {
        background: transparent !important;
        border: none !important;
    }
    
    /* Remove white blocks from main content */
    .main .block-container {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    
    .stApp > div:first-child > div:first-child {
        padding-top: 0 !important;
    }
    
    /* Remove white background from main content area */
    .main .block-container > div {
        background: transparent !important;
    }
    
    /* Hide Streamlit's default white containers */
    .stApp > div:first-child > div:first-child > div:first-child {
        background: transparent !important;
    }
    
    /* Make everything transparent except our custom containers */
    .stApp > div:first-child > div:first-child > div:first-child > div {
        background: transparent !important;
    }
    
    /* Buttons with animation */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3); }
        50% { box-shadow: 0 5px 25px rgba(102, 126, 234, 0.5); }
        100% { box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3); }
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
        animation: none;
    }
    
    /* File uploader with animation */
    .stFileUploader > div {
        border: 3px dashed #667eea;
        border-radius: 15px;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        transition: all 0.3s ease;
        animation: fadeIn 0.8s ease-out 0.4s both;
    }
    
    .stFileUploader > div:hover {
        border-color: #764ba2;
        background: linear-gradient(135deg, #f0f2f6 0%, #e3e6ea 100%);
        transform: scale(1.02);
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    
    /* Tabs with animation */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        animation: fadeIn 0.8s ease-out 0.3s both;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        transform: scale(1.05);
    }
    
    /* Progress bar with animation */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        animation: progressGlow 1.5s ease-in-out infinite;
    }
    
    @keyframes progressGlow {
        0%, 100% { box-shadow: 0 0 5px rgba(102, 126, 234, 0.5); }
        50% { box-shadow: 0 0 20px rgba(102, 126, 234, 0.8); }
    }
    
    /* Video player with animation */
    .stVideo {
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        animation: fadeIn 0.8s ease-out;
    }
    
    /* Input fields with animation */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e9ecef;
        transition: all 0.3s ease;
        animation: fadeIn 0.8s ease-out 0.5s both;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        transform: scale(1.02);
    }
    
    /* Demo section with animation */
    .demo-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
        animation: slideInUp 0.8s ease-out 0.6s both;
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
    }
    
    /* Remove all white backgrounds */
    .stApp > div:first-child > div:first-child > div:first-child > div:first-child {
        background: transparent !important;
    }
    
    /* Remove white blocks from columns */
    .stColumn > div {
        background: transparent !important;
    }
    
    /* Remove white blocks from markdown containers */
    .stMarkdown > div {
        background: transparent !important;
    }
    
    /* Make sure only our custom containers have backgrounds */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    /* Remove padding that creates white space */
    .main .block-container {
        padding-left: 0 !important;
        padding-right: 0 !important;
        max-width: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Authentication check
if not st.session_state.authenticated:
    # Clean Login Interface
    st.markdown("""
    <div class="main-header">
        <h1>🧠 NeRF VR180 Converter</h1>
        <p style="font-size: 1.1rem; margin: 0; opacity: 0.9;">Transform videos into immersive VR180 experiences</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
        
        with tab1:
            with st.form("login_form"):
                st.markdown("### Welcome Back")
                login_email = st.text_input("Email", placeholder="your@email.com", key="login_email")
                login_password = st.text_input("Password", type="password", key="login_password")
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
        
        with tab2:
            with st.form("register_form"):
                st.markdown("### Create Account")
                reg_name = st.text_input("Full Name", placeholder="John Doe", key="reg_name")
                reg_email = st.text_input("Email", placeholder="your@email.com", key="reg_email")
                reg_password = st.text_input("Password", type="password", key="reg_password")
                reg_confirm = st.text_input("Confirm Password", type="password", key="reg_confirm")
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
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Demo login with animation
        st.markdown('<div class="demo-section">', unsafe_allow_html=True)
        st.markdown("### 🚀 Quick Demo")
        st.markdown("Try the app instantly with our demo account")
        
        if st.button("Login as Demo User", use_container_width=True):
            st.session_state.authenticated = True
            st.session_state.user_email = "demo@nerfvr.com"
            st.session_state.user_name = "Demo User"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.stop()

# Main app interface (only shown when authenticated)
st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🧠 NeRF VR180 Converter</h1>
    <p style="font-size: 1.1rem; margin: 0;">Welcome, {}! Ready to convert your videos</p>
</div>
""".format(st.session_state.user_name), unsafe_allow_html=True)

# Header with logout
col1, col2 = st.columns([4, 1])
with col2:
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.user_email = None
        st.session_state.user_name = None
        st.rerun()

# Upload section
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.markdown("## 📤 Upload Your Video")
uploaded_file = st.file_uploader(
    "Choose a video file to convert to VR180",
    type=['mp4', 'avi', 'mov', 'mkv'],
    help="Supported formats: MP4, AVI, MOV, MKV"
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎬 Video Preview")
        st.video(uploaded_file)
    
    with col2:
        st.markdown("### 📋 File Details")
        st.markdown(f'<div class="file-info"><strong>📁 File:</strong><br>{uploaded_file.name}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="file-info"><strong>💾 Size:</strong><br>{uploaded_file.size / (1024*1024):.2f} MB</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="file-info"><strong>🎯 Format:</strong><br>{uploaded_file.name.split(".")[-1].upper()}</div>', unsafe_allow_html=True)

    # Convert button
    st.markdown("---")
    if st.button("🚀 Convert to VR180", type="primary", use_container_width=True):
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Save uploaded file temporarily
            status_text.text("💾 Processing file...")
            progress_bar.progress(20)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Convert using NeRF VR180 converter
            status_text.text("🧠 Converting with NeRF...")
            progress_bar.progress(50)
            
            output_path = convert_to_vr180(tmp_path)
            
            if os.path.exists(output_path):
                progress_bar.progress(100)
                status_text.text("✅ Conversion completed!")
                
                st.markdown('<div class="success-message"><h4>🎉 Success!</h4>Your VR180 video is ready!</div>', unsafe_allow_html=True)
                
                # Show output video
                st.markdown("### 🎬 VR180 Output")
                st.video(output_path)
                
                # Download button
                st.markdown("### 📥 Download")
                with open(output_path, "rb") as file:
                    st.download_button(
                        label="💾 Download VR180 Video",
                        data=file.read(),
                        file_name=f"vr180_{uploaded_file.name}",
                        mime="video/mp4",
                        use_container_width=True,
                        type="primary"
                    )
                
                # Clean up
                os.unlink(tmp_path)
                os.unlink(output_path)
            else:
                st.error("❌ Conversion failed")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

else:
    st.markdown("### 👆 Upload a video to get started")
    st.markdown("""
    <div class="file-info">
        <h4>🎯 How it works:</h4>
        <ol style="text-align: left; margin: 1rem 0;">
            <li>Upload your video file</li>
            <li>NeRF analyzes and processes it</li>
            <li>Download your VR180 video!</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Simple footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; color: #666;">
    <p>Built with <strong>NeRF</strong> • <strong>OpenCV</strong> • <strong>Streamlit</strong></p>
    <p><a href="https://github.com/Leoash14/vr180-converter" target="_blank">GitHub Repository</a></p>
</div>
""", unsafe_allow_html=True)
