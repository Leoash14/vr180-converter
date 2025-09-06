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

# Custom CSS for clean UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    .upload-container {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        border: 1px solid #e9ecef;
    }
    .success-message {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: none;
        border-radius: 10px;
        padding: 1.5rem;
        color: #155724;
        text-align: center;
        margin: 1rem 0;
    }
    .info-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 3px solid #667eea;
    }
    .login-container {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        border: 1px solid #e9ecef;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    .stFileUploader > div {
        border: 2px dashed #667eea;
        border-radius: 10px;
        background: #f8f9fa;
    }
    .stFileUploader > div:hover {
        border-color: #764ba2;
        background: #f0f2f6;
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
        
        # Demo login
        st.markdown("---")
        st.markdown("### 🚀 Quick Demo")
        st.markdown("Try the app instantly with our demo account")
        
        if st.button("Login as Demo User", use_container_width=True):
            st.session_state.authenticated = True
            st.session_state.user_email = "demo@nerfvr.com"
            st.session_state.user_name = "Demo User"
            st.rerun()
    
    st.stop()

# Main app interface (only shown when authenticated)
st.markdown("""
<div class="main-header">
    <h1>🧠 NeRF VR180 Converter</h1>
    <p style="font-size: 1.1rem; margin: 0;">Welcome, {}! Ready to convert your videos</p>
</div>
""".format(st.session_state.user_name), unsafe_allow_html=True)

# Simple header with logout
col1, col2 = st.columns([4, 1])
with col2:
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.user_email = None
        st.session_state.user_name = None
        st.rerun()

# Main content - Clean and simple
st.markdown('<div class="upload-container">', unsafe_allow_html=True)

st.markdown("## 📤 Upload Your Video")
uploaded_file = st.file_uploader(
    "Choose a video file to convert to VR180",
    type=['mp4', 'avi', 'mov', 'mkv'],
    help="Supported formats: MP4, AVI, MOV, MKV"
)

if uploaded_file is not None:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎬 Video Preview")
        st.video(uploaded_file)
    
    with col2:
        st.markdown("### 📋 File Details")
        st.markdown(f'<div class="info-card"><strong>📁 File:</strong><br>{uploaded_file.name}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="info-card"><strong>💾 Size:</strong><br>{uploaded_file.size / (1024*1024):.2f} MB</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="info-card"><strong>🎯 Format:</strong><br>{uploaded_file.name.split(".")[-1].upper()}</div>', unsafe_allow_html=True)

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
    <div class="info-card">
        <h4>🎯 How it works:</h4>
        <ol>
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
