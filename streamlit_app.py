import streamlit as st
import sqlite3
import hashlib
import datetime
import os
import time
from vr180_converter import convert_to_vr180

# Page config
st.set_page_config(
    page_title="VR180 Converter",
    page_icon="🎥",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Database functions
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    # Drop table if exists to ensure clean start
    c.execute('DROP TABLE IF EXISTS users')
    # Create fresh table with correct schema
    c.execute('''CREATE TABLE users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  email TEXT UNIQUE,
                  password_hash TEXT,
                  created_at TIMESTAMP)''')
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(email, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (email, password_hash, created_at) VALUES (?, ?, ?)",
                  (email, hash_password(password), datetime.datetime.now()))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def login_user(email, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE email = ? AND password_hash = ?",
              (email, hash_password(password)))
    user = c.fetchone()
    conn.close()
    return user is not None

def create_demo_user():
    try:
        if not login_user("demo@vr180.com", "demo123"):
            register_user("demo@vr180.com", "demo123")
    except:
        # If there's any error, just register the demo user
        register_user("demo@vr180.com", "demo123")

# Initialize database
init_db()
create_demo_user()

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'current_state' not in st.session_state:
    st.session_state.current_state = "login"
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'progress' not in st.session_state:
    st.session_state.progress = 0
if 'is_converting' not in st.session_state:
    st.session_state.is_converting = False
if 'error' not in st.session_state:
    st.session_state.error = ""
if 'video_url' not in st.session_state:
    st.session_state.video_url = ""

# Clean CSS - inspired by our Next.js design
st.markdown("""
<style>
    /* Hide Streamlit default elements */
    .stApp > header { visibility: hidden; }
    .stApp > div:first-child { padding-top: 0; }
    
    /* Clean background */
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        min-height: 100vh;
    }
    
    /* Main container */
    .main-container {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem auto;
        max-width: 500px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        border: 1px solid #e2e8f0;
    }
    
    /* Remove all white blocks */
    .main .block-container {
        background: transparent !important;
        padding: 0 !important;
        margin: 0 !important;
        max-width: none !important;
    }
    
    .stApp > div:first-child > div:first-child > div:first-child {
        background: transparent !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Clean buttons */
    .stButton > button {
        background: #059669 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        transition: all 0.2s !important;
    }
    
    .stButton > button:hover {
        background: #047857 !important;
        transform: translateY(-1px) !important;
    }
    
    /* Clean file uploader */
    .stFileUploader {
        border: 2px dashed #cbd5e1 !important;
        border-radius: 8px !important;
        padding: 2rem !important;
        text-align: center !important;
        background: #f8fafc !important;
    }
    
    .stFileUploader:hover {
        border-color: #059669 !important;
        background: #f0fdf4 !important;
    }
    
    /* Clean inputs */
    .stTextInput > div > div > input {
        border: 1px solid #d1d5db !important;
        border-radius: 8px !important;
        padding: 0.75rem !important;
        background: white !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #059669 !important;
        box-shadow: 0 0 0 3px rgba(5, 150, 105, 0.1) !important;
    }
    
    /* Clean progress bar */
    .stProgress > div > div > div > div {
        background: #059669 !important;
    }
    
    /* Clean video player */
    .stVideo {
        border-radius: 8px !important;
        overflow: hidden !important;
    }
    
    /* Clean tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0 !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #f1f5f9 !important;
        border-radius: 8px 8px 0 0 !important;
        border: 1px solid #e2e8f0 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: white !important;
        border-bottom: 1px solid white !important;
    }
    
    /* Remove all padding and margins */
    .main > div {
        padding: 0 !important;
        margin: 0 !important;
    }
    
    .main > div > div {
        padding: 0 !important;
        margin: 0 !important;
    }
    
    .main > div > div > div {
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Hide sidebar */
    .stApp > div:first-child > div:first-child > div:first-child > div:first-child {
        display: none !important;
    }
    
    /* Fix text labels - make them visible */
    .stMarkdown {
        color: #1f2937 !important;
    }
    
    .stMarkdown p {
        color: #1f2937 !important;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #1f2937 !important;
    }
    
    /* Fix form labels */
    .stForm label {
        color: #1f2937 !important;
        font-weight: 600 !important;
    }
    
    /* Fix tab labels */
    .stTabs [data-baseweb="tab"] {
        color: #1f2937 !important;
    }
    
    /* Fix any other text elements */
    .stText, .stSelectbox label, .stTextInput label {
        color: #1f2937 !important;
    }
</style>
""", unsafe_allow_html=True)

# Authentication check
if not st.session_state.authenticated:
    # Clean Login Interface
    st.markdown("""
    <div class="main-container">
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="font-size: 2rem; font-weight: bold; color: #1f2937; margin-bottom: 0.5rem;">VR180 Converter</h1>
            <p style="color: #6b7280; margin: 0;">Login to Continue</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Login/Register Tabs
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        with st.form("login_form"):
            email = st.text_input("Email", placeholder="Enter your email", type="default")
            password = st.text_input("Password", placeholder="Enter your password", type="password")
            login_submitted = st.form_submit_button("Login", use_container_width=True)
            
            if login_submitted:
                if email and password:
                    if login_user(email, password):
                        st.session_state.authenticated = True
                        st.session_state.current_state = "upload"
                        st.rerun()
                    else:
                        st.error("Invalid email or password")
                else:
                    st.error("Please fill in all fields")
    
    with tab2:
        with st.form("register_form"):
            new_email = st.text_input("Email", placeholder="Enter your email", type="default", key="reg_email")
            new_password = st.text_input("Password", placeholder="Enter your password", type="password", key="reg_password")
            register_submitted = st.form_submit_button("Register", use_container_width=True)
            
            if register_submitted:
                if new_email and new_password:
                    if register_user(new_email, new_password):
                        st.success("Registration successful! Please login.")
                    else:
                        st.error("Email already exists")
                else:
                    st.error("Please fill in all fields")
    
    # Demo account
    st.markdown("""
    <div style="text-align: center; margin-top: 1rem; padding: 1rem; background: #f0fdf4; border-radius: 8px; border: 1px solid #bbf7d0;">
        <p style="margin: 0; color: #166534; font-size: 0.9rem;">
            <strong>Demo Account:</strong> demo@vr180.com / demo123
        </p>
    </div>
    """, unsafe_allow_html=True)

else:
    # Main App Interface
    if st.session_state.current_state == "upload":
        st.markdown("""
        <div class="main-container">
            <div style="text-align: center; margin-bottom: 2rem;">
                <h1 style="font-size: 2rem; font-weight: bold; color: #1f2937; margin-bottom: 0.5rem;">Upload your 2D Clip</h1>
                <p style="color: #6b7280; margin: 0;">Select a video file to convert to VR180 format using NeRF technology</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'mov', 'avi'],
            help="MP4, MOV, AVI files supported"
        )
        
        if uploaded_file:
            st.session_state.uploaded_file = uploaded_file
            st.success(f"Selected: {uploaded_file.name} ({(uploaded_file.size / (1024 * 1024)):.2f} MB)")
        
        if st.session_state.error:
            st.error(st.session_state.error)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Upload & Convert with NeRF", disabled=not st.session_state.uploaded_file or st.session_state.is_converting, use_container_width=True):
                st.session_state.current_state = "processing"
                st.session_state.is_converting = True
                st.session_state.error = ""
                st.rerun()
        
        with col2:
            if st.button("Logout", use_container_width=True):
                st.session_state.authenticated = False
                st.session_state.current_state = "login"
                st.rerun()
    
    elif st.session_state.current_state == "processing":
        st.markdown("""
        <div class="main-container">
            <div style="text-align: center; margin-bottom: 2rem;">
                <h1 style="font-size: 2rem; font-weight: bold; color: #1f2937; margin-bottom: 0.5rem;">NeRF Processing</h1>
                <p style="color: #6b7280; margin: 0;">Converting your video to VR180 using Neural Radiance Fields...</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Processing animation
        st.markdown("""
        <div style="text-align: center; margin: 2rem 0;">
            <div style="width: 64px; height: 64px; border: 4px solid #e5e7eb; border-top: 4px solid #059669; border-radius: 50%; animation: spin 1s linear infinite; margin: 0 auto;"></div>
        </div>
        <style>
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Progress bar
        progress = st.progress(0)
        
        # Simulate processing
        for i in range(100):
            progress.progress(i + 1)
            time.sleep(0.05)
        
        # Convert video
        try:
            if st.session_state.uploaded_file:
                # Save uploaded file
                os.makedirs("uploads", exist_ok=True)
                with open(f"uploads/{st.session_state.uploaded_file.name}", "wb") as f:
                    f.write(st.session_state.uploaded_file.getbuffer())
                
                # Convert
                output_path = convert_to_vr180(f"uploads/{st.session_state.uploaded_file.name}")
                
                if output_path and os.path.exists(output_path):
                    st.session_state.video_url = output_path
                    st.session_state.current_state = "result"
                    st.session_state.is_converting = False
                    st.rerun()
                else:
                    st.session_state.error = "Conversion failed"
                    st.session_state.current_state = "upload"
                    st.session_state.is_converting = False
                    st.rerun()
        except Exception as e:
            st.session_state.error = f"Conversion failed: {str(e)}"
            st.session_state.current_state = "upload"
            st.session_state.is_converting = False
            st.rerun()
    
    elif st.session_state.current_state == "result":
        st.markdown("""
        <div class="main-container">
            <div style="text-align: center; margin-bottom: 2rem;">
                <h1 style="font-size: 2rem; font-weight: bold; color: #1f2937; margin-bottom: 0.5rem;">VR180 Conversion Complete</h1>
                <p style="color: #6b7280; margin: 0;">Your NeRF-generated VR180 video is ready</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Video preview
        if st.session_state.video_url:
            st.video(st.session_state.video_url)
            
            # Download button
            if st.button("Download VR180 Video", use_container_width=True):
                with open(st.session_state.video_url, "rb") as file:
                    st.download_button(
                        label="Download",
                        data=file.read(),
                        file_name=f"vr180_{st.session_state.uploaded_file.name}",
                        mime="video/mp4"
                    )
        
        # Convert another button
        if st.button("Convert Another Video", use_container_width=True):
            st.session_state.uploaded_file = None
            st.session_state.progress = 0
            st.session_state.video_url = ""
            st.session_state.error = ""
            st.session_state.current_state = "upload"
            st.rerun()
