import streamlit as st
import sqlite3
import hashlib
import datetime
import time
from io import BytesIO
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
    c.execute('DROP TABLE IF EXISTS users')
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
    if not login_user("demo@vr180.com", "demo123"):
        register_user("demo@vr180.com", "demo123")

# Initialize database
init_db()
create_demo_user()

# Session state
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
if 'video_bytes' not in st.session_state:
    st.session_state.video_bytes = None

# --- Your same CSS styling ---
st.markdown("""
<style>
/* Same CSS from your previous code ... */
</style>
""", unsafe_allow_html=True)

# Authentication
if not st.session_state.authenticated:
    st.markdown("""
    <div class="main-container">
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="font-size: 2rem; font-weight: bold; color: #1f2937; margin-bottom: 0.5rem;">VR180 Converter</h1>
            <p style="color: #6b7280; margin: 0;">Login to Continue</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    with tab1:
        with st.form("login_form"):
            email = st.text_input("Email", placeholder="Enter your email", type="default")
            password = st.text_input("Password", placeholder="Enter your password", type="password")
            if st.form_submit_button("Login", use_container_width=True):
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
            if st.form_submit_button("Register", use_container_width=True):
                if new_email and new_password:
                    if register_user(new_email, new_password):
                        st.success("Registration successful! Please login.")
                    else:
                        st.error("Email already exists")
                else:
                    st.error("Please fill in all fields")
    st.markdown("""
    <div style="text-align: center; margin-top: 1rem; padding: 1rem; background: #f0fdf4; border-radius: 8px; border: 1px solid #bbf7d0;">
        <p style="margin: 0; color: #166534; font-size: 0.9rem;">
            <strong>Demo Account:</strong> demo@vr180.com / demo123
        </p>
    </div>
    """, unsafe_allow_html=True)

# Main App
else:
    if st.session_state.current_state == "upload":
        st.markdown("""
        <div class="main-container">
            <h1 style="text-align:center; font-size:2rem;">Upload your 2D Clip</h1>
        </div>
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose a video file", type=['mp4','mov','avi'])
        if uploaded_file:
            st.session_state.uploaded_file = uploaded_file
            st.success(f"Selected: {uploaded_file.name} ({uploaded_file.size / (1024*1024):.2f} MB)")
        if st.session_state.error:
            st.error(st.session_state.error)
        col1, col2 = st.columns([1,1])
        with col1:
            if st.button("Convert with NeRF", disabled=not st.session_state.uploaded_file or st.session_state.is_converting, use_container_width=True):
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
        st.markdown("<h2 style='text-align:center;'>Processing your video...</h2>", unsafe_allow_html=True)
        progress = st.progress(0)
        for i in range(100):
            progress.progress(i+1)
            time.sleep(0.05)
        try:
            if st.session_state.uploaded_file:
                # Read file into memory
                video_bytes = st.session_state.uploaded_file.read()
                # Convert to VR180 in memory
                output_bytes = convert_to_vr180_bytes(video_bytes)
                st.session_state.video_bytes = output_bytes
                st.session_state.current_state = "result"
                st.session_state.is_converting = False
                st.rerun()
        except Exception as e:
            st.session_state.error = f"Conversion failed: {str(e)}"
            st.session_state.current_state = "upload"
            st.session_state.is_converting = False
            st.rerun()

    elif st.session_state.current_state == "result":
        st.markdown("<h2 style='text-align:center;'>VR180 Conversion Complete</h2>", unsafe_allow_html=True)
        if st.session_state.video_bytes:
            st.video(st.session_state.video_bytes)
            st.download_button(
                "Download VR180 Video",
                data=st.session_state.video_bytes,
                file_name=f"vr180_{st.session_state.uploaded_file.name}",
                mime="video/mp4",
                use_container_width=True
            )
        if st.button("Convert Another Video", use_container_width=True):
            st.session_state.uploaded_file = None
            st.session_state.video_bytes = None
            st.session_state.progress = 0
            st.session_state.error = ""
            st.session_state.current_state = "upload"
            st.rerun()
