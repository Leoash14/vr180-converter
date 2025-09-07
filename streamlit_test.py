import streamlit as st
import sqlite3
import hashlib
import datetime

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
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  email TEXT UNIQUE,
                  password TEXT,
                  created_at TIMESTAMP)''')
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(email, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (email, password, created_at) VALUES (?, ?, ?)",
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
    c.execute("SELECT * FROM users WHERE email = ? AND password = ?",
              (email, hash_password(password)))
    user = c.fetchone()
    conn.close()
    return user is not None

def create_demo_user():
    if not login_user("demo@vr180.com", "demo123"):
        register_user("demo@vr180.com", "demo123")

# Initialize
init_db()
create_demo_user()

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'current_state' not in st.session_state:
    st.session_state.current_state = "login"

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
    st.markdown("""
    <div class="main-container">
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="font-size: 2rem; font-weight: bold; color: #1f2937; margin-bottom: 0.5rem;">VR180 Converter</h1>
            <p style="color: #6b7280; margin: 0;">Clean UI Test - No White Blocks!</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.success("🎉 Clean UI is working! No white blocks!")
    
    if st.button("Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.current_state = "login"
        st.rerun()
