"""
Fashion Chatbot UI with Gemini AI Integration
This connects to code2_modified.py for image analysis
"""

import streamlit as st
from groq import Groq
from code2_modified import analyze_user_images
import json
import os
from datetime import datetime
from PIL import Image
import base64

# =============================================================================
# CONFIGURATION - ADD YOUR GROQ API KEY HERE
# =============================================================================
GROQ_API_KEY = "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"  # Replace with your actual Groq API key
# =============================================================================

# Configure Groq API
if GROQ_API_KEY and GROQ_API_KEY != "YOUR_GROQ_API_KEY_HERE":
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    groq_client = None

# Page configuration
st.set_page_config(
    page_title="AI Fashion Stylist",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Premium Modern Design
st.markdown("""
<style>
    /* Main Container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Header Styles */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        padding: 1rem;
        animation: fadeIn 1s ease-in;
    }
    
    .sub-header {
        font-size: 1.3rem;
        text-align: center;
        color: #6b7280;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Chat Messages */
    .chat-message {
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        display: flex;
        flex-direction: column;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        animation: slideIn 0.3s ease-out;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 15%;
        border-bottom-right-radius: 5px;
    }
    
    .bot-message {
        background: white;
        color: #1f2937;
        margin-right: 15%;
        border: 1px solid #e5e7eb;
        border-bottom-left-radius: 5px;
    }
    
    .message-header {
        font-weight: 700;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
        opacity: 0.9;
    }
    
    .message-content {
        line-height: 1.6;
        font-size: 1rem;
    }
    
    /* Upload Section */
    .upload-section {
        background: white;
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
    }
    
    /* Analysis Box */
    .analysis-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.8rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
        transition: transform 0.3s ease;
    }
    
    .analysis-box:hover {
        transform: translateY(-5px);
    }
    
    .analysis-box h4 {
        font-weight: 700;
        margin-bottom: 1rem;
        font-size: 1.2rem;
    }
    
    /* Profile Card */
    .profile-card {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.6);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    section[data-testid="stSidebar"] .stRadio label {
        color: white;
    }
    
    section[data-testid="stSidebar"] h2 {
        color: white;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Chat Input */
    .stChatInput {
        border-radius: 25px;
    }
    
    /* File uploader */
    .uploadedFile {
        border-radius: 10px;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #764ba2;
    }
    
    /* Success/Error boxes */
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'user_data' not in st.session_state:
    st.session_state.user_data = None
if 'current_session' not in st.session_state:
    st.session_state.current_session = datetime.now().strftime("%Y%m%d_%H%M%S")

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # API Key status
    if GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE":
        st.error("⚠️ Add Groq API Key")
        st.info("Edit line 14 in chatbot_ui.py")
    else:
        st.success("✅ Groq AI Connected")
    
    st.markdown("---")
    
    # Gender selection
    gender = st.radio("Select Gender", ["Male", "Female"], index=0)
    
    st.markdown("---")
    
    # Chat history
    st.markdown("## 📜 Chat History")
    if st.session_state.chat_history:
        st.info(f"Messages: {len(st.session_state.chat_history)}")
        if st.button("🗑️ Clear History"):
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.info("No messages yet")
    
    st.markdown("---")
    
    # New session
    if st.button("🔄 New Session"):
        st.session_state.chat_history = []
        st.session_state.analysis_done = False
        st.session_state.user_data = None
        st.session_state.current_session = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.rerun()

# Main content
st.markdown('<div class="main-header">✨ AI Fashion Stylist</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Your Personal Style Advisor • Powered by Advanced AI</div>', unsafe_allow_html=True)

# Step 1: Image Upload Section
if not st.session_state.analysis_done:
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### 📸 Step 1: Upload Your Images")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Face Image**")
        face_image = st.file_uploader(
            "Upload a clear front-facing photo", 
            type=['jpg', 'jpeg', 'png'],
            key="face_upload"
        )
        if face_image:
            st.image(face_image, caption="Face Image", width=300)
    
    with col2:
        st.markdown("**Body Image**")
        body_image = st.file_uploader(
            "Upload a full-body photo", 
            type=['jpg', 'jpeg', 'png'],
            key="body_upload"
        )
        if body_image:
            st.image(body_image, caption="Body Image", width=300)
    
    if face_image and body_image:
        if st.button("🔍 Analyze Images", type="primary", use_container_width=True):
            with st.spinner("Analyzing your images... This may take a moment."):
                # Save uploaded images temporarily
                face_path = f"temp_face_{st.session_state.current_session}.jpg"
                body_path = f"temp_body_{st.session_state.current_session}.jpg"
                
                with open(face_path, "wb") as f:
                    f.write(face_image.getbuffer())
                with open(body_path, "wb") as f:
                    f.write(body_image.getbuffer())
                
                # Analyze images
                analysis_result = analyze_user_images(face_path, body_path)
                
                if analysis_result:
                    st.session_state.user_data = analysis_result
                    st.session_state.analysis_done = True
                    
                    # Clean up temp files
                    if os.path.exists(face_path):
                        os.remove(face_path)
                    if os.path.exists(body_path):
                        os.remove(body_path)
                    
                    st.success("✅ Analysis complete! You can now chat with your fashion advisor.")
                    st.rerun()
                else:
                    st.error("❌ Could not analyze images. Please ensure your images are clear and try again.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Step 2: Display Analysis Results
else:
    st.markdown("### ✨ Your Personalized Style Profile")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="analysis-box">
            <h4>🎨 Color Analysis</h4>
        </div>
        """, unsafe_allow_html=True)
        st.write(f"**Skin Tone:** {st.session_state.user_data['skin_tone']}")
        st.write(f"**Undertone:** {st.session_state.user_data['undertone']}")
        st.write(f"**HEX:** {st.session_state.user_data['hex_color']}")
    
    with col2:
        st.markdown("""
        <div class="analysis-box">
            <h4>👤 Physical Features</h4>
        </div>
        """, unsafe_allow_html=True)
        st.write(f"**Hair Color:** {st.session_state.user_data['hair_color']}")
        st.write(f"**Eye Color:** {st.session_state.user_data['eye_color']}")
        st.write(f"**Body Shape:** {st.session_state.user_data['body_shape']}")
    
    with col3:
        st.markdown("""
        <div class="analysis-box">
            <h4>👕 Style Guide</h4>
        </div>
        """, unsafe_allow_html=True)
        recs = st.session_state.user_data['recommendations']
        st.write(f"**Best Colors:** {recs['Recommended Clothing Colors'][:25]}...")
        st.write(f"**Best Fit:** {recs['Recommended Fitting Style']}")
    
    # Display visualization if available
    if os.path.exists('analysis_results.png'):
        with st.expander("📊 View Detailed Analysis"):
            st.image('analysis_results.png')
    
    st.markdown("---")
    
    # Step 3: Chatbot Interface
    st.markdown("---")
    st.markdown("### 💬 Chat with Your AI Fashion Advisor")
    st.markdown("")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f'''
                <div class="chat-message user-message">
                    <div class="message-header">You</div>
                    <div class="message-content">{message["content"]}</div>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="chat-message bot-message">
                    <div class="message-header">✨ Fashion Advisor</div>
                    <div class="message-content">{message["content"]}</div>
                </div>
                ''', unsafe_allow_html=True)
    
    # Chat input
    user_input = st.chat_input("Ask me anything about your style, outfits, or fashion advice...")
    
    if user_input:
        if GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE":
            st.error("⚠️ Please add your Groq API key in chatbot_ui.py file (line 14)")
        else:
            # Add user message to history
            st.session_state.chat_history.append({
                'role': 'user',
                'content': user_input
            })
            
            # Prepare context for Gemini
            user_data = st.session_state.user_data
            recommendations = user_data['recommendations']
            
            context_prompt = f"""
Role: Personal fashion stylist and advisor.

Objective: Give personalized clothing and garment recommendations for {gender.lower()} based on the analysis below.

USER PROFILE:
- Skin Tone: {user_data['skin_tone']}
- Undertone: {user_data['undertone']}
- HEX Code: {user_data['hex_color']}
- Hair Color: {user_data['hair_color']}
- Eye Color: {user_data['eye_color']}
- Body Shape: {user_data['body_shape']}
- Torso Length: {user_data['user_profile']['Torso length']:.4f}

STYLE RECOMMENDATIONS:
- Recommended Clothing Colors: {recommendations['Recommended Clothing Colors']}
- Avoid Clothing Colors: {recommendations['Avoid Clothing Colors']}
- Recommended Fitting Style: {recommendations['Recommended Fitting Style']}
- Recommended Materials: {recommendations['Recommended Materials']}
- Recommended Patterns: {recommendations['Recommended Patterns']}
- Recommended Jewelry Metal: {recommendations['Recommended Jewelry Metal']}
- Recommended Shoes: {recommendations['Recommended Shoes']}
- Recommended Color Wheel Region: {recommendations['Recommended Clothing Color Wheel Region']}
- Avoid Color Wheel Region: {recommendations['Avoid Clothing Color Wheel Region']}
- Fabric Nature: {recommendations['Fabric Nature']}
- Don't Exaggerate: {recommendations["Don't Exaggerate"]}
- Do Exaggerate: {recommendations['Do Exaggerate']}

CURRENT SEASON: Indian climate (consider both summer and winter options)

USER QUESTION: {user_input}

Please provide personalized, specific fashion advice. Include outfit suggestions with top and bottom combinations, color palettes, and styling tips suitable for India. Be conversational, friendly, and practical.
"""
            
            try:
                # Call Groq API with system message
                chat_completion = groq_client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert fashion stylist and personal style advisor. Provide personalized, practical fashion advice based on the user's physical attributes and preferences. Be conversational, friendly, and specific in your recommendations."
                        },
                        {
                            "role": "user",
                            "content": context_prompt
                        }
                    ],
                    model="llama-3.3-70b-versatile",  # Fast and capable model
                    temperature=0.7,
                    max_tokens=2000,
                )
                bot_response = chat_completion.choices[0].message.content
                
                # Add bot response to history
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': bot_response
                })
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error communicating with Groq AI: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888; padding: 2rem;'>Powered by Groq AI (Llama 3.3) • Advanced Image Analysis • Machine Learning</div>", 
    unsafe_allow_html=True
)
