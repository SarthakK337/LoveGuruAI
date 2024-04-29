import logging
import streamlit as st
import requests
import base64
import time
import json
from PIL import Image, ImageEnhance
import google.generativeai as genai
# from langchain_google_genai
import os

class GeminiAPI:
  def __init__(self, api_key):
    genai.configure(api_key=api_key)


  def send_message(self, messages):
    # headers = {
    #   "Authorization": f"Bearer {self.api_key}",
    #   "Content-Type": "application/json"
    # }
    # payload = {
    #   "messages": messages,
    #   "model": model,
    #   "temperature": temperature,
    #   "max_tokens": max_tokens,
    # }
    model = genai.GenerativeModel('gemini-pro')
    # response = model.generate_content("Give me python code to sort a list")
    response = model.generate_content(messages)

    if response.status_code == 200:
      return response.text
    else:
      raise Exception(f"Error: {response.status_code} - {response.text}")


# Initialize Gemini API Client
# gemini_client = 

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Streamlit Page Configuration
st.set_page_config(
    page_title="Streamly Streamlit Assistant",
    page_icon="imgs/avatar_streamly.png",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": "https://github.com/AdieLaine/Streamly",
        "Report a bug": "https://github.com/AdieLaine/Streamly",
        "About": """
            ## Streamly Streamlit Assistant
            
            **GitHub**: https://github.com/AdieLaine/
            
            The AI Assistant named, Streamly, aims to provide the latest updates from Streamlit,
            generate code snippets for Streamlit widgets,
            and answer questions about Streamlit's latest features, issues, and more.
            Streamly has been trained on the latest Streamlit updates and documentation.
        """
    }
)

# Streamlit Updates and Expanders
st.title("Streamly Streamlit Assistant")

API_DOCS_URL = "https://docs.streamlit.io/library/api-reference"

@st.cache_data(show_spinner=False)
def load_and_enhance_image(image_path, enhance=False):
    """
    Load and optionally enhance an image.

    Parameters:
    - image_path: str, path of the image
    - enhance: bool, whether to enhance the image or not

    Returns:
    - img: PIL.Image.Image, (enhanced) image
    """
    img = Image.open(image_path)
    if enhance:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.8)
    return img

@st.cache_data(show_spinner=False)
def load_streamlit_updates():
    """Load the latest Streamlit updates from a local JSON file."""
    try:
        with open("data/streamlit_updates.json", "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def display_streamlit_updates():
    """It displays the latest updates of the Streamlit."""
    with st.expander("Streamlit 1.32 Announcement", expanded=False):
        image_path = "imgs/streamlit128.png"
        enhance = st.checkbox("Enhance Image?", False)
        img = load_and_enhance_image(image_path, enhance)
        st.image(img, caption="Streamlit 1.32 Announcement", use_column_width="auto", clamp=True, channels="RGB", output_format="PNG")
        st.markdown("For more details on this version, check out the [Streamlit Forum post](https://docs.streamlit.io/library/changelog#version-1320).")

def img_to_base64(image_path):
    """Convert image to base64"""
    with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()

@st.cache_data(show_spinner=False)
def on_chat_submit(chat_input, api_key, latest_updates):
    """
    Handle chat input submissions and interact with the Gemini API.

    Parameters:
        chat_input (str): The chat input from the user.
        api_key (str): The Gemini API key.
        latest_updates (dict): The latest Streamlit updates fetched from a JSON file or API.

    Returns:
        None: Updates the chat history in Streamlit's session state.
    """
    user_input = chat_input.strip().lower()

    # Initialize the conversation history with system and assistant messages
    if 'conversation_history' not in st.session_state:
        # Initialize conversation_history
        st.session_state.conversation_history = [
            {"role": "system", "content": "You are Streamly, a specialized AI assistant trained in Streamlit."},
            {"role": "system", "content": "Refer to conversation history to provide context to your response."},
            {"role": "system", "content": "You are trained up to Streamlit Version 1.32.0."},
            {"role": "assistant", "content": "Hello! I am Streamly. How can I assist you with Streamlit today?"}
        ]

    # Append user's query to conversation history
    st.session_state.conversation_history.append({"role": "user", "content": user_input})

    try:
        # Logic for assistant's reply
        assistant_reply = ""

        # Direct Gemini API call
        assistant_reply = GeminiAPI.send_message(st.secrets["GOOGLE_API_KEY"],st.session_state.conversation_history)
        # assistant_reply = GeminiAPI.send_message(messages='Hello Sarthak')

        # Append assistant's reply to the conversation history
        st.session_state.conversation_history.append({"role": "assistant", "content": assistant_reply})

        # Update the Streamlit chat history
        if "history" in st.session_state:
            st.session_state.history.append({"role": "user", "content": user_input})
            st.session_state.history.append({"role": "assistant", "content": assistant_reply})

    except requests.exceptions.RequestException as e:
        logging.error(f"Error occurred: {e}")
        error_message = f"Gemini API Error: {str(e)}"
        st.error(error_message)

def main():
    """
    Display Streamlit updates and handle the chat interface.
    """
    # Initialize session state variables for chat history and conversation history
    if "history" not in st.session_state:
        st.session_state.history = []
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    # Initialize the chat with a greeting and Streamlit updates if the history is empty
    if not st.session_state.history:
        latest_updates = load_streamlit_updates()  # This function should be defined elsewhere to load updates
        initial_bot_message = "Hello! How can I assist you with Streamlit today? Here are some of the latest highlights:\n"
        updates = latest_updates.get("Highlights", {})
        if isinstance(updates, dict):  # Check if updates is a dictionary
            initial_bot_message = "I am Streamly, what can I help with today?"
            st.session_state.history.append({"role": "assistant", "content": initial_bot_message})
            st.session_state.conversation_history = [
                {"role": "system", "content": "You are Streamly, a specialized AI assistant trained to assist with the logic and programming using Streamlit."},
                {"role": "system", "content": "Refer to conversation history to provide context to your response."},
                {"role": "system", "content": "Use the streamlit_updates.json local file to look up the latest Streamlit feature updates."},
                {"role": "system", "content": "When responding, provide code examples, links to documentation, and code examples from Streamlit API to help the user."},
                {"role": "assistant", "content": initial_bot_message}
            ]
        else:
            st.error("Unexpected structure for 'Highlights' in latest updates.")
    
    # Inject custom CSS for glowing border effect
    st.markdown(
        """
        <style>
        .cover-glow {
            width: 100%;
            height: auto;
            padding: 3px;
            box-shadow: 
                0 0 5px #330000,
                0 0 10px #660000,
                0 0 15px #990000,
                0 0 20px #CC0000,
                0 0 25px #FF0000,
                0 0 30px #FF3333,
                0 0 35px #FF6666;
            position: relative;
            z-index: -1;
            border-radius: 30px;  /* Rounded corners */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Function to convert image to base64
    def img_to_base64(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()

    # Load and display sidebar image with glowing effect
    img_path = "imgs/sidebar_streamly_avatar.png"
    img_base64 = img_to_base64(img_path)
    st.sidebar.markdown(
        f'<img src="data:image/png;base64,{img_base64}" class="cover-glow">',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")
    
    # Sidebar for Mode Selection
    mode = st.sidebar.radio("Select Mode:", options=["Latest Updates", "Chat with Streamly"], index=1)
    st.sidebar.markdown("---")
    
    # Handle Chat and Update Modes
    if mode == "Chat with Streamly":
        chat_input = st.text_input("Ask me about Streamlit updates:")
        if st.button("Submit"):
            if chat_input:
                latest_updates = load_streamlit_updates()
                GeminiAPI(st.secrets["GOOGLE_API_KEY"])
                on_chat_submit(chat_input, api_key=st.secrets["GOOGLE_API_KEY"], latest_updates=latest_updates)

        # Display chat history with custom avatars
        for message in st.session_state.history[-20:]:
            role = message["role"]
            
            # Set avatar based on role
            if role == "assistant":
                avatar_image = "imgs/avatar_streamly.png"
            elif role == "user":
                avatar_image = "imgs/stuser.png"
            else:
                avatar_image = None  # Default
            
            with st.chat_message(role, avatar=avatar_image):
                st.write(message["content"])

    else:
        display_streamlit_updates()

if __name__ == "__main__":
    main()
