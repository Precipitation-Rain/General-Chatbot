# ─────────────────────────────────────────────────────────────────
# General Q&A Chatbot
# Powered by Groq (LLaMA 3.3) + Streamlit
# ─────────────────────────────────────────────────────────────────

import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import os

# SYSTEM CONFIGURATION ____________________________________________

# load_dotenv reads .env file and puts values into environment variables
load_dotenv()

# Create Groq client — this is how your code talks to Groq API
# Think of client as the connection object — created once, reused everywhere
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Must be the first Streamlit command — no exceptions
st.set_page_config(
    page_icon='🤖',
    page_title='AI Chatbot',
    layout='wide'
)

# SYSTEM PROMPT ___________________________________________________

# Sent with every API call as the first message with role "system"
# User never sees this — it silently shapes every response
# This is where you define who the AI is and how it behaves

SYSTEM_PROMPT = """
You are a helpful, knowledgeable, and friendly AI assistant.

YOUR PERSONALITY:
- You are warm, clear, and approachable
- You explain complex topics in simple language
- You are honest — if you don't know something, you say so
- You are concise but thorough — no unnecessary padding

YOUR BEHAVIOUR:
- Answer questions on any topic the user asks about
- If a question is unclear, ask for clarification before answering
- Use examples to explain difficult concepts
- Format your answers with bullet points or numbering when listing things
- Keep responses focused — do not go off topic

WHAT YOU DO NOT DO:
- You do not make up facts
- You do not give medical, legal, or financial advice as a professional
- You do not engage with harmful or inappropriate requests
"""

# AI RESPONSE FUNCTION ____________________________________________

def get_groq_response(user_message, conversation_history):
    """
    Sends user message + full conversation history to Groq.
    Returns Groq's response as a plain string.

    How it works:
    - First message is always the system prompt with role "system"
    - Then all previous conversation messages are added
    - Finally the current user message is already inside conversation_history
    - Groq reads everything and writes the next response

    Why no format conversion needed (unlike Gemini):
    - Groq uses "user" and "assistant" as role names
    - Our session_state also uses "user" and "assistant"
    - They match — so we can pass messages directly without changing anything
    """

    # Start with system prompt as the first message
    # role "system" is special — it is instructions, not conversation
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    # Add full conversation history after the system prompt
    # Each msg is {"role": "user"/"assistant", "content": "text"}
    # This is exactly what Groq expects — no conversion needed
    for msg in conversation_history:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    # Make the API call to Groq
    # model — llama-3.3-70b-versatile is free and very capable
    # messages — full list: system + history + current message
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )

    # Extract and return the text
    # response.choices[0] — first (and only) response generated
    # .message.content    — the actual text string
    return response.choices[0].message.content


# INITIALIZE SESSION STATE ________________________________________

# Runs every rerun but only CREATES keys the first time
# After that the if check fails and existing values are preserved

if "messages" not in st.session_state:
    st.session_state.messages = []
    # Grows like this as conversation happens:
    # [
    #   {"role": "user",      "content": "What is Python?"},
    #   {"role": "assistant", "content": "Python is a programming language..."},
    #   {"role": "user",      "content": "Who created it?"},
    #   {"role": "assistant", "content": "Guido van Rossum created Python..."}
    # ]

if "total_messages" not in st.session_state:
    st.session_state.total_messages = 0
    # Simple counter — increments each time user sends a message


# SIDEBAR _________________________________________________________

with st.sidebar:

    st.title("🤖 AI Chatbot")
    st.markdown("A general purpose AI assistant powered by Groq + LLaMA 3.3")
    st.markdown("---")

    # Clear button — wipes conversation and restarts fresh
    # session_state.messages becomes empty list
    # counter resets to 0
    # st.rerun() redraws the page with empty chat
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_messages = 0
        st.rerun()

    st.markdown("---")

    # Conversation statistics
    st.markdown("**Conversation Stats**")
    st.markdown(f"Messages sent: **{st.session_state.total_messages}**")
    st.markdown(f"Total turns: **{len(st.session_state.messages) // 2}**")

    st.markdown("---")

    # Tips for user
    st.markdown("**💡 You can ask about:**")
    st.markdown("""
    - Any general knowledge topic
    - Science, history, geography
    - Coding and technology
    - Math explanations
    - Writing and grammar help
    - Concept explanations
    - Current best practices
    """)

    st.markdown("---")
    st.caption("Powered by Groq — LLaMA 3.3 70B")


# MAIN CHAT AREA __________________________________________________

st.title("💬 General Q&A Chatbot")
st.caption("Ask me anything — I am here to help.")
st.markdown("---")

# Welcome message — only shows when conversation is empty
# Once any message exists this block is skipped entirely
# Prevents welcome message from appearing mid-conversation
if len(st.session_state.messages) == 0:
    with st.chat_message("assistant"):
        st.markdown("""
        👋 Hello! I am your AI assistant powered by Groq and LLaMA 3.3.
        
        I can help you with:
        - **Answering questions** on any topic
        - **Explaining concepts** in simple terms
        - **Helping with writing** — drafts, edits, summaries
        - **Coding help** — explaining code, debugging, examples
        - **General conversation** — just chat!
        
        What would you like to know today?
        """)


# CONVERSATION HISTORY ____________________________________________

# Redraws every message on every rerun
# Fast — just rendering stored text, no API calls
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# USER INPUT ______________________________________________________

# Text box pinned to bottom of page
# Returns typed text when user presses Enter
# Returns None if nothing submitted
user_input = st.chat_input("Type your message here...")

if user_input:

    # Save user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Increment counter shown in sidebar
    st.session_state.total_messages += 1

    # Rerun so:
    # 1. User message appears in chat display loop above
    # 2. AI response logic below detects last message is "user" and fires
    st.rerun()


# AI RESPONSE LOGIC _______________________________________________

# Runs after every rerun
# Checks if last message is from user — meaning AI has not responded yet
# If yes — calls Groq and displays response
# If no  — AI already responded, nothing to do

if (st.session_state.messages and
        st.session_state.messages[-1]["role"] == "user"):

    latest_user_message = st.session_state.messages[-1]["content"]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            try:
                response = get_groq_response(
                    latest_user_message,
                    st.session_state.messages
                )

                # Display response in chat bubble
                st.markdown(response)

                # Save to history — critical, without this response
                # disappears on next rerun
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })

            except Exception as e:
                error_str = str(e)

                # Handle specific error types with clear messages
                if "429" in error_str:
                    st.warning("⏳ Rate limit hit. Please wait a moment and try again.")
                elif "401" in error_str:
                    st.error("❌ Invalid API key. Check your GROQ_API_KEY in .env file")
                elif "503" in error_str:
                    st.error("❌ Groq service unavailable. Try again in a moment.")
                else:
                    st.error(f"❌ Error: {error_str}")
                    st.info("Check your GROQ_API_KEY in the .env file")