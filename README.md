# 🤖 General Q&A Chatbot

A real-time, multi-turn conversational chatbot powered by Groq's hosted
LLaMA 3.3 70B model — built with Streamlit, with a custom system prompt and a
session-state layer that keeps the full conversation alive across Streamlit's
stateless rerun cycle.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=flat-square&logo=streamlit)
![Groq](https://img.shields.io/badge/Groq-LLaMA%203.3%2070B-orange?style=flat-square)

---

## 📌 What This Project Does

A general-purpose AI assistant that holds a real multi-turn conversation —
not just single question/answer pairs. It:
1. Sends every user message, plus the full conversation history, to Groq's
   `llama-3.3-70b-versatile` model
2. Streams the response back into a chat interface
3. Persists the conversation across reruns using Streamlit session state
4. Handles API errors (rate limits, bad keys, service downtime) gracefully

---

## ⚙️ The Core Engineering Problem This Solves

Streamlit re-executes the **entire script from top to bottom** on every user
interaction — typing a message, clicking a button, anything. Without extra
handling, that would wipe the conversation on every single message.

This app solves it with `st.session_state`, a persistent dict tied to the
user's browser session:

```python
if "messages" not in st.session_state:
    st.session_state.messages = []
```

This only initializes once. On every later rerun, the check is `False`, so
existing history survives.

**Turn-taking flow:**
1. User submits a message → appended to `st.session_state.messages` with
   `role: "user"` → `st.rerun()` is called immediately (so the message shows
   on screen right away, before waiting on the API)
2. Script reruns top to bottom. A check fires: *"is the last message's role
   `user`?"* — if yes, the AI hasn't replied yet, so Groq is called
3. The response is displayed and appended with `role: "assistant"` — on any
   further rerun, the check is now `False`, so no duplicate API calls happen

---

## 💬 System Prompt

Sent as the first message (`role: "system"`) on every API call — the user
never sees it, but it shapes every response:

- Personality: warm, clear, concise, honest about not knowing something
- Behavior: asks for clarification on unclear questions, uses examples,
  formats lists/bullets when helpful
- Boundaries: doesn't fabricate facts, doesn't give professional
  medical/legal/financial advice, doesn't engage with harmful requests

---

## 🛠️ Tech Stack

- **Python** — core logic
- **Groq API** — hosted inference for `llama-3.3-70b-versatile`
- **Streamlit** — chat UI (`st.chat_message`, `st.chat_input`) and session state
- **python-dotenv** — loads `GROQ_API_KEY` from a `.env` file (never hardcoded)

---

## 📁 Project Structure

```
General-Chatbot/
│
├── app.py                  # Full app: system prompt, Groq calls, session state,
│                            #   chat UI, sidebar stats, error handling
├── requirements.txt        # Dependencies
└── .env                    # GROQ_API_KEY (not committed — create locally)
```

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/Precipitation-Rain/General-Chatbot.git
cd General-Chatbot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create a .env file with your Groq API key
echo "GROQ_API_KEY=your_key_here" > .env

# 4. Run the app
streamlit run app.py
```

---

## 🧩 Features

- **Multi-turn context** — full conversation history sent with every request,
  so the model remembers earlier turns in the same session
- **Sidebar stats** — live message count and conversation turn count
- **Clear conversation button** — resets session state and starts fresh
- **Graceful error handling** — distinct messages for rate limits (`429`),
  invalid API key (`401`), and service downtime (`503`), with a generic
  fallback for anything else

---

## 🎯 Why Groq

Groq runs open-weight models like LLaMA on custom inference hardware (LPUs)
built for very low latency — noticeably faster response times than typical
GPU-hosted APIs, which matters directly for a real-time chat experience.

---

## 💡 What I Learned

- How to manage conversational state in a framework (Streamlit) that has no
  built-in memory between interactions by default
- Structuring system/user/assistant roles correctly for a chat completion API
- Designing a system prompt that constrains tone, scope, and refusal behavior
- Handling API failure modes (rate limits, auth errors, downtime) instead of
  letting the app crash on the user

---

## 🔧 Known Limitations / Next Steps

- No persistence beyond the browser session — closing or refreshing the tab
  clears the conversation; would need a database keyed by user/session ID for
  cross-session memory
- Full message history is sent on every call with no trimming or
  summarization — a long conversation could eventually exceed the model's
  context window
- Responses are generated in full before displaying (with a "Thinking..."
  spinner) rather than streamed token-by-token — streaming would improve
  perceived responsiveness

---

## 📬 Author

**Rajvardhan Shewale**
- [GitHub](https://github.com/Precipitation-Rain)
- [LinkedIn](https://www.linkedin.com/in/rajvardhanshewale/)
- [Portfolio](https://sites.google.com/view/rajvardhanshewale1771/home)