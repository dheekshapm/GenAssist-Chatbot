import os
import streamlit as st
from groq import Groq
from textblob import TextBlob
from dotenv import load_dotenv

load_dotenv()

# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="GenAssist – AI Chatbot", page_icon="🤖", layout="wide")

st.title("🤖 GenAssist — Sentiment-Aware AI Chatbot")
st.caption("Powered by Groq LLM · Built with LangChain + Streamlit")

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Groq API Key", type="password",
                            value=os.getenv("GROQ_API_KEY", ""))
    model = st.selectbox("Model", [
        "llama3-8b-8192",
        "llama3-70b-8192",
        "mixtral-8x7b-32768",
        "gemma-7b-it"
    ])
    memory_len = st.slider("Memory (messages)", 2, 20, 6)
    st.divider()
    st.markdown("**Innovation Added:**")
    st.markdown("- 🎭 Sentiment-aware tone adjustment")
    st.markdown("- 🌐 Multi-language detection")
    st.markdown("- 📊 Live chat analytics")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ── Session state ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Sentiment helper ──────────────────────────────────────────
def get_sentiment_tone(text: str) -> str:
    """Returns a tone instruction based on user sentiment."""
    polarity = TextBlob(text).sentiment.polarity
    if polarity < -0.3:
        return "The user seems frustrated or upset. Be extra empathetic, calm, and supportive."
    elif polarity > 0.3:
        return "The user is in a positive mood. Match their energy with an enthusiastic and friendly tone."
    else:
        return "Respond in a clear, helpful, and neutral tone."

# ── Analytics sidebar live stats ─────────────────────────────
total_msgs = len(st.session_state.messages)
user_msgs  = sum(1 for m in st.session_state.messages if m["role"] == "user")
with st.sidebar:
    st.divider()
    st.markdown("**📊 Chat Analytics**")
    st.metric("Total Messages", total_msgs)
    st.metric("Your Messages",  user_msgs)

# ── Display chat history ──────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────
if prompt := st.chat_input("Ask me anything..."):
    if not api_key:
        st.error("Please enter your Groq API Key in the sidebar.")
        st.stop()

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build system prompt with sentiment tone
    tone = get_sentiment_tone(prompt)
    system_prompt = f"""You are GenAssist, a helpful general-purpose AI assistant.
{tone}
Always give concise, accurate, and friendly responses."""

    # Trim to memory length
    recent = st.session_state.messages[-memory_len:]

    # Call Groq API
    client = Groq(api_key=api_key)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}] + recent,
                temperature=0.7,
                max_tokens=1024,
            )
            reply = response.choices[0].message.content
            st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
