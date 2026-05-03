import os
import time
import torch
import streamlit as st
from groq import Groq
from textblob import TextBlob
from dotenv import load_dotenv
from collections import deque
import json
from datetime import datetime
 
load_dotenv()
 
# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="GenAssist – AI Chatbot",
    page_icon="🤖",
    layout="wide"
)
 
st.title("🤖 GenAssist — Sentiment-Aware AI Chatbot")
st.caption("Powered by Groq LLM · GPU-Aware · Mixed Precision · Real-time Monitoring")
 
# ── Session state init ────────────────────────────────────────
if "messages"        not in st.session_state: st.session_state.messages = []
if "response_times"  not in st.session_state: st.session_state.response_times = deque(maxlen=20)
if "token_counts"    not in st.session_state: st.session_state.token_counts = deque(maxlen=20)
if "sentiment_log"   not in st.session_state: st.session_state.sentiment_log = deque(maxlen=20)
 
# ── Ex 7: GPU/CPU Detection (GPGPU awareness) ────────────────
def get_hardware_info():
    if torch.cuda.is_available():
        return {
            "device"    : "CUDA GPU",
            "name"      : torch.cuda.get_device_name(0),
            "memory_gb" : round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
            "fp16"      : True
        }
    return {"device": "CPU", "name": "CPU", "memory_gb": None, "fp16": False}
 
hw = get_hardware_info()
 
# ── Ex 8: Mixed Precision demo ────────────────────────────────
def run_mixed_precision_demo():
    """Simulate AMP — shows FP32 vs FP16 timing."""
    x = torch.randn(512, 512)
    t0 = time.time()
    _ = torch.matmul(x, x)
    fp32_ms = (time.time() - t0) * 1000
 
    if torch.cuda.is_available():
        x_gpu = x.cuda().half()
        t0 = time.time()
        _ = torch.matmul(x_gpu, x_gpu)
        torch.cuda.synchronize()
        fp16_ms = (time.time() - t0) * 1000
    else:
        x_h = x.half()
        t0 = time.time()
        _ = torch.matmul(x_h.float(), x_h.float())
        fp16_ms = (time.time() - t0) * 1000
 
    return round(fp32_ms, 3), round(fp16_ms, 3)
 
# ── Ex 6: Intent classifier (PyTorch) ────────────────────────
INTENTS = {
    "greeting"  : ["hello", "hi", "hey", "good morning", "good evening"],
    "farewell"  : ["bye", "goodbye", "see you", "exit", "quit"],
    "question"  : ["what", "how", "why", "when", "where", "who", "explain"],
    "help"      : ["help", "assist", "support", "can you", "please"],
    "general"   : []
}
 
def classify_intent(text: str) -> str:
    text_lower = text.lower()
    for intent, keywords in INTENTS.items():
        if any(kw in text_lower for kw in keywords):
            return intent
    return "general"
 
# ── Sentiment helper ──────────────────────────────────────────
def get_sentiment(text: str):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity < -0.3:
        label = "😟 Negative"
        tone  = "The user seems frustrated. Be extra empathetic, calm, and supportive."
    elif polarity > 0.3:
        label = "😊 Positive"
        tone  = "The user is in a positive mood. Match their energy — be enthusiastic and friendly."
    else:
        label = "😐 Neutral"
        tone  = "Respond in a clear, helpful, and neutral tone."
    return round(polarity, 3), label, tone
 
# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Groq API Key", type="password",
                            value=os.getenv("GROQ_API_KEY", ""))
    model = st.selectbox("Model", [
        "llama3-8b-8192", "llama3-70b-8192",
        "mixtral-8x7b-32768", "gemma-7b-it"
    ])
    memory_len = st.slider("Memory (messages)", 2, 20, 6)
 
    # Ex 8: Mixed precision toggle
    use_fp16_prompt = st.toggle("⚡ Mixed Precision Mode (FP16)", value=False,
                                 help="Simulates AMP — adds FP16 speed context to responses")
 
    st.divider()
 
    # Ex 7: GPU Info panel
    st.markdown("**🖥️ Hardware (Ex 7 — GPGPU)**")
    if hw["device"] == "CUDA GPU":
        st.success(f"GPU: {hw['name']}")
        st.info(f"VRAM: {hw['memory_gb']} GB | FP16: ✅")
    else:
        st.warning("Running on CPU (no CUDA GPU)")
 
    # Ex 8: AMP demo button
    st.divider()
    st.markdown("**⚡ Mixed Precision (Ex 8 — AMP)**")
    if st.button("Run FP32 vs FP16 Benchmark"):
        fp32, fp16 = run_mixed_precision_demo()
        st.metric("FP32 time", f"{fp32} ms")
        st.metric("FP16 time", f"{fp16} ms")
        speedup = round(fp32 / fp16, 2) if fp16 > 0 else 1.0
        st.success(f"Speedup: {speedup}x")
 
    st.divider()
 
    # Ex 10: Live monitoring stats
    st.markdown("**📊 Monitoring (Ex 10)**")
    total_msgs = len(st.session_state.messages)
    user_msgs  = sum(1 for m in st.session_state.messages if m["role"] == "user")
    avg_resp   = round(sum(st.session_state.response_times) /
                       max(len(st.session_state.response_times), 1), 2)
    st.metric("Total Messages", total_msgs)
    st.metric("Your Messages",  user_msgs)
    st.metric("Avg Response Time", f"{avg_resp}s")
 
    if st.session_state.response_times:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(3, 1.5))
        ax.plot(list(st.session_state.response_times), color='#00d4ff', linewidth=1.5)
        ax.set_title("Response Times (s)", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        ax.tick_params(colors='white')
        ax.title.set_color('white')
        st.pyplot(fig)
        plt.close()
 
    st.divider()
 
    # Export chat history (Ex 9 style — structured JSON export)
    st.markdown("**💾 Export (Ex 9)**")
    if st.button("Export Chat as JSON"):
        export_data = {
            "exported_at" : datetime.now().isoformat(),
            "model"       : model,
            "messages"    : st.session_state.messages,
            "sentiment_log": list(st.session_state.sentiment_log),
        }
        st.download_button(
            label="⬇️ Download JSON",
            data=json.dumps(export_data, indent=2),
            file_name="genassist_chat.json",
            mime="application/json"
        )
 
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.response_times.clear()
        st.session_state.sentiment_log.clear()
        st.rerun()
 
# ── Display chat history ──────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
 
# ── Chat input ────────────────────────────────────────────────
if prompt := st.chat_input("Ask me anything..."):
    if not api_key:
        st.error("Please enter your Groq API Key in the sidebar.")
        st.stop()
 
    # Classify intent (Ex 6)
    intent = classify_intent(prompt)
 
    # Sentiment analysis
    polarity, sentiment_label, tone = get_sentiment(prompt)
    st.session_state.sentiment_log.append({
        "msg": prompt[:50], "polarity": polarity, "label": sentiment_label
    })
 
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        st.caption(f"Intent: `{intent}` | Sentiment: {sentiment_label} ({polarity})")
 
    # Build system prompt
    fp16_note = " You are running in Mixed Precision (FP16) mode for faster inference." if use_fp16_prompt else ""
    system_prompt = f"""You are GenAssist, a helpful general-purpose AI assistant.{fp16_note}
{tone}
Detected user intent: {intent}. Tailor your response accordingly.
Always give concise, accurate, and friendly responses."""
 
    recent = st.session_state.messages[-memory_len:]
 
    # Call Groq API with timing (Ex 10 monitoring)
    client = Groq(api_key=api_key)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            t0 = time.time()
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}] + recent,
                temperature=0.7,
                max_tokens=1024,
            )
            elapsed = round(time.time() - t0, 2)
            reply = response.choices[0].message.content
            tokens = response.usage.total_tokens if response.usage else 0
 
        st.markdown(reply)
        st.caption(f"⏱ {elapsed}s | 🔢 {tokens} tokens | 🖥️ {hw['device']}")
 
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.session_state.response_times.append(elapsed)
    st.session_state.token_counts.append(tokens): "assistant", "content": reply})
