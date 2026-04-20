import streamlit as st
import requests
import time

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Ticket Classifier",
    page_icon="🎫",
    layout="wide"
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎫 Customer Support Ticket Classifier")
st.markdown("Powered by **DistilBERT** fine-tuned on real customer support data · 99% accuracy")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("System Status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success("API Online")
        st.metric("Model", "DistilBERT")
        st.metric("Device", health.get("device", "cpu").upper())
        st.metric("Uptime", f"{health.get('uptime_secs', 0):.0f}s")
    except:
        st.error("API Offline — start with uvicorn")

    st.divider()
    st.header("About")
    st.markdown("""
    **MLDLOps Course Project**
    
    **Student:** Abhimanyu Gupta  
    **Roll:** B22BB001
    
    **Pipeline:**
    - DVC data versioning
    - WandB experiment tracking
    - FastAPI serving
    - Docker Compose
    - Prometheus monitoring
    - Evidently drift detection
    - GitHub Actions CI/CD
    """)

# ── Main layout ───────────────────────────────────────────────────────────────
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Enter Support Ticket")
    text_input = st.text_area(
        label="Ticket text",
        placeholder="Describe the customer issue here...",
        height=160,
        label_visibility="collapsed"
    )

    st.markdown("**Try an example:**")
    examples = {
        "Billing": "I was charged twice for my subscription this month",
        "Technical": "The app keeps crashing every time I try to upload a file",
        "Refund": "I want my money back for the order I returned last week",
        "Cancellation": "Please cancel my premium subscription immediately",
        "Product": "What payment methods do you accept for annual plans?",
    }
    cols = st.columns(5)
    for i, (label, text) in enumerate(examples.items()):
        if cols[i].button(label, use_container_width=True):
            st.session_state["example_text"] = text
            st.rerun()

    if "example_text" in st.session_state:
        text_input = st.session_state.pop("example_text")
        st.rerun()

    classify_btn = st.button("Classify Ticket", type="primary", use_container_width=True)

with col2:
    st.subheader("Prediction Result")

    if classify_btn and text_input.strip():
        with st.spinner("Classifying..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"text": text_input},
                    timeout=10
                )
                result = response.json()

                predicted = result["predicted_class"]
                confidence = result["confidence"]
                all_scores = result["all_scores"]
                latency = result["latency_ms"]

                # Color map
                color_map = {
                    "Billing inquiry":       "#3B82F6",
                    "Technical issue":       "#EF4444",
                    "Refund request":        "#10B981",
                    "Cancellation request":  "#F59E0B",
                    "Product inquiry":       "#8B5CF6",
                }
                color = color_map.get(predicted, "#666")

                st.markdown(f"""
                <div style="background:{color}18; border-left:4px solid {color};
                     padding:16px; border-radius:8px; margin-bottom:16px;">
                    <div style="font-size:11px; color:{color}; font-weight:600; 
                         text-transform:uppercase; letter-spacing:1px;">Predicted Category</div>
                    <div style="font-size:22px; font-weight:700; color:{color}; margin:4px 0;">
                        {predicted}
                    </div>
                    <div style="font-size:13px; color:#666;">
                        {confidence:.1%} confidence · {latency}ms
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("**Confidence scores:**")
                for label, score in sorted(all_scores.items(), key=lambda x: -x[1]):
                    col_a, col_b = st.columns([3, 7])
                    col_a.caption(label)
                    col_b.progress(score, text=f"{score:.1%}")

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Make sure it is running on port 8000.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    elif classify_btn and not text_input.strip():
        st.warning("Please enter some ticket text first.")
    else:
        st.info("Enter a ticket on the left and click Classify.")

# ── Metrics section ───────────────────────────────────────────────────────────
st.divider()
st.subheader("Model Performance")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Test Accuracy", "99.0%", "+79% vs baseline")
m2.metric("Macro F1", "0.989")
m3.metric("Training Time", "4.5 min", "T4 GPU")
m4.metric("Inference Latency", "~60ms", "CPU")

st.divider()
st.caption("MLDLOps Course Project · Abhimanyu Gupta · B22BB001 · IIT Rajasthan · 2026")