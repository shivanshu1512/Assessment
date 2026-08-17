"""
Streamlit Live Demo — Support Ticket Classifier
================================================
Run with:  streamlit run app.py
"""

import re
import warnings
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")   # headless backend for Streamlit Cloud
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────────

CATEGORIES = ["Billing", "Technical", "HR", "General"]
CONFIDENCE_THRESHOLD = 0.60
URGENT_KEYWORDS = [
    "urgent", "immediately", "critical", "asap", "down",
    "not working", "broken", "emergency", "cannot", "failed",
]
DATA_PATH = "data/tickets.csv"

CATEGORY_ICONS = {
    "Billing": "💳",
    "Technical": "🔧",
    "HR": "👥",
    "General": "📋",
}
CATEGORY_COLORS = {
    "Billing": "#4361EE",
    "Technical": "#F72585",
    "HR": "#4CC9F0",
    "General": "#7B2FBE",
}


# ──────────────────────────────────────────────────
#  Helper Functions
# ──────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_priority(text: str) -> str:
    for kw in URGENT_KEYWORDS:
        if kw in text.lower():
            return "URGENT"
    return "NORMAL"


# ──────────────────────────────────────────────────
#  Model Training (cached so it only runs once)
# ──────────────────────────────────────────────────

@st.cache_resource
def load_and_train():
    df = pd.read_csv(DATA_PATH)
    df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")
    df["clean_text"] = df["text"].apply(clean_text)
    df["category"] = df["category"].str.strip()

    X_raw = df["clean_text"]
    y = df["category"]
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)

    nb = MultinomialNB(alpha=0.5)
    nb.fit(X_train, y_train)

    lr = LogisticRegression(max_iter=1000, C=5.0, solver="lbfgs")
    lr.fit(X_train, y_train)

    acc_nb = accuracy_score(y_test, nb.predict(X_test))
    acc_lr = accuracy_score(y_test, lr.predict(X_test))
    best_model = lr if acc_lr >= acc_nb else nb
    best_name = "Logistic Regression" if acc_lr >= acc_nb else "Naive Bayes"

    return {
        "vectorizer": vectorizer,
        "nb": nb,
        "lr": lr,
        "best_model": best_model,
        "best_name": best_name,
        "acc_nb": acc_nb,
        "acc_lr": acc_lr,
        "X_test": X_test,
        "y_test": y_test,
        "df": df,
    }


def predict_ticket(model, vectorizer, ticket_text: str) -> dict:
    clean = clean_text(ticket_text)
    X = vectorizer.transform([clean])
    proba = model.predict_proba(X)[0]
    classes = model.classes_

    predicted_idx = np.argmax(proba)
    predicted_category = classes[predicted_idx]
    confidence = proba[predicted_idx]
    needs_review = confidence < CONFIDENCE_THRESHOLD
    priority = get_priority(ticket_text)

    return {
        "predicted_category": predicted_category,
        "confidence": round(float(confidence) * 100, 2),
        "needs_human_review": needs_review,
        "priority": priority,
        "all_probabilities": {
            cat: round(float(p) * 100, 2)
            for cat, p in zip(classes, proba)
        },
    }


# ──────────────────────────────────────────────────
#  Streamlit UI
# ──────────────────────────────────────────────────

st.set_page_config(
    page_title="Ticket Classifier | Shivanshu Shukla",
    page_icon="🎫",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
    }
    .main-header h1 { margin: 0; font-size: 2rem; font-weight: 700; }
    .main-header p  { margin: 0.5rem 0 0; opacity: 0.85; font-size: 1rem; }

    .metric-card {
        background: white;
        border: 1px solid #e8ecf0;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #4361EE; }
    .metric-label { font-size: 0.85rem; color: #6b7280; margin-top: 0.2rem; }

    .result-box {
        background: #f8faff;
        border: 2px solid #4361EE;
        border-radius: 14px;
        padding: 1.5rem 2rem;
        margin-top: 1rem;
    }
    .badge {
        display: inline-block;
        padding: 0.3rem 0.9rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        color: white;
        margin-right: 0.5rem;
    }
    .urgent-badge { background: #ef4444; }
    .normal-badge { background: #22c55e; }
    .review-badge { background: #f59e0b; }
    .auto-badge   { background: #4361EE; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🎫 Support Ticket Classifier</h1>
    <p>AI-powered ticket routing · TF-IDF + Naive Bayes / Logistic Regression · Built by Shivanshu Shukla</p>
</div>
""", unsafe_allow_html=True)

# Load model
with st.spinner("Training model on ticket dataset..."):
    artifacts = load_and_train()

vectorizer = artifacts["vectorizer"]
best_model = artifacts["best_model"]
best_name  = artifacts["best_name"]

# ── Tabs ──
tab1, tab2, tab3 = st.tabs(["🚀 Classify Ticket", "📊 Model Performance", "🔍 Dataset Explorer"])

# ─── TAB 1: Classify ───
with tab1:
    st.subheader("Enter a Support Ticket")

    col1, col2 = st.columns([3, 1])
    with col1:
        ticket_input = st.text_area(
            "Ticket text (subject + body)",
            placeholder="e.g. I was charged twice this month and need a refund...",
            height=140,
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        model_choice = st.radio("Model", ["Best (Auto)", "Logistic Regression", "Naive Bayes"])
        threshold = st.slider("Confidence threshold (%)", 40, 90, 60)

    classify_btn = st.button("🔍 Classify Ticket", type="primary", use_container_width=True)

    if classify_btn and ticket_input.strip():
        if model_choice == "Logistic Regression":
            model_used = artifacts["lr"]
        elif model_choice == "Naive Bayes":
            model_used = artifacts["nb"]
        else:
            model_used = best_model

        result = predict_ticket(model_used, vectorizer, ticket_input)
        # override threshold from slider
        result["needs_human_review"] = result["confidence"] < threshold

        cat = result["predicted_category"]
        icon = CATEGORY_ICONS.get(cat, "📋")
        color = CATEGORY_COLORS.get(cat, "#4361EE")
        priority = result["priority"]
        review = result["needs_human_review"]

        priority_badge = f'<span class="badge urgent-badge">🔴 URGENT</span>' if priority == "URGENT" else f'<span class="badge normal-badge">🟢 NORMAL</span>'
        review_badge   = f'<span class="badge review-badge">⚠️ Needs Human Review</span>' if review else f'<span class="badge auto-badge">✅ Auto-Assigned</span>'

        st.markdown(f"""
        <div class="result-box">
            <h3 style="margin:0 0 0.8rem; color:{color};">{icon} {cat}</h3>
            <p style="font-size:1.1rem; margin:0 0 1rem;">
                Confidence: <strong>{result['confidence']}%</strong>
                &nbsp;&nbsp; {priority_badge} {review_badge}
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Probability Breakdown")
        probs_df = pd.DataFrame(
            result["all_probabilities"].items(), columns=["Category", "Confidence (%)"]
        ).sort_values("Confidence (%)", ascending=False)
        st.bar_chart(probs_df.set_index("Category"))

    elif classify_btn:
        st.warning("Please enter some ticket text first.")

    # ── Sample Tickets ──
    st.markdown("---")
    st.subheader("🧪 Try Sample Tickets")
    samples = {
        "💳 Billing — double charge": "I was charged twice for my subscription this month. Need an immediate refund.",
        "🔧 Technical — app crash": "The mobile app crashes every time I open it after the latest update.",
        "👥 HR — leave request": "I need to apply for 5 days of sick leave starting next Monday.",
        "📋 General — feature request": "Could you add dark mode support? Many users have been asking for it.",
        "⚠️ Edge case — ambiguous": "I have a problem and need help right away.",
    }
    selected_sample = st.selectbox("Pick a sample to auto-fill:", list(samples.keys()))
    if st.button("Load Sample →"):
        st.session_state["sample_text"] = samples[selected_sample]
        st.info(f"Sample loaded: *{samples[selected_sample]}*")


# ─── TAB 2: Performance ───
with tab2:
    st.subheader("Model Evaluation Metrics")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{artifacts['acc_lr']:.1%}</div>
            <div class="metric-label">Logistic Regression</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{artifacts['acc_nb']:.1%}</div>
            <div class="metric-label">Naive Bayes</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">50</div>
            <div class="metric-label">Training Samples</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">4</div>
            <div class="metric-label">Categories</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    # Confusion matrix for best model
    with col_left:
        st.markdown(f"#### Confusion Matrix — {best_name}")
        y_pred = best_model.predict(artifacts["X_test"])
        cm = confusion_matrix(artifacts["y_test"], y_pred, labels=CATEGORIES)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=CATEGORIES, yticklabels=CATEGORIES, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.tight_layout()
        st.pyplot(fig)

    # Classification report
    with col_right:
        st.markdown("#### Classification Report")
        report = classification_report(
            artifacts["y_test"], y_pred, target_names=CATEGORIES, output_dict=True
        )
        report_df = pd.DataFrame(report).T.round(2)
        st.dataframe(report_df.style.background_gradient(cmap="Blues", axis=None), height=280)

        st.markdown("""
        **Reading the report:**
        - **Precision** — of tickets predicted as X, how many actually are X?
        - **Recall** — of all actual X tickets, how many did we catch?
        - **F1-score** — harmonic mean of precision & recall (balanced metric)
        """)


# ─── TAB 3: Dataset ───
with tab3:
    st.subheader("Dataset Overview")
    df = artifacts["df"]

    col_a, col_b = st.columns([2, 1])
    with col_a:
        category_filter = st.multiselect("Filter by category", CATEGORIES, default=CATEGORIES)
        filtered_df = df[df["category"].isin(category_filter)][["subject", "body", "category"]]
        st.dataframe(filtered_df.reset_index(drop=True), height=350)
    with col_b:
        st.markdown("#### Category Distribution")
        dist = df["category"].value_counts()
        fig2, ax2 = plt.subplots(figsize=(4, 4))
        colors = [CATEGORY_COLORS.get(c, "#999") for c in dist.index]
        ax2.pie(dist.values, labels=dist.index, colors=colors, autopct="%1.0f%%",
                startangle=90, wedgeprops=dict(width=0.6))
        ax2.set_title("Ticket Distribution", pad=10)
        st.pyplot(fig2)


# ── Footer ──
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#9ca3af; font-size:0.85rem;'>"
    "Built by <strong>Shivanshu Shukla</strong> · Support Ticket Classifier Assessment · "
    "Stack: Python · scikit-learn · TF-IDF · Streamlit</p>",
    unsafe_allow_html=True,
)
