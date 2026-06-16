import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
import streamlit as st
import pandas as pd
import numpy as np
import torch
import pickle
import re
import matplotlib.pyplot as plt
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

st.set_page_config(
    page_title="FinSight",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

from pathlib import Path
import os
os.environ["TRANSFORMERS_OFFLINE"] = "0"

BASE_DIR = Path(__file__).parent
MODEL_PATH = "Moyoshabz/finsight-transaction-classifier"
ENCODER_PATH = str(BASE_DIR / "label_encoder.pkl")
SAMPLE_PATH = str(BASE_DIR / "sample_monthly_statement.csv")
@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    with open(ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, tokenizer, le, device

CORRECTION_RULES = {
    "NETFLIX": "Entertainment", "SPOTIFY": "Entertainment",
    "HULU": "Entertainment", "DISNEY": "Entertainment",
    "HBO": "Entertainment", "APPLE MUSIC": "Entertainment",
    "YOUTUBE": "Entertainment", "PEACOCK": "Entertainment",
    "PARAMOUNT": "Entertainment", "TICKETMASTER": "Entertainment",
    "AMC THEATER": "Entertainment", "UBER TRIP": "Transport",
    "LYFT": "Transport", "WALGREENS": "Healthcare",
    "CVS": "Healthcare", "RITE AID": "Healthcare",
    "FPL": "Utilities", "DUKE ENERGY": "Utilities",
    "DOMINION": "Utilities", "XCEL ENERGY": "Utilities",
    "WALMART": "Shopping", "TARGET": "Shopping",
    "COSTCO": "Shopping", "HOME DEPOT": "Shopping",
    "AMZN": "Shopping", "AMAZON": "Shopping",
}

CAT_COLORS = {
    "Food & Dining": "#FF6B6B",
    "Transport": "#4ECDC4",
    "Shopping": "#45B7D1",
    "Entertainment": "#96CEB4",
    "Healthcare": "#FFEAA7",
    "Utilities": "#DDA0DD",
    "Financial Services": "#98D8C8"
}

def clean_transaction(text):
    text = str(text).upper()
    prefixes = [
        "CHECK CRD PURCHASE", "RECUR DEBIT CRD PMT",
        "DEBIT CARD PURCHASE", "POS PURCHASE",
        "ACH DEBIT", "ONLINE TRANSFER", "ATM WITHDRAWAL"
    ]
    for prefix in prefixes:
        text = text.replace(prefix, "")
    text = re.sub(r"\d{1,2}/\d{1,2}", "", text)
    text = re.sub(r"\b[A-Z0-9]*\d[A-Z0-9]*\d[A-Z0-9]*\b", "", text)
    text = re.sub(r"X{2,}", "", text)
    text = re.sub(r"MCC=?\s*\d*", "", text)
    text = re.sub(r"\bMCC\b", "", text)
    text = re.sub(r"\d{3}[-.\s]\d{3}[-.\s]\d{4}", "", text)
    text = re.sub(r"\b\d+\b", "", text)
    us_states = ["AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA",
                 "HI","ID","IL","IN","IA","KS","KY","LA","ME","MD",
                 "MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
                 "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC",
                 "SD","TN","TX","UT","VT","VA","WA","WV","WI","WY"]
    for state in us_states:
        text = re.sub(r"\b" + state + r"\b", "", text)
    text = re.sub(r"[^A-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def apply_rules(text, predicted):
    for keyword, category in CORRECTION_RULES.items():
        if keyword in text.upper():
            return category, True
    return predicted, False

def predict(text, model, tokenizer, le, device):
    cleaned = clean_transaction(text)
    encoding = tokenizer(
        cleaned, truncation=True, padding=True,
        max_length=64, return_tensors="pt"
    )
    with torch.no_grad():
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        confidence, predicted_idx = torch.max(probs, dim=1)
    raw_category = le.inverse_transform([predicted_idx.item()])[0]
    final_category, corrected = apply_rules(text, raw_category)
    return {
        "category": final_category,
        "confidence": round(confidence.item() * 100, 1),
        "corrected": corrected
    }

def predict_batch(transactions, model, tokenizer, le, device):
    results = []
    for txn in transactions:
        r = predict(txn, model, tokenizer, le, device)
        results.append({
            "transaction": txn,
            "category": r["category"],
            "confidence": r["confidence"],
            "corrected": r["corrected"]
        })
    return pd.DataFrame(results)

def spending_breakdown(df):
    breakdown = df.groupby("category")["amount"].agg(
        total="sum", count="count", avg="mean"
    ).round(2)
    breakdown["percentage"] = (
        breakdown["total"] / breakdown["total"].sum() * 100
    ).round(1)
    return breakdown.sort_values("total", ascending=False)

def health_score(df):
    total = df["amount"].sum()
    by_cat = df.groupby("category")["amount"].sum()
    needs_cats = ["Healthcare", "Utilities", "Transport", "Financial Services"]
    wants_cats = ["Food & Dining", "Entertainment", "Shopping"]
    needs = sum(by_cat.get(c, 0) for c in needs_cats)
    wants = sum(by_cat.get(c, 0) for c in wants_cats)
    needs_pct = (needs / total) * 100
    wants_pct = (wants / total) * 100
    needs_score = max(0, 100 - max(0, needs_pct - 50) * 2)
    wants_score = max(0, 100 - max(0, wants_pct - 30) * 2)
    diversity_score = min(100, len(by_cat) / 7 * 100)
    final = round(needs_score * 0.40 + wants_score * 0.40 + diversity_score * 0.20)
    return {
        "score": final,
        "needs_pct": round(needs_pct, 1),
        "wants_pct": round(wants_pct, 1),
        "needs_score": round(needs_score),
        "wants_score": round(wants_score),
        "diversity_score": round(diversity_score),
        "total": total,
        "by_cat": by_cat.to_dict()
    }

def detect_anomalies(df, multiplier=1.5):
    anomalies = []
    for cat in df["category"].unique():
        cat_df = df[df["category"] == cat]
        if len(cat_df) < 2:
            continue
        mean = cat_df["amount"].mean()
        std = cat_df["amount"].std()
        threshold = mean + multiplier * std
        for _, row in cat_df[cat_df["amount"] > threshold].iterrows():
            anomalies.append({
                "transaction": row["transaction"],
                "amount": row["amount"],
                "category": cat,
                "category_avg": round(mean, 2),
                "times_above_avg": round(row["amount"] / mean, 1)
            })
    return pd.DataFrame(anomalies) if anomalies else pd.DataFrame()

with st.sidebar:
    st.title("FinSight")
    st.caption("Intelligent Transaction Categorization")
    st.divider()
    page = st.radio(
        "Navigate",
        ["Home", "Upload & Categorize", "Spending Dashboard",
         "Health Score", "Anomaly Detection", "Try a Transaction"],
        label_visibility="collapsed"
    )
    st.divider()
    st.caption("Built with DistilBERT + Streamlit")
    st.caption("ML Course Project 2026")

model, tokenizer, le, device = load_model()

if "df_results" not in st.session_state:
    st.session_state.df_results = None

if page == "Home":
    st.title("FinSight")
    st.subheader("Automated Bank Transaction Categorization")
    st.write("")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model", "DistilBERT")
    with col2:
        st.metric("F1 Score", "0.6565")
    with col3:
        st.metric("Categories", "7")
    with col4:
        st.metric("vs Baseline", "+28%")
    st.divider()
    st.markdown("### How it works")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**1. Upload** your bank transactions as a CSV file")
    with col2:
        st.info("**2. FinSight** categorizes every transaction using fine-tuned BERT")
    with col3:
        st.info("**3. Explore** spending insights, health score and anomaly alerts")
    st.divider()
    st.markdown("### Spending Categories")
    cols = st.columns(7)
    for cat, col in zip(CAT_COLORS.keys(), cols):
        col.markdown(
            f"<div style='background:{CAT_COLORS[cat]};padding:8px;"
            f"border-radius:8px;text-align:center;font-size:11px;"
            f"font-weight:500'>{cat}</div>",
            unsafe_allow_html=True
        )
    st.divider()
    st.markdown("### CSV Format Required")
    sample = pd.DataFrame({
        "transaction": ["STARBUCKS STORE 12345", "NETFLIX MONTHLY SUB"],
        "amount": [6.75, 15.99]
    })
    st.dataframe(sample, use_container_width=True)
    sample_csv = pd.read_csv(SAMPLE_PATH)[
        ["transaction", "amount"]].to_csv(index=False)
    st.download_button(
        "Download Sample CSV",
        sample_csv,
        "sample_transactions.csv",
        "text/csv"
    )

elif page == "Upload & Categorize":
    st.title("Upload & Categorize")
    st.write("Upload a CSV with transaction and amount columns.")
    uploaded = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        if "transaction" not in df.columns or "amount" not in df.columns:
            st.error("CSV must have transaction and amount columns.")
        else:
            st.success(f"Loaded {len(df)} transactions")
            with st.spinner("Categorizing with DistilBERT..."):
                results = predict_batch(
                    df["transaction"].tolist(),
                    model, tokenizer, le, device
                )
                df["category"] = results["category"]
                df["confidence"] = results["confidence"]
                df["corrected"] = results["corrected"]
                st.session_state.df_results = df
            st.success("Categorization complete")
            st.divider()
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Transactions", len(df))
            with col2:
                st.metric("Total Spend", f"${df['amount'].sum():,.2f}")
            with col3:
                st.metric("Categories Found", df["category"].nunique())
            with col4:
                st.metric("Avg Confidence", f"{df['confidence'].mean():.1f}%")
            st.divider()
            st.markdown("### Categorized Transactions")
            st.dataframe(
                df[["transaction", "amount", "category", "confidence"]],
                use_container_width=True,
                height=400
            )
            st.download_button(
                "Download Categorized CSV",
                df.to_csv(index=False),
                "finsight_categorized.csv",
                "text/csv"
            )
            st.info("Navigate to other pages to explore insights")
    else:
        st.info("Upload a CSV file to get started.")

elif page == "Spending Dashboard":
    st.title("Spending Dashboard")
    df = st.session_state.df_results
    if df is None:
        st.warning("Please upload transactions first.")
    else:
        breakdown = spending_breakdown(df)
        total = df["amount"].sum()
        top_cat = breakdown.index[0]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Spend", f"${total:,.2f}")
        with col2:
            st.metric("Top Category", top_cat)
        with col3:
            st.metric("Top Category Share",
                      f"{breakdown.loc[top_cat, 'percentage']}%")
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Spending by Category")
            fig, ax = plt.subplots(figsize=(6, 5))
            colors = [CAT_COLORS.get(c, "#cccccc") for c in breakdown.index]
            ax.pie(
                breakdown["total"],
                labels=breakdown.index,
                autopct="%1.1f%%",
                colors=colors,
                startangle=90,
                pctdistance=0.85
            )
            st.pyplot(fig)
            plt.close()
        with col2:
            st.markdown("#### Amount by Category ($)")
            fig, ax = plt.subplots(figsize=(6, 5))
            bars = ax.barh(
                breakdown.index,
                breakdown["total"],
                color=colors,
                edgecolor="white"
            )
            for bar, val in zip(bars, breakdown["total"]):
                ax.text(bar.get_width() + 5,
                        bar.get_y() + bar.get_height() / 2,
                        f"${val:,.0f}", va="center", fontsize=9)
            ax.set_xlabel("Amount ($)")
            ax.set_xlim(0, breakdown["total"].max() * 1.3)
            ax.grid(True, alpha=0.2, axis="x")
            st.pyplot(fig)
            plt.close()
        st.divider()
        st.markdown("#### Detailed Breakdown")
        display_df = breakdown.copy()
        display_df.columns = ["Total ($)", "Transactions", "Avg ($)", "% of Budget"]
        display_df["Total ($)"] = display_df["Total ($)"].apply(lambda x: f"${x:,.2f}")
        display_df["Avg ($)"] = display_df["Avg ($)"].apply(lambda x: f"${x:,.2f}")
        display_df["% of Budget"] = display_df["% of Budget"].apply(lambda x: f"{x}%")
        st.dataframe(display_df, use_container_width=True)

elif page == "Health Score":
    st.title("Financial Health Score")
    st.caption("Based on the 50/30/20 budgeting framework")
    df = st.session_state.df_results
    if df is None:
        st.warning("Please upload transactions first.")
    else:
        h = health_score(df)
        score = h["score"]
        if score >= 75:
            score_color = "#2ECC71"
            score_label = "Good"
        elif score >= 50:
            score_color = "#F39C12"
            score_label = "Fair"
        else:
            score_color = "#E74C3C"
            score_label = "Needs Attention"
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(
                f"<div style='text-align:center;padding:30px;"
                f"background:{score_color}22;border-radius:16px;"
                f"border:2px solid {score_color}'>"
                f"<h1 style='font-size:80px;color:{score_color};margin:0'>{score}</h1>"
                f"<h3 style='color:{score_color};margin:0'>{score_label}</h3>"
                f"<p style='color:gray'>out of 100</p></div>",
                unsafe_allow_html=True
            )
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Needs Score", f"{h['needs_score']}/100",
                      f"{h['needs_pct']}% of budget (target 50%)")
        with col2:
            st.metric("Wants Score", f"{h['wants_score']}/100",
                      f"{h['wants_pct']}% of budget (target 30%)")
        with col3:
            st.metric("Diversity Score", f"{h['diversity_score']}/100",
                      "Spending across categories")
        st.divider()
        st.markdown("### Personalized Insights")
        if h["wants_pct"] > 30:
            excess = round(h["wants_pct"] - 30, 1)
            top_want = max(
                ["Food & Dining", "Entertainment", "Shopping"],
                key=lambda c: h["by_cat"].get(c, 0)
            )
            amt = h["by_cat"].get(top_want, 0)
            st.warning(
                f"Discretionary spending is {h['wants_pct']}% of budget "
                f"— {excess}% above the recommended 30%. "
                f"Highest wants category: {top_want} (${amt:,.2f}). "
                f"Consider reducing by ${amt * 0.10:,.2f}/month."
            )
        else:
            st.success(f"Discretionary spending well managed at {h['wants_pct']}%.")
        if h["needs_pct"] > 50:
            st.warning(
                f"Needs spending is {h['needs_pct']}% of budget — above 50% target. "
                f"This may include large fixed costs like mortgage or loan payments."
            )
        else:
            st.success(f"Needs spending well managed at {h['needs_pct']}%.")
        if score >= 75:
            st.success(f"Strong financial health score of {score}/100.")
        elif score >= 50:
            st.info(f"Moderate score of {score}/100. Small adjustments would help.")
        else:
            st.error(f"Score of {score}/100 suggests a review of spending priorities.")
        st.divider()
        st.markdown("### The 50/30/20 Framework")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("**50% Needs**\nHousing, transport, utilities, healthcare")
        with col2:
            st.info("**30% Wants**\nDining, entertainment, shopping")
        with col3:
            st.info("**20% Savings**\nEmergency fund, investments")

elif page == "Anomaly Detection":
    st.title("Anomaly Detection")
    st.caption("Transactions significantly above their category average")
    df = st.session_state.df_results
    if df is None:
        st.warning("Please upload transactions first.")
    else:
        multiplier = st.slider(
            "Detection sensitivity",
            min_value=1.0, max_value=3.0,
            value=1.5, step=0.1,
            help="Lower = more sensitive"
        )
        anomalies = detect_anomalies(df, multiplier)
        if len(anomalies) == 0:
            st.success("No anomalies detected at current sensitivity.")
        else:
            st.error(f"{len(anomalies)} anomalous transaction(s) detected")
            st.divider()
            for _, row in anomalies.iterrows():
                color = CAT_COLORS.get(row["category"], "#cccccc")
                st.markdown(
                    f"<div style='padding:16px;border-radius:10px;"
                    f"border-left:5px solid {color};background:{color}22;"
                    f"margin-bottom:12px'>"
                    f"<b>{row['transaction']}</b><br>"
                    f"<span style='font-size:20px;font-weight:bold'>"
                    f"${row['amount']:,.2f}</span> — "
                    f"<span style='color:red'>"
                    f"{row['times_above_avg']}x above category average "
                    f"(${row['category_avg']:,.2f})</span><br>"
                    f"<span style='color:gray'>{row['category']}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

elif page == "Try a Transaction":
    st.title("Try a Transaction")
    st.write("Type any bank transaction and see FinSight categorize it live.")
    txn_input = st.text_input(
        "Transaction string",
        placeholder="e.g. UBER EATS PAYMENT or NETFLIX MONTHLY SUB"
    )
    if txn_input:
        with st.spinner("Classifying..."):
            result = predict(txn_input, model, tokenizer, le, device)
        color = CAT_COLORS.get(result["category"], "#cccccc")
        corrected_note = " (rule corrected)" if result["corrected"] else ""
        st.markdown(
            f"<div style='padding:24px;border-radius:12px;"
            f"background:{color}33;border:2px solid {color};"
            f"text-align:center;margin-top:20px'>"
            f"<h2 style='color:{color};margin:0'>{result['category']}</h2>"
            f"<p style='font-size:18px;margin:8px 0'>"
            f"Confidence: <b>{result['confidence']}%</b>{corrected_note}</p>"
            f"</div>",
            unsafe_allow_html=True
        )
        st.divider()
        st.markdown("**Try these examples:**")
        examples = [
            "STARBUCKS STORE 12345",
            "UBER TRIP 4X9K2",
            "NETFLIX MONTHLY SUB",
            "WELLS FARGO LOAN PMT",
            "CVS PHARMACY 00234",
            "DELTA AIR LINES TICKET",
            "AMZN MKTP US*2K4LP",
            "FPL ENERGY BILL"
        ]
        cols = st.columns(4)
        for i, ex in enumerate(examples):
            with cols[i % 4]:
                st.code(ex, language=None)