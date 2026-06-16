
import pandas as pd
import numpy as np

def generate_spending_breakdown(df):
    breakdown = df.groupby("category")["amount"].agg(
        total="sum", count="count", avg="mean"
    ).round(2)
    breakdown["percentage"] = (
        breakdown["total"] / breakdown["total"].sum() * 100
    ).round(1)
    return breakdown.sort_values("total", ascending=False)

def calculate_health_score(df):
    total = df["amount"].sum()
    by_category = df.groupby("category")["amount"].sum()
    needs_cats = ["Healthcare", "Utilities", "Transport", "Financial Services"]
    wants_cats = ["Food & Dining", "Entertainment", "Shopping"]
    needs = sum(by_category.get(c, 0) for c in needs_cats)
    wants = sum(by_category.get(c, 0) for c in wants_cats)
    needs_pct = (needs / total) * 100
    wants_pct = (wants / total) * 100
    needs_score = max(0, 100 - max(0, needs_pct - 50) * 2)
    wants_score = max(0, 100 - max(0, wants_pct - 30) * 2)
    diversity_score = min(100, len(by_category) / 7 * 100)
    final_score = (needs_score * 0.40 + wants_score * 0.40 + 
                   diversity_score * 0.20)
    return {
        "score": round(final_score),
        "needs_pct": round(needs_pct, 1),
        "wants_pct": round(wants_pct, 1),
        "needs_score": round(needs_score),
        "wants_score": round(wants_score),
        "diversity_score": round(diversity_score),
        "total_spend": total,
        "by_category": by_category.to_dict()
    }

def detect_anomalies(df, threshold_multiplier=1.5):
    anomalies = []
    for category in df["category"].unique():
        cat_df = df[df["category"] == category]
        if len(cat_df) < 2:
            continue
        mean_amt = cat_df["amount"].mean()
        std_amt = cat_df["amount"].std()
        threshold = mean_amt + (threshold_multiplier * std_amt)
        flagged = cat_df[cat_df["amount"] > threshold]
        for _, row in flagged.iterrows():
            anomalies.append({
                "transaction": row["transaction"],
                "amount": row["amount"],
                "category": row["category"],
                "category_avg": round(mean_amt, 2),
                "times_above_avg": round(row["amount"] / mean_amt, 1)
            })
    return pd.DataFrame(anomalies) if anomalies else pd.DataFrame()
