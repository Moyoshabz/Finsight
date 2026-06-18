# FinSight
### Automated Bank Transaction Categorization Using Fine-Tuned DistilBERT

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co)

> FinSight transforms raw bank transaction strings like `AMZN MKTP US*2K4LP` 
> into meaningful spending categories and financial insights.

---

## Live Demo
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://finsight-kwvxwf5vdsk4x55kyks6wf.streamlit.app/)

![iamge_1](./image7.png)
---

## What it does
Upload a CSV of bank transactions and FinSight will:
- Categorize every transaction into 7 spending categories
- Generate a spending breakdown with percentages
- Calculate a Financial Health Score based on the 50/30/20 rule
- Flag unusual transactions as anomalies
- Visualize spending with an interactive dashboard

**Categories:** Food & Dining · Transport · Shopping · Entertainment · Healthcare · Utilities · Financial Services

---

## Model Performance

| Model | Weighted F1 | Accuracy |
|---|---|---|
| Logistic Regression + TF-IDF | 0.5152 | 51% |
| Random Forest + TF-IDF | 0.5085 | 50% |
| **DistilBERT fine-tuned** | **0.6565** | **66%** |

---

## Architecture

```mermaid
flowchart TD
    A[Raw Transaction String]
    B[Text Cleaning Pipeline]
    C[DistilBERT Fine-Tuned Classifier]
    D[Business Rule Correction Layer]
    E[Category + Confidence Score]
    F[Analytics Engine<br/>Health Score + Anomalies]
    G[Streamlit Dashboard]

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
```


## Dataset
- **Primary**: Wells Fargo Campus Analytics Challenge (40,000 real transactions)
- **Supplementary**: Hand-curated synthetic US merchant dataset (Food & Dining)
- **Final balanced training set**: 12,752 transactions across 7 categories

---

## Quick Start
- git clone https://github.com/Moyoshabz/Finsight.git
- cd Finsight
- pip install -r requirements.txt
- streamlit run app.py

---

## References
- Devlin et al. (2019) — BERT: Pre-training of Deep Bidirectional Transformers
- Sanh et al. (2019) — DistilBERT: Smaller, Faster, Cheaper
- Sun et al. (2019) — How to Fine-Tune BERT for Text Classification

